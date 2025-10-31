// TimeSyncClient_TCP.cpp
// 時刻同期クライアント（TCP版・1分おきにリクエスト送信→オフセット更新）
// - 接続を維持し、60秒ごとにNTP式4タイムスタンプで offset(θ) と RTT(δ) を再計算
// - 切断/失敗時は自動再接続をリトライ
// Windows / C++17 / Winsock2 / TCP
// Link: Ws2_32.lib

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <WinSock2.h>
#include <WS2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <mutex>
#include <limits>
#include "DebugLog.h"

//////////////////////////////// 設定 ////////////////////////////////
static constexpr const char* kServerIP    = "192.168.0.2";
static constexpr uint16_t    kServerPort  = 50000;
static constexpr int         kPeriodMs    = 60 * 1000; // 1分おき
static constexpr int         kRecvTOms    = 2000;      // 応答待ちタイムアウト(ms)
static constexpr bool        kVerboseLog  = true;
//////////////////////////////////////////////////////////////////////

////////////////////////////// グローバル（重要） /////////////////////
// ▼ これが “時刻差(オフセット)” ▼
// server_time ≈ client_time + g_TimeOffsetNs  （単位: ns, θ）
std::atomic<int64_t>  g_TimeOffsetNs{ 0 };    // θ
std::atomic<bool>     g_TimeSyncValid{ false };
// 参考：RTT(δ) と 4タイムスタンプ
std::atomic<int64_t>  g_RttNs{ 0 };
std::atomic<uint64_t> g_T1_ns{ 0 }, g_T2_ns{ 0 }, g_T3_ns{ 0 }, g_T4_ns{ 0 };

// 外部から停止したい場合に true をセット（任意利用）
std::atomic<bool>     g_ClientStop{ false };
//////////////////////////////////////////////////////////////////////

namespace {
    using Clock = std::chrono::system_clock;
    using ns    = std::chrono::nanoseconds;

    inline uint64_t NowNs() {
        auto t = Clock::now().time_since_epoch();
        return (uint64_t)std::chrono::duration_cast<ns>(t).count();
    }
    // host<->network 64bit big-endian
    inline uint64_t HostToNetwork64(uint64_t x) {
        return ( (uint64_t)htonl((uint32_t)(x >> 32)) ) | ( ((uint64_t)htonl((uint32_t)x)) << 32 );
    }
    inline uint64_t NetworkToHost64(uint64_t x) {
        return ( (uint64_t)ntohl((uint32_t)(x >> 32)) ) | ( ((uint64_t)ntohl((uint32_t)x)) << 32 );
    }
    bool IpPortToSockaddrIPv4(const char* ip, uint16_t port_host, sockaddr_in& out) {
        std::memset(&out, 0, sizeof(out));
        out.sin_family = AF_INET;
        out.sin_port   = htons(port_host);
        return ::InetPtonA(AF_INET, ip, &out.sin_addr) == 1;
    }
    void EnsureWinsockOnce() {
        static std::once_flag once;
        std::call_once(once, []{
            WSADATA wsa{};
            if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) {
                DebugLog(L"WSAStartup failed (client)");
            } else {
                DebugLog(L"WSAStartup OK (client)");
            }
        });
    }
    // 固定長送受信（TCP）
    bool SendAll(SOCKET s, const char* buf, int len) {
        int sentTotal = 0;
        while (sentTotal < len) {
            int n = send(s, buf + sentTotal, len - sentTotal, 0);
            if (n <= 0) return false;
            sentTotal += n;
        }
        return true;
    }
    bool RecvAll(SOCKET s, char* buf, int len) {
        int recvd = 0;
        while (recvd < len) {
            int n = recv(s, buf + recvd, len - recvd, 0);
            if (n <= 0) return false;
            recvd += n;
        }
        return true;
    }

    // パケット（NTP 4タイムスタンプ互換）
    constexpr uint32_t kMagic   = 0x5453594E; // 'T','S','Y','N'
    constexpr uint32_t kVersion = 1;
#pragma pack(push, 1)
    struct TimeSyncRequest {
        uint32_t magic_be;
        uint32_t version_be;
        uint64_t t1_ns_be; // client send time (client clock)
    };
    struct TimeSyncResponse {
        uint32_t magic_be;
        uint32_t version_be;
        uint64_t t1_ns_be; // echo
        uint64_t t2_ns_be; // server recv time (server clock)
        uint64_t t3_ns_be; // server send time (server clock)
    };
#pragma pack(pop)
} // namespace

// 補助：補正遅延 [ns] を返す（任意利用）
inline int64_t ComputeCorrectedLatencyNs(uint64_t client_render_ns, uint64_t server_capture_ns) {
    return (int64_t)client_render_ns - (int64_t)server_capture_ns + g_TimeOffsetNs.load(std::memory_order_acquire);
}

// 引数なしスレッド関数：常駐で 1 分おきに同期、切断時は再接続
void TimeSyncClientThread() {
    EnsureWinsockOnce();

    while (!g_ClientStop.load(std::memory_order_acquire)) {
        // 接続確立
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == INVALID_SOCKET) {
            DebugLog(L"Client socket() failed");
            ::Sleep(1000);
            continue;
        }
        // Nagle無効
        {
            BOOL yes = TRUE;
            setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&yes, sizeof(yes));
        }
        // recv タイムアウト
        if (kRecvTOms > 0) {
            DWORD to = (DWORD)kRecvTOms;
            setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&to, sizeof(to));
        }

        sockaddr_in srv{};
        if (!IpPortToSockaddrIPv4(kServerIP, kServerPort, srv)) {
            DebugLog(L"Invalid server IP");
            closesocket(s);
            ::Sleep(1000);
            continue;
        }
        if (connect(s, (sockaddr*)&srv, sizeof(srv)) == SOCKET_ERROR) {
            DebugLog(L"Client connect() failed");
            closesocket(s);
            ::Sleep(1000);
            continue;
        }
        DebugLog(L"Client connected to server");

        // 接続が生きている間、周期的に要求
        for (;;) {
            if (g_ClientStop.load(std::memory_order_acquire)) break;

            // ---- 送信（T1）----
            TimeSyncRequest req{};
            req.magic_be   = htonl(kMagic);
            req.version_be = htonl(kVersion);
            uint64_t T1 = NowNs();
            req.t1_ns_be   = HostToNetwork64(T1);

            if (!SendAll(s, (const char*)&req, sizeof(req))) {
                DebugLog(L"Client send error");
                break; // 再接続
            }

            // ---- 受信（T2,T3 を含む応答）----
            TimeSyncResponse resp{};
            if (!RecvAll(s, (char*)&resp, sizeof(resp))) {
                DebugLog(L"Client recv error");
                break; // 再接続
            }
            uint64_t T4 = NowNs();

            if (ntohl(resp.magic_be) != kMagic || ntohl(resp.version_be) != kVersion) {
                DebugLog(L"Client: invalid response header");
                break; // 再接続
            }

            uint64_t T1e = NetworkToHost64(resp.t1_ns_be);
            uint64_t T2  = NetworkToHost64(resp.t2_ns_be);
            uint64_t T3  = NetworkToHost64(resp.t3_ns_be);

            if (T1e != T1) {
                DebugLog(L"Client: T1 echo mismatch");
                // 続行してもよいが、ここでは安全側で次回に期待
            }

            // ---- NTP式で θ と δ を算出 ----
            int64_t t1 = (int64_t)T1;
            int64_t t2 = (int64_t)T2;
            int64_t t3 = (int64_t)T3;
            int64_t t4 = (int64_t)T4;

            int64_t offset_ns = ((t2 - t1) + (t3 - t4)) / 2;     // θ
            int64_t rtt_ns    = (t4 - t1) - (t3 - t2);           // δ
            if (rtt_ns < 0) rtt_ns = 0;

            // 保存
            g_TimeOffsetNs.store(offset_ns, std::memory_order_release);
            g_RttNs.store(rtt_ns, std::memory_order_release);
            g_T1_ns.store(T1, std::memory_order_release);
            g_T2_ns.store(T2, std::memory_order_release);
            g_T3_ns.store(T3, std::memory_order_release);
            g_T4_ns.store(T4, std::memory_order_release);
            g_TimeSyncValid.store(true, std::memory_order_release);

            // ログ
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                "Client sync: offset(theta)=%.3f ms, RTT=%.3f ms (one-way~%.3f ms)",
                offset_ns/1e6, rtt_ns/1e6, (rtt_ns/2.0)/1e6);
            
            // Convert to wide string for DebugLog
            wchar_t wmsg[256];
            MultiByteToWideChar(CP_UTF8, 0, msg, -1, wmsg, 256);
            DebugLog(wmsg);

            // ---- 周期待ち（1分） ----
            for (int slept = 0; slept < kPeriodMs && !g_ClientStop.load(std::memory_order_acquire);) {
                constexpr int kStep = 200; // 200ms 粒度で抜けられるように
                ::Sleep(kStep);
                slept += kStep;
            }
        }

        closesocket(s);
        DebugLog(L"Client: disconnected, retrying...");
        ::Sleep(1000); // 再接続までの待機
    }

    DebugLog(L"Client: stopped");
}

/*
/////////////////////// wWinMain からの簡単サンプル起動（参考） ///////////////////////
#include <thread>
int APIENTRY wWinMain(HINSTANCE, HINSTANCE, LPWSTR, int) {
    std::thread th(&TimeSyncClientThread); // 引数なし
    th.detach(); // 常駐させるなら detach でもOK

    // 利用例：補正遅延
    // server_capture_ns: サーバ刻印の system_clock(ns)
    // client_render_ns : クライアント描画時刻 system_clock(ns)
    // int64_t corrected = ComputeCorrectedLatencyNs(client_render_ns, server_capture_ns);

    return 0;
}
*/
