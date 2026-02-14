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
#include <vector>
#include <map>
#include "DebugLog.h"
#include "ReedSolomon.h"

//////////////////////////////// 設定 ////////////////////////////////
static constexpr const char* kServerIP    = "192.168.0.2";
static constexpr uint16_t    kServerPort  = 50000;
static constexpr int         kPeriodMs    = 60 * 1000; // 1分おき
static constexpr int         kRecvTOms    = 2000;      // 応答待ちタイムアウト(ms)
static constexpr bool        kVerboseLog  = true;
static constexpr int         kTimeSyncFecK = 14;      // FEC data shards
static constexpr int         kTimeSyncFecM = 8;       // FEC parity shards
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
struct TimeSyncUdpShardHeader {
    uint32_t magic_be;          // 'TSYN' (0x5453594E)
    uint16_t version_be;        // 1
    uint16_t header_bytes_be;   // sizeof(TimeSyncUdpShardHeader)

    uint16_t shard_index_be;    // 0..(k+m-1)
    uint16_t shard_count_be;    // k+m
    uint8_t  k;                 // data shards
    uint8_t  m;                 // parity shards
    uint16_t shard_bytes_be;    // bytes per shard payload

    uint32_t original_bytes_be; // original TimeSyncResponse bytes
};
#pragma pack(pop)
bool ReceiveTimeSyncResponseUdp(
    SOCKET s,
    const sockaddr_in& expectedServer,
    TimeSyncResponse& outResp,
    uint64_t& outT4_ns)
{
    const int k = kTimeSyncFecK;
    const int m = kTimeSyncFecM;

    std::map<uint32_t, std::vector<uint8_t>> receivedShards;
    uint16_t shard_bytes = 0;
    uint32_t original_bytes = 0;
    bool shard_meta_initialized = false;
    bool sawFecShard = false;

    auto start = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(kRecvTOms > 0 ? kRecvTOms : 2000);

    for (;;) {
        uint8_t buffer[1500];
        sockaddr_in from{};
        int fromLen = sizeof(from);

        int n = ::recvfrom(
            s,
            reinterpret_cast<char*>(buffer),
            static_cast<int>(sizeof(buffer)),
            0,
            reinterpret_cast<sockaddr*>(&from),
            &fromLen);

        if (n == SOCKET_ERROR) {
            int err = ::WSAGetLastError();
            if (err == WSAETIMEDOUT) {
                break;
            }
            DebugLog(L"TimeSyncClient(UDP): recvfrom() error");
            return false;
        }
        if (n <= 0) {
            auto now = std::chrono::steady_clock::now();
            if (kRecvTOms > 0 && now - start > timeout) {
                break;
            }
            continue;
        }

        // 他ホストからのパケットは無視
        if (from.sin_family != AF_INET ||
            from.sin_addr.s_addr != expectedServer.sin_addr.s_addr ||
            from.sin_port != expectedServer.sin_port) {
            continue;
        }

        const int packet_len = n;
        bool processed_as_fec = false;

        // まず FEC 付き TimeSyncUdpShardHeader として解釈を試みる
        if (packet_len >= static_cast<int>(sizeof(TimeSyncUdpShardHeader))) {
            TimeSyncUdpShardHeader hdr{};
            std::memcpy(&hdr, buffer, sizeof(TimeSyncUdpShardHeader));

            const uint32_t magic        = ntohl(hdr.magic_be);
            const uint16_t version      = ntohs(hdr.version_be);
            const uint16_t header_bytes = ntohs(hdr.header_bytes_be);
            const uint16_t shard_index  = ntohs(hdr.shard_index_be);
            const uint16_t shard_count  = ntohs(hdr.shard_count_be);
            const uint8_t  kk           = hdr.k;
            const uint8_t  mm           = hdr.m;
            const uint16_t shard_bytes_be = ntohs(hdr.shard_bytes_be);
            const uint32_t orig_bytes     = ntohl(hdr.original_bytes_be);

            if (magic == kMagic &&
                version == kVersion &&
                header_bytes == sizeof(TimeSyncUdpShardHeader) &&
                static_cast<int>(kk) == k &&
                static_cast<int>(mm) == m &&
                shard_count == static_cast<uint16_t>(k + m) &&
                shard_index < shard_count &&
                packet_len == static_cast<int>(header_bytes + shard_bytes_be)) {

                sawFecShard = true;
                processed_as_fec = true;

                if (!shard_meta_initialized) {
                    shard_bytes = shard_bytes_be;
                    original_bytes = orig_bytes;
                    shard_meta_initialized = true;
                } else {
                    if (shard_bytes != shard_bytes_be || original_bytes != orig_bytes) {
                        // メタ情報が不一致のシャードは無視
                        continue;
                    }
                }

                const uint8_t* payload_ptr = buffer + header_bytes;
                std::vector<uint8_t> shard(payload_ptr, payload_ptr + shard_bytes);
                receivedShards[shard_index] = std::move(shard);

                if (static_cast<int>(receivedShards.size()) >= k) {
                    std::vector<uint8_t> decoded;
                    if (!DecodeFEC_ISAL(receivedShards, k, m, original_bytes, decoded)) {
                        DebugLog(L"TimeSyncClient(UDP/FEC): DecodeFEC_ISAL failed");
                        return false;
                    }
                    if (decoded.size() < sizeof(TimeSyncResponse)) {
                        DebugLog(L"TimeSyncClient(UDP/FEC): decoded size too small");
                        return false;
                    }

                    std::memcpy(&outResp, decoded.data(), sizeof(TimeSyncResponse));
                    outT4_ns = NowNs();
                    return true;
                }
            }
        }

        // FEC ではなかった場合は、従来の TimeSyncResponse 生パケットとして解釈
        if (!processed_as_fec) {
            if (packet_len >= static_cast<int>(sizeof(TimeSyncResponse))) {
                TimeSyncResponse resp{};
                std::memcpy(&resp, buffer, sizeof(TimeSyncResponse));
                if (ntohl(resp.magic_be) == kMagic &&
                    ntohl(resp.version_be) == kVersion) {
                    outResp = resp;
                    outT4_ns = NowNs();
                    return true;
                }
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (kRecvTOms > 0 && now - start > timeout) {
            break;
        }
    }

    if (sawFecShard) {
        DebugLog(L"TimeSyncClient(UDP/FEC): timeout before collecting enough shards");
    }
    return false;
}
} // namespace

// 補助：補正遅延 [ns] を返す（任意利用）
inline int64_t ComputeCorrectedLatencyNs(uint64_t client_render_ns, uint64_t server_capture_ns) {
    return (int64_t)client_render_ns - (int64_t)server_capture_ns + g_TimeOffsetNs.load(std::memory_order_acquire);
}

// 引数なしスレッド関数：常駐で 1 分おきに同期、切断時は再接続
void TimeSyncClientThread() {
    EnsureWinsockOnce();

    while (!g_ClientStop.load(std::memory_order_acquire)) {
        // UDP ソケット作成
        SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (s == INVALID_SOCKET) {
            DebugLog(L"TimeSyncClient(UDP): socket() failed");
            ::Sleep(1000);
            continue;
        }

        // 受信タイムアウトを設定（FEC シャード収集全体の上限目安）
        if (kRecvTOms > 0) {
            DWORD to = static_cast<DWORD>(kRecvTOms);
            setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&to), sizeof(to));
        }

        sockaddr_in srv{};
        if (!IpPortToSockaddrIPv4(kServerIP, kServerPort, srv)) {
            DebugLog(L"TimeSyncClient(UDP): invalid server IP");
            closesocket(s);
            ::Sleep(1000);
            continue;
        }

        if (kVerboseLog) {
            DebugLog(L"TimeSyncClient(UDP/FEC): ready to sync with server");
        }

        // ソケットを維持しつつ、1 分おきにリクエスト送信
        while (!g_ClientStop.load(std::memory_order_acquire)) {
            // ---- リクエスト送信（T1）----
            TimeSyncRequest req{};
            req.magic_be   = htonl(kMagic);
            req.version_be = htonl(kVersion);
            uint64_t T1    = NowNs();
            req.t1_ns_be   = HostToNetwork64(T1);

            int sent = ::sendto(
                s,
                reinterpret_cast<const char*>(&req),
                static_cast<int>(sizeof(req)),
                0,
                reinterpret_cast<sockaddr*>(&srv),
                sizeof(srv));

            if (sent != static_cast<int>(sizeof(req))) {
                DebugLog(L"TimeSyncClient(UDP): sendto() failed");
                break; // ソケットを作り直す
            }

            // ---- 応答受信（FEC 付き or フォールバック生パケット）----
            TimeSyncResponse resp{};
            uint64_t T4 = 0;
            if (!ReceiveTimeSyncResponseUdp(s, srv, resp, T4)) {
                DebugLog(L"TimeSyncClient(UDP): failed to receive valid response");
                break; // 再試行のためソケットを作り直す
            }

            if (ntohl(resp.magic_be) != kMagic || ntohl(resp.version_be) != kVersion) {
                DebugLog(L"TimeSyncClient(UDP): invalid response header");
                break;
            }

            uint64_t T1e = NetworkToHost64(resp.t1_ns_be);
            uint64_t T2  = NetworkToHost64(resp.t2_ns_be);
            uint64_t T3  = NetworkToHost64(resp.t3_ns_be);

            if (T1e != T1) {
                DebugLog(L"TimeSyncClient(UDP): T1 echo mismatch");
                // 続行はするが、念のためログのみ残す
            }

            // ---- NTP 式で θ と δ を算出 ----
            int64_t t1 = static_cast<int64_t>(T1);
            int64_t t2 = static_cast<int64_t>(T2);
            int64_t t3 = static_cast<int64_t>(T3);
            int64_t t4 = static_cast<int64_t>(T4);

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
            if (kVerboseLog) {
                char msg[256];
                std::snprintf(
                    msg,
                    sizeof(msg),
                    "Client sync(UDP/FEC): offset(theta)=%.3f ms, RTT=%.3f ms (one-way~%.3f ms)",
                    offset_ns / 1e6,
                    rtt_ns / 1e6,
                    (rtt_ns / 2.0) / 1e6);

                wchar_t wmsg[256];
                MultiByteToWideChar(CP_UTF8, 0, msg, -1, wmsg, 256);
                DebugLog(wmsg);
            }

            // ---- 周期待ち（1分） ----
            for (int slept = 0;
                 slept < kPeriodMs && !g_ClientStop.load(std::memory_order_acquire);) {
                constexpr int kStep = 200; // 200ms 粒度で抜けられるように
                ::Sleep(kStep);
                slept += kStep;
            }
        }

        closesocket(s);
        DebugLog(L"TimeSyncClient(UDP): socket closed, retrying...");
        ::Sleep(1000); // 再作成までの待機
    }

    DebugLog(L"TimeSyncClient: stopped");
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
