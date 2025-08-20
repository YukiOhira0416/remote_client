#ifndef NOMINMAX
#define NOMINMAX 
#endif 
// Force _WIN32_WINNT to be at least Windows Vista (0x0600) for WSARecvMsg.
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0600 
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h> // Required for WSARecvMsg and WSASendMsg
#include <windows.h>
#include <mmsystem.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>
#include <mutex>
#include <ctime>
#include "window.h"
#include <filesystem>
#include <regex>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <map>
#include <queue>
#include <condition_variable>
#include "DebugLog.h"
// 追加：非同期ロガー初期化/終了用
using namespace DebugLogAsync;
#include "ReedSolomon.h"
#include <gf_complete.h>
#include <jerasure.h>
#include <reed_sol.h>
#include <cauchy.h>
#include <sstream>
#include "concurrentqueue/concurrentqueue.h"
#include <enet/enet.h>
#include "Globals.h"
#include "nvdec.h"
#include "AppShutdown.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <d3dx12.h>
#include <d3d12.h>

// === 新規：ネットワーク準備・解像度ペンディング管理 ===
std::atomic<bool> g_networkReady{false};
std::atomic<bool> g_pendingResolutionValid{false};
std::atomic<int>  g_pendingW{0}, g_pendingH{0};

// window.cpp 側で実装されている既存API（エクスポート）
void SendFinalResolution(int width, int height); // Existing function from window.h
void ClearReorderState();                        // New function to be implemented in window.cpp
extern std::atomic<int> currentResolutionWidth;  // Assumed to be in window.cpp per instructions
extern std::atomic<int> currentResolutionHeight; // Assumed to be in window.cpp per instructions


// 既存定義を流用（main.cpp 冒頭にある定数と同じもの）
#ifndef SERVER_IP_RESEND
#define SERVER_IP_RESEND "127.0.0.1"
#endif
#ifndef SERVER_PORT_RESEND
#define SERVER_PORT_RESEND 8120
#endif

// 失敗しても致命傷にしない（ベストエフォート）
static void RequestIDRNow()
{
    SOCKET sock = INVALID_SOCKET;
    ADDRINFOA hints{};
    ADDRINFOA* result = nullptr;

    try {
        hints.ai_family   = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;
        hints.ai_protocol = IPPROTO_UDP;

        if (getaddrinfo(SERVER_IP_RESEND, std::to_string(SERVER_PORT_RESEND).c_str(), &hints, &result) != 0 || !result) {
            DebugLog(L"RequestIDRNow: getaddrinfo failed.");
            return;
        }

        sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (sock == INVALID_SOCKET) {
            DebugLog(L"RequestIDRNow: socket() failed.");
            freeaddrinfo(result);
            return;
        }

        const char* msg = "REQUEST_IDR#";
        int sent = sendto(sock, msg, (int)strlen(msg), 0, result->ai_addr, (int)result->ai_addrlen);
        if (sent <= 0) {
            DebugLog(L"RequestIDRNow: sendto failed.");
        } else {
            DebugLog(L"RequestIDRNow: sent REQUEST_IDR#");
        }
        freeaddrinfo(result);
    } catch (...) {
        DebugLog(L"RequestIDRNow: unexpected exception.");
    }

    if (sock != INVALID_SOCKET) closesocket(sock);
}

// どこからでも呼べるように：解像度が確定/変更されたときの通知
void OnResolutionChanged_GatedSend(int w, int h, bool forceResendNow = false)
{
    currentResolutionWidth  = w;
    currentResolutionHeight = h;

    // 常にペンディング更新（最後の値を保持）
    g_pendingW = w; g_pendingH = h; g_pendingResolutionValid = true;

    if (forceResendNow || g_networkReady.load()) {
        // 送出してリオーダをクリア、IDR要求を発行
        DebugLog(L"OnResolutionChanged_GatedSend: sending now.");
        SendFinalResolution(w, h);
        ClearReorderState();
        RequestIDRNow();

        g_pendingResolutionValid = false;
    } else {
        DebugLog(L"OnResolutionChanged_GatedSend: network not ready, pending.");
    }
}

// ネットワーク（ENet受信側）が「接続完了」になったときに必ず呼ぶ
void OnNetworkReady()
{
    g_networkReady = true;
    DebugLog(L"OnNetworkReady: network is ready.");

    if (g_pendingResolutionValid.load()) {
        int w = g_pendingW.load(), h = g_pendingH.load();
        DebugLog(L"OnNetworkReady: flushing pending resolution.");
        SendFinalResolution(w, h);
        ClearReorderState();
        RequestIDRNow();
        g_pendingResolutionValid = false;
    }

    // 念のためワンショット再送（取りこぼし対策）
    std::thread([]{
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (g_networkReady.load()) {
            DebugLog(L"OnNetworkReady: one-shot re-announce.");
            int w = currentResolutionWidth.load(), h = currentResolutionHeight.load();
            if (w > 0 && h > 0) {
                SendFinalResolution(w, h);
                RequestIDRNow();
            }
        }
    }).detach();
}

// ==== [GPU Policy Support - BEGIN] ====
#include <dxgi1_6.h>
#pragma comment(lib, "dxgi.lib")
#include <vector>
#include <string>

struct DetectedAdapter {
    DXGI_ADAPTER_DESC1 desc{};
    bool isSoftware = false;
    bool isDiscrete = false;   // Heuristic: DedicatedVideoMemory > 0
    bool isIntegrated = false; // Heuristic: Intel iGPU (Vendor 0x8086) 等
    bool isNvidia = false;     // Vendor 0x10DE
};

static std::vector<DetectedAdapter> EnumerateAdaptersDXGI() {
    std::vector<DetectedAdapter> out;
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    UINT flags = 0;
#if defined(_DEBUG)
    flags |= DXGI_CREATE_FACTORY_DEBUG;
#endif
    HRESULT hr = CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory));
    if (FAILED(hr) || !factory) {
        // 取得失敗時は空リスト（上位で扱う）
        return out;
    }

    for (UINT i = 0;; ++i) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
        if (factory->EnumAdapters1(i, &adapter1) == DXGI_ERROR_NOT_FOUND) break;

        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter1->GetDesc1(&desc))) continue;

        DetectedAdapter a{};
        a.desc = desc;
        a.isSoftware = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        if (a.isSoftware) continue; // WARP等は除外

        a.isNvidia    = (desc.VendorId == 0x10DE);
        const bool isIntel = (desc.VendorId == 0x8086);
        // 離散/統合の簡易判定（DXGIだけでは厳密にeGPU判定不可）
        a.isDiscrete   = (desc.DedicatedVideoMemory > 0);
        a.isIntegrated = (!a.isDiscrete) || isIntel;

        out.push_back(a);
    }
    return out;
}

static bool EvaluateGpuPolicy(const std::vector<DetectedAdapter>& adapters,
                              std::wstring& outUserMessage) {
    // ルール適用のため、離散GPUのみ抽出
    int discreteCount = 0;
    int nvidiaDiscrete = 0;
    int nonNvidiaDiscrete = 0;
    bool hasIntegrated = false;

    for (auto& a : adapters) {
        if (a.isDiscrete) {
            ++discreteCount;
            if (a.isNvidia) ++nvidiaDiscrete;
            else ++nonNvidiaDiscrete;
        } else {
            // ヘuristic: Intel等を統合扱い
            if (a.isIntegrated) hasIntegrated = true;
        }
    }

    // 判定
    if (discreteCount >= 2) {
        outUserMessage = L"Multiple discrete (external) GPUs were detected. "
                         L"This app supports exactly one NVIDIA GPU. Click OK to exit.";
        return false;
    }
    if (discreteCount == 1 && nonNvidiaDiscrete == 1 && hasIntegrated) {
        outUserMessage = L"A non-NVIDIA discrete GPU and an integrated GPU were detected. "
                         L"This app requires a single NVIDIA GPU. Click OK to exit.";
        return false;
    }
    if (discreteCount == 1 && nvidiaDiscrete == 1 && hasIntegrated) {
        outUserMessage = L"An NVIDIA discrete GPU and an integrated GPU were detected. "
                         L"This build requires a single NVIDIA GPU only (no integrated GPU present). Click OK to exit.";
        return false;
    }
    if (discreteCount == 1 && nvidiaDiscrete == 1 && !hasIntegrated) {
        // 唯一の許可パターン
        return true;
    }

    // その他（離散なし、NVIDIA不在 等）
    outUserMessage = L"No supported GPU configuration was found. "
                     L"This app requires exactly one discrete NVIDIA GPU. Click OK to exit.";
    return false;
}

static bool EnforceGpuPolicyOrExit() {
    try {
        auto adapters = EnumerateAdaptersDXGI();
        if (adapters.empty()) {
            MessageBoxW(nullptr,
                L"Could not enumerate GPUs (DXGI factory failed). "
                L"This app requires exactly one discrete NVIDIA GPU. Click OK to exit.",
                L"GPU Requirement",
                MB_OK | MB_ICONINFORMATION);
            return false;
        }
        std::wstring msg;
        const bool allow = EvaluateGpuPolicy(adapters, msg);
        if (!allow) {
            MessageBoxW(nullptr, msg.c_str(), L"GPU Requirement", MB_OK | MB_ICONINFORMATION);
            return false;
        }
        return true;
    } catch (...) {
        MessageBoxW(nullptr,
            L"An unexpected error occurred while checking GPU configuration. "
            L"This app requires exactly one discrete NVIDIA GPU. Click OK to exit.",
            L"GPU Requirement",
            MB_OK | MB_ICONINFORMATION);
        return false;
    }
}
// ==== [GPU Policy Support - END] ====
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "Mswsock.lib")          // For WSARecvMsg, WSASendMsg
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "dxgi.lib")             // For DXGI functions like CreateDXGIFactory
#pragma comment(lib, "ole32.lib")            // COM オブジェクト関連
#pragma comment(lib, "uuid.lib")             // IID 関連
#pragma comment(lib, "mfplat.lib")           // MediaFoundation コア
#pragma comment(lib, "mfuuid.lib")           // MF GUID
#pragma comment(lib, "mfreadwrite.lib")      // MF リーダライタ（必要なら）
#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "Strmiids.lib")
#pragma comment(lib, "winmm.lib") // For timeBeginPeriod / timeEndPeriod
#pragma comment(lib, "secur32.lib")
#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")

#define SERVER_IP_DATA "127.0.0.1"
#define SERVER_IP_RESEND "127.0.0.1"// 再送信要求用のIPアドレス
#define CLIENT_IP_BANDWIDTH "127.0.0.1"
#define SERVER_PORT_RESEND 8120 // 再送信要求用のポート番号
#define SERVER_PORT_DATA 8130 // パケット受信用のポート番号
#define CLIENT_PORT_BANDWIDTH 8200// 帯域幅測定用のポート番号
#define BANDWIDTH_DATA_SIZE 60 * 1024  // 60KB(帯域幅測定時のデータサイズ)
#define DATA_PACKET_SIZE 1400 // UDPパケットサイズ
#define WSARECV_BUFFER_SIZE 65000

// FEC worker threads and control variables
std::atomic<bool> send_bandw_Running = true;
std::atomic<bool> receive_resend_Running = true;
std::atomic<bool> receive_raw_packet_Running = true;

// Mutexes for synchronization
std::mutex logMutex;
std::mutex processvideosequenceMutex;
std::mutex g_frameBufferMutex;

// Frame and fragment related data structures
std::map<int, std::map<int, std::vector<uint8_t>>> g_frameBuffer;
std::map<int, int> expectedFrameCounts;

// Fragment assembly data structures
std::map<uint64_t, std::map<uint16_t, std::vector<uint8_t>>> g_fragmentAssemblyBuffer;
std::mutex g_fragmentAssemblyMutex;
std::map<uint64_t, uint16_t> g_expectedFragmentsCount;
std::map<uint64_t, uint16_t> g_receivedFragmentsCount;
std::map<uint64_t, size_t> g_accumulatedFragmentDataSize;

// Frame metadata structure and management
struct FrameMetadata {
    uint64_t firstTimestamp = 0;
    uint32_t originalDataLen = 0;
};
std::map<int, FrameMetadata> g_frameMetadata;
std::mutex g_frameMetadataMutex;

// Fragment timing and frame ID management
std::map<uint64_t, std::chrono::steady_clock::time_point> g_fragmentFirstPacketTime;
std::atomic<uint64_t> g_rgbaFrameIdCounter{0};

// Number of packet processing threads
const int NUM_FEC_WORKER_THREADS = 1;        // Threads that process ParsedShardInfo, assemble shards, and do FEC

const int NUM_RECEIVER_THREADS = 4; // Number of receiver threads

// Structures (assuming they are defined in Globals.h or a shared header, replicating here for clarity if not)
// Ensure these match the definitions used by the sender (CaptureManager.cpp)
#ifndef SHARED_PACKET_STRUCTURES_DEFINED
#define SHARED_PACKET_STRUCTURES_DEFINED
struct ShardInfoHeader {
    uint32_t frameNumber;        // Network byte order
    uint32_t shardIndex;         // Network byte order (0 to k-1 for data, k to k+m-1 for parity)
    uint32_t totalDataShards;    // Network byte order (RS_K)
    uint32_t totalParityShards;  // Network byte order (RS_M)
    uint32_t originalDataLen;    // Network byte order (length of H.264 frame before padding and FEC)
};

#endif

// ENet Packet type prefixes (consistent with sender in CaptureManager.cpp)
const uint8_t PACKET_TYPE_FULL_SHARD      = 0x01;
const uint8_t ENET_PACKET_TYPE_APP_FRAGMENT = 0x02;

// ENet Application-level Fragment Header (consistent with sender in CaptureManager.cpp)
struct ENetAppFragmentHeader {
    uint32_t original_packet_id_net; // Network byte order (e.g., frameNumber << 16 | shardIndex)
    uint16_t fragment_index_net;     // Network byte order
    uint16_t total_fragments_net;    // Network byte order
}; // Size: 4 + 2 + 2 = 8 bytes

struct AppFragmentAssemblyState {
    std::map<uint16_t, std::vector<uint8_t>> fragments; // key: fragment_index (host byte order)
    uint16_t total_fragments = 0;
    std::chrono::steady_clock::time_point first_fragment_received_time;
};

// For ENet app fragments - Define it here if not defined elsewhere
const std::chrono::seconds APP_FRAGMENT_ASSEMBLY_TIMEOUT(5); 

// Assume these are declared globally, accessible by this function
extern std::atomic<bool> receive_raw_packet_Running; // Controls the main loop

// Buffer for reassembling application-level ENet fragments
std::map<uint32_t, AppFragmentAssemblyState> appFragmentBuffers; // key: original_packet_id (host byte order)
std::mutex g_appFragmentBuffersMutex; // Mutex to protect appFragmentBuffers

// New struct for parsed shard information
struct ParsedShardInfo {
    uint64_t wgcCaptureTimestamp;
    uint64_t rawpacket_to_fec_timestamp;
    uint32_t frameNumber;        // Host byte order
    uint32_t shardIndex;         // Host byte order
    uint32_t totalDataShards;    // Host byte order
    uint32_t totalParityShards;  // Host byte order
    uint32_t originalDataLen;    // Host byte order
    std::vector<uint8_t> shardData;
};



// Comparator for ParsedShardInfo to prioritize by wgcCaptureTimestamp (older first)
struct ParsedShardInfoComparator {
    bool operator()(const ParsedShardInfo& lhs, const ParsedShardInfo& rhs) const {
        // std::priority_queue is a max-heap by default.
        // To make it a min-heap (smallest timestamp at top), this comparator
        // should return true if lhs is "greater than" rhs in terms of priority value.
        // Since we want smaller timestamp to have higher priority (be at the top),
        // if lhs.wgcCaptureTimestamp > rhs.wgcCaptureTimestamp, it means lhs has lower priority.
        return lhs.wgcCaptureTimestamp > rhs.wgcCaptureTimestamp;
    }
};

template<typename T,
         typename Container = std::vector<T>,
         typename Compare = std::less<typename Container::value_type>>
class ThreadSafePriorityQueue {
public:
    ThreadSafePriorityQueue() = default;
    ~ThreadSafePriorityQueue() = default;

    // Disable copy and assignment
    ThreadSafePriorityQueue(const ThreadSafePriorityQueue&) = delete;
    ThreadSafePriorityQueue& operator=(const ThreadSafePriorityQueue&) = delete;

    void enqueue(T item) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pq.push(std::move(item)); // item is moved into the priority queue
    }

    bool try_dequeue(T& item_out) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_pq.empty()) {
            return false;
        }
        item_out = m_pq.top(); // This is a copy operation from const T&
        m_pq.pop();
        return true;
    }

    size_t size_approx() const { // Keep the same method name for minimal changes
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pq.size();
    }

private:
    mutable std::mutex m_mutex;
    std::priority_queue<T, Container, Compare> m_pq;
};

ThreadSafePriorityQueue<ParsedShardInfo, std::vector<ParsedShardInfo>, ParsedShardInfoComparator> g_parsedShardQueue;

// フレーム受信時に I フレームのみを送信するかどうか
bool sendIFrameOnly = true; // This seems like a debug/test flag, consider removing or making configurable

// Helper function to check for and clear timed-out fragment assembly buffers
void ClearTimedOutAppFragments() {
    std::lock_guard<std::mutex> lock(g_appFragmentBuffersMutex); // Protect access to appFragmentBuffers
    auto now = std::chrono::steady_clock::now();
    for (auto it = appFragmentBuffers.begin(); it != appFragmentBuffers.end(); /* manual increment */) {
        if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second.first_fragment_received_time) > APP_FRAGMENT_ASSEMBLY_TIMEOUT) {
            // Consider adding thread ID if this function is called from multiple threads,
            // though with a single global buffer, the log source might be less critical here.
            //DebugLog(L"ClearTimedOutAppFragments: Clearing timed-out app fragments for packet ID " + std::to_wstring(it->first));
            it = appFragmentBuffers.erase(it);
        } else {
            ++it;
        }
    }
}

void SaveH264ToFile_NUM(const std::vector<uint8_t>& prepared_h264Buffer, const std::string& baseName) {
    if (prepared_h264Buffer.empty()) {
        return;  // 0バイトの場合は何もしない
    }
    
    static int fileCounter = 0;

    // 実行ファイルのパスを取得
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH) == 0) {
        DebugLog(L"Failed to get executable path.");
        return;
    }

    std::filesystem::path folderPath(exePath);
    folderPath.remove_filename();
    folderPath /= "ffplay";

    if (!std::filesystem::exists(folderPath)) {
        std::filesystem::create_directories(folderPath);
    }

    // ファイル名を生成
    std::ostringstream oss;
    oss << baseName << "_" << std::setw(4) << std::setfill('0') << fileCounter++ << ".h264";
    std::string numberedFilename = (folderPath / oss.str()).string();

    // タイムスタンプを付加
    std::ofstream ofs(numberedFilename, std::ios::binary);
    if (!ofs) {
        DebugLog(L"Error opening file " + std::wstring(numberedFilename.begin(), numberedFilename.end()));
        return;
    }
    ofs.write(reinterpret_cast<const char*>(prepared_h264Buffer.data()), prepared_h264Buffer.size());
    ofs.close();

    DebugLog(L"Saved " + std::to_wstring(prepared_h264Buffer.size()) + L" bytes to " + std::wstring(numberedFilename.begin(), numberedFilename.end()));
}

// NALユニットの先頭バイトからタイプを抽出する関数
int getNALType(const uint8_t* data, size_t size) {
    if (size < 1) return -1;
    // H.264 の NAL ヘッダー (1バイト): 0b[forbidden_zero_bit][nal_ref_idc][nal_unit_type]
    return data[0] & 0x1F;
}

// エンコード済みバッファから I フレームのみを抽出する関数（Iフレーム: NAL type 5）
std::vector<uint8_t> filterIFrameAndParameterSets(const std::vector<uint8_t>& encodedBuffer) {
    std::vector<uint8_t> output;
    bool hasIDR = false;
    bool hasSPS = false;
    bool hasPPS = false;

    size_t i = 0;
    while (i + 4 < encodedBuffer.size()) {
        if (encodedBuffer[i] == 0x00 &&
            encodedBuffer[i + 1] == 0x00 &&
            encodedBuffer[i + 2] == 0x00 &&
            encodedBuffer[i + 3] == 0x01) {

            size_t nalStart = i + 4;
            size_t nalEnd = encodedBuffer.size();

            for (size_t j = nalStart; j + 4 < encodedBuffer.size(); ++j) {
                if (encodedBuffer[j] == 0x00 &&
                    encodedBuffer[j + 1] == 0x00 &&
                    encodedBuffer[j + 2] == 0x00 &&
                    encodedBuffer[j + 3] == 0x01) {
                    nalEnd = j;
                    break;
                }
            }

            int nalType = getNALType(&encodedBuffer[nalStart], nalEnd - nalStart);

            if (nalType == 5) hasIDR = true;
            if (nalType == 7) hasSPS = true;
            if (nalType == 8) hasPPS = true;

            if (nalType == 5 || nalType == 7 || nalType == 8) {
                output.insert(output.end(), encodedBuffer.begin() + i, encodedBuffer.begin() + nalEnd);
            }

            i = nalEnd;
        }
        else {
            ++i;
        }
    }

    if (hasIDR && hasSPS && hasPPS) {
        return output;
    }
    else {
        return {}; // 0バイト
    }
}


// 送信前の準備（呼び出し側で処理済み）
std::vector<uint8_t> prepareFrameForSending(const std::vector<uint8_t>& encodedBuffer) {
    if (sendIFrameOnly) {
        return filterIFrameAndParameterSets(encodedBuffer);
    }
    else {
        return encodedBuffer; // そのまま送信
    }
}

void InitializeRSMatrix() {
    std::call_once(g_matrix_init_flag, []() {
        // 1. 従来の Vandermonde ベースのエンコード行列を生成
        // → Cauchy ベースのエンコード行列を生成するよう変更 ←
        g_vandermonde_matrix = cauchy_original_coding_matrix(RS_K, RS_M, 8);
        if (g_vandermonde_matrix == nullptr) { // ※ g_vandermonde_matrix を直接使用
            DebugLog(L"ERROR: cauchy_original_coding_matrix failed."); // ※ ログメッセージ修正
            g_matrix_initialized = false;
            return;
        }
        // 2. Vandermonde 行列をビット行列に変換
        //g_jerasure_matrix = jerasure_matrix_to_bitmatrix(RS_K, RS_M, 8, g_vandermonde_matrix);
        // free(vandermonde_matrix); // ※ g_vandermonde_matrix として保持するので、ここでは解放しない

        // ※ デバッグ用の g_vandermonde_matrix のコピーを作成し jerasure_matrix_to_bitmatrix に渡す ※
        int* temp_cauchy_for_bitmatrix = (int*)malloc(sizeof(int) * RS_M * RS_K); // m x k 行列
        if (temp_cauchy_for_bitmatrix == nullptr) {
            DebugLog(L"ERROR: Failed to allocate memory for temp_cauchy_for_bitmatrix.");
            free(g_vandermonde_matrix);
            g_vandermonde_matrix = nullptr;
            g_matrix_initialized = false;
            return;
        }
        memcpy(temp_cauchy_for_bitmatrix, g_vandermonde_matrix, sizeof(int) * RS_M * RS_K);
        g_jerasure_matrix = jerasure_matrix_to_bitmatrix(RS_K, RS_M, 8, temp_cauchy_for_bitmatrix);
        free(temp_cauchy_for_bitmatrix); // コピーはここで解放

        if (g_jerasure_matrix == nullptr) {
            DebugLog(L"ERROR: jerasure_matrix_to_bitmatrix failed."); // エラーログ修正
            // ※ g_vandermonde_matrix も解放しておく
            if (g_vandermonde_matrix != nullptr) {
                free(g_vandermonde_matrix);
                g_vandermonde_matrix = nullptr;
            }
            g_matrix_initialized = false;
            return;
        }

        g_matrix_initialized = true;
        //DebugLog(L"Jerasure Cauchy-based bitmatrix generated (k=" + std::to_wstring(RS_K) + L", m=" + std::to_wstring(RS_M) + L", w=8)"); // ※ ログメッセージ修正

        /*// --- [Debug] Print part of the g_vandermonde_matrix (Cauchy) after all operations ---
        std::wstringstream wss_g_mat;
        wss_g_mat << L"[Debug Init] g_vandermonde_matrix (Cauchy) (first 4xK or less): ";
        for(int r=0; r<std::min(4,RS_M); ++r) for(int c=0; c<RS_K; ++c) wss_g_mat << std::hex << std::setw(2) << std::setfill(L'0') << (int)g_vandermonde_matrix[r*RS_K+c] << L" ";
        DebugLog(wss_g_mat.str());
        // --- [Debug] ---*/

    });
}

void ListenForResendRequests() {
    DebugLog(L"ListenForResendRequests thread started.");

    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        DebugLog(L"Failed to create socket.");
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT_RESEND); // ポート8120で待機
    inet_pton(AF_INET, SERVER_IP_RESEND, &serverAddr.sin_addr); // 127.0.0.1で待機

    if (bind(udpSocket, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
        DebugLog(L"Failed to bind socket.");
        closesocket(udpSocket);
        return;
    }

    DebugLog(L"Waiting for RESEND signal...");

    while (receive_resend_Running) {
        try {
            char buffer[SIZE_PACKET_SIZE] = {0};
            sockaddr_in clientAddr{};
            int clientAddrSize = sizeof(clientAddr);

            // データ受信
            int bytesReceived = recvfrom(udpSocket, buffer, sizeof(buffer) - 1, 0,
                                         reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrSize);
            if (bytesReceived == SOCKET_ERROR) {
                if (WSAGetLastError() == WSAEINTR || WSAGetLastError() == WSAECONNRESET) {
                    DebugLog(L"recvfrom interrupted or connection reset in ListenForResendRequests.");
                    continue;
                }
                DebugLog(L"Failed to receive data in ListenForResendRequests. Error: " + std::to_wstring(WSAGetLastError()));
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Avoid busy loop on error
                continue;
            }

            buffer[bytesReceived] = '\0'; // Null-terminate
            std::string narrowReceivedData(buffer);
            std::wstring receivedData(narrowReceivedData.begin(), narrowReceivedData.end());


            // デバッグログ
            DebugLog(L"Received data: " + receivedData);

            // 該当のメッセージを受信した場合の処理
            if (receivedData == L"RESEND_DATA") {
                // SendWindowSize(); // This is obsolete. The client now proactively sends its final resolution.
                DebugLog(L"Received obsolete RESEND_DATA command. Ignoring.");
            }
        } catch (const std::exception& ex) {
            std::string narrowExceptionMsg(ex.what());
            std::wstring wideExceptionMsg(narrowExceptionMsg.begin(), narrowExceptionMsg.end());
            DebugLog(L"Exception in ListenForResendRequests: " + wideExceptionMsg);
            continue;
        }
    }

    // ソケットのクローズとWSAのクリーンアップ
    closesocket(udpSocket);
    DebugLog(L"ListenForResendRequests thread stopped.");
}

void CountBandW() {
    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        DebugLog(L"Failed to create socket.");
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(CLIENT_PORT_BANDWIDTH);
    inet_pton(AF_INET, CLIENT_IP_BANDWIDTH, &serverAddr.sin_addr);

    std::vector<char> data(BANDWIDTH_DATA_SIZE, 'A');

    //DebugLog("Start bandwidth measurement loop.");

    while (send_bandw_Running) {
        const char* startMessage = "START";
        sendto(udpSocket, startMessage, strlen(startMessage), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));
        //DebugLog("Sent START message");

        int offset = 0;
        while (offset < BANDWIDTH_DATA_SIZE) {
            int packetSize = std::min(DATA_PACKET_SIZE, BANDWIDTH_DATA_SIZE - offset);
            sendto(udpSocket, data.data() + offset, packetSize, 0, (sockaddr*)&serverAddr, sizeof(serverAddr));
            offset += packetSize;
        }

        //DebugLog("Sent 2MB payload.");

        const char* endMessage = "END";
        sendto(udpSocket, endMessage, strlen(endMessage), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));
        //DebugLog("Sent END message");

        std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }

    closesocket(udpSocket);
    DebugLog(L"Stopped bandwidth measurement.");
}


void ReceiveRawPacketsThread(int threadId) { // Renaming to ReceiveENetPacketsThread would be clearer
    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"] started.");

    // ENet should be initialized in wWinMain before starting this thread.
    // And deinitialized in wWinMain after this thread stops.

    ENetHost *server_host;
    ENetAddress address;

    address.host = ENET_HOST_ANY; // Listen on all available interfaces
    address.port = static_cast<enet_uint16>(SERVER_PORT_DATA + threadId); // Each thread listens on a different port

    server_host = enet_host_create(&address /* the address to bind the server host to */,
                                   32      /* allow up to 32 clients and/or outgoing connections */,
                                   2       /* allow up to 2 channels to be used, 0 and 1 */,
                                   0       /* assume any amount of incoming bandwidth */,
                                   0       /* assume any amount of outgoing bandwidth */);

    if (server_host == NULL) {
                DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: An error occurred while trying to create an ENet server host on port " + std::to_wstring(address.port));
        return;
    }

    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: ENet server started, listening on port " + std::to_wstring(address.port));


    ENetEvent event;
    UINT64 count = 0;
    std::chrono::steady_clock::time_point last_timeout_check = std::chrono::steady_clock::now();

    while (receive_raw_packet_Running) {
        
        // Service ENet events with a timeout (e.g., 10ms)
        int service_result = enet_host_service(server_host, &event, 10);

        if (service_result > 0) {
            switch (event.type) {
                case ENET_EVENT_TYPE_CONNECT:
                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Client connected from " +
                             std::to_wstring(event.peer->address.host) + L":" + std::to_wstring(event.peer->address.port));
                    OnNetworkReady(); // ★ 追加：接続成立トリガ
                    // You can store peer-specific data if needed:
                    // event.peer->data = (void*)("Some client info");
                    break;

                case ENET_EVENT_TYPE_RECEIVE:
                {
                    // DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Packet of length " + std::to_wstring(event.packet->dataLength) +
                    //          L" received from client on channel " + std::to_wstring(event.channelID));

                    if (event.packet->dataLength < 1) {
                        DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Received empty ENet packet.");
                        enet_packet_destroy(event.packet);
                        continue;
                    }

                    uint8_t packet_type = event.packet->data[0];
                    const uint8_t* payload_data = event.packet->data + 1;// payload_data is [WorkerTS (8B)][WGCCaptureTS (8B)][SIH][Data]
                    size_t payload_size = event.packet->dataLength - 1;

                    if (packet_type == PACKET_TYPE_FULL_SHARD) {
                        if (payload_size >= sizeof(uint64_t)) { // Check for WorkerTS
                            uint64_t worker_ts_val = *reinterpret_cast<const uint64_t*>(payload_data);//worker_ts_valはFECWorkerTSの値
                            

                            if (payload_size > sizeof(uint64_t)) { // Check for data after WorkerTS
                                const uint8_t* data_after_worker_ts = payload_data + sizeof(uint64_t);//data_after_worker_tsは[WGCCaptureTS (8B)][SIH][Data]の先頭アドレス
                                size_t size_after_worker_ts = payload_size - sizeof(uint64_t);//[WGCCaptureTS (8B)][SIH][Data]のサイズ

                                if (size_after_worker_ts >= (sizeof(uint64_t) + sizeof(ShardInfoHeader))) {//データが十分にある場合
                                    ParsedShardInfo parsedInfoLocal;
                                    const uint8_t* current_ptr_parse = data_after_worker_ts;//current_ptr_parseは[WGCCaptureTS (8B)][SIH][Data]の先頭アドレス

                                    parsedInfoLocal.wgcCaptureTimestamp = *reinterpret_cast<const uint64_t*>(current_ptr_parse); //parsedInfoLocal.wgcCaptureTimestampにWGCTSを設定
                                    current_ptr_parse += sizeof(uint64_t);//current_ptr_parseは[SIH][Data]の先頭アドレス

                                    const ShardInfoHeader* sih_parse = reinterpret_cast<const ShardInfoHeader*>(current_ptr_parse);
                                    current_ptr_parse += sizeof(ShardInfoHeader);//current_ptr_parseは[Data]の先頭アドレス

                                    parsedInfoLocal.frameNumber = ntohl(sih_parse->frameNumber);
                                    parsedInfoLocal.shardIndex = ntohl(sih_parse->shardIndex);
                                    parsedInfoLocal.totalDataShards = ntohl(sih_parse->totalDataShards);
                                    parsedInfoLocal.totalParityShards = ntohl(sih_parse->totalParityShards);
                                    parsedInfoLocal.originalDataLen = ntohl(sih_parse->originalDataLen);

                                    uint64_t rawpacket_enqueue_time_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                                    parsedInfoLocal.rawpacket_to_fec_timestamp = rawpacket_enqueue_time_ts;

                                    size_t shardDataSize_parse = data_after_worker_ts + size_after_worker_ts - current_ptr_parse;//shardDataSize_parseは[Data]のサイズ
                                    if (shardDataSize_parse > 0) {
                                        parsedInfoLocal.shardData.assign(current_ptr_parse, current_ptr_parse + shardDataSize_parse);
                                    }

                                    g_parsedShardQueue.enqueue(std::move(parsedInfoLocal));

                                    uint64_t enqueue_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                                    int64_t delay_ms = static_cast<int64_t>(enqueue_time_ms) - static_cast<int64_t>(worker_ts_val);
                                    if(count % 120 == 0)DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Full Shard - Delay (Server_FECWorker to Receiver_Enqueue time): " + std::to_wstring(delay_ms) + L" ms.");

                                } else {
                                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Full shard packet (after WorkerTS) too small for WGCCaptureTS and SIH. Size: " + std::to_wstring(size_after_worker_ts));
                                }
                            } else {
                                DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Full shard packet has WorkerTS but no further data (WGCCaptureTS etc.).");
                            }
                        } else {
                            DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Full shard packet too small for WorkerTS. Size: " + std::to_wstring(payload_size));
                        }
                        
                    } else if (packet_type == ENET_PACKET_TYPE_APP_FRAGMENT) {
                        if (payload_size < sizeof(ENetAppFragmentHeader)) {
                            DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: ENET_PACKET_TYPE_APP_FRAGMENT too small for header.");
                            enet_packet_destroy(event.packet);
                            continue;
                        }

                        ENetAppFragmentHeader app_header;
                        memcpy(&app_header, payload_data, sizeof(ENetAppFragmentHeader));                      
                        uint32_t original_packet_id = ntohl(app_header.original_packet_id_net);
                        uint16_t fragment_index = ntohs(app_header.fragment_index_net);
                        uint16_t total_fragments = ntohs(app_header.total_fragments_net);

                        const uint8_t* fragment_actual_data_ptr = payload_data + sizeof(ENetAppFragmentHeader);
                        size_t fragment_actual_data_size = payload_size - sizeof(ENetAppFragmentHeader);

                        // Data needed for reassembly, to be populated under lock
                        std::map<uint16_t, std::vector<uint8_t>> fragments_for_reassembly;
                        uint16_t total_fragments_for_reassembly = 0;
                        bool ready_for_reassembly = false;

                        { // Lock scope for appFragmentBuffers - minimized
                            std::lock_guard<std::mutex> lock(g_appFragmentBuffersMutex);
                            auto& assembly_state = appFragmentBuffers[original_packet_id];
                            if (assembly_state.fragments.empty()) { // First fragment for this ID
                                assembly_state.first_fragment_received_time = std::chrono::steady_clock::now();
                                assembly_state.total_fragments = total_fragments;
                            } else if (assembly_state.total_fragments != total_fragments) {
                                DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Mismatch in total_fragments for packet ID " + std::to_wstring(original_packet_id) + L". Discarding old fragments.");
                                assembly_state.fragments.clear();
                                assembly_state.first_fragment_received_time = std::chrono::steady_clock::now();
                                assembly_state.total_fragments = total_fragments;

                            }
                            // Store the fragment if not already received
                            if (assembly_state.fragments.find(fragment_index) == assembly_state.fragments.end()) {
                                assembly_state.fragments[fragment_index].assign(fragment_actual_data_ptr, fragment_actual_data_ptr + fragment_actual_data_size);
                            }

                            if (assembly_state.fragments.size() == assembly_state.total_fragments && assembly_state.total_fragments > 0) {
                                // All fragments received, prepare for reassembly outside the lock
                                fragments_for_reassembly = std::move(assembly_state.fragments); // Move data
                                total_fragments_for_reassembly = assembly_state.total_fragments;
                                ready_for_reassembly = true;
                                appFragmentBuffers.erase(original_packet_id); // Clean up
                            }
                        }

                        // Reassembly logic moved outside the lock
                        if (ready_for_reassembly) {
                            std::vector<uint8_t> reassembled_packet;
                            size_t total_reassembled_size = 0;
                            for(const auto& frag_pair : fragments_for_reassembly) {
                                total_reassembled_size += frag_pair.second.size();
                            }
                            reassembled_packet.reserve(total_reassembled_size);

                            bool reassembly_ok = true;
                            for (uint16_t i = 0; i < total_fragments_for_reassembly; ++i) {
                                auto it_frag = fragments_for_reassembly.find(i);
                                if (it_frag != fragments_for_reassembly.end()) {
                                    reassembled_packet.insert(reassembled_packet.end(), it_frag->second.begin(), it_frag->second.end());
                                } else {
                                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Missing fragment " + std::to_wstring(i) + L" for packet ID " + std::to_wstring(original_packet_id) + L" during reassembly.");
                                    reassembly_ok = false;
                                    break;
                                }
                            }
                            
                            if (reassembly_ok && !reassembled_packet.empty()) {
                                // reassembled_packet is [WorkerTS (8B)][WGCCaptureTS (8B)][SIH][Data]
                                if (reassembled_packet.size() >= sizeof(uint64_t)) { // Check for WorkerTS
                                    uint64_t worker_ts_val = *reinterpret_cast<const uint64_t*>(reassembled_packet.data());
                                    
                                    if (reassembled_packet.size() > sizeof(uint64_t)) { // Check for data after WorkerTS
                                        const uint8_t* data_after_worker_ts_frag = reassembled_packet.data() + sizeof(uint64_t);
                                        size_t size_after_worker_ts_frag = reassembled_packet.size() - sizeof(uint64_t);

                                        if (size_after_worker_ts_frag >= (sizeof(uint64_t) + sizeof(ShardInfoHeader))) {
                                            ParsedShardInfo parsedInfoLocalFrag;
                                            const uint8_t* current_ptr_parse_frag = data_after_worker_ts_frag;

                                            parsedInfoLocalFrag.wgcCaptureTimestamp = *reinterpret_cast<const uint64_t*>(current_ptr_parse_frag);
                                            current_ptr_parse_frag += sizeof(uint64_t);

                                            const ShardInfoHeader* sih_parse_frag = reinterpret_cast<const ShardInfoHeader*>(current_ptr_parse_frag);
                                            current_ptr_parse_frag += sizeof(ShardInfoHeader);

                                            parsedInfoLocalFrag.frameNumber = ntohl(sih_parse_frag->frameNumber);
                                            parsedInfoLocalFrag.shardIndex = ntohl(sih_parse_frag->shardIndex);
                                            parsedInfoLocalFrag.totalDataShards = ntohl(sih_parse_frag->totalDataShards);
                                            parsedInfoLocalFrag.totalParityShards = ntohl(sih_parse_frag->totalParityShards);
                                            parsedInfoLocalFrag.originalDataLen = ntohl(sih_parse_frag->originalDataLen);
                                            
                                            uint64_t rawpacket_enqueue_time_ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                                            parsedInfoLocalFrag.rawpacket_to_fec_timestamp = rawpacket_enqueue_time_ts;     

                                            size_t shardDataSize_parse_frag = data_after_worker_ts_frag + size_after_worker_ts_frag - current_ptr_parse_frag;
                                            if (shardDataSize_parse_frag > 0) {
                                                parsedInfoLocalFrag.shardData.assign(current_ptr_parse_frag, current_ptr_parse_frag + shardDataSize_parse_frag);
                                            }
                                            g_parsedShardQueue.enqueue(std::move(parsedInfoLocalFrag));

                                            uint64_t enqueue_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
                                            int64_t delay_ms = static_cast<int64_t>(enqueue_time_ms) - static_cast<int64_t>(worker_ts_val);
                                            if(count % 120 == 0)DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Fragment Shard - Delay (Server_FECWorker to Receiver_Enqueue time): " + std::to_wstring(delay_ms) + L" ms.");
                                        } else {
                                             DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Reassembled AppFragment ID " + std::to_wstring(original_packet_id) + L" (after WorkerTS) too small for WGCCaptureTS and SIH. Size: " + std::to_wstring(size_after_worker_ts_frag));
                                        }
                                    } else {
                                        DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Reassembled packet ID " + std::to_wstring(original_packet_id) + L" has WorkerTS but no further data.");
                                    }
                                } else {
                                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Reassembled packet ID " + std::to_wstring(original_packet_id) + L" too small for WorkerTS. Size: " + std::to_wstring(reassembled_packet.size()));
                                }
                            } else if (!reassembly_ok) {
                                DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Failed to reassemble packet ID " + std::to_wstring(original_packet_id));
                            }
                        } // End of reassembly logic (outside lock)
                    } else {
                        DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Unknown packet type " + std::to_wstring(packet_type));
                    }
                    enet_packet_destroy(event.packet);
                    break;
                }
                case ENET_EVENT_TYPE_DISCONNECT:
                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Client disconnected.");
                    // event.peer->data = NULL; // Reset peer data if you stored something
                    break;

                case ENET_EVENT_TYPE_NONE:
                    break;
            }
        } else if (service_result < 0) {
            DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Error in enet_host_service: " + std::to_wstring(service_result));
            // Potentially break or handle error
        }
        // Else, service_result == 0, meaning no event occurred within the timeout

        // Periodically check for timed-out fragment assemblies
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_timeout_check) >= std::chrono::seconds(1)) {
            // ClearTimedOutAppFragments is now internally locked
            ClearTimedOutAppFragments(); 
            last_timeout_check = now;
        }
        count++;
    }

    if (server_host) {
        enet_host_destroy(server_host);
    }
    
    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"] stopped.");
}

void FecWorkerThread(int threadId) {
    DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"] started.");
    UINT64 count = 0;
    const std::chrono::milliseconds EMPTY_QUEUE_WAIT_MS(1);

    while (g_fec_worker_Running || g_parsedShardQueue.size_approx() > 0) { // Process remaining items after flag is false
        auto fec_worker_thread_start = std::chrono::system_clock::now();
        uint64_t fec_worker_thread_start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(fec_worker_thread_start.time_since_epoch()).count();
        ParsedShardInfo parsedInfo;

        if(g_parsedShardQueue.size_approx() == 0){
            continue;
        }

        if (g_parsedShardQueue.try_dequeue(parsedInfo)) {
            uint64_t packetTimestamp = parsedInfo.wgcCaptureTimestamp;
            int frameNumber = parsedInfo.frameNumber; // Already host order
            int shardIndex = parsedInfo.shardIndex;   // Already host order
            uint32_t originalDataLenHost = parsedInfo.originalDataLen; // Already host order
            std::vector<uint8_t>& payload = parsedInfo.shardData; // Use reference or move if appropriate

            bool tryDecode = false;
            std::map<uint32_t, std::vector<uint8_t>> shardsForDecodeAttempt;
            FrameMetadata currentFrameMetaForAttempt;

            auto fec_worker_thread_dequeue = std::chrono::system_clock::now();
            uint64_t fec_worker_thread_dequeue_ts = std::chrono::duration_cast<std::chrono::milliseconds>(fec_worker_thread_dequeue.time_since_epoch()).count();

            if(count % 120 == 0)DebugLog(L"FecWorkerThread: Queue Size " + std::to_wstring(g_parsedShardQueue.size_approx()));
            if(count % 120 == 0)DebugLog(L"FecWorkerThread: RawPackets to FECWorker " + std::to_wstring(static_cast<int>(fec_worker_thread_dequeue_ts) - static_cast<int>(parsedInfo.rawpacket_to_fec_timestamp)) + L" ms");

            { // Metadata and FrameBuffer scope
                // Metadata access first
                {
                    std::lock_guard<std::mutex> metaLock(g_frameMetadataMutex);
                    if (g_frameMetadata.find(frameNumber) == g_frameMetadata.end()) {
                        g_frameMetadata[frameNumber] = {packetTimestamp, originalDataLenHost};
                    }
                    // Parameter check (totalDataShards and totalParityShards from parsedInfo vs RS_K/RS_M)
                    if (parsedInfo.totalDataShards != RS_K || parsedInfo.totalParityShards != RS_M) {
                        DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"]: Mismatch in FEC parameters! Packet K/M: " +
                                 std::to_wstring(parsedInfo.totalDataShards) + L"/" + std::to_wstring(parsedInfo.totalParityShards) +
                                 L", Client K/M: " + std::to_wstring(RS_K) + L"/" + std::to_wstring(RS_M) +
                                 L". Frame: " + std::to_wstring(frameNumber) + L", Shard: " + std::to_wstring(shardIndex));
                        continue; // Skip this shard
                    }
                }

                std::lock_guard<std::mutex> bufferLock(g_frameBufferMutex);
                if (g_frameBuffer.find(frameNumber) == g_frameBuffer.end()) {
                    g_frameBuffer[frameNumber] = std::map<int, std::vector<uint8_t>>();
                }

                if (g_frameBuffer[frameNumber].find(shardIndex) == g_frameBuffer[frameNumber].end()) {
                    g_frameBuffer[frameNumber][shardIndex] = std::move(payload); // payload is moved here
                    // DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"]: Added Shard F#" + std::to_wstring(frameNumber) + L" Idx:" + std::to_wstring(shardIndex) + L". Current count: " + std::to_wstring(g_frameBuffer[frameNumber].size()));
                }

                if (g_frameBuffer.count(frameNumber) && g_frameBuffer[frameNumber].size() >= RS_K) {
                    std::lock_guard<std::mutex> metaLock(g_frameMetadataMutex); // Lock metadata while accessing
                    if (g_frameMetadata.count(frameNumber)) {
                        currentFrameMetaForAttempt = g_frameMetadata[frameNumber]; // This is a copy
                        for (auto& pair_entry : g_frameBuffer[frameNumber]) {
                            shardsForDecodeAttempt[static_cast<uint32_t>(pair_entry.first)] = std::move(pair_entry.second);
                        }
                        g_frameBuffer.erase(frameNumber);
                        g_frameMetadata.erase(frameNumber);
                        tryDecode = true;
                    } else {
                        DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"]: Metadata missing for F#" + std::to_wstring(frameNumber) + L" when K shards reached. Cleaning buffer.");
                        g_frameBuffer.erase(frameNumber);
                    }
                } else if (g_frameBuffer.count(frameNumber)) {
                    if(count % 200 == 0)DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"]: F#" + std::to_wstring(frameNumber) + L" has " + std::to_wstring(g_frameBuffer[frameNumber].size()) + L" shards, (less than K=" + std::to_wstring(RS_K) + L") waiting for more.");
                }
            } // End Metadata and FrameBuffer scope

            if (tryDecode && !shardsForDecodeAttempt.empty()) {
                std::vector<uint8_t> decodedFrameData;
                // currentFrameMetaForAttempt.originalDataLen is already host order
                uint32_t originalLenForDecode = currentFrameMetaForAttempt.originalDataLen;

                if (g_matrix_initialized && DecodeFEC_Jerasure(shardsForDecodeAttempt, RS_K, RS_M, originalLenForDecode, decodedFrameData, g_jerasure_matrix)) {
                    // Save H264 file after successful FEC decode, before enqueue
                    // SaveH264ToFile_NUM(decodedFrameData, "decoded_frame");

                    H264Frame frame_to_decode;
                    frame_to_decode.timestamp = currentFrameMetaForAttempt.firstTimestamp;
                    frame_to_decode.frameNumber = frameNumber;
                    frame_to_decode.data = std::move(decodedFrameData);

                    g_h264FrameQueue.enqueue(std::move(frame_to_decode));
                    
                    // 追加のデバッグログ
                    auto fec_worker_thread_end = std::chrono::system_clock::now();
                    uint64_t fec_worker_thread_end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(fec_worker_thread_end.time_since_epoch()).count();
                    int64_t elapsed_fec_worker_thread = static_cast<int64_t>(fec_worker_thread_end_ts) - static_cast<int64_t>(fec_worker_thread_start_ts);
                    if(count % 120 == 0)DebugLog(L"FEC Worker Thread Process Time: " + std::to_wstring(elapsed_fec_worker_thread) + L" ms");
                } else {
                    DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"]: FEC Decode failed for frame " + std::to_wstring(frameNumber));
                }
            }
            count++;
        } else {
            // Queue was empty
            if (!g_fec_worker_Running && g_parsedShardQueue.size_approx() == 0) { // Exit if flag is false AND queue is empty
                break;
            }
            std::this_thread::sleep_for(EMPTY_QUEUE_WAIT_MS);
        }
    }
    DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"] stopped.");
}


ThreadConfig getOptimalThreadConfig(){
    ThreadConfig config;

    config.receiver = 5;
    config.fec = 4;
    config.decoder = 1;
    config.render = 1;
    config.RS_K = 8;
    config.RS_M = 2;

    return config;
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow) {
    // Enforce GPU policy first
    if (!EnforceGpuPolicyOrExit()) {
        return 0; // Exit early per policy
    }

    // SetProcessDpiAwarenessContextが使えない場合はSetProcessDPIAwareを使う
    HMODULE hUser32 = LoadLibraryA("user32.dll");
    if (hUser32) {
        typedef BOOL (WINAPI *SetDpiAwarenessContextFunc)(HANDLE);
        SetDpiAwarenessContextFunc pSetDpiAwarenessContext = (SetDpiAwarenessContextFunc)GetProcAddress(hUser32, "SetProcessDpiAwarenessContext");
        if (pSetDpiAwarenessContext) {
            pSetDpiAwarenessContext((HANDLE)-4/*DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2*/);
        } else {
            SetProcessDPIAware();
        }
        FreeLibrary(hUser32);
    } else {
        SetProcessDPIAware();
    }
    InitializeRSMatrix();

    // Get executable path and set up logging
    char exePath[MAX_PATH];
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    std::string exeDir = std::string(exePath);
    exeDir = exeDir.substr(0, exeDir.find_last_of("\\/"));

    // Log file management
    std::string logFilePath = exeDir + "\\debuglog_client.log";
    std::time_t now = std::time(nullptr);
    std::tm localTime;
    localtime_s(&localTime, &now);
    char timeBuffer[64];
    std::strftime(timeBuffer, sizeof(timeBuffer), "%Y%m%d%H%M%S", &localTime);
    std::string backupFilePath = exeDir + "\\" + timeBuffer + "_debuglog_client.log.back";

    // Backup current log file
    if (std::filesystem::exists(logFilePath)) {
        try {
            std::filesystem::rename(logFilePath, backupFilePath);
        } catch (const std::filesystem::filesystem_error& e) {
            DebugLog(L"Failed to rename log file: " + std::wstring(e.what(), e.what() + strlen(e.what())));
        }
    }

    // Clean up old log files - keep only 5 backup files
    std::vector<std::filesystem::path> backupFiles;
    for (const auto& entry : std::filesystem::directory_iterator(exeDir)) {
        if (entry.is_regular_file()) {
            std::string fileName = entry.path().filename().string();
            std::regex pattern(R"((\d{14})_debuglog_client\.log\.back)");
            if (std::regex_match(fileName, pattern)) {
                backupFiles.push_back(entry.path());
            }
        }
    }
    
    // Sort by filename (timestamp), newest first
    std::sort(backupFiles.begin(), backupFiles.end(), std::greater<std::filesystem::path>());
    
    // Remove files beyond the 5 most recent ones
    if (backupFiles.size() > 5) {
        for (size_t i = 5; i < backupFiles.size(); ++i) {
            try {
                std::filesystem::remove(backupFiles[i]);
            } catch (const std::filesystem::filesystem_error& e) {
                DebugLog(L"Failed to remove old log file: " + std::wstring(e.what(), e.what() + strlen(e.what())));
            }
        }
    }

    // === 非同期ロガー初期化 ===
    // 既存と同じファイル名で良い。ODSは有効、キュー容量は適宜調整。
    DebugLogAsync::Init(L"debuglog_client.log", /*queueCapacity=*/16384, /*alsoOutputDebugString=*/true);
    DebugLog(L"Async DebugLog initialized.");

    // Initialize network and DirectX
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        DebugLog(L"WSAStartup failed in wWinMain.");
        return 1;
    }

    if (enet_initialize() != 0) {
        DebugLog(L"ENet initialization failed in wWinMain.");
        WSACleanup();
        return 1;
    }

    timeBeginPeriod(1);

    // Initialize window and DirectX
    if (!InitWindow(hInstance, nCmdShow)) {
        timeEndPeriod(1);
        WSACleanup();
        enet_deinitialize();
        return -1;
    }

    if (!InitD3D()) {
        DebugLog(L"wWinMain: Failed to initialize Direct3D for rendering after InitWindow.");
        return -1;
    }

    // After InitWindow(...) and InitD3D() == true
    RECT rc{}; GetClientRect(g_hWnd, &rc);
    int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
    int tw, th; SnapToKnownResolution(cw, ch, tw, th);
    currentResolutionWidth  = tw;
    currentResolutionHeight = th;

    // Enqueue an initial resize for the render thread (to align swap-chain)
    g_pendingResize.w.store(tw, std::memory_order_relaxed);
    g_pendingResize.h.store(th, std::memory_order_relaxed);
    g_pendingResize.has.store(true, std::memory_order_release);

    // Force-send to server now (single gate):
    OnResolutionChanged_GatedSend(tw, th, /*force=*/true);

    // Initialize CUDA and NVDEC
    // Per Yuki's recommendation, use the primary context to ensure Runtime and Driver APIs work together.
    cudaSetDevice(0); // Ensure runtime API is targeting the correct device.
    CUdevice cuDev = 0;
    cuDeviceGet(&cuDev, 0);
    CUcontext cuContext = nullptr;
    cuDevicePrimaryCtxRetain(&cuContext, cuDev);

    g_frameDecoder = std::make_unique<FrameDecoder>(cuContext, g_d3d12Device.Get());
    if (!g_frameDecoder->Init()) {
        DebugLog(L"wWinMain: Failed to initialize FrameDecoder.");
        return -1;
    }

    // Start worker threads
    std::thread bandwidthThread(CountBandW);
    std::thread resendThread(ListenForResendRequests);

    std::vector<std::thread> receiverThreads;
    for (int i = 0; i < getOptimalThreadConfig().receiver; ++i) {
        receiverThreads.emplace_back(ReceiveRawPacketsThread, i);
    }

    std::vector<std::thread> fecWorkerThreads;
    for (int i = 0; i < getOptimalThreadConfig().fec; ++i) {
        fecWorkerThreads.emplace_back(FecWorkerThread, i);
    }

    std::vector<std::thread> nvdecThreads;
    for (int i = 0; i < getOptimalThreadConfig().decoder; ++i) {
        nvdecThreads.emplace_back(NvdecThread, i);
    }

    // The app_running_atomic is now a global atomic defined in Globals.cpp
    // The windowSenderThread is removed as it's part of the old logic.

    // Main render loop
    const double TARGET_FPS = 60.0;
    const std::chrono::nanoseconds TARGET_FRAME_DURATION(static_cast<long long>(1'000'000'000.0 / TARGET_FPS));
    auto lastFrameRenderTime = std::chrono::high_resolution_clock::now();

    MSG msg = {};
    while (msg.message != WM_QUIT) {
        // Process all pending messages in the queue first.
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                break;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (msg.message == WM_QUIT) {
            break;
        }

        // After processing messages, render a frame, respecting the frame rate.
        // This ensures rendering continues even during a message-heavy event like resizing.
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto timeSinceLastRender = currentTime - lastFrameRenderTime;

        if (timeSinceLastRender >= TARGET_FRAME_DURATION) {
            RenderFrame();
            lastFrameRenderTime = currentTime;
        } else {
            // Yield CPU time if we are ahead of schedule to avoid spinning.
            auto timeToWait = TARGET_FRAME_DURATION - timeSinceLastRender;
            long long msToWait = std::chrono::duration_cast<std::chrono::milliseconds>(timeToWait).count();
            if (msToWait > 1) {
                Sleep(static_cast<DWORD>(msToWait - 1));
            }
        }
    }

    // Centralized Cleanup
    DebugLog(L"Exited message loop. Initiating final resource cleanup...");
    AppThreads appThreads{};
    appThreads.bandwidthThread = &bandwidthThread;
    appThreads.resendThread = &resendThread;
    appThreads.receiverThreads = &receiverThreads;
    appThreads.fecWorkerThreads = &fecWorkerThreads;
    appThreads.nvdecThreads = &nvdecThreads;
    // appThreads.windowSenderThread is removed

    // This single call handles joining all threads and releasing all resources idempotently.
    ReleaseAllResources(appThreads);
    
    DebugLog(L"Cleanup complete. Exiting wWinMain.");

    // === 非同期ロガーを安全に停止（全ログを flush） ===
    Shutdown();
    return 0;
}
