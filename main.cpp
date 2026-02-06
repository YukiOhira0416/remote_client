#ifndef NOMINMAX
#define NOMINMAX 
#endif 
// Force _WIN32_WINNT to be at least Windows Vista (0x0600) for WSARecvMsg.
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0600 
#include <winsock2.h>
#include <cstdint> // For uint64_t
#include <chrono>
#include <ws2tcpip.h>
#include <mswsock.h> // Required for WSARecvMsg and WSASendMsg
#include <windows.h>
#include <mmsystem.h>
#include <thread>
#include <atomic>
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
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include "DebugLog.h"
#include "ReedSolomon.h"
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
#include "TimeSyncClient.h"
#include "AudioClient.h"
#include "InputSender.h"
#include <QApplication>
#include <QTimer>
#include "main_window.h"
using namespace DebugLogAsync;

// ==== [GPU Policy Support - BEGIN] ====
#include <dxgi1_6.h>
#pragma comment(lib, "dxgi.lib")
#include <vector>
#include <string>
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
#define SEND_IP_BANDWIDTH "192.168.0.2"
//#define SEND_IP_BANDWIDTH "127.0.0.1"
#define RECEIVE_PORT_DATA 8130 // パケット受信用のポート番号
#define SEND_PORT_BANDWIDTH 8200// 帯域幅測定用のポート番号
#define BANDWIDTH_DATA_SIZE 60 * 1024  // 60KB(帯域幅測定時のデータサイズ)
#define DATA_PACKET_SIZE 1300 // UDPパケットサイズ
#define WSARECV_BUFFER_SIZE 65000
#define RECEIVE_IP_REBOOT "0.0.0.0"
//#define RECEIVE_IP_REBOOT "127.0.0.1"
#define RECEIVE_PORT_REBOOT_START 8150
#define RECEIVE_PORT_REBOOT_END 8151

// FEC worker threads and control variables
std::atomic<bool> send_bandw_Running = true;
std::atomic<bool> receive_resend_Running = true;
std::atomic<bool> receive_raw_packet_Running = true;

// Mutexes for synchronization
std::mutex logMutex;
std::mutex processvideosequenceMutex;
std::mutex g_frameBufferMutex;

// Frame and fragment related data structures
std::unordered_map<int, std::unordered_map<int, std::vector<uint8_t>>> g_frameBuffer;
std::unordered_map<int, int> expectedFrameCounts;

// Fragment assembly data structures
std::unordered_map<uint64_t, std::unordered_map<uint16_t, std::vector<uint8_t>>> g_fragmentAssemblyBuffer;
std::mutex g_fragmentAssemblyMutex;
std::unordered_map<uint64_t, uint16_t> g_expectedFragmentsCount;
std::unordered_map<uint64_t, uint16_t> g_receivedFragmentsCount;
std::unordered_map<uint64_t, size_t> g_accumulatedFragmentDataSize;

// Frame metadata structure and management
struct FrameMetadata {
    uint64_t firstTimestamp = 0;
    uint32_t originalDataLen = 0;
    uint64_t first_seen_time_ms = 0;
    // FEC parameters (per-frame, from ShardInfoHeader)
    uint32_t rs_k = 0;
    uint32_t rs_m = 0;
};
std::unordered_map<int, FrameMetadata> g_frameMetadata;
std::mutex g_frameMetadataMutex;

// Fragment timing and frame ID management
std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> g_fragmentFirstPacketTime;
std::atomic<uint64_t> g_rgbaFrameIdCounter{0};

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
    std::unordered_map<uint16_t, std::vector<uint8_t>> fragments; // key: fragment_index (host byte order)
    uint16_t total_fragments = 0;
    std::chrono::steady_clock::time_point first_fragment_received_time;
};

// For ENet app fragments - Define it here if not defined elsewhere
const std::chrono::seconds APP_FRAGMENT_ASSEMBLY_TIMEOUT(5); 

// Assume these are declared globally, accessible by this function
extern std::atomic<bool> receive_raw_packet_Running; // Controls the main loop

// Buffer for reassembling application-level ENet fragments
std::unordered_map<uint32_t, AppFragmentAssemblyState> appFragmentBuffers; // key: original_packet_id (host byte order)
std::mutex g_appFragmentBuffersMutex; // Mutex to protect appFragmentBuffers

// New struct for parsed shard information
struct ParsedShardInfo {
    uint64_t wgcCaptureTimestamp;
    uint64_t server_fec_timestamp;
    uint32_t frameNumber;        // Host byte order
    uint32_t shardIndex;         // Host byte order
    uint32_t totalDataShards;    // Host byte order
    uint32_t totalParityShards;  // Host byte order
    uint32_t originalDataLen;    // Host byte order
    std::vector<uint8_t> shardData;
    uint64_t generation;
};

// === 新規：ネットワーク準備・解像度ペンディング管理 ===
std::atomic<bool> g_networkReady(false);
std::atomic<bool> g_pendingResolutionValid(false);
std::atomic<int>  g_pendingW(0), g_pendingH(0);
std::atomic<bool> g_didInitialAnnounce(false);

// HEVC 出力ファイル保存用のミューテックス
std::mutex hevcoutputMutex;

void SendFinalResolution(int width, int height);
void ClearReorderState();

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
        item_out = std::move(const_cast<T&>(m_pq.top()));
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

ThreadConfig getOptimalThreadConfig(){
    ThreadConfig config;

    config.receiver = 4;
    config.fec = 3;
    config.decoder = 1;
    config.render = 1;
    config.RS_K = 14;
    config.RS_M = 8;

    return config;
}


// この関数はネットワーク接続確立のタイミングで、クライアントの解像度情報をサーバーに確実に伝達するために使用される。
void OnNetworkReady()
{
    g_networkReady = true;
    DebugLog(L"OnNetworkReady: network is ready.");

    bool already_announced = g_didInitialAnnounce.exchange(true);
    if (already_announced) {
        DebugLog(L"OnNetworkReady: initial announce already done, skipping.");
        return;
    }

    if (g_pendingResolutionValid.load()) {
        int w = g_pendingW.load(), h = g_pendingH.load();
        DebugLog(L"OnNetworkReady: flushing pending resolution.");
        SendFinalResolution(w, h);
        ClearReorderState();
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
            }
        }
    }).detach();
}



// Helper function to check for and clear timed-out fragment assembly buffers
void ClearTimedOutAppFragments() {
    std::lock_guard<std::mutex> lock(g_appFragmentBuffersMutex); // Protect access to appFragmentBuffers
    auto now = std::chrono::steady_clock::now();
    for (auto it = appFragmentBuffers.begin(); it != appFragmentBuffers.end(); /* manual increment */) {
        if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second.first_fragment_received_time) > APP_FRAGMENT_ASSEMBLY_TIMEOUT) {
            DebugLog(L"ClearTimedOutAppFragments: Timing out assembly for original_packet_id " + std::to_wstring(it->first));
            it = appFragmentBuffers.erase(it);
        } else {
            ++it;
        }
    }
}



void SaveEncodedStreamToFile(const std::vector<uint8_t>& prepared_encodedBuffer, const std::string& baseName) {
    // ★ スレッドIDをログに追加して、どのスレッドからの呼び出しから分かるようにする
    std::wstringstream wss_log_prefix;
    wss_log_prefix << L"SaveEncodedStreamToFile (Thread " << std::this_thread::get_id() << L"): ";

    if (prepared_encodedBuffer.empty()) {
        DebugLog(wss_log_prefix.str() + L"prepared_encodedBuffer is empty. Skipping file save.");
        return;
    }

    static std::atomic<int> fileCounter(0); // ★ スレッドセーフなカウンターに変更
    int currentFileCounterValue = fileCounter.fetch_add(1); // ★ アトミックにインクリメントし、古い値を取得

    // 実行ファイルのパスからディレクトリを取得
    char exePath[MAX_PATH];
    if (GetModuleFileNameA(NULL, exePath, MAX_PATH) == 0) {
        DebugLog(wss_log_prefix.str() + L"Failed to get executable path.");
        return;
    }

    std::filesystem::path folderPath(exePath);
    folderPath.remove_filename();
    folderPath /= "ffplay";

    if (!std::filesystem::exists(folderPath)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(folderPath, ec)) {
            DebugLog(wss_log_prefix.str() + L"Failed to create directory: " + folderPath.wstring() + L" Error: " + ConvertToWString(ec.message()));
            return;
        }

    }

    // 番号付きファイル名を生成
    std::ostringstream oss;
    oss << baseName << "_" << std::setw(4) << std::setfill('0') << currentFileCounterValue << ".hevc";
    std::string numberedFilename = (folderPath / oss.str()).string();

    // ファイルに書き込み
    std::ofstream ofs(numberedFilename, std::ios::binary);
    if (!ofs) {
        DebugLog(wss_log_prefix.str() + L"Error opening file " + ConvertToWString(numberedFilename));
        return;
    }
    ofs.write(reinterpret_cast<const char*>(prepared_encodedBuffer.data()), prepared_encodedBuffer.size());
    ofs.close();

    // std::cout はデバッグログには出ないので、DebugLog を使う
    DebugLog(wss_log_prefix.str() + L"Saved " + std::to_wstring(prepared_encodedBuffer.size()) + L" bytes to " + ConvertToWString(numberedFilename) + L" (Counter: " + std::to_wstring(currentFileCounterValue) + L")");
}


void CountBandW() {
    {
        std::unique_lock<std::mutex> lock(g_windowShownMutex);
        g_windowShownCv.wait(lock, [] { return g_windowShown; });
    }

    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        DebugLog(L"Failed to create socket.");
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SEND_PORT_BANDWIDTH);
    inet_pton(AF_INET, SEND_IP_BANDWIDTH, &serverAddr.sin_addr);

    std::vector<char> data(BANDWIDTH_DATA_SIZE, 'A');

    const char* endMessage = "END";
    sendto(udpSocket, endMessage, strlen(endMessage), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    int w = currentResolutionWidth.load();
    int h = currentResolutionHeight.load();
    SendFinalResolution(w, h);

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

        const char* endMessage = "END";
        sendto(udpSocket, endMessage, strlen(endMessage), 0, (sockaddr*)&serverAddr, sizeof(serverAddr));

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    closesocket(udpSocket);
    DebugLog(L"Stopped bandwidth measurement.");
}


void ReceiveRawPacketsThread(int threadId) { // Renaming to ReceiveENetPacketsThread would be clearer
    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"] started.");

    // === 追加: 再構成作業用リングバッファ（このスレッド専用） ===
    // 断片再構成後の最大想定サイズに合わせて reserve。
    // 値は運用環境に応じて調整。まずは安全側で 128 KiB。
    static constexpr size_t kRingSlots = 256;
    static constexpr size_t kRingSlotCap = 128 * 1024;

    struct ReassembleSlot {
        std::vector<uint8_t> buf; // 一度だけ確保して使い回す
    };
    std::vector<ReassembleSlot> rb(kRingSlots);
    for (auto& s : rb) s.buf.reserve(kRingSlotCap);
    size_t rbHead = 0;

    auto nextSlot = [&]()->std::vector<uint8_t>& {
        auto& v = rb[rbHead].buf;
        rbHead = (rbHead + 1) % kRingSlots;
        return v;
    };
    // === 追加ここまで ===

    // ENet should be initialized in wWinMain before starting this thread.
    // And deinitialized in wWinMain after this thread stops.

    ENetHost *server_host;
    ENetAddress address;

    address.host = ENET_HOST_ANY; // Listen on all available interfaces
    address.port = static_cast<enet_uint16>(RECEIVE_PORT_DATA + threadId); // Each thread listens on a different port

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
        int service_result;
        service_result = enet_host_service(server_host, &event, 0);

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
                    const uint8_t* payload_data = event.packet->data + 1;// payload_data is [FEC完了時タイムスタンプ]{[WGCキャプチャ時][ShardInfoHeader][実際のデータ]}
                    size_t payload_size = event.packet->dataLength - 1;

                    if (packet_type == PACKET_TYPE_FULL_SHARD) {
                        if (payload_size >= sizeof(uint64_t)) { // Check for WorkerTS
                            uint64_t worker_ts_val = *reinterpret_cast<const uint64_t*>(payload_data);//worker_ts_valはFEC完了時タイムスタンプの値
                            
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
                                    parsedInfoLocal.server_fec_timestamp = worker_ts_val;

                                    size_t shardDataSize_parse = data_after_worker_ts + size_after_worker_ts - current_ptr_parse;//shardDataSize_parseは[Data]のサイズ
                                    if (shardDataSize_parse > 0) {
                                        parsedInfoLocal.shardData.resize(shardDataSize_parse);
                                        std::memcpy(parsedInfoLocal.shardData.data(),
                                                    current_ptr_parse,
                                                    shardDataSize_parse);
                                    }
                                    parsedInfoLocal.generation = g_streamGeneration.load(std::memory_order_acquire);
                                    g_parsedShardQueue.enqueue(std::move(parsedInfoLocal));

                                    auto receive_data_time = std::chrono::system_clock::now();
                                    uint64_t receive_data_time_ts = std::chrono::duration_cast<std::chrono::milliseconds>(receive_data_time.time_since_epoch()).count();

                                    if (count % 60 == 0) DebugLog(L"ReceiveRawPacketsThread: Server FEC End to Client Receive End latency (ms): " +
                                                     std::to_wstring(receive_data_time_ts - worker_ts_val + g_TimeOffsetNs / 1000000) + L" ms");

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
                        std::unordered_map<uint16_t, std::vector<uint8_t>> fragments_for_reassembly;
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
                            // === 変更: 一時 reassembled_packet の生成をやめ、リングバッファを使う ===
                            // 合計サイズを先に算出（既存処理と同じ）
                            size_t total_reassembled_size = 0;
                            for (const auto& kv : fragments_for_reassembly)
                                total_reassembled_size += kv.second.size();

                            std::vector<uint8_t>& reassembled = nextSlot();
                            if (reassembled.capacity() < total_reassembled_size) {
                                // まれに閾値超えた場合のみ再確保（ログは残す）
                                DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) +
                                         L"]: ring slot grow from " + std::to_wstring(reassembled.capacity()) +
                                         L" to " + std::to_wstring(total_reassembled_size));
                                reassembled.reserve(total_reassembled_size);
                            }
                            reassembled.resize(total_reassembled_size);

                            size_t off = 0;
                            bool reassembly_ok = true;
                            for (uint16_t i = 0; i < total_fragments_for_reassembly; ++i) {
                                auto it = fragments_for_reassembly.find(i);
                                if (it == fragments_for_reassembly.end()) {
                                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) +
                                             L"]: Missing fragment " + std::to_wstring(i) + L" during reassembly.");
                                    reassembly_ok = false;
                                    break;
                                }
                                const auto& frag = it->second;
                                std::memcpy(reassembled.data() + off, frag.data(), frag.size());
                                off += frag.size();
                            }

                            if (reassembly_ok && !reassembled.empty()) {
                                // 以降のパースは従来どおり（WorkerTS / WGCTS / SIH / Data）
                                if (reassembled.size() >= sizeof(uint64_t)) {
                                    uint64_t worker_ts_val = *reinterpret_cast<const uint64_t*>(reassembled.data());
                                    if (reassembled.size() > sizeof(uint64_t)) {
                                        const uint8_t* data_after_worker = reassembled.data() + sizeof(uint64_t);
                                        size_t size_after_worker = reassembled.size() - sizeof(uint64_t);

                                        if (size_after_worker >= (sizeof(uint64_t) + sizeof(ShardInfoHeader))) {
                                            ParsedShardInfo parsed{};
                                            const uint8_t* cur = data_after_worker;

                                            parsed.wgcCaptureTimestamp = *reinterpret_cast<const uint64_t*>(cur);
                                            cur += sizeof(uint64_t);

                                            const ShardInfoHeader* sih = reinterpret_cast<const ShardInfoHeader*>(cur);
                                            cur += sizeof(ShardInfoHeader);

                                            parsed.frameNumber       = ntohl(sih->frameNumber);
                                            parsed.shardIndex        = ntohl(sih->shardIndex);
                                            parsed.totalDataShards   = ntohl(sih->totalDataShards);
                                            parsed.totalParityShards = ntohl(sih->totalParityShards);
                                            parsed.originalDataLen   = ntohl(sih->originalDataLen);
                                            parsed.server_fec_timestamp = worker_ts_val;
                                            parsed.generation = g_streamGeneration.load(std::memory_order_acquire);

                                            const size_t shardDataSize = (data_after_worker + size_after_worker) - cur;
                                            if (shardDataSize > 0) {
                                                // ※ここは所有権を他スレッドに渡すため従来どおり vector にコピー（安全最優先）
                                                parsed.shardData.resize(shardDataSize);
                                                std::memcpy(parsed.shardData.data(), cur, shardDataSize);
                                            }

                                            g_parsedShardQueue.enqueue(std::move(parsed));

                                            auto receive_data_time = std::chrono::system_clock::now();
                                            uint64_t receive_data_time_ts = std::chrono::duration_cast<std::chrono::milliseconds>(receive_data_time.time_since_epoch()).count();
                                            if (count % 60 == 0) DebugLog(L"ReceiveRawPacketsThread: Server FEC End to Client Receive End latency (ms): " +
                                                     std::to_wstring(receive_data_time_ts - worker_ts_val + g_TimeOffsetNs / 1000000) + L" ms");
                                        } else {
                                            DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) +
                                                     L"]: Reassembled size too small for headers.");
                                        }
                                    } else {
                                        DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) +
                                                 L"]: Reassembled has WorkerTS but no further data.");
                                    }
                                } else {
                                    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) +
                                             L"]: Reassembled too small for WorkerTS.");
                                }
                            } else if (!reassembly_ok) {
                                DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"]: Failed to reassemble packet ID " + std::to_wstring(original_packet_id));
                            }
                        }
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
        // カウンタを60,000でリセット（約1000分=16.7時間分のログ）
        count = (count + 1) % 60000;
    }

    if (server_host) {
        enet_host_destroy(server_host);
    }
    
    DebugLog(L"ReceiveRawPacketsThread [" + std::to_wstring(threadId) + L"] stopped.");
}

void FecWorkerThread(int threadId) {
    DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"] started.");
    const std::chrono::milliseconds EMPTY_QUEUE_WAIT_MS(1);
    const uint64_t ASSEMBLY_TIMEOUT_MS = 300; // Timeout for clearing stale frames
    UINT64 processed_count = 0;
    while (g_fec_worker_Running || g_parsedShardQueue.size_approx() > 0) {
        ParsedShardInfo parsedInfo;

        if (!g_parsedShardQueue.try_dequeue(parsedInfo)) {
            if (!g_fec_worker_Running && g_parsedShardQueue.size_approx() == 0) {
                break; // Exit condition: stopped and queue is empty
            }
            std::this_thread::sleep_for(EMPTY_QUEUE_WAIT_MS);
            continue;
        }

        // Discard stale shards from previous stream generations (e.g., after a resize)
        if (parsedInfo.generation != g_streamGeneration.load(std::memory_order_acquire)) {
            continue;
        }

        int frameNumber = parsedInfo.frameNumber;
        int shardIndex = parsedInfo.shardIndex;

        bool readyToDecode = false;
        std::map<uint32_t, std::vector<uint8_t>> shardsForDecode;
        FrameMetadata metadataForDecode;

        // --- Start of critical section for shard processing ---
        {
            std::lock_guard<std::mutex> metaLock(g_frameMetadataMutex);
            std::lock_guard<std::mutex> bufferLock(g_frameBufferMutex);

            // Periodically clear out old, incomplete frames
            static std::chrono::steady_clock::time_point last_cleanup = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (now - last_cleanup > std::chrono::seconds(1)) {
                for (auto it = g_frameMetadata.begin(); it != g_frameMetadata.end(); ) {
                    if (SteadyNowMs() - it->second.first_seen_time_ms > ASSEMBLY_TIMEOUT_MS) {
                        g_frameBuffer.erase(it->first);
                        it = g_frameMetadata.erase(it);
                    } else {
                        ++it;
                    }
                }
                last_cleanup = now;
            }

            FrameMetadata& meta = g_frameMetadata[frameNumber];

            // If this is the first shard for this frame, initialize its metadata
            if (meta.first_seen_time_ms == 0) {
                meta.firstTimestamp = parsedInfo.wgcCaptureTimestamp;
                meta.originalDataLen = parsedInfo.originalDataLen;
                meta.first_seen_time_ms = SteadyNowMs();
                // Keep per-frame RS parameters from ShardInfoHeader
                meta.rs_k = parsedInfo.totalDataShards;
                meta.rs_m = parsedInfo.totalParityShards;
            }

            auto& frameBuf = g_frameBuffer[frameNumber];

            // Store the shard if it's new
            if (frameBuf.find(shardIndex) == frameBuf.end()) {
                frameBuf[shardIndex] = std::move(parsedInfo.shardData);
            }

            // Check if we have enough shards to attempt decoding (use per-frame K)
            const uint32_t rsKForFrame_local =
                (meta.rs_k != 0) ? meta.rs_k : static_cast<uint32_t>(RS_K);
            if (frameBuf.size() >= static_cast<size_t>(rsKForFrame_local)) {
                readyToDecode = true;
                // Copy data needed for decoding to local variables, to release locks sooner
                for (const auto& pair : frameBuf) {
                    shardsForDecode[static_cast<uint32_t>(pair.first)] = pair.second;
                }
                metadataForDecode = meta;

                // Crucially, remove the frame from the global buffers *before* attempting to decode.
                // This prevents other threads from trying to decode the same frame.
                g_frameBuffer.erase(frameNumber);
                g_frameMetadata.erase(frameNumber);
            }
        }
        // --- End of critical section ---

        if (readyToDecode) {
            std::vector<uint8_t> decodedFrameData;
            // Use per-frame K/M (fallback to globals if not set, for safety)
            const uint32_t rsKForFrame = (metadataForDecode.rs_k != 0) ? metadataForDecode.rs_k : static_cast<uint32_t>(RS_K);
            const uint32_t rsMForFrame = (metadataForDecode.rs_m != 0) ? metadataForDecode.rs_m : static_cast<uint32_t>(RS_M);
            if (DecodeFEC_ISAL(
                    shardsForDecode, rsKForFrame, rsMForFrame, metadataForDecode.originalDataLen, decodedFrameData)) {

                if (dumpEncodedStreamToFiles.load()) {
                    std::lock_guard<std::mutex> lock(hevcoutputMutex);
                    SaveEncodedStreamToFile(decodedFrameData, "output_hevc");
                }

                EncodedFrame frame_to_decode;
                frame_to_decode.timestamp = metadataForDecode.firstTimestamp;
                frame_to_decode.frameNumber = frameNumber;
                frame_to_decode.data = std::move(decodedFrameData);
                frame_to_decode.rx_done_ms = SteadyNowMs();

                {
                    std::lock_guard<std::mutex> lk(g_fecEndTimeMutex);
                    g_fecEndTimeByStreamFrame[frame_to_decode.frameNumber] = frame_to_decode.rx_done_ms;
                }
                {
                    std::lock_guard<std::mutex> lk(g_wgcTsMutex);
                    g_wgcCaptureTimestampByStreamFrame[frame_to_decode.frameNumber] = frame_to_decode.timestamp;
                }
                g_encodedFrameQueue.enqueue(std::move(frame_to_decode));

                auto fec_end_time = std::chrono::system_clock::now();
                uint64_t fec_end_time_ts = std::chrono::duration_cast<std::chrono::milliseconds>(fec_end_time.time_since_epoch()).count();

                if (processed_count % 60 == 0) DebugLog(L"FecWorkerThread: Server FEC End to Client FEC End latency for frame " + std::to_wstring(frameNumber) + L": " +
                         std::to_wstring(fec_end_time_ts - parsedInfo.server_fec_timestamp + g_TimeOffsetNs / 1000000) + L" ms");

            } else {
                // Decoding failed, but we might get more shards. Since we already removed it from the buffer,
                // we can't retry. This is a trade-off to prevent multiple threads from decoding the same frame.
                // With a robust shard validation (which is now implicitly done by the server), this path should be rare.
                 DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"]: FEC Decode failed for frame " + std::to_wstring(frameNumber));
            }
        }
        processed_count++;
    }
    DebugLog(L"FecWorkerThread [" + std::to_wstring(threadId) + L"] stopped.");
}

void ListenForRebootCommands() {
    DebugLog(L"ListenForRebootCommands thread started.");

    SOCKET listenSocketStart = INVALID_SOCKET;
    SOCKET listenSocketEnd = INVALID_SOCKET;
    SOCKET clientSocket = INVALID_SOCKET;

    struct sockaddr_in serverAddr;
    int addrLen = sizeof(serverAddr);

    // Create and bind the REBOOTSTART socket
    listenSocketStart = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSocketStart == INVALID_SOCKET) {
        DebugLog(L"ListenForRebootCommands: socket() failed for START socket with error: " + std::to_wstring(WSAGetLastError()));
        return;
    }
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr(RECEIVE_IP_REBOOT);
    serverAddr.sin_port = htons(RECEIVE_PORT_REBOOT_START);
    if (bind(listenSocketStart, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        DebugLog(L"ListenForRebootCommands: bind() failed for START socket with error: " + std::to_wstring(WSAGetLastError()));
        closesocket(listenSocketStart);
        return;
    }
    if (listen(listenSocketStart, SOMAXCONN) == SOCKET_ERROR) {
        DebugLog(L"ListenForRebootCommands: listen() failed for START socket with error: " + std::to_wstring(WSAGetLastError()));
        closesocket(listenSocketStart);
        return;
    }

    // Create and bind the REBOOTEND socket
    listenSocketEnd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSocketEnd == INVALID_SOCKET) {
        DebugLog(L"ListenForRebootCommands: socket() failed for END socket with error: " + std::to_wstring(WSAGetLastError()));
        closesocket(listenSocketStart);
        return;
    }
    serverAddr.sin_port = htons(RECEIVE_PORT_REBOOT_END);
    if (bind(listenSocketEnd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        DebugLog(L"ListenForRebootCommands: bind() failed for END socket with error: " + std::to_wstring(WSAGetLastError()));
        closesocket(listenSocketStart);
        closesocket(listenSocketEnd);
        return;
    }
    if (listen(listenSocketEnd, SOMAXCONN) == SOCKET_ERROR) {
        DebugLog(L"ListenForRebootCommands: listen() failed for END socket with error: " + std::to_wstring(WSAGetLastError()));
        closesocket(listenSocketStart);
        closesocket(listenSocketEnd);
        return;
    }

    DebugLog(L"ListenForRebootCommands: Sockets bound and listening on ports 8150 and 8151.");

    fd_set readSet;

    while (reboot_listener_running) {
        FD_ZERO(&readSet);
        FD_SET(listenSocketStart, &readSet);
        FD_SET(listenSocketEnd, &readSet);

        // Set a timeout so the loop can check the running flag periodically
        timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int total = select(0, &readSet, NULL, NULL, &timeout);
        if (total == SOCKET_ERROR) {
            DebugLog(L"ListenForRebootCommands: select() failed with error: " + std::to_wstring(WSAGetLastError()));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (total == 0) {
            // Timeout, loop again
            continue;
        }

        auto PrepareRebootHandshake = [](bool entering) {
            // 旧ストリーム破棄（受信済みシャード/描画バッファを世代で捨てる）
            BumpStreamGeneration();
            ClearReorderState();

            // 再接続扱いにする
            g_networkReady.store(false, std::memory_order_release);

            // ★重要：再起動後のENet connectで OnNetworkReady が “初回解像度通知” を必ず流すため
            g_didInitialAnnounce.store(false, std::memory_order_release);

            // pending を立てて、OnNetworkReady で確実に flush させる
            int w = currentResolutionWidth.load();
            int h = currentResolutionHeight.load();
            if (w > 0 && h > 0) {
                g_pendingW.store(w);
                g_pendingH.store(h);
                g_pendingResolutionValid.store(true, std::memory_order_release);
            }
        };

        // Check for REBOOTSTART connection
        if (FD_ISSET(listenSocketStart, &readSet)) {
            clientSocket = accept(listenSocketStart, NULL, NULL);
            if (clientSocket != INVALID_SOCKET) {
                char recvbuf[32] = {0};
                int iResult = recv(clientSocket, recvbuf, sizeof(recvbuf) -1, 0);
                if (iResult > 0) {
                    recvbuf[iResult] = '\0';
                    if (strcmp(recvbuf, "REBOOTSTART") == 0) {
                        DebugLog(L"ListenForRebootCommands: Received REBOOTSTART. Showing reboot overlay + preparing handshake.");
                        g_showRebootOverlay.store(true);
                        PrepareRebootHandshake(true);
                    }
                }
                closesocket(clientSocket);
            }
        }

        // Check for REBOOTEND connection
        if (FD_ISSET(listenSocketEnd, &readSet)) {
            clientSocket = accept(listenSocketEnd, NULL, NULL);
            if (clientSocket != INVALID_SOCKET) {
                char recvbuf[32] = {0};
                int iResult = recv(clientSocket, recvbuf, sizeof(recvbuf) - 1, 0);
                if (iResult > 0) {
                     recvbuf[iResult] = '\0';
                    if (strcmp(recvbuf, "REBOOTEND") == 0) {
                        DebugLog(L"ListenForRebootCommands: Received REBOOTEND. Hiding reboot overlay + preparing handshake.");
                        g_showRebootOverlay.store(false);
                        PrepareRebootHandshake(false);

                        // ベストエフォートで即送（ただし繋がってないと落ちる可能性があるので pending も必須）
                        int w = currentResolutionWidth.load();
                        int h = currentResolutionHeight.load();
                        if (w > 0 && h > 0) {
                            SendFinalResolution(w, h);
                        }
                    }
                }
                closesocket(clientSocket);
            }
        }
    }

    closesocket(listenSocketStart);
    closesocket(listenSocketEnd);
    DebugLog(L"ListenForRebootCommands thread stopped.");
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow) {
    // SetProcessDpiAwarenessContextが使えない場合はSetProcessDPIAwareを使う
    timeBeginPeriod(1);

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

    int argc = __argc;
    char** argv = __argv;

    // --- HiDPI対策: DPIスケールに関係なく「指定ピクセル数」通りのサイズで表示したい ---
    // Qt6は既定でHigh-DPIスケーリングが有効で、resize(1504,846) がDIP扱いになり、
    // Windows 125% では物理ピクセル相当で約 1880x1058 になってしまう。
    //
    // ここでQtの座標系を 96DPI 固定に寄せ、1Qt単位≒1物理px として扱うことで、
    // DPI倍率に関係なく初期サイズを 1504x846 に固定する。
    //
    // 注意: これらは QApplication 生成前に行う必要がある。
    qputenv("QT_ENABLE_HIGHDPI_SCALING", "0");
    qputenv("QT_FONT_DPI", "96");

    QApplication a(argc, argv);

    MainWindow mainWindow;
    mainWindow.show();

    HWND parentHwnd = mainWindow.getRenderFrame()->getHostHwnd();

    // Initialize window and DirectX
    if (!InitWindow(hInstance, nCmdShow, parentHwnd)) {
        timeEndPeriod(1);
        WSACleanup();
        enet_deinitialize();
        return -1;
    }

    if (!InitD3D()) {
        DebugLog(L"wWinMain: Failed to initialize Direct3D for rendering after InitWindow.");
        return -1;
    }

    // 子HWNDを親ウィジェットに同期させ、初回解像度をサーバーに通知
    mainWindow.getRenderFrame()->syncChildWindow();

    // After InitWindow(...) and InitD3D() == true
    RECT rc{}; GetClientRect(g_hWnd, &rc);
    int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
    int tw, th; SnapToKnownResolution(cw, ch, tw, th);
    currentResolutionWidth  = tw;
    currentResolutionHeight = th;

    // Enqueue an initial resize for the render thread (swap-chain must match CLIENT size, not video)
    g_pendingResize.w.store(cw, std::memory_order_relaxed);
    g_pendingResize.h.store(ch, std::memory_order_relaxed);
    g_pendingResize.has.store(true, std::memory_order_release);

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
    std::atomic<bool> input_sender_running(true);
    std::thread timeSyncThread(TimeSyncClientThread);
    StartAudioThreads();
    std::thread bandwidthThread(CountBandW);
    std::thread rebootListenerThread(ListenForRebootCommands);
    std::thread inputSenderThread(InputSendThread, std::ref(input_sender_running));

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

    std::thread frameMonitorThread([]{
        DebugLog(L"FrameMonitorThread started.");
        while (app_running_atomic.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));

            // Only monitor if network is ready
            if (!g_networkReady.load(std::memory_order_acquire)) continue;

            uint64_t last = g_lastFrameTickMs.load(std::memory_order_relaxed);
            if (last == 0) continue; // No frame rendered yet

            uint64_t now = SteadyNowMs();
            // 1.5s timeout
            if (now > last && (now - last) > 1500) {
                 static uint64_t lastRequestTime = 0;
                 // Don't spam requests. Let's say once every 1.5 seconds if still stuck.
                 if (now > lastRequestTime && (now - lastRequestTime) > 1500) {
                     DebugLog(L"FrameMonitorThread: No new frame for " + std::to_wstring(now - last) + L" ms. Requesting resync.");
                     int w = currentResolutionWidth.load();
                     int h = currentResolutionHeight.load();
                     if (w > 0 && h > 0) {
                        SendFinalResolution(w, h);
                     }
                     lastRequestTime = now;
                 }
            }
        }
        DebugLog(L"FrameMonitorThread stopped.");
    });

    // Main render loop integrated into Qt
    QTimer renderTimer;
    QObject::connect(&renderTimer, &QTimer::timeout, []() {
        RenderFrame();
    });
    renderTimer.start(0); // Pacing is handled inside RenderFrame by DXGI waitable object

    a.exec();

    // Signal app is stopping for threads checking this flag
    app_running_atomic.store(false, std::memory_order_relaxed);

    // Centralized Cleanup
    StopAudioThreads();
    DebugLog(L"Exited message loop. Initiating final resource cleanup...");
    AppThreads appThreads{};
    appThreads.bandwidthThread = &bandwidthThread;
    appThreads.rebootListenerThread = &rebootListenerThread;
    appThreads.receiverThreads = &receiverThreads;
    appThreads.fecWorkerThreads = &fecWorkerThreads;
    appThreads.nvdecThreads = &nvdecThreads;
    appThreads.inputSenderThread = &inputSenderThread;
    appThreads.input_sender_running = &input_sender_running;
    appThreads.frameMonitorThread = &frameMonitorThread;

    // This single call handles joining all threads and releasing all resources idempotently.
    ReleaseAllResources(appThreads);
    
    DebugLog(L"Cleanup complete. Exiting wWinMain.");

    // === 非同期ロガーを安全に停止（全ログを flush） ===
    DebugLogAsync::Shutdown();
    return 0;
}
