#ifndef GLOBALS_H
#define GLOBALS_H
#include <windows.h>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <utility>
#include <cstdint>
#include <d3d12.h>
#include <wrl/client.h>
#include <deque>
#include <mutex>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <condition_variable>
#include <unordered_map>
#include "concurrentqueue/concurrentqueue.h"

#define SIZE_PACKET_SIZE 258

const double TARGET_FPS = 60.0;
const std::chrono::nanoseconds TARGET_FRAME_DURATION(static_cast<long long>(1'000'000'000.0 / TARGET_FPS));

// New struct for pending resize requests
struct PendingResize {
    std::atomic<bool> has{false};
    std::atomic<int>  w{0};
    std::atomic<int>  h{0};
};

extern PendingResize g_pendingResize;

// Forward declarations for NVDEC
class FrameDecoder;
void NvdecThread(int threadId);


extern std::atomic<bool> g_fec_worker_Running;
extern std::atomic<bool> g_decode_worker_Running;
extern std::atomic<int> currentResolutionWidth;
extern std::atomic<int> currentResolutionHeight;
extern HWND g_hWnd;
// D3D12 Globals (defined in window.cpp)
extern Microsoft::WRL::ComPtr<ID3D12Device> g_d3d12Device;

extern const int RS_K;
extern const int RS_M;
extern const int RS_N;
extern std::once_flag g_matrix_init_flag;
extern std::vector<uint8_t> g_encode_matrix;
extern std::atomic<bool> g_matrix_initialized;
extern int* g_jerasure_matrix;
extern int* g_vandermonde_matrix;

// Helper to convert pointer to wstring (moved from window.cpp)
template <typename T>
std::wstring PointerToWString(T* ptr) {
    if (!ptr) {
        return L"nullptr";
    }
    std::wstringstream wss;
    wss << L"0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
    return wss.str();
}

// Helper to convert HRESULT to hex wstring
inline std::wstring HResultToHexWString(HRESULT hr) {
    std::wstringstream wss;
    // Output HRESULT as a 32-bit hex number (e.g., 0x80070057)
    // static_cast to unsigned long to ensure it's treated as an unsigned value for hex formatting.
    wss << L"0x" << std::hex << std::setw(8) << std::setfill(L'0') << static_cast<unsigned long>(hr);
    return wss.str();
}

// ReadyGpuFrame struct for D3D12
struct ReadyGpuFrame {
    uint64_t timestamp;    
    Microsoft::WRL::ComPtr<ID3D12Resource> hw_decoded_texture_Y;  // Y plane texture
    Microsoft::WRL::ComPtr<ID3D12Resource> hw_decoded_texture_UV; // UV plane texture
    int width;
    int height;
    uint32_t originalFrameNumber;
    uint64_t id;
    // 新規: 送信（ストリーム）側のフレーム番号
    uint32_t streamFrameNumber = 0;
    // Latency metric
    uint64_t client_fec_end_to_render_end_time_ms = 0;

    // Client-side steady clock timestamps for latency measurement.
    uint64_t rx_done_ms = 0;       // Client steady clock: when FEC assembly completed (set in FEC -> propagated)
    uint64_t nvdec_done_ms = 0;    // Client steady clock: after NVDEC copy + cuCtxSynchronize
    uint64_t render_start_ms = 0;  // Client steady clock: just before CPU populates commands
    uint64_t submit_ms = 0;        // Client steady clock: right after ExecuteCommandLists
    uint64_t present_ms = 0;       // Client steady clock: right after Present returns
    uint64_t fence_done_ms = 0;    // Client steady clock: after per-frame fence wait (if any)
};

// H264 Frame Data for decoder queue
struct H264Frame {
    uint64_t timestamp;
    uint32_t frameNumber;
    std::vector<uint8_t> data;

    // Timing fields for accurate end-to-end latency
    uint64_t rx_done_ms = 0;        // set when frame is fully received/reconstructed (K shards reached)
    uint64_t decode_start_ms = 0;   // set when NVDEC begins decoding this frame
};

// Global running flag for wWinMain loop
extern std::atomic<bool> app_running_atomic;
extern std::atomic<bool> g_isSizing;
extern std::atomic<bool> g_forcePresentOnce; // Present at least once even if no new decoded frame

// Global queues and synchronization for frame management
extern moodycamel::ConcurrentQueue<H264Frame> g_h264FrameQueue;
extern std::deque<ReadyGpuFrame> g_readyGpuFrameQueue;
extern std::mutex g_readyGpuFrameQueueMutex;
extern std::condition_variable g_readyGpuFrameQueueCV;

// FEC end times keyed by stream frame number (steady clock ms since epoch-like)
extern std::unordered_map<uint32_t, uint64_t> g_fecEndTimeByStreamFrame;
extern std::mutex g_fecEndTimeMutex;

// New: WGC capture timestamps keyed by stream frame number (server system_clock ms)
extern std::unordered_map<uint32_t, uint64_t> g_wgcCaptureTimestampByStreamFrame;
extern std::mutex g_wgcTsMutex;

// Global instance for the decoder
extern std::unique_ptr<FrameDecoder> g_frameDecoder;

struct ThreadConfig {
    int receiver = 1;
    int fec = 1;
    int decoder = 1;
    int render = 1;
    int RS_K = 1;
    int RS_M = 1;
};

#endif