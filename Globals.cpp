#include <Windows.h>
#include <atomic>
#include <vector>
#include <mutex>
#include "Globals.h"
#include "Nvdec.h"
#include "main.h"

// Reed-Solomon encoding parameters
const int RS_K = getOptimalThreadConfig().RS_K;  // Data shards
const int RS_M = getOptimalThreadConfig().RS_M;  // Parity shards
const int RS_N = RS_K + RS_M;

// Jerasure encoding matrix (bit matrix)
int* g_jerasure_matrix = nullptr;
std::vector<uint8_t> g_encode_matrix(RS_N * RS_K);
int* g_vandermonde_matrix = nullptr;
std::once_flag g_matrix_init_flag;
std::atomic<bool> g_matrix_initialized(false);

// Window and resolution management
std::atomic<int> currentResolutionWidth(1920);
std::atomic<int> currentResolutionHeight(1080);
HWND g_hWnd = nullptr;
HANDLE g_frameLatencyWaitableObject = nullptr;

// Pending resize request state
PendingResize g_pendingResize;

// D3D12 Frame queue management
std::deque<ReadyGpuFrame> g_readyGpuFrameQueue;
std::mutex g_readyGpuFrameQueueMutex;
std::condition_variable g_readyGpuFrameQueueCV;

// Global running flags
std::atomic<bool> g_fec_worker_Running(true);
std::atomic<bool> g_decode_worker_Running(true);
std::atomic<bool> reboot_listener_running(true);
std::atomic<bool> app_running_atomic(true);
std::atomic<bool> g_isSizing{false};
std::atomic<bool> g_showRebootOverlay{ false };
std::atomic<bool> g_forcePresentOnce{false};
std::atomic<bool> dumpH264ToFiles{false};

// H264 Frame queue
moodycamel::ConcurrentQueue<H264Frame> g_h264FrameQueue;

// FEC end times keyed by stream frame number (steady clock ms since epoch-like)
std::unordered_map<uint32_t, uint64_t> g_fecEndTimeByStreamFrame;
std::mutex g_fecEndTimeMutex;

// WGC capture timestamps keyed by stream frame number (server system_clock ms)
std::unordered_map<uint32_t, uint64_t> g_wgcCaptureTimestampByStreamFrame;
std::mutex g_wgcTsMutex;

// Decoder instance
std::unique_ptr<FrameDecoder> g_frameDecoder;

// Generation/epoch for resize events
std::atomic<uint64_t> g_streamGeneration{0};
std::atomic<uint64_t> g_latencyEpochMs{0};
std::atomic<uint64_t> g_lastIdrMs{0};
