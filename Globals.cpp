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

// Pending resize request state
PendingResize g_pendingResize;

// D3D12 Frame queue management
LUID g_dxgiAdapterLuid = {}; // zero-init
std::deque<ReadyGpuFrame> g_readyGpuFrameQueue;
std::mutex g_readyGpuFrameQueueMutex;
std::condition_variable g_readyGpuFrameQueueCV;

// Global running flags
std::atomic<bool> g_fec_worker_Running(true);
std::atomic<bool> g_decode_worker_Running(true);
std::atomic<bool> app_running_atomic(true);
std::atomic<bool> g_isSizing{false};
std::atomic<bool> g_forcePresentOnce{false};
std::atomic<bool> g_deviceLost{false};
std::atomic<bool> dumpH264ToFiles{false};

// H264 Frame queue
moodycamel::ConcurrentQueue<H264Frame> g_h264FrameQueue;
moodycamel::ConcurrentQueue<std::vector<uint8_t>*> g_h264BufferPool;

// Decoder instance
std::unique_ptr<FrameDecoder> g_frameDecoder;
