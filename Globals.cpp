#include <Windows.h>
#include <atomic>
#include <vector>
#include <mutex>
#include "Globals.h"

// Reed-Solomon encoding parameters
const int RS_K = 8;  // Data shards
const int RS_M = 2;  // Parity shards
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

// D3D12 Frame queue management
std::deque<ReadyGpuFrame> g_readyGpuFrameQueue;
std::mutex g_readyGpuFrameQueueMutex;
std::condition_variable g_readyGpuFrameQueueCV;
