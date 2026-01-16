#include "Globals.h"
#include "nvdec.h"

// Instantiate the global queues
moodycamel::ConcurrentQueue<MouseInputMessage> g_mouseInputQueue;
moodycamel::ConcurrentQueue<EncodedFrame> g_encodedFrameQueue;
std::deque<ReadyGpuFrame> g_readyGpuFrameQueue;
std::mutex g_readyGpuFrameQueueMutex;
std::condition_variable g_readyGpuFrameQueueCV;

// Instantiate other global variables
std::atomic<bool> app_running_atomic(true);
std::atomic<bool> g_isSizing(false);
std::atomic<bool> g_showRebootOverlay(false);
std::atomic<bool> g_forcePresentOnce(false);
std::atomic<bool> dumpEncodedStreamToFiles(false);

std::atomic<uint64_t> g_streamGeneration(0);
std::atomic<uint64_t> g_latencyEpochMs(0);
std::atomic<uint64_t> g_lastIdrMs(0);

PendingResize g_pendingResize;

HWND g_hWnd = nullptr;
HANDLE g_frameLatencyWaitableObject = nullptr;

const int RS_K = 14;
const int RS_M = 8;
const int RS_N = RS_K + RS_M;

std::unordered_map<uint32_t, uint64_t> g_fecEndTimeByStreamFrame;
std::mutex g_fecEndTimeMutex;

std::unordered_map<uint32_t, uint64_t> g_wgcCaptureTimestampByStreamFrame;
std::mutex g_wgcTsMutex;

std::unique_ptr<FrameDecoder> g_frameDecoder;

std::atomic<bool> g_fec_worker_Running(true);
std::atomic<bool> g_decode_worker_Running(true);
std::atomic<bool> reboot_listener_running(true);
std::atomic<int> currentResolutionWidth(0);
std::atomic<int> currentResolutionHeight(0);

std::chrono::high_resolution_clock::time_point g_lastFrameRenderTimeForKick;

Microsoft::WRL::ComPtr<ID3D12Device> g_d3d12Device;

// Window shown synchronization
std::mutex g_windowShownMutex;
std::condition_variable g_windowShownCv;
bool g_windowShown = false;

// String conversion utility implementation
std::wstring ConvertToWString(const std::string& str) {
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    if (size_needed <= 0) return L"(invalid UTF-8 string)";
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size_needed);
    // Remove the null terminator at the end
    if (!wstr.empty() && wstr.back() == L'\0') {
        wstr.pop_back();
    }
    return wstr;
}
