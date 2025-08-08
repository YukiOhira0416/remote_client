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
#include "concurrentqueue/concurrentqueue.h"

#define SIZE_PACKET_SIZE 258

// Struct to hold data for the NVDEC decoder
struct H264Frame {
    uint64_t timestamp;
    uint32_t frameNumber;
    std::vector<uint8_t> frameData;
};

// Queue for passing H.264 frames from FEC workers to NVDEC workers
extern moodycamel::ConcurrentQueue<H264Frame> g_h264FrameQueue;

extern std::atomic<int> currentResolutionWidth;
extern std::atomic<int> currentResolutionHeight;
extern HWND g_hWnd;
// D3D12 Globals (defined in window.cpp)
extern Microsoft::WRL::ComPtr<ID3D12Device> g_d3d12Device;
extern Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_d3d12CommandQueue;

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
};

// Global queues and synchronization for frame management
extern std::deque<ReadyGpuFrame> g_readyGpuFrameQueue;
extern std::mutex g_readyGpuFrameQueueMutex;
extern std::condition_variable g_readyGpuFrameQueueCV;

struct ThreadConfig {
    int receiver = 1;
    int fec = 1;
    int decoder = 1;
    int render = 1;
};

#endif