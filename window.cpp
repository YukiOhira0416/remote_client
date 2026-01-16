#include <d3dcompiler.h> // For shader compilation
#include <DirectXColors.h> // For DirectX::Colors
#include <windows.h>
#include <d3d12.h> // D3D12
#include <d3dx12.h> // D3D12 Helper Structures (for CD3DX12_CPU_DESCRIPTOR_HANDLE, etc.)
#include <wrl/client.h>
#include <dxgi1_6.h> // For IDXGIFactory6 and CreateSwapChainForHwnd
#include <cmath>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include "DebugLog.h"
#include "ReedSolomon.h"
#include <algorithm> // For std::min, std::abs
#include <vector> // For std::vector
#include <deque>  // For std::deque (used by Globals.h)
#include <mutex>  // For std::mutex (used by Globals.h)
#include <condition_variable> // For std::condition_variable (used by Globals.h)
#include <atomic> // For std::atomic (used by Globals.h)
#include <sstream>      // Required for std::wstringstream (for PointerToWString)
#include <iomanip>      // Required for std::hex (for PointerToWString)
#include <chrono> // For time measurement
#include <map>
#include <queue>
#include "Globals.h"
#include "AppShutdown.h"
#include "main.h"
#include "TimeSyncClient.h"
#include "AudioClient.h"

// CUDA includes
#include <cuda.h>
#include <cuda_runtime_api.h>

// ==== [Multi-monitor helpers - BEGIN] ====
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <ShellScalingApi.h> // AdjustWindowRectExForDpi 等
#pragma comment(lib, "Shcore.lib")

UINT64 RenderCount = 0;

static UINT GetDpiForMonitorOrDefault(HMONITOR hMon) {
    UINT dpiX = 96, dpiY = 96;
    HMODULE hShcore = LoadLibraryW(L"Shcore.dll");
    if (hShcore) {
        typedef HRESULT (WINAPI *GetDpiForMonitorFunc)(HMONITOR, int, UINT*, UINT*);
        auto pGetDpiForMonitor = (GetDpiForMonitorFunc)GetProcAddress(hShcore, "GetDpiForMonitor");
        if (pGetDpiForMonitor) {
            // MDT_EFFECTIVE_DPI = 0
            if (SUCCEEDED(pGetDpiForMonitor(hMon, 0, &dpiX, &dpiY))) {
                FreeLibrary(hShcore);
                return dpiX;
            }
        }
        FreeLibrary(hShcore);
    }
    // Fallback: GetDpiForWindow(User32) or default 96
    HMODULE hUser32 = LoadLibraryW(L"user32.dll");
    if (hUser32) {
        typedef UINT (WINAPI *GetDpiForWindowFunc)(HWND);
        auto pGetDpiForWindow = (GetDpiForWindowFunc)GetProcAddress(hUser32, "GetDpiForWindow");
        if (pGetDpiForWindow && g_hWnd) {
            UINT d = pGetDpiForWindow(g_hWnd);
            FreeLibrary(hUser32);
            return d ? d : 96;
        }
        FreeLibrary(hUser32);
    }
    return 96;
}

static bool CreateWindowOnBestMonitor(HINSTANCE hInstance, int nCmdShow,
                                      int desiredClientWidth, int desiredClientHeight,
                                      HWND parentHwnd = NULL) {
    // カーソル位置のモニタを初期ターゲットにする
    POINT pt; GetCursorPos(&pt);
    HMONITOR hMon = MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi{ sizeof(MONITORINFO) };
    if (!GetMonitorInfoW(hMon, &mi)) {
        mi.rcWork = {0, 0, desiredClientWidth, desiredClientHeight};
    }
    UINT dpi = GetDpiForMonitorOrDefault(hMon);

    DWORD dwStyle = WS_OVERLAPPEDWINDOW;
    DWORD dwExStyle = 0;

    if (parentHwnd) {
        dwStyle = WS_CHILD | WS_VISIBLE;
    }

    RECT rc = {0, 0, desiredClientWidth, desiredClientHeight};

    // AdjustWindowRectExForDpi が使えるなら正しいDPIで枠を計算
    HMODULE hUser32 = LoadLibraryW(L"user32.dll");
    if (hUser32) {
        typedef BOOL (WINAPI *AdjustForDpi)(LPRECT, DWORD, BOOL, DWORD, UINT);
        auto pAdj = (AdjustForDpi)GetProcAddress(hUser32, "AdjustWindowRectExForDpi");
        if (pAdj) {
            pAdj(&rc, dwStyle, FALSE, dwExStyle, dpi);
        } else {
            AdjustWindowRectEx(&rc, dwStyle, FALSE, dwExStyle);
        }
        FreeLibrary(hUser32);
    } else {
        AdjustWindowRectEx(&rc, dwStyle, FALSE, dwExStyle);
    }

    int winW = rc.right - rc.left;
    int winH = rc.bottom - rc.top;

    int x, y;
    if (parentHwnd) {
        RECT parentRect;
        GetClientRect(parentHwnd, &parentRect);
        x = 0;
        y = 0;
        winW = parentRect.right - parentRect.left;
        winH = parentRect.bottom - parentRect.top;
        desiredClientWidth = winW;
        desiredClientHeight = winH;
    } else {
        x = mi.rcWork.left + ((mi.rcWork.right  - mi.rcWork.left) - winW) / 2;
        y = mi.rcWork.top  + ((mi.rcWork.bottom - mi.rcWork.top) - winH) / 2;
    }

    g_hWnd = CreateWindowExW(dwExStyle, L"MyWindowClass", L"Remote Desktop Viewer",
                             dwStyle, x, y, winW, winH,
                             parentHwnd, nullptr, hInstance, nullptr);
    if (!g_hWnd) {
        DebugLog(L"InitWindow: CreateWindowExW failed. Error: " + std::to_wstring(GetLastError()));
        return false;
    }

    ShowWindow(g_hWnd, nCmdShow);
    UpdateWindow(g_hWnd);

    {
        std::lock_guard<std::mutex> lock(g_windowShownMutex);
        g_windowShown = true;
    }
    g_windowShownCv.notify_one();

    currentResolutionWidth  = desiredClientWidth;
    currentResolutionHeight = desiredClientHeight;
    return true;
}
// ==== [Multi-monitor helpers - END] ====

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "d3d12.lib") // Link D3D12 lib
#pragma comment(lib, "dxgi.lib") // DXGI is still used
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

// Renders the video with its aspect ratio preserved, adding black bars (letterboxing/pillarboxing).
static void SetLetterboxViewport(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_DESC backbufferDesc, int videoWidthInt, int videoHeightInt)
{
    if (!cmd) return;

    const float bbWidth  = static_cast<float>(backbufferDesc.Width);
    const float bbHeight = static_cast<float>(backbufferDesc.Height);
    const float videoW   = static_cast<float>(videoWidthInt);
    const float videoH   = static_cast<float>(videoHeightInt);

    if (bbWidth <= 0.0f || bbHeight <= 0.0f || videoW <= 0.0f || videoH <= 0.0f) {
        return; // avoid div-by-zero or nonsense
    }

    const float bbAspect    = bbWidth / bbHeight;
    const float videoAspect = videoW / videoH;

    float vpW, vpH, vpX, vpY;
    if (bbAspect > videoAspect) {
        // Pillarbox (window wider than video)
        vpH = bbHeight;
        vpW = vpH * videoAspect;
        vpX = (bbWidth - vpW) * 0.5f;
        vpY = 0.0f;
    } else {
        // Letterbox (window taller than video)
        vpW = bbWidth;
        vpH = vpW / videoAspect;
        vpX = 0.0f;
        vpY = (bbHeight - vpH) * 0.5f;
    }

    // Viewport (float)
    D3D12_VIEWPORT vp{};
    vp.TopLeftX = vpX;
    vp.TopLeftY = vpY;
    vp.Width    = vpW;
    vp.Height   = vpH;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    cmd->RSSetViewports(1, &vp);

    // Scissor (int): use ceilf on right/bottom to avoid truncation losing the last pixel.
    D3D12_RECT sc{};
    sc.left   = static_cast<LONG>(vpX);
    sc.top    = static_cast<LONG>(vpY);
    sc.right  = static_cast<LONG>(std::ceil(vpX + vpW));
    sc.bottom = static_cast<LONG>(std::ceil(vpY + vpH));
    cmd->RSSetScissorRects(1, &sc);
}

static inline void SetViewportScissorToBackbuffer(
    ID3D12GraphicsCommandList* cmd,
    ID3D12Resource* backbuffer)
{
    if (!cmd || !backbuffer) return;

    const D3D12_RESOURCE_DESC desc = backbuffer->GetDesc();
    const float w = static_cast<float>(desc.Width);
    const float h = static_cast<float>(desc.Height);

    D3D12_VIEWPORT vp{};
    vp.TopLeftX = 0.0f;
    vp.TopLeftY = 0.0f;
    vp.Width    = w;
    vp.Height   = h;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    cmd->RSSetViewports(1, &vp);

    D3D12_RECT sc{};
    sc.left   = 0;
    sc.top    = 0;
    sc.right  = static_cast<LONG>(desc.Width);
    sc.bottom = static_cast<LONG>(desc.Height);
    cmd->RSSetScissorRects(1, &sc);
}

// D3D12 Global Variables
// Overlay Resources
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_overlayQuadPso;
Microsoft::WRL::ComPtr<ID3D12RootSignature> g_overlayQuadRootSignature;
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_overlayTextPso;
Microsoft::WRL::ComPtr<ID3D12RootSignature> g_overlayRootSignature;
Microsoft::WRL::ComPtr<ID3D12Resource> g_overlayVertexBuffer;
D3D12_VERTEX_BUFFER_VIEW g_overlayVertexBufferView;
Microsoft::WRL::ComPtr<ID3D12Resource> g_textTexture;
Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> g_textSrvHeap;

static constexpr UINT kSwapChainBufferCount = 3; // was 2
Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_d3d12CommandQueue;
Microsoft::WRL::ComPtr<IDXGISwapChain3> g_swapChain; // Use IDXGISwapChain3 or 4
Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> g_rtvHeap;
Microsoft::WRL::ComPtr<ID3D12Resource> g_renderTargets[kSwapChainBufferCount]; // Triple buffering
// One allocator per swap-chain buffer (triple buffering respected)
Microsoft::WRL::ComPtr<ID3D12CommandAllocator> g_commandAllocator[kSwapChainBufferCount];
Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> g_commandList;
Microsoft::WRL::ComPtr<ID3D12Fence> g_fence;
UINT g_rtvDescriptorSize;
UINT g_currentFrameBufferIndex; // Current back buffer index
UINT64 g_fenceValue;
HANDLE g_fenceEvent;
UINT64 g_renderFenceValues[kSwapChainBufferCount]; // Fence values for each frame in flight for rendering

// [NEW] Shared fence used for CUDA→D3D12 GPU-GPU sync (do not reflow existing code)
static Microsoft::WRL::ComPtr<ID3D12Fence> g_copyFence;
static HANDLE g_copyFenceSharedHandle = nullptr; // passed to CUDA
static UINT64 g_copyFenceValue = 0;              // monotonically increasing per ready frame

// [NEW] Accessor for CUDA side (do not change signature or layout elsewhere)
extern "C" HANDLE GetCopyFenceSharedHandleForCuda() {
    return g_copyFenceSharedHandle;
}

static std::atomic<bool> g_deviceResetInProgress{false};

// ==== [Deferred resource retire - BEGIN] ====
// Keep layout/comments as-is around this block.
struct RetiredGpuResources {
    Microsoft::WRL::ComPtr<ID3D12Resource> y;
    Microsoft::WRL::ComPtr<ID3D12Resource> u;
    Microsoft::WRL::ComPtr<ID3D12Resource> v;
    CUevent copyDone;         // may be nullptr
    UINT64 fenceValue;        // render-queue fence that must be completed before release
    RetiredGpuResources() : copyDone(nullptr), fenceValue(0) {}
};

static std::deque<RetiredGpuResources> g_retireBin;
static std::mutex g_retireBinMutex;

// Signal the render fence now and return the value; safe if device/queue/fence exist.
static UINT64 SignalFenceNow() noexcept {
    if (!g_d3d12CommandQueue || !g_fence) return 0;
    const UINT64 v = ++g_fenceValue;
    HRESULT hr = g_d3d12CommandQueue->Signal(g_fence.Get(), v);
    if (FAILED(hr)) {
        DebugLog(L"SignalFenceNow: Signal failed. HR: " + HResultToHexWString(hr));
        return 0;
    }
    return v;
}

// Drain any retired resources whose CUDA event and fence are complete.
// Maintains timing/logging; no sleeps; no reformatting.
static void DrainRetireBin() {
    // NEW: bail out if sync primitives or device are not in a trustworthy state.
    if (g_deviceResetInProgress.load(std::memory_order_acquire)) {
        return; // device/pipeline is being reset; defer releasing GPU resources
    }
    if (!g_d3d12CommandQueue || !g_fence /*|| !g_d3d12Device*/) {
        return; // NEW: do not assume completion when fence/queue are unavailable
    }

    std::lock_guard<std::mutex> g(g_retireBinMutex);
    const UINT64 completed = g_fence ? g_fence->GetCompletedValue() : ~0ULL;

    size_t released = 0;
    while (!g_retireBin.empty()) {
        RetiredGpuResources &r = g_retireBin.front();
        const bool cudaDone =
            (r.copyDone == nullptr) || (cuEventQuery(r.copyDone) == CUDA_SUCCESS);
        const bool fenceDone = (r.fenceValue == 0) || (completed >= r.fenceValue);

        if (!cudaDone || !fenceDone) {
            // Oldest not ready; later ones may not be ready either.
            break;
        }

        if (r.copyDone) {
            cuEventDestroy(r.copyDone);
            r.copyDone = nullptr;
        }

        // Final release now that both queues are quiesced for this resource set.
        r.y.Reset();
        r.u.Reset();
        r.v.Reset();

        g_retireBin.pop_front();
        ++released;
    }

    if (released) {
        std::wstringstream wss;
        wss << L"DrainRetireBin: released " << released << L" retired resource set(s).";
        DebugLog(wss.str());
    }
}
// ==== [Deferred resource retire - END] ====

// D3D11 specific globals (to be removed or replaced)
#define SEND_PORT_FEC 8080// FEC用ポート番号
#define SEND_IP_FEC "192.168.0.2"// FEC用IPアドレス
//#define SEND_IP_FEC "127.0.0.1"// FEC用IPアドレス

bool g_allowTearing = false; // ティアリングを許可するかどうか

// Forward declarations for recovery helper
void WaitForGpu();
void ClearReorderState();
bool CreateOverlayResources(); // New function for overlay
void CleanupOverlayResources(); // New function for overlay cleanup
bool InitD3D();

// ---- Device-loss recovery (no goto, preserve layout/comments) ----

static void HandleDeviceRemovedAndReinit() noexcept {
    if (g_deviceResetInProgress.exchange(true)) {
        // Already handling a reset in another code path; do nothing.
        return;
    }

    // Log remove/reset reason if available
    if (g_d3d12Device) {
        HRESULT gr = g_d3d12Device->GetDeviceRemovedReason();
        DebugLog(L"D3D12 device removed/reset. Reason: " + HResultToHexWString(gr));
    } else {
        DebugLog(L"D3D12 device removed/reset. Reason: (device nullptr)");
    }

    // Drain GPU work while pumping messages; preserves timing/logging
    WaitForGpu();

    // Clear in-flight frames and reorder buffers to avoid stale resources
    {
        std::lock_guard<std::mutex> qlock(g_readyGpuFrameQueueMutex);
        g_readyGpuFrameQueue.clear();
    }
    ClearReorderState(); // do not touch MonitorSharedMemory()

    // Tear down and re-create the D3D12 pipeline (InitD3D already resets & recreates)
    if (!InitD3D()) {
        DebugLog(L"HandleDeviceRemovedAndReinit: InitD3D() failed.");
        // We intentionally return; the main loop/logging survives.
        g_deviceResetInProgress.store(false, std::memory_order_release);
        return;
    }

    // Force at least one Present to refresh the screen after reset
    g_forcePresentOnce.store(true, std::memory_order_release);

    DebugLog(L"HandleDeviceRemovedAndReinit: D3D12 re-initialized successfully.");
    g_deviceResetInProgress.store(false, std::memory_order_release);
}


// ==== [Resize helpers - BEGIN] ====

// 16:9のアスペクト比を維持するようにサイズを調整するユーティリティ
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH) {
    if (srcW <= 0 || srcH <= 0) {
        outW = 1280; outH = 720;
        return;
    }
    // 16:9を維持しつつ、元の矩形に収まる最大のサイズを計算する
    if (srcW * 9 > srcH * 16) {
        outW = srcH * 16 / 9;
        outH = srcH;
    } else {
        outW = srcW;
        outH = srcW * 9 / 16;
    }
}

// サーバーへ最終解像度を送る（旧 SendWindowSize 代替）
void SendFinalResolution(int width, int height); // Forward declaration (removed static for external linkage)
void WaitForGpu(); // D3D12: Helper function to wait for GPU to finish commands

// ==== [Resize helpers - END] ====

// from main.cpp
extern std::atomic<bool> g_networkReady;
extern std::atomic<bool> g_pendingResolutionValid;
extern std::atomic<int> g_pendingW, g_pendingH;

void OnResolutionChanged_GatedSend(int w, int h, bool forceResendNow = false)
{
    currentResolutionWidth  = w;
    currentResolutionHeight = h;

    // 常にペンディング更新（最後の値を保持）
    g_pendingW = w; g_pendingH = h; g_pendingResolutionValid = true;

    if (forceResendNow || g_networkReady.load()) {
        DebugLog(L"OnResolutionChanged_GatedSend: sending now.");
        SendFinalResolution(w, h);
        ClearReorderState();

        g_pendingResolutionValid = false;
    } else {
        DebugLog(L"OnResolutionChanged_GatedSend: network not ready, pending.");
    }
}

// 送信フレーム番号で並べる小さなバッファ
static std::map<uint32_t, ReadyGpuFrame> g_reorderBuffer;
static std::mutex g_reorderMutex;
static ReadyGpuFrame g_lastDrawnFrame; // 最後に描画されたフレームをキャッシュ

static uint32_t g_expectedStreamFrame = 0;
static bool     g_expectedInitialized = false;

// 描画側の「期待フレーム番号」やバッファをクリアして“待ち”を防ぐ
void ClearReorderState()
{
    // Keep this to preserve current behavior and logs:
    WaitForGpu(); // drains graphics queue only (CUDA is handled per frame)  // (implementation references)

    std::lock_guard<std::mutex> lk(g_reorderMutex);

    // Record a fence point that future releases must pass
    const UINT64 retireFence = SignalFenceNow();

    // Move all frames' resources into the retire bin (no immediate Reset)
    {
        std::lock_guard<std::mutex> gb(g_retireBinMutex);
        for (auto &pair : g_reorderBuffer) {
            ReadyGpuFrame &rf = pair.second;

            RetiredGpuResources r;
            r.y = std::move(rf.hw_decoded_texture_Y);
            r.u = std::move(rf.hw_decoded_texture_U);
            r.v = std::move(rf.hw_decoded_texture_V);
            r.copyDone = rf.copyDone;      // hand the event to retire bin
            r.fenceValue = retireFence;    // conservative: release after this fence

            rf.copyDone = nullptr;         // ownership transferred

            g_retireBin.emplace_back(std::move(r));
        }
        g_reorderBuffer.clear();
    }

    // Handle the cached last-drawn frame the same way
    {
        RetiredGpuResources r;
        r.y = std::move(g_lastDrawnFrame.hw_decoded_texture_Y);
        r.u = std::move(g_lastDrawnFrame.hw_decoded_texture_U);
        r.v = std::move(g_lastDrawnFrame.hw_decoded_texture_V);
        r.copyDone = g_lastDrawnFrame.copyDone;
        r.fenceValue = retireFence;
        g_lastDrawnFrame.copyDone = nullptr;
        g_lastDrawnFrame = {}; // keep existing layout/behavior

        if (r.y || r.u || r.v || r.copyDone) {
            std::lock_guard<std::mutex> gb(g_retireBinMutex);
            g_retireBin.emplace_back(std::move(r));
        }
    }

    g_expectedInitialized = false;
    g_expectedStreamFrame = 0;
    DebugLog(L"ClearReorderState: reorder state moved to retire bin.");
}

// 調整パラメータ
static constexpr size_t REORDER_MAX_BUFFER = 4; // tighten to reduce in-buffer latency
static constexpr int    REORDER_WAIT_MS    = 2; // base wait smaller

static inline int GetReorderWaitMsForDepth(size_t depth) {
    if (depth <= 1) return 0;
    return 1;
}

static std::chrono::steady_clock::time_point g_lastReorderDecision = std::chrono::steady_clock::now();

// Rendering specific D3D12 globals
Microsoft::WRL::ComPtr<ID3D12RootSignature> g_rootSignature;
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_pipelineStateYuv444;
Microsoft::WRL::ComPtr<ID3D12Resource> g_vertexBuffer;
D3D12_VERTEX_BUFFER_VIEW g_vertexBufferView;
Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> g_srvHeap; // For Y and UV textures
UINT g_srvDescriptorSize;

// [FIX] SRV Heap Ring Buffer
static const UINT kSrvHeapSize = 256;
static UINT g_srvDescriptorHeapIndex = 0;

// Constant buffer for cropping
struct CropCBData {
    float uvScale[2];
    float uvBias[2];
    float _pad[2]; // 16B-align
};
Microsoft::WRL::ComPtr<ID3D12Resource> g_cropCB;
UINT8* g_cropCBMapped = nullptr;

bool CreateCropCB() {
    if (!g_d3d12Device) return false;
    const UINT cbSize = (sizeof(CropCBData) + 255) & ~255u;
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto resDesc = CD3DX12_RESOURCE_DESC::Buffer(cbSize);
    HRESULT hr = g_d3d12Device->CreateCommittedResource(
        &heapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_cropCB));
    if (FAILED(hr)) { DebugLog(L"CreateCropCB failed."); return false; }
    // Persistently map the buffer
    CD3DX12_RANGE readRange(0, 0); // We do not intend to read from this buffer on the CPU.
    hr = g_cropCB->Map(0, &readRange, reinterpret_cast<void**>(&g_cropCBMapped));
    if (FAILED(hr)) { DebugLog(L"CropCB Map failed."); return false; }
    return true;
}


std::atomic<uint32_t> g_globalFrameNumber(0);// FEC用フレーム番号

extern uint64_t g_lastRenderedRgbaFrameId; // To track the last rendered frameB

struct VertexPosTex { float x, y, z; float u, v; };

void CleanupD3DRenderResources(); // Forward declaration

// Helper to handle the logic for snapping, and notifying after a resize event.
static void FinalizeResize(HWND hWnd, bool forceAnnounce = false)
{
    // 子ウィンドウ（埋め込み）の場合はトップレベルウィンドウの調整をスキップ
    if (GetWindowLong(hWnd, GWL_STYLE) & WS_CHILD) {
        RECT rc{}; GetClientRect(hWnd, &rc);
        int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
        int tw, th;
        SnapToKnownResolution(cw, ch, tw, th);

        currentResolutionWidth = tw;
        currentResolutionHeight = th;

        g_pendingResize.w.store(cw, std::memory_order_relaxed);
        g_pendingResize.h.store(ch, std::memory_order_relaxed);
        g_pendingResize.has.store(true, std::memory_order_release);

        OnResolutionChanged_GatedSend(tw, th, /*force=*/true);
        InvalidateRect(hWnd, nullptr, FALSE);
        return;
    }

    RECT rc{}; GetClientRect(hWnd, &rc);
    int cw = rc.right - rc.left, ch = rc.bottom - rc.top;

    int tw = 0, th = 0;
    SnapToKnownResolution(cw, ch, tw, th); // tw,th = snapped *video* size (16:9)

    // If the snapped resolution and size are already correct, do nothing,
    // UNLESS a force announce is requested (e.g., after a monitor move).
    if (currentResolutionWidth.load() == tw && currentResolutionHeight.load() == th &&
        cw == tw && ch == th) {
        if (forceAnnounce) {
            // This nudge ensures the server gets a resolution update even if the
            // window size itself didn't change, which can happen on monitor moves.
            OnResolutionChanged_GatedSend(tw, th, /*force=*/true);
        }
        return;
    }

    // Update globals: these represent the *video* resolution (server-side encode)
    currentResolutionWidth  = tw;
    currentResolutionHeight = th;

    // Adjust *outer* window so that client becomes exactly tw x th
    RECT wr{0, 0, tw, th};
    DWORD style = GetWindowLong(hWnd, GWL_STYLE);
    DWORD ex    = GetWindowLong(hWnd, GWL_EXSTYLE);
    AdjustWindowRectEx(&wr, style, GetMenu(hWnd)!=NULL, ex);
    const int ww = wr.right - wr.left, wh = wr.bottom - wr.top;

    RECT wrNow{}; GetWindowRect(hWnd, &wrNow);
    if ((wrNow.right - wrNow.left) != ww || (wrNow.bottom - wrNow.top) != wh) {
        SetWindowPos(hWnd, nullptr, 0, 0, ww, wh,
                     SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    // Enqueue swap-chain resize to exactly tw x th.
    g_pendingResize.w.store(tw, std::memory_order_relaxed);
    g_pendingResize.h.store(th, std::memory_order_relaxed);
    g_pendingResize.has.store(true, std::memory_order_release);

    // Notify server (single gate) with the *video* resolution ONLY
    OnResolutionChanged_GatedSend(tw, th, /*force=*/true);

    InvalidateRect(hWnd, nullptr, FALSE);
}

// State for mouse input handling
static int lastValidX = -1;
static int lastValidY = -1;
static bool isLeftButtonDown = false;
static bool isRightButtonDown = false;

// Helper function to transform window coordinates to video coordinates
bool TransformCursorPos(HWND hWnd, int& outX, int& outY) {
    RECT clientRect;
    GetClientRect(hWnd, &clientRect);

    POINT p;
    GetCursorPos(&p);
    ScreenToClient(hWnd, &p);

    const float bbWidth = static_cast<float>(clientRect.right - clientRect.left);
    const float bbHeight = static_cast<float>(clientRect.bottom - clientRect.top);
    const float videoW = static_cast<float>(currentResolutionWidth.load());
    const float videoH = static_cast<float>(currentResolutionHeight.load());

    if (bbWidth <= 0.0f || bbHeight <= 0.0f || videoW <= 0.0f || videoH <= 0.0f) {
        return false;
    }

    const float bbAspect = bbWidth / bbHeight;
    const float videoAspect = videoW / videoH;

    float vpW, vpH, vpX, vpY;
    if (bbAspect > videoAspect) { // Pillarbox
        vpH = bbHeight;
        vpW = vpH * videoAspect;
        vpX = (bbWidth - vpW) * 0.5f;
        vpY = 0.0f;
    } else { // Letterbox
        vpW = bbWidth;
        vpH = vpW / videoAspect;
        vpX = 0.0f;
        vpY = (bbHeight - vpH) * 0.5f;
    }

    bool isInside = (p.x >= vpX && p.x < vpX + vpW && p.y >= vpY && p.y < vpY + vpH);

    if (isInside) {
        float u = (p.x - vpX) / vpW;
        float v = (p.y - vpY) / vpH;
        outX = static_cast<int>(round(u * (videoW - 1)));
        outY = static_cast<int>(round(v * (videoH - 1)));
        // Clamp to be safe
        outX = std::max(0, std::min((int)videoW - 1, outX));
        outY = std::max(0, std::min((int)videoH - 1, outY));

        lastValidX = outX;
        lastValidY = outY;
    }

    return isInside;
}


UINT64 count = 0;
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    try {
        switch (message) {
        case WM_CLOSE:
            RequestShutdown(); // Signal all threads to exit and post WM_QUIT
            DestroyWindow(hWnd);
            return 0;

        case WM_PAINT:
            {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hWnd, &ps);
                EndPaint(hWnd, &ps);
            }
            return 0;
        case WM_DESTROY:
            RequestShutdown(); // Ensure shutdown is requested, even if WM_CLOSE was bypassed
            return 0;

        case WM_ENTERSIZEMOVE:
            g_isSizing = true;
            return 0;

        case WM_EXITSIZEMOVE:
        {
            g_isSizing = false;

            // 1) Finalize and FORCE the resolution announce at drag end.
            FinalizeResize(hWnd, /*forceAnnounce=*/true);

            // 2) Resume the streaming pipeline even if the snapped size didn’t change.
            BumpStreamGeneration();
            ClearReorderState();

            // 3) Kick the render loop once so we draw immediately.
            g_lastFrameRenderTimeForKick = std::chrono::high_resolution_clock::now() - TARGET_FRAME_DURATION;
            g_forcePresentOnce.store(true, std::memory_order_release); // Present at least once even if no new decoded frame
            DebugLog(L"RenderKick: forced immediate frame after WM_EXITSIZEMOVE.");
            return 0;
        }

        case WM_DPICHANGED:
        {
            const RECT* suggested = reinterpret_cast<const RECT*>(lParam);
            if (suggested) {
                SetWindowPos(hWnd, nullptr,
                             suggested->left, suggested->top,
                             suggested->right - suggested->left,
                             suggested->bottom - suggested->top,
                             SWP_NOZORDER | SWP_NOACTIVATE);
            }
            // After a DPI change, the size and position have changed, so we must
            // trigger the same logic as a completed resize to restart the stream.
            FinalizeResize(hWnd);
            g_forcePresentOnce.store(true, std::memory_order_release);
            break;
        }
        case WM_DISPLAYCHANGE:
        {
            RECT rc{};
            if (GetClientRect(hWnd, &rc)) {
                PostMessageW(hWnd, WM_SIZE, SIZE_RESTORED,
                             MAKELPARAM(rc.right - rc.left, rc.bottom - rc.top));
            }
            break;
        }
        case WM_MOVE:
        {
            // Posting WM_SIZE on every move is unnecessary and can cause issues with
            // the resize/move logic. A move does not imply a size change.
            break;
        }

        case WM_SIZE:
        {
            const int width  = LOWORD(lParam);
            const int height = HIWORD(lParam);

            // Skip only when minimized/zero; still accept sizes during sizing to enable live resize.
            if (wParam == SIZE_MINIMIZED || width == 0 || height == 0) {
                DebugLog(L"WM_SIZE: minimized or zero. skip.");
                return 0;
            }

            // Enqueue a *local* swap-chain resize (render thread will pick this up).
            // The server-side resolution announce continues to be handled in WM_EXITSIZEMOVE via FinalizeResize.
            g_pendingResize.w.store(width,  std::memory_order_relaxed);
            g_pendingResize.h.store(height, std::memory_order_relaxed);
            g_pendingResize.has.store(true, std::memory_order_release);
            g_lastFrameRenderTimeForKick = std::chrono::high_resolution_clock::now(); // render kick

            return 0;
        }

        // Mouse Input Handling
        case WM_MOUSEMOVE:
        {
            int x, y;
            if (TransformCursorPos(hWnd, x, y)) {
                g_mouseInputQueue.enqueue({MOUSE_MOVE, x, y, (unsigned short)wParam, 0, 0, 0});
            }
            return 0;
        }
        case WM_LBUTTONDOWN:
        {
            int x, y;
            if (TransformCursorPos(hWnd, x, y)) {
                isLeftButtonDown = true;
                SetCapture(hWnd);
                g_mouseInputQueue.enqueue({LBUTTON_DOWN, x, y, (unsigned short)wParam, 0, 0, 0});
            }
            return 0;
        }
        case WM_LBUTTONUP:
        {
            isLeftButtonDown = false;
            if (!isRightButtonDown) ReleaseCapture();
            int x, y;
            unsigned char flags = 0;
            if (!TransformCursorPos(hWnd, x, y)) {
                x = lastValidX;
                y = lastValidY;
                flags = POS_IS_LAST_VALID;
            }
            g_mouseInputQueue.enqueue({LBUTTON_UP, x, y, (unsigned short)wParam, 0, 0, flags});
            return 0;
        }
        case WM_RBUTTONDOWN:
        {
            int x, y;
            if (TransformCursorPos(hWnd, x, y)) {
                isRightButtonDown = true;
                SetCapture(hWnd);
                g_mouseInputQueue.enqueue({RBUTTON_DOWN, x, y, (unsigned short)wParam, 0, 0, 0});
            }
            return 0;
        }
        case WM_RBUTTONUP:
        {
            isRightButtonDown = false;
            if (!isLeftButtonDown) ReleaseCapture();
            int x, y;
            unsigned char flags = 0;
            if (!TransformCursorPos(hWnd, x, y)) {
                x = lastValidX;
                y = lastValidY;
                flags = POS_IS_LAST_VALID;
            }
            g_mouseInputQueue.enqueue({RBUTTON_UP, x, y, (unsigned short)wParam, 0, 0, flags});
            return 0;
        }
        case WM_MOUSEWHEEL:
        {
            int x, y;
            if (TransformCursorPos(hWnd, x, y)) {
                g_mouseInputQueue.enqueue({WHEEL, x, y, (unsigned short)GET_KEYSTATE_WPARAM(wParam), (short)GET_WHEEL_DELTA_WPARAM(wParam), 0, 0});
            }
            return 0;
        }
        case WM_MOUSEHWHEEL:
        {
            int x, y;
            if (TransformCursorPos(hWnd, x, y)) {
                g_mouseInputQueue.enqueue({HWHEEL, x, y, (unsigned short)GET_KEYSTATE_WPARAM(wParam), 0, (short)GET_WHEEL_DELTA_WPARAM(wParam), 0});
            }
            return 0;
        }

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }

    } catch (const std::bad_alloc& e) {
        std::wstringstream wss;
        wss << L"!!! std::bad_alloc caught in WndProc !!! message: 0x" << std::hex << message << L", error: " << e.what();
        DebugLog(wss.str());
        // ここで例外を再スローして、呼び出し元での適切なエラーハンドリングを促す
        throw; // 例外を再スローして、呼び出し元での適切なエラーハンドリングを促す
    } catch (const std::exception& e) {
        std::wstringstream wss;
        wss << L"!!! std::exception caught in WndProc !!! message: 0x" << std::hex << message << L", error: " << e.what();
        DebugLog(wss.str());
        throw;
    } catch (...) {
        std::wstringstream wss;
        wss << L"!!! Unknown exception caught in WndProc !!! message: 0x" << std::hex << message;
        DebugLog(wss.str());
        throw;
    }
    return 0;
}

bool InitWindow(HINSTANCE hInstance, int nCmdShow, HWND parentHwnd) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc   = WndProc;
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.hInstance     = hInstance;
    wc.lpszClassName = L"MyWindowClass";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    if (!RegisterClassW(&wc)) {
        if (GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            DebugLog(L"InitWindow: Failed to register window class. Error: " + std::to_wstring(GetLastError()));
            return false;
        }
    }

    int initialWidth = 1920;
    int initialHeight = 1080;
    if (parentHwnd) {
        RECT pr;
        GetClientRect(parentHwnd, &pr);
        initialWidth = pr.right - pr.left;
        initialHeight = pr.bottom - pr.top;
        if (initialWidth <= 0 || initialHeight <= 0) {
            initialWidth = 1280; initialHeight = 720;
        }
    }
    if (!CreateWindowOnBestMonitor(hInstance, nCmdShow, initialWidth, initialHeight, parentHwnd)) {
        return false;
    }

    // クライアント実寸からアスペクト比を維持するように一回送信
    RECT rc{}; GetClientRect(g_hWnd, &rc);
    int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
    int tw, th;
    SnapToKnownResolution(cw, ch, tw, th);

    currentResolutionWidth = tw;
    currentResolutionHeight = th;

    // 初期クライアントサイズが16:9でない場合は調整
    if (cw != tw || ch != th) {
        RECT wr = {0, 0, tw, th};
        DWORD style = GetWindowLong(g_hWnd, GWL_STYLE);
        DWORD ex    = GetWindowLong(g_hWnd, GWL_EXSTYLE);
        AdjustWindowRectEx(&wr, style, GetMenu(g_hWnd) != NULL, ex);
        const int ww = wr.right - wr.left;
        const int wh = wr.bottom - wr.top;
        SetWindowPos(g_hWnd, nullptr, 0, 0, ww, wh, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    // Notify the server with the *video* resolution only.
    // The swap-chain resize to the padded client size will be triggered by the WM_SIZE
    // message that SetWindowPos generates, which is handled by the render thread.
    // OnResolutionChanged_GatedSend(tw, th, /*force=*/true);
    return true;
}

bool InitD3D() {
    // Elevate priority to reduce scheduling jitter.
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
        // Release existing D3D12 resources if re-initializing
    if (g_fence) g_fence.Reset();
    if (g_commandList) g_commandList.Reset();
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        if (g_commandAllocator[i]) g_commandAllocator[i].Reset();
    }
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        if (g_renderTargets[i]) g_renderTargets[i].Reset();
    }
    if (g_rtvHeap) g_rtvHeap.Reset();
    if (g_swapChain) g_swapChain.Reset();
    
    if (g_d3d12CommandQueue) g_d3d12CommandQueue.Reset();
    // Release rendering specific resources
    if (g_srvHeap) g_srvHeap.Reset();
    if (g_vertexBuffer) g_vertexBuffer.Reset();
    if (g_pipelineStateYuv444) g_pipelineStateYuv444.Reset();
    if (g_rootSignature) g_rootSignature.Reset();
    if (g_d3d12Device) g_d3d12Device.Reset();
    if (g_fenceEvent) { CloseHandle(g_fenceEvent); g_fenceEvent = nullptr; }


    // 1. Create Device
    Microsoft::WRL::ComPtr<IDXGIFactory6> dxgiFactory; // Use a newer factory for better features
    // Check for tearing support
    Microsoft::WRL::ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory5)))) {
        BOOL allowTearing = FALSE;
        if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
            g_allowTearing = static_cast<bool>(allowTearing);
            DebugLog(g_allowTearing ? L"InitD3D (D3D12): Tearing is supported." : L"InitD3D (D3D12): Tearing is NOT supported.");
        }
    }

    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create DXGIFactory. HRESULT: " + HResultToHexWString(hr));
        return false;
    }

    UINT createDeviceFlags = 0;
    
#if defined(_DEBUG)
    Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
        debugController->EnableDebugLayer();
        createDeviceFlags |= DXGI_CREATE_FACTORY_DEBUG; // Enable additional debug layers
    }
#endif
    // Recreate factory with debug flag if enabled
    hr = CreateDXGIFactory2(createDeviceFlags, IID_PPV_ARGS(&dxgiFactory));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create DXGIFactory (with debug). HRESULT: " + HResultToHexWString(hr));
        return false;
    }

    // Find a hardware adapter
    Microsoft::WRL::ComPtr<IDXGIAdapter1> hardwareAdapter;
    for (UINT adapterIndex = 0; dxgiFactory->EnumAdapters1(adapterIndex, &hardwareAdapter) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
        DXGI_ADAPTER_DESC1 desc;
        hardwareAdapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue; // Skip software adapter
        if (SUCCEEDED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) break; // Check if D3D12 is supported
        hardwareAdapter.Reset();
    }
    if (hardwareAdapter == nullptr) {
        DebugLog(L"InitD3D (D3D12): No suitable D3D12 hardware adapter found.");
        return false;
    }

    hr = D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g_d3d12Device));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create D3D12 device. HRESULT: " + HResultToHexWString(hr));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): D3D12 Device created.");

    // 2. Create Command Queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = g_d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&g_d3d12CommandQueue));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create command queue. HRESULT: " + HResultToHexWString(hr));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): Command Queue created.");

    // 3. Create Swap Chain
    DXGI_SWAP_CHAIN_DESC1 scd1 = {};
    scd1.BufferCount = kSwapChainBufferCount; // Triple buffering
    RECT rcClient;
    GetClientRect(g_hWnd, &rcClient);
    scd1.Width = (rcClient.right - rcClient.left > 0) ? (rcClient.right - rcClient.left) : 1;
    scd1.Height = (rcClient.bottom - rcClient.top > 0) ? (rcClient.bottom - rcClient.top) : 1;
    scd1.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd1.Stereo = FALSE;
    scd1.SampleDesc.Count = 1;
    scd1.SampleDesc.Quality = 0;
    scd1.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd1.Scaling = DXGI_SCALING_NONE;   // 既存は DXGI_SCALING_STRETCH
    scd1.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; // Recommended for D3D12
    scd1.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
    scd1.Flags = (g_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0) | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

    Microsoft::WRL::ComPtr<IDXGISwapChain1> tempSwapChain;
    hr = dxgiFactory->CreateSwapChainForHwnd(g_d3d12CommandQueue.Get(), g_hWnd, &scd1, nullptr, nullptr, &tempSwapChain);
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create swap chain. HRESULT: " + HResultToHexWString(hr));
        return false;
    }
    hr = tempSwapChain.As(&g_swapChain); // QueryInterface to IDXGISwapChain3
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to QI for IDXGISwapChain3. HRESULT: " + HResultToHexWString(hr));
        return false;
    }

    {
        Microsoft::WRL::ComPtr<IDXGISwapChain2> sc2;
        HRESULT hr_sc2 = g_swapChain.As(&sc2);
        if (SUCCEEDED(hr_sc2) && sc2) {
            sc2->SetMaximumFrameLatency(1);
            g_frameLatencyWaitableObject = sc2->GetFrameLatencyWaitableObject();
        }
    }

    g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();
    DebugLog(L"InitD3D (D3D12): Swap Chain created.");

    // 4. Create RTV Descriptor Heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = kSwapChainBufferCount; // For triple buffering
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    hr = g_d3d12Device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&g_rtvHeap));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create RTV descriptor heap. HRESULT: " + HResultToHexWString(hr));
        return false;
    }
    g_rtvDescriptorSize = g_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    DebugLog(L"InitD3D (D3D12): RTV Descriptor Heap created.");

    // 5. Create Render Target Views
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(g_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        hr = g_swapChain->GetBuffer(i, IID_PPV_ARGS(&g_renderTargets[i]));
        if (FAILED(hr)) {
            DebugLog(L"InitD3D (D3D12): Failed to get swap chain buffer " + std::to_wstring(i) + L". HRESULT: " + HResultToHexWString(hr));
            return false;
        }
        g_d3d12Device->CreateRenderTargetView(g_renderTargets[i].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, g_rtvDescriptorSize);
    }
    DebugLog(L"InitD3D (D3D12): Render Target Views created.");

    // 6. Create Command Allocator
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        hr = g_d3d12Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&g_commandAllocator[i]));
        if (FAILED(hr)) {
            DebugLog(L"InitD3D (D3D12): Failed to create command allocator[" + std::to_wstring(i) + L"].");
            return false;
        }
    }
    DebugLog(L"InitD3D (D3D12): Command Allocators created.");

    // 7. Create Command List
    hr = g_d3d12Device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        g_commandAllocator[g_currentFrameBufferIndex].Get(), nullptr,
        IID_PPV_ARGS(&g_commandList));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create command list. HRESULT: " + HResultToHexWString(hr));
        return false;
    }
    g_commandList->Close(); // Close it initially, reset before use
    DebugLog(L"InitD3D (D3D12): Command List created.");

    // 8. Create Fence and Event
    hr = g_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create fence. HRESULT: " + HResultToHexWString(hr));
        return false;
    }

// [NEW] Create a *shared* fence for CUDA interop (do not alter formatting around here)
hr = g_d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_copyFence));
if (FAILED(hr)) {
    DebugLog(L"InitD3D (D3D12): Failed to create shared fence for CUDA interop. HRESULT: " + HResultToHexWString(hr));
    return false;
}
if (g_copyFenceSharedHandle) { CloseHandle(g_copyFenceSharedHandle); g_copyFenceSharedHandle = nullptr; }
hr = g_d3d12Device->CreateSharedHandle(g_copyFence.Get(), nullptr, GENERIC_ALL, nullptr, &g_copyFenceSharedHandle);
if (FAILED(hr) || !g_copyFenceSharedHandle) {
    DebugLog(L"InitD3D (D3D12): CreateSharedHandle(copy fence) failed. HRESULT: " + HResultToHexWString(hr));
    return false;
}
    g_fenceValue = 1;
    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (g_fenceEvent == nullptr) {
        DebugLog(L"InitD3D (D3D12): Failed to create fence event. Error: " + std::to_wstring(GetLastError()));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): Fence and Event created.");

    // --- Create rendering resources (Root Signature, PSO, Vertex Buffer, SRV Heap) ---

    // 9. Create Root Signature
    D3D12_DESCRIPTOR_RANGE1 srvRanges[3];
    srvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[0].NumDescriptors = 1; // For Y Texture
    srvRanges[0].BaseShaderRegister = 0; // t0
    srvRanges[0].RegisterSpace = 0;
    srvRanges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
    srvRanges[0].OffsetInDescriptorsFromTableStart = 0;

    srvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[1].NumDescriptors = 1; // For U Texture
    srvRanges[1].BaseShaderRegister = 1; // t1
    srvRanges[1].RegisterSpace = 0;
    srvRanges[1].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
    srvRanges[1].OffsetInDescriptorsFromTableStart = 1;

    srvRanges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[2].NumDescriptors = 1; // For V Texture
    srvRanges[2].BaseShaderRegister = 2; // t2
    srvRanges[2].RegisterSpace = 0;
    srvRanges[2].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE;
    srvRanges[2].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER1 rootParameters[2]; // ★ 1 → 2
    // [0] SRV descriptor table (既存)
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    rootParameters[0].DescriptorTable.NumDescriptorRanges = _countof(srvRanges);
    rootParameters[0].DescriptorTable.pDescriptorRanges = srvRanges;

    // ★ [1] 追加: Crop用 CBV (b0)
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    rootParameters[1].Descriptor.ShaderRegister = 0; // b0
    rootParameters[1].Descriptor.RegisterSpace = 0;
    rootParameters[1].Descriptor.Flags = D3D12_ROOT_DESCRIPTOR_FLAG_NONE;

    D3D12_STATIC_SAMPLER_DESC staticSampler = {};
    staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSampler.MipLODBias = 0;
    staticSampler.MaxAnisotropy = 0;
    staticSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    staticSampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    staticSampler.MinLOD = 0.0f;
    staticSampler.MaxLOD = D3D12_FLOAT32_MAX;
    staticSampler.ShaderRegister = 0; // s0
    staticSampler.RegisterSpace = 0;
    staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 1, &staticSampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    Microsoft::WRL::ComPtr<ID3DBlob> signatureBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
    hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signatureBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) DebugLog(L"InitD3D (D3D12): D3D12SerializeVersionedRootSignature failed: " + std::wstring(static_cast<wchar_t*>(errorBlob->GetBufferPointer()), static_cast<wchar_t*>(errorBlob->GetBufferPointer()) + errorBlob->GetBufferSize() / sizeof(wchar_t)));
        else DebugLog(L"InitD3D (D3D12): D3D12SerializeVersionedRootSignature failed. HR: " + HResultToHexWString(hr));
        return false;
    }
    hr = g_d3d12Device->CreateRootSignature(0, signatureBlob->GetBufferPointer(), signatureBlob->GetBufferSize(), IID_PPV_ARGS(&g_rootSignature));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create root signature. HR: " + HResultToHexWString(hr));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): Root Signature created.");

    // 10. Create PSO
    Microsoft::WRL::ComPtr<ID3DBlob> vertexShaderBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> pixelShaderBlobYuv444;

    UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
    compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

    hr = D3DCompileFromFile(L"Shader/FullScreenQuadVS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "vs_5_1", compileFlags, 0, &vertexShaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) DebugLog(L"InitD3D (D3D12): Vertex shader compilation failed: " + std::wstring(static_cast<wchar_t*>(errorBlob->GetBufferPointer()), static_cast<wchar_t*>(errorBlob->GetBufferPointer()) + errorBlob->GetBufferSize() / sizeof(wchar_t)));
        else DebugLog(L"InitD3D (D3D12): Vertex shader compilation failed. HR: " + HResultToHexWString(hr));
        return false;
    }
    hr = D3DCompileFromFile(L"Shader/YUV444ToRGBA709Full.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_1", compileFlags, 0, &pixelShaderBlobYuv444, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) DebugLog(L"InitD3D (D3D12): Pixel shader compilation failed: " + std::wstring(static_cast<wchar_t*>(errorBlob->GetBufferPointer()), static_cast<wchar_t*>(errorBlob->GetBufferPointer()) + errorBlob->GetBufferSize() / sizeof(wchar_t)));
        else DebugLog(L"InitD3D (D3D12): Pixel shader compilation failed. HR: " + HResultToHexWString(hr));
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = g_rootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShaderBlob.Get());
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShaderBlobYuv444.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    // No input layout for full-screen quad generated in VS
    psoDesc.InputLayout = { nullptr, 0 };

    hr = g_d3d12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&g_pipelineStateYuv444));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create PSO. HR: " + HResultToHexWString(hr));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): PSO created.");

    // 11. Create SRV Descriptor Heap (for Y and UV textures)
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    // srvHeapDesc.NumDescriptors = 2 * 2; // 2 textures (Y, UV) per frame buffer for double buffering (or more if needed)
                                        // For simplicity, let's assume we update descriptors for the current frame's textures.
                                        // So, 2 descriptors are enough if we update them each frame.
                                        // If we pre-create for all decoder surfaces, it'd be NUM_DECODE_SURFACES_IN_POOL * 2.
                                        // Let's start with 2, for the current frame's Y and UV.
    srvHeapDesc.NumDescriptors = kSrvHeapSize; // [FIX] Use a large ring buffer for SRVs to avoid overwriting descriptors in-flight.
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = g_d3d12Device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&g_srvHeap));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create SRV descriptor heap. HR: " + HResultToHexWString(hr));
        return false;
    }
    g_srvDescriptorSize = g_d3d12Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    DebugLog(L"InitD3D (D3D12): SRV Descriptor Heap created.");

    // Vertex buffer is not strictly needed for a full-screen triangle generated in VS,
    // but if you were to use one:
    // Create vertex buffer for a quad (or use SV_VertexID in VS to generate one)

    if (!CreateCropCB()) {
        DebugLog(L"InitD3D (D3D12): Failed to create crop constant buffer.");
        return false;
    }

    if (!CreateOverlayResources()) {
        DebugLog(L"InitD3D (D3D12): Failed to create overlay resources.");
        return false;
    }

    return true;
}

void CleanupOverlayResources() {
    g_overlayQuadPso.Reset();
    g_overlayQuadRootSignature.Reset();
    g_overlayTextPso.Reset();
    g_overlayRootSignature.Reset();
    g_overlayVertexBuffer.Reset();
    g_textTexture.Reset();
    g_textSrvHeap.Reset();
    DebugLog(L"CleanupOverlayResources: Overlay resources cleaned up.");
}

bool CreateOverlayResources() {
    HRESULT hr;

    // --- Root Signature for Text ---
    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;
    if (FAILED(g_d3d12Device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))) {
        featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    D3D12_DESCRIPTOR_RANGE1 ranges[1];
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER1 rootParameters[1];
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    rootParameters[0].DescriptorTable.NumDescriptorRanges = _countof(ranges);
    rootParameters[0].DescriptorTable.pDescriptorRanges = ranges;

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MipLODBias = 0;
    sampler.MaxAnisotropy = 0;
    sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    sampler.BorderColor = D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderRegister = 0;
    sampler.RegisterSpace = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    Microsoft::WRL::ComPtr<ID3DBlob> signature;
    Microsoft::WRL::ComPtr<ID3DBlob> error;
    hr = D3D12SerializeVersionedRootSignature(&rootSignatureDesc, &signature, &error);
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: D3D12SerializeVersionedRootSignature for text failed.");
        return false;
    }
    hr = g_d3d12Device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&g_overlayRootSignature));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateRootSignature for text failed.");
        return false;
    }

    // --- Root Signature for Quad ---
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC quadRootSignatureDesc;
    quadRootSignatureDesc.Init_1_1(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    Microsoft::WRL::ComPtr<ID3DBlob> quadSignature;
    hr = D3D12SerializeVersionedRootSignature(&quadRootSignatureDesc, &quadSignature, &error);
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: D3D12SerializeVersionedRootSignature for quad failed.");
        return false;
    }
    hr = g_d3d12Device->CreateRootSignature(0, quadSignature->GetBufferPointer(), quadSignature->GetBufferSize(), IID_PPV_ARGS(&g_overlayQuadRootSignature));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateRootSignature for quad failed.");
        return false;
    }


    // --- PSO for Transparent Quad ---
    Microsoft::WRL::ComPtr<ID3DBlob> vertexShaderBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> pixelShaderBlob;
    UINT compileFlags = D3DCOMPILE_ENABLE_STRICTNESS;
    #if defined(_DEBUG)
    compileFlags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
    #endif

    hr = D3DCompileFromFile(L"Shader/FullScreenQuadVS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "vs_5_1", compileFlags, 0, &vertexShaderBlob, &error);
    if (FAILED(hr)) {
         if (error) DebugLog(L"CreateOverlayResources: FullScreenQuadVS.hlsl compilation failed.");
        return false;
    }
    hr = D3DCompileFromFile(L"Shader/TransparentQuadPS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_1", compileFlags, 0, &pixelShaderBlob, &error);
    if (FAILED(hr)) {
        if (error) DebugLog(L"CreateOverlayResources: TransparentQuadPS.hlsl compilation failed.");
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = g_overlayQuadRootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShaderBlob.Get());
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShaderBlob.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);

    D3D12_BLEND_DESC blendDesc = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    psoDesc.BlendState = blendDesc;

    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.InputLayout = { nullptr, 0 };

    hr = g_d3d12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&g_overlayQuadPso));
     if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateGraphicsPipelineState for quad failed.");
        return false;
    }

    // --- PSO for Text ---
    Microsoft::WRL::ComPtr<ID3DBlob> textVertexShaderBlob;
    Microsoft::WRL::ComPtr<ID3DBlob> textPixelShaderBlob;

    hr = D3DCompileFromFile(L"Shader/OverlayVS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "vs_5_1", compileFlags, 0, &textVertexShaderBlob, &error);
     if (FAILED(hr)) {
        if (error) DebugLog(L"CreateOverlayResources: OverlayVS.hlsl compilation failed.");
        return false;
    }
    hr = D3DCompileFromFile(L"Shader/OverlayPS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_1", compileFlags, 0, &textPixelShaderBlob, &error);
    if (FAILED(hr)) {
        if (error) DebugLog(L"CreateOverlayResources: OverlayPS.hlsl compilation failed.");
        return false;
    }

    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC textPsoDesc = {}; // Initialize from scratch
    textPsoDesc.pRootSignature = g_overlayRootSignature.Get();
    textPsoDesc.VS = CD3DX12_SHADER_BYTECODE(textVertexShaderBlob.Get());
    textPsoDesc.PS = CD3DX12_SHADER_BYTECODE(textPixelShaderBlob.Get());
    textPsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    textPsoDesc.BlendState = psoDesc.BlendState; // We can safely copy the blend state
    textPsoDesc.DepthStencilState.DepthEnable = FALSE;
    textPsoDesc.DepthStencilState.StencilEnable = FALSE;
    textPsoDesc.SampleMask = UINT_MAX;
    textPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    textPsoDesc.NumRenderTargets = 1;
    textPsoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    textPsoDesc.SampleDesc.Count = 1;
    textPsoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
    hr = g_d3d12Device->CreateGraphicsPipelineState(&textPsoDesc, IID_PPV_ARGS(&g_overlayTextPso));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateGraphicsPipelineState for text failed.");
        return false;
    }

    // --- Vertex Buffer for Text Quad ---
    VertexPosTex vertices[] = {
        { -0.5f,  0.25f, 0.0f, 0.0f, 0.0f },
        { -0.5f, -0.25f, 0.0f, 0.0f, 1.0f },
        {  0.5f,  0.25f, 0.0f, 1.0f, 0.0f },
        {  0.5f, -0.25f, 0.0f, 1.0f, 1.0f }
    };
    const UINT vertexBufferSize = sizeof(vertices);

    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize);
    hr = g_d3d12Device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&g_overlayVertexBuffer));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateCommittedResource for vertex buffer failed.");
        return false;
    }

    UINT8* pVertexDataBegin;
    CD3DX12_RANGE readRange(0, 0);
    hr = g_overlayVertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin));
     if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: Map vertex buffer failed.");
        return false;
    }
    memcpy(pVertexDataBegin, vertices, sizeof(vertices));
    g_overlayVertexBuffer->Unmap(0, nullptr);

    g_overlayVertexBufferView.BufferLocation = g_overlayVertexBuffer->GetGPUVirtualAddress();
    g_overlayVertexBufferView.StrideInBytes = sizeof(VertexPosTex);
    g_overlayVertexBufferView.SizeInBytes = vertexBufferSize;

    // --- Create Text Texture using GDI ---
    const int textureWidth = 512;
    const int textureHeight = 128;
    const wchar_t* text = L"System Rebooting...";

    HDC hdc = CreateCompatibleDC(nullptr);
    if (!hdc) return false;

    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = textureWidth;
    bmi.bmiHeader.biHeight = -textureHeight; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* pPixels = nullptr;
    HBITMAP hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &pPixels, NULL, 0);
    if (!hBitmap) {
        DeleteDC(hdc);
        return false;
    }

    HBITMAP hOldBitmap = (HBITMAP)SelectObject(hdc, hBitmap);

    SetBkColor(hdc, RGB(0, 0, 0));
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(255, 255, 255));
    HFONT hFont = CreateFont(48, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Arial");
    HFONT hOldFont = (HFONT)SelectObject(hdc, hFont);

    RECT rect = { 0, 0, textureWidth, textureHeight };
    FillRect(hdc, &rect, (HBRUSH)GetStockObject(BLACK_BRUSH)); // Fill with black for debugging
    TextOutW(hdc, 10, 30, text, (int)wcslen(text));

    // Cleanup GDI objects
    SelectObject(hdc, hOldFont);
    DeleteObject(hFont);
    SelectObject(hdc, hOldBitmap);

    // Create D3D12 Texture
    auto texHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_B8G8R8A8_UNORM, textureWidth, textureHeight, 1, 1);
    hr = g_d3d12Device->CreateCommittedResource(&texHeapProps, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_textTexture));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateCommittedResource for text texture failed.");
        DeleteObject(hBitmap);
        DeleteDC(hdc);
        return false;
    }

    // Upload pixel data
    Microsoft::WRL::ComPtr<ID3D12Resource> uploadHeap;
    UINT64 uploadBufferSize = GetRequiredIntermediateSize(g_textTexture.Get(), 0, 1);
    auto uploadHeapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize);
    hr = g_d3d12Device->CreateCommittedResource(&uploadHeapProps, D3D12_HEAP_FLAG_NONE, &uploadBufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadHeap));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateCommittedResource for upload heap failed.");
        DeleteObject(hBitmap);
        DeleteDC(hdc);
        return false;
    }

    D3D12_SUBRESOURCE_DATA textureData = {};
    textureData.pData = pPixels;
    textureData.RowPitch = textureWidth * 4;
    textureData.SlicePitch = textureData.RowPitch * textureHeight;

    // The command list needs to be open to do this
    ID3D12CommandAllocator* allocator = g_commandAllocator[g_currentFrameBufferIndex].Get();
    allocator->Reset();
    g_commandList->Reset(allocator, nullptr);

    UpdateSubresources(g_commandList.Get(), g_textTexture.Get(), uploadHeap.Get(), 0, 0, 1, &textureData);
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(g_textTexture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    g_commandList->ResourceBarrier(1, &barrier);

    g_commandList->Close();
    ID3D12CommandList* ppCommandLists[] = { g_commandList.Get() };
    g_d3d12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    WaitForGpu(); // Wait for the upload to complete

    // Cleanup GDI resources now that data is on GPU
    DeleteObject(hBitmap);
    DeleteDC(hdc);

    // Create SRV for the text texture
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = 1;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = g_d3d12Device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&g_textSrvHeap));
    if (FAILED(hr)) {
        DebugLog(L"CreateOverlayResources: CreateDescriptorHeap for text SRV failed.");
        return false;
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Format = texDesc.Format;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    g_d3d12Device->CreateShaderResourceView(g_textTexture.Get(), &srvDesc, g_textSrvHeap->GetCPUDescriptorHandleForHeapStart());

    DebugLog(L"CreateOverlayResources: Successfully created all overlay resources.");
    return true;
}

UINT64 PopulateCommandListCount = 0;
// 修正対象ファイル: window.cpp

// 既存の PopulateCommandList 関数をまるごと置き換えてください
bool PopulateCommandList(ReadyGpuFrame& outFrameToRender) { // Return bool, pass ReadyGpuFrame by reference
    DrainRetireBin(); // safe: runs on render thread, preserves timing/logging

    // Reset command allocator and command list
    ID3D12CommandAllocator* allocator = g_commandAllocator[g_currentFrameBufferIndex].Get();
    HRESULT hr;
    hr = allocator->Reset();
    if (FAILED(hr)) { DebugLog(L"PopulateCommandList: allocator Reset failed."); return false; }

    hr = g_commandList->Reset(allocator, nullptr);
    if (FAILED(hr)) { DebugLog(L"PopulateCommandList: cmdlist Reset failed."); return false; }

    // Transition the current back buffer from PRESENT to RENDER_TARGET.
    if (!g_renderTargets[g_currentFrameBufferIndex]) {
        g_commandList->Close();
        return false;
    }
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = g_renderTargets[g_currentFrameBufferIndex].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_commandList->ResourceBarrier(1, &barrier);

    // Set RTV
    g_commandList->SetPipelineState(g_pipelineStateYuv444.Get());
    g_commandList->SetGraphicsRootSignature(g_rootSignature.Get());
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(g_rtvHeap->GetCPUDescriptorHandleForHeapStart(), g_currentFrameBufferIndex, g_rtvDescriptorSize);
    g_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    // [修正点 1] レンダーターゲットをセットした直後にクリア処理を行う
    const float clearColor[4] = {0.f, 0.f, 0.f, 1.f};
    g_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

    // [修正点 2] クラッシュの原因となっていた冗長な SetViewportScissorToBackbuffer 呼び出しを削除 (※このコメントは元コードの意図を汲んだものです)
    // この処理は、描画するフレームがある場合は SetLetterboxViewport で、
    // ない場合は後段の else ブロックで安全に実行されるため、ここでの呼び出しは不要です。

    // --- Crash and Flicker Fix ---
    bool isNewFrame = false;
    ReadyGpuFrame frameToDraw;

    // 1. Pull all new frames into the reorder buffer.
    {
        std::unique_lock<std::mutex> qlock(g_readyGpuFrameQueueMutex);
        while (!g_readyGpuFrameQueue.empty()) {
            ReadyGpuFrame f = std::move(g_readyGpuFrameQueue.back());
            g_readyGpuFrameQueue.pop_back();
            std::lock_guard<std::mutex> rlock(g_reorderMutex);
            auto it = g_reorderBuffer.find(f.streamFrameNumber);
            if (it != g_reorderBuffer.end()) {
                if (it->second.copyDone) cuEventDestroy(it->second.copyDone);
                it->second = std::move(f);
            } else {
                g_reorderBuffer.emplace(f.streamFrameNumber, std::move(f));
            }
        }
    }

    // 2. Decide which frame to draw (new or cached) and manage event ownership.
    {
        std::lock_guard<std::mutex> rlock(g_reorderMutex);

        // Optional low-latency mode: if multiple frames have queued up,
        // drop all but the newest one to minimize latency.
        static const bool kLowLatencyDropOld =
            (GetEnvironmentVariableW(L"LOW_LATENCY_DROP_OLD", nullptr, 0) != 0);

        if (kLowLatencyDropOld) {
            if (g_reorderBuffer.size() > 1) {
                // Keep only the greatest streamFrameNumber
                auto it_last = std::prev(g_reorderBuffer.end());
                uint32_t newest_key = it_last->first;
                ReadyGpuFrame newest = std::move(it_last->second);
                g_reorderBuffer.clear(); // Invalidate iterators
                g_reorderBuffer.emplace(newest_key, std::move(newest));
                g_expectedStreamFrame = newest_key; // align expected
            }
        }

        ReadyGpuFrame newFrameFromReorder;
        bool hasNewFrame = false;

        // --- Reorder logic to find a new frame ---
        auto now = std::chrono::steady_clock::now();
        if (!g_expectedInitialized) {
            // Initialization logic
            if (!g_reorderBuffer.empty()) {
                g_expectedStreamFrame = g_reorderBuffer.begin()->first;
                g_expectedInitialized = true;
            }
        }
        const bool haveExpected = g_reorderBuffer.count(g_expectedStreamFrame) != 0;
        const bool waitedLong = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_lastReorderDecision).count() > GetReorderWaitMsForDepth(g_reorderBuffer.size());

        if (haveExpected || (!g_reorderBuffer.empty() && waitedLong)) {
            auto it = g_reorderBuffer.find(g_expectedStreamFrame);
            if (it == g_reorderBuffer.end()) {
                it = g_reorderBuffer.begin();
            }
            newFrameFromReorder = std::move(it->second);
            g_expectedStreamFrame = it->first + 1;
            g_reorderBuffer.erase(it);
            hasNewFrame = true;
            g_lastReorderDecision = now;
        }
        // --- End reorder logic ---

        if (hasNewFrame) {
            isNewFrame = true;
            if (g_lastDrawnFrame.copyDone) {
                cuEventDestroy(g_lastDrawnFrame.copyDone);
            }
            g_lastDrawnFrame = newFrameFromReorder;
            newFrameFromReorder.copyDone = nullptr; // Ownership transferred to cache
            UpdateVideoTimestamp(g_lastDrawnFrame.timestamp);
        }

        if (g_lastDrawnFrame.hw_decoded_texture_Y) {
            frameToDraw = g_lastDrawnFrame;
            frameToDraw.copyDone = nullptr;
        }
    }

    // 3. Draw the selected frame.
    const bool canDrawFrame = frameToDraw.hw_decoded_texture_Y && frameToDraw.hw_decoded_texture_U &&
        frameToDraw.hw_decoded_texture_V;
    if (canDrawFrame) {
        if (isNewFrame) {
            frameToDraw.render_start_ms = SteadyNowMs();
            // We must wait on the event from the master copy in the cache.
            std::lock_guard<std::mutex> rlock(g_reorderMutex);
            // Propagate the render-start time to the master cached frame, so that
            // latency stats are correct.
            g_lastDrawnFrame.render_start_ms = frameToDraw.render_start_ms;
            // GPU-GPU sync: ensure render queue does not sample until CUDA copy signaled completion.
            if (frameToDraw.fenceValue != 0 && g_copyFence) {
                g_d3d12CommandQueue->Wait(g_copyFence.Get(), static_cast<UINT64>(frameToDraw.fenceValue));
            } else if (g_lastDrawnFrame.copyDone) {
                // Existing CUDA-event fallback logic (no-op wait, for diagnostics).
                (void)cuEventQuery(g_lastDrawnFrame.copyDone);
            }
        }

        g_commandList->SetPipelineState(g_pipelineStateYuv444.Get());

        // Use display dimensions for letterboxing, not coded dimensions or window dimensions.
        const int videoW = frameToDraw.displayW;
        const int videoH = frameToDraw.displayH;
        ID3D12Resource* backbuffer = g_renderTargets[g_currentFrameBufferIndex].Get();
        if (backbuffer) {
            SetLetterboxViewport(g_commandList.Get(), backbuffer->GetDesc(), videoW, videoH);
        }

        // --- Update and bind crop constant buffer ---
        if (g_cropCBMapped) {
            CropCBData cb{};
            cb.uvBias[0]  = frameToDraw.uvMinX;
            cb.uvBias[1]  = frameToDraw.uvMinY;
            cb.uvScale[0] = frameToDraw.uvMaxX - frameToDraw.uvMinX;
            cb.uvScale[1] = frameToDraw.uvMaxY - frameToDraw.uvMinY;
            std::memcpy(g_cropCBMapped, &cb, sizeof(cb));
        }
        g_commandList->SetGraphicsRootConstantBufferView(1, g_cropCB->GetGPUVirtualAddress());


        // --- SRV setup (Y, U[, V]) ---
        // 連続スロットを保証するための wrap 前処理（性能影響なし）
        const UINT kNeededSrv = 3u;
        if (g_srvDescriptorHeapIndex + kNeededSrv > kSrvHeapSize) {
            // ★ wrap：ヒープ末尾近くで3連続確保できない場合は先頭に巻き戻す
            g_srvDescriptorHeapIndex = 0;
        }
        const UINT startIndex = g_srvDescriptorHeapIndex;

        // CPU/GPU 両方のハンドルを同じ startIndex から計算（従来コメントは維持）
        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandleCpu(
            g_srvHeap->GetCPUDescriptorHandleForHeapStart(), startIndex, g_srvDescriptorSize);
        CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandleGpu(
            g_srvHeap->GetGPUDescriptorHandleForHeapStart(), startIndex, g_srvDescriptorSize);

        // SRV 共通設定
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;

        // Y plane
        srvDesc.Format = frameToDraw.hw_decoded_texture_Y->GetDesc().Format;
        g_d3d12Device->CreateShaderResourceView(
            frameToDraw.hw_decoded_texture_Y.Get(), &srvDesc, srvHandleCpu);
        srvHandleCpu.Offset(1, g_srvDescriptorSize);

        // U plane
        srvDesc.Format = frameToDraw.hw_decoded_texture_U->GetDesc().Format;
        g_d3d12Device->CreateShaderResourceView(
            frameToDraw.hw_decoded_texture_U.Get(), &srvDesc, srvHandleCpu);
        srvHandleCpu.Offset(1, g_srvDescriptorSize);

        // V plane
        srvDesc.Format = frameToDraw.hw_decoded_texture_V->GetDesc().Format;
        g_d3d12Device->CreateShaderResourceView(
            frameToDraw.hw_decoded_texture_V.Get(), &srvDesc, srvHandleCpu);

        // シェーダ可視ヒープ設定（従来通り）
        ID3D12DescriptorHeap* ppHeaps[] = { g_srvHeap.Get() };
        g_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
        g_commandList->SetGraphicsRootDescriptorTable(0, srvHandleGpu);

        // リングの書き込み位置を必要数進める（従来のモジュロ運用は維持）
        g_srvDescriptorHeapIndex = (startIndex + kNeededSrv) % kSrvHeapSize;
        g_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        g_commandList->DrawInstanced(4, 1, 0, 0);
    } else {
        // [修正点 3] 描画するフレームがない場合、ビューポートをバックバッファ全体に設定する
        ID3D12Resource* backbuffer = g_renderTargets[g_currentFrameBufferIndex].Get();
        if (backbuffer) {
            SetViewportScissorToBackbuffer(g_commandList.Get(), backbuffer);
        }
    }

    if (isNewFrame) {
        std::lock_guard<std::mutex> rlock(g_reorderMutex);
        outFrameToRender = g_lastDrawnFrame;
    }

    // --- Draw Overlay if enabled ---
    if (g_showRebootOverlay.load(std::memory_order_relaxed)) {
        // Draw semi-transparent black quad
        g_commandList->SetPipelineState(g_overlayQuadPso.Get());
        g_commandList->SetGraphicsRootSignature(g_overlayQuadRootSignature.Get());
        g_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        g_commandList->DrawInstanced(4, 1, 0, 0);

        // Draw "System Rebooting..." text
        g_commandList->SetPipelineState(g_overlayTextPso.Get());
        g_commandList->SetGraphicsRootSignature(g_overlayRootSignature.Get());
        ID3D12DescriptorHeap* ppHeaps[] = { g_textSrvHeap.Get() };
        g_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
        g_commandList->SetGraphicsRootDescriptorTable(0, g_textSrvHeap->GetGPUDescriptorHandleForHeapStart());
        g_commandList->IASetVertexBuffers(0, 1, &g_overlayVertexBufferView);
        g_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        g_commandList->DrawInstanced(4, 1, 0, 0);
    }


    // Transition the current back buffer from RENDER_TARGET to PRESENT.
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_commandList->ResourceBarrier(1, &barrier);

    hr = g_commandList->Close();
    if (FAILED(hr)) { DebugLog(L"PopulateCommandList: Failed to close command list. HR: " + HResultToHexWString(hr)); return false; }
    return isNewFrame;
}

static void ResizeSwapChainOnRenderThread(int newW, int newH) {
    if (!g_swapChain || !g_d3d12Device || !g_d3d12CommandQueue) return;

    // Get existing desc; skip if already correct
    DXGI_SWAP_CHAIN_DESC1 desc{};
    g_swapChain->GetDesc1(&desc);
    if (desc.Width == (UINT)newW && desc.Height == (UINT)newH) return;

    // Bounded wait so we never hard-freeze the pipeline
    auto WaitForGpuWithTimeout = [](DWORD totalTimeoutMs)->bool {
        // Signal
        UINT64 fenceValueToSignal = g_fenceValue;
        HRESULT hr = g_d3d12CommandQueue->Signal(g_fence.Get(), fenceValueToSignal);
        if (FAILED(hr)) { DebugLog(L"WaitForGpuWithTimeout: Signal failed. HR: " + HResultToHexWString(hr)); return false; }

        const DWORD step = 50; // ms
        DWORD waited = 0;
        while (g_fence->GetCompletedValue() < fenceValueToSignal) {
            hr = g_fence->SetEventOnCompletion(fenceValueToSignal, g_fenceEvent);
            if (FAILED(hr)) { DebugLog(L"WaitForGpuWithTimeout: SetEventOnCompletion failed. HR: " + HResultToHexWString(hr)); return false; }
            DWORD r = MsgWaitForMultipleObjects(1, &g_fenceEvent, FALSE, step, QS_ALLINPUT);
            if (r == WAIT_OBJECT_0) break; // fence signaled
            // pump messages to keep app responsive
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            waited += step;
            if (waited >= totalTimeoutMs) {
                DebugLog(L"WaitForGpuWithTimeout: timed out; proceeding cautiously.");
                break;
            }
        }
        g_fenceValue++;
        return true;
    };

    if (newW == 0 || newH == 0) { DebugLog(L"Resize skipped: zero size."); return; }
    DebugLog(L"RenderThread: resizing swap-chain to " + std::to_wstring(newW) + L"x" + std::to_wstring(newH));

    // **Wait BEFORE clearing and releasing any resources**
    WaitForGpuWithTimeout(500); // or WaitForGpu();

    // Clear any buffered frames, as their underlying resources might be invalid
    // after a resize. This prevents using stale D3D resources from the decoder.
    DebugLog(L"Resize detected. Clearing in-flight frame queues.");
    {
        std::lock_guard<std::mutex> qlock(g_readyGpuFrameQueueMutex);
        // The ComPtrs in the deque will automatically be released.
        g_readyGpuFrameQueue.clear();
    }
    ClearReorderState(); // This clears the reorder buffer and resets sequence.

    for (UINT i = 0; i < kSwapChainBufferCount; ++i) g_renderTargets[i].Reset();

    HRESULT hr = g_swapChain->ResizeBuffers(
        kSwapChainBufferCount, newW, newH, DXGI_FORMAT_R8G8B8A8_UNORM,
        (g_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0) | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
    if (FAILED(hr)) {
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
            DebugLog(L"RenderThread: Device removed/reset on ResizeBuffers. HR: " + HResultToHexWString(hr));
            HandleDeviceRemovedAndReinit();
            return; // Recovery was handled, exit this function
        }
        DebugLog(L"RenderThread: ResizeBuffers failed. HR: " + HResultToHexWString(hr));
        // As a fallback, try automatic size:
        hr = g_swapChain->ResizeBuffers(kSwapChainBufferCount, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM,
                                        (g_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0) | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        if (FAILED(hr)) {
            DebugLog(L"RenderThread: ResizeBuffers(0,0) also failed. HR: " + HResultToHexWString(hr));
            return;
        }
    }

    g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(g_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        hr = g_swapChain->GetBuffer(i, IID_PPV_ARGS(&g_renderTargets[i]));
        if (FAILED(hr)) {
            DebugLog(L"RenderThread: GetBuffer(" + std::to_wstring(i) + L") failed. HR: " + HResultToHexWString(hr));
            return;
        }
        g_d3d12Device->CreateRenderTargetView(g_renderTargets[i].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, g_rtvDescriptorSize);
    }
    DebugLog(L"RenderThread: RTVs recreated after resize.");

    // After ResizeBuffers, any old fence targets are invalid for the new back buffers.
    // This reset prevents waiting on never-signaled values.
    const UINT64 completed = g_fence->GetCompletedValue();
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        g_renderFenceValues[i] = completed + 1;
    }
    // Refresh the current index, as it might have changed.
    g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();

    // Ensure we present at least one frame post-resize to refresh the screen.
    g_forcePresentOnce.store(true, std::memory_order_release);
}

void RenderFrame() {
    // Frame latency wait scope
    {
        bool hasReadyFrame = false;
        bool hasReorderFrame = false;

        // Check ready-queue (GPU-ready frames)
        {
            std::lock_guard<std::mutex> qlock(g_readyGpuFrameQueueMutex);
            hasReadyFrame = !g_readyGpuFrameQueue.empty();
        }

        // Check reorder buffer
        {
            std::lock_guard<std::mutex> rlock(g_reorderMutex);
            hasReorderFrame = !g_reorderBuffer.empty();
        }

        // If we already have something to render, skip the blocking wait.
        // Pump messages briefly so UI remains responsive, then continue.
        if (hasReadyFrame || hasReorderFrame) {
            // Non-blocking message pump (preserve layout & surrounding comments)
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    PostQuitMessage((int)msg.wParam);
                    return; // keep existing behavior: let the main loop handle exit
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            // Skip the blocking wait; proceed with the rest of RenderFrame()
        } else {
            if (g_frameLatencyWaitableObject) {
                // Wait on frame-latency object but remain responsive to window messages.
                HANDLE handles[1] = { g_frameLatencyWaitableObject };
                for (;;) {
                    DWORD r = MsgWaitForMultipleObjectsEx(
                        1,
                        handles,
                        100, // タイムアウトを100msに設定してフリーズを回避
                        QS_ALLINPUT,
                        MWMO_INPUTAVAILABLE | MWMO_ALERTABLE
                    );
                    if (r == WAIT_OBJECT_0) {
                        // frame-latency object signaled -> proceed to next frame
                        break;
                    } else if (r == WAIT_OBJECT_0 + 1) {
                        // pump messages; preserve layout and existing logging if present
                        MSG msg;
                        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                            if (msg.message == WM_QUIT) {
                                PostQuitMessage((int)msg.wParam);
                                return; // メインループに終了を任せるため、即時復帰
                            }
                            TranslateMessage(&msg);
                            DispatchMessage(&msg);
                        }
                        // loop and wait again
                        continue;
                    } else if (r == WAIT_TIMEOUT) {
                        // タイムアウトは想定内の動作。ループを抜けて処理を続ける
                        break;
                    }
                    else {
                        // Unexpected; if you already have logging, reuse it.
                        break;
                    }
                }
            }
        }
    }
    // Use existing fence and queue types; keep comments and layout intact.

    // Pick up pending resize, if any
    if (g_pendingResize.has.load(std::memory_order_acquire)) {
        int newW = g_pendingResize.w.load(std::memory_order_relaxed);
        int newH = g_pendingResize.h.load(std::memory_order_relaxed);
        g_pendingResize.has.store(false, std::memory_order_release);
        ResizeSwapChainOnRenderThread(newW, newH);
    }

    // --- D3D12 RenderFrame ---
    if (!g_commandList || !g_d3d12CommandQueue || !g_swapChain || !g_fence) {
        DebugLog(L"RenderFrame (D3D12): Core D3D12 objects not initialized.");
        return;
    }

    HRESULT hr; // for error checks
    ReadyGpuFrame renderedFrameData{}; // 描画したフレームのメタ（ログ用）
    const bool frameWasRendered = PopulateCommandList(renderedFrameData); // 今回のフレームで実描画コマンドを記録したか？

    // Command list is populated (at least with a clear). Now execute and present.
    ID3D12CommandList* ppCommandLists[] = { g_commandList.Get() };
    g_d3d12CommandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    const bool forcePresent = g_forcePresentOnce.exchange(false, std::memory_order_acq_rel);
    const bool shouldPresent = frameWasRendered || forcePresent;

    if (frameWasRendered) {
        // ---- Logging ----
        const auto log_render_start_ms = renderedFrameData.render_start_ms;
        const auto log_stream_frame_no = renderedFrameData.streamFrameNumber;
        const auto log_original_frame_no = renderedFrameData.originalFrameNumber;
        const auto log_id = renderedFrameData.id;
        const uint64_t render_end_ms = SteadyNowMs();
        const uint64_t render_total_ms =
            (render_end_ms >= log_render_start_ms) ? (render_end_ms - log_render_start_ms) : 0;

        uint64_t client_e2e_ms = 0;
        {
            std::lock_guard<std::mutex> lk(g_fecEndTimeMutex);
            auto it = g_fecEndTimeByStreamFrame.find(log_stream_frame_no);
            if (it != g_fecEndTimeByStreamFrame.end()) {
                const uint64_t fec_end_ms = it->second;
                client_e2e_ms = (render_end_ms >= fec_end_ms) ? (render_end_ms - fec_end_ms) : 0;
                g_fecEndTimeByStreamFrame.erase(it);
            }
        }
        if (RenderCount++ % 60 == 0) {
            DebugLog(L"RenderFrame Total (wall): " + std::to_wstring(render_total_ms) + L" ms.");
            DebugLog(L"Client FEC End->RenderEnd: " + std::to_wstring(client_e2e_ms) + L" ms.");
        }

        // --- ここから WGC→RenderEnd 計測と出力条件 ---
        int64_t wgc_to_renderend_ms = 0;
        {
            uint64_t wgc_ts_ms = 0;
            {
                std::lock_guard<std::mutex> lk(g_wgcTsMutex);
                auto it = g_wgcCaptureTimestampByStreamFrame.find(log_stream_frame_no);
                if (it != g_wgcCaptureTimestampByStreamFrame.end()) {
                    wgc_ts_ms = it->second;
                    g_wgcCaptureTimestampByStreamFrame.erase(it);
                }
            }
            if (wgc_ts_ms != 0) {
                auto frameEndSys = std::chrono::system_clock::now();
                uint64_t frameEndMs =
                    std::chrono::duration_cast<std::chrono::milliseconds>(frameEndSys.time_since_epoch()).count();
                wgc_to_renderend_ms = static_cast<int64_t>(frameEndMs) - static_cast<int64_t>(wgc_ts_ms) + (g_TimeOffsetNs.load() / 1000000);
            }
        }

        // [追加] 直近に WGC→RenderEnd をログ出力した StreamFrameNo を覚えて、同じなら再出力しない
        static uint32_t s_lastLoggedStreamFrame = 0;
        static bool     s_hasLastLoggedStreamFrame = false;
        const bool isDuplicateFrameLog =
            s_hasLastLoggedStreamFrame && (log_stream_frame_no == s_lastLoggedStreamFrame);

        // 既存の 60 フレーム毎の条件は維持しつつ、(1) 同一フレームの再レンダリング、(2) 差分が 0 以下 を抑制
        if ((RenderCount % 1) == 0) {
            if (!isDuplicateFrameLog && wgc_to_renderend_ms > 0) {
                DebugLog(L"RenderFrame Latency (unsynced clocks): StreamFrame #"
                    + std::to_wstring(log_stream_frame_no)
                    + L", OriginalFrame #" + std::to_wstring(log_original_frame_no)
                    + L" (ID: " + std::to_wstring(log_id) + L")"
                    + L" - WGC to RenderEnd: " + std::to_wstring(wgc_to_renderend_ms) + L" ms.");
                // 次回以降、同じフレームでの重複出力を抑制
                s_lastLoggedStreamFrame = log_stream_frame_no;
                s_hasLastLoggedStreamFrame = true;
            }
        }

    }

    //常にPresentを呼び出して、ウィンドウが描画され続けるようにする
    // Signal end-of-frame on the render fence and remember the value for this backbuffer
    const UINT64 fenceValue = ++g_fenceValue;
    g_d3d12CommandQueue->Signal(g_fence.Get(), fenceValue);
    g_renderFenceValues[g_currentFrameBufferIndex] = fenceValue;

    // Present
    HRESULT hrPresent;
    hrPresent = g_swapChain->Present(0, g_allowTearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
    if (FAILED(hrPresent)) {
        if (hrPresent == DXGI_ERROR_DEVICE_REMOVED || hrPresent == DXGI_ERROR_DEVICE_RESET) {
            DebugLog(L"RenderFrame (D3D12): Device removed/reset on Present. HR: " + HResultToHexWString(hrPresent));
            HandleDeviceRemovedAndReinit();
            return; // let the next iteration render with the new device
        } else {
            DebugLog(L"RenderFrame (D3D12): Present failed. HR: " + HResultToHexWString(hrPresent));
        }
    }

    // We are about to use the *next* backbuffer index; ensure its previous work is done
    UINT nextIndex = g_swapChain->GetCurrentBackBufferIndex();
    const UINT64 valueToWait = g_renderFenceValues[nextIndex];
    if (valueToWait != 0 && g_fence->GetCompletedValue() < valueToWait) {
        g_fence->SetEventOnCompletion(valueToWait, g_fenceEvent);
        while (true) {
            DWORD wr = MsgWaitForMultipleObjects(1, &g_fenceEvent, FALSE, INFINITE, QS_ALLINPUT);
            if (wr == WAIT_OBJECT_0) break; // fence signaled
            if (wr == WAIT_OBJECT_0 + 1) {
                MSG msg;
                while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            } else {
                break; // unexpected result; fail open
            }
        }
    }
    g_currentFrameBufferIndex = nextIndex;
}

// Minimal blocking wait, only for shutdown.
void WaitForGpu() {
    if (!g_d3d12CommandQueue || !g_fence || !g_fenceEvent) return;
    // Signal the command queue.
    UINT64 fenceValueToSignal = g_fenceValue;
    HRESULT hr = g_d3d12CommandQueue->Signal(g_fence.Get(), fenceValueToSignal);
    if (FAILED(hr)) { return; }

    // Wait until the fence has been processed.
    if (g_fence->GetCompletedValue() < fenceValueToSignal) {
        hr = g_fence->SetEventOnCompletion(fenceValueToSignal, g_fenceEvent);
        if (FAILED(hr)) { return; }

        while (true) {
            DWORD waitResult = MsgWaitForMultipleObjects(1, &g_fenceEvent, FALSE, INFINITE, QS_ALLINPUT);
            if (waitResult == WAIT_OBJECT_0) {
                // The fence was signaled.
                break;
            }
            if (waitResult == WAIT_OBJECT_0 + 1) {
                // A message is available. Pump the message queue.
                MSG msg;
                while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            } else {
                // An error or unexpected result occurred.
                break;
            }
        }
    }
    g_fenceValue++;
}


void CleanupD3DRenderResources() {
    WaitForGpu(); // Ensure GPU is idle before releasing resources

    CleanupOverlayResources();

    if (g_cropCB) {
        if (g_cropCBMapped) {
            g_cropCB->Unmap(0, nullptr);
            g_cropCBMapped = nullptr;
        }
        g_cropCB.Reset();
    }

    if (g_fenceEvent) {
        CloseHandle(g_fenceEvent);
        g_fenceEvent = nullptr;
    }
// [NEW] Close shared fence handle
if (g_copyFenceSharedHandle) { CloseHandle(g_copyFenceSharedHandle); g_copyFenceSharedHandle = nullptr; }
g_copyFence.Reset();
    if (g_fence) g_fence.Reset();
    if (g_commandList) g_commandList.Reset();
    for (UINT i = 0; i < kSwapChainBufferCount; ++i) {
        if (g_commandAllocator[i]) g_commandAllocator[i].Reset();
    }
    for (UINT i = 0; i < 2; ++i) {
        if (g_renderTargets[i]) g_renderTargets[i].Reset();
    }
    if (g_rtvHeap) g_rtvHeap.Reset();
    if (g_swapChain) g_swapChain.Reset();
    if (g_d3d12CommandQueue) g_d3d12CommandQueue.Reset();
    if (g_srvHeap) g_srvHeap.Reset();
    if (g_vertexBuffer) g_vertexBuffer.Reset();
    if (g_pipelineStateYuv444) g_pipelineStateYuv444.Reset();
    if (g_rootSignature) g_rootSignature.Reset();
    if (g_d3d12Device) g_d3d12Device.Reset();

    DebugLog(L"CleanupD3DRenderResources (D3D12): D3D12 resources cleaned up.");

    // Clean up resources
}


void SendFinalResolution(int width, int height) {
    DebugLog(L"SendFinalResolution: Sending resolution " + std::to_wstring(width) + L"x" + std::to_wstring(height));

    // サイズ変更メッセージを作成 (this is the logic from the old SendWindowSize)
    std::string message = std::to_string(width) + ":" + std::to_string(height) + "#";
    std::vector<uint8_t> data(message.begin(), message.end());

    size_t original_data_len = data.size();

    size_t shard_len = 0; // シャードの長さ
    bool fec_success = false;
    std::vector<std::vector<uint8_t>> dataShards;
    std::vector<std::vector<uint8_t>> parityShards;

    if (original_data_len > 0) { // データが存在する場合は FEC を適用
        fec_success = EncodeFEC_ISAL(
            data.data(),
            original_data_len,
            dataShards,
            parityShards,
            shard_len,
            RS_K,
            RS_M
        );
        if (!fec_success) {
            DebugLog(L"SendFinalResolution: EncodeFEC_ISAL failed.");
        }
    } else {
        DebugLog(L"SendFinalResolution: original_data_len is 0. Skipping FEC.");
        fec_success = false;
    }

    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        DebugLog(L"SendFinalResolution: Failed to create socket.");
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SEND_PORT_FEC);
    inet_pton(AF_INET, SEND_IP_FEC, &serverAddr.sin_addr);

    if (fec_success && original_data_len > 0) {
        uint32_t currentFrameNumber = g_globalFrameNumber.fetch_add(1);
        const size_t shardHeaderSize = sizeof(ShardInfoHeader);

        // データシャードの送信
        for (int i = 0; i < RS_K; ++i) {
            std::vector<uint8_t> packetData;
            packetData.reserve(shardHeaderSize + shard_len);

            ShardInfoHeader header;
            header.frameNumber = htonl(currentFrameNumber);
            header.shardIndex = htonl(i);
            header.totalDataShards = htonl(RS_K);
            header.totalParityShards = htonl(RS_M);
            header.originalDataLen = htonl(static_cast<uint32_t>(original_data_len));

            packetData.insert(packetData.end(), reinterpret_cast<uint8_t*>(&header), reinterpret_cast<uint8_t*>(&header) + shardHeaderSize);
            packetData.insert(packetData.end(), dataShards[i].begin(), dataShards[i].end());

            if (packetData.data() != nullptr && packetData.size() > 0 && packetData.size() <= SIZE_PACKET_SIZE) {
                if (sendto(udpSocket, reinterpret_cast<const char*>(packetData.data()), static_cast<int>(packetData.size()), 0, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
                    DebugLog(L"SendFinalResolution: Failed to send data shard: " + std::to_wstring(WSAGetLastError()));
                }
            } else {
                DebugLog(L"SendFinalResolution: Invalid packetData for data shard.");
            }
        }

        // パリティシャードの送信 (変更なし)
        for (int i = 0; i < RS_M; ++i) {
            std::vector<uint8_t> packetData;
            packetData.reserve(shardHeaderSize + shard_len);

            ShardInfoHeader header;
            header.frameNumber = htonl(currentFrameNumber);
            header.shardIndex = htonl(RS_K + i);
            header.totalDataShards = htonl(RS_K);
            header.totalParityShards = htonl(RS_M);
            header.originalDataLen = htonl(static_cast<uint32_t>(original_data_len));

            packetData.insert(packetData.end(), reinterpret_cast<uint8_t*>(&header), reinterpret_cast<uint8_t*>(&header) + shardHeaderSize);
            packetData.insert(packetData.end(), parityShards[i].begin(), parityShards[i].end());

            if (packetData.data() != nullptr && packetData.size() > 0 && packetData.size() <= SIZE_PACKET_SIZE) {
                if (sendto(udpSocket, reinterpret_cast<const char*>(packetData.data()), static_cast<int>(packetData.size()), 0, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
                    DebugLog(L"SendFinalResolution: Failed to send parity shard: " + std::to_wstring(WSAGetLastError()));
                }
            } else {
                DebugLog(L"SendFinalResolution: Invalid packetData for parity shard.");
            }
        }
    }
    closesocket(udpSocket);
}
