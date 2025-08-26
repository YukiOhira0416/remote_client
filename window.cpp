#define WIN32_LEAN_AND_MEAN // winsock.h の含まれる量を減らす
#define NOMINMAX
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
#include "main.h" // For RequestIDRNow

// ==== [Multi-monitor helpers - BEGIN] ====
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <ShellScalingApi.h> // AdjustWindowRectExForDpi 等
#pragma comment(lib, "Shcore.lib")

// Use a single monotonic clock for all latency metrics.
static inline uint64_t SteadyNowMs() noexcept {
    using clock = std::chrono::steady_clock;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            clock::now().time_since_epoch()
        ).count()
    );
}

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
                                      int desiredClientWidth, int desiredClientHeight) {
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

    int x = mi.rcWork.left + ((mi.rcWork.right  - mi.rcWork.left) - winW) / 2;
    int y = mi.rcWork.top  + ((mi.rcWork.bottom - mi.rcWork.top) - winH) / 2;

    g_hWnd = CreateWindowExW(dwExStyle, L"MyWindowClass", L"Remote Desktop Viewer",
                             dwStyle, x, y, winW, winH,
                             nullptr, nullptr, hInstance, nullptr);
    if (!g_hWnd) {
        DebugLog(L"InitWindow: CreateWindowExW failed. Error: " + std::to_wstring(GetLastError()));
        return false;
    }

    ShowWindow(g_hWnd, nCmdShow);
    UpdateWindow(g_hWnd);
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
Microsoft::WRL::ComPtr<ID3D12Device> g_d3d12Device;
Microsoft::WRL::ComPtr<ID3D12CommandQueue> g_d3d12CommandQueue;
Microsoft::WRL::ComPtr<IDXGISwapChain3> g_swapChain; // Use IDXGISwapChain3 or 4
Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> g_rtvHeap;
Microsoft::WRL::ComPtr<ID3D12Resource> g_renderTargets[2]; // Double buffering
Microsoft::WRL::ComPtr<ID3D12CommandAllocator> g_commandAllocator;
Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> g_commandList;
Microsoft::WRL::ComPtr<ID3D12Fence> g_fence;
UINT g_rtvDescriptorSize;
UINT g_currentFrameBufferIndex; // Current back buffer index
UINT64 g_fenceValue;
HANDLE g_fenceEvent;
UINT64 g_renderFenceValues[2]; // Fence values for each frame in flight for rendering

// D3D11 specific globals (to be removed or replaced)
#define CLIENT_PORT_FEC 8080// FEC用ポート番号
#define CLIENT_IP_FEC "127.0.0.1"// FEC用IPアドレス

bool g_allowTearing = false; // ティアリングを許可するかどうか

// ==== [Resize helpers - BEGIN] ====

// --- Window client padding around the video (in pixels) ---
static constexpr int kClientPaddingX = 16;
static constexpr int kClientPaddingY = 16;

// 「既知解像度」テーブルを共通化
struct TargetResolution { int width; int height; };
// Dense 16:9 ladder from 8K to 360p + practical intermediate widths.
static const TargetResolution kKnownResolutions[] = {
    {3840, 2160}, // 4K UHD
    {3456, 1944}, // 1.8x 1080p
    {3200, 1800}, // "3.2K"
    {3072, 1728}, // 1.6x 1080p

    // QHD family and dense steps down to FHD
    {2880, 1620},
    {2715, 1528}, // uncommon but keeps the ladder dense
    {2560, 1440}, // QHD
    {2432, 1368},
    {2304, 1296},
    {2240, 1260},
    {2176, 1224},
    {2048, 1152}, // 2K-ish 16:9

    // FHD and dense steps
    {1984, 1116},
    {1920, 1080}, // FHD
    {1856, 1044},
    {1792, 1008},
    {1728, 972},
    {1680, 945},
    {1600, 900},  // HD+
    {1536, 864},
    {1504, 846},
    {1440, 810},
    {1408, 792},
    {1366, 768},  // very common laptop (approx 16:9)
    {1360, 765},  // defensive step near 1366×768
    {1344, 756},
    {1280, 720},  // HD
    {1216, 684},
    {1152, 648},
    {1120, 630},
    {1088, 612},
    {1056, 594},
    {1024, 576},  // 576p (16:9)
    {992,  558},
    {960,  540},  // qHD
    {928,  522},
    {896,  504},
    {864,  486},
    {832,  468},
    {800,  450},
    {768,  432},  // 432p
    {736,  414},
    {704,  396},
    {672,  378},
    {640,  360}   // 360p
};

// 既知解像度にスナップするユーティリティ
void SnapToKnownResolution(int srcW, int srcH, int& outW, int& outH) {
    int bestW = srcW, bestH = srcH, best = INT_MAX;
    for (auto& r : kKnownResolutions) {
        int d = abs(srcW - r.width) + abs(srcH - r.height);
        if (d < best) { best = d; bestW = r.width; bestH = r.height; }
    }
    outW = bestW; outH = bestH;
}

// サーバーへ最終解像度を送る（旧 SendWindowSize 代替）
void SendFinalResolution(int width, int height); // Forward declaration (removed static for external linkage)
void WaitForGpu(); // D3D12: Helper function to wait for GPU to finish commands

// ==== [Resize helpers - END] ====

// from main.cpp
extern void OnResolutionChanged_GatedSend(int w, int h, bool forceResendNow);
extern std::chrono::high_resolution_clock::time_point g_lastFrameRenderTimeForKick;

// 送信フレーム番号で並べる小さなバッファ
static std::map<uint32_t, ReadyGpuFrame> g_reorderBuffer;
static std::mutex g_reorderMutex;

// ==== [In-flight Frame Management - BEGIN] ====
// GPUがまだ使用中の可能性のあるフレームのリソースを管理するための構造体
struct InFlightFrame {
    ReadyGpuFrame frameData;
    UINT64 fenceValue;
};
// GPUへ投入済みのフレームを保持するキュー
static std::queue<InFlightFrame> g_inFlightFrames;
static std::mutex g_inFlightFramesMutex;
// ==== [In-flight Frame Management - END] ====

static uint32_t g_expectedStreamFrame = 0;
static bool     g_expectedInitialized = false;

// 描画側の「期待フレーム番号」やバッファをクリアして“待ち”を防ぐ
void ClearReorderState()
{
    std::lock_guard<std::mutex> lk(g_reorderMutex);
    g_reorderBuffer.clear();
    g_expectedInitialized = false;
    g_expectedStreamFrame = 0;
    DebugLog(L"ClearReorderState: reorder state cleared.");
}

// 調整パラメータ
static constexpr size_t REORDER_MAX_BUFFER = 8; // これを超えたら妥協して前進
static constexpr int    REORDER_WAIT_MS    = 1; // N+1 を待つ最大時間（ms） (from 3ms)
static std::chrono::steady_clock::time_point g_lastReorderDecision = std::chrono::steady_clock::now();

// Rendering specific D3D12 globals
Microsoft::WRL::ComPtr<ID3D12RootSignature> g_rootSignature;
Microsoft::WRL::ComPtr<ID3D12PipelineState> g_pipelineState;
Microsoft::WRL::ComPtr<ID3D12Resource> g_vertexBuffer;
D3D12_VERTEX_BUFFER_VIEW g_vertexBufferView;
Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> g_srvHeap; // For Y and UV textures
UINT g_srvDescriptorSize;

std::atomic<uint32_t> g_globalFrameNumber(0);// FEC用フレーム番号

// 各シャードパケットに付与するヘッダー
struct ShardInfoHeader {
    uint32_t frameNumber;       // フレーム番号 (ネットワークバイトオーダー)
    uint32_t shardIndex;        // シャードインデックス (0..k-1: データシャード, k..n-1: パリティシャード) (リトルエンディアン)
    uint32_t totalDataShards;   // データシャードの総数 (k) (リトルエンディアン)
    uint32_t totalParityShards; // パリティシャードの総数 (m) (リトルエンディアン)
    uint32_t originalDataLen;   // 元のH.264データの長さ (パケットサイズ) (リトルエンディアン)
};

extern uint64_t g_lastRenderedRgbaFrameId; // To track the last rendered frameB

struct VertexPosTex { float x, y, z; float u, v; };



void CleanupD3DRenderResources(); // Forward declaration

// Helper to handle the logic for snapping, padding, and notifying after a resize event.
static void FinalizeResize(HWND hWnd, bool forceAnnounce = false)
{
    RECT rc{}; GetClientRect(hWnd, &rc);
    int cw = rc.right - rc.left, ch = rc.bottom - rc.top;

    int tw = 0, th = 0;
    SnapToKnownResolution(cw, ch, tw, th); // tw,th = snapped *video* size (16:9)

    const int paddedClientW = tw + kClientPaddingX * 2;
    const int paddedClientH = th + kClientPaddingY * 2;

    // If the snapped resolution and padded size are already correct, do nothing,
    // UNLESS a force announce is requested (e.g., after a monitor move).
    if (currentResolutionWidth.load() == tw && currentResolutionHeight.load() == th &&
        cw == paddedClientW && ch == paddedClientH) {
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

    // Adjust *outer* window so that client becomes padded size
    RECT wr{0, 0, paddedClientW, paddedClientH};
    DWORD style = GetWindowLong(hWnd, GWL_STYLE);
    DWORD ex    = GetWindowLong(hWnd, GWL_EXSTYLE);
    AdjustWindowRectEx(&wr, style, GetMenu(hWnd)!=NULL, ex);
    const int ww = wr.right - wr.left, wh = wr.bottom - wr.top;

    RECT wrNow{}; GetWindowRect(hWnd, &wrNow);
    if ((wrNow.right - wrNow.left) != ww || (wrNow.bottom - wrNow.top) != wh) {
        SetWindowPos(hWnd, nullptr, 0, 0, ww, wh,
                     SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    }

    // Enqueue swap-chain resize to *padded client size*.
    // This might be redundant if SetWindowPos triggers a WM_SIZE that is handled,
    // but it's important to ensure the resize is correctly queued.
    g_pendingResize.w.store(paddedClientW, std::memory_order_relaxed);
    g_pendingResize.h.store(paddedClientH, std::memory_order_relaxed);
    g_pendingResize.has.store(true, std::memory_order_release);

    // Notify server (single gate) with the *video* resolution ONLY
    OnResolutionChanged_GatedSend(tw, th, /*force=*/forceAnnounce);

    InvalidateRect(hWnd, nullptr, FALSE);
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
            ClearReorderState();
            RequestIDRNow();

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

            // Do not queue resizes while the user is actively dragging the window.
            // This prevents a storm of resize requests. The final resize is handled
            // by the logic in WM_EXITSIZEMOVE.
            if (g_isSizing) {
                return 0;
            }

            if (wParam == SIZE_MINIMIZED || width == 0 || height == 0) {
                DebugLog(L"WM_SIZE: minimized or zero. skip.");
                return 0;
            }

            // This will now only run for programmatic resizes or the final resize
            // triggered after a drag operation.
            g_pendingResize.w.store(width, std::memory_order_relaxed);
            g_pendingResize.h.store(height, std::memory_order_relaxed);
            g_pendingResize.has.store(true, std::memory_order_release);

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

bool InitWindow(HINSTANCE hInstance, int nCmdShow) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc   = WndProc;
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.hInstance     = hInstance;
    wc.lpszClassName = L"MyWindowClass";
    wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    if (!RegisterClassW(&wc)) {
        DebugLog(L"InitWindow: Failed to register window class. Error: " + std::to_wstring(GetLastError()));
        return false;
    }

    const int initialWidth = 1920;
    const int initialHeight = 1080;
    if (!CreateWindowOnBestMonitor(hInstance, nCmdShow, initialWidth, initialHeight)) {
        return false;
    }

    // クライアント実寸から既知解像度へスナップして一回送信
    RECT rc{}; GetClientRect(g_hWnd, &rc);
    int cw = rc.right - rc.left, ch = rc.bottom - rc.top;
    int tw, th;
    SnapToKnownResolution(cw, ch, tw, th);

    currentResolutionWidth = tw;
    currentResolutionHeight = th;

    // Make the *client area* slightly larger than the video to leave margins
    const int paddedClientW = tw + kClientPaddingX * 2;
    const int paddedClientH = th + kClientPaddingY * 2;

    // If the initial client size differs from the snapped+padded size, adjust the window.
    if (cw != paddedClientW || ch != paddedClientH) {
        RECT wr = {0, 0, paddedClientW, paddedClientH};
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
    OnResolutionChanged_GatedSend(tw, th, false);
    return true;
}

bool InitD3D() {
        // Release existing D3D12 resources if re-initializing
    if (g_fence) g_fence.Reset();
    if (g_commandList) g_commandList.Reset();
    if (g_commandAllocator) g_commandAllocator.Reset();
    for (UINT i = 0; i < 2; ++i) {
        if (g_renderTargets[i]) g_renderTargets[i].Reset();
    }
    if (g_rtvHeap) g_rtvHeap.Reset();
    if (g_swapChain) g_swapChain.Reset();
    
    if (g_d3d12CommandQueue) g_d3d12CommandQueue.Reset();
    // Release rendering specific resources
    if (g_srvHeap) g_srvHeap.Reset();
    if (g_vertexBuffer) g_vertexBuffer.Reset();
    if (g_pipelineState) g_pipelineState.Reset();
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
    scd1.BufferCount = 2; // Double buffering
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
    scd1.Flags = g_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

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
    g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();
    DebugLog(L"InitD3D (D3D12): Swap Chain created.");

    // 4. Create RTV Descriptor Heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 2; // For double buffering
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
    for (UINT i = 0; i < 2; ++i) {
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
    hr = g_d3d12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_commandAllocator));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create command allocator. HRESULT: " + HResultToHexWString(hr));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): Command Allocator created.");

    // 7. Create Command List
    hr = g_d3d12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_commandAllocator.Get(), nullptr, IID_PPV_ARGS(&g_commandList));
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
    g_fenceValue = 1;
    g_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (g_fenceEvent == nullptr) {
        DebugLog(L"InitD3D (D3D12): Failed to create fence event. Error: " + std::to_wstring(GetLastError()));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): Fence and Event created.");

    // --- Create rendering resources (Root Signature, PSO, Vertex Buffer, SRV Heap) ---

    // 9. Create Root Signature
    D3D12_DESCRIPTOR_RANGE1 srvRanges[2];
    srvRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[0].NumDescriptors = 1; // For Y Texture
    srvRanges[0].BaseShaderRegister = 0; // t0
    srvRanges[0].RegisterSpace = 0;
    srvRanges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    srvRanges[0].OffsetInDescriptorsFromTableStart = 0;

    srvRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRanges[1].NumDescriptors = 1; // For UV Texture
    srvRanges[1].BaseShaderRegister = 1; // t1
    srvRanges[1].RegisterSpace = 0;
    srvRanges[1].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC;
    srvRanges[1].OffsetInDescriptorsFromTableStart = 1;

    D3D12_ROOT_PARAMETER1 rootParameters[1];
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    rootParameters[0].DescriptorTable.NumDescriptorRanges = _countof(srvRanges);
    rootParameters[0].DescriptorTable.pDescriptorRanges = srvRanges;

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
    Microsoft::WRL::ComPtr<ID3DBlob> pixelShaderBlob;

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
    hr = D3DCompileFromFile(L"Shader/NV12ToRGBPS.hlsl", nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE, "main", "ps_5_1", compileFlags, 0, &pixelShaderBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) DebugLog(L"InitD3D (D3D12): Pixel shader compilation failed: " + std::wstring(static_cast<wchar_t*>(errorBlob->GetBufferPointer()), static_cast<wchar_t*>(errorBlob->GetBufferPointer()) + errorBlob->GetBufferSize() / sizeof(wchar_t)));
        else DebugLog(L"InitD3D (D3D12): Pixel shader compilation failed. HR: " + HResultToHexWString(hr));
        return false;
    }

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = g_rootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShaderBlob.Get());
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShaderBlob.Get());
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

    hr = g_d3d12Device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&g_pipelineState));
    if (FAILED(hr)) {
        DebugLog(L"InitD3D (D3D12): Failed to create PSO. HR: " + HResultToHexWString(hr));
        return false;
    }
    DebugLog(L"InitD3D (D3D12): PSO created.");

    // 11. Create SRV Descriptor Heap (for Y and UV textures)
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = 2 * 2; // 2 textures (Y, UV) per frame buffer for double buffering (or more if needed)
                                        // For simplicity, let's assume we update descriptors for the current frame's textures.
                                        // So, 2 descriptors are enough if we update them each frame.
                                        // If we pre-create for all decoder surfaces, it'd be NUM_DECODE_SURFACES_IN_POOL * 2.
                                        // Let's start with 2, for the current frame's Y and UV.
    srvHeapDesc.NumDescriptors = 2; // One for Y, one for UV of the current frame to render
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

    return true;
}

UINT64 PopulateCommandListCount = 0;
bool PopulateCommandList(ReadyGpuFrame& outFrameToRender) { // Return bool, pass ReadyGpuFrame by reference
    // Reset command allocator and command list
    HRESULT hr = g_commandAllocator->Reset();
    if (FAILED(hr)) { DebugLog(L"PopulateCommandList: Failed to reset command allocator. HR: " + HResultToHexWString(hr)); return false; }
    hr = g_commandList->Reset(g_commandAllocator.Get(), nullptr); // No initial PSO for clear
    if (FAILED(hr)) { DebugLog(L"PopulateCommandList: Failed to reset command list. HR: " + HResultToHexWString(hr)); return false; }

    // Transition the current back buffer from PRESENT to RENDER_TARGET.
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = g_renderTargets[g_currentFrameBufferIndex].Get();
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    g_commandList->ResourceBarrier(1, &barrier);

    // Set RTV
    // Set PSO and Root Signature
    g_commandList->SetPipelineState(g_pipelineState.Get());
    g_commandList->SetGraphicsRootSignature(g_rootSignature.Get());

    // Set RTV
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(g_rtvHeap->GetCPUDescriptorHandleForHeapStart(), g_currentFrameBufferIndex, g_rtvDescriptorSize);
    g_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    // >>> NEW: Ensure viewport/scissor initially cover the full backbuffer <<<
    if (g_renderTargets[g_currentFrameBufferIndex]) {
        SetViewportScissorToBackbuffer(g_commandList.Get(), g_renderTargets[g_currentFrameBufferIndex].Get());
    }

    // Optional: clear to black so gutters are black (no blue flicker around the content)
    const float clearColor[4] = {0.f, 0.f, 0.f, 1.f};
    g_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

    // --- Render the decoded frame ---
    bool hasFrameToRender = false;

    // 1) NVDEC からの準備完了キューを全部吸い込み、番号で並べる
    {
        std::unique_lock<std::mutex> qlock(g_readyGpuFrameQueueMutex);
        while (!g_readyGpuFrameQueue.empty()) {
            ReadyGpuFrame f = std::move(g_readyGpuFrameQueue.back());
            g_readyGpuFrameQueue.pop_back();

            std::lock_guard<std::mutex> rlock(g_reorderMutex);
            auto it = g_reorderBuffer.find(f.streamFrameNumber);
            if (it == g_reorderBuffer.end()) {
                g_reorderBuffer.emplace(f.streamFrameNumber, std::move(f));
            } else {
                // 同じ番号が2枚来たら後着で上書き
                it->second = std::move(f);
                DebugLog(L"[REORDER] duplicate streamFrameNumber, replaced: " + std::to_wstring(it->first));
            }
        }
    }

    // 2) N と N+1 がそろったら N を描画。なければ短い待ち or 圧迫で妥協
    {
        std::lock_guard<std::mutex> rlock(g_reorderMutex);
        auto now = std::chrono::steady_clock::now();

        // 初期化：隣接ペアの最小キーを expected にする
        if (!g_expectedInitialized) {
            for (auto it = g_reorderBuffer.begin(); it != g_reorderBuffer.end(); ++it) {
                uint32_t k = it->first;
                if (g_reorderBuffer.find(k + 1) != g_reorderBuffer.end()) {
                    g_expectedStreamFrame = k;
                    g_expectedInitialized = true;
                    break;
                }
            }
            if (!g_expectedInitialized && !g_reorderBuffer.empty()) {
                g_expectedStreamFrame = g_reorderBuffer.begin()->first;
                g_expectedInitialized = true;
            }
        }

        const bool haveExpected = g_reorderBuffer.count(g_expectedStreamFrame) != 0;
        const bool haveNext     = g_reorderBuffer.count(g_expectedStreamFrame + 1) != 0;
        const bool bufferTooBig = g_reorderBuffer.size() > REORDER_MAX_BUFFER;
        const bool waitedLong   = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_lastReorderDecision).count() > REORDER_WAIT_MS;

        if (haveExpected && haveNext) {
            // 理想：N と N+1 がそろった
            outFrameToRender = std::move(g_reorderBuffer[g_expectedStreamFrame]);
            outFrameToRender.render_start_ms = SteadyNowMs(); // steady clock start
            g_reorderBuffer.erase(g_expectedStreamFrame);
            g_expectedStreamFrame++;
            hasFrameToRender = true;
            g_lastReorderDecision = now;
        } else if (bufferTooBig || waitedLong) {
            // 妥協：溜まりすぎor待ちすぎ → 最小キーを描画
            if (!g_reorderBuffer.empty()) {
                auto it = g_reorderBuffer.begin();
                outFrameToRender = std::move(it->second);
                outFrameToRender.render_start_ms = SteadyNowMs(); // steady clock start
                uint32_t drawn = it->first;
                g_reorderBuffer.erase(it);
                g_expectedStreamFrame = drawn + 1;
                hasFrameToRender = true;
                g_lastReorderDecision = now;

                if (!(haveExpected && haveNext)) {
                    if(PopulateCommandListCount++ % 60 == 0)DebugLog(L"[REORDER] fallback draw, key=" + std::to_wstring(drawn));
                }
            }
        }
    }

    if (hasFrameToRender && outFrameToRender.hw_decoded_texture_Y && outFrameToRender.hw_decoded_texture_UV) {
        // Use the *video* size for centering (what the server encodes)
        const int videoW = currentResolutionWidth.load();
        const int videoH = currentResolutionHeight.load();

        // Center the draw region inside the (possibly larger) backbuffer
        ID3D12Resource* backbuffer = g_renderTargets[g_currentFrameBufferIndex].Get();
        if (backbuffer) {
            const D3D12_RESOURCE_DESC bbDesc = backbuffer->GetDesc();
            SetLetterboxViewport(g_commandList.Get(), bbDesc, videoW, videoH);
        }

        // Create SRVs for Y and UV textures (only if resource changed)
        static ID3D12Resource* s_lastY = nullptr;
        static ID3D12Resource* s_lastUV = nullptr;

        CD3DX12_CPU_DESCRIPTOR_HANDLE srvHandleCpu(g_srvHeap->GetCPUDescriptorHandleForHeapStart());

        if (outFrameToRender.hw_decoded_texture_Y.Get() != s_lastY) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDescY = {};
            srvDescY.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDescY.Format = DXGI_FORMAT_R8_UNORM; // Y plane
            if (!outFrameToRender.hw_decoded_texture_Y) { DebugLog(L"PopulateCommandList: frameToRender.hw_decoded_texture_Y is NULL."); return false; }
            srvDescY.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDescY.Texture2D.MipLevels = 1;
            g_d3d12Device->CreateShaderResourceView(
                outFrameToRender.hw_decoded_texture_Y.Get(), &srvDescY, srvHandleCpu);
            s_lastY = outFrameToRender.hw_decoded_texture_Y.Get();
        }

        // Advance handle to UV and repeat only on change
        srvHandleCpu.Offset(1, g_srvDescriptorSize); // (match existing offset logic)

        if (outFrameToRender.hw_decoded_texture_UV.Get() != s_lastUV) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDescUV = {};
            srvDescUV.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srvDescUV.Format = DXGI_FORMAT_R8G8_UNORM; // UV plane (NV12)
            if (!outFrameToRender.hw_decoded_texture_UV) { DebugLog(L"PopulateCommandList: frameToRender.hw_decoded_texture_UV is NULL."); return false; }
            srvDescUV.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srvDescUV.Texture2D.MipLevels = 1;
            g_d3d12Device->CreateShaderResourceView(
                outFrameToRender.hw_decoded_texture_UV.Get(), &srvDescUV, srvHandleCpu);
            s_lastUV = outFrameToRender.hw_decoded_texture_UV.Get();
        }

        // Set descriptor heap for SRVs
        ID3D12DescriptorHeap* ppHeaps[] = { g_srvHeap.Get() };
        g_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
        g_commandList->SetGraphicsRootDescriptorTable(0, g_srvHeap->GetGPUDescriptorHandleForHeapStart());

        // フルスクリーンクアッド（TRIANGLESTRIP で 4 頂点）
        g_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        g_commandList->DrawInstanced(4, 1, 0, 0);

        // Release the ComPtrs now that they are submitted for rendering
        // The resources themselves are managed by the decoder's pool
        // frameToRender.hw_decoded_texture_Y.Reset(); // ComPtr will auto-release
        // frameToRender.hw_decoded_texture_UV.Reset();
    }

    // Transition the current back buffer from RENDER_TARGET to PRESENT.
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    g_commandList->ResourceBarrier(1, &barrier);

    hr = g_commandList->Close();
    if (FAILED(hr)) { DebugLog(L"PopulateCommandList: Failed to close command list. HR: " + HResultToHexWString(hr)); return false; }
    return hasFrameToRender; // Return whether a frame was actually prepared for rendering
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

    DebugLog(L"RenderThread: resizing swap-chain to " + std::to_wstring(newW) + L"x" + std::to_wstring(newH));

    WaitForGpuWithTimeout(500); // keep pumping; don’t freeze

    for (UINT i = 0; i < 2; ++i) g_renderTargets[i].Reset();

    HRESULT hr = g_swapChain->ResizeBuffers(
        2, newW, newH, DXGI_FORMAT_R8G8B8A8_UNORM,
        g_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    if (FAILED(hr)) {
        DebugLog(L"RenderThread: ResizeBuffers failed. HR: " + HResultToHexWString(hr));
        // As a fallback, try automatic size:
        hr = g_swapChain->ResizeBuffers(2, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM,
                                        g_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
        if (FAILED(hr)) {
            DebugLog(L"RenderThread: ResizeBuffers(0,0) also failed. HR: " + HResultToHexWString(hr));
            return;
        }
    }

    g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(g_rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < 2; ++i) {
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
    for (UINT i = 0; i < 2; ++i) {
        g_renderFenceValues[i] = completed + 1;
    }
    // Refresh the current index, as it might have changed.
    g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();

    // Ensure at least one post-resize Present to advance swap-chain even if no new frame is ready.
    g_forcePresentOnce.store(true, std::memory_order_release);
}

void RenderFrame() {
    // ---- [Release completed frame resources - BEGIN] ----
    // GPUがどこまで処理を終えたかを確認
    const UINT64 completedFenceValue = g_fence->GetCompletedValue();
    {
        std::lock_guard<std::mutex> lock(g_inFlightFramesMutex);
        // キューの先頭から、完了済みのフレームを解放
        while (!g_inFlightFrames.empty() && g_inFlightFrames.front().fenceValue <= completedFenceValue) {
            g_inFlightFrames.pop(); // Dequeue and destroy the frame object, releasing its resources.
        }
    }
    // ---- [Release completed frame resources - END] ----

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

    if (shouldPresent) {
        // Present (VSync=0, tearing depends on support)
        hr = g_swapChain->Present(0, g_allowTearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
                DebugLog(L"RenderFrame (D3D12): Device removed/reset on Present. HR: " + HResultToHexWString(hr));
                // TODO: Handle device loss (e.g., re-initialize)
            } else {
                DebugLog(L"RenderFrame (D3D12): Present failed. HR: " + HResultToHexWString(hr));
            }
        }

        if (frameWasRendered) {
            // ---- Standard path for a rendered frame ----
            // Fence: Track this frame's completion
            const UINT64 currentFenceVal = g_renderFenceValues[g_currentFrameBufferIndex];
            hr = g_d3d12CommandQueue->Signal(g_fence.Get(), currentFenceVal);
            if (FAILED(hr)) {
                DebugLog(L"RenderFrame: Failed to signal fence. HR: " + HResultToHexWString(hr));
            }

            // GPUに投入したフレームのリソースを、完了まで破棄しないようにキューへ移動
            {
                std::lock_guard<std::mutex> lock(g_inFlightFramesMutex);
                g_inFlightFrames.push({std::move(renderedFrameData), currentFenceVal});
            }

            // Update back buffer index for the next frame
            g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();

            // Wait only if we risk getting too far ahead of the GPU (bounded backlog)
            bool shouldWait = false;
            {
                std::lock_guard<std::mutex> lock(g_inFlightFramesMutex);
                // Keep at most 2 frames in flight for this back buffer index
                // (Adjust threshold carefully if you observe pipe underflow/overflow)
                shouldWait = (g_inFlightFrames.size() >= 2);
            }

            if (shouldWait) {
                if (g_fence->GetCompletedValue() < g_renderFenceValues[g_currentFrameBufferIndex]) {
                    hr = g_fence->SetEventOnCompletion(g_renderFenceValues[g_currentFrameBufferIndex], g_fenceEvent);
                    if (FAILED(hr)) {
                        DebugLog(L"RenderFrame: Failed to set event on completion. HR: " + HResultToHexWString(hr));
                    } else {
                        // Bounded wait keeps pipeline responsive; still measured in the existing logging
                        const DWORD timeoutMs = 5; // short, non-infinite
                        WaitForSingleObjectEx(g_fenceEvent, timeoutMs, FALSE);
                    }
                }
            }

            // Increment fence value for the next use of this RTV
            g_renderFenceValues[g_currentFrameBufferIndex] = currentFenceVal + 1;
        } else {
            // ---- Forced present path (no new frame rendered) ----
            // Don't perform fence waits or signals that assume a rendered frame.
            // Just refresh the back buffer index so the next real frame uses the correct RTV.
            g_currentFrameBufferIndex = g_swapChain->GetCurrentBackBufferIndex();

            // To be extra safe, align the fence value for this buffer index to what's known to be completed.
            // This prevents a future wait on a stale, never-to-be-signaled value.
            const UINT64 completed = g_fence->GetCompletedValue();
            g_renderFenceValues[g_currentFrameBufferIndex] = completed + 1;
        }
    }

    // ---- Logging (only if a new frame was actually rendered) ----
    if (frameWasRendered) {
        const uint64_t render_end_ms = SteadyNowMs();
        renderedFrameData.present_ms = render_end_ms;   // Present just finished (approx)
        renderedFrameData.fence_done_ms = render_end_ms; // after waits; same endpoint for now

        // Render totals (steady)
        const uint64_t render_total_ms = (render_end_ms >= renderedFrameData.render_start_ms)
            ? (render_end_ms - renderedFrameData.render_start_ms)
            : 0;

        // Client FEC End->RenderEnd (steady) using per-frame mapping
        uint64_t client_e2e_ms = 0;
        {
            std::lock_guard<std::mutex> lk(g_fecEndTimeMutex);
            auto it = g_fecEndTimeByStreamFrame.find(renderedFrameData.streamFrameNumber);
            if (it != g_fecEndTimeByStreamFrame.end()) {
                const uint64_t fec_end_ms = it->second;
                client_e2e_ms = (render_end_ms >= fec_end_ms) ? (render_end_ms - fec_end_ms) : 0;
                // cleanup: we used it for this frame
                g_fecEndTimeByStreamFrame.erase(it);
            }
        }

        // Finalize back-compat field for older log readers:
        renderedFrameData.client_fec_end_to_render_end_time_ms = client_e2e_ms;

        // Logging cadence preserved
        if (RenderCount++ % 60 == 0) {
            DebugLog(L"RenderFrame Total (wall): " + std::to_wstring(render_total_ms) + L" ms.");
            DebugLog(L"Client FEC End->RenderEnd: " + std::to_wstring(client_e2e_ms) + L" ms.");
            // NOTE: NVDEC End->RenderEnd log removed per spec
        }

        // Keep the existing cross-machine log, but compute from per-frame map (unsynced clocks).
        int64_t wgc_to_renderend_ms = 0;
        {
            // server WGC timestamp (ms since epoch, system_clock at server)
            uint64_t wgc_ts_ms = 0;
            {
                std::lock_guard<std::mutex> lk(g_wgcTsMutex);
                auto it = g_wgcCaptureTimestampByStreamFrame.find(renderedFrameData.streamFrameNumber);
                if (it != g_wgcCaptureTimestampByStreamFrame.end()) {
                    wgc_ts_ms = it->second;
                    g_wgcCaptureTimestampByStreamFrame.erase(it); // consume once
                }
            }
            if (wgc_ts_ms != 0) {
                auto frameEndSys   = std::chrono::system_clock::now();
                uint64_t frameEndMs = std::chrono::duration_cast<std::chrono::milliseconds>(frameEndSys.time_since_epoch()).count();
                wgc_to_renderend_ms = static_cast<int64_t>(frameEndMs) - static_cast<int64_t>(wgc_ts_ms);
            } else {
                // No timestamp found (e.g., rare drop/miss). Keep 0; still log as unsynced.
                wgc_to_renderend_ms = 0;
            }
        }

        if (RenderCount % 60 == 0) {
            DebugLog(L"RenderFrame Latency (unsynced clocks): StreamFrame #"
                + std::to_wstring(renderedFrameData.streamFrameNumber)
                + L", OriginalFrame #" + std::to_wstring(renderedFrameData.originalFrameNumber)
                + L" (ID: " + std::to_wstring(renderedFrameData.id) + L")"
                + L" - WGC to RenderEnd: " + std::to_wstring(wgc_to_renderend_ms) + L" ms.");
        }
    }
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
        WaitForSingleObject(g_fenceEvent, INFINITE);
    }
    g_fenceValue++;
}


void CleanupD3DRenderResources() {
    WaitForGpu(); // Ensure GPU is idle before releasing resources

    if (g_fenceEvent) {
        CloseHandle(g_fenceEvent);
        g_fenceEvent = nullptr;
    }
    if (g_fence) g_fence.Reset();
    if (g_commandList) g_commandList.Reset();
    if (g_commandAllocator) g_commandAllocator.Reset();
    for (UINT i = 0; i < 2; ++i) {
        if (g_renderTargets[i]) g_renderTargets[i].Reset();
    }
    if (g_rtvHeap) g_rtvHeap.Reset();
    if (g_swapChain) g_swapChain.Reset();
    if (g_d3d12CommandQueue) g_d3d12CommandQueue.Reset();
    if (g_srvHeap) g_srvHeap.Reset();
    if (g_vertexBuffer) g_vertexBuffer.Reset();
    if (g_pipelineState) g_pipelineState.Reset();
    if (g_rootSignature) g_rootSignature.Reset();
    if (g_d3d12Device) g_d3d12Device.Reset();

    DebugLog(L"CleanupD3DRenderResources (D3D12): D3D12 resources cleaned up.");

    // Clean up resources
}


void SendFinalResolution(int width, int height) {
    DebugLog(L"SendFinalResolution: Sending resolution " + std::to_wstring(width) + L"x" + std::to_wstring(height));

    // サイズ変更メッセージを作成 (this is the logic from the old SendWindowSize)
    std::string message = std::to_string(height) + ":" + std::to_string(width) + "#";
    std::vector<uint8_t> data(message.begin(), message.end());

    size_t original_data_len = data.size();

    size_t shard_len = 0; // シャードの長さ
    bool fec_success = false;
    std::vector<std::vector<uint8_t>> parityShards;
    if (original_data_len > 0) { // データが存在する場合は FEC を適用
        size_t min_shard_len = (original_data_len + RS_K - 1) / RS_K; // 最小シャード長を計算

        const int w = 8; // シャードの幅
        const int alignment = w * sizeof(long); // 8 * 8 = 64 (64bit 整数の場合)
        shard_len = (min_shard_len + alignment - 1) / alignment * alignment;

        size_t padded_data_len = shard_len * RS_K;
        if (original_data_len < padded_data_len) {
            data.resize(padded_data_len, 0); // 0 でパディング
        } else if (original_data_len > padded_data_len) {
            DebugLog(L"Warning: original_data_len > padded_data_len. This should not happen.");
            data.resize(padded_data_len);
        }

        if (g_matrix_initialized) {
            fec_success = EncodeFEC_Jerasure(
                data.data(),
                padded_data_len,
                parityShards,
                RS_K,
                RS_M,
                g_jerasure_matrix,
                shard_len
            );
            if (!fec_success) {
                DebugLog(L"SendFinalResolution: EncodeFEC_Jerasure failed.");
            }
        } else {
            DebugLog(L"SendFinalResolution: Jerasure matrix not initialized. Skipping FEC.");
            fec_success = false;
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
    serverAddr.sin_port = htons(CLIENT_PORT_FEC);
    inet_pton(AF_INET, CLIENT_IP_FEC, &serverAddr.sin_addr);

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
            const uint8_t* shardStart = data.data() + i * shard_len;
            packetData.insert(packetData.end(), shardStart, shardStart + shard_len);

            if (packetData.data() != nullptr && packetData.size() > 0 && packetData.size() <= SIZE_PACKET_SIZE) {
                if (sendto(udpSocket, reinterpret_cast<const char*>(packetData.data()), static_cast<int>(packetData.size()), 0, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR) {
                    DebugLog(L"SendFinalResolution: Failed to send data shard: " + std::to_wstring(WSAGetLastError()));
                }
            } else {
                DebugLog(L"SendFinalResolution: Invalid packetData for data shard.");
            }
        }

        // パリティシャードの送信
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
