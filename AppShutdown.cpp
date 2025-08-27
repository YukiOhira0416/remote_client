#include "AppShutdown.h"
#include "DebugLog.h"
#include "Globals.h"
#include "nvdec.h"
#include "window.h"

#include <Windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <enet/enet.h>
#include <cuda.h>
#include <atomic>
#include <exception>
#include <mutex>
#include <string>
#include <mmsystem.h>

// ==== External Symbols from main.cpp ====
// These are defined in main.cpp and control thread loops.
extern std::atomic<bool> send_bandw_Running;
extern std::atomic<bool> receive_resend_Running;
extern std::atomic<bool> receive_raw_packet_Running;
// This will be created in a future step to replace the local bool in wWinMain.
extern std::atomic<bool> app_running_atomic;

// ==== Idempotency Guard ====
static std::once_flag g_shutdownOnce;

// ==== Helper Functions ====
static void SafeJoin(std::thread* t, const wchar_t* name) noexcept {
    if (!t) return;
    try {
        if (t->joinable()) t->join();
    } catch (const std::exception& e) {
        // Safely convert narrow char exception message to wide string
        const char* what_cstr = e.what();
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, what_cstr, (int)strlen(what_cstr), NULL, 0);
        std::wstring wide_what(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, what_cstr, (int)strlen(what_cstr), &wide_what[0], size_needed);
        DebugLog(L"SafeJoin: exception while joining [" + std::wstring(name) + L"]: " + wide_what);
    } catch (...) {
        DebugLog(L"SafeJoin: unknown exception while joining [" + std::wstring(name) + L"]");
    }
}

static void SafeJoinVector(std::vector<std::thread>* vec, const wchar_t* name) noexcept {
    if (!vec) return;
    for (size_t i = 0; i < vec->size(); ++i) {
        // Check joinable before accessing the thread object to avoid issues
        if (vec->at(i).joinable()) {
            SafeJoin(&(*vec)[i], (std::wstring(name) + L"[" + std::to_wstring(i) + L"]").c_str());
        }
    }
}

static void FinalWSACleanup() noexcept {
    try {
        if (WSACleanup() != 0) {
            int w = WSAGetLastError();
            // WSANOTINITIALISED is an expected error if WSAStartup failed or was never called.
            if (w != WSANOTINITIALISED) {
                 DebugLog(L"FinalWSACleanup failed. Error: " + std::to_wstring(w));
            }
        } else {
            DebugLog(L"WSACleanup succeeded.");
        }
    } catch(...) {
        DebugLog(L"FinalWSACleanup: unexpected exception (ignored).");
    }
}

// ==== Core Shutdown Logic ====
void RequestShutdown(std::atomic<bool>* appRunningPtr) {
    try {
        // 1) Set all running flags to false to signal threads to stop.
        send_bandw_Running.store(false, std::memory_order_relaxed);
        receive_resend_Running.store(false, std::memory_order_relaxed);
        receive_raw_packet_Running.store(false, std::memory_order_relaxed);
        g_fec_worker_Running.store(false, std::memory_order_relaxed);
        g_decode_worker_Running.store(false, std::memory_order_relaxed);

        // This handles the main render loop flag in wWinMain.
        if (appRunningPtr) {
            appRunningPtr->store(false, std::memory_order_relaxed);
        }

        // 2) Post WM_QUIT to exit the message loop in wWinMain.
        PostQuitMessage(0);
        DebugLog(L"RequestShutdown: Flags cleared and WM_QUIT posted.");
    } catch (...) {
        // This function must not throw, as it's a critical part of shutdown.
        DebugLog(L"RequestShutdown: unexpected exception (ignored).");
    }
}

void ReleaseAllResources(const AppThreads& threads) {
    std::call_once(g_shutdownOnce, [&]() {
        DebugLog(L"ReleaseAllResources: Begin");

        // 1) Join all threads to ensure they have exited cleanly.
        try {
            SafeJoin(threads.bandwidthThread, L"bandwidthThread");
            SafeJoin(threads.resendThread, L"resendThread");
            SafeJoinVector(threads.receiverThreads, L"receiverThreads");
            SafeJoinVector(threads.fecWorkerThreads, L"fecWorkerThreads");
            SafeJoinVector(threads.nvdecThreads, L"nvdecThreads");
            SafeJoin(threads.windowSenderThread, L"windowSenderThread");
        } catch (...) {
            DebugLog(L"ReleaseAllResources: exception during thread join (ignored to continue cleanup).");
        }

        // 2) Wait for the GPU to finish its work, then release D3D12 resources.
        try {
            WaitForGpu();
            CleanupD3DRenderResources();
        } catch (...) {
            DebugLog(L"ReleaseAllResources: exception during D3D12 cleanup (ignored).");
        }

        // 3) Release NVDEC decoder and the primary CUDA context.
        try {
            if (g_frameDecoder) {
                g_frameDecoder.reset();
            }
            cuDevicePrimaryCtxRelease(0);
        } catch (...) {
            DebugLog(L"ReleaseAllResources: exception during NVDEC/CUDA cleanup (ignored).");
        }

        // 4) Release Network and Timer resources.
        try {
            enet_deinitialize();
            FinalWSACleanup(); // Centralized WSA cleanup.
        } catch (...) {
            DebugLog(L"ReleaseAllResources: exception during network/timer cleanup (ignored).");
        }

        // 5) Release Reed-Solomon matrices.
        try {
            if (g_vandermonde_matrix) { free(g_vandermonde_matrix); g_vandermonde_matrix = nullptr; }
            if (g_jerasure_matrix)    { free(g_jerasure_matrix);    g_jerasure_matrix    = nullptr; }
        } catch (...) {
            DebugLog(L"ReleaseAllResources: exception during matrix cleanup (ignored).");
        }

        DebugLog(L"ReleaseAllResources: Done");
    });
}
