#include "RemoteKeyboard.h"
#include "Globals.h"
#include "ReedSolomon.h"
#include "DebugLog.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <vector>
#include <chrono>
#include <thread>
#include <cstring>

#pragma comment(lib, "Ws2_32.lib")

// Global state for the low-level keyboard hook.
static HHOOK g_keyboardHook = nullptr;
static std::atomic<bool> g_remoteKeyboardEnabled{true};

// Low-level keyboard hook procedure.
// This intercepts physical keyboard input at the OS level so we can:
//  - Forward it to the remote host, and
//  - Prevent it from affecting the local OS and other applications
//    when this client process owns the foreground window.
static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    if (nCode < 0) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    if (!g_remoteKeyboardEnabled.load(std::memory_order_acquire)) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    KBDLLHOOKSTRUCT* p = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);
    if (!p) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    // Only intercept when this process's window is in the foreground.
    HWND fg = GetForegroundWindow();
    if (!fg) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    DWORD fgPid = 0;
    GetWindowThreadProcessId(fg, &fgPid);
    DWORD thisPid = GetCurrentProcessId();
    if (fgPid != thisPid) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    // We only care about key down/up (including system keys).
    uint16_t state = 0;
    switch (wParam) {
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
        state = 0x00; // INTERCEPTION_KEY_DOWN
        break;

    case WM_KEYUP:
    case WM_SYSKEYUP:
        state = 0x01; // INTERCEPTION_KEY_UP
        break;

    default:
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    // Set extended-key flag (E0) if present.
    // INTERCEPTION_KEY_E0 is 0x02 in the InterceptionKeyState enum.
    if (p->flags & LLKHF_EXTENDED) {
        state = static_cast<uint16_t>(state | 0x02);
    }

    KeyboardInputMessage msg{};
    msg.scancode  = static_cast<uint16_t>(p->scanCode);
    msg.state     = state;
    msg.timestamp = static_cast<uint32_t>(GetTickCount());

    g_keyboardInputQueue.enqueue(msg);

    // Returning a non-zero value prevents the event from being passed
    // to the rest of the system. This ensures that while our client
    // has the foreground focus, physical keyboard input does NOT
    // control the local OS or other applications.
    return 1;
}

bool InitializeRemoteKeyboard()
{
    if (g_keyboardHook) {
        return true; // already installed
    }

    HINSTANCE hInstance = GetModuleHandleW(nullptr);
    if (!hInstance) {
        DebugLog(L"InitializeRemoteKeyboard: GetModuleHandleW(nullptr) failed.");
        return false;
    }

    g_keyboardHook = SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc, hInstance, 0);
    if (!g_keyboardHook) {
        DWORD err = GetLastError();
        DebugLog(L"InitializeRemoteKeyboard: SetWindowsHookExW failed. Error: " + std::to_wstring(err));
        return false;
    }

    g_remoteKeyboardEnabled.store(true, std::memory_order_release);
    DebugLog(L"InitializeRemoteKeyboard: low-level keyboard hook installed.");
    return true;
}

void ShutdownRemoteKeyboard()
{
    try {
        if (g_keyboardHook) {
            if (UnhookWindowsHookEx(g_keyboardHook) == 0) {
                DWORD err = GetLastError();
                DebugLog(L"ShutdownRemoteKeyboard: UnhookWindowsHookEx failed. Error: " + std::to_wstring(err));
            } else {
                DebugLog(L"ShutdownRemoteKeyboard: keyboard hook removed.");
            }
            g_keyboardHook = nullptr;
        }
    } catch (...) {
        // Must not throw during shutdown.
        DebugLog(L"ShutdownRemoteKeyboard: unexpected exception (ignored).");
    }
}

void SetRemoteKeyboardEnabled(bool enabled)
{
    g_remoteKeyboardEnabled.store(enabled, std::memory_order_release);
}

// Thread that sends keyboard input messages to the remote agent over UDP
// using the same FEC scheme as the mouse input sender.
void KeyboardSendThread(std::atomic<bool>& running)
{
    DebugLog(L"KeyboardSendThread started.");

    SOCKET udpSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSocket == INVALID_SOCKET) {
        DebugLog(L"KeyboardSendThread: socket() failed: " + std::to_wstring(WSAGetLastError()));
        // Disable the remote keyboard hook if we cannot talk to the server,
        // otherwise we would block local keyboard input without sending it.
        SetRemoteKeyboardEnabled(false);
        return;
    }

    // Bind to the configured local client IP/port so that all keyboard traffic
    // originates from the expected endpoint (REMOTE_KEYBOARD_CLIENT_IP:REMOTE_KEYBOARD_CLIENT_PORT).
    sockaddr_in localAddr{};
    localAddr.sin_family = AF_INET;
    localAddr.sin_port   = htons(REMOTE_KEYBOARD_CLIENT_PORT);
    if (inet_pton(AF_INET, REMOTE_KEYBOARD_CLIENT_IP, &localAddr.sin_addr) != 1) {
        DebugLog(L"KeyboardSendThread: inet_pton failed for REMOTE_KEYBOARD_CLIENT_IP.");
        closesocket(udpSocket);
        SetRemoteKeyboardEnabled(false);
        return;
    }
    if (bind(udpSocket, reinterpret_cast<sockaddr*>(&localAddr), sizeof(localAddr)) == SOCKET_ERROR) {
        DebugLog(L"KeyboardSendThread: bind() failed: " + std::to_wstring(WSAGetLastError()));
        closesocket(udpSocket);
        SetRemoteKeyboardEnabled(false);
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port   = htons(REMOTE_KEYBOARD_SERVER_PORT);
    if (inet_pton(AF_INET, REMOTE_KEYBOARD_SERVER_IP, &serverAddr.sin_addr) != 1) {
        DebugLog(L"KeyboardSendThread: inet_pton failed for REMOTE_KEYBOARD_SERVER_IP.");
        closesocket(udpSocket);
        SetRemoteKeyboardEnabled(false);
        return;
    }

    std::atomic<uint32_t> frameCounter(0);

    while (running.load(std::memory_order_acquire)) {
        KeyboardInputMessage msg{};
        if (!g_keyboardInputQueue.try_dequeue(msg)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        const uint8_t* original_data = reinterpret_cast<const uint8_t*>(&msg);
        const size_t   original_len  = sizeof(KeyboardInputMessage);

        std::vector<std::vector<uint8_t>> dataShards;
        std::vector<std::vector<uint8_t>> parityShards;
        size_t shard_len = 0;

        if (!EncodeFEC_ISAL(original_data,
                            original_len,
                            dataShards,
                            parityShards,
                            shard_len,
                            RS_K,
                            RS_M)) {
            DebugLog(L"KeyboardSendThread: EncodeFEC_ISAL failed.");
            continue;
        }

        const uint32_t frameNumber = frameCounter.fetch_add(1, std::memory_order_relaxed);

        // Send data shards
        for (int i = 0; i < RS_K; ++i) {
            ShardInfoHeader header{};
            header.frameNumber       = htonl(frameNumber);
            header.shardIndex        = htonl(static_cast<uint32_t>(i));
            header.totalDataShards   = htonl(RS_K);
            header.totalParityShards = htonl(RS_M);
            header.originalDataLen   = htonl(static_cast<uint32_t>(original_len));

            std::vector<uint8_t> packet(sizeof(ShardInfoHeader) + shard_len);
            std::memcpy(packet.data(), &header, sizeof(header));
            std::memcpy(packet.data() + sizeof(header),
                        dataShards[i].data(),
                        shard_len);

            int sent = sendto(udpSocket,
                              reinterpret_cast<const char*>(packet.data()),
                              static_cast<int>(packet.size()),
                              0,
                              reinterpret_cast<sockaddr*>(&serverAddr),
                              sizeof(serverAddr));
            if (sent == SOCKET_ERROR) {
                DebugLog(L"KeyboardSendThread: sendto(data) failed: " + std::to_wstring(WSAGetLastError()));
            }
        }

        // Send parity shards
        for (int i = 0; i < RS_M; ++i) {
            ShardInfoHeader header{};
            header.frameNumber       = htonl(frameNumber);
            header.shardIndex        = htonl(static_cast<uint32_t>(RS_K + i));
            header.totalDataShards   = htonl(RS_K);
            header.totalParityShards = htonl(RS_M);
            header.originalDataLen   = htonl(static_cast<uint32_t>(original_len));

            std::vector<uint8_t> packet(sizeof(ShardInfoHeader) + shard_len);
            std::memcpy(packet.data(), &header, sizeof(header));
            std::memcpy(packet.data() + sizeof(header),
                        parityShards[i].data(),
                        shard_len);

            int sent = sendto(udpSocket,
                              reinterpret_cast<const char*>(packet.data()),
                              static_cast<int>(packet.size()),
                              0,
                              reinterpret_cast<sockaddr*>(&serverAddr),
                              sizeof(serverAddr));
            if (sent == SOCKET_ERROR) {
                DebugLog(L"KeyboardSendThread: sendto(parity) failed: " + std::to_wstring(WSAGetLastError()));
            }
        }
    }

    closesocket(udpSocket);
    DebugLog(L"KeyboardSendThread stopped.");
}
