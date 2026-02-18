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

// Helper to convert a virtual-key code to a scan code suitable for
// our KeyboardInputMessage. If MapVirtualKeyW fails, we fall back
// to the provided default scan code.
static uint16_t GetScanCodeForVk(UINT vk, uint16_t fallback)
{
    UINT sc = MapVirtualKeyW(vk, MAPVK_VK_TO_VSC);
    if (sc == 0) {
        return fallback;
    }
    return static_cast<uint16_t>(sc & 0xFF);
}

// Helper to enqueue a single keyboard event (down/up) into the global
// keyboard input queue, using the same encoding convention as the
// low-level hook:
//   state bit 0 -> 0 = DOWN, 1 = UP
//   state bit 1 -> E0 flag (extended key)
//   state bit 2 -> E1 flag (unused here)
static void EnqueueSyntheticKey(uint16_t scancode, bool isExtended, bool keyDown)
{
    KeyboardInputMessage msg{};
    msg.scancode  = scancode;
    uint16_t state = keyDown ? 0x00 : 0x01; // DOWN or UP
    if (isExtended) {
        state = static_cast<uint16_t>(state | 0x02); // KEY_E0
    }
    msg.state     = state;
    msg.timestamp = static_cast<uint32_t>(GetTickCount());

    g_keyboardInputQueue.enqueue(msg);
}

// Send Win + L (lock screen) to the remote host.
void SendShortcutWinL()
{
    const uint16_t scWin = GetScanCodeForVk(VK_LWIN, 0x5B);
    const uint16_t scL   = GetScanCodeForVk('L',    0x26);

    // Win down, L down, L up, Win up
    EnqueueSyntheticKey(scWin, true,  true);
    EnqueueSyntheticKey(scL,   false, true);
    EnqueueSyntheticKey(scL,   false, false);
    EnqueueSyntheticKey(scWin, true,  false);
}

// Send Win + G to the remote host.
void SendShortcutWinG()
{
    const uint16_t scWin = GetScanCodeForVk(VK_LWIN, 0x5B);
    const uint16_t scG   = GetScanCodeForVk('G',    0x22);

    // Win down, G down, G up, Win up
    EnqueueSyntheticKey(scWin, true,  true);
    EnqueueSyntheticKey(scG,   false, true);
    EnqueueSyntheticKey(scG,   false, false);
    EnqueueSyntheticKey(scWin, true,  false);
}

// Send Win + Alt + B to the remote host.
void SendShortcutWinAltB()
{
    const uint16_t scWin = GetScanCodeForVk(VK_LWIN, 0x5B);
    const uint16_t scAlt = GetScanCodeForVk(VK_MENU, 0x38); // Left Alt
    const uint16_t scB   = GetScanCodeForVk('B',     0x30);

    // Win down, Alt down, B down, B up, Alt up, Win up
    EnqueueSyntheticKey(scWin, true,  true);
    EnqueueSyntheticKey(scAlt, false, true);
    EnqueueSyntheticKey(scB,   false, true);
    EnqueueSyntheticKey(scB,   false, false);
    EnqueueSyntheticKey(scAlt, false, false);
    EnqueueSyntheticKey(scWin, true,  false);
}

// Send Ctrl + Alt + Delete to the remote host.
void SendShortcutCtrlAltDel()
{
    const uint16_t scCtrl   = GetScanCodeForVk(VK_CONTROL, 0x1D); // Left Ctrl
    const uint16_t scAlt    = GetScanCodeForVk(VK_MENU,    0x38); // Left Alt
    const uint16_t scDelete = GetScanCodeForVk(VK_DELETE,  0x53); // Extended Delete

    // Ctrl down, Alt down, Delete down, Delete up, Alt up, Ctrl up
    EnqueueSyntheticKey(scCtrl,   false, true);
    EnqueueSyntheticKey(scAlt,    false, true);
    EnqueueSyntheticKey(scDelete, true,  true);
    EnqueueSyntheticKey(scDelete, true,  false);
    EnqueueSyntheticKey(scAlt,    false, false);
    EnqueueSyntheticKey(scCtrl,   false, false);
}



// Low-level keyboard hook procedure.
// This intercepts physical keyboard input at the OS level so we can:
//  - Forward it to the remote host, and
//  - Prevent it from affecting the local OS and other applications
//    when this client process owns the foreground window.
static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    // Ctrl+Alt の同時押し状態をトラッキングして、ローカル側の
    // フルスクリーントグル専用ホットキーとして扱う。
    // キーイベント自体は従来通りリモートへも送信する。
    static bool sCtrlDown = false;
    static bool sAltDown = false;
    static bool sCtrlAltHandledForChord = false;

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
    const DWORD thisPid = GetCurrentProcessId();
    if (fgPid != thisPid) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    // Ignore events that we injected ourselves to avoid feedback loops.
    if (p->flags & LLKHF_INJECTED) {
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

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
        // Do not intercept other message types.
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }

    // Ctrl+Alt のコンビネーションを検出して、Qt メインウィンドウに
    // WM_APP+1 を投げる。1回の押下（Ctrl/Alt どちらかが離されるまで）
    // に対して 1 回だけトグルが走るようにラッチする。
    if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN) {
        if (p->vkCode == VK_LCONTROL || p->vkCode == VK_RCONTROL || p->vkCode == VK_CONTROL) {
            sCtrlDown = true;
        } else if (p->vkCode == VK_LMENU || p->vkCode == VK_RMENU || p->vkCode == VK_MENU) {
            sAltDown = true;
        }

        if (sCtrlDown && sAltDown && !sCtrlAltHandledForChord) {
            sCtrlAltHandledForChord = true;

            if (g_mainWindowHwnd) {
                PostMessageW(g_mainWindowHwnd, WM_APP + 1, 0, 0);
            }
        }
    } else if (wParam == WM_KEYUP || wParam == WM_SYSKEYUP) {
        if (p->vkCode == VK_LCONTROL || p->vkCode == VK_RCONTROL || p->vkCode == VK_CONTROL) {
            sCtrlDown = false;
        } else if (p->vkCode == VK_LMENU || p->vkCode == VK_RMENU || p->vkCode == VK_MENU) {
            sAltDown = false;
        }

        if (!sCtrlDown && !sAltDown) {
            sCtrlAltHandledForChord = false;
        }
    }

    // Map the LL hook data (vkCode / scanCode / flags) to Interception-style
    // scan code + state bits. Pause and NumLock are special because they share
    // the same physical scan code (0x45) but are distinguished by the E0/E1
    // flags on the remote side.
    uint16_t scancode = static_cast<uint16_t>(p->scanCode & 0xFF);

    if (p->vkCode == VK_PAUSE) {
        // Pause (and Ctrl+Pause / Break) use the E1 flag in Interception.
        // Use the shared scan code 0x45 and set KEY_E1 (0x04).
        scancode = 0x45;
        state = static_cast<uint16_t>(state | 0x04); // INTERCEPTION_KEY_E1
    } else {
        // For all non-Pause keys, set the E0 flag when Windows reports the
        // LLKHF_EXTENDED flag. This covers arrow keys, Insert/Delete, etc.
        if (p->flags & LLKHF_EXTENDED) {
            state = static_cast<uint16_t>(state | 0x02); // INTERCEPTION_KEY_E0
        }

        // NumLock does not reliably set LLKHF_EXTENDED in the low-level hook,
        // but Interception expects the E0 flag to distinguish it from Pause.
        // Force KEY_E0 here so that NumLock is delivered correctly.
        if (p->vkCode == VK_NUMLOCK) {
            state = static_cast<uint16_t>(state | 0x02); // INTERCEPTION_KEY_E0
        }
    }

    KeyboardInputMessage msg{};
    msg.scancode  = scancode;
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
