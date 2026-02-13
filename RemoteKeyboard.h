#pragma once
#include <atomic>

// Initialize the low-level keyboard hook used for remote keyboard capture.
// Returns true on success, false on failure (in which case remote keyboard is disabled).
bool InitializeRemoteKeyboard();

// Remove the low-level keyboard hook if it was installed.
void ShutdownRemoteKeyboard();

// Enable or disable remote keyboard interception globally.
// When disabled, all key events are passed through to the OS.
void SetRemoteKeyboardEnabled(bool enabled);

// Thread procedure for sending queued keyboard input to the remote agent.
// The thread will exit when 'running' is set to false.
void KeyboardSendThread(std::atomic<bool>& running);
