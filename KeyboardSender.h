#pragma once
#include <atomic>
#include <cstdint>

// enqueue from UI thread
void EnqueueKeyboardRawEvent(uint16_t makeCode, uint16_t rawFlags, uint16_t vkey); // RAWKEYBOARD.Flags + VirtualKey(0=unknown)
void EnqueueKeyboardFocusChanged(bool active);

// sender thread entry
void KeyboardSendThread(std::atomic<bool>& running);
