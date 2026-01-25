#pragma once
#include <atomic>
#include <cstdint>

// enqueue from UI thread
void EnqueueKeyboardRawEvent(uint16_t makeCode, uint16_t rawFlags); // RAWKEYBOARD.Flags
void EnqueueKeyboardFocusChanged(bool active);

// sender thread entry
void KeyboardSendThread(std::atomic<bool>& running);
