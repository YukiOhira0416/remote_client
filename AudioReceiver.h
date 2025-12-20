#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

// Starts the audio receiver thread.
void StartAudioReceiver();

// Signals the audio receiver thread to stop and waits for it to exit.
void StopAudioReceiver();

