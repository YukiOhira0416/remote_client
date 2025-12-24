#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

// Starts the audio receive / FEC / playback pipeline.
// Safe to call after WSAStartup and COM initialization in the main thread.
void StartAudioPipeline();

// Signals all audio threads to stop and waits for them to finish.
void ShutdownAudioPipeline();

// Optional: allow the video path to report measured delay so audio sync can follow it.
// video_capture_ts_ms: server capture timestamp from the video header (ms).
// client_present_ns:   system_clock timestamp on the client when the frame finished presenting (ns).
void NotifyVideoPresent(uint64_t video_capture_ts_ms, uint64_t client_present_ns);

