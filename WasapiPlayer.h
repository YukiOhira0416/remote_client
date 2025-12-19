#pragma once
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <thread>
#include <atomic>
#include <vector>
#include "JitterBuffer.h"

class WasapiPlayer {
public:
    WasapiPlayer(JitterBuffer& jitterBuffer);
    ~WasapiPlayer();

    // Initializes WASAPI and starts the playback thread.
    bool Start();

    // Stops the playback thread and releases WASAPI resources.
    void Stop();

private:
    void PlaybackThread();
    bool InitializeWasapi();
    void ReleaseWasapi();

    JitterBuffer& m_jitterBuffer;

    std::atomic<bool> m_isRunning{false};
    std::thread m_thread;

    // WASAPI interfaces
    IMMDeviceEnumerator* m_pEnumerator = nullptr;
    IMMDevice* m_pDevice = nullptr;
    IAudioClient* m_pAudioClient = nullptr;
    IAudioRenderClient* m_pRenderClient = nullptr;

    HANDLE m_hAudioEvent = nullptr;
    UINT32 m_bufferFrameCount = 0;
    WAVEFORMATEX m_waveFormat{};

    // State for playback synchronization
    uint32_t m_expectedBlockId{0};
    bool m_isSynchronized{false};

    // Buffer for handling audio data that spans multiple WASAPI callbacks.
    std::vector<uint8_t> m_partialBlockBuffer;
};
