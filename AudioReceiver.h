#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <atlbase.h>
#include <winsock2.h>

struct AudioPacketHeader;

class WasapiPlayer {
public:
    WasapiPlayer();
    ~WasapiPlayer();

    bool Initialize();
    void EnqueuePcm(const std::vector<uint8_t>& pcmData);
    void Stop();

private:
    void PlaybackThread();
    void ShutdownInternal();

    std::atomic<bool> m_running{false};
    std::thread m_thread;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCv;
    std::queue<std::vector<uint8_t>> m_queue;

    CComPtr<IAudioClient> m_audioClient;
    CComPtr<IAudioRenderClient> m_renderClient;
    WAVEFORMATEX m_waveFormat{};
    UINT32 m_bufferFrameCount = 0;
    uint32_t m_frameSizeBytes = 0;
};

class AudioReceiver {
public:
    AudioReceiver();
    ~AudioReceiver();

    bool Start();
    void Stop();

    struct FrameKey {
        uint32_t streamId;
        uint32_t frameId;
        bool operator<(const FrameKey& other) const {
            if (streamId != other.streamId) return streamId < other.streamId;
            return frameId < other.frameId;
        }
    };

    struct FrameBufferEntry {
        uint8_t endian = 1;
        uint16_t flags = 0;
        uint32_t streamId = 0;
        uint32_t frameId = 0;
        uint64_t captureTimeNs = 0;
        uint32_t sampleRate = 48000;
        uint16_t channels = 2;
        uint16_t bitsPerSample = 16;
        uint16_t frameDurationUs = 20000;
        uint16_t shardSize = 0;
        uint32_t originalBytes = 0;
        uint8_t fecK = 0;
        uint8_t fecM = 0;
        std::map<uint32_t, std::vector<uint8_t>> shards;
        std::chrono::steady_clock::time_point firstShardTime{};
    };

private:
    struct DecodedFrame {
        FrameBufferEntry headerInfo;
        std::vector<uint8_t> pcm;
    };

    bool InitializeSocket();
    void ReceiverThread();
    void TimeoutThread();
    void ProcessDecodedFrame(const DecodedFrame& frame);
    bool TryDecodeFrame(const FrameKey& key, FrameBufferEntry entry);
    void EnqueueSilence(const FrameKey& key, const FrameBufferEntry& entry);
    void PumpPlaybackQueue();

    std::atomic<bool> m_running{false};
    std::thread m_receiverThread;
    std::thread m_timeoutThread;
    SOCKET m_socket = INVALID_SOCKET;

    std::mutex m_frameMutex;
    std::map<FrameKey, FrameBufferEntry> m_frames;
    std::map<FrameKey, DecodedFrame> m_pendingPlayback;

    uint32_t m_expectedFrameId = 0;
    uint32_t m_currentStreamId = 0;
    bool m_haveStream = false;

    WasapiPlayer m_player;
};

bool ParseAudioPacketHeader(const uint8_t* data, size_t len, AudioPacketHeader& outHeader);
uint32_t AudioCrc32(const uint8_t* data, size_t len);
uint32_t AudioCrc32(const uint8_t* data, size_t len, uint32_t previous);
std::vector<uint8_t> BuildDecodedLogHeader(const AudioReceiver::FrameBufferEntry& info, const std::vector<uint8_t>& pcm);
std::wstring HexDump(const std::vector<uint8_t>& data);
