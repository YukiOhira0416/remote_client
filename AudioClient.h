#pragma once
#include <atomic>
#include <cstdint>
#include <thread>

// Audio packet header definition (packed 64 bytes)
#pragma pack(push, 1)
struct AudioPacketHeader {
    char     Magic[4];           // "AUDP"
    uint8_t  Version;            // 1
    uint8_t  HeaderSize;         // 64
    uint8_t  Flags;              // bit0=IS_PARITY
    uint8_t  Reserved0;
    uint32_t SessionId;
    uint32_t BlockId;
    uint16_t ShardIndex;
    uint16_t ShardTotal;
    uint16_t K;
    uint16_t M;
    uint16_t ShardBytes;
    uint16_t PcmBytesInBlock;    // 1920
    uint16_t BlockSamplesPerCh;  // 480
    uint16_t Reserved1;
    uint32_t SampleRate;         // 48000
    uint16_t Channels;           // 2
    uint16_t BitsPerSample;      // 16
    uint64_t CaptureTimestampNs;
    uint32_t BlockCrc32;
    uint32_t PayloadCrc32;
    uint32_t HeaderCrc32;
    uint32_t Reserved2;
};
#pragma pack(pop)

// External running flags to coordinate shutdown.
extern std::atomic<bool> g_audioReceiverRunning;
extern std::atomic<bool> g_audioPlaybackRunning;

// Starts the audio receiver and playback threads.
// Returns true on success; threads are placed into the provided references.
bool StartAudioClient(std::thread& receiverThread, std::thread& playbackThread);

// Signals audio threads to stop.
void StopAudioClient();
