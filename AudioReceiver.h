#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <vector>
#include <winsock2.h>
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include "DebugLog.h"

// Audio receiver implementing 10ms RS-coded audio playback.
class AudioReceiver {
public:
    AudioReceiver();
    ~AudioReceiver();

    bool Start();
    void Stop();

private:
    #pragma pack(push, 1)
    struct AudioPacketHeader {
        char     Magic[4];
        uint8_t  Version;
        uint8_t  HeaderSize;
        uint8_t  Flags;
        uint8_t  Reserved0;
        uint32_t SessionId;
        uint32_t BlockId;
        uint16_t ShardIndex;
        uint16_t ShardTotal;
        uint16_t K;
        uint16_t M;
        uint16_t ShardBytes;
        uint16_t PcmBytesInBlock;
        uint16_t BlockSamplesPerCh;
        uint16_t Reserved1;
        uint32_t SampleRate;
        uint16_t Channels;
        uint16_t BitsPerSample;
        uint64_t CaptureTimestampNs;
        uint32_t BlockCrc32;
        uint32_t PayloadCrc32;
        uint32_t HeaderCrc32;
        uint32_t Reserved2;
    };
    static_assert(sizeof(AudioPacketHeader) == 64, "AudioPacketHeader must be 64 bytes");
    #pragma pack(pop)

    struct ReceivedPacket {
        AudioPacketHeader header{};
        std::vector<uint8_t> payload;
        uint64_t arrival_ms = 0;
    };

    struct BlockKey {
        uint32_t sessionId;
        uint32_t blockId;
        bool operator<(const BlockKey& other) const {
            return std::tie(sessionId, blockId) < std::tie(other.sessionId, other.blockId);
        }
    };

    struct BlockBuffer {
        uint16_t k = 0;
        uint16_t m = 0;
        uint16_t shardBytes = 0;
        uint16_t pcmBytes = 0;
        uint16_t shardTotal = 0;
        uint32_t blockCrc32 = 0;
        uint64_t captureTimestampNs = 0;
        uint64_t firstSeenMs = 0;
        std::map<uint16_t, std::vector<uint8_t>> dataShards;
    };

    struct DecodedBlock {
        AudioPacketHeader header{};
        std::vector<uint8_t> pcm;
        uint64_t completed_ms = 0;
    };

    bool InitializeSocket();
    void ReceiveLoop();
    void RenderLoop();

    bool ParsePacket(const uint8_t* data, size_t len, ReceivedPacket& out);
    uint32_t ComputeCrc32(const uint8_t* data, size_t len) const;
    void TryAssemble(const ReceivedPacket& pkt);
    bool AttemptDecode(BlockKey key, BlockBuffer& buffer);
    void LogDecodedBlock(const DecodedBlock& block);

    void ResetSession(uint32_t sessionId);

    SOCKET udpSocket_ = INVALID_SOCKET;
    std::atomic<bool> running_{false};
    std::thread recvThread_;
    std::thread renderThread_;

    std::mutex buffersMutex_;
    std::map<BlockKey, BlockBuffer> buffers_;
    uint32_t currentSessionId_ = 0;
    uint32_t expectedBlockId_ = 0;
    bool haveExpectedBlock_ = false;

    std::condition_variable renderCv_;
    std::mutex renderMutex_;
    std::map<uint32_t, DecodedBlock> jitterBuffer_;
    const uint32_t jitterBlocks_ = 2;

    // WASAPI
    bool InitAudioClient();
    void ShutdownAudioClient();
    bool WriteBlockToRenderClient(const uint8_t* data, size_t bytes);

    IMMDevice* device_ = nullptr;
    IAudioClient* audioClient_ = nullptr;
    IAudioRenderClient* renderClient_ = nullptr;
    HANDLE audioEvent_ = nullptr;
    WAVEFORMATEX* mixFormat_ = nullptr;
    uint32_t bufferFrameCount_ = 0;
};

