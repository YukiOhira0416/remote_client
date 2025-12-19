#pragma once
#include <thread>
#include <atomic>
#include <map>
#include <vector>
#include "AudioReceiver.h"
#include "JitterBuffer.h"

class AudioDecoder {
public:
    // The decoder needs sources for shards and a destination for decoded blocks.
    AudioDecoder(AudioReceiver& receiver, JitterBuffer& jitterBuffer);
    ~AudioDecoder();

    // Starts the decoder thread.
    void Start();

    // Stops the decoder thread.
    void Stop();

private:
    // The main function for the decoder thread.
    void DecoderThread();

    // Handles the processing of a single incoming shard.
    void ProcessShard(const AudioShard& shard);

    // Performs the required hex dump logging for a completed block.
    void LogDecodedBlock(const AudioPacketHeader& header, const std::vector<uint8_t>& pcmData);

    // Private struct to hold the state of a block being assembled.
    struct BlockAssembler {
        // Metadata from the first shard received for this block
        uint16_t k = 0;
        uint16_t m = 0;
        uint16_t shardBytes = 0;
        uint16_t pcmBytesInBlock = 0;
        uint32_t blockCrc32 = 0;
        uint64_t captureTimestampNs = 0;

        // Storage for received shards, keyed by ShardIndex
        std::map<uint32_t, std::vector<uint8_t>> receivedShards;
    };

    std::atomic<bool> m_isRunning{false};
    std::thread m_thread;

    AudioReceiver& m_receiver;
    JitterBuffer& m_jitterBuffer;

    uint32_t m_currentSessionId{0};
    std::map<uint32_t, BlockAssembler> m_pendingBlocks;
};
