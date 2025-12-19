#include "AudioDecoder.h"
#include "ReedSolomon.h"
#include "Crc32.h"
#include "DebugLog.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

AudioDecoder::AudioDecoder(AudioReceiver& receiver, JitterBuffer& jitterBuffer)
    : m_receiver(receiver), m_jitterBuffer(jitterBuffer) {}

AudioDecoder::~AudioDecoder() {
    Stop();
}

void AudioDecoder::Start() {
    if (m_isRunning) {
        return;
    }
    m_isRunning = true;
    m_thread = std::thread(&AudioDecoder::DecoderThread, this);
    DebugLog(L"AudioDecoder: Started.");
}

void AudioDecoder::Stop() {
    if (!m_isRunning) {
        return;
    }
    m_isRunning = false;
    if (m_thread.joinable()) {
        m_thread.join();
    }
    DebugLog(L"AudioDecoder: Stopped.");
}

void AudioDecoder::DecoderThread() {
    while (m_isRunning) {
        AudioShard shard;
        // The TryDequeue will block for a short time if the queue is empty.
        // We can add a small sleep if it's too busy.
        if (m_receiver.TryDequeue(shard)) {
            ProcessShard(shard);
        } else {
            // No shard, wait a bit to prevent busy-looping.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void AudioDecoder::ProcessShard(const AudioShard& shard) {
    // Session ID change check
    if (shard.header.SessionId != m_currentSessionId) {
        DebugLog(L"AudioDecoder: New SessionId detected (old: %u, new: %u). Clearing state.", m_currentSessionId, shard.header.SessionId);
        m_currentSessionId = shard.header.SessionId;
        m_pendingBlocks.clear();
    }

    // Ignore shards for blocks we have already processed and removed.
    // This is a simple protection against very late packets.
    // A more robust solution might use a sequence number check.

    auto& assembler = m_pendingBlocks[shard.header.BlockId];

    // If this is the first shard for this block, store the metadata.
    if (assembler.receivedShards.empty()) {
        assembler.k = shard.header.K;
        assembler.m = shard.header.M;
        assembler.shardBytes = shard.header.ShardBytes;
        assembler.pcmBytesInBlock = shard.header.PcmBytesInBlock;
        assembler.blockCrc32 = shard.header.BlockCrc32;
        assembler.captureTimestampNs = shard.header.CaptureTimestampNs;
    }

    // Add the shard to the assembler
    assembler.receivedShards[shard.header.ShardIndex] = shard.payload;

    // Check if we have enough shards to attempt decoding
    if (assembler.receivedShards.size() >= assembler.k) {
        std::vector<uint8_t> decodedData;
        bool success = DecodeFEC_ISAL(
            assembler.receivedShards,
            assembler.k,
            assembler.m,
            assembler.k * assembler.shardBytes, // We need the padded size for reconstruction
            decodedData
        );

        if (success) {
            // Trim to the actual PCM data size specified in the header.
            if (decodedData.size() >= assembler.pcmBytesInBlock) {
                decodedData.resize(assembler.pcmBytesInBlock);

                // Final verification: Block CRC32
                if (Crc32(decodedData.data(), decodedData.size()) == assembler.blockCrc32) {
                    // Success!
                    DecodedAudioBlock block;
                    block.blockId = shard.header.BlockId;
                    block.pcmData = std::move(decodedData);

                    // Log before moving the data.
                    LogDecodedBlock(shard.header, block.pcmData);

                    m_jitterBuffer.AddBlock(std::move(block));
                } else {
                    DebugLog(L"AudioDecoder: Block %u failed BlockCrc32 check after decode.", shard.header.BlockId);
                }
            } else {
                 DebugLog(L"AudioDecoder: Block %u decoded data is smaller than PcmBytesInBlock.", shard.header.BlockId);
            }
        } else {
             DebugLog(L"AudioDecoder: ISA-L decode failed for block %u.", shard.header.BlockId);
        }

        // Whether decoding succeeded or failed, we are done with this block.
        m_pendingBlocks.erase(shard.header.BlockId);
    }
}

void AudioDecoder::LogDecodedBlock(const AudioPacketHeader& header, const std::vector<uint8_t>& pcmData) {
    // As per spec, create a representative header.
    // We'll use the header from the last received shard and normalize it.
    AudioPacketHeader logHeader = header;
    logHeader.ShardIndex = 0;
    logHeader.Flags &= ~0x01; // Clear IS_PARITY flag
    logHeader.PayloadCrc32 = 0; // Not relevant for the combined log
    logHeader.HeaderCrc32 = 0; // Zero out before calculation
    logHeader.HeaderCrc32 = Crc32(&logHeader, sizeof(logHeader)); // Recalculate CRC

    // Combine header and PCM data
    std::vector<uint8_t> logBuffer;
    logBuffer.resize(sizeof(logHeader) + pcmData.size());
    memcpy(logBuffer.data(), &logHeader, sizeof(logHeader));
    memcpy(logBuffer.data() + sizeof(logHeader), pcmData.data(), pcmData.size());

    // Create hex string
    std::wstringstream wss;
    wss << L"Decoded Block " << header.BlockId << L": ";
    for (size_t i = 0; i < logBuffer.size(); ++i) {
        wss << std::hex << std::setw(2) << std::setfill(L'0') << static_cast<int>(logBuffer[i]);
        if (i < logBuffer.size() - 1) {
            wss << L" ";
        }
    }

    DebugLog(wss.str().c_str());
}
