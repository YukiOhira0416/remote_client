#pragma once
#include <vector>
#include <cstdint>

// A simple structure to hold a decoded block of audio data.
struct DecodedAudioBlock {
    uint32_t blockId;
    std::vector<uint8_t> pcmData; // Should be 1920 bytes
};

class JitterBuffer {
public:
    virtual ~JitterBuffer() = default;

    // Adds a decoded block to the buffer.
    virtual void AddBlock(DecodedAudioBlock&& block) = 0;

    // Gets the next block in sequence for playback.
    // If the block is not available, this should handle it (e.g., return silence).
    virtual bool GetNextBlock(std::vector<uint8_t>& out_pcm, uint32_t expectedBlockId) = 0;

    // Peeks at the ID of the earliest block in the buffer.
    // Returns true if a block is available and sets out_blockId.
    virtual bool PeekNextBlockId(uint32_t& out_blockId) = 0;
};
