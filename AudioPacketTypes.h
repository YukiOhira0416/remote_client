#pragma once
#include <cstdint>

// AudioPacketHeader v1 (64 bytes, little-endian)
// This struct is packed to ensure it matches the exact 64-byte layout of the network protocol.
#pragma pack(push, 1)
struct AudioPacketHeader {
    char     Magic[4];           // 0-3: "AUDP"
    uint8_t  Version;            // 4: Constant 1
    uint8_t  HeaderSize;         // 5: Constant 64
    uint8_t  Flags;              // 6: Bit flags (IS_PARITY, etc.)
    uint8_t  Reserved0;          // 7: 0
    uint32_t SessionId;          // 8-11: Server session ID
    uint32_t BlockId;            // 12-15: Block sequence number
    uint16_t ShardIndex;         // 16-17: 0 to k+m-1
    uint16_t ShardTotal;         // 18-19: k+m
    uint16_t K;                  // 20-21: k
    uint16_t M;                  // 22-23: m
    uint16_t ShardBytes;         // 24-25: Size of each shard's payload
    uint16_t PcmBytesInBlock;    // 26-27: Constant 1920
    uint16_t BlockSamplesPerCh;  // 28-29: Constant 480
    uint16_t Reserved1;          // 30-31: 0
    uint32_t SampleRate;         // 32-35: Constant 48000
    uint16_t Channels;           // 36-37: Constant 2
    uint16_t BitsPerSample;      // 38-39: Constant 16
    uint64_t CaptureTimestampNs; // 40-47: Server-side timestamp
    uint32_t BlockCrc32;         // 48-51: CRC32 of original 1920 PCM bytes
    uint32_t PayloadCrc32;       // 52-55: CRC32 of the shard payload
    uint32_t HeaderCrc32;        // 56-59: CRC32 of this header (with this field set to 0)
    uint32_t Reserved2;          // 60-63: 0
};
#pragma pack(pop)

// Sanity check to ensure the struct is exactly 64 bytes.
static_assert(sizeof(AudioPacketHeader) == 64, "AudioPacketHeader must be 64 bytes");
