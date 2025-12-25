#pragma once

#include <cstdint>

// Packet types
enum : uint8_t {
    AUDIO_PACKET_TYPE_FORMAT    = 1,
    AUDIO_PACKET_TYPE_SHARD     = 2,
    AUDIO_PACKET_TYPE_HEARTBEAT = 3,
};

#pragma pack(push, 1)
struct AudioUdpHeader {
    uint32_t magic_be;            // 'RAUD' (0x52415544)
    uint16_t version_be;          // 1
    uint16_t header_bytes_be;     // sizeof(AudioUdpHeader)

    uint8_t  packet_type;         // 1/2/3
    uint8_t  flags;               // bit0: format_changed 等
    uint16_t reserved0_be;        // 0

    uint64_t stream_id_be;        // stream identifier

    uint32_t block_id_be;         // AUDIO_SHARD対象のブロックID
    uint16_t shard_index_be;      // 0..(K+M-1) / 非該当は 0xFFFF
    uint8_t  k;                   // 14..19
    uint8_t  m;                   // 1..8
    uint16_t shard_bytes_be;      // このパケットのシャードpayload長

    uint32_t original_pcm_bytes_be; // 生PCM長

    uint64_t audio_capture_ts_ns_be; // サーバsystem_clock(ns)
    uint64_t video_capture_ts_ms_be; // サーバsystem_clock(ms)
};

struct AudioFormatPayload {
    uint32_t sample_rate_be;      // WASAPI mix format
    uint16_t channels_be;         // 2
    uint16_t bits_per_sample_be;  // 32 (float想定)
    uint16_t block_ms_be;         // AUDIO_BLOCK_MS (20)
    uint16_t bytes_per_frame_be;  // channels * (bits/8)
    uint32_t frames_per_block_be; // sample_rate * block_ms / 1000
    uint32_t channel_mask_be;     // WAVEFORMATEXTENSIBLE相当
    uint32_t sample_format_be;    // 1=float32, 2=int16 など
};
#pragma pack(pop)
