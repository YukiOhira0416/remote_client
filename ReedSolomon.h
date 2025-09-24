#pragma once
#include <cstdint>
#include <vector>
#include <map>

// Jerasure/GFcomplete 由来の宣言は全削除すること。
// K/M は getOptimalThreadConfig() → RS_K/RS_M を既存のとおり使用。

// ISA-L 版 API（宣言のみ）
bool EncodeFEC_ISAL(
    const uint8_t* original_data,
    size_t original_data_len,
    std::vector<std::vector<uint8_t>>& parityShards,
    size_t& shard_len_out,
    int k,
    int m);

bool DecodeFEC_ISAL(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards,
    int k,
    int m,
    uint32_t originalDataLen,
    std::vector<uint8_t>& decodedData);
