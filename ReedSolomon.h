#pragma once

#include <vector>
#include <map>
#include <cstdint>

// Decodes a set of received data and parity shards into the original data.
//
// @param receivedShards A map where the key is the shard index (0 to k+m-1) and
//                       the value is the shard data. At least 'k' shards must be present.
// @param k The number of original data shards.
// @param m The number of parity shards.
// @param originalDataLen The exact length of the original data before padding and encoding.
// @param decodedData A vector that will be cleared and filled with the reconstructed original data.
// @return True if decoding was successful, false otherwise.
bool DecodeFEC_ISAL(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards,
    int k,
    int m,
    uint32_t originalDataLen,
    std::vector<uint8_t>& decodedData
);
