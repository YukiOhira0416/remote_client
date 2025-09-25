// ISA-L ベースの Reed-Solomon 実装

// Jerasure/GFcomplete 由来の include は削除。
#include <isa-l/include/erasure_code.h>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "Globals.h"
#include "ReedSolomon.h"
#include "DebugLog.h"

bool EncodeFEC_ISAL(
    const uint8_t* original_data,
    size_t original_data_len,
    std::vector<std::vector<uint8_t>>& dataShards,
    std::vector<std::vector<uint8_t>>& parityShards,
    size_t& shard_len_out,
    int k,
    int m)
{
    if (!original_data || original_data_len == 0 || k <= 0 || m < 0) { // m=0 is valid
        DebugLog(L"EncodeFEC_ISAL: Invalid arguments (data=" + std::to_wstring(reinterpret_cast<uintptr_t>(original_data))
                 + L", len=" + std::to_wstring(original_data_len) 
                 + L", k=" + std::to_wstring(k) 
                 + L", m=" + std::to_wstring(m) + L").");
        return false;
    }

    // Calculate shard length with padding to be a multiple of 64 bytes (for AVX performance)
    size_t shard_len = (original_data_len + static_cast<size_t>(k) - 1) / static_cast<size_t>(k);
    shard_len = (shard_len + 63) & ~63; // Align up to nearest multiple of 64
    shard_len_out = shard_len;

    if (shard_len == 0) {
        DebugLog(L"EncodeFEC_ISAL: Calculated shard length is zero.");
        return false;
    }

    const size_t padded_data_len = shard_len * static_cast<size_t>(k);
    std::vector<uint8_t> padded_data(padded_data_len, 0);
    std::memcpy(padded_data.data(), original_data, original_data_len);

    dataShards.assign(k, std::vector<uint8_t>(shard_len));
    std::vector<uint8_t*> data_ptrs(k);
    for (int i = 0; i < k; ++i) {
        std::memcpy(dataShards[i].data(), padded_data.data() + i * shard_len, shard_len);
        data_ptrs[i] = dataShards[i].data();
    }

    if (m == 0) {
        parityShards.clear();
        return true;
    }

    parityShards.assign(m, std::vector<uint8_t>(shard_len, 0));
    std::vector<uint8_t*> parity_ptrs(m);
    for (int i = 0; i < m; ++i) {
        parity_ptrs[i] = parityShards[i].data();
    }

    // Generate encode matrix
    const size_t matrix_size = static_cast<size_t>(k + m) * k;
    std::vector<uint8_t> encode_matrix(matrix_size);
    gf_gen_cauchy1_matrix(encode_matrix.data(), k + m, k);

    // Get the parity-generating part of the matrix
    uint8_t* parity_matrix_ptr = encode_matrix.data() + static_cast<size_t>(k) * k;

    // Generate parity shards
    ec_encode_data(static_cast<int>(shard_len), k, m, parity_matrix_ptr, data_ptrs.data(), parity_ptrs.data());

    return true;
}

bool DecodeFEC_ISAL(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards,
    int k,
    int m,
    uint32_t originalDataLen,
    std::vector<uint8_t>& decodedData)
{
    if (k <= 0 || m < 0) {
        DebugLog(L"DecodeFEC_ISAL Error: Invalid Reed-Solomon parameters (k=" + std::to_wstring(k) + L", m=" + std::to_wstring(m) + L").");
        return false;
    }

    if (receivedShards.size() < static_cast<size_t>(k)) {
        // This is a normal condition, not an error, so a less alarming log is better.
        // DebugLog(L"DecodeFEC_ISAL Info: Not enough shards received (" + std::to_wstring(receivedShards.size()) + L" < " + std::to_wstring(k) + L")");
        return false;
    }

    size_t shard_len = 0;
    if (!receivedShards.empty()) {
        shard_len = receivedShards.begin()->second.size();
    }

    if (shard_len == 0) {
        DebugLog(L"DecodeFEC_ISAL Error: Invalid shard length (0).");
        return false;
    }

    const int n = k + m;
    std::vector<uint8_t*> fragment_ptrs(n, nullptr);
    std::vector<int> available_indices;
    available_indices.reserve(receivedShards.size());
    
    // Create a temporary copy of shards to avoid modifying the input map's data
    std::map<uint32_t, std::vector<uint8_t>> temp_shards = receivedShards;

    for (auto const& [index, data] : temp_shards) {
        if (index < static_cast<uint32_t>(n)) {
            if (data.size() != shard_len) {
                 DebugLog(L"DecodeFEC_ISAL Error: Shard " + std::to_wstring(index) + L" has inconsistent length.");
                 return false;
            }
            fragment_ptrs[index] = temp_shards[index].data();
            available_indices.push_back(index);
        }
    }

    if (available_indices.size() < static_cast<size_t>(k)) {
        DebugLog(L"DecodeFEC_ISAL Error: Not enough valid shards for decoding.");
        return false;
    }

    // If all k data shards are present, just reconstruct
    bool all_data_shards_present = true;
    for (int i = 0; i < k; ++i) {
        if (temp_shards.find(i) == temp_shards.end()) {
            all_data_shards_present = false;
            break;
        }
    }

    if (all_data_shards_present) {
        decodedData.clear();
        decodedData.reserve(static_cast<size_t>(k) * shard_len);
        for (int i = 0; i < k; ++i) {
            decodedData.insert(decodedData.end(), temp_shards[i].begin(), temp_shards[i].end());
        }
        if (decodedData.size() > originalDataLen) {
            decodedData.resize(originalDataLen);
        }
        return true;
    }

    // If we need to reconstruct from parity
    if (m == 0) {
        DebugLog(L"DecodeFEC_ISAL Error: Missing data shards but no parity shards available (m=0).");
        return false;
    }

    std::vector<uint8_t> encode_matrix(static_cast<size_t>(n) * k);
    gf_gen_cauchy1_matrix(encode_matrix.data(), n, k);

    std::vector<uint8_t> sub_matrix(static_cast<size_t>(k) * k);
    std::vector<uint8_t*> source_ptrs(k);
    
    for (int i = 0; i < k; ++i) {
        int shard_idx = available_indices[i];
        source_ptrs[i] = fragment_ptrs[shard_idx];
        if (shard_idx < k) { // It's a data shard
            for (int j = 0; j < k; ++j) {
                sub_matrix[i * k + j] = (shard_idx == j) ? 1 : 0;
            }
        } else { // It's a parity shard
            memcpy(sub_matrix.data() + i * k, encode_matrix.data() + shard_idx * k, k);
        }
    }

    std::vector<uint8_t> inverted_matrix(static_cast<size_t>(k) * k);
    if (gf_invert_matrix(sub_matrix.data(), inverted_matrix.data(), k) != 0) {
        DebugLog(L"DecodeFEC_ISAL Error: Could not invert matrix. Shards might be corrupted or linearly dependent.");
        return false;
    }
    
    std::vector<uint8_t*> recovered_ptrs(k);
    std::vector<std::vector<uint8_t>> recovered_data(k, std::vector<uint8_t>(shard_len));
    
    std::vector<int> missing_indices;
    for (int i = 0; i < k; ++i) {
        bool found = false;
        for(int idx : available_indices) {
            if (i == idx) {
                found = true;
                break;
            }
        }
        if (!found) {
            missing_indices.push_back(i);
        }
    }

    for (size_t i = 0; i < missing_indices.size(); ++i) {
        recovered_ptrs[i] = recovered_data[i].data();
    }

    std::vector<uint8_t> decode_matrix(static_cast<size_t>(k) * k);
    for (size_t i = 0; i < missing_indices.size(); ++i) {
        int missing_idx = missing_indices[i];
        memcpy(decode_matrix.data() + i * k, inverted_matrix.data() + missing_idx * k, k);
    }
    
    ec_encode_data(static_cast<int>(shard_len), k, (int)missing_indices.size(), decode_matrix.data(), source_ptrs.data(), recovered_ptrs.data());

    for (size_t i = 0; i < missing_indices.size(); ++i) {
        temp_shards[missing_indices[i]] = recovered_data[i];
    }
    
    decodedData.clear();
    decodedData.reserve(static_cast<size_t>(k) * shard_len);
    for (int i = 0; i < k; ++i) {
        if (temp_shards.find(i) == temp_shards.end()) {
             DebugLog(L"DecodeFEC_ISAL Internal Error: Failed to reconstruct data shard " + std::to_wstring(i));
             return false;
        }
        decodedData.insert(decodedData.end(), temp_shards[i].begin(), temp_shards[i].end());
    }

    if (decodedData.size() > originalDataLen) {
        decodedData.resize(originalDataLen);
    }

    return true;
}
