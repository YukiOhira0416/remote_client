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
    std::vector<std::vector<uint8_t>>& parityShards,
    size_t& shard_len_out,
    int k,
    int m)
{
    if (!original_data || original_data_len == 0 || k <= 0 || m <= 0) {
        DebugLog(L"EncodeFEC_ISAL: Invalid arguments.");
        return false;
    }

    size_t shard_len = (original_data_len + static_cast<size_t>(k) - 1) / static_cast<size_t>(k);
    shard_len = (shard_len + 63) / 64 * 64;
    shard_len_out = shard_len;

    if (shard_len == 0 || shard_len > static_cast<size_t>(std::numeric_limits<int>::max())) {
        DebugLog(L"EncodeFEC_ISAL: Computed shard length is out of range.");
        return false;
    }

    const size_t padded_data_len = shard_len * static_cast<size_t>(k);
    std::vector<uint8_t> padded_data(padded_data_len, 0);
    std::memcpy(padded_data.data(), original_data, original_data_len);

    std::vector<uint8_t*> data_ptrs(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        data_ptrs[static_cast<size_t>(i)] = padded_data.data() + static_cast<size_t>(i) * shard_len;
    }

    parityShards.resize(static_cast<size_t>(m));
    std::vector<uint8_t*> parity_ptrs(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
        parityShards[static_cast<size_t>(i)].assign(shard_len, 0);
        parity_ptrs[static_cast<size_t>(i)] = parityShards[static_cast<size_t>(i)].data();
    }

    const size_t matrix_rows = static_cast<size_t>(k) + static_cast<size_t>(m);
    std::vector<uint8_t> encode_matrix(matrix_rows * static_cast<size_t>(k));
    gf_gen_cauchy1_matrix(encode_matrix.data(), static_cast<int>(matrix_rows), k);
    uint8_t* parity_matrix = encode_matrix.data() + static_cast<size_t>(k) * static_cast<size_t>(k);
    ec_encode_data(static_cast<int>(shard_len), k, m, parity_matrix, data_ptrs.data(), parity_ptrs.data());

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
        DebugLog(L"DecodeFEC_ISAL Error: Invalid Reed-Solomon parameters.");
        return false;
    }

    if (receivedShards.size() < static_cast<size_t>(k)) {
        DebugLog(L"DecodeFEC_ISAL Error: Not enough shards received (" + std::to_wstring(receivedShards.size()) + L" < " + std::to_wstring(k) + L")");
        return false;
    }

    size_t shard_len = 0;
    if (!receivedShards.empty()) {
        shard_len = receivedShards.begin()->second.size();
    }
    if (shard_len == 0 || shard_len > static_cast<size_t>(std::numeric_limits<int>::max())) {
        DebugLog(L"DecodeFEC_ISAL Error: Invalid shard length (" + std::to_wstring(shard_len) + L")");
        return false;
    }

    const int n = k + m;
    std::vector<std::vector<uint8_t>> temp_shards(static_cast<size_t>(n));
    std::vector<uint8_t*> fragment_ptrs(static_cast<size_t>(n), nullptr);

    std::vector<int> available_indices;
    available_indices.reserve(static_cast<size_t>(k));

    for (int i = 0; i < n; ++i) {
        auto it = receivedShards.find(static_cast<uint32_t>(i));
        if (it != receivedShards.end()) {
            if (it->second.size() != shard_len) {
                DebugLog(L"DecodeFEC_ISAL Error: Shard " + std::to_wstring(i) + L" has unexpected length.");
                return false;
            }
            temp_shards[static_cast<size_t>(i)] = it->second;
            fragment_ptrs[static_cast<size_t>(i)] = temp_shards[static_cast<size_t>(i)].data();
            if (available_indices.size() < static_cast<size_t>(k)) {
                available_indices.push_back(i);
            }
        } else {
            temp_shards[static_cast<size_t>(i)].assign(shard_len, 0);
            fragment_ptrs[static_cast<size_t>(i)] = temp_shards[static_cast<size_t>(i)].data();
        }
    }

    if (available_indices.size() < static_cast<size_t>(k)) {
        DebugLog(L"DecodeFEC_ISAL Error: Could not gather enough shards for inversion.");
        return false;
    }

    const size_t matrix_rows = static_cast<size_t>(k) + static_cast<size_t>(m);
    std::vector<uint8_t> encode_matrix(matrix_rows * static_cast<size_t>(k));
    gf_gen_cauchy1_matrix(encode_matrix.data(), static_cast<int>(matrix_rows), k);
    const uint8_t* parity_matrix = encode_matrix.data() + static_cast<size_t>(k) * static_cast<size_t>(k);

    auto fill_generator_row = [&](int shard_index, uint8_t* dest) {
        if (shard_index < k) {
            for (int j = 0; j < k; ++j) {
                dest[j] = (shard_index == j) ? 1 : 0;
            }
        } else {
            std::memcpy(dest, parity_matrix + static_cast<size_t>(shard_index - k) * static_cast<size_t>(k), static_cast<size_t>(k));
        }
    };

    std::vector<uint8_t> sub_matrix(static_cast<size_t>(k) * static_cast<size_t>(k));
    for (int row = 0; row < k; ++row) {
        fill_generator_row(available_indices[static_cast<size_t>(row)], sub_matrix.data() + static_cast<size_t>(row) * static_cast<size_t>(k));
    }

    std::vector<uint8_t> inverted_matrix(static_cast<size_t>(k) * static_cast<size_t>(k));
    std::vector<uint8_t> temp_matrix = sub_matrix;
    if (gf_invert_matrix(temp_matrix.data(), inverted_matrix.data(), k) != 0) {
        DebugLog(L"DecodeFEC_ISAL Error: Could not invert matrix.");
        return false;
    }

    std::vector<uint8_t*> source_ptrs(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        source_ptrs[static_cast<size_t>(i)] = fragment_ptrs[static_cast<size_t>(available_indices[static_cast<size_t>(i)])];
    }

    std::vector<std::vector<uint8_t>> recovered_data(static_cast<size_t>(k), std::vector<uint8_t>(shard_len));
    std::vector<uint8_t*> recovered_ptrs(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        recovered_ptrs[static_cast<size_t>(i)] = recovered_data[static_cast<size_t>(i)].data();
    }

    ec_encode_data(static_cast<int>(shard_len), k, k, inverted_matrix.data(), source_ptrs.data(), recovered_ptrs.data());

    for (int i = 0; i < k; ++i) {
        temp_shards[static_cast<size_t>(i)] = std::move(recovered_data[static_cast<size_t>(i)]);
    }

    decodedData.clear();
    decodedData.reserve(static_cast<size_t>(k) * shard_len);
    for (int i = 0; i < k; ++i) {
        decodedData.insert(decodedData.end(), temp_shards[static_cast<size_t>(i)].begin(), temp_shards[static_cast<size_t>(i)].end());
    }

    if (decodedData.size() > originalDataLen) {
        decodedData.resize(originalDataLen);
    }

    return true;
}
