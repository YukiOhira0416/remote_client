// Reed-Solomon implementation without ISA-L dependency

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "Globals.h"
#include "ReedSolomon.h"
#include "DebugLog.h"

namespace {
constexpr int kFieldSize = 256;
constexpr int kFieldOrder = kFieldSize - 1;  // 255
constexpr uint16_t kPrimitivePolynomial = 0x11d;  // x^8 + x^4 + x^3 + x^2 + 1

std::array<uint8_t, kFieldSize * 2> g_expTable{};
std::array<uint8_t, kFieldSize> g_logTable{};
std::once_flag g_initTablesFlag;

void InitializeTables() {
    uint16_t value = 1;
    for (int i = 0; i < kFieldOrder; ++i) {
        uint8_t byteValue = static_cast<uint8_t>(value);
        g_expTable[static_cast<size_t>(i)] = byteValue;
        g_logTable[byteValue] = static_cast<uint8_t>(i);
        value <<= 1;
        if (value & kFieldSize) {
            value ^= kPrimitivePolynomial;
        }
    }

    // Extend the exponent table so we can avoid taking modulus in hot paths.
    for (int i = kFieldOrder; i < static_cast<int>(g_expTable.size()); ++i) {
        g_expTable[static_cast<size_t>(i)] = g_expTable[static_cast<size_t>(i - kFieldOrder)];
    }

    g_logTable[0] = 0;
}

inline void EnsureTables() {
    std::call_once(g_initTablesFlag, InitializeTables);
}

inline uint8_t GfMul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    EnsureTables();
    int sum = g_logTable[a] + g_logTable[b];
    if (sum >= kFieldOrder) {
        sum -= kFieldOrder;
    }
    return g_expTable[static_cast<size_t>(sum)];
}

inline uint8_t GfInv(uint8_t value) {
    if (value == 0) {
        return 0;
    }
    EnsureTables();
    int logValue = g_logTable[value];
    int inverseLog = kFieldOrder - logValue;
    if (inverseLog >= kFieldOrder) {
        inverseLog -= kFieldOrder;
    }
    return g_expTable[static_cast<size_t>(inverseLog)];
}

bool InvertMatrix(std::vector<uint8_t>& matrix, std::vector<uint8_t>& inverse, int size) {
    EnsureTables();

    inverse.assign(static_cast<size_t>(size) * static_cast<size_t>(size), 0);
    for (int i = 0; i < size; ++i) {
        inverse[static_cast<size_t>(i) * static_cast<size_t>(size) + static_cast<size_t>(i)] = 1;
    }

    for (int col = 0; col < size; ++col) {
        int pivot = col;
        while (pivot < size && matrix[static_cast<size_t>(pivot) * static_cast<size_t>(size) + static_cast<size_t>(col)] == 0) {
            ++pivot;
        }
        if (pivot == size) {
            return false;
        }

        if (pivot != col) {
            for (int j = 0; j < size; ++j) {
                std::swap(
                    matrix[static_cast<size_t>(pivot) * static_cast<size_t>(size) + static_cast<size_t>(j)],
                    matrix[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)]);
                std::swap(
                    inverse[static_cast<size_t>(pivot) * static_cast<size_t>(size) + static_cast<size_t>(j)],
                    inverse[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)]);
            }
        }

        uint8_t pivotValue = matrix[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(col)];
        uint8_t pivotInv = GfInv(pivotValue);
        if (pivotInv == 0) {
            return false;
        }

        for (int j = 0; j < size; ++j) {
            matrix[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)] =
                GfMul(matrix[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)], pivotInv);
            inverse[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)] =
                GfMul(inverse[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)], pivotInv);
        }

        for (int row = 0; row < size; ++row) {
            if (row == col) {
                continue;
            }
            uint8_t factor = matrix[static_cast<size_t>(row) * static_cast<size_t>(size) + static_cast<size_t>(col)];
            if (factor == 0) {
                continue;
            }
            for (int j = 0; j < size; ++j) {
                matrix[static_cast<size_t>(row) * static_cast<size_t>(size) + static_cast<size_t>(j)] ^=
                    GfMul(factor, matrix[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)]);
                inverse[static_cast<size_t>(row) * static_cast<size_t>(size) + static_cast<size_t>(j)] ^=
                    GfMul(factor, inverse[static_cast<size_t>(col) * static_cast<size_t>(size) + static_cast<size_t>(j)]);
            }
        }
    }

    return true;
}

bool BuildGeneratorMatrix(int k, int m, std::vector<uint8_t>& parityMatrix) {
    if (m <= 0) {
        parityMatrix.clear();
        return true;
    }

    if (k + m > kFieldOrder) {
        return false;
    }

    EnsureTables();

    std::vector<uint8_t> vandermonde(static_cast<size_t>(k) * static_cast<size_t>(k));
    for (int row = 0; row < k; ++row) {
        uint8_t value = 1;
        uint8_t evaluation = g_expTable[static_cast<size_t>(row)];
        for (int col = 0; col < k; ++col) {
            vandermonde[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(col)] = value;
            value = GfMul(value, evaluation);
        }
    }

    std::vector<uint8_t> vandermondeCopy = vandermonde;
    std::vector<uint8_t> vandermondeInv;
    if (!InvertMatrix(vandermondeCopy, vandermondeInv, k)) {
        return false;
    }

    parityMatrix.assign(static_cast<size_t>(m) * static_cast<size_t>(k), 0);

    for (int row = 0; row < m; ++row) {
        uint8_t evaluation = g_expTable[static_cast<size_t>(k + row)];
        std::vector<uint8_t> evaluationRow(static_cast<size_t>(k));
        uint8_t value = 1;
        for (int col = 0; col < k; ++col) {
            evaluationRow[static_cast<size_t>(col)] = value;
            value = GfMul(value, evaluation);
        }

        for (int col = 0; col < k; ++col) {
            uint8_t accumulator = 0;
            for (int i = 0; i < k; ++i) {
                accumulator ^= GfMul(
                    evaluationRow[static_cast<size_t>(i)],
                    vandermondeInv[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(col)]);
            }
            parityMatrix[static_cast<size_t>(row) * static_cast<size_t>(k) + static_cast<size_t>(col)] = accumulator;
        }
    }

    return true;
}

void MultiplyShards(
    size_t shardLength,
    int rowCount,
    int columnCount,
    const std::vector<uint8_t>& matrix,
    const std::vector<uint8_t*>& inputs,
    const std::vector<uint8_t*>& outputs) {
    for (int row = 0; row < rowCount; ++row) {
        uint8_t* destination = outputs[static_cast<size_t>(row)];
        if (!destination) {
            continue;
        }
        std::fill(destination, destination + shardLength, 0);
        const uint8_t* coefficients = matrix.data() + static_cast<size_t>(row) * static_cast<size_t>(columnCount);
        for (int column = 0; column < columnCount; ++column) {
            uint8_t coefficient = coefficients[static_cast<size_t>(column)];
            if (coefficient == 0) {
                continue;
            }
            const uint8_t* source = inputs[static_cast<size_t>(column)];
            if (!source) {
                continue;
            }
            for (size_t index = 0; index < shardLength; ++index) {
                destination[index] ^= GfMul(coefficient, source[index]);
            }
        }
    }
}

}  // namespace

bool EncodeFEC_ISAL(
    const uint8_t* original_data,
    size_t original_data_len,
    std::vector<std::vector<uint8_t>>& parityShards,
    size_t& shard_len_out,
    int k,
    int m) {
    if (!original_data || original_data_len == 0 || k <= 0 || m <= 0) {
        DebugLog(L"EncodeFEC_ISAL: Invalid arguments.");
        return false;
    }

    if (k + m > kFieldOrder) {
        DebugLog(L"EncodeFEC_ISAL: RS_K + RS_M exceeds supported field size (255).");
        return false;
    }

    size_t shard_len = (original_data_len + static_cast<size_t>(k) - 1) / static_cast<size_t>(k);
    shard_len = (shard_len + 63) / 64 * 64;
    shard_len_out = shard_len;

    if (shard_len == 0 || shard_len > static_cast<size_t>(std::numeric_limits<int>::max())) {
        DebugLog(L"EncodeFEC_ISAL: Computed shard length is out of range.");
        return false;
    }

    std::vector<uint8_t> encodeMatrix;
    if (!BuildGeneratorMatrix(k, m, encodeMatrix)) {
        DebugLog(L"EncodeFEC_ISAL: Failed to build generator matrix.");
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

    MultiplyShards(shard_len, m, k, encodeMatrix, data_ptrs, parity_ptrs);

    return true;
}

bool DecodeFEC_ISAL(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards,
    int k,
    int m,
    uint32_t originalDataLen,
    std::vector<uint8_t>& decodedData) {
    if (k <= 0 || m < 0) {
        DebugLog(L"DecodeFEC_ISAL Error: Invalid Reed-Solomon parameters.");
        return false;
    }

    if (k + m > kFieldOrder) {
        DebugLog(L"DecodeFEC_ISAL Error: RS_K + RS_M exceeds supported field size (255).");
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

    std::vector<uint8_t> encodeMatrix;
    if (!BuildGeneratorMatrix(k, m, encodeMatrix)) {
        DebugLog(L"DecodeFEC_ISAL Error: Failed to build generator matrix.");
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

    std::vector<uint8_t> sub_matrix(static_cast<size_t>(k) * static_cast<size_t>(k));
    for (int row = 0; row < k; ++row) {
        int shard_index = available_indices[static_cast<size_t>(row)];
        uint8_t* dest = sub_matrix.data() + static_cast<size_t>(row) * static_cast<size_t>(k);
        if (shard_index < k) {
            for (int col = 0; col < k; ++col) {
                dest[static_cast<size_t>(col)] = (shard_index == col) ? 1 : 0;
            }
        } else {
            int parity_index = shard_index - k;
            if (parity_index < 0 || parity_index >= m) {
                DebugLog(L"DecodeFEC_ISAL Error: Invalid parity shard index.");
                return false;
            }
            const uint8_t* sourceRow = encodeMatrix.data() + static_cast<size_t>(parity_index) * static_cast<size_t>(k);
            std::copy(sourceRow, sourceRow + static_cast<size_t>(k), dest);
        }
    }

    std::vector<uint8_t> inverted_matrix(static_cast<size_t>(k) * static_cast<size_t>(k));
    std::vector<uint8_t> temp_matrix = sub_matrix;
    if (!InvertMatrix(temp_matrix, inverted_matrix, k)) {
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

    MultiplyShards(shard_len, k, k, inverted_matrix, source_ptrs, recovered_ptrs);

    for (int i = 0; i < k; ++i) {
        temp_shards[static_cast<size_t>(i)] = std::move(recovered_data[static_cast<size_t>(i)]);
    }

    decodedData.clear();
    decodedData.reserve(static_cast<size_t>(k) * shard_len);
    for (int i = 0; i < k; ++i) {
        decodedData.insert(
            decodedData.end(),
            temp_shards[static_cast<size_t>(i)].begin(),
            temp_shards[static_cast<size_t>(i)].end());
    }

    if (decodedData.size() > originalDataLen) {
        decodedData.resize(originalDataLen);
    }

    return true;
}
