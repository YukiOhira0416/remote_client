// ReadSolomon.cpp などに追加

#include <vector>
#include <map>     // ※ std::map を使用するためにインクルード
#include <unordered_map> // ※ std::unordered_map を使用するためにインクルード
#include <cstdint>
#include <cstring> // for memcpy
#include <algorithm> // for std::max
#include "DebugLog.h"
#include <gf_complete.h>
#include <jerasure.h>
#include "ReedSolomon.h" // ※ ヘッダーファイルのインクルードを追加
#include "Globals.h"

// Static/file-local FECContext for reuse (no public header changes)
namespace {
    struct FECContext {
        int k{-1}, m{-1};
        size_t shard_len{0};
        std::vector<char*> data_ptrs;
        std::vector<char*> coding_ptrs;
        std::vector<int>   erasures;      // size m+1
        std::vector<uint8_t> parity_scratch; // m * shard_len
        void ensure(int k_, int m_, size_t shard_len_) {
            if (k != k_ || m != m_ || shard_len != shard_len_) {
                k = k_; m = m_; shard_len = shard_len_;
                data_ptrs.assign(k, nullptr);
                coding_ptrs.assign(m, nullptr);
                erasures.assign(m + 1, -1);
                parity_scratch.resize(static_cast<size_t>(m) * shard_len);
            }
        }
    };
    static thread_local FECContext g_fecctx; // reuse across calls
}

// Jerasure を使った FEC エンコード関数
bool EncodeFEC_Jerasure(
    const uint8_t* data,           // パディング済み入力データ
    size_t padded_data_len,        // パディング済みデータ長
    std::vector<std::vector<uint8_t>>& parityShards, // 出力パリティシャード
    int k,
    int m,
    int* matrix,                   // jerasure_matrix_encode で生成した行列
    size_t shard_len)              // シャード長
{
    if (!data || !matrix || k <= 0 || m <= 0 || shard_len == 0 || padded_data_len != k * shard_len) {
        DebugLog(L"EncodeFEC_Jerasure: Invalid arguments.");
        return false;
    }

    // Ensure reusable arrays are sized correctly.
    g_fecctx.ensure(k, m, shard_len);

    // Grow-only buffer strategy for parity shards to avoid re-allocation.
    if (parityShards.size() != static_cast<size_t>(m)) {
        parityShards.resize(m);
    }
    for (int i = 0; i < m; ++i) {
        if (parityShards[i].size() != shard_len) {
            parityShards[i].resize(shard_len);
        }
        g_fecctx.coding_ptrs[i] = reinterpret_cast<char*>(parityShards[i].data());
    }

    // Data shard pointers point directly to the input data slices.
    for (int i = 0; i < k; ++i) {
        g_fecctx.data_ptrs[i] = const_cast<char*>(reinterpret_cast<const char*>(data)) + i * shard_len;
    }

    // Jerasure でエンコードを実行 (ビット行列を使用, w=8)
    jerasure_bitmatrix_encode(k, m, 8, matrix, g_fecctx.data_ptrs.data(), g_fecctx.coding_ptrs.data(), shard_len, static_cast<int>(shard_len / 8));

    // Jerasure 関数は通常エラーコードを返さないため、成功とみなす
    return true;
}


// Jerasure を使った FEC デコード関数
bool DecodeFEC_Jerasure(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards, // 受信シャード
    int k,
    int m,
    uint32_t originalDataLen,       // 元データ長
    std::vector<uint8_t>& decodedData,  // 出力先
    const int* bitmatrix)             // ※ デコードに使用するビット行列
{
    if (receivedShards.size() < static_cast<size_t>(k)) {
        DebugLog(L"DecodeFEC_Jerasure Error: Not enough shards received (" +
                 std::to_wstring(receivedShards.size()) + L" < " + std::to_wstring(k) + L")");
        return false;
    }
    if (!bitmatrix) {
        DebugLog(L"DecodeFEC_Jerasure Error: Jerasure bitmatrix not provided for decoding.");
        return false;
    }

    // === shard_len を堅牢に決定 ===
    size_t shard_len = 0;
    if (!receivedShards.empty()) {
        // 長さの頻度を数え、多数派（mode）を採用。引き分け時は最大値。
        std::unordered_map<size_t, size_t> freq;
        size_t mode_len = 0, mode_cnt = 0, max_len = 0;
        for (auto& kv : receivedShards) {
            size_t len = kv.second.size();
            if (len > 0) {
                size_t c = ++freq[len];
                if (c > mode_cnt) { mode_cnt = c; mode_len = len; }
                if (len > max_len) max_len = len;
            }
        }
        shard_len = (mode_cnt > 0) ? mode_len : 0;
        if (shard_len == 0) shard_len = max_len; // 全部0は異常系だが一応最大にフォールバック
    }
    if (shard_len == 0) {
        DebugLog(L"DecodeFEC_Jerasure Error: Invalid shard length (0)");
        return false;
    }
    if (static_cast<uint64_t>(k) * static_cast<uint64_t>(shard_len) < static_cast<uint64_t>(originalDataLen)) {
        DebugLog(L"DecodeFEC_Jerasure Error: originalDataLen exceeds k*shard_len.");
        return false;
    }

    // Ensure reusable arrays
    g_fecctx.ensure(k, m, shard_len);

    // 1) Pre-size output once (k * shard_len)
    decodedData.clear();
    decodedData.resize(static_cast<size_t>(k) * shard_len);

    // 2) Data shard pointers point directly into decodedData.
    for (int i = 0; i < k; ++i) {
        g_fecctx.data_ptrs[i] = reinterpret_cast<char*>(decodedData.data() + static_cast<size_t>(i) * shard_len);
    }

    // 3) Coding shard pointers point into one reusable parity_scratch block.
    for (int j = 0; j < m; ++j) {
        g_fecctx.coding_ptrs[j] = reinterpret_cast<char*>(g_fecctx.parity_scratch.data() + static_cast<size_t>(j) * shard_len);
    }

    // 4) Fill received data shards directly into decodedData; record erasures for missing shards.
    int erasure_count = 0;
    const int n = k + m;
    // Iterate through all possible shard indices to identify missing ones.
    for (int i = 0; i < n; ++i) {
        auto it = receivedShards.find(i);
        const bool shard_is_present = (it != receivedShards.end() && it->second.size() == shard_len);

        if (i < k) { // This is a data shard
            if (shard_is_present) {
                // Copy received shard data into its final destination in the output buffer. This is the ONLY copy.
                std::memcpy(decodedData.data() + static_cast<size_t>(i) * shard_len, it->second.data(), shard_len);
            } else {
                // This data shard is missing. Record it as an erasure.
                if (erasure_count < m) {
                    g_fecctx.erasures[erasure_count++] = i;
                    // No need to zero-fill the buffer; Jerasure will write the recovered data here.
                } else {
                    // This should not happen if receivedShards.size() >= k, but as a safeguard:
                    DebugLog(L"DecodeFEC_Jerasure Error: Too many erasures detected (" +
                             std::to_wstring(erasure_count + 1) + L" > " + std::to_wstring(m) + L")");
                    return false;
                }
            }
        } else { // This is a coding (parity) shard
            if (shard_is_present) {
                // We have a parity shard. Copy it into the reusable parity scratch buffer.
                std::memcpy(g_fecctx.parity_scratch.data() + static_cast<size_t>(i - k) * shard_len, it->second.data(), shard_len);
            } else {
                // This coding shard is missing. Record it as an erasure.
                if (erasure_count < m) {
                    g_fecctx.erasures[erasure_count++] = i;
                } else {
                    DebugLog(L"DecodeFEC_Jerasure Error: Too many erasures detected (" +
                             std::to_wstring(erasure_count + 1) + L" > " + std::to_wstring(m) + L")");
                    return false;
                }
            }
        }
    }
    g_fecctx.erasures[erasure_count] = -1; // Null-terminate the erasures list for Jerasure.

    // packetsize は最低 1 に丸める（極端な shard_len にも耐える）
    const int packetsize = std::max(1, static_cast<int>(shard_len / 8));

    // 5) Decode — recovered bytes go straight into decodedData via data_ptrs.
    const int ret = jerasure_bitmatrix_decode(
        k, m, 8, const_cast<int*>(bitmatrix),
        /*row_k_ones*/ 0,
        g_fecctx.erasures.data(),
        g_fecctx.data_ptrs.data(),
        g_fecctx.coding_ptrs.data(),
        static_cast<int>(shard_len),
        packetsize
    );
    if (ret == -1) {
        DebugLog(L"DecodeFEC_Jerasure Error: jerasure_matrix_decode failed (matrix not invertible?).");
        return false;
    }

    // 6) Trim to original length (preserve timing/logs around this function)
    if (decodedData.size() > originalDataLen) {
        decodedData.resize(originalDataLen);
    } else if (decodedData.size() < originalDataLen) {
        DebugLog(L"DecodeFEC_Jerasure Warning: Reconstructed data shorter than originalDataLen. Size: " +
                 std::to_wstring(decodedData.size()) + L", Original: " + std::to_wstring(originalDataLen));
    }
    return true;
}
