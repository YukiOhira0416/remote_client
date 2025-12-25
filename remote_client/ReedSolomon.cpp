// ISA-L ベースの Reed-Solomon 実装

// Jerasure/GFcomplete 由来の include は削除。
#include <isa-l/include/erasure_code.h>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <numeric>
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
    // 引数チェック（m=0 は許容）
    if (!original_data || original_data_len == 0 || k <= 0 || m < 0) {
        DebugLog(L"EncodeFEC_ISAL: Invalid args.");
        return false;
    }

    // シャード長（64B アライン）計算
    size_t shard_len = (original_data_len + static_cast<size_t>(k) - 1) / static_cast<size_t>(k);
    shard_len = (shard_len + 63) & ~static_cast<size_t>(63);
    if (shard_len == 0) {
        DebugLog(L"EncodeFEC_ISAL: Calculated shard_len is 0.");
        return false;
    }
    shard_len_out = shard_len;

    // パディングして k * shard_len に拡張
    const size_t padded_len = shard_len * static_cast<size_t>(k);
    std::vector<uint8_t> padded(padded_len, 0);
    std::memcpy(padded.data(), original_data, original_data_len);

    // データシャードを作成
    dataShards.assign(k, std::vector<uint8_t>(shard_len));
    std::vector<uint8_t*> data_ptrs(k);
    for (int i = 0; i < k; ++i) {
        std::memcpy(dataShards[i].data(), padded.data() + static_cast<size_t>(i) * shard_len, shard_len);
        data_ptrs[i] = dataShards[i].data();
    }

    // パリティ不要ならここで終了
    if (m == 0) {
        parityShards.clear();
        return true;
    }

    // パリティシャードを確保
    parityShards.assign(m, std::vector<uint8_t>(shard_len, 0));
    std::vector<uint8_t*> parity_ptrs(m);
    for (int i = 0; i < m; ++i) parity_ptrs[i] = parityShards[i].data();

    // 生成行列 G（系統 RS 行列）: (k + m) x k
    const size_t mat_elems = static_cast<size_t>(k + m) * k;
    std::vector<uint8_t> G(mat_elems);
    // ★重要: 系統符号になる gf_gen_rs_matrix を使う
    gf_gen_rs_matrix(G.data(), k + m, k);

    // パリティ部分（下 m 行）を取り出し
    uint8_t* parity_rows = G.data() + static_cast<size_t>(k) * k;

    // gftbls を初期化してからエンコード（これが正しい ISA-L の使い方）
    std::vector<uint8_t> gftbls(static_cast<size_t>(k) * m * 32);
    ec_init_tables(k, m, parity_rows, gftbls.data());

    // パリティ生成
    ec_encode_data(static_cast<int>(shard_len), k, m, gftbls.data(), data_ptrs.data(), parity_ptrs.data());

    return true;
}

bool DecodeFEC_ISAL(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards,
    int k,
    int m,
    uint32_t originalDataLen,
    std::vector<uint8_t>& decodedData)
{
    decodedData.clear();

    const int n = k + m;
    if (k <= 0 || m < 0 || static_cast<int>(receivedShards.size()) < k) {
        DebugLog(L"DecodeFEC_ISAL: insufficient shards.");
        return false;
    }

    // すべての受信シャードが同じ長さか確認
    size_t shard_len = 0;
    {
        auto it = receivedShards.begin();
        shard_len = it->second.size();
        if (shard_len == 0) {
            DebugLog(L"DecodeFEC_ISAL: first shard len is 0.");
            return false;
        }
        for (const auto& kv : receivedShards) {
            if (kv.second.size() != shard_len) {
                DebugLog(L"DecodeFEC_ISAL: shard length mismatch.");
                return false;
            }
        }
    }

    // 0..n-1 のインデックスで受信済みをマーク
    std::vector<std::vector<uint8_t>> shard_storage(n);     // 受信 / 復元データの置き場
    std::vector<uint8_t*>             shard_ptrs(n, nullptr);

    for (const auto& kv : receivedShards) {
        const uint32_t idx = kv.first;
        if (idx >= static_cast<uint32_t>(n)) continue; // 異常値は無視
        shard_storage[idx] = kv.second;                // コピー（元 map の vector をそのまま移動できないため）
        shard_ptrs[idx] = shard_storage[idx].data();
    }

    // データシャード [0..k-1] が全部あるなら連結するだけ
    bool all_data_present = true;
    for (int i = 0; i < k; ++i) {
        if (!shard_ptrs[i]) { all_data_present = false; break; }
    }
    if (all_data_present) {
        decodedData.resize(static_cast<size_t>(k) * shard_len);
        for (int i = 0; i < k; ++i) {
            std::memcpy(decodedData.data() + static_cast<size_t>(i) * shard_len, shard_ptrs[i], shard_len);
        }
        if (originalDataLen <= decodedData.size())
            decodedData.resize(originalDataLen);
        return true;
    }

    // ここから復元パス
    // 生成行列 G（系統 RS 行列）
    std::vector<uint8_t> G(static_cast<size_t>(k + m) * k);
    gf_gen_rs_matrix(G.data(), k + m, k);

    // 利用する k 本のシャードを選択（先着 k 本）
    std::vector<int> avail_idx; avail_idx.reserve(k);
    for (int idx = 0; idx < n && static_cast<int>(avail_idx.size()) < k; ++idx) {
        if (shard_ptrs[idx]) avail_idx.push_back(idx);
    }
    if (static_cast<int>(avail_idx.size()) < k) {
        DebugLog(L"DecodeFEC_ISAL: available shards < k after selection.");
        return false;
    }

    // S（k×k）＝ G の avail_idx 行からなる部分行列
    std::vector<uint8_t> S(static_cast<size_t>(k) * k, 0);
    for (int r = 0; r < k; ++r) {
        const int src_row = avail_idx[r]; // 0..n-1
        // G は (k+m) x k の行列で、行 major として [row*k + col]
        std::memcpy(&S[static_cast<size_t>(r) * k], &G[static_cast<size_t>(src_row) * k], static_cast<size_t>(k));
    }

    // S を反転して S^{-1} を得る
    std::vector<uint8_t> Sinv(static_cast<size_t>(k) * k, 0);
    if (gf_invert_matrix(S.data(), Sinv.data(), k) < 0) {
        DebugLog(L"DecodeFEC_ISAL: gf_invert_matrix failed.");
        return false;
    }

    // 欠損しているデータシャードのインデックスを列挙（0..k-1 の範囲）
    std::vector<int> missing_data_rows;
    for (int i = 0; i < k; ++i) {
        if (!shard_ptrs[i]) missing_data_rows.push_back(i);
    }
    if (missing_data_rows.empty()) {
        // ここに来るのは「データのうち一部欠損だが既に直列化した」等の競合パス。念のため全データ再構成。
        missing_data_rows.resize(k);
        std::iota(missing_data_rows.begin(), missing_data_rows.end(), 0);
    }

    // 受信ソース（k 本）をポインタ配列にまとめる（順序は S の行順＝avail_idx 順）
    std::vector<uint8_t*> source_ptrs(k, nullptr);
    for (int r = 0; r < k; ++r) {
        source_ptrs[r] = shard_ptrs[avail_idx[r]];
    }

    // 欠損データ行用の復元係数行列（rows = missing_data_rows.size(), cols = k）
    const int rnum = static_cast<int>(missing_data_rows.size());
    std::vector<uint8_t> dec_rows(static_cast<size_t>(rnum) * k, 0);
    for (int r = 0; r < rnum; ++r) {
        const int data_row = missing_data_rows[r];
        // S^{-1} の「data_row 行」をそのまま使用（D = S^{-1} * R なので、行単位で出力を得られる）
        std::memcpy(&dec_rows[static_cast<size_t>(r) * k],
                    &Sinv[static_cast<size_t>(data_row) * k],
                    static_cast<size_t>(k));
    }

    // 復元先バッファ
    std::vector<std::vector<uint8_t>> recovered(rnum, std::vector<uint8_t>(shard_len, 0));
    std::vector<uint8_t*> recovered_ptrs(rnum, nullptr);
    for (int r = 0; r < rnum; ++r) recovered_ptrs[r] = recovered[r].data();

    // gftbls を作って復元（これが ISA-L の正しいデコード手順）
    std::vector<uint8_t> gftbls(static_cast<size_t>(k) * rnum * 32);
    ec_init_tables(k, rnum, dec_rows.data(), gftbls.data());
    ec_encode_data(static_cast<int>(shard_len), k, rnum, gftbls.data(), source_ptrs.data(), recovered_ptrs.data());

    // 完全なデータシャード列（0..k-1）を並べる
    std::vector<const uint8_t*> data_rows(k, nullptr);
    for (int i = 0; i < k; ++i) {
        if (shard_ptrs[i]) {
            data_rows[i] = shard_ptrs[i];
        } else {
            // 欠損だった行に対応する recovered を割り当て
            auto it = std::find(missing_data_rows.begin(), missing_data_rows.end(), i);
            const int r = static_cast<int>(std::distance(missing_data_rows.begin(), it));
            data_rows[i] = recovered[static_cast<size_t>(r)].data();
        }
    }

    // 連結して originalDataLen まで切り詰め
    decodedData.resize(static_cast<size_t>(k) * shard_len);
    for (int i = 0; i < k; ++i) {
        std::memcpy(decodedData.data() + static_cast<size_t>(i) * shard_len, data_rows[i], shard_len);
    }
    if (originalDataLen <= decodedData.size())
        decodedData.resize(originalDataLen);

    return true;
}