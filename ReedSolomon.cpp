// ReadSolomon.cpp などに追加

#include <vector>
#include <map>     // ※ std::map を使用するためにインクルード
#include <cstdint>
#include <cstring> // for memcpy
#include "DebugLog.h"
#include <gf_complete.h>
#include <jerasure.h>
#include "ReedSolomon.h" // ※ ヘッダーファイルのインクルードを追加
#include "Globals.h"

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
    if (!data || !matrix || k <= 0 || m <= 0 || shard_len <= 0 || padded_data_len != k * shard_len) {
        DebugLog(L"EncodeFEC_Jerasure: Invalid arguments.");
        return false;
    }

    // データシャードとパリティシャードのポインタ配列を作成 (char** 型)
    std::vector<char*> data_ptrs(k);
    std::vector<char*> coding_ptrs(m);

    // パリティシャード用のメモリ確保とポインタ設定
    parityShards.resize(m);
    for (int i = 0; i < m; ++i) {
        parityShards[i].resize(shard_len);
        coding_ptrs[i] = reinterpret_cast<char*>(parityShards[i].data());
    }

    // データシャードポインタ設定 (入力データを直接参照)
    for (int i = 0; i < k; ++i) {
        data_ptrs[i] = const_cast<char*>(reinterpret_cast<const char*>(data)) + i * shard_len;
    }

    // Jerasure でエンコードを実行 (ビット行列を使用, w=8)
    jerasure_bitmatrix_encode(k, m, 8, matrix, data_ptrs.data(), coding_ptrs.data(), shard_len, static_cast<int>(shard_len / 8));

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
    if (receivedShards.size() < k) {
        DebugLog(L"DecodeFEC_Jerasure Error: Not enough shards received (" + std::to_wstring(receivedShards.size()) + L" < " + std::to_wstring(k) + L")");
        return false;
    }

    int n = k + m;
    size_t shard_len = 0;

    // シャード長を取得
    if (!receivedShards.empty()) {
        shard_len = receivedShards.begin()->second.size();
    }
    if (shard_len == 0) {
        DebugLog(L"DecodeFEC_Jerasure Error: Invalid shard length (0)");
        return false;
    }

    // --- デコード処理 ---
    std::vector<char*> data_ptrs(k);   // 元のデータシャードへのポインタ
    std::vector<char*> coding_ptrs(m); // 元のパリティシャードへのポインタ
    std::vector<int> erasures(m + 1);  // 消失したシャードのインデックスリスト (-1終了)
    int num_erasures = 0;
    std::vector<std::vector<uint8_t>> temp_shards(n); // 受信/復元シャードの一時格納用

    // --- 受信シャードを一時バッファにコピーし、消失を記録 ---
    for (int i = 0; i < n; ++i) {
        auto it = receivedShards.find(i);
        bool is_shard_valid = (it != receivedShards.end() && it->second.size() == shard_len);

        if (is_shard_valid) {
            // Valid shard received
            temp_shards[i] = it->second;
        } else {
            // Shard is missing or invalid, treat as an erasure.
            if (num_erasures < m) {
                erasures[num_erasures++] = i;
                temp_shards[i].resize(shard_len, 0); // Allocate a buffer for the recovery
            } else {
                // Too many erasures to recover.
                DebugLog(L"DecodeFEC_Jerasure Error: Too many erasures (" + std::to_wstring(num_erasures + 1) + L" > " + std::to_wstring(m) + L")");
                return false;
            }
        }

        // Set up the pointers for Jerasure
        if (i < k) {
            data_ptrs[i] = reinterpret_cast<char*>(temp_shards[i].data());
        } else {
            coding_ptrs[i - k] = reinterpret_cast<char*>(temp_shards[i].data());
        }
    }
    erasures[num_erasures] = -1; // 消失リストの終了マーカー

    // --- デコード実行 ---
    if (bitmatrix == nullptr) { // ※ 引数で渡されたビット行列をチェック
        DebugLog(L"DecodeFEC_Jerasure Error: Jerasure bitmatrix not provided for decoding.");
        return false;
    }


    // ※ jerasure_bitmatrix_decode を使用 ※
    // The last two arguments are data_ptrs and coding_ptrs. Erasures must come before them.
    // The 'row_k_ones' parameter is not used in this decoding mode, set to 0.
    int ret = jerasure_bitmatrix_decode(k, m, 8, const_cast<int*>(bitmatrix), 0, erasures.data(), data_ptrs.data(), coding_ptrs.data(), static_cast<int>(shard_len), static_cast<int>(shard_len / 8));
    if (ret != 0) { // Jerasure returns 0 on success, -1 on failure.
        DebugLog(L"DecodeFEC_Jerasure Error: jerasure_bitmatrix_decode failed. Return code: " + std::to_wstring(ret));
        return false; // デコード失敗
    }

    // --- 元データの再構築 ---
    decodedData.clear();
    decodedData.reserve(k * shard_len); // 最大のサイズを予約 (パディング込み)

    /*DebugLog(L"DecodeFEC_Jerasure: Reconstructing data. k=" + std::to_wstring(k) +
             L", m=" + std::to_wstring(m) +
             L", shard_len=" + std::to_wstring(shard_len) +
             L", originalDataLen=" + std::to_wstring(originalDataLen));*/


    for (int i = 0; i < k; ++i) {
        if (temp_shards[i].size() < shard_len) {
            DebugLog(L"DecodeFEC_Jerasure Warning: temp_shards[" + std::to_wstring(i) +
                     L"].size() (" + std::to_wstring(temp_shards[i].size()) +
                     L") is less than shard_len (" + std::to_wstring(shard_len) + L")");
            // ここで処理を中断するか、エラーとするかは仕様次第
        }
        // temp_shards[i] には、受信したか復元されたデータシャードが格納されている
        // shard_len バイト分を decodedData に追加する
        decodedData.insert(decodedData.end(), temp_shards[i].begin(), temp_shards[i].begin() + shard_len);
    }

    //DebugLog(L"DecodeFEC_Jerasure: Before resize, decodedData.size()=" + std::to_wstring(decodedData.size()));

    // パディングを除去
    // decodedData の現在のサイズが originalDataLen より大きい場合、originalDataLen にリサイズする
    if (decodedData.size() > originalDataLen) {
        decodedData.resize(originalDataLen);
    } else if (decodedData.size() < originalDataLen) {
        // このケースは通常発生しないはずだが、念のため警告を出す
        DebugLog(L"DecodeFEC_Jerasure Warning: Reconstructed data is shorter than originalDataLen. Size: " +
            std::to_wstring(decodedData.size()) + L", Original: " + std::to_wstring(originalDataLen));
    }
    // decodedData.size() == originalDataLen の場合は何もしない

    //DebugLog(L"DecodeFEC_Jerasure: After resize, decodedData.size()=" + std::to_wstring(decodedData.size()));

    return true;
}
