#pragma once
#include <cstdint>
#include <vector>
#include <map>

/**
 * @brief ISA-Lを使用してデータをFECエンコードする
 * @param original_data エンコード対象の元データ
 * @param original_data_len 元データの長さ
 * @param dataShards [out] 生成されたデータシャードが格納される
 * @param parityShards [out] 生成されたパリティシャードが格納される
 * @param shard_len_out [out] 計算されたシャード長（パディング後）が格納される
 * @param k データシャード数
 * @param m パリティシャード数
 * @return 成功した場合はtrue、失敗した場合はfalse
 */
bool EncodeFEC_ISAL(
    const uint8_t* original_data,
    size_t original_data_len,
    std::vector<std::vector<uint8_t>>& dataShards,
    std::vector<std::vector<uint8_t>>& parityShards,
    size_t& shard_len_out,
    int k,
    int m);

/**
 * @brief ISA-Lを使用してFECシャードから元データをデコード（復元）する
 * @param receivedShards 受信したシャードのマップ (キー: シャードインデックス, 値: シャードデータ)
 * @param k データシャード数
 * @param m パリティシャード数
 * @param originalDataLen パディング除去前の元データの長さ
 * @param decodedData [out] デコードされたデータが格納される
 * @return 成功した場合はtrue、失敗した場合はfalse
 */
bool DecodeFEC_ISAL(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards,
    int k,
    int m,
    uint32_t originalDataLen,
    std::vector<uint8_t>& decodedData);
