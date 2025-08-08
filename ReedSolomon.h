#include <vector>
#include <cstdint>
#include <map>     // ※ std::map を使用するためにインクルード

bool EncodeFEC_Jerasure(
    const uint8_t* data,           // パディング済み入力データ
    size_t padded_data_len,        // パディング済みデータ長
    std::vector<std::vector<uint8_t>>& parityShards, // 出力パリティシャード
    int k,
    int m,
    int* matrix,                   // jerasure_matrix_encode で生成した行列
    size_t shard_len);

bool DecodeFEC_Jerasure(
    const std::map<uint32_t, std::vector<uint8_t>>& receivedShards, // 受信シャード
    int k,
    int m,
    uint32_t originalDataLen, // 元データ長
    std::vector<uint8_t>& decodedData,  // 出力先
    const int* bitmatrix);             // ※ デコードに使用するビット行列

