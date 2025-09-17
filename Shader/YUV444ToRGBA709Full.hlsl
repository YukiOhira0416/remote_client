// YUV444ToRGBA709Full.hlsl
// 目的: HEVC 4:4:4 (BT.709, Full Range) 入力の YUV444 → RGBA 変換
// 注意: 余分な処理は行わない / goto は使わない / 既存のログ・計測は C++ 側で維持

Texture2D TextureY : register(t0); // R8_UNORM or R16_UNORM (0..1 に正規化)
Texture2D TextureU : register(t1); // R8_UNORM or R16_UNORM
Texture2D TextureV : register(t2); // R8_UNORM or R16_UNORM
SamplerState Sampler : register(s0); // point/clamp を推奨（D3D12 側の静的サンプラ設定）

// 切替マクロ
#ifndef FULL_RANGE
  #define FULL_RANGE 1 // このシェーダは Full Range 前提
#endif
#ifndef USE_BT601
  #define USE_BT601 0 // 0: BT.709, 1: BT.601
#endif

// BT.709 Full Range 係数（Y はそのまま, U/V は ±0.5 中心）
float3 YUV709FullToRGB(float y, float u, float v)
{
    // Uc, Vc は中心 0 の偏差（-0.5..+0.5）
    float uc = u - 0.5f;
    float vc = v - 0.5f;

#if USE_BT601
    // 参考: BT.601 Full Range（必要なら切替）
    float r = y + 1.4020f * vc;
    float g = y - 0.3441f * uc - 0.7141f * vc;
    float b = y + 1.7720f * uc;
#else
    // BT.709 Full Range
    float r = y + 1.5748f * vc;
    float g = y - 0.1873f * uc - 0.4681f * vc;
    float b = y + 1.8556f * uc;
#endif

    return saturate(float3(r, g, b));
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    // Y, U, V はいずれもフル解像度（アップサンプル不要）
    float y = TextureY.Sample(Sampler, uv).r;
    float u = TextureU.Sample(Sampler, uv).r;
    float v = TextureV.Sample(Sampler, uv).r;

#if FULL_RANGE
    // そのまま（Y:0..1, U/V:0..1）
#else
    // Limited→Full 変換を行う場合はここに追加（今回は Full 前提）
#endif

    float3 rgb = YUV709FullToRGB(y, u, v);
    return float4(rgb, 1.0f);
}
