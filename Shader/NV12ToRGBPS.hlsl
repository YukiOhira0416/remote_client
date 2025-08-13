// NV12ToRGBPS.hlsl (完全版: 置き換え)
// デフォルト: BT.709 + Limited Range
// 必要に応じてマクロで切替: 例) /D USE_BT601=1 /D FULL_RANGE=1
Texture2D    TextureY  : register(t0); // R8_UNORM
Texture2D    TextureUV : register(t1); // R8G8_UNORM
SamplerState Sampler   : register(s0); // static POINT+CLAMP（root signature側）

#ifndef USE_BT601
  #define USE_BT601 0  // 0: BT.709, 1: BT.601
#endif
#ifndef FULL_RANGE
  #define FULL_RANGE 0 // 0: Limited(16-235/240), 1: Full(0-255)
#endif

// 係数: ITU-R BT.709/601 (Y'CbCr -> R'G'B')、Limited時は Yoffset=16, Coffset=128, scale=219/224 等を内部で扱う
float3x3 GetMat()
{
#if USE_BT601
    // BT.601
    // R = 1.164*(Y-16) + 1.596*(Cr-128)
    // G = 1.164*(Y-16) - 0.392*(Cb-128) - 0.813*(Cr-128)
    // B = 1.164*(Y-16) + 2.017*(Cb-128)
    return float3x3(
        1.164383f,  0.000000f,  1.596027f,
        1.164383f, -0.391762f, -0.812968f,
        1.164383f,  2.017232f,  0.000000f
    );
#else
    // BT.709
    // R = 1.164*(Y-16) + 1.793*(Cr-128)
    // G = 1.164*(Y-16) - 0.213*(Cb-128) - 0.534*(Cr-128)
    // B = 1.164*(Y-16) + 2.115*(Cb-128)
    return float3x3(
        1.164383f,  0.000000f,  1.792741f,
        1.164383f, -0.213249f, -0.532909f,
        1.164383f,  2.112402f,  0.000000f
    );
#endif
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    // NV12: Y はフル解像度、UV は半解像度のインタリーブ (U=rg.x, V=rg.y)
    float  Y  = TextureY.Sample(Sampler,  uv).r;      // 0..1
    float2 UV = TextureUV.Sample(Sampler, uv).rg;     // 0..1

    // 0..255 相当にスケール
    float y = Y  * 255.0f;
    float u = UV.x * 255.0f;
    float v = UV.y * 255.0f;

#if FULL_RANGE
    // Full Range: オフセット無し、スケール1
    float yc = y;
    float uc = u - 128.0f;
    float vc = v - 128.0f;
    // 行列はそのまま使用
#else
    // Limited Range: Y-16, C-128
    float yc = y - 16.0f;
    float uc = u - 128.0f;
    float vc = v - 128.0f;
#endif

    float3 rgb255 = mul(float3(yc, uc, vc), transpose(GetMat()));
    float3 rgb = saturate(rgb255 / 255.0f);

    return float4(rgb, 1.0f);
}
