// NV12ToRGBA709Full.hlsl
// 目的: HEVC/H.264 などの NV12 (4:2:0, Full Range) 入力を RGBA BT.709 に変換
// 注意: 既存の計測・ログと整合するように、C++ 側の CropCB(b0) をそのまま利用する

Texture2D<float>   TextureY  : register(t0); // R8/R16 UNORM
Texture2D<float2>  TextureUV : register(t1); // R8G8/R16G16 UNORM (U = .x, V = .y)
SamplerState       Sampler   : register(s0);

cbuffer CropCB : register(b0)
{
    float2 uvScale; // = (uvMax - uvMin)
    float2 uvBias;  // = uvMin
    float2 _pad;    // 16B 整列維持
};

static float3 Nv12ToRgb709Full(float y, float u, float v)
{
    float uc = u - 0.5f;
    float vc = v - 0.5f;
    float r = y + 1.5748f * vc;
    float g = y - 0.1873f * uc - 0.4681f * vc;
    float b = y + 1.8556f * uc;
    return saturate(float3(r, g, b));
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    float2 sampleUv = uv * uvScale + uvBias;
    float y = TextureY.Sample(Sampler, sampleUv).r;
    float2 uvSample = TextureUV.Sample(Sampler, sampleUv);
    float3 rgb = Nv12ToRgb709Full(y, uvSample.x, uvSample.y);
    return float4(rgb, 1.0f);
}
