// NV12ToRGB_Lanczos3_PS.hlsl
// 入力: texY (R8_UNORM, フル解像度), texUV (R8G8_UNORM, 半解像度; R=U, G=V)
// 出力: float4(RGB, 1.0)
// 仕様: U,V を Lanczos3 (半径3, 7x7 tap) で 4:2:0->4:4:4 にアップサンプル後、BT.709 で YUV->RGB 変換。
// 注意: サンプラ s0 は POINT + CLAMP を前提とする（既存 RS/PSO と同一）。
// 余分なことはせずに指示に従い、goto は使用しない。時間計測やログは C++ 側を維持し、ここでは変更しない。

Texture2D<float>     texY   : register(t0); // Y
Texture2D<float2>    texUV  : register(t1); // U=R, V=G
SamplerState         Sampler: register(s0); // 推奨: POINT + CLAMP（既存ルートシグネチャに合わせる）

#ifndef USE_BT601
  #define USE_BT601 0  // 0: BT.709, 1: BT.601
#endif
#ifndef FULL_RANGE
  #define FULL_RANGE 0 // 0: Limited(16-235/240), 1: Full(0-255)
#endif

static const float PI = 3.14159265358979323846f;

// sinc(x) with sinc(0)=1
float sinc(float x)
{
    x = abs(x);
    if (x < 1e-6f) return 1.0f;
    float pix = PI * x;
    return sin(pix) / pix;
}

// Lanczos3 kernel: w(x) = sinc(x) * sinc(x/3) for |x| < 3, else 0
float lanczos3(float x)
{
    x = abs(x);
    if (x >= 3.0f) return 0.0f;
    return sinc(x) * sinc(x / 3.0f);
}

// ITU-R BT.709/601 matrix (for 8bit-scaled inputs Y-16, C-128 on Limited)
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

// Lanczos3 で UV（半解像度）を 4:4:4 へアップサンプル
float2 SampleUV_Lanczos3(float2 uv)
{
    // UV テクスチャのサイズ（テクセル数）を取得
    uint wUV, hUV;
    texUV.GetDimensions(wUV, hUV);
    float2 sizeUV = float2((float)wUV, (float)hUV);

    // 正規化座標 → テクセル座標（中心基準）。0.5 を引いて texel center を整数とみなす。
    float2 uvTexel = uv * sizeUV - 0.5f;
    float2 f = frac(uvTexel); // 中心の小数部 (0..1)

    // 1D の Lanczos 重み（横・縦）を分離計算
    float wx[7];
    float wy[7];
    float sumx = 0.0f;
    float sumy = 0.0f;

    [unroll]
    for (int i = -3; i <= 3; ++i)
    {
        float w = lanczos3((float)i - f.x);
        wx[i + 3] = w;
        sumx += w;
    }
    [unroll]
    for (int j = -3; j <= 3; ++j)
    {
        float w = lanczos3((float)j - f.y);
        wy[j + 3] = w;
        sumy += w;
    }

    // 個別正規化（境界繰り返しでも総和は 1 になるが、数値安定性のために正規化）
    float invSumX = (sumx != 0.0f) ? (1.0f / sumx) : 0.0f;
    float invSumY = (sumy != 0.0f) ? (1.0f / sumy) : 0.0f;

    [unroll] for (int i = 0; i < 7; ++i) wx[i] *= invSumX;
    [unroll] for (int j = 0; j < 7; ++j) wy[j] *= invSumY;

    // 2D 合成
    float2 acc = float2(0.0f, 0.0f);
    float  wsum = 0.0f;

    [unroll]
    for (int j = -3; j <= 3; ++j)
    {
        float wyj = wy[j + 3];
        float dv  = ((float)j - f.y) / sizeUV.y;

        [unroll]
        for (int i = -3; i <= 3; ++i)
        {
            float wxi = wx[i + 3];
            float du  = ((float)i - f.x) / sizeUV.x;

            float2 samp = texUV.SampleLevel(Sampler, uv + float2(du, dv), 0.0f).rg; // POINT+CLAMP 前提
            float  w    = wxi * wyj;
            acc  += samp * w;
            wsum += w;
        }
    }

    // 2D の総和でも一応正規化（境界での数値誤差対策）
    if (wsum > 1e-6f) acc /= wsum;

    return acc; // (U,V) in 0..1
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    // Luma はフル解像度そのまま
    float Y = texY.Sample(Sampler, uv).r; // 0..1

    // Chroma は Lanczos3 でアップサンプリング
    float2 UV = SampleUV_Lanczos3(uv);    // 0..1

    // 0..255 スケール（既存実装と整合）
    float y = Y      * 255.0f;
    float u = UV.x   * 255.0f;
    float v = UV.y   * 255.0f;

#if FULL_RANGE
    float yc = y;
    float uc = u - 128.0f;
    float vc = v - 128.0f;
#else
    float yc = y - 16.0f;
    float uc = u - 128.0f;
    float vc = v - 128.0f;
#endif

    float3 rgb255 = mul(float3(yc, uc, vc), transpose(GetMat()));
    float3 rgb    = saturate(rgb255 / 255.0f);

    return float4(rgb, 1.0f);
}
