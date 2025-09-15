// NV12ToRGBPS.hlsl (高品質版: 置き換え)
// 既存機能（BT.709/601, Full/Limited Range）を維持しつつ、以下を追加：
// - [HQ] UV（クロマ）の双三次(Catmull-Rom)アップサンプリング（切替可）
// - [HQ] ルーマ限定アンシャープ（弱め、切替可）
// - [HQ] ディザ追加（切替可）
// - [HQ] saturate と境界/異常系フォールバックで安全性を確保
// 余分なコメントを消さず、追記は // [HQ] で示す。gotoは使用しない。

Texture2D    TextureY  : register(t0); // R8_UNORM
Texture2D    TextureUV : register(t1); // R8G8_UNORM (NV12 interleaved UV)
SamplerState Sampler   : register(s0); // static（ルートシグネチャ側定義を使用）

#ifndef USE_BT601
  #define USE_BT601 0  // 0: BT.709, 1: BT.601
#endif
#ifndef FULL_RANGE
  #define FULL_RANGE 0 // 0: Limited(16-235/240), 1: Full(0-255)
#endif

// [HQ] 追加マクロ（切替用）
#ifndef HQ_CHROMA_BICUBIC
  #define HQ_CHROMA_BICUBIC 1   // 1: UVを双三次（Catmull-Rom）でアップサンプル、0: 従来のSample
#endif
#ifndef HQ_LUMA_UNSHARP
  #define HQ_LUMA_UNSHARP 1     // 1: ルーマ限定アンシャープを軽く適用、0: 無効
#endif
#ifndef HQ_DITHER
  #define HQ_DITHER 1           // 1: 出力直前に微小ディザ、0: 無効
#endif

// [HQ] Catmull-Romカーネル
float CatmullRom(float x)
{
    x = abs(x);
    if (x < 1.0) {
        return 1.0 * (1.0 - 2.0*x*x + x*x*x);
    } else if (x < 2.0) {
        return 1.0 * (4.0 - 8.0*x + 5.0*x*x - x*x*x);
    }
    return 0.0;
}

// 係数: ITU-R BT.709/601 (Y'CbCr -> R'G'B')。Limited時は Y-16, C-128 を前段で処理。
float3x3 GetMat()
{
#if USE_BT601
    // BT.601
    return float3x3(
        1.164383f,  0.000000f,  1.596027f,
        1.164383f, -0.391762f, -0.812968f,
        1.164383f,  2.017232f,  0.000000f
    );
#else
    // BT.709
    return float3x3(
        1.164383f,  0.000000f,  1.792741f,
        1.164383f, -0.213249f, -0.532909f,
        1.164383f,  2.112402f,  0.000000f
    );
#endif
}

// [HQ] UVを双三次でサンプル（Catmull-Rom）。UVは半解像度。
//      失敗時/異常時はフォールバックで安全に返す。
float2 SampleUV_HQ(float2 uv)
{
#if HQ_CHROMA_BICUBIC
    uint w, h;
    TextureUV.GetDimensions(w, h);
    if (w == 0 || h == 0) {
        return TextureUV.Sample(Sampler, uv).rg; // フォールバック
    }

    // 正規化UV -> テクセル座標
    float2 texSize = float2(w, h);
    float2 pos = uv * texSize - 0.5;    // 中心合わせ
    float2 f   = frac(pos);
    float2 base = floor(pos);

    float2 result = float2(0.0, 0.0);
    float  totalW = 0.0;

    // 4x4タップ（Catmull-Rom、分離可能）
    [unroll] for (int j = -1; j <= 2; ++j)
    {
        float wy = CatmullRom(j - f.y);
        float y = base.y + j;
        y = clamp(y, 0.0, (float)h - 1.0);

        [unroll] for (int i = -1; i <= 2; ++i)
        {
            float wx = CatmullRom(i - f.x);
            float x = base.x + i;
            x = clamp(x, 0.0, (float)w - 1.0);

            float wxy = wx * wy;
            float2 uvSample = TextureUV.Load(int3((int)x, (int)y, 0)).rg;
            result += uvSample * wxy;
            totalW += wxy;
        }
    }

    if (totalW > 0.0) {
        result /= totalW;
    }
    return result;
#else
    return TextureUV.Sample(Sampler, uv).rg;
#endif
}

// [HQ] ルーマ（Y')で軽いアンシャープ。3x3ガウシアンからの差分を弱く加算。
//      入出力は 0..1 のY（R'G'B'からの計算ではなく、Y面から取得して使う）。
float UnsharpLuma(float2 uv)
{
#if HQ_LUMA_UNSHARP
    uint yw, yh;
    TextureY.GetDimensions(yw, yh);
    if (yw == 0 || yh == 0) {
        return TextureY.Sample(Sampler, uv).r;
    }

    float2 texel = 1.0 / float2(yw, yh);

    // 3x3 Gaussian (1 2 1; 2 4 2; 1 2 1) / 16
    float k00 = 1.0/16.0, k01 = 2.0/16.0, k02 = 1.0/16.0;
    float k10 = 2.0/16.0, k11 = 4.0/16.0, k12 = 2.0/16.0;
    float k20 = 1.0/16.0, k21 = 2.0/16.0, k22 = 1.0/16.0;

    float y00 = TextureY.Sample(Sampler, uv + float2(-texel.x, -texel.y)).r;
    float y01 = TextureY.Sample(Sampler, uv + float2(0.0,       -texel.y)).r;
    float y02 = TextureY.Sample(Sampler, uv + float2(+texel.x, -texel.y)).r;

    float y10 = TextureY.Sample(Sampler, uv + float2(-texel.x, 0.0)).r;
    float y11 = TextureY.Sample(Sampler, uv).r;
    float y12 = TextureY.Sample(Sampler, uv + float2(+texel.x, 0.0)).r;

    float y20 = TextureY.Sample(Sampler, uv + float2(-texel.x, +texel.y)).r;
    float y21 = TextureY.Sample(Sampler, uv + float2(0.0,       +texel.y)).r;
    float y22 = TextureY.Sample(Sampler, uv + float2(+texel.x, +texel.y)).r;

    float blur = y00*k00 + y01*k01 + y02*k02 +
                 y10*k10 + y11*k11 + y12*k12 +
                 y20*k20 + y21*k21 + y22*k22;

    // 強調量は弱め（ハロー抑制）
    const float amount = 0.35; // 調整可：0.25〜0.5程度
    float ySharp = saturate(y11 + (y11 - blur) * amount);
    return ySharp;
#else
    return TextureY.Sample(Sampler, uv).r;
#endif
}

// [HQ] 8bit向け微小ディザ（三角分布風）
float Dither8bit(float2 pos)
{
#if HQ_DITHER
    // 低コストハッシュ
    float2 p = pos;
    float n = dot(p, float2(12.9898, 78.233));
    float s = frac(sin(n) * 43758.5453);
    // 三角分布に近い形に変換
    float t = s + frac(s * 1.2154);
    t = t - floor(t);
    // 1LSB相当（約1/255）より小さく
    return (t - 0.5) * (1.0/255.0);
#else
    return 0.0;
#endif
}

float4 main(float4 pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    // NV12: Y はフル解像度、UV は半解像度 (U=rg.x, V=rg.y)
    float  Y_lin  = UnsharpLuma(uv);          // [HQ] アンシャープ込み/なしでY取得 (0..1)
    float2 UVsamp = SampleUV_HQ(uv);          // [HQ] 高品質 or 従来 (0..1)

    // 0..255 へスケール
    float y = Y_lin * 255.0f;
    float u = UVsamp.x * 255.0f;
    float v = UVsamp.y * 255.0f;

#if FULL_RANGE
    float yc = y;              // Full: オフセット無し
    float uc = u - 128.0f;
    float vc = v - 128.0f;
#else
    float yc = y - 16.0f;      // Limited: Y-16
    float uc = u - 128.0f;     //          C-128
    float vc = v - 128.0f;
#endif

    float3 rgb255 = mul(float3(yc, uc, vc), transpose(GetMat()));
    float3 rgb = saturate(rgb255 / 255.0f);

    // [HQ] 微小ディザで8bitバンディングの抑制（位置はY面の座標系で近似）
#if HQ_DITHER
    uint yw, yh;
    TextureY.GetDimensions(yw, yh);
    float2 ipos = float2(uv.x * (float)yw, uv.y * (float)yh);
    float d = Dither8bit(ipos);
    rgb = saturate(rgb + d);
#endif

    return float4(rgb, 1.0f);
}
