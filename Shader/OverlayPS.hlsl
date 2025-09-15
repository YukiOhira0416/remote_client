// Pixel shader for rendering a texture with transparency
// [HQ] テキスト/オーバレイをテクセル中心にスナップして最近傍等価で取得する版。
// [HQ] 既存コメントは維持しつつ、機能切替はマクロで可能。
// [HQ] 余分な処理は行わず、ログ/時間計測等はC++側を維持。

Texture2D g_texture : register(t0);
SamplerState g_sampler : register(s0);

struct PSInput
{
    float2 texcoord : TEXCOORD;
    float4 position : SV_POSITION;
};

// [HQ] マクロで挙動切替
#ifndef ENABLE_TEXEL_SNAP
#define ENABLE_TEXEL_SNAP 1   // 1: テクセルスナップ(Load)、0: 従来のSample
#endif

#ifndef ALLOW_LINEAR_FALLBACK
#define ALLOW_LINEAR_FALLBACK 1 // 1: 失敗時や特殊条件でSampleにフォールバック可
#endif

float4 main(PSInput input) : SV_TARGET
{
#if ENABLE_TEXEL_SNAP
    // [HQ] テクセル中心にスナップして最近傍等価のサンプルを行う
    uint w, h;
    g_texture.GetDimensions(w, h); // [HQ] 正常系: 0ではないはず
    if (w == 0 || h == 0) {
    #if ALLOW_LINEAR_FALLBACK
        // [HQ] フォールバック：従来のサンプル
        return g_texture.Sample(g_sampler, input.texcoord);
    #else
        return float4(0,0,0,0);
    #endif
    }

    // [HQ] 正規化UV -> テクセル座標へ
    float2 pix = input.texcoord * float2(w, h);
    // [HQ] 四捨五入で最も近いテクセル中心に揃える
    int2 ipix = int2(floor(pix + 0.5));
    // [HQ] 境界クランプ
    ipix = clamp(ipix, int2(0, 0), int2(int(w) - 1, int(h) - 1));

    // [HQ] Sampler無関係のLoadでフィルタなし取得（最もくっきり）
    float4 c = g_texture.Load(int3(ipix, 0));
    return c;
#else
    // [HQ] 従来どおり
    return g_texture.Sample(g_sampler, input.texcoord);
#endif
}
