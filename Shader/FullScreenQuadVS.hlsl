struct VS_OUTPUT {
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VS_OUTPUT main(uint VertexID : SV_VertexID)
{
    VS_OUTPUT o;

    // 4頂点フルスクリーンクアッド（TL,TR,BL,BR）
    // 画面全域を覆い、UVは 0..1 に限定
    static const float2 POS[4] = {
        float2(-1.0f,  1.0f), // 0: TL
        float2( 1.0f,  1.0f), // 1: TR
        float2(-1.0f, -1.0f), // 2: BL
        float2( 1.0f, -1.0f)  // 3: BR
    };
    static const float2 UVS[4] = {
        float2(0.0f, 0.0f),
        float2(1.0f, 0.0f),
        float2(0.0f, 1.0f),
        float2(1.0f, 1.0f)
    };

    o.Position = float4(POS[VertexID], 0.0f, 1.0f);
    o.TexCoord = UVS[VertexID];
    return o;
}