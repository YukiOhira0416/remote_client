// Vertex shader for rendering a textured quad
struct VSInput
{
    float3 pos : POSITION;
    float2 tex : TEXCOORD;
};

struct VSOutput
{
    float2 tex : TEXCOORD;
    float4 pos : SV_POSITION;
};

VSOutput main(VSInput input)
{
    VSOutput output;
    output.pos = float4(input.pos, 1.0f);
    output.tex = input.tex;
    return output;
}
