struct VS_OUTPUT
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD0;
};

VS_OUTPUT main(uint VertexID : SV_VertexID)
{
    VS_OUTPUT Output;
    // Generate a full-screen triangle strip (or two triangles)
    // This example generates a full-screen quad using a triangle strip.
    // Adjust TexCoord as needed if your UV convention is different.
    if (VertexID == 0) { Output.Position = float4(-1.0f,  1.0f, 0.0f, 1.0f); Output.TexCoord = float2(0.0f, 0.0f); } // Top-left
    if (VertexID == 1) { Output.Position = float4( 3.0f,  1.0f, 0.0f, 1.0f); Output.TexCoord = float2(2.0f, 0.0f); } // Top-right (extended)
    if (VertexID == 2) { Output.Position = float4(-1.0f, -3.0f, 0.0f, 1.0f); Output.TexCoord = float2(0.0f, 2.0f); } // Bottom-left (extended)
    // For a simple quad (2 triangles):
    // if (VertexID == 0) { Output.Position = float4(-1.0f,  1.0f, 0.0f, 1.0f); Output.TexCoord = float2(0.0f, 0.0f); } // TL
    // if (VertexID == 1) { Output.Position = float4( 1.0f,  1.0f, 0.0f, 1.0f); Output.TexCoord = float2(1.0f, 0.0f); } // TR
    // if (VertexID == 2) { Output.Position = float4(-1.0f, -1.0f, 0.0f, 1.0f); Output.TexCoord = float2(0.0f, 1.0f); } // BL
    // if (VertexID == 3) { Output.Position = float4( 1.0f, -1.0f, 0.0f, 1.0f); Output.TexCoord = float2(1.0f, 1.0f); } // BR
    return Output;
}