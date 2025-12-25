// Simple pixel shader for rendering a semi-transparent quad
float4 main() : SV_TARGET
{
    return float4(0.0f, 0.0f, 0.0f, 0.5f); // Black, 50% transparent
}
