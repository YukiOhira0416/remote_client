Texture2D    TextureY : register(t0); // Y plane (R8_UNORM)
Texture2D    TextureUV: register(t1); // UV plane (R8G8_UNORM)
SamplerState Sampler  : register(s0); // Sampler

float4 main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD0) : SV_TARGET
{
    // Sample YUV values (range [0.0, 1.0])
    float y_sample = TextureY.Sample(Sampler, TexCoord).r;
    float2 uv_sample = TextureUV.Sample(Sampler, TexCoord).rg; // .r for U, .g for V

    // Convert sampled values [0.0, 1.0] to integer-like range [0, 255]
    // This mimics the byte range before the SaveSeparateYUVTexturesAsBmp's integer math.
    float y_byte = y_sample * 255.0f;
    float u_byte = uv_sample.x * 255.0f;
    float v_byte = uv_sample.y * 255.0f;

    // Apply offsets similar to SaveSeparateYUVTexturesAsBmp
    // Y: 16-235, U/V: 16-240 (conceptually, after 0-255 scaling)
    // c = Y - 16
    // d = U - 128
    // e = V - 128
    float c = y_byte - 16.0f;
    float d = u_byte - 128.0f;
    float e = v_byte - 128.0f;

    // RGB conversion using coefficients from SaveSeparateYUVTexturesAsBmp
    // r = (298 * c + 409 * e + 128) >> 8
    // g = (298 * c - 100 * d - 208 * e + 128) >> 8
    // b = (298 * c + 516 * d + 128) >> 8
    // The '>> 8' is equivalent to dividing by 256.
    // The '+ 128' before shifting is for rounding in integer arithmetic.
    // We'll do floating point math and then normalize.

    float r_calc = (298.0f * c + 409.0f * e + 128.0f) / 256.0f;
    float g_calc = (298.0f * c - 100.0f * d - 208.0f * e + 128.0f) / 256.0f;
    float b_calc = (298.0f * c + 516.0f * d + 128.0f) / 256.0f;

    // Normalize and clamp to [0.0, 1.0] range for output
    // The integer calculations in SaveSeparateYUVTexturesAsBmp implicitly clamp to [0, 255].
    // We then divide by 255.0 to get to [0.0, 1.0] for SV_TARGET.
    float r = saturate(r_calc / 255.0f);
    float g = saturate(g_calc / 255.0f);
    float b = saturate(b_calc / 255.0f);

    return float4(r, g, b, 1.0f);
}

