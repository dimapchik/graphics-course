#version 450
#extension GL_GOOGLE_include_directive : require

#define METERING_ACCESS readonly
#include "metering.glsl"

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Parameters
{
    uint useTonemapping;
    uint encodeSrgb;
} params;

vec3 linearToSrgb(vec3 color)
{
    return mix(1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055,
               12.92 * color,
               lessThanEqual(color, vec3(0.0031308)));
}

void main()
{
    vec3 color = texelFetch(hdrImage, ivec2(gl_FragCoord.xy), 0).rgb;
    if (params.useTonemapping != 0u)
    {
        color *= exp2(metering.logExposure);
        // Compress highlights with one scale for all three channels.
        color /= 1.0 + max(max(color.r, color.g), color.b);
    }
    if (params.encodeSrgb != 0u)
        color = linearToSrgb(color);

    // The 8-bit attachment handles clamping and quantization.
    outColor = vec4(color, 1.0);
}
