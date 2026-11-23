#include "metering_config.h"

layout(set = 0, binding = 0) uniform sampler2D hdrImage;
layout(std430, set = 0, binding = 1) METERING_ACCESS buffer Metering
{
    float logExposure;
    uint minimumBits;
    uint maximumBits;
    uint histogram[HISTOGRAM_BINS];
} metering;

float luminance(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float logLuminance(float value)
{
    return log2(max(value, 0.0001));
}

vec2 logRange()
{
    return vec2(logLuminance(uintBitsToFloat(metering.minimumBits)),
                logLuminance(uintBitsToFloat(metering.maximumBits)));
}
