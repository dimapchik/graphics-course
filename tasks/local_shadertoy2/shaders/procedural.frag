#version 450

layout(push_constant) uniform Parameters
{
    vec2 iResolution;
    float iTime;
    uint targetIsSrgb;
};

layout(location = 0) out vec4 albedoHeight;
layout(location = 1) out vec4 roughnessMetalness;

// RGB = процедурная цветная текстура альбедо; A = карта высот.
void generateAlbedoHeight(out vec4 fragColor, in vec2 fragCoord)
{
    const float TAU = 6.28318530718;
    vec2 uv = fragCoord / iResolution.xy;
    float warp = sin(TAU * (uv.x - uv.y) + 0.32 * iTime);
    float phase = TAU * (2.0 * uv.x + uv.y) + 1.8 * warp + 0.4 * iTime;
    float height = 0.5 + 0.5 * sin(phase);
    float marble = smoothstep(0.12, 0.88, height);
    vec3 darkColor = vec3(0.28, 0.40, 0.52);
    vec3 lightColor = vec3(0.92, 0.96, 1.0);
    vec3 albedoTexture = mix(darkColor, lightColor, marble);
    fragColor = vec4(albedoTexture, height);
}

// R = roughness, G = metalness; без гамма-коррекции.
void generateRoughnessMetalness(out vec4 fragColor, in vec2 fragCoord)
{
    const float TAU = 6.28318530718;
    vec2 uv = fragCoord / iResolution.xy;
    float wave = sin(TAU * (2.0 * uv.x + uv.y) + 0.3 * iTime
                   + 0.6 * sin(TAU * uv.y));
    float roughness = mix(0.24, 0.72, 0.5 + 0.5 * wave);
    float patches = sin(TAU * uv.x + 0.18 * iTime)
                  * cos(TAU * uv.y - 0.14 * iTime);
    float metalness = smoothstep(-0.22, 0.22, patches);
    fragColor = vec4(roughness, metalness, 0.0, 1.0);
}

void main()
{
    // Texture row 0 must correspond to Shadertoy's v = 0 for later sampling.
    generateAlbedoHeight(albedoHeight, gl_FragCoord.xy);
    generateRoughnessMetalness(roughnessMetalness, gl_FragCoord.xy);
}
