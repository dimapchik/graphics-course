#version 450

layout(push_constant) uniform Parameters
{
    vec2 iResolution;
    float iTime;
    uint targetIsSrgb;
};

layout(set = 0, binding = 0) uniform sampler2D iChannel0;
layout(set = 0, binding = 1) uniform sampler2D iChannel1;

layout(location = 0) out vec4 outColor;

// Image: iChannel0 = Buffer A, iChannel1 = Buffer B.
// Filter = Linear, Wrap = Repeat. Buffer A: RGB=albedo, A=height.
const float PI = 3.14159265359;
const float EPS = 0.001;
const float TEX_SCALE = 0.75;
const float RELIEF = 0.003; // Мягкий рельеф; 0.0 отключает bump mapping.
// 0: сцена; 1: альбедо; 2: roughness; 3: metalness; 4: нормаль; 5: Buffer A.
const int DEBUG_VIEW = 0;

vec2 objectDistances(vec3 p)
{
    float rotation = 0.5 * iTime;
    float angle = 0.4;
    vec3 q = p;
    q.xz = mat2(cos(rotation), -sin(rotation),
                sin(rotation), cos(rotation)) * q.xz;
    q.xy = mat2(cos(angle), -sin(angle),
                sin(angle), cos(angle)) * q.xy;
    return vec2(length(p) - 0.8,
                length(vec2(length(q.xz) - 1.25, q.y)) - 0.12);
}

float objectSDF(vec3 p)
{
    vec2 d = objectDistances(p);
    return min(d.x, d.y);
}

// Пол пересекается аналитически: нет дыр от лимита шагов на горизонте.
// Результат: расстояние, ID материала (0: шар, 1: кольцо, 2: пол).
vec2 traceScene(vec3 ro, vec3 rd)
{
    float floorT = rd.y < -0.00001 ? (-1.1 - ro.y) / rd.y : 1e5;
    float limit = min(floorT, 25.0);
    float t = 0.0;
    for (int i = 0; i < 112; i++)
    {
        if (t >= limit) break;
        vec2 d = objectDistances(ro + rd * t);
        float nearest = min(d.x, d.y);
        if (nearest < EPS) return vec2(t, d.x < d.y ? 0.0 : 1.0);
        t += max(nearest * 0.9, 0.0002);
    }
    if (floorT > 0.0 && floorT < 1e5) return vec2(floorT, 2.0);
    return vec2(-1.0);
}

vec3 geometricNormal(vec3 p, float materialID)
{
    if (materialID > 1.5) return vec3(0.0, 1.0, 0.0);
    vec2 e = vec2(EPS, 0.0);
    return normalize(vec3(
        objectSDF(p + e.xyy) - objectSDF(p - e.xyy),
        objectSDF(p + e.yxy) - objectSDF(p - e.yxy),
        objectSDF(p + e.yyx) - objectSDF(p - e.yyx)));
}

// Приближение мягкой тени от протяжённого источника.
float softShadow(vec3 p, vec3 ng, vec3 lightPosition)
{
    vec3 origin = p + ng * 0.015;
    vec3 toLight = lightPosition - origin;
    float distanceToLight = length(toLight);
    vec3 direction = toLight / distanceToLight;
    float visibility = 1.0;
    float t = 0.025;
    for (int i = 0; i < 64; i++)
    {
        if (t >= distanceToLight) break;
        float d = objectSDF(origin + direction * t);
        if (d < 0.0005) return 0.0;
        visibility = min(visibility, 5.0 * d / t);
        t += clamp(d, 0.008, 0.4);
    }
    return smoothstep(0.0, 1.0, clamp(visibility, 0.0, 1.0));
}

vec2 heightSlope(vec2 uv)
{
    vec2 texel = 1.0 / max(vec2(textureSize(iChannel0, 0)), vec2(1.0));
    float l = textureLod(iChannel0, fract(uv - vec2(texel.x, 0.0)), 0.0).a;
    float r = textureLod(iChannel0, fract(uv + vec2(texel.x, 0.0)), 0.0).a;
    float d = textureLod(iChannel0, fract(uv - vec2(0.0, texel.y)), 0.0).a;
    float u = textureLod(iChannel0, fract(uv + vec2(0.0, texel.y)), 0.0).a;
    return RELIEF * vec2((r - l) / (2.0 * texel.x), (u - d) / (2.0 * texel.y));
}

struct Material
{
    vec3 albedo;
    vec3 normal;
    float roughness;
    float metalness;
};

Material readMaterial(vec3 p, vec3 ng, float materialID)
{
    Material m;
    m.normal = ng;
    m.roughness = 0.82;
    m.metalness = 0.0;
    m.albedo = vec3(0.22, 0.26, 0.32);
    if (materialID > 1.5) return m; // Спокойный матовый пол.

    vec3 weights = pow(abs(ng), vec3(4.0));
    weights /= max(weights.x + weights.y + weights.z, 0.0001);
    vec3 q = p * TEX_SCALE;
    // RGB ИЗ ПРОЦЕДУРНОЙ ТЕКСТУРЫ BUFFER A непосредственно задаёт альбедо.
    vec3 texColor = weights.x * textureLod(iChannel0, fract(q.yz), 0.0).rgb
                  + weights.y * textureLod(iChannel0, fract(q.zx), 0.0).rgb
                  + weights.z * textureLod(iChannel0, fract(q.xy), 0.0).rgb;
    vec3 tint = materialID < 0.5 ? vec3(0.20, 0.72, 0.94) : vec3(1.0, 0.57, 0.18);
    m.albedo = texColor * tint;
    vec2 sx = heightSlope(q.yz);
    vec2 sy = heightSlope(q.zx);
    vec2 sz = heightSlope(q.xy);
    vec3 gradient = TEX_SCALE * (weights.x * vec3(0.0, sx.x, sx.y)
                               + weights.y * vec3(sy.y, 0.0, sy.x)
                               + weights.z * vec3(sz.x, sz.y, 0.0));
    gradient -= ng * dot(gradient, ng);
    m.normal = normalize(ng - gradient);
    // Параметры PBR считываются из отдельной процедурной текстуры Buffer B.
    vec2 rm = weights.x * textureLod(iChannel1, fract(q.yz), 0.0).rg
            + weights.y * textureLod(iChannel1, fract(q.zx), 0.0).rg
            + weights.z * textureLod(iChannel1, fract(q.xy), 0.0).rg;
    m.roughness = clamp(rm.r, 0.08, 1.0);
    m.metalness = clamp(rm.g, 0.0, 1.0);
    return m;
}

// Процедурное студийное окружение: небо, земля и две световые панели.
vec3 environment(vec3 direction)
{
    float sky = smoothstep(-0.35, 0.8, direction.y);
    vec3 color = mix(vec3(0.13, 0.15, 0.19), vec3(0.42, 0.53, 0.72), sky);
    float key = pow(max(dot(direction, normalize(vec3(-3.0, 4.0, 3.0))), 0.0), 18.0);
    float fill = pow(max(dot(direction, normalize(vec3(3.0, 2.0, 1.0))), 0.0), 10.0);
    return color + vec3(2.5, 2.2, 1.8) * key + vec3(0.6, 0.8, 1.2) * fill;
}

float distributionGGX(float NoH, float alpha)
{
    float a2 = alpha * alpha;
    float d = (1.0 - NoH * NoH) + a2 * NoH * NoH;
    return a2 / max(PI * d * d, 1e-10);
}
float visibilityGGX(float NoV, float NoL, float alpha)
{
    float a2 = alpha * alpha;
    float v = NoL * sqrt(a2 + (1.0 - a2) * NoV * NoV);
    float l = NoV * sqrt(a2 + (1.0 - a2) * NoL * NoL);
    return 0.5 / max(v + l, 1e-6); // Уже G / (4 NoV NoL).
}
vec3 fresnel(float VoH, vec3 F0)
{
    return F0 + (vec3(1.0) - F0) * pow(1.0 - clamp(VoH, 0.0, 1.0), 5.0);
}
vec3 brdf(vec3 N, vec3 V, vec3 L, Material m)
{
    float NoV = max(dot(N, V), 0.0), NoL = max(dot(N, L), 0.0);
    if (NoV <= 0.0 || NoL <= 0.0) return vec3(0.0);
    vec3 H = normalize(V + L);
    vec3 F0 = mix(vec3(0.04), m.albedo, m.metalness);
    vec3 F = fresnel(max(dot(V, H), 0.0), F0);
    float alpha = m.roughness * m.roughness;
    float D = distributionGGX(clamp(dot(N, H), 0.0, 1.0), alpha);
    vec3 specular = D * visibilityGGX(NoV, NoL, alpha) * F;
    vec3 diffuse = (vec3(1.0) - F) * (1.0 - m.metalness) * m.albedo / PI;
    return diffuse + specular;
}

float radicalInverse(int index)
{
    float value = 0.0, weight = 0.5;
    int n = index;
    for (int i = 0; i < 4; i++)
    {
        value += float(n % 2) * weight;
        n /= 2;
        weight *= 0.5;
    }
    return value;
}

// Интегрирование процедурного окружения: 16 GGX-выборок для specular,
// 16 cosine-weighted выборок для diffuse. Без загруженных HDR-текстур.
vec3 environmentLight(vec3 N, vec3 V, Material m)
{
    vec3 helper = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(helper, N));
    vec3 B = cross(N, T);
    float NoV = max(dot(N, V), 0.001);
    float alpha = m.roughness * m.roughness;
    float a2 = alpha * alpha;
    vec3 F0 = mix(vec3(0.04), m.albedo, m.metalness);
    vec3 result = vec3(0.0);
    for (int i = 0; i < 16; i++)
    {
        vec2 xi = vec2((float(i) + 0.5) / 16.0, radicalInverse(i));
        float phi = 2.0 * PI * xi.x;
        float cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
        vec3 H = normalize(T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta);
        float VoH = max(dot(V, H), 0.0);
        vec3 L = reflect(-V, H);
        float NoL = max(dot(N, L), 0.0);
        if (NoL > 0.0 && VoH > 0.0)
        {
            // BRDF * NoL / pdf_GGX; D сокращается с плотностью выборки.
            float weight = 4.0 * VoH * NoL * visibilityGGX(NoV, NoL, alpha)
                         / max(dot(N, H), 0.001);
            result += environment(L) * fresnel(VoH, F0) * weight;
        }
        float r = sqrt(xi.y);
        vec3 Ld = normalize(T * (cos(phi) * r) + B * (sin(phi) * r)
                           + N * sqrt(1.0 - xi.y));
        vec3 Hd = normalize(V + Ld);
        vec3 kD = (vec3(1.0) - fresnel(max(dot(V, Hd), 0.0), F0)) * (1.0 - m.metalness);
        result += environment(Ld) * kD * m.albedo;
    }
    return result / 16.0;
}

vec3 shade(vec3 p, vec3 ng, vec3 V, Material m)
{
    vec3 N = m.normal;
    vec3 color = environmentLight(N, V, m);
    vec3 lightPosition = vec3(-3.0, 4.0, 3.0);
    vec3 toLight = lightPosition - p;
    float distanceSquared = max(dot(toLight, toLight), 0.001);
    vec3 L = toLight * inversesqrt(distanceSquared);
    float NoL = max(dot(N, L), 0.0);
    if (dot(ng, L) > 0.0)
        color += brdf(N, V, L, m) * vec3(65.0, 56.0, 44.0) / distanceSquared
               * NoL * softShadow(p, ng, lightPosition);
    return color;
}


void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Явная диагностика отсутствующего канала, без подмены текстуры формулой.
    bool missing = any(lessThan(textureSize(iChannel0, 0), ivec2(1)))
                || any(lessThan(textureSize(iChannel1, 0), ivec2(1)));
    if (missing)
    {
        float stripe = step(0.5, fract((fragCoord.x + fragCoord.y) / 48.0));
        fragColor = vec4(mix(vec3(1.0, 0.02, 0.45), vec3(1.0), stripe), 1.0);
        return;
    }
    if (DEBUG_VIEW == 5)
    {
        vec3 c = textureLod(iChannel0, fragCoord / iResolution.xy, 0.0).rgb;
        fragColor = vec4(pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
        return;
    }
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 1.5, 4.5);
    vec3 forward = normalize(vec3(0.0, -0.1, 0.0) - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);
    vec3 rd = normalize(uv.x * right + uv.y * up + 2.35 * forward);
    vec3 color = environment(rd) * 0.55;
    vec2 hit = traceScene(ro, rd);
    if (hit.x >= 0.0)
    {
        vec3 p = ro + rd * hit.x;
        vec3 ng = geometricNormal(p, hit.y);
        Material m = readMaterial(p, ng, hit.y);
        // У самой кромки не допускаем разворота нормали от камеры.
        float facing = smoothstep(0.0, 0.15, dot(ng, -rd));
        m.normal = normalize(mix(ng, m.normal, facing));
        if (DEBUG_VIEW == 1) { fragColor = vec4(pow(m.albedo, vec3(1.0 / 2.2)), 1.0); return; }
        if (DEBUG_VIEW == 2) { fragColor = vec4(vec3(m.roughness), 1.0); return; }
        if (DEBUG_VIEW == 3) { fragColor = vec4(vec3(m.metalness), 1.0); return; }
        if (DEBUG_VIEW == 4) { fragColor = vec4(m.normal * 0.5 + 0.5, 1.0); return; }
        color = shade(p, ng, -rd, m);
        // Атмосферная дымка скрывает резкую границу дальнего плана.
        float fog = 1.0 - exp(-0.016 * hit.x);
        color = mix(color, environment(rd) * 0.55, fog);
    }
    color = max(color, vec3(0.0));
    color = color / (vec3(1.0) + color); // Reinhard, затем gamma.
    fragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}

void main()
{
    // Shadertoy has a bottom-left origin; Vulkan's viewport has a top-left origin.
    vec2 fragCoord = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
    mainImage(outColor, fragCoord);

    // mainImage already produces display-encoded RGB. Decode it for an sRGB
    // attachment so the hardware encoding preserves the original Shadertoy image.
    if (targetIsSrgb != 0u)
    {
        vec3 low = outColor.rgb / 12.92;
        vec3 high = pow((outColor.rgb + 0.055) / 1.055, vec3(2.4));
        outColor.rgb = mix(high, low, lessThanEqual(outColor.rgb, vec3(0.04045)));
    }
}
