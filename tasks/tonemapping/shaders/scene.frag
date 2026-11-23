#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Parameters
{
    vec2 resolution;
    float cameraYaw;
    float illumination;
} params;

const vec3 sunDirection = normalize(vec3(-0.46, 0.27, -0.85));
const vec3 sunColor = vec3(1.0, 0.68, 0.36);
const vec3 centers[3] = vec3[3](vec3(-1.12, 0.83, 0.0),
                              vec3(0.72, 0.67, 0.5),
                              vec3(1.70, 0.37, -1.05));
const vec3 colors[3] = vec3[3](vec3(0.70, 0.19, 0.06),
                              vec3(0.03, 0.32, 0.31),
                              vec3(0.59, 0.44, 0.19));
const float radii[3] = float[3](0.83, 0.67, 0.37);

vec3 sky(vec3 ray)
{
    float height = sqrt(max(ray.y, 0.0));
    vec3 gradient = mix(vec3(0.50, 0.29, 0.18), vec3(0.10, 0.29, 0.59), height);
    float towardSun = max(dot(ray, sunDirection), 0.0);
    float glow = 2.0 * pow(towardSun, 24.0);
    float disc = 38.0 * smoothstep(0.9986, 0.9990, towardSun);
    return (gradient + sunColor * (glow + disc)) * params.illumination;
}

float hitSphere(vec3 origin, vec3 ray, int sphere)
{
    vec3 offset = origin - centers[sphere];
    float b = dot(offset, ray);
    float d = b * b - dot(offset, offset) + radii[sphere] * radii[sphere];
    return d >= 0.0 ? -b - sqrt(d) : -1.0;
}

void main()
{
    vec2 uv = (2.0 * gl_FragCoord.xy - params.resolution) / params.resolution.y;
    vec3 origin = vec3(0.0, 1.4, 4.5);
    vec3 ray = normalize(vec3(uv.x, -uv.y - 0.08, -1.65));
    float c = cos(params.cameraYaw);
    float s = sin(params.cameraYaw);
    ray.xz = mat2(c, s, -s, c) * ray.xz;

    float distanceToHit = ray.y < -0.001 ? -origin.y / ray.y : 1000.0;
    int object = -1;
    for (int i = 0; i < 3; ++i)
    {
        float t = hitSphere(origin, ray, i);
        if (t > 0.0 && t < distanceToHit)
        {
            distanceToHit = t;
            object = i;
        }
    }

    vec3 color = sky(ray);
    if (distanceToHit < 1000.0)
    {
        vec3 point = origin + distanceToHit * ray;
        vec3 normal = vec3(0.0, 1.0, 0.0);
        vec3 albedo;
        if (object >= 0)
        {
            normal = normalize(point - centers[object]);
            albedo = colors[object];
        }
        else
        {
            vec2 tile = point.xz * 0.75;
            float checker = mod(floor(tile.x) + floor(tile.y), 2.0);
            float detail = 1.0 - smoothstep(8.0, 25.0, distanceToHit);
            albedo = mix(vec3(0.075, 0.085, 0.10), vec3(0.26, 0.23, 0.19),
                         mix(0.5, checker, detail));
        }

        float visibility = 1.0;
        for (int i = 0; i < 3; ++i)
        {
            if (i != object && hitSphere(point + 0.002 * normal, sunDirection, i) > 0.0)
                visibility = 0.0;
        }

        float diffuse = max(dot(normal, sunDirection), 0.0);
        vec3 reflected = reflect(ray, normal);
        float highlight = pow(max(dot(reflected, sunDirection), 0.0), 80.0);
        color = albedo * (vec3(0.18, 0.24, 0.34) + 4.0 * sunColor * diffuse * visibility)
            * params.illumination;
        if (object >= 0)
            color += 0.24 * sky(reflected) + 12.0 * sunColor * highlight * visibility
                * params.illumination;

        // Blend distant tiles into the horizon.
        color = mix(color, sky(ray), smoothstep(20.0, 80.0, distanceToHit));
    }

    outColor = vec4(color, 1.0);
}
