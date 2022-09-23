
const float PI = 3.1415926;

float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);

    return float(bits) * 2.3283064365386963e-10;    // 0x100000000
}

vec2 hammersleySamplePoint(uint i, uint n) {
    return vec2(float(i) / float(n), radicalInverseVdC(i));
}

vec3 importanceSamplingGGX(vec2 samplePoint, vec3 normal, float roughness) {
    const float a = roughness * roughness;

    const float phi = 2.0 * PI * samplePoint.x;
    const float cosTheta = sqrt((1.0f - samplePoint.y) / (1.0f + (a * a - 1.0f) * samplePoint.y));
    const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    const vec3 h = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    const vec3 up = abs(normal.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
    const vec3 tangent = normalize(cross(up, normal));
    const vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * h.x + bitangent * h.y + normal * h.z);
}
