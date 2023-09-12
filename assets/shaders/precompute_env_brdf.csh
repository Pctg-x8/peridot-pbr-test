VertexInput {
    Binding 0 [PerVertex] { pos: vec4; }
}
VertexShader {
    RasterPosition = pos;
    localPos = pos.xy * 0.5f + 0.5f;
}
Varyings VertexShader -> FragmentShader {
    localPos: vec2;
}

Header[FragmentShader] {
    const uint SAMPLE_COUNT = 1024;
    #include "specular_prefilter_common.glsl"

    float geometrySchlickGGX(float ndv, float roughness) {
        const float k = (roughness * roughness) / 2.0f;

        const float nom = ndv;
        const float denom = ndv * (1.0f - k) + k;

        return nom / denom;
    }
    float geometrySmith(vec3 n, vec3 v, vec3 l, float roughness) {
        const float ggx1 = geometrySchlickGGX(max(0.0f, dot(n, l)), roughness);
        const float ggx2 = geometrySchlickGGX(max(0.0f, dot(n, v)), roughness);

        return ggx1 * ggx2;
    }

    vec2 integrateBRDF(float ndv, float roughness) {
        const vec3 v = vec3(sqrt(1.0f - ndv * ndv), 0.0f, ndv);
        float a = 0.0f;
        float b = 0.0f;
        vec3 n = vec3(0.0f, 0.0f, 1.0f);

        for (uint i = 0; i < SAMPLE_COUNT; i++) {
            const vec3 h = importanceSamplingGGX(hammersleySamplePoint(i, SAMPLE_COUNT), n, roughness);
            const vec3 l = normalize(2.0f * dot(v, h) * h - v);

            const float ndl = max(0.0f, l.z);
            const float ndh = max(0.0f, h.z);
            const float vdh = max(0.0f, dot(v, h));

            if (ndl > 0.0f) {
                const float g = geometrySmith(n, v, l, roughness);
                const float gv = (g * vdh) / (ndh * ndv);
                const float fc = pow(1.0f - vdh, 5.0f);

                a += (1.0f - fc) * gv;
                b += fc * gv;
            }
        }

        return vec2(a, b) / float(SAMPLE_COUNT);
    }
}
FragmentShader {
    Target[0] = vec4(integrateBRDF(localPos.x, localPos.y), 0.0f, 1.0f);
}
