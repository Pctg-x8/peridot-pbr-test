VertexInput {
    Binding 0 [PerVertex] { pos: vec4; }
    Binding 1 [PerVertex] { cubeRefPosition: vec4; }
}
VertexShader {
    RasterPosition = pos;
    localPos = cubeRefPosition.xyz;
}
Varyings VertexShader -> FragmentShader {
    localPos: vec3;
}

Header[FragmentShader] {
    const uint SAMPLE_COUNT = 1024;
    #include "specular_prefilter_common.glsl"
}
FragmentShader {
    const vec3 normal = normalize(localPos);

    vec3 sum = vec3(0.0, 0.0, 0.0);
    float totalSampleCount = 0;
    for (uint i = 0; i < SAMPLE_COUNT; i++) {
        const vec3 h = importanceSamplingGGX(hammersleySamplePoint(i, SAMPLE_COUNT), normal, roughness);
        const vec3 l = normalize(2.0f * dot(normal, h) * h - normal);

        const float ndl = max(0.0f, dot(normal, l));
        if (ndl > 0.0f) {
            // opt: omit texture sampling if this factor will not be affected
            sum += texture(environmentMap, l).rgb * ndl;
            totalSampleCount += ndl;
        }
    }

    Target[0] = vec4(sum / totalSampleCount, 1.0f);
}

PushConstant[FragmentShader] Parameters { float roughness; }
SamplerCube[FragmentShader](0, 0) environmentMap
