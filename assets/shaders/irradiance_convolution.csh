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
    const float SAMPLE_DELTA = 0.025;
    const float PI = 3.1415926;
}
FragmentShader {
    const vec3 normal = normalize(localPos);
    const vec3 up1 = vec3(0.0, 1.0, 0.0);
    const vec3 right = normalize(cross(up1, normal));
    const vec3 up = normalize(cross(normal, right));

    vec3 sum = vec3(0.0, 0.0, 0.0);
    uint totalSampleCount = 0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += SAMPLE_DELTA) {
        for (float th = 0.0; th < PI * 0.5; th += SAMPLE_DELTA) {
            const vec3 tangentSamplePoint = vec3(sin(th) * cos(phi), sin(th) * sin(phi), cos(th));
            // TODO mat3の乗算でできたりしそうな気がするけど
            const vec3 samplePoint = tangentSamplePoint.x * right + tangentSamplePoint.y * up + tangentSamplePoint.z * normal;

            sum += texture(environmentMap, samplePoint).rgb * cos(th) * sin(th);
            totalSampleCount += 1;
        }
    }

    Target[0] = vec4(PI * sum * (1.0 / float(totalSampleCount)), 1.0);
}

SamplerCube[FragmentShader](0, 0) environmentMap
