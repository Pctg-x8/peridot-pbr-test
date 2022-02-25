// ported from https://learnopengl.com/PBR/IBL/Diffuse-irradiance

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
    const vec2 invAtan = vec2(0.1591, 0.3183);
    vec2 sampleSphericalMap(vec3 v) {
        return vec2(atan(v.z, v.x), -asin(v.y)) * invAtan + 0.5;
    }
}
FragmentShader {
    const vec2 uv = sampleSphericalMap(normalize(localPos));
    const vec3 color = texture(equirectangularMap, uv).rgb;

    Target[0] = vec4(color, 1.0);
}

Sampler2D[FragmentShader](0, 0) equirectangularMap
