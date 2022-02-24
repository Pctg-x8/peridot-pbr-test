// ported from https://learnopengl.com/PBR/IBL/Diffuse-irradiance

VertexInput {
    Binding 0 [PerVertex] { pos: vec4; }
}
VertexShader {
    localPos = pos.xyz;
    // removes translation part
    const mat4 rotView = mat4(mat3(transpose(view)));
    const vec4 transformed = transpose(projection) * rotView * pos;
    // Force z=1.0
    RasterPosition = transformed.xyww;
}
Varyings VertexShader -> FragmentShader {
    localPos: vec3;
}

FragmentShader {
    const vec3 envColor = texture(envmap, localPos).rgb;
    // reverse gamma
    Target[0] = vec4(pow(envColor / (1.0 + envColor), vec3(1.0 / 2.2)), 1.0);
}

Uniform[VertexShader](0, 0) Camera {
    mat4 projection, view;
}
SamplerCube[FragmentShader](1, 0) envmap
