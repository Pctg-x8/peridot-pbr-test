VertexInput {
    Binding 0 [PerVertex] { pos: vec2; }
}
VertexShader {
    RasterPosition = transpose(mvp) * vec4(pos, 0.0f, 1.0f);
}
FragmentShader {
    Target[0] = col;
    Target[0].rgb *= Target[0].a;
}

Uniform[VertexShader](0, 0) Transform { mat4 mvp; }
PushConstant[FragmentShader] Attributes { vec4 col; }
