VertexInput {
    Binding 0 [PerVertex] { pos: vec4; color: vec4; }
}
VertexShader {
    RasterPosition = transpose(mvp) * pos;
    c = color;
}
FragmentShader {
    Target[0] = c;
}

Varyings VertexShader -> FragmentShader {
    c: vec4;
}

Uniform[VertexShader](0, 0) Transform { mat4 mvp; }
