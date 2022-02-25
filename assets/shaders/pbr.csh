VertexInput {
    Binding 0 [PerVertex] { pos: vec4; normal: vec4; tangent: vec4; binormal: vec4; }
}
VertexShader {
    worldPos = transpose(mvp) * pos;
    RasterPosition = worldPos;
    normal_v = (transpose(modelTransform) * normal).xyz;
    tangent_v = (transpose(modelTransform) * tangent).xyz;
    binormal_v = (transpose(modelTransform) * binormal).xyz;
}

Header[FragmentShader] {
    const float PI = 3.1415926;

    // Subset of Disney Principled BRDF Functions
    float DistributionGGXAnisotropic(vec3 h) {
        const float aspect = sqrt(1.0 - 0.9 * anisotropic);
        const vec2 alpha = max(vec2(0.001, 0.001), vec2(pow(roughness, 2.0) / aspect, pow(roughness, 2.0) * aspect));
        const float th2 = pow(abs(dot(tangent_v, h)), 2.0);
        const float bh2 = pow(abs(dot(binormal_v, h)), 2.0);
        const float nm2 = pow(abs(dot(normal_v, h)), 2.0);
        const float denom1 = (th2 / pow(alpha.x, 2.0)) + (bh2 / pow(alpha.y, 2.0)) + nm2;
        const float denom = PI * alpha.x * alpha.y * pow(denom1, 2.0);

        return 1.0 / denom;
    }

    float GeometryShadowingSubGGX(vec3 v) {
        const float nv = max(0.0, dot(normal_v, v));
        const float a2 = pow(roughness, 2.0);

        return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * pow(nv, 2.0)));
    }
    // Smiths method with GGX
    float GeometryShadowing(vec3 light, vec3 outDir) {
        return GeometryShadowingSubGGX(light) * GeometryShadowingSubGGX(outDir);
    }

    // Schlicks approximation method
    vec3 Fresnel(vec3 outDir, vec3 h, vec3 albedo) {
        const vec3 f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + albedo * metallic;
        const float u = clamp(dot(lightDir.xyz, h), 0.0, 1.0);

        return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
    }

    vec3 SpecularBRDF(vec3 light, vec3 outDir, vec3 albedo) {
        const vec3 h = normalize(light + outDir);

        const vec3 num = DistributionGGXAnisotropic(h) * GeometryShadowing(light, outDir) * Fresnel(outDir, h, albedo);
        const float denom = 4.0 * dot(normal_v, light) * dot(normal_v, outDir);

        return num / (denom + 0.00001);
    }

    // Simple Lambert approximation
    vec3 DiffuseBRDF(vec3 albedo) {
        return max(texture(envIrradianceMap, normal_v).rgb, 0.0) * albedo / PI;
    }

    vec3 CalcDirectionalLightReflectIntensity(vec3 outDir) {
        const vec3 albedo = baseColor.xyz;
        const vec3 ks = Fresnel(outDir, normalize(lightDir.xyz + outDir), albedo);
        const vec3 kd = (vec3(1.0) - ks) * (1.0 - metallic);
        const vec3 brdf = DiffuseBRDF(albedo) + SpecularBRDF(lightDir.xyz, outDir, albedo) * lightIntensity.xyz * dot(normal_v, lightDir.xyz);
        // const vec3 brdf = DiffuseBRDF(albedo);

        return brdf;
    }
}
FragmentShader {
    const vec3 outDir = normalize((cameraPos - worldPos).xyz);
    Target[0] = vec4(CalcDirectionalLightReflectIntensity(outDir), 1.0) * baseColor.a;
}

Varyings VertexShader -> FragmentShader {
    worldPos: vec4;
    normal_v: vec3;
    tangent_v: vec3;
    binormal_v: vec3;
}

Uniform[VertexShader](0, 0) Transform { mat4 mvp; mat4 modelTransform; }
Uniform[FragmentShader](1, 0) Camera {
    vec4 cameraPos;
}
Uniform[FragmentShader](1, 1) DirectionalLight {
    vec4 lightDir, lightIntensity;
}
Uniform[FragmentShader](2, 0) Material {
    vec4 baseColor;
    float roughness, anisotropic, metallic, reflectance;
}
SamplerCube[FragmentShader](3, 0) envIrradianceMap