VertexInput {
    Binding 0 [PerVertex] { pos: vec4; normal: vec4; tangent: vec4; binormal: vec4; }
}
VertexShader {
    worldPos = transpose(modelTransform) * pos;
    RasterPosition = transpose(mvp) * pos;
    normal_v = normalize((transpose(modelTransform) * normal).xyz);
    tangent_v = normalize((transpose(modelTransform) * tangent).xyz);
    binormal_v = normalize((transpose(modelTransform) * binormal).xyz);
}

Header[FragmentShader] {
    const float PI = 3.1415926;

    // Subset of Disney Principled BRDF Functions
    float DistributionGGXAnisotropic(vec3 h) {
        const vec2 alpha = max(vec2(0.001, 0.001), vec2(pow(roughness, 2.0) * (1.0 + anisotropic), pow(roughness, 2.0) * (1.0 - anisotropic)));
        const float th = dot(tangent_v, h);
        const float bh = dot(binormal_v, h);
        const float nh = clamp(dot(normal_v, h), 0.0, 1.0);
        const float a2 = alpha.x * alpha.y;
        const vec3 v = vec3(alpha.y * th, alpha.x * bh, a2 * nh);
        const float v2 = dot(v, v);
        const float w2 = a2 / v2;

        return a2 * w2 * w2 / PI;
        /*const float th2 = pow(abs(dot(tangent_v, h)), 2.0);
        const float bh2 = pow(abs(dot(binormal_v, h)), 2.0);
        const float nm2 = pow(abs(dot(normal_v, h)), 2.0);
        const float denom1 = (th2 / pow(alpha.x, 2.0)) + (bh2 / pow(alpha.y, 2.0)) + nm2;
        const float denom = PI * alpha.x * alpha.y * pow(denom1, 2.0);

        return 1.0 / denom;*/
    }

    // Smiths method with GGX
    float GeometryVisibility(vec3 light, vec3 outDir) {
        const float a = roughness * roughness;
        const float a2 = pow(a, 2.0);
        const float nl = clamp(dot(normal_v, light), 0.0, 1.0);
        const float nv = abs(dot(normal_v, outDir)) + 1e-5;
        const float denomLight = (nl + sqrt(a2 + pow(nl, 2.0) - pow(nl * a, 2.0)));
        const float denomView = (nv + sqrt(a2 + pow(nv, 2.0) - pow(nv * a, 2.0)));

        return 1.0 / (denomLight * denomView);
    }

    // Schlicks approximation method
    vec3 Fresnel1(vec3 f0, float f90, float u) {
        return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
    }

    vec3 Fresnel(vec3 outDir, vec3 h, vec3 albedo) {
        const vec3 f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + albedo * metallic;
        const float u = clamp(dot(lightDir.xyz, h), 0.0, 1.0);

        return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
    }

    vec3 SpecularBRDF(vec3 light, vec3 outDir, vec3 albedo) {
        const vec3 h = normalize(light + outDir);

        return DistributionGGXAnisotropic(h) * GeometryVisibility(light, outDir) * Fresnel(outDir, h, albedo);
    }

    // Simple Lambert approximation
    /*vec3 DiffuseBRDF(vec3 albedo) {
        return max(texture(envIrradianceMap, normal_v).rgb, 0.0) * albedo / PI;
    }*/
    // Energy-conservation Diffuse BRDF from https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
    vec3 DiffuseBRDF(float ndv, float ndl, float ldh, float linearRoughness, vec3 albedo) {
        const float energyBias = mix(0.0, 0.5, linearRoughness);
        const float energyFactor = mix(1.0, 1.0 / 1.51, linearRoughness);
        const float fd90 = energyBias + 2.0 * ldh * ldh * linearRoughness;
        const vec3 f0 = vec3(1.0, 1.0, 1.0);
        const float lightScatter = Fresnel1(f0, fd90, ndl).r;
        const float viewScatter = Fresnel1(f0, fd90, ndv).r;

        const float diffuseLevel = lightScatter * viewScatter * energyFactor;
        return albedo * diffuseLevel / PI;
    }

    vec3 CalcDirectionalLightReflectIntensity(vec3 outDir) {
        const vec3 h = normalize(outDir + lightDir.xyz);
        const float ndv = abs(dot(normal_v, outDir)) + 1e-5;
        const float ndl = dot(normal_v, lightDir.xyz);
        const float ldh = dot(lightDir.xyz, h);
        const vec3 albedo = baseColor.xyz;
        const vec3 ks = Fresnel(outDir, h, albedo);
        const vec3 kd = (vec3(1.0) - ks) * (1.0 - metallic);
        const vec3 diffuse = DiffuseBRDF(ndv, ndl, ldh, roughness, albedo);
        const vec3 brdf = (diffuse + SpecularBRDF(normalize(lightDir.xyz), outDir, albedo)) * lightIntensity.xyz * dot(normal_v, lightDir.xyz);
        // const vec3 brdf = DiffuseBRDF(albedo);

        return brdf;
    }

    vec3 CalcImageLightReflectIntensity(vec3 outDir) {
        return max(texture(envIrradianceMap, normal_v).rgb, 0.0) * baseColor.xyz;
    }
}
FragmentShader {
    const vec3 outDir = normalize((cameraPos - worldPos).xyz);
    Target[0] = vec4(CalcImageLightReflectIntensity(outDir) + CalcDirectionalLightReflectIntensity(outDir), 1.0) * baseColor.a;
}

Varyings VertexShader -> FragmentShader {
    worldPos: vec4;
    normal_v: vec3;
    tangent_v: vec3;
    binormal_v: vec3;
}

Uniform[VertexShader](0, 0) Transform { mat4 mvp; mat4 modelTransform; mat4 viewProjection; }
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