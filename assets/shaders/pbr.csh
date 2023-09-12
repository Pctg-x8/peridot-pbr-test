VertexInput {
    Binding 0 [PerVertex] { pos: vec4; normal: vec4; tangent: vec4; binormal: vec4; }
}
VertexShader {
    worldPos = transpose(modelTransform) * pos;
    RasterPosition = transpose(mvp) * pos;
    normal_v = normalize(mat3(transpose(modelTransform)) * normal.xyz);
    tangent_v = normalize(mat3(transpose(modelTransform)) * tangent.xyz);
    binormal_v = normalize(mat3(transpose(modelTransform)) * binormal.xyz);
}

Header[FragmentShader] {
    const float PI = 3.1415926f;
    const float MAX_IBL_REFLECTION_LOD = 4.0f;

    // PBR Fragment Shader
    // based: https://learnopengl.com/PBR/Lighting https://google.github.io/filament/Filament.html 

    // Subset of Disney Principled BRDF Functions
    float DistributionGGXAnisotropic(vec3 h) {
        const vec2 alpha = max(vec2(0.001, 0.001), vec2(roughness * (1.0 + anisotropic), roughness * (1.0 - anisotropic)));

        const float th = dot(tangent_v, h);
        const float bh = dot(binormal_v, h);
        const float nh = dot(normal_v, h);
        const float a2 = alpha.x * alpha.y;
        const vec3 v = vec3(alpha.y * th, alpha.x * bh, a2 * nh);
        const float v2 = dot(v, v);
        const float w2 = a2 / v2;

        return a2 * w2 * w2 / PI;

        /*// GGX Anisotropic Model
        const float th2 = pow(abs(dot(tangent_v, h)), 2.0);
        const float bh2 = pow(abs(dot(binormal_v, h)), 2.0);
        const float nm2 = pow(abs(dot(normal_v, h)), 2.0);
        const float denom1 = (th2 / pow(alpha.x, 2.0)) + (bh2 / pow(alpha.y, 2.0)) + nm2;
        const float denom = PI * alpha.x * alpha.y * denom1 * denom1;

        return 1.0 / denom;*/

        /*const float a = roughness * roughness;
        // (nh^2 * a^2 - nh^2 + 1) ^ 2
        const float nh2 = pow(abs(dot(normal_v, h)), 2.0);
        const float w = a / (nh2 * a * a - nh2 + 1);

        return w * w / PI;*/

        /*// ellipsoid ndf: http://www.flycooler.com/download/SupplementalEllipsoidNDF.pdf
        const vec2 alpha = max(vec2(0.001f, 0.001f), vec2(pow(roughness, 2.0f) * (1.0f + anisotropic), pow(roughness, 2.0f) * (1.0f - anisotropic)));
        const mat3 a = mat3(alpha.x, 0.0f, 0.0f, 0.0f, alpha.y, 0.0f, 0.0f, 0.0f, 1.0f);
        const float detA = alpha.x * alpha.y;
        const float an = length(a * normal_v);
        const vec3 athV = inverse(transpose(a)) * h;
        const float ath2 = dot(athV, athV);

        const float nh = dot(h, normal_v);

        return (nh >= 0.0f ? 1.0f : 0.0f) / (PI * detA * an * ath2 * ath2);*/
    }

    float GeometryMasking1(vec3 u, vec3 m) {
        const vec2 alpha = max(vec2(0.001, 0.001), vec2(roughness * (1.0 + anisotropic), roughness * (1.0 - anisotropic)));
        const mat3 a = mat3(alpha.x, 0.0f, 0.0f, 0.0f, alpha.y, 0.0f, 0.0f, 0.0f, 1.0f);

        const vec3 anV = a * normal_v;
        const vec3 auV = a * u;

        return min(1.0f, 2.0f * dot(anV, anV) * abs(dot(u, normal_v)) / (length(auV) * length(anV) + dot(auV, anV))) * (dot(u, m) >= 0.0f ? 1.0f : 0.0f);
    }
    float GeometryMasking(vec3 light, vec3 outDir) {
        const vec3 h = normalize(light + outDir);
        return GeometryMasking1(light, h) * GeometryMasking1(outDir, h);
    }

    // Smiths method with GGX
    float GeometryVisibility(vec3 light, vec3 outDir) {
        /*const float a = roughness * roughness;
        const float a2 = a * a;
        const float nl = clamp(dot(normal_v, light), 0.0, 1.0);
        const float nv = abs(dot(normal_v, outDir)) + 1e-5;
        const float denomLight = nl * sqrt(nv * nv * (1.0 - a2) + a2);
        const float denomView = nv * sqrt(nl * nl * (1.0 - a2) + a2);

        return 0.5 / max(denomLight + denomView, 1e-6);*/
        
        // from filament anisotropic model
        const vec2 alpha = max(vec2(0.001, 0.001), vec2(roughness * (1.0 + anisotropic), roughness * (1.0 - anisotropic)));
        const float ndl = max(0.0f, dot(normal_v, light));
        const float tdl = max(0.0f, dot(tangent_v, light));
        const float bdl = max(0.0f, dot(binormal_v, light));
        const float ndv = max(0.0f, dot(normal_v, outDir));
        const float tdv = max(0.0f, dot(tangent_v, outDir));
        const float bdv = max(0.0f, dot(binormal_v, outDir));
        /*const float lv = ndl * length(vec3(alpha.x * tdv, alpha.y * bdv, ndv));
        const float ll = ndv * length(vec3(alpha.x * tdl, alpha.y * bdl, ndl));*/
        const float lv = ndl * (ndv * (1.0f - roughness) + roughness);
        const float ll = ndv * (ndl * (1.0f - roughness) + roughness);
        return min(0.5 / (lv + ll), 65504.0f);
    }

    // Schlicks approximation method
    vec3 Fresnel1(vec3 f0, float f90, float u) {
        return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
    }

    vec3 Fresnel(vec3 outDir, vec3 h, vec3 albedo) {
        const vec3 f0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + albedo * metallic;
        const float u = clamp(dot(lightDir.xyz, h), 0.0, 1.0);

        return Fresnel1(f0, 1.0, u);
    }

    vec3 SpecularBRDF(vec3 light, vec3 outDir, vec3 albedo) {
        const vec3 h = normalize(outDir + light);

        return vec3(DistributionGGXAnisotropic(h) * GeometryVisibility(light, outDir)) * Fresnel(outDir, h, albedo);
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
        const vec3 kd = (1.0 - Fresnel(outDir, h, albedo)) * (1.0 - metallic);
        const vec3 diffuse = DiffuseBRDF(ndv, clamp(ndl, 0.0, 1.0), ldh, roughness, albedo);
        const vec3 brdf = (diffuse * kd + SpecularBRDF(lightDir.xyz, outDir, albedo)) * lightIntensity.xyz * dot(normal_v, lightDir.xyz);
        // const vec3 brdf = DiffuseBRDF(albedo);

        return brdf;
    }

    vec3 CalcImageLightReflectIntensity(vec3 outDir) {
        const vec3 h = normalize(outDir + lightDir.xyz);
        const vec3 albedo = baseColor.xyz;

        // bending normal for anisotropic: https://google.github.io/filament/Filament.html#lighting/imagebasedlights/anisotropy
        const vec3 anisoTan = cross(binormal_v, outDir);
        const vec3 anisoNormal = cross(anisoTan, binormal_v);
        const vec3 bentNormal = normalize(mix(normal_v, anisoNormal, anisotropic));

        const vec3 f = Fresnel(outDir, h, albedo);
        const vec3 kd = (1.0f - f) * (1.0f - metallic);

        const vec3 diffuse = texture(envIrradianceMap, bentNormal).xyz * albedo;

        const vec2 envBRDF = texture(envPrecomputedBRDF, vec2(max(0.0f, dot(bentNormal, outDir)), roughness)).xy;
        const vec3 specular = (f * envBRDF.x + envBRDF.y) * textureLod(envPrefilteredMap, reflect(-outDir, bentNormal), roughness * MAX_IBL_REFLECTION_LOD).xyz;

        return kd * diffuse + specular;
    }
}
FragmentShader {
    // outdir: raster point to camera ray (outgoing from surface)
    const vec3 outDir = normalize(cameraPos.xyz - worldPos.xyz);
    Target[0] = vec4(CalcImageLightReflectIntensity(outDir) + CalcDirectionalLightReflectIntensity(outDir), 1.0) * baseColor.a;
    // Target[0] = vec4(binormal_v * 0.5f + 0.5f, 1.0f);
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
SamplerCube[FragmentShader](3, 1) envPrefilteredMap
Sampler2D[FragmentShader](3, 2) envPrecomputedBRDF
