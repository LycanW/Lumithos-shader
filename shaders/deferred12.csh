#version 430 compatibility

// Pass: Combine Direct Sunlight + Emission + Diffuse GI + Specular GI + Sky + Debug Views,
// and publish prevGuideImg (exact float32 integers!) + diffuse-safe colortex5 (lit history).
#include "/lib/SkyShadow.glsl"
#if DEBUG_VIEW == 4
    #include "/lib/RadianceCache.glsl"
#endif

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D colorimg0;
layout(rgba16f) uniform writeonly image2D colorimg5;
layout(rgba32f) uniform writeonly image2D prevGuideImg;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    float depth = texelFetch(depthtex1, px, 0).r;

    if (depth >= 0.999999) {
        vec3 skyDir = normalize(mat3(gbufferModelViewInverse) * screenToView(vec3(uv, 1.0)));
        vec3 sky = evalSky(skyDir, true);
        imageStore(colorimg0, px, vec4(sky, 1.0));
        imageStore(colorimg5, px, vec4(sky, 1.0));
        imageStore(prevGuideImg, px, vec4(0.0));
        return;
    }

    GBufferMat mat = readGBuffer(px);
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);
    vec3 viewDir  = normalize(mat3(gbufferModelViewInverse) * (-viewPos));

    // Publish exact float32 integer-encoded 16-bit normals (never denormal uintBitsToFloat!).
    imageStore(prevGuideImg, px, vec4(
        float(packNormal16(mat.geoNormal)),
        float(packNormal16(mat.shadeNormal)),
        mat.roughness,
        -viewPos.z
    ));

    // Direct Sunlight / Moonlight (Contact-Hardening Blue-Noise PCSS on Shadow Map)
    // Smooth grazing-horizon fade on geoNormal prevents normal-map bumps from binary-flashing
    // when sunlight skims parallel to a wall (dot(geoNormal, lightDir) ≈ 0).
    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();
    float NdotL = max(dot(mat.shadeNormal, lightDir), 0.0);
    float geoHorizon = smoothstep(0.0, 0.08, dot(mat.geoNormal, lightDir));
    vec3 shadowVis = (NdotL > 0.0 && geoHorizon > 0.0 && maxOf(directCol) > 1e-4)
        ? sampleShadowPCSS(worldPos, mat.geoNormal, stbn2D(px, frameCounter, 7u).x) * geoHorizon
        : vec3(0.0);
    vec3 directDiffuse = mat.albedo * INV_PI * directCol * (NdotL * shadowVis);

    // LabPBR 1.3 subsurface scattering (thin-slab back-lit transmission, Henyey-Greenstein forward lobe).
    // Back-face visibility is evaluated against the shadow map from the opposite side of the slab.
    vec3 sssLight = vec3(0.0);
    if (mat.sss > 0.0 && maxOf(directCol) > 1e-4) {
        float backNL = dot(-mat.geoNormal, lightDir);
        if (backNL > 0.0) {
            vec3 transVis = sampleShadowPCSS(worldPos, -mat.geoNormal, stbn2D(px, frameCounter, 11u).x);
            float mu = dot(-viewDir, lightDir);
            const float g = 0.55;
            float hg = (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * mu, 1e-4), 1.5));
            float phase = 0.35 * INV_PI + 0.65 * hg;
            sssLight = mat.albedo * mat.albedo * directCol * transVis * (mat.sss * phase * saturate(backNL + 0.25));
        }
    }

    // Direct Specular Highlight (Cook-Torrance GGX with Height-Correlated Smith Visibility & LabPBR RGB F0)
    vec3 F0_rgb = evalLabPBRF0(mat.f0, mat.albedo);
    bool isMetal = isLabPBRMetal(mat.f0);
    float NdotV = max(dot(mat.shadeNormal, viewDir), 1e-4);
    vec3 directSpec = vec3(0.0);
    vec3 kD_direct = isMetal ? vec3(0.0) : (1.0 - evalEnvBRDF(F0_rgb, NdotV, mat.roughness));
    if (NdotL > 0.0 && maxOf(shadowVis) > 0.0) {
        vec3 H = normalize(lightDir + viewDir);
        float NdotH = max(dot(mat.shadeNormal, H), 0.0);
        float VdotH = saturate(dot(viewDir, H));
        float alpha = max(mat.roughness * mat.roughness, 0.015);
        float a2 = alpha * alpha;
        float dDenom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = min(a2 / max(PI * dDenom * dDenom, 1e-5), 256.0);
        float Vis = visSmithGGX(NdotV, NdotL, alpha);
        vec3 F = fresnelSchlick(F0_rgb, VdotH);
        directSpec = F * (D * Vis) * directCol * (NdotL * shadowVis);
        if (!isMetal) kD_direct = 1.0 - F;
    }

    vec3 diffuseGI  = max(texelFetch(colortex3, px, 0).rgb, vec3(0.0));
    vec3 specularGI = max(texelFetch(colortex4, px, 0).rgb, vec3(0.0));
    vec3 emissive   = (mat.emission > 0.0) ? (evalDirectEmitterColor(mat.albedo) * (mat.emission * EMISSIVE_SCALE)) : vec3(0.0);

    // Physical GGX split-sum environment BRDF remodulation + energy conservation:
    // - Metals use traced GGX specular across the entire [0..1] roughness range (never black!).
    // - Dielectrics smoothly cross-fade from traced GGX to diffuseGI across [0.50..0.65] (zero cutoff cliff!).
    vec3 envBrdf = evalEnvBRDF(F0_rgb, NdotV, mat.roughness);
    vec3 kD_env  = isMetal ? vec3(0.0) : (1.0 - envBrdf);
    float specTraceFade = isMetal ? 1.0 : smoothstep(SPECULAR_ROUGH_CUTOFF, 0.50, mat.roughness);
    vec3 specEnv = mat.isHand ? diffuseGI : mix(diffuseGI, specularGI, specTraceFade);

    // LabPBR material AO (normals.b) on indirect diffuse + Lagarde specular occlusion on indirect specular.
    float specOcc = saturate(pow(NdotV + mat.ao, exp2(-16.0 * mat.roughness - 1.0)) - 1.0 + mat.ao);

    vec3 diffuseLit = directDiffuse * kD_direct + sssLight * kD_direct + (diffuseGI * mat.albedo) * (kD_env * mat.ao) + emissive;
    vec3 finalLit   = diffuseLit + directSpec + specEnv * envBrdf * specOcc;

    // Store pure diffuse+emissive radiance in colortex5.rgb, and emission flag in .a so
    // DIFFUSE_SSRT can stay enabled alongside NEE without double-counting block lamps!
    imageStore(colorimg5, px, vec4(diffuseLit, mat.emission));

    // Optional Debug Visualizations
    #if DEBUG_VIEW == 1 || DEBUG_VIEW == 2
        finalLit = diffuseGI;
    #elif DEBUG_VIEW == 3
        finalLit = specularGI;
    #elif DEBUG_VIEW == 4
        finalLit = queryRadianceCache(worldPos, mat.geoNormal, 0u, false);
    #elif DEBUG_VIEW == 5
        ivec3 vCoord = ivec3(floor(worldToVoxelPos(worldPos - mat.geoNormal * 0.05)));
        vec3 dbg = vec3(0.05, 0.05, 0.15);
        if (inVoxelBounds(vCoord)) {
            if (texelFetch(voxelGridTex, vCoord, 0).r != 0u) {
                SurfaceVoxel sv = surfacePool.surfaces[voxelLinearIndex(vCoord)];
                bool hasT1 = (sv.face[0] | sv.face[1] | sv.face[2] | sv.face[3] | sv.face[4] | sv.face[5]) != 0u;
                bool hasT2 = (sv.primHead != PRIM_END);
                dbg = vec3(hasT2 ? 1.0 : 0.0, hasT1 ? 1.0 : 0.0, 0.0);
            }
        }
        finalLit = dbg;
    #endif

    imageStore(colorimg0, px, vec4(finalLit, 1.0));
}
