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

vec3 evalHeldLightColor(int itemId) {
    if (itemId == 20001) return vec3(0.24, 0.82, 1.00); // Soul fire
    if (itemId == 20002) return vec3(0.68, 0.94, 1.00); // Sea lantern / beacon
    if (itemId == 20003) return vec3(1.00, 0.16, 0.08); // Redstone
    if (itemId == 20004) return vec3(1.00, 0.90, 0.72); // End rod / glowstone
    if (itemId == 20005) return vec3(0.76, 0.36, 1.00); // Amethyst / arcane
    if (itemId == 20006) return vec3(0.45, 1.00, 0.56); // Verdant froglight
    return vec3(1.00, 0.62, 0.26);                      // Torch / lantern / lava default
}

// Perspective-exact clip-space depth contact shadow for held-item dynamic lights:
// - Interpolates homogeneous clip coordinates (sampleClip.z == exact linear view depth) to avoid
//   non-linear NDC depth self-intersection on front-facing close surfaces.
// - Uses pixel-world-size grazing bias, near-plane ray clamping, cubic contact-concentrated steps,
//   and smooth front-penumbra / back-thickness occlusion (ignoring the hand holding the light).
float traceHeldLightShadow(vec3 viewPos, vec3 viewGeoN, vec3 lightViewPos, vec3 viewLightDir, bool isOriginHand, float noise) {
    float viewZ = -viewPos.z;
    vec2 projScale = vec2(gbufferProjection[0][0], gbufferProjection[1][1]);
    vec2 invRes = texelSize();
    float pixelWorldSize = 2.0 * viewZ * max(invRes.x / max(abs(projScale.x), 1e-4), invRes.y / max(abs(projScale.y), 1e-4));
    float geoNdotL = saturate(dot(viewGeoN, viewLightDir));

    vec3 startView = viewPos
        + viewGeoN * min(pixelWorldSize * (0.9 + 1.0 / max(geoNdotL, 0.15)), 0.08)
        + viewLightDir * min(pixelWorldSize * 1.0, 0.04);

    vec3 endView = lightViewPos;
    if (-endView.z < 0.15) {
        float tClamp = saturate((startView.z + 0.15) / min(startView.z - lightViewPos.z, -1e-4));
        endView = mix(startView, lightViewPos, tClamp);
    }
    float traceDist = length(endView - startView);
    if (traceDist <= 1e-3) return 1.0;

    vec3 startClip = vec3(projScale * startView.xy + gbufferProjection[2].xy * startView.z, -startView.z);
    vec3 endClip   = vec3(projScale * endView.xy   + gbufferProjection[2].xy * endView.z,   -endView.z);
    vec3 rayClip   = endClip - startClip;

    const int STEPS = 16;
    const float invSamples = 1.0 / float(STEPS);
    vec2 res = screenSize();
    float maxOcc = 0.0;

    for (int i = 0; i < STEPS; ++i) {
        float u = (float(i) + noise) * invSamples;
        float t = u * (0.10 + u * (0.30 + 0.60 * u));

        vec3 sampleClip = startClip + rayClip * t;
        if (sampleClip.z <= 0.15) break;

        vec2 sampleUV = (sampleClip.xy / sampleClip.z) * 0.5 + 0.5;
        #ifdef TAA_ENABLED
            sampleUV += taaJitter * 0.5;
        #endif
        if (any(lessThan(sampleUV, vec2(0.001))) || any(greaterThanEqual(sampleUV, vec2(0.999)))) break;

        ivec2 q = ivec2(sampleUV * res);
        float sd = texelFetch(depthtex1, q, 0).r;
        if (sd >= 0.999999) continue;

        float sampleViewZ = linearizeDepth(sd);
        float depthDiff = sampleClip.z - sampleViewZ;
        if (depthDiff > 0.0) {
            uint w1 = texelFetch(colortex1, q, 0).g;
            bool isHandSample = ((w1 >> 24u) & 1u) != 0u;
            // Ignore the hand holding this light source while keeping the opposite hand as a caster.
            if (isHandSample && sampleClip.x * lightViewPos.x > 0.0) continue;

            float travelDist = traceDist * t;
            float stepZ = abs(rayClip.z) * (0.10 + u * (0.60 + 1.80 * u)) * invSamples;
            float penumbra;
            float thickness;
            if (isOriginHand) {
                penumbra  = max(pixelWorldSize * 2.0, 0.006 + travelDist * 0.05);
                thickness = clamp(0.10 + travelDist * 0.50 + stepZ * 1.5, 0.10, 0.55);
            } else if (isHandSample) {
                penumbra  = max(pixelWorldSize * 2.0, 0.015 + travelDist * 0.06);
                thickness = clamp(0.20 + travelDist * 0.50 + stepZ * 1.5, 0.20, 0.80);
            } else {
                penumbra  = max(pixelWorldSize * 2.5, 0.008 + travelDist * 0.07);
                thickness = clamp(0.15 + travelDist * 0.65 + stepZ * 2.2, 0.15, 2.40);
            }

            float frontOcc = smoothstep(0.0, penumbra, depthDiff);
            float backFade = 1.0 - smoothstep(thickness * 0.55, thickness, depthDiff);
            maxOcc = max(maxOcc, frontOcc * backFade);
            if (maxOcc > 0.99) break;
        }
    }

    return 1.0 - 0.82 * maxOcc;
}

void evalHeldHandLight(int lightVal, int itemId, vec3 handViewPos, GBufferMat mat, vec3 viewPos, vec3 viewDir,
                       vec3 F0_rgb, bool isMetal, float NdotV, vec3 viewGeoN, float dither,
                       inout vec3 outDiffuse, inout vec3 outSpec) {
    if (lightVal <= 0) return;

    bool isSameHand = mat.isHand && (viewPos.x * handViewPos.x > 0.0);
    vec3 effectiveLightPos;
    if (isSameHand) {
        effectiveLightPos = vec3(0.0);
    } else if (mat.isHand) {
        effectiveLightPos = handViewPos;
    } else {
        effectiveLightPos = handViewPos * smoothstep(0.05, 0.35, -viewPos.z);
    }

    vec3 toLightView = effectiveLightPos - viewPos;
    float dist = length(toLightView);
    float maxRange = 4.0 + float(lightVal) * 0.95;
    if (dist >= maxRange) return;

    vec3 viewLightDir = toLightView / max(dist, 1e-4);
    vec3 L = normalize(mat3(gbufferModelViewInverse) * viewLightDir);
    float NdotL = max(dot(mat.shadeNormal, L), 0.0);
    float geoHorizon = smoothstep(0.0, 0.08, dot(mat.geoNormal, L));
    float backNL = (mat.sss > 0.0 && !mat.isHand) ? max(dot(-mat.geoNormal, L), 0.0) : 0.0;
    if ((NdotL <= 0.0 || geoHorizon <= 0.0) && backNL <= 0.0) return;

    float rNorm = dist / maxRange;
    float r2 = rNorm * rNorm;
    float window = saturate(1.0 - r2 * r2);
    float atten = (window * window) / (dist * dist + 0.35);
    float power = (float(lightVal) / 15.0) * 5.2;
    vec3 lightRad = evalHeldLightColor(itemId) * (power * atten);

    float vis = 0.0;
    if (NdotL > 0.0 && geoHorizon > 0.0) {
        float shadow = 1.0;
        if (!isSameHand) {
            shadow = traceHeldLightShadow(viewPos, viewGeoN, effectiveLightPos, viewLightDir, mat.isHand, dither);
            float shadowDistFade = 1.0 - smoothstep(maxRange * 0.65, maxRange, dist);
            shadow = mix(1.0, shadow, shadowDistFade);
        }
        vis = shadow * geoHorizon;
    }

    if (vis > 1e-4) {
        vec3 H = normalize(L + viewDir);
        float NdotH = max(dot(mat.shadeNormal, H), 0.0);
        float VdotH = saturate(dot(viewDir, H));
        float alpha = max(mat.roughness * mat.roughness, 0.015);
        float a2 = alpha * alpha;
        float dDenom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = min(a2 / max(PI * dDenom * dDenom, 1e-5), 256.0);
        float Vis = visSmithGGX(NdotV, NdotL, alpha);
        vec3 F = fresnelSchlick(F0_rgb, VdotH);
        vec3 kD = isMetal ? vec3(0.0) : (1.0 - F);

        outDiffuse += mat.albedo * INV_PI * kD * lightRad * (NdotL * vis);
        outSpec    += F * (D * Vis) * lightRad * (NdotL * vis);
    }

    if (backNL > 0.0) {
        outDiffuse += mat.albedo * mat.albedo * INV_PI * lightRad * (mat.sss * backNL * 0.45);
    }
}

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
    vec3 emissive   = (mat.emission > 0.0) ? (mat.albedo * (mat.emission * EMISSIVE_SCALE)) : vec3(0.0);

    // Physical GGX split-sum environment BRDF remodulation + energy conservation:
    // - Metals use traced GGX specular across the entire [0..1] roughness range (never black!).
    // - Dielectrics smoothly cross-fade from traced GGX to diffuseGI across [0.50..0.65] (zero cutoff cliff!).
    vec3 envBrdf = evalEnvBRDF(F0_rgb, NdotV, mat.roughness);
    vec3 kD_env  = isMetal ? vec3(0.0) : (1.0 - envBrdf);
    float specTraceFade = isMetal ? 1.0 : smoothstep(SPECULAR_ROUGH_CUTOFF, 0.50, mat.roughness);
    vec3 specEnv = mat.isHand ? diffuseGI : mix(diffuseGI, specularGI, specTraceFade);

    // LabPBR material AO (normals.b) on indirect diffuse + Lagarde specular occlusion on indirect specular.
    float specOcc = saturate(pow(NdotV + mat.ao, exp2(-16.0 * mat.roughness - 1.0)) - 1.0 + mat.ao);

    // Handheld dynamic lighting (main hand + off-hand) with perspective-exact screen-space contact shadows.
    // Uniform early-out costs 0.00ms whenever neither hand holds an emissive item!
    vec3 heldDiffuse = vec3(0.0);
    vec3 heldSpec    = vec3(0.0);
    if (heldBlockLightValue > 0 || heldBlockLightValue2 > 0) {
        vec3 viewGeoN = normalize(mat3(gbufferModelView) * mat.geoNormal);
        float heldDither = stbn2D(px, frameCounter, 14u).x;
        evalHeldHandLight(heldBlockLightValue,  heldItemId,  vec3( 0.45, 0.0, 0.0), mat, viewPos, viewDir,
                          F0_rgb, isMetal, NdotV, viewGeoN, heldDither, heldDiffuse, heldSpec);
        evalHeldHandLight(heldBlockLightValue2, heldItemId2, vec3(-0.45, 0.0, 0.0), mat, viewPos, viewDir,
                          F0_rgb, isMetal, NdotV, viewGeoN, fract(heldDither + 0.5), heldDiffuse, heldSpec);
    }

    vec3 diffuseLit = directDiffuse * kD_direct + sssLight * kD_direct + heldDiffuse + (diffuseGI * mat.albedo) * (kD_env * mat.ao) + emissive;
    vec3 finalLit   = diffuseLit + directSpec + heldSpec + specEnv * envBrdf * specOcc;

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
