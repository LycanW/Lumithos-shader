#ifndef LIB_SKY_SHADOW_GLSL
#define LIB_SKY_SHADOW_GLSL

#include "/lib/Resources.glsl"
#include "/lib/Atmosphere.glsl"

// Single-transform shadow-screen helper: computes undistorted clip once to derive exact
// local distorted texel size, sin(theta)-exact normal offset (clears single-texel staircase
// quantization at every grazing angle with minimal corner bias), and the exact 2D analytical
// receiver-plane depth gradient d(sp.z)/d(uv) in distorted shadow-screen space.
vec3 worldToShadowScreenBiased(vec3 worldPos, vec3 normal, vec3 pullbackDir, float NdotL,
                               float normalTexelMul, float minOff, float maxOff,
                               out float outTexelWorld, out vec2 outDepthGrad) {
    mat3 smRot = mat3(shadowModelView);
    vec3 svPos = (shadowModelView * vec4(worldPos, 1.0)).xyz;
    vec3 rawClip = (shadowProjection * vec4(svPos, 1.0)).xyz;
    float rRaw = length(rawClip.xy);
    float b = shadowDistortBias();
    float factor = shadowDistortFactor(rawClip.xy);
    float baseTexel = (2.0 * shadowDistance) / float(shadowMapResolution);
    outTexelWorld = baseTexel * (factor * factor / b);

    // Exact minimal normal offset to clear 1 distorted shadow texel: sqrt(2)/2 * texelWorld * sin(theta).
    float sinTheta = sqrt(max(1.0 - NdotL * NdotL, 0.0));
    float normalOffset = clamp(outTexelWorld * normalTexelMul * (0.35 + 0.95 * sinTheta), minOff, maxOff);

    vec3 worldBias = normal * normalOffset + pullbackDir * 0.008;
    vec3 clipBias  = mat3(shadowProjection) * (smRot * worldBias);
    vec3 sp = rawClip + clipBias;

    // Exact analytical receiver-plane depth gradient d(sp.z)/d(uv) via inverse Jacobian of distortShadowXY:
    // keeps blocker-search and PCF taps strictly on the tilted receiver plane even when nearly parallel to sunlight!
    vec3 Ns = normalize(smRot * normal);
    float zScale = 0.25 * shadowProjection[2][2]; // negative for OpenGL ortho
    vec2 invProjXY = vec2(1.0 / shadowProjection[0][0], 1.0 / shadowProjection[1][1]);
    vec2 dSpZ_dRaw = (-(Ns.xy / max(Ns.z, 0.05)) * invProjXY) * zScale;
    vec2 radDir = (rRaw > 1e-5) ? (rawClip.xy / rRaw) : vec2(0.0);
    float radExtra = factor * (factor - b) * (1.0 / b);
    outDepthGrad = 2.0 * (factor * dSpZ_dRaw + radExtra * dot(dSpZ_dRaw, radDir) * radDir);

    sp.xy = distortShadowXY(sp.xy);
    sp.z *= 0.5;
    return sp * 0.5 + 0.5;
}

float shadowScreenZPerBlock() {
    // clipPos.z *= 0.5 and sp.z = clipPos.z * 0.5 + 0.5 => 0.25 * |shadowProjection[2][2]|
    return 0.25 * abs(shadowProjection[2][2]);
}

// Evaluate single shadow tap with opaque (shadowtex1) + colored translucent (shadowtex0 + shadowcolor0).
vec3 evalShadowTapColored(vec2 uv, float zRef) {
    float dOpaque = textureLod(shadowtex1, uv, 0.0).r;
    if (zRef > dOpaque) return vec3(0.0);

    float dAll = textureLod(shadowtex0, uv, 0.0).r;
    if (zRef > dAll) {
        vec4 sc = textureLod(shadowcolor0, uv, 0.0);
        if (sc.a < 0.05) {
            float waterPath = min((zRef - dAll) / max(shadowScreenZPerBlock(), 1e-6), 64.0);
            return pow(sc.rgb, vec3(waterPath));
        }
        return evalGlassTransmittance(sc.rgb, sc.a);
    }
    return vec3(1.0);
}

// 2x2 Sub-Texel Bilinear Shadow Tap via single-instruction textureGather:
// - Smoothly interpolates shadow edges INSIDE every shadow texel (eliminates staircase jaggedness
//   even at razor-sharp contact shadows where PCSS penumbra < 1 texel!).
// - Tracks exact per-texel receiver-plane depth gradient across all 4 corners of the 2x2 quad.
// - Preserves raw shadowtex0/shadowtex1 depth reads for PCSS blocker distance, caustics, and SSS.
vec3 evalShadowTapBilinear(vec2 uv, float zRef, vec2 depthGrad, float invRes, float zPerBlock) {
    float fRes = float(shadowMapResolution);
    vec2 texelPos = uv * fRes - 0.5;
    vec2 f = fract(texelPos);
    vec2 gatherUV = (floor(texelPos) + 1.0) * invRes;

    // Exact per-texel receiver-plane depth offset across the 2x2 gather footprint (clamped for corner safety).
    float maxCornerDz = 0.65 * zPerBlock;
    float dZx = clamp(depthGrad.x * invRes, -maxCornerDz, maxCornerDz);
    float dZy = clamp(depthGrad.y * invRes, -maxCornerDz, maxCornerDz);
    float z00 = zRef - (f.x * dZx + f.y * dZy);
    // OpenGL textureGather layout: .x=(0,1), .y=(1,1), .z=(1,0), .w=(0,0)
    vec4 z4 = vec4(z00 + dZy, z00 + dZx + dZy, z00 + dZx, z00);

    vec4 dOpaque4 = textureGather(shadowtex1, gatherUV, 0);
    vec4 litOp4 = step(z4, dOpaque4); // 1.0 when zRef <= dOpaque (unoccluded)
    float visOpaque = mix(mix(litOp4.w, litOp4.z, f.x), mix(litOp4.x, litOp4.y, f.x), f.y);
    if (visOpaque <= 1e-4) return vec3(0.0);

    vec4 dAll4 = textureGather(shadowtex0, gatherUV, 0);
    // Fast path (~99% of world): no translucent occluder (glass/water) in front of opaque shadowtex1.
    if (all(greaterThanEqual(dAll4, dOpaque4 - vec4(1e-6)))) {
        return vec3(visOpaque);
    }

    vec4 hitTrans4 = step(dAll4, z4) * litOp4;
    float transCov = mix(mix(hitTrans4.w, hitTrans4.z, f.x), mix(hitTrans4.x, hitTrans4.y, f.x), f.y);
    if (transCov <= 1e-4) return vec3(visOpaque);

    vec4 sc = textureLod(shadowcolor0, uv, 0.0);
    vec3 tint;
    if (sc.a < 0.05) {
        float dCenter = textureLod(shadowtex0, uv, 0.0).r;
        float waterPath = min(max(zRef - dCenter, 0.0) / max(zPerBlock, 1e-6), 64.0);
        tint = pow(max(sc.rgb, vec3(1e-4)), vec3(waterPath));
    } else {
        tint = evalGlassTransmittance(sc.rgb, sc.a);
    }
    return mix(vec3(visOpaque), tint * visOpaque, saturate(transCov / max(visOpaque, 1e-4)));
}

// Fast 1-tap colored shadow lookup for secondary bounces, reflections & Radiance Cache probes.
vec3 sampleShadowFastRay(vec3 worldPos, vec3 normal, vec3 inRayDir) {
    vec3 lightDir = getShadowLightDir();
    float NdotL = dot(normal, lightDir);
    if (NdotL <= 0.0) return vec3(0.0);

    float texelWorld;
    vec2 depthGrad;
    vec3 sp = worldToShadowScreenBiased(worldPos, normal, -inRayDir, NdotL, 1.25, 0.012, 0.22, texelWorld, depthGrad);
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return vec3(1.0);

    float bias = max(0.015 * shadowScreenZPerBlock(), 2.0e-5);
    return evalShadowTapColored(sp.xy, sp.z - bias);
}

// Contact-Hardening Blue-Noise PCSS (8-tap Vogel Blocker Search + 16-tap 2x2 Bilinear Gather PCF)
// - Every tap uses single-instruction 2x2 textureGather bilinear filtering (64 effective sub-texel
//   comparisons/frame * TAA -> silky smooth edges even at razor-sharp contact points!).
// - Exact receiver-plane depth bias keeps grazing walls (parallel to sunlight) 100% free of diagonal
//   self-shadow banding while preserving zero corner light leaks.
// - Outputs smooth spherical + border coverage (`outShadowCoverage`) so deferred12.csh can
//   seamlessly hand off to distant screen-space sun shadows + skyLight occlusion with zero ring,
//   and `outWaterPath` (sunlight travel distance through water) for projected underwater caustics.
vec3 sampleShadowPCSS(vec3 worldPos, vec3 normal, float blueNoise01, out float outShadowCoverage, out float outWaterPath) {
    outShadowCoverage = 0.0;
    outWaterPath = 0.0;
    vec3 lightDir = getShadowLightDir();
    float NdotL = dot(normal, lightDir);
    if (NdotL <= 0.0) return vec3(0.0);

    float worldDist = length(worldPos);
    float distFade = smoothstep(shadowDistance, shadowDistance * 0.82, worldDist);
    if (distFade <= 1e-4) return vec3(1.0);

    vec3 eyeToSurf = worldPos - gbufferModelViewInverse[3].xyz;
    float dist = length(eyeToSurf);
    vec3 toCam = -eyeToSurf / max(dist, 1e-4);
    float texelWorld;
    vec2 depthGrad;
    // Tight max normal offset near player (zero corner leaks), smoothly relaxed for far mountains (zero distant acne).
    float maxNormalOff = mix(0.18, 0.65, smoothstep(48.0, 220.0, worldDist));
    vec3 sp = worldToShadowScreenBiased(worldPos, normal, toCam, NdotL, 1.35, 0.012, maxNormalOff, texelWorld, depthGrad);
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return vec3(1.0);

    float edgeDist = max(abs(sp.x - 0.5), abs(sp.y - 0.5));
    float uvFade = smoothstep(0.50, 0.44, edgeDist);
    outShadowCoverage = min(distFade, uvFade);
    if (outShadowCoverage <= 1e-4) return vec3(1.0);

    float zPerBlock = shadowScreenZPerBlock();
    float baseBias = max(0.014 * zPerBlock, 2.0e-5);
    float zRef = sp.z - baseBias;
    float invRes = 1.0 / float(shadowMapResolution);
    float maxPlaneDelta = 2.5 * zPerBlock;

    float angle = TWO_PI * blueNoise01;
    const float GOLDEN_ANGLE = 2.39996323;

    // 1) Blocker search (8-tap Vogel disk) with exact receiver-plane depth tracking.
    float searchRadiusUV = clamp((0.55 / max(texelWorld, 1e-4)) * invRes, 1.8 * invRes, 10.0 * invRes);
    float blockerSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < 8; ++i) {
        float r = sqrt((float(i) + 0.5) * (1.0 / 8.0)) * searchRadiusUV;
        float a = float(i) * GOLDEN_ANGLE + angle;
        vec2 uvOff = vec2(cos(a), sin(a)) * r;
        float tapZRef = zRef + clamp(dot(uvOff, depthGrad), -maxPlaneDelta, maxPlaneDelta);
        float d = textureLod(shadowtex0, sp.xy + uvOff, 0.0).r;
        if (d < tapZRef) {
            blockerSum += max(tapZRef - d, 0.0);
            ++blockerCount;
        }
    }

    // Also inspect the center 2x2 gather quad so tight contact shadow edges never early-out!
    vec2 centerGatherUV = (floor(sp.xy * float(shadowMapResolution) - 0.5) + 1.0) * invRes;
    vec4 centerD4 = textureGather(shadowtex0, centerGatherUV, 0);
    if (blockerCount == 0 && all(greaterThanEqual(centerD4, vec4(zRef)))) {
        return vec3(1.0);
    }

    // 1 L1-cached center tap detects water entry depth along lightDir for exact underwater caustics.
    float centerAll = textureLod(shadowtex0, sp.xy, 0.0).r;
    if (zRef > centerAll) {
        vec4 centerSc = textureLod(shadowcolor0, sp.xy, 0.0);
        if (centerSc.a < 0.05) {
            outWaterPath = min((zRef - centerAll) / max(zPerBlock, 1e-6), 64.0);
        }
    }

    float avgBlockerDistZ = (blockerCount > 0) ? (blockerSum / float(blockerCount)) : 0.0;
    float blockerDistBlocks = avgBlockerDistZ / max(zPerBlock, 1e-6);

    // Physical sun angular size (~1.5 deg radius) -> crisp at contact, soft far from occluder.
    float penumbraWorld = clamp(blockerDistBlocks * 0.028, 0.025, 0.85);
    float pcfRadiusUV = clamp((penumbraWorld / max(texelWorld, 1e-4)) * invRes, 0.85 * invRes, 12.0 * invRes);
    float pcfBias = max((0.014 + 0.020 * penumbraWorld) * zPerBlock, 2.0e-5);

    // 2) Variable-penumbra colored 2x2-bilinear PCF (16-tap Vogel disk -> 64 sub-texel compares/frame!).
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 16; ++i) {
        float r = sqrt((float(i) + 0.5) * (1.0 / 16.0)) * pcfRadiusUV;
        float a = float(i) * GOLDEN_ANGLE + angle + 0.7;
        vec2 uvOff = vec2(cos(a), sin(a)) * r;
        float tapZRef = sp.z - pcfBias + clamp(dot(uvOff, depthGrad), -maxPlaneDelta, maxPlaneDelta);
        sum += evalShadowTapBilinear(sp.xy + uvOff, tapZRef, depthGrad, invRes, zPerBlock);
    }
    return sum * (1.0 / 16.0);
}

vec3 sampleShadowPCSS(vec3 worldPos, vec3 normal, float blueNoise01, out float outShadowCoverage) {
    float wp;
    return sampleShadowPCSS(worldPos, normal, blueNoise01, outShadowCoverage, wp);
}

// Convenience overload for translucent pass (fades smoothly to 1.0 past shadowDistance).
vec3 sampleShadowPCSS(vec3 worldPos, vec3 normal, float blueNoise01) {
    float cov, wp;
    vec3 vis = sampleShadowPCSS(worldPos, normal, blueNoise01, cov, wp);
    return mix(vec3(1.0), vis, cov);
}

// Physical shadow-map optical thickness transmission for Subsurface Scattering:
// - Never steps +lightDir blindly through 1-block solid walls (eliminates corridor light leaks
//   and grazing diagonal shadow-texel stripes on LabPBR SSS stone/quartz/wool/ice!).
// - Solid blocks step inward (-geoNormal) with short-range extinction (sigma = 7.5) so 1-block
//   walls stay airtight while thin edges/candles/crystals glow softly.
// - Thin foliage/plants (leaves, grass, vines) use porous canopy extinction (sigma = 1.35).
vec3 evalShadowSSS(vec3 worldPos, vec3 geoNormal, vec3 lightDir, bool isFoliage, float blueNoise01) {
    float worldDist = length(worldPos);
    float distFade = smoothstep(shadowDistance, shadowDistance * 0.78, worldDist);
    if (distFade <= 1e-4) return vec3(0.0);

    vec3 svRaw = (shadowModelView * vec4(worldPos, 1.0)).xyz;
    vec3 rawClip = (shadowProjection * vec4(svRaw, 1.0)).xyz;
    float b = shadowDistortBias();
    float factor = shadowDistortFactor(rawClip.xy);
    float baseTexel = (2.0 * shadowDistance) / float(shadowMapResolution);
    float texelWorld = baseTexel * (factor * factor / b);

    // Solid SSS stays strictly inside the block (-geoNormal) so grazing taps never spill into
    // open air behind the wall; thin foliage uses a tiny forward offset along lightDir.
    vec3 samplePos = isFoliage
        ? (worldPos + lightDir * 0.04)
        : (worldPos - geoNormal * clamp(1.6 * texelWorld, 0.05, 0.22));
    vec3 svPos = (shadowModelView * vec4(samplePos, 1.0)).xyz;
    vec3 sp = (shadowProjection * vec4(svPos, 1.0)).xyz;
    sp.xy = distortShadowXY(sp.xy);
    sp.z *= 0.5;
    sp = sp * 0.5 + 0.5;
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return vec3(0.0);

    float edgeDist = max(abs(sp.x - 0.5), abs(sp.y - 0.5));
    float uvFade = smoothstep(0.50, 0.44, edgeDist);
    float cov = min(distFade, uvFade);
    if (cov <= 1e-4) return vec3(0.0);

    float zPerBlock = shadowScreenZPerBlock();
    float invRes = 1.0 / float(shadowMapResolution);
    float sigma = isFoliage ? 1.35 : 7.5;
    float radiusUV = isFoliage
        ? clamp((0.14 / max(texelWorld, 1e-4)) * invRes, 1.0 * invRes, 4.0 * invRes)
        : (0.75 * invRes);

    float angle = TWO_PI * blueNoise01;
    const float GOLDEN_ANGLE = 2.39996323;

    float transSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        float r = sqrt((float(i) + 0.5) * 0.25) * radiusUV;
        float a = float(i) * GOLDEN_ANGLE + angle;
        vec2 uv = sp.xy + vec2(cos(a), sin(a)) * r;
        float dOpaque = textureLod(shadowtex1, uv, 0.0).r;
        // On a back-lit solid wall, a tap with dOpaque >= sp.z stepped off the grazing edge into background air.
        if (!isFoliage && dOpaque >= sp.z - 0.005 * zPerBlock) continue;
        float thickness = max(sp.z - dOpaque, 0.0) / max(zPerBlock, 1e-6);
        transSum += exp(-sigma * thickness);
    }
    return vec3((transSum * 0.25) * cov);
}

#endif
