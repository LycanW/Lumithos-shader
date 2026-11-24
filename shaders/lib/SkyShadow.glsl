#ifndef LIB_SKY_SHADOW_GLSL
#define LIB_SKY_SHADOW_GLSL

#include "/lib/Resources.glsl"

// Clean procedural day/sunset/night sky for MVP (no volumetric clouds/fog yet).
vec3 evalSky(vec3 dir, bool includeDisc) {
    vec3 sunDir = getSunDir();
    float sunElev = sunDir.y;
    float dayFactor = smoothstep(-0.12, 0.18, sunElev);
    float sDist = (sunElev - 0.02) * 8.0;
    float sunsetFactor = exp(-(sDist * sDist));

    float up = saturate(dir.y);
    float horizon = pow(1.0 - up, 4.0);

    vec3 dayZenith   = vec3(0.14, 0.36, 0.85);
    vec3 dayHorizon  = vec3(0.55, 0.74, 0.96);
    vec3 sunsetHoriz = vec3(1.15, 0.42, 0.16);
    vec3 nightZenith = vec3(0.004, 0.007, 0.018);
    vec3 nightHoriz  = vec3(0.012, 0.018, 0.036);

    vec3 zenith  = mix(nightZenith, dayZenith, dayFactor);
    vec3 horiz   = mix(nightHoriz, dayHorizon, dayFactor);
    horiz = mix(horiz, sunsetHoriz, sunsetFactor * 0.65);

    vec3 sky = mix(zenith, horiz, horizon);
    sky *= mix(1.0, 0.45, rainStrength);

    if (includeDisc) {
        float sunCos = dot(dir, sunDir);
        float sunDisc = smoothstep(0.9993, 0.9997, sunCos) * dayFactor * (1.0 - rainStrength);
        float sunGlow = pow(max(sunCos, 0.0), 32.0) * (0.25 * dayFactor + 0.5 * sunsetFactor) * (1.0 - rainStrength);
        sky += vec3(12.0, 10.5, 8.5) * sunDisc + vec3(1.0, 0.65, 0.35) * sunGlow;

        vec3 moonDir = -sunDir;
        float moonCos = dot(dir, moonDir);
        float moonDisc = smoothstep(0.9995, 0.9998, moonCos) * (1.0 - dayFactor) * (1.0 - rainStrength);
        sky += vec3(1.2, 1.3, 1.6) * moonDisc;
    }

    return sky;
}

vec3 getDirectLightColor() {
    vec3 sunDir = getSunDir();
    float sunElev = sunDir.y;
    float dayFactor = smoothstep(-0.08, 0.12, sunElev);
    float sDist = (sunElev - 0.02) * 8.0;
    float sunsetFactor = exp(-(sDist * sDist));

    vec3 sunCol = mix(vec3(1.0, 0.95, 0.88), vec3(1.0, 0.48, 0.22), sunsetFactor * 0.75) * 4.2;
    vec3 moonCol = vec3(0.18, 0.24, 0.38) * 0.16;

    float fade = smoothstep(0.0, 0.08, abs(sunElev));
    vec3 lightCol = isNight() ? moonCol : sunCol;
    return lightCol * fade * (1.0 - 0.8 * rainStrength);
}

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
    float factor = shadowDistortFactor(rawClip.xy);
    float baseTexel = (2.0 * shadowDistance) / float(shadowMapResolution);
    outTexelWorld = baseTexel * (factor * factor / 0.15);

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
    float radExtra = factor * (factor - 0.15) * (1.0 / 0.15);
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

// Contact-Hardening Blue-Noise PCSS (8-tap Vogel Blocker Search + 12-tap Variable-Penumbra Colored PCF)
// - Exact receiver-plane depth bias keeps grazing walls (parallel to sunlight) 100% free of diagonal
//   self-shadow banding while preserving razor-sharp contact shadows and zero corner light leaks.
// - Lossless early-out when 0 blockers are found or outside shadowDistance.
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
    float distFade = smoothstep(shadowDistance, shadowDistance * 0.78, worldDist);
    if (distFade <= 1e-4) return vec3(1.0);

    vec3 eyeToSurf = worldPos - gbufferModelViewInverse[3].xyz;
    float dist = length(eyeToSurf);
    vec3 toCam = -eyeToSurf / max(dist, 1e-4);
    float texelWorld;
    vec2 depthGrad;
    vec3 sp = worldToShadowScreenBiased(worldPos, normal, toCam, NdotL, 1.35, 0.012, 0.18, texelWorld, depthGrad);
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
    float searchRadiusUV = clamp((0.55 / max(texelWorld, 1e-4)) * invRes, 1.5 * invRes, 10.0 * invRes);
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

    // Fully lit early-out (0 blockers within search cone).
    if (blockerCount == 0) return vec3(1.0);

    // 1 L1-cached center tap detects water entry depth along lightDir for exact underwater caustics.
    float centerAll = textureLod(shadowtex0, sp.xy, 0.0).r;
    if (zRef > centerAll) {
        vec4 centerSc = textureLod(shadowcolor0, sp.xy, 0.0);
        if (centerSc.a < 0.05) {
            outWaterPath = min((zRef - centerAll) / max(zPerBlock, 1e-6), 64.0);
        }
    }

    float avgBlockerDistZ = blockerSum / float(blockerCount);
    float blockerDistBlocks = avgBlockerDistZ / max(zPerBlock, 1e-6);

    // Physical sun angular size (~1.5 deg radius) -> crisp at contact, soft far from occluder.
    float penumbraWorld = clamp(blockerDistBlocks * 0.028, 0.02, 0.85);
    float pcfRadiusUV = clamp((penumbraWorld / max(texelWorld, 1e-4)) * invRes, 0.6 * invRes, 12.0 * invRes);
    float pcfBias = max((0.014 + 0.020 * penumbraWorld) * zPerBlock, 2.0e-5);

    // 2) Variable-penumbra colored PCF (12-tap Vogel disk) with exact receiver-plane depth tracking.
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 12; ++i) {
        float r = sqrt((float(i) + 0.5) * (1.0 / 12.0)) * pcfRadiusUV;
        float a = float(i) * GOLDEN_ANGLE + angle + 0.7;
        vec2 uvOff = vec2(cos(a), sin(a)) * r;
        float tapZRef = sp.z - pcfBias + clamp(dot(uvOff, depthGrad), -maxPlaneDelta, maxPlaneDelta);
        sum += evalShadowTapColored(sp.xy + uvOff, tapZRef);
    }
    return sum * (1.0 / 12.0);
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
    float factor = shadowDistortFactor(rawClip.xy);
    float baseTexel = (2.0 * shadowDistance) / float(shadowMapResolution);
    float texelWorld = baseTexel * (factor * factor / 0.15);

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
