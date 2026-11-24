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
// local distorted texel size, then applies linear clip-space normal/pullback offset (saves 50% mat4 ALU!).
vec3 worldToShadowScreenBiased(vec3 worldPos, vec3 normal, vec3 pullbackDir, float normalTexelMul, float minOff, float maxOff, out float outTexelWorld) {
    mat3 smRot = mat3(shadowModelView);
    vec3 svPos = (shadowModelView * vec4(worldPos, 1.0)).xyz;
    vec3 rawClip = (shadowProjection * vec4(svPos, 1.0)).xyz;
    float factor = shadowDistortFactor(rawClip.xy);
    float baseTexel = (2.0 * shadowDistance) / float(shadowMapResolution);
    outTexelWorld = baseTexel * (factor * factor / 0.15);
    float normalOffset = clamp(outTexelWorld * normalTexelMul, minOff, maxOff);

    vec3 worldBias = normal * normalOffset + pullbackDir * 0.008;
    vec3 clipBias  = mat3(shadowProjection) * (smRot * worldBias);
    vec3 sp = rawClip + clipBias;
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
    vec3 sp = worldToShadowScreenBiased(worldPos, normal, -inRayDir, 1.25, 0.014, 0.25, texelWorld);
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return vec3(1.0);

    float bias = max(0.015 * shadowScreenZPerBlock(), 2.0e-5);
    return evalShadowTapColored(sp.xy, sp.z - bias);
}

// Contact-Hardening Blue-Noise PCSS (8-tap Vogel Blocker Search + 12-tap Variable-Penumbra Colored PCF)
// - Razor-sharp shadows at contact points, physically softening with blocker distance.
// - Lossless early-out when 0 blockers are found or outside shadowDistance.
// - Outputs smooth spherical + border coverage (`outShadowCoverage`) so deferred12.csh can
//   seamlessly hand off to distant screen-space sun shadows + skyLight occlusion with zero ring!
vec3 sampleShadowPCSS(vec3 worldPos, vec3 normal, float blueNoise01, out float outShadowCoverage) {
    outShadowCoverage = 0.0;
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
    vec3 sp = worldToShadowScreenBiased(worldPos, normal, toCam, 1.65, 0.018, 0.30, texelWorld);
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return vec3(1.0);

    float edgeDist = max(abs(sp.x - 0.5), abs(sp.y - 0.5));
    float uvFade = smoothstep(0.50, 0.44, edgeDist);
    outShadowCoverage = min(distFade, uvFade);
    if (outShadowCoverage <= 1e-4) return vec3(1.0);

    float zPerBlock = shadowScreenZPerBlock();
    float baseBias = max(0.016 * zPerBlock, 2.0e-5);
    float zRef = sp.z - baseBias;
    float invRes = 1.0 / float(shadowMapResolution);

    float angle = TWO_PI * blueNoise01;
    const float GOLDEN_ANGLE = 2.39996323;

    // 1) Blocker search (8-tap Vogel disk): search radius scales with local distorted texel size (~0.55 blocks).
    float searchRadiusUV = clamp((0.55 / max(texelWorld, 1e-4)) * invRes, 1.5 * invRes, 10.0 * invRes);
    float blockerSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < 8; ++i) {
        float r = sqrt((float(i) + 0.5) * (1.0 / 8.0)) * searchRadiusUV;
        float a = float(i) * GOLDEN_ANGLE + angle;
        vec2 uv = sp.xy + vec2(cos(a), sin(a)) * r;
        float d = textureLod(shadowtex0, uv, 0.0).r;
        if (d < zRef) {
            blockerSum += d;
            ++blockerCount;
        }
    }

    // Fully lit early-out (0 blockers within search cone).
    if (blockerCount == 0) return vec3(1.0);

    float avgBlockerZ = blockerSum / float(blockerCount);
    float blockerDistBlocks = max(zRef - avgBlockerZ, 0.0) / max(zPerBlock, 1e-6);

    // Physical sun angular size (~1.5 deg radius) -> crisp at contact, soft far from occluder.
    float penumbraWorld = clamp(blockerDistBlocks * 0.028, 0.02, 0.85);
    float pcfRadiusUV = clamp((penumbraWorld / max(texelWorld, 1e-4)) * invRes, 0.6 * invRes, 12.0 * invRes);
    float pcfBias = max((0.016 + 0.022 * penumbraWorld) * zPerBlock, 2.0e-5);

    // 2) Variable-penumbra colored PCF (12-tap Vogel disk with blue-noise rotation).
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 12; ++i) {
        float r = sqrt((float(i) + 0.5) * (1.0 / 12.0)) * pcfRadiusUV;
        float a = float(i) * GOLDEN_ANGLE + angle + 0.7;
        vec2 uv = sp.xy + vec2(cos(a), sin(a)) * r;
        sum += evalShadowTapColored(uv, sp.z - pcfBias);
    }
    return sum * (1.0 / 12.0);
}

// Convenience overload for translucent pass (fades smoothly to 1.0 past shadowDistance).
vec3 sampleShadowPCSS(vec3 worldPos, vec3 normal, float blueNoise01) {
    float cov;
    vec3 vis = sampleShadowPCSS(worldPos, normal, blueNoise01, cov);
    return mix(vec3(1.0), vis, cov);
}

#endif
