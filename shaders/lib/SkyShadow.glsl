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

// Exact world-space size of 1 shadow texel at worldPos under radial shadow distortion,
// plus exact shadow-screen Z delta per world block (matches clipPos.z *= 0.5 in shadow.vsh).
float shadowTexelWorldSize(vec3 worldPos) {
    vec2 clipXY = (shadowProjection * (shadowModelView * vec4(worldPos, 1.0))).xy;
    float factor = shadowDistortFactor(clipXY);
    float baseTexel = (2.0 * shadowDistance) / float(shadowMapResolution);
    return baseTexel * (factor * factor / 0.15);
}

float shadowScreenZPerBlock() {
    // clipPos.z *= 0.5 and sp.z = clipPos.z * 0.5 + 0.5 => 0.25 * |shadowProjection[2][2]|
    return 0.25 * abs(shadowProjection[2][2]);
}

// Fast 1-tap shadow lookup for secondary bounces, reflections & Radiance Cache probes.
// Uses exact distortion-aware normal offset (1.2x local texel) + minimal 0.015-block Z epsilon
// so shadows stay tightly locked to block bases and never punch through 1-block indoor corners.
float sampleShadowFastRay(vec3 worldPos, vec3 normal, vec3 inRayDir) {
    vec3 lightDir = getShadowLightDir();
    float NdotL = dot(normal, lightDir);
    if (NdotL <= 0.0) return 0.0;

    float texelWorld = shadowTexelWorldSize(worldPos);
    float normalOffset = clamp(texelWorld * 1.25, 0.014, 0.25);
    vec3 biasedPos = worldPos + normal * normalOffset - inRayDir * 0.008;
    vec3 sp = worldToShadowScreen(biasedPos);
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return 1.0;

    float bias = max(0.015 * shadowScreenZPerBlock(), 2.0e-5);
    return step(sp.z - bias, textureLod(shadowtex0, sp.xy, 0.0).r);
}

float sampleShadowFast(vec3 worldPos, vec3 normal) {
    return sampleShadowFastRay(worldPos, normal, vec3(0.0));
}

// Soft 4-tap jittered PCF shadow for primary direct lighting.
// Normal offset scales with exact distorted shadow texel size (eliminates acne at all angles
// without needing large sunward Z-bias, preventing both peter-panning and indoor corner leaks).
float sampleShadowPCF(vec3 worldPos, vec3 normal, inout uint rng) {
    vec3 lightDir = getShadowLightDir();
    float NdotL = dot(normal, lightDir);
    if (NdotL <= 0.0) return 0.0;

    float dist = length(worldPos);
    vec3 toCam = -worldPos / max(dist, 1e-4);
    float texelWorld = shadowTexelWorldSize(worldPos);
    float normalOffset = clamp(texelWorld * 1.65, 0.018, 0.30);
    vec3 biasedPos = worldPos + normal * normalOffset + toCam * 0.008;
    vec3 sp = worldToShadowScreen(biasedPos);
    if (any(lessThan(sp, vec3(0.0))) || any(greaterThan(sp, vec3(1.0)))) return 1.0;

    float bias = max(0.015 * shadowScreenZPerBlock(), 2.0e-5);
    float invRes = 1.0 / float(shadowMapResolution);
    float angle = TWO_PI * nextFloat(rng);
    mat2 rot = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));

    const vec2 offsets[4] = vec2[4](
        vec2(-0.7, -0.3), vec2(0.7, 0.3),
        vec2(-0.3,  0.7), vec2(0.3, -0.7)
    );
    float sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        vec2 uv = sp.xy + (rot * offsets[i]) * (0.75 * invRes);
        sum += step(sp.z - bias, textureLod(shadowtex0, uv, 0.0).r);
    }
    return sum * 0.25;
}

#endif
