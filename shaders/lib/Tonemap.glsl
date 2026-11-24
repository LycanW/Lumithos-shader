#ifndef LIB_TONEMAP_GLSL
#define LIB_TONEMAP_GLSL

#include "/lib/Resources.glsl"

// ============================================================================
// Color Management: AgX (Default), GT7 (2025 Perceptual ICtCp), PBR Neutral, ACES
// All tonemappers accept linear Rec.709 HDR and return linear Rec.709 display [0, 1].
// ============================================================================

// ---- 1. AgX (Blender 4.0 / Troy Sobotka Inset + 6th-order Sigmoid + Outset + Mild Punchy Look) ----
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

vec3 tonemapAgX(vec3 col) {
    // Column-major Rec.709 -> AgX inset matrix (each row sums to 1.0)
    const mat3 AGX_INSET = mat3(
        vec3(0.842479062253094,  0.0423282422610123, 0.0423756549057051),
        vec3(0.0784335999999992, 0.878468636469772,  0.0784336),
        vec3(0.0792237451477643, 0.0791661274605434, 0.879142973793104)
    );
    // Column-major AgX outset -> Rec.709 matrix (each row sums to 1.0)
    const mat3 AGX_OUTSET = mat3(
        vec3( 1.19687900512017,  -0.0528968517574562, -0.0529716355144438),
        vec3(-0.0980208811401368, 1.15190312990417,   -0.0980434501171241),
        vec3(-0.0990297440797205, -0.0989611768448433, 1.15107367264116)
    );

    const float MIN_EV = -12.47393;
    const float MAX_EV =  4.026069;

    vec3 val = AGX_INSET * max(col, vec3(0.0));
    val = clamp(log2(max(val, vec3(1e-10))), MIN_EV, MAX_EV);
    val = (val - MIN_EV) * (1.0 / (MAX_EV - MIN_EV));
    val = saturate(agxDefaultContrastApprox(val));

    // Gentle Punchy AgX look (restores crisp midtone contrast & foliage/stained-glass vibrancy)
    float l = luma(val);
    val = pow(val, vec3(1.16));
    val = mix(vec3(l), val, 1.18);

    val = saturate(AGX_OUTSET * val);
    // AgX contrast curve outputs ~2.2 display-encoded values -> return linear [0, 1]
    return pow(val, vec3(2.2));
}

// ---- 2. GT7 Tone Mapping (Perceptual ICtCp Hue-Preserving + C1 Uchimura Shoulder) ----
const float GT7_PQ_M1 = 2610.0 / 16384.0;
const float GT7_PQ_M2 = 2523.0 / 32.0;
const float GT7_PQ_C1 = 3424.0 / 4096.0;
const float GT7_PQ_C2 = 2413.0 / 128.0;
const float GT7_PQ_C3 = 2392.0 / 128.0;

vec3 gt7LinearToPQ(vec3 x) {
    // Reference white 1.0 == 100 nits -> normalized to ST.2084 10000-nit range (0.01)
    vec3 y = pow(max(x * 0.01, vec3(0.0)), vec3(GT7_PQ_M1));
    return pow(max((GT7_PQ_C1 + GT7_PQ_C2 * y) / (1.0 + GT7_PQ_C3 * y), vec3(0.0)), vec3(GT7_PQ_M2));
}

vec3 gt7PQToLinear(vec3 pq) {
    vec3 p = pow(max(pq, vec3(0.0)), vec3(1.0 / GT7_PQ_M2));
    vec3 num = max(p - GT7_PQ_C1, vec3(0.0));
    vec3 den = max(GT7_PQ_C2 - GT7_PQ_C3 * p, vec3(1e-6));
    return pow(num / den, vec3(1.0 / GT7_PQ_M1)) * 100.0;
}

vec3 gt7Rec709ToICtCp(vec3 rgb) {
    // Rec.709 -> Rec.2020 -> LMS (Dolby ICtCp combined matrix, column-major)
    const mat3 REC709_TO_LMS = mat3(
        vec3(0.41210938, 0.16674805, 0.02416992),
        vec3(0.52392578, 0.72045898, 0.07543945),
        vec3(0.06396484, 0.11279297, 0.90039063)
    );
    vec3 lmsPQ = gt7LinearToPQ(max(REC709_TO_LMS * rgb, vec3(0.0)));
    float I  =  0.5000000 * lmsPQ.x + 0.5000000 * lmsPQ.y;
    float Ct =  1.6137695 * lmsPQ.x - 3.3234863 * lmsPQ.y + 1.7097168 * lmsPQ.z;
    float Cp =  4.3781738 * lmsPQ.x - 4.2456055 * lmsPQ.y - 0.1325683 * lmsPQ.z;
    return vec3(I, Ct, Cp);
}

vec3 gt7ICtCpToRec709(vec3 ictcp) {
    float L = ictcp.x + 0.0086090 * ictcp.y + 0.1110296 * ictcp.z;
    float M = ictcp.x - 0.0086090 * ictcp.y - 0.1110296 * ictcp.z;
    float S = ictcp.x + 0.5600313 * ictcp.y - 0.3206272 * ictcp.z;
    vec3 lms = gt7PQToLinear(vec3(L, M, S));
    const mat3 LMS_TO_REC709 = mat3(
        vec3( 3.4366067, -0.7913296, -0.0259499),
        vec3(-2.5064521,  1.9836005, -0.0989137),
        vec3( 0.0698454, -0.1922709,  1.1248636)
    );
    return LMS_TO_REC709 * lms;
}

// C1-continuous GT7 tone curve on [0, +inf) -> [0, 1]:
// Crisp subtle toe below linearSection (0.444), 1:1 linear midtones, smooth exponential shoulder to 1.0.
float gt7Curve(float x) {
    const float L = 0.444;
    const float TOE = 1.24;
    x = max(x, 0.0);
    if (x < L) {
        float toe = L * pow(x / L, TOE);
        return mix(toe, x, smoothstep(0.0, L, x));
    }
    return 1.0 - (1.0 - L) * exp(-(x - L) / (1.0 - L));
}

vec3 tonemapGT7(vec3 col) {
    col = max(col, vec3(0.0));
    vec3 ucs = gt7Rec709ToICtCp(col);

    // Per-channel shoulder curve provides natural white-core highlight roll-off
    vec3 skewedRgb = vec3(gt7Curve(col.r), gt7Curve(col.g), gt7Curve(col.b));
    vec3 skewedUcs = gt7Rec709ToICtCp(skewedRgb);

    // Luminance-mapped chroma-attenuated ICtCp preserves exact hue across high-intensity colored lights
    float peakPQ = gt7LinearToPQ(vec3(1.0)).x;
    float chromaFade = 1.0 - smoothstep(0.60, 1.15, ucs.x / max(peakPQ, 1e-4));
    vec3 huePreservedUcs = vec3(skewedUcs.x, ucs.yz * chromaFade);

    // GT7 reference blend: 60% hue-preserving perceptual ICtCp + 40% smooth per-channel desaturation
    vec3 blendedUcs = mix(skewedUcs, huePreservedUcs, 0.60);
    vec3 outRgb = gt7ICtCpToRec709(blendedUcs);

    // Luminance-preserving Rec.709 gamut compression for extreme out-of-gamut colors
    float y = luma(max(outRgb, vec3(0.0)));
    float minC = minOf(outRgb);
    if (minC < 0.0) {
        outRgb = mix(vec3(y), outRgb, y / max(y - minC, 1e-5));
    }
    float maxC = maxOf(outRgb);
    if (maxC > 1.0) {
        outRgb = mix(vec3(1.0), outRgb, (1.0 - y) / max(maxC - y, 1e-5));
    }
    return saturate(outRgb);
}

// ---- 3. Khronos PBR Neutral (2024 Exact Reference) ----
vec3 tonemapPBRNeutral(vec3 color) {
    color = max(color, vec3(0.0));
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return saturate(color);

    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return saturate(mix(color, vec3(newPeak), g));
}

// ---- 4. ACES Filmic (Narkowicz 2015) ----
vec3 tonemapACES(vec3 x) {
    x = max(x, vec3(0.0));
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

vec3 applyTonemap(vec3 hdr) {
    #if TONEMAP_MODE == 1
        return tonemapGT7(hdr);
    #elif TONEMAP_MODE == 2
        return tonemapPBRNeutral(hdr);
    #elif TONEMAP_MODE == 3
        return tonemapACES(hdr);
    #else
        return tonemapAgX(hdr);
    #endif
}

// ============================================================================
// Shared 5-Level Bloom Pyramid Atlas Layout inside scratchImg (2*W x H, zero extra VRAM!)
// Tile 0 (x in [0, W)):   Downsample levels L0..L4
// Tile 1 (x in [W, 2W)):  Upsample levels   U0..U3 (mirrored + ivec2(W, 0), race-free!)
// ============================================================================
struct BloomRect {
    ivec2 origin;
    ivec2 size;
};

BloomRect getBloomRect(int level, bool upsampleTile, ivec2 res) {
    int w0 = max(res.x >> 1, 1);
    int h0 = max(res.y >> 1, 1);
    int w1 = max(res.x >> 2, 1);
    int h1 = max(res.y >> 2, 1);
    int w2 = max(res.x >> 3, 1);
    int h2 = max(res.y >> 3, 1);
    int w3 = max(res.x >> 4, 1);
    int h3 = max(res.y >> 4, 1);
    int w4 = max(res.x >> 5, 1);
    int h4 = max(res.y >> 5, 1);

    BloomRect r;
    if (level == 0)      { r.origin = ivec2(0,       0);       r.size = ivec2(w0, h0); }
    else if (level == 1) { r.origin = ivec2(w0,      0);       r.size = ivec2(w1, h1); }
    else if (level == 2) { r.origin = ivec2(w0,      h1);      r.size = ivec2(w2, h2); }
    else if (level == 3) { r.origin = ivec2(w0 + w2, h1);      r.size = ivec2(w3, h3); }
    else                 { r.origin = ivec2(w0 + w2, h1 + h3); r.size = ivec2(w4, h4); }

    if (upsampleTile) r.origin.x += res.x;
    return r;
}

// Exact 4x4 symmetric cubic B-spline downsample ([1,3,3,1]/8) using texelFetch:
// Zero half-pixel drift across 5 mip levels, zero cross-tile bleeding, optional Karis weight on L0!
vec3 bloomDownsample4x4(sampler2D srcTex, ivec2 dstPx, ivec2 srcOrigin, ivec2 srcSize, bool karisWeight) {
    const float w1d[4] = float[4](0.125, 0.375, 0.375, 0.125);
    ivec2 srcMax = srcSize - 1;
    ivec2 base = dstPx * 2 - 1;
    vec3 sum = vec3(0.0);
    float wSum = 0.0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            ivec2 sp = clamp(base + ivec2(x, y), ivec2(0), srcMax) + srcOrigin;
            vec3 c = max(texelFetch(srcTex, sp, 0).rgb, vec3(0.0));
            if (any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
            float w = w1d[x] * w1d[y];
            if (karisWeight) {
                w /= (1.0 + luma(c));
            }
            sum += c * w;
            wSum += w;
        }
    }
    return sum / max(wSum, 1e-6);
}

// Exact 3x3 tent upsample using texelFetch (zero cross-tile bleeding, sampler-state independent!).
vec3 bloomUpsample3x3(sampler2D srcTex, ivec2 dstPx, ivec2 dstSize, ivec2 srcOrigin, ivec2 srcSize) {
    vec2 uv = (vec2(dstPx) + 0.5) / vec2(dstSize);
    vec2 srcPos = uv * vec2(srcSize) - 0.5;
    ivec2 center = ivec2(round(srcPos));
    ivec2 srcMax = srcSize - 1;
    vec3 sum = vec3(0.0);
    float wSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 p = center + ivec2(x, y);
            vec2 d = abs(vec2(p) - srcPos);
            float w = max(1.5 - d.x, 0.0) * max(1.5 - d.y, 0.0);
            if (w <= 1e-5) continue;
            vec3 c = max(texelFetch(srcTex, clamp(p, ivec2(0), srcMax) + srcOrigin, 0).rgb, vec3(0.0));
            sum += c * w;
            wSum += w;
        }
    }
    return sum / max(wSum, 1e-6);
}

#endif
