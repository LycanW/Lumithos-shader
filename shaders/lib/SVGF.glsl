#ifndef LIB_SVGF_GLSL
#define LIB_SVGF_GLSL

#include "/lib/Resources.glsl"

uniform sampler2D diffHistTex;
uniform sampler2D diffMomTex;
uniform sampler2D specHistTex;
uniform sampler2D specMomTex;
uniform sampler2D prevGuideTex;
uniform sampler2D scratchTex;

// Exact previous-frame R2 TAA jitter matching shaders.properties (for jitter-aligned prevGuideTex lookups).
vec2 getPrevTaaJitter() {
    #ifdef TAA_ENABLED
        float pf = float(max(frameCounter - 1, 0));
        vec2 r2 = fract(pf * vec2(1.0 / 1.3247179572, 1.0 / 1.7548776662) + 0.5) * 2.0 - 1.0;
        return r2 * texelSize();
    #else
        return vec2(0.0);
    #endif
}

// Reproject current world position to previous unjittered screen UV.
vec2 reprojectWorldToPrevUV(vec3 worldPos) {
    vec3 prevWorld = worldPos + cameraVelocity;
    vec4 prevClip = gbufferPreviousProjection * (gbufferPreviousModelView * vec4(prevWorld, 1.0));
    if (prevClip.w <= 0.0) return vec2(-1.0);
    return (prevClip.xy / prevClip.w) * 0.5 + 0.5;
}

// Jitter-compensated screen-space reprojection:
// Computes exact unjittered motion (prevUnjitUV - currUnjitUV) so static pixels
// stay locked to currUV with zero subpixel jitter or drift.
vec2 reprojectScreenUV(vec2 currUV, vec3 worldPos) {
    vec2 prevUnjitUV = reprojectWorldToPrevUV(worldPos);
    if (prevUnjitUV.x < 0.0) return vec2(-1.0);
    vec4 currClip = gbufferProjection * (gbufferModelView * vec4(worldPos, 1.0));
    if (currClip.w <= 0.0) return vec2(-1.0);
    vec2 currUnjitUV = (currClip.xy / currClip.w) * 0.5 + 0.5;
    return currUV + (prevUnjitUV - currUnjitUV);
}

// Filter-independent 12-tap cornerless Catmull-Rom bicubic history fetch via texelFetch:
// - 1-tap exact fast-path when static (zero overhead).
// - Tile-clamped (`tileOff`) for race-free ping-pong TAA history without cross-tile bleed!
// - Returns RGB (Catmull-Rom) + A (non-ringing bilinear accumulated frame count).
vec4 fetchHistoryFast4(sampler2D tex, vec2 uv, ivec2 res, ivec2 tileOff) {
    vec2 pos = uv * vec2(res) - 0.5;
    ivec2 base = ivec2(floor(pos));
    vec2 f = pos - vec2(base);

    if (abs(f.x) < 1e-4 && abs(f.y) < 1e-4) {
        ivec2 p = clamp(base, ivec2(0), res - 1) + tileOff;
        vec4 tap = texelFetch(tex, p, 0);
        return vec4(max(tap.rgb, vec3(0.0)), max(tap.a, 0.0));
    }

    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 = 0.5 * (f3 - f2);

    ivec2 maxPx = res - 1;
    int x0 = clamp(base.x - 1, 0, maxPx.x) + tileOff.x;
    int x1 = clamp(base.x,     0, maxPx.x) + tileOff.x;
    int x2 = clamp(base.x + 1, 0, maxPx.x) + tileOff.x;
    int x3 = clamp(base.x + 2, 0, maxPx.x) + tileOff.x;
    int y0 = clamp(base.y - 1, 0, maxPx.y) + tileOff.y;
    int y1 = clamp(base.y,     0, maxPx.y) + tileOff.y;
    int y2 = clamp(base.y + 1, 0, maxPx.y) + tileOff.y;
    int y3 = clamp(base.y + 2, 0, maxPx.y) + tileOff.y;

    // 12-tap cornerless Catmull-Rom for RGB + exact bilinear of center 2x2 for frame count A.
    vec4 t11 = texelFetch(tex, ivec2(x1, y1), 0);
    vec4 t21 = texelFetch(tex, ivec2(x2, y1), 0);
    vec4 t12 = texelFetch(tex, ivec2(x1, y2), 0);
    vec4 t22 = texelFetch(tex, ivec2(x2, y2), 0);

    vec3 sum = vec3(0.0);
    sum += max(texelFetch(tex, ivec2(x1, y0), 0).rgb, vec3(0.0)) * (w1.x * w0.y);
    sum += max(texelFetch(tex, ivec2(x2, y0), 0).rgb, vec3(0.0)) * (w2.x * w0.y);

    sum += max(texelFetch(tex, ivec2(x0, y1), 0).rgb, vec3(0.0)) * (w0.x * w1.y);
    sum += max(t11.rgb, vec3(0.0)) * (w1.x * w1.y);
    sum += max(t21.rgb, vec3(0.0)) * (w2.x * w1.y);
    sum += max(texelFetch(tex, ivec2(x3, y1), 0).rgb, vec3(0.0)) * (w3.x * w1.y);

    sum += max(texelFetch(tex, ivec2(x0, y2), 0).rgb, vec3(0.0)) * (w0.x * w2.y);
    sum += max(t12.rgb, vec3(0.0)) * (w1.x * w2.y);
    sum += max(t22.rgb, vec3(0.0)) * (w2.x * w2.y);
    sum += max(texelFetch(tex, ivec2(x3, y2), 0).rgb, vec3(0.0)) * (w3.x * w2.y);

    sum += max(texelFetch(tex, ivec2(x1, y3), 0).rgb, vec3(0.0)) * (w1.x * w3.y);
    sum += max(texelFetch(tex, ivec2(x2, y3), 0).rgb, vec3(0.0)) * (w2.x * w3.y);

    float wNorm = 1.0 - (w0.x + w3.x) * (w0.y + w3.y);
    vec3 catmull = sum / max(wNorm, 1e-4);
    // Anti-ringing clamp against the center 2x2 neighborhood eliminates bicubic overshoot halos on sharp edges!
    vec3 c11 = max(t11.rgb, vec3(0.0));
    vec3 c21 = max(t21.rgb, vec3(0.0));
    vec3 c12 = max(t12.rgb, vec3(0.0));
    vec3 c22 = max(t22.rgb, vec3(0.0));
    vec3 minC = min(min(c11, c21), min(c12, c22));
    vec3 maxC = max(max(c11, c21), max(c12, c22));
    float aBilin = mix(mix(t11.a, t21.a, f.x), mix(t12.a, t22.a, f.x), f.y);
    return vec4(clamp(catmull, minC, maxC), max(aBilin, 0.0));
}

// Geometry-aware 3x3 RCRS firefly despike (preserves chromaticity) with depth-only fallback
// for intricate 3D latticework / armor stands / curved geometry where <3 neighbors share the same normal!
vec3 despike3x3(sampler2D noisyTex, ivec2 px, ivec2 res, vec3 centerCol, vec3 geoN, vec3 shadeN, float centerDepth, float ratio, float floorVal) {
    float cLuma = luma(centerCol);
    float maxN = 0.0, secondMax = 0.0;
    float dMaxN = 0.0, dSecondMax = 0.0;
    int valid = 0, dValid = 0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 q = px + ivec2(x, y);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
            float qd = texelFetch(depthtex1, q, 0).r;
            if (qd >= 0.999999 || abs(qd - centerDepth) > 0.020 * max(centerDepth, 0.01)) continue;
            vec4 qt = texelFetch(noisyTex, q, 0);
            if (qt.a < 0.0) continue;
            vec3 qc = max(qt.rgb, vec3(0.0));
            if (any(isnan(qc)) || any(isinf(qc))) continue;
            float ql = luma(qc);
            if (ql > dMaxN) { dSecondMax = dMaxN; dMaxN = ql; }
            else if (ql > dSecondMax) { dSecondMax = ql; }
            ++dValid;

            vec3 qGeoN, qShadeN;
            readGBufferNormals(q, qGeoN, qShadeN);
            if (dot(geoN, qGeoN) < 0.9 || dot(shadeN, qShadeN) < 0.45) continue;
            if (ql > maxN) { secondMax = maxN; maxN = ql; }
            else if (ql > secondMax) { secondMax = ql; }
            ++valid;
        }
    }
    if (valid >= 3) {
        float refLuma = (valid >= 5) ? secondMax : maxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) return centerCol * (lim / max(cLuma, 1e-6));
    } else if (dValid >= 3) {
        float refLuma = (dValid >= 5) ? dSecondMax : dMaxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) return centerCol * (lim / max(cLuma, 1e-6));
    }
    return centerCol;
}

// Same-parity 8-neighbor RCRS firefly despike for checkerboard-traced buffers (with depth-only fallback!).
vec3 despikeCheckerboard(sampler2D noisyTex, ivec2 px, ivec2 res, vec3 centerCol, vec3 geoN, vec3 shadeN, float centerDepth, float ratio, float floorVal) {
    const ivec2 offs[8] = ivec2[8](
        ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1),
        ivec2(-2,  0), ivec2(2,  0), ivec2( 0,-2), ivec2(0, 2)
    );
    float cLuma = luma(centerCol);
    float maxN = 0.0, secondMax = 0.0;
    float dMaxN = 0.0, dSecondMax = 0.0;
    int valid = 0, dValid = 0;
    for (int i = 0; i < 8; ++i) {
        ivec2 q = px + offs[i];
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
        float qd = texelFetch(depthtex1, q, 0).r;
        if (qd >= 0.999999 || abs(qd - centerDepth) > 0.022 * max(centerDepth, 0.01)) continue;
        vec4 qt = texelFetch(noisyTex, q, 0);
        if (qt.a < 0.0) continue;
        vec3 qc = max(qt.rgb, vec3(0.0));
        if (any(isnan(qc)) || any(isinf(qc))) continue;
        float ql = luma(qc);
        if (ql > dMaxN) { dSecondMax = dMaxN; dMaxN = ql; }
        else if (ql > dSecondMax) { dSecondMax = ql; }
        ++dValid;

        vec3 qGeoN, qShadeN;
        readGBufferNormals(q, qGeoN, qShadeN);
        if (dot(geoN, qGeoN) < 0.9 || dot(shadeN, qShadeN) < 0.45) continue;
        if (ql > maxN) { secondMax = maxN; maxN = ql; }
        else if (ql > secondMax) { secondMax = ql; }
        ++valid;
    }
    if (valid >= 3) {
        float refLuma = (valid >= 5) ? secondMax : maxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) return centerCol * (lim / max(cLuma, 1e-6));
    } else if (dValid >= 3) {
        float refLuma = (dValid >= 5) ? dSecondMax : dMaxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) return centerCol * (lim / max(cLuma, 1e-6));
    }
    return centerCol;
}

// Geometry-aware 4-neighbor axial cross-fill for disoccluded checkerboard-untraced pixels:
// All 4 axial neighbors ((±1,0), (0,±1)) are guaranteed traced on the current frame!
vec4 checkerboardCrossFill(sampler2D noisyTex, ivec2 px, ivec2 res, vec3 geoN, vec3 shadeN, float centerDepth,
                           bool checkRough, float centerRough) {
    const ivec2 offs[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
    vec3 sumCol = vec3(0.0);
    float sumDist = 0.0;
    float sumW = 0.0;
    vec3 fallbackCol = vec3(0.0);
    float fallbackDist = 64.0;
    float bestFallW = -1.0;

    for (int i = 0; i < 4; ++i) {
        ivec2 q = px + offs[i];
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
        float qd = texelFetch(depthtex1, q, 0).r;
        if (qd >= 0.999999) continue;

        vec4 qt = texelFetch(noisyTex, q, 0);
        if (qt.a < 0.0) continue;
        vec3 qc = max(qt.rgb, vec3(0.0));
        if (any(isnan(qc)) || any(isinf(qc))) continue;

        vec3 qGeoN, qShadeN;
        readGBufferNormals(q, qGeoN, qShadeN);
        float dGeo = dot(geoN, qGeoN);
        float depthDiff = abs(qd - centerDepth) / max(centerDepth, 0.01);
        float fallW = max(dGeo, 0.0) / (1.0 + depthDiff * 40.0);
        if (fallW > bestFallW) {
            bestFallW = fallW;
            fallbackCol = qc;
            fallbackDist = qt.a;
        }

        if (dGeo < 0.90 || depthDiff > 0.025) continue;
        float dShade = max(dot(shadeN, qShadeN), 0.0);
        if (dShade < 0.45) continue;

        float w = pow(dShade, 4.0);
        if (checkRough) {
            float qRough, qF0;
            bool qHand;
            readGBufferSpecMask(q, qRough, qF0, qHand);
            if (!hasSpecularTrace(qRough, qF0, qHand)) continue;
            w *= exp(-abs(centerRough - qRough) / max(0.03, 0.25 * max(centerRough, qRough)));
        }
        sumCol  += w * qc;
        sumDist += w * qt.a;
        sumW    += w;
    }

    if (sumW > 1e-5) {
        return vec4(sumCol / sumW, sumDist / sumW);
    }
    if (bestFallW >= 0.0) {
        return vec4(fallbackCol, fallbackDist);
    }
    return vec4(0.0, 0.0, 0.0, 64.0);
}

// Bayer 2x2 subpixel offset for quarter-resolution interleaved diffuse tracing:
// (0,0) -> (1,1) -> (1,0) -> (0,1) eliminates directional sweep waves across 4 frames!
ivec2 diffuseQuarterOffset(int frame) {
    int f = frame & 3;
    return ivec2(((f >> 1) ^ (f & 1)) & 1, f & 1);
}

// Same-phase 8-neighbor (stride-2 3x3) RCRS firefly despike for 2x2 quarter-res traced buffers
// with depth-only fallback for intricate 3D latticework / armor stands / stairs!
vec3 despikeQuarter(sampler2D noisyTex, ivec2 px, ivec2 res, vec3 centerCol, vec3 geoN, vec3 shadeN, float centerDepth, float ratio, float floorVal) {
    float cLuma = luma(centerCol);
    float maxN = 0.0, secondMax = 0.0;
    float dMaxN = 0.0, dSecondMax = 0.0;
    int valid = 0, dValid = 0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 q = px + ivec2(x, y) * 2;
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
            float qd = texelFetch(depthtex1, q, 0).r;
            if (qd >= 0.999999 || abs(qd - centerDepth) > 0.025 * max(centerDepth, 0.01)) continue;
            vec3 qc = max(texelFetch(noisyTex, q, 0).rgb, vec3(0.0));
            if (any(isnan(qc)) || any(isinf(qc))) continue;
            float ql = luma(qc);
            if (ql > dMaxN) { dSecondMax = dMaxN; dMaxN = ql; }
            else if (ql > dSecondMax) { dSecondMax = ql; }
            ++dValid;

            vec3 qGeoN, qShadeN;
            readGBufferNormals(q, qGeoN, qShadeN);
            if (dot(geoN, qGeoN) < 0.9 || dot(shadeN, qShadeN) < 0.45) continue;
            if (ql > maxN) { secondMax = maxN; maxN = ql; }
            else if (ql > secondMax) { secondMax = ql; }
            ++valid;
        }
    }
    if (valid >= 3) {
        float refLuma = (valid >= 5) ? secondMax : maxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) return centerCol * (lim / max(cLuma, 1e-6));
    } else if (dValid >= 3) {
        float refLuma = (dValid >= 5) ? dSecondMax : dMaxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) return centerCol * (lim / max(cLuma, 1e-6));
    }
    return centerCol;
}

// Geometry-aware 2x2 bilinear fill from the 4 nearest traced quarter-res pixels around `px`.
vec3 quarterBilinearFill(sampler2D noisyTex, ivec2 px, ivec2 res, ivec2 qOff, vec3 geoN, vec3 shadeN, float centerDepth) {
    ivec2 qBase = ((px - qOff) >> 1) * 2 + qOff;
    vec2 f = vec2(px - qBase) * 0.5;

    vec3 sumCol = vec3(0.0);
    float sumW = 0.0;
    vec3 fallbackCol = vec3(0.0);
    float bestFallW = -1.0;

    // Pre-fetch the 4 surrounding traced corners and cap any isolated single-corner firefly
    // so raw spikes never smear into 2x2 bright blocks on disoccluded narrow geometry!
    vec3 cCols[4];
    float cLumas[4];
    float cDepths[4];
    bool cValid[4];
    float minL = 1e6, secondMinL = 1e6;
    int validCnt = 0;

    for (int idx = 0; idx < 4; ++idx) {
        int i = idx & 1;
        int j = idx >> 1;
        ivec2 q = qBase + ivec2(i * 2, j * 2);
        cValid[idx] = false;
        cCols[idx] = vec3(0.0);
        cLumas[idx] = 0.0;
        cDepths[idx] = 1.0;
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
        float qd = texelFetch(depthtex1, q, 0).r;
        if (qd >= 0.999999) continue;
        vec3 qc = max(texelFetch(noisyTex, q, 0).rgb, vec3(0.0));
        if (any(isnan(qc)) || any(isinf(qc))) continue;
        float ql = luma(qc);
        cValid[idx] = true;
        cCols[idx] = qc;
        cLumas[idx] = ql;
        cDepths[idx] = qd;
        if (ql < minL) { secondMinL = minL; minL = ql; }
        else if (ql < secondMinL) { secondMinL = ql; }
        ++validCnt;
    }
    float cornerCap = (validCnt >= 3) ? (secondMinL * 5.0 + 0.8) : 16.0;

    for (int idx = 0; idx < 4; ++idx) {
        if (!cValid[idx]) continue;
        int i = idx & 1;
        int j = idx >> 1;
        ivec2 q = qBase + ivec2(i * 2, j * 2);
        float qd = cDepths[idx];
        vec3 qc = cCols[idx];
        if (cLumas[idx] > cornerCap) {
            qc *= cornerCap / max(cLumas[idx], 1e-6);
        }

        vec3 qGeoN, qShadeN;
        readGBufferNormals(q, qGeoN, qShadeN);
        float bw = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
        float dGeo = dot(geoN, qGeoN);
        float depthDiff = abs(qd - centerDepth) / max(centerDepth, 0.01);
        float fallW = (max(dGeo, 0.0) + 0.05) * (bw + 0.05) / (1.0 + depthDiff * 40.0);
        if (fallW > bestFallW) {
            bestFallW = fallW;
            fallbackCol = qc;
        }

        if (dGeo < 0.90 || depthDiff > 0.03) continue;
        float dShade = max(dot(shadeN, qShadeN), 0.0);
        if (dShade < 0.45) continue;

        float s2 = dShade * dShade;
        float w = max(bw, 0.05) * (s2 * s2);
        sumCol += w * qc;
        sumW   += w;
    }
    if (sumW > 1e-5) return sumCol / sumW;
    return (bestFallW >= 0.0) ? fallbackCol : vec3(0.0);
}

// Edge-stopping weight for lightweight 3x3 A-Trous (view-space plane test + fast perspective unprojection:
// eliminates TWO full mat4 multiplies and transcendental pow() on every single tap!).
float atrousWeight(vec3 centerViewPos, vec3 centerViewGeoN, vec3 centerGeoN, vec3 centerShadeN,
                   ivec2 q, bool isSpecular, float centerRough) {
    float qd = texelFetch(depthtex1, q, 0).r;
    if (qd >= 0.999999) return 0.0;

    vec3 qGeoN, qShadeN;
    readGBufferNormals(q, qGeoN, qShadeN);
    float geoDot = dot(centerGeoN, qGeoN);
    if (geoDot < 0.9) return 0.0;

    vec2 qNdc = ((vec2(q) + 0.5) * texelSize()) * 2.0 - 1.0;
    #ifdef TAA_ENABLED
        qNdc -= taaJitter;
    #endif
    vec3 qView = fastScreenToViewNdc(qNdc, qd);
    float planeDist = abs(dot(qView - centerViewPos, centerViewGeoN));
    float wPlane = exp(-planeDist * 14.0);

    float sDot = max(dot(centerShadeN, qShadeN), 0.0);
    float s2 = sDot * sDot;
    float s4 = s2 * s2;
    float wNorm;
    if (isSpecular) {
        float s8  = s4 * s4;
        float s16 = s8 * s8;
        float s32 = s16 * s16;
        wNorm = s32 * s32; // exact sDot^64 without exp2/log2!
    } else {
        wNorm = s4 * s4 * s4; // exact sDot^12
    }

    float wRough = 1.0;
    if (isSpecular) {
        float qRough, qF0;
        bool qHand;
        readGBufferSpecMask(q, qRough, qF0, qHand);
        if (!hasSpecularTrace(qRough, qF0, qHand)) return 0.0;
        wRough = exp(-abs(centerRough - qRough) / max(0.02, 0.25 * max(centerRough, qRough)));
    }

    float g2  = geoDot * geoDot;
    float g4  = g2 * g2;
    float g8  = g4 * g4;
    float g16 = g8 * g8;
    return wPlane * wNorm * wRough * (g16 * g16);
}

#endif
