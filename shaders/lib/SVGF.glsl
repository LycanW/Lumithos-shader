#ifndef LIB_SVGF_GLSL
#define LIB_SVGF_GLSL

#include "/lib/Resources.glsl"

uniform sampler2D diffHistTex;
uniform sampler2D diffMomTex;
uniform sampler2D specHistTex;
uniform sampler2D specMomTex;
uniform sampler2D prevGuideTex;
uniform sampler2D scratchTex;

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

// Reversible Karis perceptual tonemap for flicker-free HDR TAA blending.
vec3 karisForward(vec3 c) {
    return c / (1.0 + luma(max(c, vec3(0.0))));
}

vec3 karisInverse(vec3 c) {
    return c / max(1.0 - luma(max(c, vec3(0.0))), 1e-4);
}

// Fast filter-independent bilinear fetch via texelFetch (1-tap exact when static, 4-tap moving).
vec3 fetchHistoryFast(sampler2D tex, vec2 uv, ivec2 res) {
    vec2 pos = uv * vec2(res) - 0.5;
    ivec2 base = ivec2(floor(pos));
    vec2 f = pos - vec2(base);

    if (abs(f.x) < 1e-4 && abs(f.y) < 1e-4) {
        ivec2 p = clamp(base, ivec2(0), res - 1);
        return max(texelFetch(tex, p, 0).rgb, vec3(0.0));
    }

    ivec2 p0 = clamp(base,              ivec2(0), res - 1);
    ivec2 p1 = clamp(base + ivec2(1,0), ivec2(0), res - 1);
    ivec2 p2 = clamp(base + ivec2(0,1), ivec2(0), res - 1);
    ivec2 p3 = clamp(base + ivec2(1,1), ivec2(0), res - 1);

    vec3 c0 = max(texelFetch(tex, p0, 0).rgb, vec3(0.0));
    vec3 c1 = max(texelFetch(tex, p1, 0).rgb, vec3(0.0));
    vec3 c2 = max(texelFetch(tex, p2, 0).rgb, vec3(0.0));
    vec3 c3 = max(texelFetch(tex, p3, 0).rgb, vec3(0.0));
    return mix(mix(c0, c1, f.x), mix(c2, c3, f.x), f.y);
}

// Geometry-aware 3x3 RCRS firefly despike (preserves chromaticity).
vec3 despike3x3(sampler2D noisyTex, ivec2 px, ivec2 res, vec3 centerCol, vec3 geoN, vec3 shadeN, float centerDepth, float ratio, float floorVal) {
    float cLuma = luma(centerCol);
    float maxN = 0.0;
    float secondMax = 0.0;
    int valid = 0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 q = px + ivec2(x, y);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
            float qd = texelFetch(depthtex1, q, 0).r;
            if (qd >= 0.999999 || abs(qd - centerDepth) > 0.015 * max(centerDepth, 0.01)) continue;
            GBufferMat qm = readGBuffer(q);
            if (dot(geoN, qm.geoNormal) < 0.9 || dot(shadeN, qm.shadeNormal) < 0.45) continue;
            vec3 qc = max(texelFetch(noisyTex, q, 0).rgb, vec3(0.0));
            if (any(isnan(qc)) || any(isinf(qc))) continue;
            float ql = luma(qc);
            if (ql > maxN) {
                secondMax = maxN;
                maxN = ql;
            } else if (ql > secondMax) {
                secondMax = ql;
            }
            ++valid;
        }
    }
    if (valid >= 3) {
        // Use 2nd-brightest neighbor so even adjacent firefly pairs get suppressed cleanly.
        float refLuma = (valid >= 5) ? secondMax : maxN;
        float lim = refLuma * ratio + floorVal;
        if (cLuma > lim) {
            return centerCol * (lim / max(cLuma, 1e-6));
        }
    }
    return centerCol;
}

// Edge-stopping weight for lightweight 3x3 A-Trous.
float atrousWeight(vec3 centerPos, vec3 centerGeoN, vec3 centerShadeN, float centerLuma, float phiLuma,
                   ivec2 q, bool isSpecular, float centerRough) {
    float qd = texelFetch(depthtex1, q, 0).r;
    if (qd >= 0.999999) return 0.0;

    GBufferMat qm = readGBuffer(q);
    float geoDot = dot(centerGeoN, qm.geoNormal);
    if (geoDot < 0.9) return 0.0;

    vec2 qUv = (vec2(q) + 0.5) * texelSize();
    vec3 qWorld = viewToWorld(screenToView(vec3(qUv, qd)));
    float planeDist = abs(dot(qWorld - centerPos, centerGeoN));
    float wPlane = exp(-planeDist * 14.0);

    float wNorm = pow(max(dot(centerShadeN, qm.shadeNormal), 0.0), isSpecular ? 64.0 : 12.0);
    float wRough = isSpecular ? exp(-abs(centerRough - qm.roughness) / max(0.02, 0.25 * max(centerRough, qm.roughness))) : 1.0;

    return wPlane * wNorm * wRough * pow(geoDot, 32.0);
}

#endif
