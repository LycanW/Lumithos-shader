#ifndef LIB_BLOOM_GLSL
#define LIB_BLOOM_GLSL

#include "/lib/Resources.glsl"

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
