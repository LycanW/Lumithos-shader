#ifndef LIB_RADIANCE_CACHE_GLSL
#define LIB_RADIANCE_CACHE_GLSL

#include "/lib/Resources.glsl"

// ============================================================================
// World-Space 4-Way Set-Associative Surface Hash Cache
// - Stores directed surface patches keyed by (worldCell, faceIdx, distanceLOD):
//   0.5m near-field, 1.0m mid-field, 2.0m far-field, 4.0m horizon!
// - Full linear FP16 RGB irradiance per face (zero L1 SH chroma collapse, zero wall bleed!).
// - Cache-line-aligned 4-slot buckets (64 bytes/bucket) with inline frame-stamp activity
//   and automatic LRU eviction — zero global worklist SSBO, zero atomicOr bitmasks!
// ============================================================================

uint scSelectLod(vec3 worldPos) {
    float d2 = dot(worldPos, worldPos);
    if (d2 < 196.0)  return 0u; // < 14m: 0.5m half-block precision
    if (d2 < 1296.0) return 1u; // < 36m: 1.0m block precision
    if (d2 < 6400.0) return 2u; // < 80m: 2.0m voxel-grid extent
    return 3u;                  // >= 80m: 4.0m far-field horizon
}

float scCellSize(uint lod) {
    return 0.5 * float(1u << lod);
}

uint scNormalToFace(vec3 n) {
    vec3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) return n.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.z)               return n.y >= 0.0 ? 2u : 3u;
    return n.z >= 0.0 ? 4u : 5u;
}

void scFaceBasis(uint faceIdx, out vec3 N, out vec3 T, out vec3 B) {
    if (faceIdx == 0u)      { N = vec3( 1.0,  0.0,  0.0); T = vec3( 0.0,  1.0,  0.0); B = vec3( 0.0,  0.0,  1.0); }
    else if (faceIdx == 1u) { N = vec3(-1.0,  0.0,  0.0); T = vec3( 0.0,  1.0,  0.0); B = vec3( 0.0,  0.0, -1.0); }
    else if (faceIdx == 2u) { N = vec3( 0.0,  1.0,  0.0); T = vec3( 1.0,  0.0,  0.0); B = vec3( 0.0,  0.0, -1.0); }
    else if (faceIdx == 3u) { N = vec3( 0.0, -1.0,  0.0); T = vec3( 1.0,  0.0,  0.0); B = vec3( 0.0,  0.0,  1.0); }
    else if (faceIdx == 4u) { N = vec3( 0.0,  0.0,  1.0); T = vec3( 1.0,  0.0,  0.0); B = vec3( 0.0,  1.0,  0.0); }
    else                    { N = vec3( 0.0,  0.0, -1.0); T = vec3(-1.0,  0.0,  0.0); B = vec3( 0.0,  1.0,  0.0); }
}

ivec3 scWorldCell(vec3 worldPos, vec3 normal, uint lod) {
    float cSize = scCellSize(lod);
    vec3 shifted = worldPos + cameraPositionFract + normal * (0.04 * cSize);
    if (lod == 0u) {
        return ((cameraPositionInt & 255) << 1) + ivec3(floor(shifted * 2.0));
    }
    int shift = int(lod - 1u);
    return (cameraPositionInt + ivec3(floor(shifted))) >> shift;
}

// 100% lossless bijective 32-bit surface key:
// bits [0..8] = cell.x, [9..17] = cell.y, [18..26] = cell.z, [27..29] = faceIdx+1 (1..6, never 0!), [30..31] = lod
uint scPackKey(ivec3 cell, uint faceIdx, uint lod) {
    uint kx = uint(cell.x) & 511u;
    uint ky = uint(cell.y) & 511u;
    uint kz = uint(cell.z) & 511u;
    return kx | (ky << 9u) | (kz << 18u) | ((faceIdx + 1u) << 27u) | ((lod & 3u) << 30u);
}

void scUnpackKey(uint key, out ivec3 modCell, out uint faceIdx, out uint lod) {
    modCell = ivec3(int(key & 511u), int((key >> 9u) & 511u), int((key >> 18u) & 511u));
    faceIdx = ((key >> 27u) & 7u) - 1u;
    lod     = (key >> 30u) & 3u;
}

uint scHashBucket(uint key) {
    uint h = key * 0x9E3779B9u;
    h ^= h >> 16u;
    h *= 0x45D9F3Bu;
    h ^= h >> 16u;
    return h & (SC_NUM_BUCKETS - 1u);
}

uint scCurrFrame14() {
    return uint(frameCounter) & 16383u;
}

uint scPackGuideUV(vec3 dir, uint faceIdx) {
    vec3 N, T, B;
    scFaceBasis(faceIdx, N, T, B);
    float tx = clamp(dot(dir, T), -1.0, 1.0);
    float ty = clamp(dot(dir, B), -1.0, 1.0);
    // Even scale 14.0 maps exact center 0.0 -> 7 without rounding bias.
    uint u4 = uint(clamp(round((tx * 0.5 + 0.5) * 14.0), 0.0, 14.0));
    uint v4 = uint(clamp(round((ty * 0.5 + 0.5) * 14.0), 0.0, 14.0));
    return u4 | (v4 << 4u);
}

vec3 scUnpackGuideDir(uint uv8, uint faceIdx) {
    vec3 N, T, B;
    scFaceBasis(faceIdx, N, T, B);
    float tx = float(uv8 & 15u) * (2.0 / 14.0) - 1.0;
    float ty = float((uv8 >> 4u) & 15u) * (2.0 / 14.0) - 1.0;
    float r2 = tx * tx + ty * ty;
    if (r2 > 0.96) {
        float s = sqrt(0.96 / r2);
        tx *= s;
        ty *= s;
        r2 = 0.96;
    }
    float tz = sqrt(max(1.0 - r2, 0.04));
    return normalize(T * tx + B * ty + N * tz);
}

uint scPackMeta(uint accum, uint frame14, uint guideUV8) {
    return min(accum, 1023u) | ((frame14 & 16383u) << 10u) | ((guideUV8 & 255u) << 24u);
}

// Helper: true iff world-space block coordinate is an opaque Track-1 solid full cube.
bool scIsSolidCubeBlock(ivec3 worldBlock) {
    #ifdef STAGE_VOXEL_GRID_WRITE
        return false;
    #else
        ivec3 vPos = (worldBlock - cameraPositionInt) + ivec3(VOXEL_RES_XZ / 2, VOXEL_RES_Y / 2, VOXEL_RES_XZ / 2);
        if (!inVoxelBounds(vPos)) return false;
        if (texelFetch(voxelGridTex, vPos, 0).r == 0u) return false;
        SurfaceVoxel sv = surfacePool.surfaces[voxelLinearIndex(vPos)];
        if (sv.primHead != PRIM_END) return false;
        uint anyFace = sv.face[0] | sv.face[1] | sv.face[2] | sv.face[3] | sv.face[4] | sv.face[5];
        if (anyFace == 0u) return false;
        for (int f = 0; f < 6; ++f) {
            if (sv.face[f] != 0u && ((sv.face[f] >> 26u) & 1u) == 0u) return true;
        }
        return false;
    #endif
}

bool scCellVisibleFromBase(ivec3 baseWorldBlock, ivec3 candWorldBlock, uint lod) {
    if (lod > 1u || all(equal(baseWorldBlock, candWorldBlock))) return true;
    if (scIsSolidCubeBlock(candWorldBlock)) return false;
    ivec3 d = candWorldBlock - baseWorldBlock;
    int manhattan = abs(d.x) + abs(d.y) + abs(d.z);
    if (manhattan > 1) {
        if (d.x != 0 && scIsSolidCubeBlock(baseWorldBlock + ivec3(d.x, 0, 0))) return false;
        if (d.y != 0 && scIsSolidCubeBlock(baseWorldBlock + ivec3(0, d.y, 0))) return false;
        if (d.z != 0 && scIsSolidCubeBlock(baseWorldBlock + ivec3(0, 0, d.z))) return false;
    }
    return true;
}

// Single cache-line (4-slot bucket) lookup with idempotent frame-stamp touch and lock-free insert.
bool scLookupSlot(uint key, bool allowTouch, bool allowInsert,
                  out vec3 outRad, out uint outAccum, out uint outGuideUV, out float outGuideRho) {
    outRad      = vec3(0.0);
    outAccum    = 0u;
    outGuideUV  = 0x77u;
    outGuideRho = 0.0;

    uint base = scHashBucket(key) << 2u;
    uint currF = scCurrFrame14();

    int emptyIdx = -1;
    int evictIdx = -1;
    uint evictKey = 0u;
    uint oldestAge = SC_STALE_FRAMES;

    for (int i = 0; i < 4; ++i) {
        uint slot = base + uint(i);
        uint k = scTable.entries[slot].key;
        if (k == key) {
            uint meta = scTable.entries[slot].meta;
            uvec2 r   = scTable.entries[slot].rad;
            if (allowTouch) {
                uint oldF = (meta >> 10u) & 16383u;
                if (oldF != currF) {
                    // Idempotent non-atomic stamp update during deferred passes (prepare3 never runs concurrently!).
                    scTable.entries[slot].meta = (meta & 0xFF0003FFu) | (currF << 10u);
                }
            }
            outAccum    = meta & 1023u;
            outGuideUV  = (meta >> 24u) & 255u;
            vec2 rg     = unpackHalf2x16(r.x);
            vec2 bg     = unpackHalf2x16(r.y);
            outRad      = max(vec3(rg, bg.x), vec3(0.0));
            outGuideRho = saturate(bg.y);
            return outAccum > 0u;
        }
        if (allowInsert) {
            if (k == 0u) {
                if (emptyIdx < 0) emptyIdx = i;
            } else {
                uint meta = scTable.entries[slot].meta;
                uint age = (currF - ((meta >> 10u) & 16383u)) & 16383u;
                if (age > oldestAge) {
                    oldestAge = age;
                    evictIdx  = i;
                    evictKey  = k;
                }
            }
        }
    }

    if (allowInsert) {
        int target = (emptyIdx >= 0) ? emptyIdx : evictIdx;
        if (target >= 0) {
            uint slot = base + uint(target);
            uint expected = (emptyIdx >= 0) ? 0u : evictKey;
            if (atomicCompSwap(scTable.entries[slot].key, expected, key) == expected) {
                scTable.entries[slot].meta = (currF << 10u) | (0x77u << 24u);
                scTable.entries[slot].rad  = uvec2(0u);
            }
        }
    }
    return false;
}

// O(1) Surface Cache lookup with coarser-LOD cold fallback (for diffuse bounce, water, debug view).
vec3 queryRadianceCache(vec3 worldPos, vec3 normal, uint rank, bool allowInsert) {
    uint lod = scSelectLod(worldPos);
    uint faceIdx = scNormalToFace(normal);
    ivec3 cell = scWorldCell(worldPos, normal, lod);
    uint key = scPackKey(cell, faceIdx, lod);

    vec3 rad;
    uint accum, gUV;
    float gRho;
    bool hit = scLookupSlot(key, allowInsert, allowInsert, rad, accum, gUV, gRho);
    if (hit && accum >= 3u) return rad;

    // Coarser LOD read-only fallback avoids black pop-in on newly seen 0.5m patches!
    if (lod < 3u) {
        uint pLod = lod + 1u;
        ivec3 pCell = scWorldCell(worldPos, normal, pLod);
        vec3 pRad;
        uint pAccum, pGUV;
        float pGRho;
        if (scLookupSlot(scPackKey(pCell, faceIdx, pLod), false, false, pRad, pAccum, pGUV, pGRho) && pAccum >= 2u) {
            return hit ? mix(pRad, rad, float(accum) * (1.0 / 3.0)) : pRad;
        }
    }
    return rad;
}

// Tangent-plane blue-noise jittered O(1) lookup (eliminates cell grid seams while staying wall-safe!).
vec3 queryRadianceCacheStochastic(vec3 worldPos, vec3 normal, uint rank, vec2 rand) {
    uint lod = scSelectLod(worldPos);
    uint faceIdx = scNormalToFace(normal);
    float cSize = scCellSize(lod);

    vec3 baseShift = worldPos + cameraPositionFract + normal * (0.04 * cSize);
    ivec3 baseBlock = cameraPositionInt + ivec3(floor(baseShift));

    vec3 N, T, B;
    scFaceBasis(faceIdx, N, T, B);
    float phi = TWO_PI * rand.x;
    float r = 0.45 * cSize * sqrt(rand.y);
    vec3 jPos = worldPos + T * (r * cos(phi)) + B * (r * sin(phi));

    ivec3 candBlock = cameraPositionInt + ivec3(floor(jPos + cameraPositionFract + normal * (0.04 * cSize)));
    vec3 queryPos = scCellVisibleFromBase(baseBlock, candBlock, lod) ? jPos : worldPos;

    ivec3 cell = scWorldCell(queryPos, normal, lod);
    uint key = scPackKey(cell, faceIdx, lod);

    vec3 rad;
    uint accum, gUV;
    float gRho;
    bool hit = scLookupSlot(key, true, true, rad, accum, gUV, gRho);
    if (hit && accum >= 3u) return rad;

    if (lod < 3u) {
        uint pLod = lod + 1u;
        ivec3 pCell = scWorldCell(queryPos, normal, pLod);
        vec3 pRad;
        uint pAccum, pGUV;
        float pGRho;
        if (scLookupSlot(scPackKey(pCell, faceIdx, pLod), true, true, pRad, pAccum, pGUV, pGRho) && pAccum >= 2u) {
            return hit ? mix(pRad, rad, float(accum) * (1.0 / 3.0)) : pRad;
        }
    }
    return rad;
}

// Smooth 4-tap tangent-plane bilinear lookup on the exact same face (for razor-sharp mirrors & first-person hand!).
vec3 queryRadianceCacheSmooth(vec3 worldPos, vec3 normal, vec3 skyFallback, uint rank, vec2 rand) {
    uint lod = scSelectLod(worldPos);
    uint faceIdx = scNormalToFace(normal);
    float cSize = scCellSize(lod);

    vec3 N, T, B;
    scFaceBasis(faceIdx, N, T, B);
    // Use strictly positive canonical tangent axes so [minCell, minCell + ax0 + ax1] brackets fCell!
    ivec3 ax0 = ivec3(round(abs(T)));
    ivec3 ax1 = ivec3(round(abs(B)));

    vec3 shifted = worldPos + cameraPositionFract + normal * (0.04 * cSize);
    ivec3 baseBlock = cameraPositionInt + ivec3(floor(shifted));
    ivec3 centerCell = scWorldCell(worldPos, normal, lod);

    vec3 fCell;
    if (lod == 0u) {
        fCell = vec3((cameraPositionInt & 255) << 1) + shifted * 2.0 - 0.5;
    } else {
        int shift = int(lod - 1u);
        ivec3 localBase = cameraPositionInt - ((cameraPositionInt >> shift) << shift);
        fCell = vec3(cameraPositionInt >> shift) + (vec3(localBase) + shifted) / cSize - 0.5;
    }
    ivec3 minCell = ivec3(floor(fCell));
    vec3 frac3 = fCell - vec3(minCell);
    vec2 tf = vec2(dot(frac3, vec3(ax0)), dot(frac3, vec3(ax1)));
    // Keep the fixed normal-axis cell identical to centerCell so bilinear taps stay strictly coplanar!
    vec3 nAbs = abs(N);
    if (nAbs.x > 0.5)      minCell.x = centerCell.x;
    else if (nAbs.y > 0.5) minCell.y = centerCell.y;
    else                   minCell.z = centerCell.z;

    vec3 sumRad = vec3(0.0);
    float sumW = 0.0;
    int reqSlot = frameCounter & 3;

    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            float bw = (i == 0 ? 1.0 - tf.x : tf.x) * (j == 0 ? 1.0 - tf.y : tf.y);
            ivec3 c = minCell + ax0 * i + ax1 * j;
            bool isCenter = all(equal(c, centerCell));

            if (!isCenter && lod <= 1u) {
                vec3 tapPos = worldPos + vec3(ax0) * ((float(i) - tf.x) * cSize) + vec3(ax1) * ((float(j) - tf.y) * cSize);
                ivec3 tapBlock = cameraPositionInt + ivec3(floor(tapPos + cameraPositionFract + normal * (0.04 * cSize)));
                if (!scCellVisibleFromBase(baseBlock, tapBlock, lod)) continue;
            }

            bool doInsert = isCenter || ((j * 2 + i) == reqSlot);
            vec3 r;
            uint accum, gUV;
            float gRho;
            bool tapHit = scLookupSlot(scPackKey(c, faceIdx, lod), doInsert, doInsert, r, accum, gUV, gRho);
            if (tapHit) {
                float conf = smoothstep(0.0, 6.0, float(accum));
                sumRad += mix(skyFallback, r, conf) * bw;
                sumW   += bw;
            } else if (lod < 3u && isCenter) {
                ivec3 pc = scWorldCell(worldPos, normal, lod + 1u);
                vec3 pr;
                uint pAccum, pGUV;
                float pGRho;
                if (scLookupSlot(scPackKey(pc, faceIdx, lod + 1u), false, false, pr, pAccum, pGUV, pGRho)) {
                    float conf = smoothstep(0.0, 6.0, float(pAccum));
                    sumRad += mix(skyFallback, pr, conf) * (bw * 0.6);
                    sumW   += bw * 0.6;
                }
            }
        }
    }

    return (sumW > 1e-4) ? (sumRad / sumW) : skyFallback;
}

// Native Surface-Hemisphere vMF Path Guiding lookup (reads 8-bit tangent-plane dominant direction + concentration).
bool queryRCPathGuide(vec3 worldPos, vec3 normal, bool allowInsert, vec2 jitterRand, out vec3 guideDir, out float guideKappa, out float guideProb) {
    guideDir   = normal;
    guideKappa = 0.0;
    guideProb  = 0.0;

    #ifndef PATH_GUIDING
        return false;
    #endif

    uint lod = scSelectLod(worldPos);
    uint faceIdx = scNormalToFace(normal);
    float cSize = scCellSize(lod);

    vec3 N, T, B;
    scFaceBasis(faceIdx, N, T, B);
    float phi = TWO_PI * jitterRand.x;
    float r = 0.35 * cSize * sqrt(jitterRand.y);
    vec3 jPos = worldPos + T * (r * cos(phi)) + B * (r * sin(phi));

    ivec3 baseBlock = cameraPositionInt + ivec3(floor(worldPos + cameraPositionFract + normal * (0.04 * cSize)));
    ivec3 candBlock = cameraPositionInt + ivec3(floor(jPos + cameraPositionFract + normal * (0.04 * cSize)));
    vec3 queryPos = scCellVisibleFromBase(baseBlock, candBlock, lod) ? jPos : worldPos;

    ivec3 cell = scWorldCell(queryPos, normal, lod);
    uint key = scPackKey(cell, faceIdx, lod);

    vec3 rad;
    uint accum, gUV;
    float gRho;
    bool hit = scLookupSlot(key, allowInsert, allowInsert, rad, accum, gUV, gRho);
    if (hit && accum >= 10u && gRho > 0.14) {
        float conf = saturate((float(accum) - 6.0) / 24.0);
        float rho  = clamp(gRho * conf, 0.0, 0.85);
        if (rho > 0.12) {
            vec3 rawDir = scUnpackGuideDir(gUV, faceIdx);
            if (dot(rawDir, normal) > 0.05) {
                guideDir   = normalize(rawDir + normal * 0.15);
                guideKappa = clamp((3.0 * rho - rho * rho * rho) / max(1.0 - rho * rho, 0.12), 0.5, 6.0);
                guideProb  = clamp(0.55 * rho, 0.0, 0.50);
                return true;
            }
        }
    }
    return false;
}

#endif
