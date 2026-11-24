#ifndef LIB_RADIANCE_CACHE_GLSL
#define LIB_RADIANCE_CACHE_GLSL

#include "/lib/Resources.glsl"

// ============================================================================
// Revoxelation-Style O(1) 3-Cascade Toroidal 3D Clipmap Radiance Cache
// - ZERO hash probing loops, ZERO collisions, ZERO tombstones, ZERO duplicates!
// - 16-byte aligned probe (1x 128-bit load) storing full-sphere L1 SH + normalized CoCg + frames.
// - 1-bit atomicOr deduplicated active-set worklist updates ONLY visible/reflected cells.
// ============================================================================

uint rcSelectCascade(vec3 worldPos) {
    vec3 a = abs(worldPos);
    if (max(a.x, a.z) < 28.0 && a.y < 14.0) return 0u;
    if (max(a.x, a.z) < 56.0 && a.y < 28.0) return 1u;
    return 2u;
}

ivec3 rcWorldCell(vec3 worldPos, vec3 normal, uint cascade) {
    float cellSize = float(1u << cascade);
    vec3 shifted = worldPos + cameraPositionFract + normal * (0.45 * cellSize);
    ivec3 worldBlock = cameraPositionInt + ivec3(floor(shifted));
    return worldBlock >> int(cascade);
}

uint rcStorageIndex(ivec3 worldCell, uint cascade) {
    uvec3 c = uvec3(worldCell) & uvec3(63u, 31u, 63u);
    return cascade * RC_CASCADE_CELLS + c.x + (c.z << 6u) + (c.y << 12u);
}

uint rcPackWorldKey(ivec3 worldCell, uint cascade) {
    uvec3 c = uvec3(worldCell) & uvec3(1023u);
    uint k = (c.x << 22u) | (c.y << 12u) | (c.z << 2u) | (cascade & 3u);
    return (k == 0u) ? 1u : k;
}

// Co/Y spans [-2, +2] (pure red = +2, pure blue = -2); Cg/Y spans [-1, +1].
// Packing full [-2, +2] Co/Y prevents orange/red lantern light from clipping toward purple!
uint rcPackChromaFrames(vec2 normCoCg, uint frames) {
    uint co = uint(clamp(round((normCoCg.x * 0.25 + 0.5) * 1023.0), 0.0, 1023.0));
    uint cg = uint(clamp(round((normCoCg.y * 0.5  + 0.5) * 1023.0), 0.0, 1023.0));
    return (co << 22u) | (cg << 12u) | min(frames, 4095u);
}

vec2 rcUnpackNormCoCg(uint word) {
    float co = float((word >> 22u) & 1023u) * (4.0 / 1023.0) - 2.0;
    float cg = float((word >> 12u) & 1023u) * (2.0 / 1023.0) - 1.0;
    return vec2(co, cg);
}

uint rcUnpackFrames(uint word) {
    return word & 4095u;
}

uint rcUpdatePhase(uint key) {
    uint p = key;
    p ^= p >> 16u;
    p *= 0x7feb352du;
    p ^= p >> 15u;
    return p & (RC_UPDATE_PERIOD - 1u);
}

// Submit probe to current frame's deduplicated active-set worklist (1-bit atomicOr).
void rcRequestProbe(ivec3 worldCell, uint cascade, bool isValid, uint accumFrames) {
    uint key = rcPackWorldKey(worldCell, cascade);
    if (isValid && accumFrames >= RC_WARMUP_UPDATES && rcUpdatePhase(key) != (uint(frameCounter) & (RC_UPDATE_PERIOD - 1u))) {
        return;
    }

    ivec3 camCell = cameraPositionInt >> int(cascade);
    ivec3 rel = worldCell - camCell + ivec3(32, 16, 32);
    if (any(lessThan(rel, ivec3(0))) || any(greaterThanEqual(rel, ivec3(64, 32, 64)))) return;

    uint slot = rcStorageIndex(worldCell, cascade);
    uint side = uint(frameCounter) & 1u;
    uint wordIdx = side * RC_DEDUP_WORDS + (slot >> 5u);
    uint bit = 1u << (slot & 31u);
    if ((atomicOr(rcWork.data[wordIdx], bit) & bit) != 0u) return;

    uint dst = atomicAdd(rcWork.count[side], 1u);
    if (dst < RC_WORKLIST_CAP) {
        uint packedReq = uint(rel.x) | (uint(rel.y) << 6u) | (uint(rel.z) << 11u) | (cascade << 17u);
        rcWork.data[2u * RC_DEDUP_WORDS + side * RC_WORKLIST_CAP + dst] = packedReq;
    }
}

// Evaluate L1 SH probe radiance along queryNormal (O(1), directional!).
vec3 rcEvalProbe(RCProbe p, vec3 queryNormal) {
    vec4 shLuma = unpackHalf4x16(p.shLuma);
    float L0 = max(shLuma.x, 0.0);
    if (L0 <= 1e-6) return vec3(0.0);
    vec2 normCoCg = rcUnpackNormCoCg(p.chromaFrames);
    vec3 rgb = yCoCgToRgb(vec3(L0, normCoCg * L0));

    // Cosine-lobe L1 SH directional modulation
    float dirMod = clamp(1.0 + 0.65 * dot(shLuma.yzw, queryNormal) / max(L0, 1e-5), 0.25, 1.75);
    return rgb * dirMod;
}

// Direct O(1) fetch from one cascade (ZERO loops!).
bool rcFetchDirect(ivec3 worldCell, uint cascade, vec3 queryNormal, out vec3 outRad, out uint outFrames) {
    outRad = vec3(0.0);
    outFrames = 0u;
    uint slot = rcStorageIndex(worldCell, cascade);
    RCProbe p = rcClipmap.cells[slot];
    uint key = rcPackWorldKey(worldCell, cascade);
    if (p.worldKey != key) return false;
    uint nf = rcUnpackFrames(p.chromaFrames);
    if (nf == 0u) return false;
    outRad = rcEvalProbe(p, queryNormal);
    outFrames = nf;
    return true;
}

// Helper: true iff world-space block coordinate `worldBlock` is an opaque Track-1 solid full cube.
// Used to make 3-way / 90-degree indoor corners 100% airtight against diagonal RC sampling leaks!
bool rcIsSolidCubeBlock(ivec3 worldBlock) {
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
        // Treat translucent glass cubes as open air for RC light propagation, opaque cubes as solid walls.
        for (int f = 0; f < 6; ++f) {
            if (sv.face[f] != 0u && ((sv.face[f] >> 26u) & 1u) == 0u) return true;
        }
        return false;
    #endif
}

// Helper: checks whether stepping from `baseCell` to `candCell` (cascade 0) is open-air connected
// without entering a solid wall OR jumping diagonally across a 90-degree / 3-way concave corner!
bool rcCellVisibleFromBase(ivec3 baseCell, ivec3 candCell, uint cascade) {
    if (cascade != 0u || all(equal(baseCell, candCell))) return true;
    if (rcIsSolidCubeBlock(candCell)) return false;
    ivec3 d = candCell - baseCell;
    int manhattan = abs(d.x) + abs(d.y) + abs(d.z);
    if (manhattan > 1) {
        // Reject diagonal jump across a hollow wall-ceiling corner if ANY axial bridge voxel is solid!
        if (d.x != 0 && rcIsSolidCubeBlock(baseCell + ivec3(d.x, 0, 0))) return false;
        if (d.y != 0 && rcIsSolidCubeBlock(baseCell + ivec3(0, d.y, 0))) return false;
        if (d.z != 0 && rcIsSolidCubeBlock(baseCell + ivec3(0, 0, d.z))) return false;
    }
    return true;
}

// O(1) lookup with coarser cascade fallback (for diffuse bounce & debug view).
vec3 queryRadianceCache(vec3 worldPos, vec3 normal, uint rank, bool allowRequest) {
    uint cascade = rcSelectCascade(worldPos);
    ivec3 cell = rcWorldCell(worldPos, normal, cascade);
    vec3 rad;
    uint nf;
    bool hit = rcFetchDirect(cell, cascade, normal, rad, nf);
    if (allowRequest) {
        rcRequestProbe(cell, cascade, hit, nf);
    }
    // Never bleed coarse 2x2x2 outdoor sky probes into a valid cascade-0 indoor cell!
    if (hit && (cascade == 0u || nf >= 4u)) return rad;

    // Fallback to parent cascade only if current cell has no data yet.
    if (cascade < 2u) {
        ivec3 pCell = cell >> 1;
        vec3 pRad;
        uint pNf;
        if (rcFetchDirect(pCell, cascade + 1u, normal, pRad, pNf) && pNf >= 2u) {
            return hit ? mix(pRad, rad, smoothstep(0.0, 4.0, float(nf))) : pRad;
        }
    }
    return rad;
}

// Fast stochastic tangent-jittered O(1) lookup (wall- & 3-way-corner-safe: never jitters into outdoor sky!).
vec3 queryRadianceCacheStochastic(vec3 worldPos, vec3 normal, uint rank, vec2 rand) {
    uint cascade = rcSelectCascade(worldPos);
    float cellSize = float(1u << cascade);
    ivec3 baseCell = rcWorldCell(worldPos, normal, cascade);

    float phi = TWO_PI * rand.x;
    float r = 0.45 * cellSize * sqrt(rand.y);
    vec3 jPos = worldPos + buildTBN(normal) * vec3(r * cos(phi), r * sin(phi), 0.0);
    ivec3 cell = rcWorldCell(jPos, normal, cascade);
    if (!rcCellVisibleFromBase(baseCell, cell, cascade)) {
        cell = baseCell;
    }

    vec3 rad;
    uint nf;
    bool hit = rcFetchDirect(cell, cascade, normal, rad, nf);
    rcRequestProbe(cell, cascade, hit, nf);
    if (hit && (cascade == 0u || nf >= 4u)) return rad;

    if (cascade < 2u) {
        ivec3 pCell = cell >> 1;
        vec3 pRad;
        uint pNf;
        if (rcFetchDirect(pCell, cascade + 1u, normal, pRad, pNf) && pNf >= 2u) {
            return hit ? mix(pRad, rad, smoothstep(0.0, 4.0, float(nf))) : pRad;
        }
    }
    return rad;
}

// O(1) 4-tap tangent-plane bilinear smooth lookup for mirror/smooth reflections:
// Only 4 direct O(1) L2-cached clipmap reads (ZERO loops -> instant compile, silky smooth mirrors!).
vec3 queryRadianceCacheSmooth(vec3 worldPos, vec3 normal, vec3 skyFallback, uint rank, vec2 rand) {
    uint cascade = rcSelectCascade(worldPos);
    float cellSize = float(1u << cascade);
    vec3 shifted = worldPos + cameraPositionFract + normal * (0.45 * cellSize);
    vec3 fCell = (vec3(cameraPositionInt) + shifted) / cellSize - 0.5;
    ivec3 baseCell = ivec3(floor(fCell));
    vec3 frac = fCell - vec3(baseCell);

    // Request center cell for worklist updates.
    ivec3 centerCell = rcWorldCell(worldPos, normal, cascade);
    vec3 dummyRad;
    uint dummyNf;
    bool centerHit = rcFetchDirect(centerCell, cascade, normal, dummyRad, dummyNf);
    rcRequestProbe(centerCell, cascade, centerHit, dummyNf);

    // Pick the 2 tangent axes perpendicular to the dominant normal axis for exact 2x2 bilinear blend!
    vec3 an = abs(normal);
    ivec3 ax0, ax1;
    vec2 tf;
    int fixedCoord;
    if (an.x >= an.y && an.x >= an.z) {
        ax0 = ivec3(0, 1, 0); ax1 = ivec3(0, 0, 1); tf = frac.yz;
        baseCell.x = centerCell.x;
    } else if (an.y >= an.z) {
        ax0 = ivec3(1, 0, 0); ax1 = ivec3(0, 0, 1); tf = frac.xz;
        baseCell.y = centerCell.y;
    } else {
        ax0 = ivec3(1, 0, 0); ax1 = ivec3(0, 1, 0); tf = frac.xy;
        baseCell.z = centerCell.z;
    }

    vec3 sumRad = vec3(0.0);
    float sumW = 0.0;

    int reqSlot = frameCounter & 3;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            float bw = (i == 0 ? 1.0 - tf.x : tf.x) * (j == 0 ? 1.0 - tf.y : tf.y);
            ivec3 c = baseCell + ax0 * i + ax1 * j;
            bool isCenter = all(equal(c, centerCell));
            // Block both solid wall voxels AND diagonal jumps across hollow 3-way indoor corners!
            if (!isCenter && !rcCellVisibleFromBase(centerCell, c, cascade)) {
                continue;
            }
            vec3 r;
            uint nf;
            bool tapHit = rcFetchDirect(c, cascade, normal, r, nf);
            // Round-robin request bilinear neighbor cells so reflected walls keep all 4 corners converged!
            if (!isCenter && ((j * 2 + i) == reqSlot)) {
                rcRequestProbe(c, cascade, tapHit, nf);
            }
            if (tapHit) {
                float conf = smoothstep(0.0, 8.0, float(nf));
                sumRad += mix(skyFallback, r, conf) * bw;
                sumW   += bw;
            } else if (cascade < 2u && (isCenter || !centerHit)) {
                vec3 pr;
                uint pnf;
                if (rcFetchDirect(c >> 1, cascade + 1u, normal, pr, pnf)) {
                    float conf = smoothstep(0.0, 6.0, float(pnf));
                    sumRad += mix(skyFallback, pr, conf) * (bw * 0.6);
                    sumW   += bw * 0.6;
                }
            }
        }
    }

    return (sumW > 1e-4) ? (sumRad / sumW) : skyFallback;
}

// O(1) L1 SH vMF Path Guiding lookup with tangent jitter (ZERO loops!).
bool queryRCPathGuide(vec3 worldPos, vec3 normal, bool allowRequest, vec2 jitterRand, out vec3 guideDir, out float guideKappa, out float guideProb) {
    guideDir = normal;
    guideKappa = 0.0;
    guideProb = 0.0;

    #ifndef PATH_GUIDING
        return false;
    #endif

    uint cascade = rcSelectCascade(worldPos);
    float cellSize = float(1u << cascade);
    ivec3 baseCell = rcWorldCell(worldPos, normal, cascade);
    float phi = TWO_PI * jitterRand.x;
    float r = 0.40 * cellSize * sqrt(jitterRand.y);
    vec3 jPos = worldPos + buildTBN(normal) * vec3(r * cos(phi), r * sin(phi), 0.0);

    ivec3 cell = rcWorldCell(jPos, normal, cascade);
    if (!rcCellVisibleFromBase(baseCell, cell, cascade)) {
        cell = baseCell;
    }
    uint slot = rcStorageIndex(cell, cascade);
    RCProbe p = rcClipmap.cells[slot];
    uint key = rcPackWorldKey(cell, cascade);
    bool valid = (p.worldKey == key);
    uint nf = valid ? rcUnpackFrames(p.chromaFrames) : 0u;

    if (allowRequest) {
        rcRequestProbe(cell, cascade, valid, nf);
    }

    if (valid && nf >= 12u) {
        vec4 sh = unpackHalf4x16(p.shLuma);
        if (sh.x > 1e-4 && !any(isnan(sh)) && !any(isinf(sh))) {
            float lenL1 = length(sh.yzw);
            float conf = saturate((float(nf) - 8.0) / 32.0);
            float rho = clamp(lenL1 / max(sh.x, 1e-6), 0.0, 0.85) * conf;
            if (rho > 0.12) {
                vec3 rawDir = sh.yzw / max(lenL1, 1e-6);
                if (dot(rawDir, normal) > -0.2) {
                    guideDir = normalize(rawDir + normal * 0.20);
                    guideKappa = clamp((3.0 * rho - rho * rho * rho) / max(1.0 - rho * rho, 0.12), 0.5, 6.0);
                    guideProb = clamp(0.55 * rho, 0.0, 0.50);
                    return true;
                }
            }
        }
    }
    return false;
}

#endif
