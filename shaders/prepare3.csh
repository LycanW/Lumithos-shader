#version 430 compatibility

// Pass: Zero-Queue In-Place World-Space Surface Hash Cache Updater
// - Scans half of the 262,144-slot 4-way hash table per frame (L2-resident key+meta early-out!).
// - Auto-GCs stale slots (> SC_STALE_FRAMES) to keep bucket load factor < 15% everywhere.
// - Traces strictly 1 SPP cosine-weighted hemisphere + exact face-normal Local-Grid RIS NEE
//   (2x higher sample efficiency than omni-directional air probes, zero wall-backside leaks!).
#include "/lib/EmitterSampling.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(2048, 1, 1); // 2048 * 64 = 131,072 = SC_TOTAL_SLOTS / 2

void main() {
    uint slot = gl_GlobalInvocationID.x * 2u + (uint(frameCounter) & 1u);
    if (slot >= SC_TOTAL_SLOTS) return;

    uint key = scTable.entries[slot].key;
    if (key == 0u) return;

    uint meta  = scTable.entries[slot].meta;
    uint currF = scCurrFrame14();
    uint lastF = (meta >> 10u) & 16383u;
    uint frameAge = (currF - lastF) & 16383u;

    // Automatic garbage collection of unseen surface patches keeps 4-slot buckets permanently sparse.
    if (frameAge > SC_STALE_FRAMES) {
        scTable.entries[slot] = SurfaceCacheEntry(0u, 0u, uvec2(0u));
        return;
    }
    if (frameAge > SC_ACTIVE_WINDOW) return;

    ivec3 modCell;
    uint faceIdx, lod;
    scUnpackKey(key, modCell, faceIdx, lod);
    if (faceIdx >= 6u) return;

    uint accum = meta & 1023u;

    // 3D spatial lattice refresh cadence (zero hash-stagger overlap):
    // - Freshly allocated patches (< SC_WARMUP_SAMPLES): warm up every 2 frames!
    // - Mature patches: lod 0 (0.5m) every 4 frames, lod 1 (1m) every 8 frames, lod 2/3 (2m/4m) every 16 frames!
    uint halfFrame = uint(frameCounter) >> 1u;
    uint lattice = uint(modCell.x + modCell.y * 2 + modCell.z * 3);
    if (accum >= SC_WARMUP_SAMPLES) {
        if (lod == 0u) {
            if (((lattice + halfFrame) & 1u) != 0u) return;
        } else if (lod == 1u) {
            if (((lattice + halfFrame) & 3u) != 0u) return;
        } else {
            if (((lattice + halfFrame) & 7u) != 0u) return;
        }
    }

    vec3 faceNormal, T, B;
    scFaceBasis(faceIdx, faceNormal, T, B);
    float cSize = scCellSize(lod);

    // Exact modular reconstruction of camera-relative world position from the lossless 9-bit cell key.
    ivec3 relCell;
    vec3 patchWorld;
    if (lod == 0u) {
        ivec3 camCell = (cameraPositionInt & 255) << 1;
        relCell = ((modCell - camCell + 256) & 511) - 256;
        patchWorld = (vec3(relCell) + 0.5) * 0.5 - cameraPositionFract;
    } else {
        int shift = int(lod - 1u);
        ivec3 camCell = cameraPositionInt >> shift;
        relCell = ((modCell - (camCell & 511) + 256) & 511) - 256;
        ivec3 absOrigin = (camCell + relCell) << shift;
        patchWorld = vec3(absOrigin - cameraPositionInt) - cameraPositionFract + vec3(0.5 * cSize);
    }

    // Drop patches that drifted far outside our 160x96x160 voxel horizon.
    if (max(abs(patchWorld.x), abs(patchWorld.z)) > 110.0 || abs(patchWorld.y) > 72.0) {
        return;
    }

    // 3D spatiotemporal blue noise indexed by absolute cell coordinate & per-LOD update step.
    uint periodShift = (accum < SC_WARMUP_SAMPLES) ? 1u : (lod == 0u ? 2u : (lod == 1u ? 3u : 4u));
    uint updateSeq = uint(frameCounter) >> periodShift;
    vec3 uDir3 = stbn3D(modCell, updateSeq, faceIdx);
    vec3 uJit3 = stbn3D(modCell, updateSeq, faceIdx + 6u);
    vec3 uNee3 = stbn3D(modCell, updateSeq, faceIdx + 12u);

    // Tangent-plane sub-patch jitter stays strictly inside open air adjacent to the surface face.
    float jitRadius = min(0.22 * cSize, 0.28);
    vec3 rayOrig = patchWorld + T * ((uJit3.x - 0.5) * 2.0 * jitRadius) + B * ((uJit3.y - 0.5) * 2.0 * jitRadius);
    ivec3 origVox = ivec3(floor(worldToVoxelPos(rayOrig)));
    if (inVoxelBounds(origVox) && texelFetch(voxelGridTex, origVox, 0).r != 0u) {
        rayOrig = patchWorld + faceNormal * min(0.35 * cSize, 0.45);
        origVox = ivec3(floor(worldToVoxelPos(rayOrig)));
        if (inVoxelBounds(origVox) && texelFetch(voxelGridTex, origVox, 0).r != 0u) {
            rayOrig = patchWorld + faceNormal * 0.55;
        }
    }

    bool useNEE = (lightList.lightCount > 0u);

    // 1) 1-SPP Local-Grid RIS NEE using exact surface faceNormal (not crude 0.25 spherical average!).
    vec3 neeDir = faceNormal;
    vec2 surfaceRandom = stbn3D(modCell, updateSeq, faceIdx + 18u).xy;
    vec3 neeRad = useNEE ? sampleDirectLightsNEE(rayOrig, faceNormal, 24, uNee3.xy, surfaceRandom, neeDir) : vec3(0.0);
    neeRad = min(max(neeRad, vec3(0.0)), vec3(16.0));
    if (any(isnan(neeRad)) || any(isinf(neeRad))) neeRad = vec3(0.0);

    // 2) 1-SPP Cosine-Weighted Hemisphere indirect bounce + sky ray (exact diffuse irradiance estimator!).
    vec3 rayDir = sampleCosineHemisphere(faceNormal, uDir3.xy);

    vec3 lightDir  = getShadowLightDir();
    vec3 directCol = getDirectLightColor();

    TraceHit hit = traceScene(rayOrig, rayDir, RC_MAX_STEPS, 1.0, true);
    vec3 indRad = vec3(0.0);
    if (hit.hit) {
        indRad += indirectHitEmission(hit, rayOrig, faceNormal);
        float NdotL = max(dot(hit.normal, lightDir), 0.0);
        vec3 direct = vec3(0.0);
        if (NdotL > 0.0) {
            vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
            direct = directCol * (NdotL * vis);
        }
        // Recursive multi-bounce read-only lookup from the Surface Hash Cache!
        vec3 bounce = queryRadianceCache(hit.worldPos, hit.normal, 1u, false);
        indRad += secondaryHitLighting(hit, rayDir, lightDir, direct, bounce, false);
        indRad *= hit.throughput;
    } else if (hit.exitedGrid) {
        indRad = evalSky(rayDir, false) * hit.throughput;
    } else {
        indRad = queryRadianceCache(hit.worldPos, -rayDir, 1u, false) * hit.throughput;
    }

    indRad = min(max(indRad, vec3(0.0)), vec3(16.0));
    if (any(isnan(indRad)) || any(isinf(indRad))) indRad = vec3(0.0);

    // Decode previous full-RGB irradiance + dominant incoming guide direction.
    uvec2 prevPack = scTable.entries[slot].rad;
    vec2 pRG = unpackHalf2x16(prevPack.x);
    vec2 pBG = unpackHalf2x16(prevPack.y);
    vec3 prevRad = (accum > 0u) ? max(vec3(pRG, pBG.x), vec3(0.0)) : vec3(0.0);
    float prevRho = (accum > 0u) ? saturate(pBG.y) : 0.0;
    if (any(isnan(prevRad)) || any(isinf(prevRad))) { prevRad = vec3(0.0); prevRho = 0.0; accum = 0u; }
    vec3 prevGuideDir = (accum > 0u) ? scUnpackGuideDir((meta >> 24u) & 255u, faceIdx) : faceNormal;

    // Coplanar 4-neighbor same-face relaxation for blue-noise cancellation along walls/floors (0 cross-wall bleed!).
    ivec3 ax0 = ivec3(round(T));
    ivec3 ax1 = ivec3(round(B));
    const ivec2 offs2[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
    ivec3 baseBlock = cameraPositionInt + ivec3(floor(patchWorld + cameraPositionFract));
    vec3 nbrRad = vec3(0.0);
    float nbrW  = 0.0;

    for (int k = 0; k < 4; ++k) {
        ivec3 nc = (modCell + ax0 * offs2[k].x + ax1 * offs2[k].y) & 511;
        if (lod <= 1u) {
            vec3 nWorld = patchWorld + T * (float(offs2[k].x) * cSize) + B * (float(offs2[k].y) * cSize);
            ivec3 nBlock = cameraPositionInt + ivec3(floor(nWorld + cameraPositionFract));
            if (!scCellVisibleFromBase(baseBlock, nBlock, lod)) continue;
        }
        vec3 nr;
        uint nAcc, nGUV;
        float nRho;
        if (scLookupSlot(scPackKey(nc, faceIdx, lod), false, false, nr, nAcc, nGUV, nRho) && nAcc >= 4u) {
            float nw = min(float(nAcc), 16.0) * (1.0 / 16.0);
            nbrRad += nr * nw;
            nbrW   += nw;
        }
    }
    vec3 nbrMean = (nbrW > 0.25) ? (nbrRad / nbrW) : prevRad;

    vec3 currRad = indRad + neeRad;

    // Gentle firefly guard on extreme outliers (only after warm-up).
    float prevLuma = luma(prevRad);
    float currLuma = luma(currRad);
    if (accum >= 6u && prevLuma > 1e-5) {
        float refL = max(prevLuma, luma(nbrMean));
        float maxL = max(refL * 4.5 + 0.5, 1.0);
        if (currLuma > maxL) {
            float s = maxL / currLuma;
            currRad *= s;
            indRad  *= s;
            neeRad  *= s;
            currLuma = maxL;
        }
    }

    float maxAccum = (lod == 0u) ? SC_MAX_ACCUM_NEAR : ((lod == 1u) ? SC_MAX_ACCUM_MID : SC_MAX_ACCUM_FAR);
    float nextAccum = min(float(accum) + 1.0, maxAccum);
    float blend = 1.0 / nextAccum;

    vec3 nextRad = mix(prevRad, currRad, blend);
    if (nbrW > 0.25) {
        float diffBlend = (accum < 8u) ? 0.22 : ((lod == 0u) ? 0.05 : 0.03);
        nextRad = mix(nextRad, nbrMean, diffBlend);
    }

    // Update surface-hemisphere dominant light vector for native vMF Path Guiding (zero L1 SH!).
    vec3 prevVec = prevGuideDir * (prevRho * max(prevLuma, 1e-4));
    vec3 sampleVec = rayDir * luma(indRad) + neeDir * luma(neeRad);
    vec3 nextVec = mix(prevVec, sampleVec, blend);
    float nextLuma = max(luma(nextRad), 1e-5);
    float vecLen = length(nextVec);
    vec3 nextGuideDir = (vecLen > 1e-5) ? (nextVec / vecLen) : faceNormal;
    if (dot(nextGuideDir, faceNormal) < 0.05) {
        nextGuideDir = normalize(nextGuideDir + faceNormal * 0.25);
    }
    float nextRho = saturate(vecLen / nextLuma);
    uint nextGuideUV = scPackGuideUV(nextGuideDir, faceIdx);

    // Preserve any fresher lastUsedFrame written by deferred queries while updating accum & guideUV.
    uint freshF = (scTable.entries[slot].meta >> 10u) & 16383u;
    scTable.entries[slot].meta = scPackMeta(uint(round(nextAccum)), freshF, nextGuideUV);
    scTable.entries[slot].rad  = uvec2(packHalf2x16(nextRad.rg), packHalf2x16(vec2(nextRad.b, nextRho)));
}
