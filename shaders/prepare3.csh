#version 430 compatibility

// Pass: Compacted Active-Set Radiance Cache Update (max 8,192 active probes/frame -> 8x fewer threads!).
// Traces ONLY probes actually requested on screen / in reflections from previous frame's worklist.
#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(128, 1, 1); // 128 * 64 = 8,192 max active probes

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint prevSide = (uint(frameCounter) & 1u) ^ 1u;
    uint workCount = min(rcWork.count[prevSide], RC_WORKLIST_CAP);
    if (idx >= workCount) return;

    uint packedReq = rcWork.data[2u * RC_DEDUP_WORDS + prevSide * RC_WORKLIST_CAP + idx];
    ivec3 rel = ivec3(int(packedReq & 63u), int((packedReq >> 6u) & 31u), int((packedReq >> 11u) & 63u));
    uint cascade = (packedReq >> 17u) & 3u;
    if (cascade >= uint(RC_CASCADES)) return;

    // Reconstruct exact world cell from previous frame's camera grid origin.
    ivec3 prevCamCell = previousCameraPositionInt >> int(cascade);
    ivec3 worldCell = prevCamCell + (rel - ivec3(32, 16, 32));
    uint slot = rcStorageIndex(worldCell, cascade);
    uint key  = rcPackWorldKey(worldCell, cascade);

    RCProbe prev = rcClipmap.cells[slot];
    bool isSameCell = (prev.worldKey == key);
    uint accumFrames = isSameCell ? rcUnpackFrames(prev.chromaFrames) : 0u;

    vec4 prevShLuma = isSameCell ? unpackHalf4x16(prev.shLuma) : vec4(0.0);
    vec2 prevNormCoCg = isSameCell ? rcUnpackNormCoCg(prev.chromaFrames) : vec2(0.0);
    if (any(isnan(prevShLuma)) || any(isinf(prevShLuma))) { prevShLuma = vec4(0.0); accumFrames = 0u; }
    vec3 prevYcc = vec3(max(prevShLuma.x, 0.0), prevNormCoCg * max(prevShLuma.x, 0.0));

    float cellSize = float(1u << cascade);
    vec3 probeWorld = vec3((worldCell << int(cascade)) - cameraPositionInt) - cameraPositionFract + vec3(0.5 * cellSize);

    // Strictly 1 SPP Radiance Cache update:
    // 1) 1-ray RIS NEE against Emissive Voxel LightList (zero-variance direct torch/lantern/lamp lighting!)
    // 2) 1-ray 3D Blue-Noise indirect bounce + sky trace (skipping blockEmit double-counting when NEE active)
    uint updateSeq = uint(frameCounter) / RC_UPDATE_PERIOD;
    vec3 uDir3 = stbn3D(worldCell, updateSeq, 0u);
    vec3 uJit3 = stbn3D(worldCell, updateSeq, 1u);
    vec3 uNee3 = stbn3D(worldCell, updateSeq, 2u);

    // Blue-noise sub-voxel position jitter (capped <=0.20 blocks) stays strictly inside air.
    float maxJit = min(0.20 * cellSize, 0.20);
    vec3 jitter = (uJit3 - 0.5) * maxJit;
    vec3 rayOrig = probeWorld + jitter;
    ivec3 origVox = ivec3(floor(worldToVoxelPos(rayOrig)));
    if (inVoxelBounds(origVox) && texelFetch(voxelGridTex, origVox, 0).r != 0u) {
        rayOrig = probeWorld;
    }

    bool useNEE = (lightList.lightCount > 0u);

    // 1) 1-SPP Local-Grid RIS Next-Event Estimation for all block light sources (torches, lanterns, lamps, lava).
    vec3 neeDir = vec3(0.0, 1.0, 0.0);
    vec3 neeRad = useNEE ? sampleDirectLightsNEE(rayOrig, vec3(0.0, 1.0, 0.0), true, 24, uNee3.xy, neeDir) : vec3(0.0);
    neeRad = min(max(neeRad, vec3(0.0)), vec3(16.0));
    if (any(isnan(neeRad)) || any(isinf(neeRad))) neeRad = vec3(0.0);
    vec3 neeYcc = rgbToYCoCg(neeRad);

    // 2) 1-SPP 3D Blue-Noise full-sphere indirect bounce + sky ray.
    float z = 1.0 - 2.0 * uDir3.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = TWO_PI * uDir3.y;
    vec3 rayDir = vec3(r * cos(phi), z, r * sin(phi));

    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();

    TraceHit hit = traceScene(rayOrig, rayDir, RC_MAX_STEPS, 1.0, true);
    vec3 indRad = vec3(0.0);
    if (hit.hit) {
        // Avoid double-counting block emitters already integrated cleanly by NEE!
        if (!useNEE || !hit.isBlockEmit) {
            indRad += hit.emitColor * (hit.emission * EMISSIVE_GI_SCALE * hit.emitFluxScale);
        }
        float NdotL = max(dot(hit.normal, lightDir), 0.0);
        if (NdotL > 0.0) {
            vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
            indRad += hit.albedo * INV_PI * directCol * (NdotL * vis);
        }
        // Recursive multi-bounce from O(1) clipmap (no new worklist request inside update).
        vec3 bounce = queryRadianceCache(hit.worldPos, hit.normal, 1u, false);
        indRad += hit.albedo * bounce * 0.85;
        indRad *= hit.throughput;
    } else if (hit.exitedGrid) {
        indRad = evalSky(rayDir, false) * hit.throughput;
    } else {
        indRad = queryRadianceCache(hit.worldPos, -rayDir, 1u, false) * hit.throughput;
    }

    indRad = min(max(indRad, vec3(0.0)), vec3(16.0));
    if (any(isnan(indRad)) || any(isinf(indRad))) indRad = vec3(0.0);

    vec3 indYcc  = rgbToYCoCg(indRad);
    vec3 currYcc = indYcc + neeYcc;
    vec3 currL1  = indYcc.x * rayDir + neeYcc.x * neeDir;

    // Wall-safe 6-neighbor air-space gather for pairwise 3D blue-noise error cancellation.
    const ivec3 offsets[6] = ivec3[6](
        ivec3( 1, 0, 0), ivec3(-1, 0, 0),
        ivec3( 0, 1, 0), ivec3( 0,-1, 0),
        ivec3( 0, 0, 1), ivec3( 0, 0,-1)
    );
    vec3 nbrYcc = vec3(0.0);
    vec3 nbrL1  = vec3(0.0);
    float nbrW  = 0.0;
    for (int k = 0; k < 6; ++k) {
        ivec3 nc = worldCell + offsets[k];
        if (cascade == 0u) {
            ivec3 nv = (nc - cameraPositionInt) + ivec3(VOXEL_RES_XZ / 2, VOXEL_RES_Y / 2, VOXEL_RES_XZ / 2);
            if (inVoxelBounds(nv) && texelFetch(voxelGridTex, nv, 0).r != 0u) {
                SurfaceVoxel svCheck = surfacePool.surfaces[voxelLinearIndex(nv)];
                bool isSolidCube = (svCheck.primHead == PRIM_END)
                                && ((svCheck.face[0] | svCheck.face[1] | svCheck.face[2] | svCheck.face[3] | svCheck.face[4] | svCheck.face[5]) != 0u);
                if (isSolidCube) continue;
            }
        } else {
            ivec3 midBlock = ((worldCell << int(cascade)) + (nc << int(cascade)) + ivec3(int(cellSize))) >> 1;
            ivec3 mv = (midBlock - cameraPositionInt) + ivec3(VOXEL_RES_XZ / 2, VOXEL_RES_Y / 2, VOXEL_RES_XZ / 2);
            if (inVoxelBounds(mv) && texelFetch(voxelGridTex, mv, 0).r != 0u) {
                SurfaceVoxel svCheck = surfacePool.surfaces[voxelLinearIndex(mv)];
                bool isSolidCube = (svCheck.primHead == PRIM_END)
                                && ((svCheck.face[0] | svCheck.face[1] | svCheck.face[2] | svCheck.face[3] | svCheck.face[4] | svCheck.face[5]) != 0u);
                if (isSolidCube) continue;
            }
        }

        uint nSlot = rcStorageIndex(nc, cascade);
        RCProbe np = rcClipmap.cells[nSlot];
        if (np.worldKey != rcPackWorldKey(nc, cascade)) continue;
        uint nFrames = rcUnpackFrames(np.chromaFrames);
        if (nFrames < 4u) continue;

        vec4 nSh = unpackHalf4x16(np.shLuma);
        if (any(isnan(nSh)) || any(isinf(nSh))) continue;
        float nY = max(nSh.x, 0.0);
        vec2 nCoCg = rcUnpackNormCoCg(np.chromaFrames);
        float nw = min(float(nFrames), 16.0) * (1.0 / 16.0);
        nbrYcc += vec3(nY, nCoCg * nY) * nw;
        nbrL1  += nSh.yzw * nw;
        nbrW   += nw;
    }

    vec3 nbrMeanYcc = (nbrW > 0.25) ? (nbrYcc / nbrW) : prevYcc;
    vec3 nbrMeanL1  = (nbrW > 0.25) ? (nbrL1  / nbrW) : prevShLuma.yzw;

    // Gentle firefly guard only on extreme outliers (NEE already removes 99% of emitter variance!).
    if (accumFrames >= 6u && prevYcc.x > 1e-5) {
        float refY = max(prevYcc.x, nbrMeanYcc.x);
        float maxY = max(refY * 4.5 + 0.5, 1.0);
        if (currYcc.x > maxY) {
            float s = maxY / currYcc.x;
            currYcc *= s;
            currL1  *= s;
        }
    }

    float nextFrames = min(float(accumFrames) + 1.0, RC_MAX_ACCUM);
    float blend = 1.0 / nextFrames;
    vec3 nextYcc = mix(prevYcc, currYcc, blend);
    vec3 nextL1  = mix(prevShLuma.yzw, currL1, blend);

    // Light air-space blue-noise cancellation across open neighbors (zero wall bleed).
    if (nbrW > 0.25) {
        float settledDiff = (cascade == 0u) ? 0.06 : 0.04;
        float diffBlend = (accumFrames < 8u) ? 0.25 : settledDiff;
        nextYcc = mix(nextYcc, nbrMeanYcc, diffBlend);
        nextL1  = mix(nextL1,  nbrMeanL1,  diffBlend);
    }

    float l1Len = length(nextL1);
    if (l1Len > nextYcc.x && l1Len > 1e-6) {
        nextL1 *= nextYcc.x / l1Len;
    }

    vec2 normCoCg = (nextYcc.x > 1e-6)
        ? clamp(nextYcc.yz / nextYcc.x, vec2(-2.0, -1.0), vec2(2.0, 1.0))
        : vec2(0.0);

    RCProbe outProbe;
    outProbe.worldKey     = key;
    outProbe.chromaFrames = rcPackChromaFrames(normCoCg, uint(round(nextFrames)));
    outProbe.shLuma       = packHalf4x16(vec4(nextYcc.x, nextL1));
    rcClipmap.cells[slot] = outProbe;
}
