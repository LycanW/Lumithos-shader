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

    uint rng = pcgHash(slot ^ (uint(frameCounter) * 0x9e3779b9u));
    int spp = (accumFrames < 4u) ? 4 : ((accumFrames < RC_WARMUP_UPDATES) ? 2 : 1);

    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();

    vec3 sumYcc = vec3(0.0);
    vec3 sumL1  = vec3(0.0);
    vec2 baseStrat = nextVec2(rng);

    for (int i = 0; i < spp; ++i) {
        // Stratified full-sphere direction sampling (encodes all 6 face directions cleanly in L1 SH!).
        vec2 u = fract(baseStrat + vec2(float(i) / float(spp), float(i) * 0.6180339887));
        float z = 1.0 - 2.0 * u.x;
        float r = sqrt(max(0.0, 1.0 - z * z));
        float phi = TWO_PI * u.y;
        vec3 rayDir = vec3(r * cos(phi), z, r * sin(phi));

        // Cap jitter to <=0.35 blocks and verify ray origin stays inside an air voxel
        // so coarse cascade cells never poke through a 1-block wall into outdoor sunlight!
        float maxJit = min(0.35 * cellSize, 0.35);
        vec3 jitter = (vec3(nextFloat(rng), nextFloat(rng), nextFloat(rng)) - 0.5) * maxJit;
        vec3 rayOrig = probeWorld + jitter;
        ivec3 origVox = ivec3(floor(worldToVoxelPos(rayOrig)));
        if (inVoxelBounds(origVox) && texelFetch(voxelGridTex, origVox, 0).r != 0u) {
            rayOrig = probeWorld;
        }

        TraceHit hit = traceScene(rayOrig, rayDir, RC_MAX_STEPS, 1.0);
        vec3 rad = vec3(0.0);
        if (hit.hit) {
            rad += hit.albedo * (hit.emission * EMISSIVE_GI_SCALE * hit.emitFluxScale);
            float NdotL = max(dot(hit.normal, lightDir), 0.0);
            if (NdotL > 0.0) {
                float vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
                rad += hit.albedo * INV_PI * directCol * (NdotL * vis);
            }
            // Recursive multi-bounce from O(1) clipmap (no new worklist request inside update).
            vec3 bounce = queryRadianceCache(hit.worldPos, hit.normal, 1u, false);
            rad += hit.albedo * bounce * 0.85;
        } else if (hit.exitedGrid) {
            rad = evalSky(rayDir, false);
        } else {
            rad = queryRadianceCache(hit.worldPos, -rayDir, 1u, false);
        }

        rad = min(max(rad, vec3(0.0)), vec3(64.0));
        if (any(isnan(rad)) || any(isinf(rad))) rad = vec3(0.0);

        vec3 ycc = rgbToYCoCg(rad);
        sumYcc += ycc;
        sumL1  += ycc.x * rayDir;
    }

    float invSpp = 1.0 / float(spp);
    vec3 currYcc = sumYcc * invSpp;
    vec3 currL1  = sumL1  * invSpp;

    // Gentle firefly guard on settled probes (high enough floor so nearby torches aren't clipped!)
    if (accumFrames >= 6u && prevYcc.x > 1e-5) {
        float maxY = max(prevYcc.x * 6.0, 4.0);
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

    float l1Len = length(nextL1);
    if (l1Len > nextYcc.x && l1Len > 1e-6) {
        nextL1 *= nextYcc.x / l1Len;
    }

    vec2 normCoCg = (nextYcc.x > 1e-6) ? clamp(nextYcc.yz / nextYcc.x, vec2(-1.0), vec2(1.0)) : vec2(0.0);

    RCProbe outProbe;
    outProbe.worldKey     = key;
    outProbe.chromaFrames = rcPackChromaFrames(normCoCg, uint(round(nextFrames)));
    outProbe.shLuma       = packHalf4x16(vec4(nextYcc.x, nextL1));
    rcClipmap.cells[slot] = outProbe;
}
