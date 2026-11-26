#version 430 compatibility

// Resolve the material emission of the voxel lights registered by shadow.gsh and bin them for NEE.
#include "/lib/SceneGeo.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(64, 1, 1); // 64 * 64 = 4,096 = MAX_SCENE_LIGHTS

float lightTexelEmission(vec2 uv, ivec2 specSize, float blockEmit, vec3 albedo, out bool authored) {
    ivec2 px = clamp(ivec2(uv * vec2(specSize)), ivec2(0), specSize - 1);
    vec2 ga = texelFetch(colortex15, px, 0).ga;
    float pbr = decodeLabPBREmission(ga.y);
    authored = pbr > 0.0;
    return surfaceEmission(pbr, blockEmit, ga.x, albedo);
}

void binLight(ivec3 voxel, uint head, uint idx, float strength, vec3 color) {
    if (strength <= 1e-4 || maxOf(color) <= 0.0) return;
    uint address = voxelLinearIndex(voxel);
    if (atomicCompSwap(surfacePool.surfaces[address].primHead, head, head | 0x80000000u) != head) return;
    ivec3 cell = voxel / 32;
    uint id = uint(cell.x + LIGHT_CELL_XZ * (cell.z + LIGHT_CELL_XZ * cell.y));
    uint slot = atomicAdd(lightList.cellCount[id], 1u);
    if (slot < LIGHT_CELL_CAP) {
        lightList.cellLights[id * LIGHT_CELL_CAP + slot] = idx;
    }
    surfacePool.surfaces[address].primHead = head | ((slot < LIGHT_CELL_CAP ? slot + 1u : 0u) << 21u);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint lCount = min(lightList.lightCount, MAX_SCENE_LIGHTS);
    if (idx < lCount) {
        vec4 pAndP = lightList.lights[idx].posAndPower;
        ivec3 v = ivec3(floor(pAndP.xyz));
        if (!inVoxelBounds(v)) {
            lightList.lights[idx].posAndPower.w = 0.0;
            return;
        }
        uint slot = voxelLinearIndex(v);
        SurfaceVoxel sv = readSurface(slot);
        ivec2 atlasTexSize = max(textureSize(colortex14, 0), ivec2(1));
        vec2  invAtlasSize = 1.0 / vec2(atlasTexSize);
        ivec2 specSize = textureSize(colortex15, 0);

        // 1) Track 1 full-cube emitters
        uint bestDesc = 0u;
        uint bestFace = 0u;
        uint bestEmit4 = 0u;
        for (uint f = 0u; f < 6u; ++f) {
            uint d = sv.face[f];
            if (d == 0u) continue;
            uint e4 = (d >> 27u) & 15u;
            if (e4 > bestEmit4) {
                bestEmit4 = e4;
                bestDesc = d;
                bestFace = f;
            }
        }
        if (bestEmit4 > 0u) {
            bool isTinted, isTrans;
            float blockEmit;
            vec2 uv = decodeFaceAtlasUV(bestDesc, bestFace, vec3(0.5), invAtlasSize, isTinted, isTrans, blockEmit);
            vec4 tex = textureLod(colortex14, uv, 2.0);
            vec3 tintSrgb = isTinted ? unpackTint(sv.tintAndSky) : vec3(1.0);
            vec3 albLin = srgbToLinear(tex.rgb * tintSrgb);
            bool authored;
            float emitStr = lightTexelEmission(uv, specSize, blockEmit * (isTrans ? tex.a : 1.0), albLin, authored);
            vec3 finalPos = vec3(v) + 0.5;
            lightList.lights[idx].posAndPower  = vec4(finalPos, emitStr);
            lightList.lights[idx].colorAndMeta = vec4(albLin, 0.5);
            binLight(v, sv.primHead, idx, emitStr, albLin);
            return;
        }

        // 2) Track 2 non-cube emitters (torches, lanterns, candles, end rods):
        // Sample 4 barycentric points per emissive triangle and average the hottest core texels
        // so tall thin meshes (like torch sticks + tiny top flame) always extract the true flame color & centroid!
        uint primIdx = sv.primHead;
        vec3  sumCol  = vec3(0.0);
        vec3  sumPos  = vec3(0.0);
        float sumW    = 0.0;
        float bestStr = 0.0;
        bool hasAuthored = false;
        const vec3 baryPts[4] = vec3[4](
            vec3(0.3333, 0.3333, 0.3334),
            vec3(0.6666, 0.1667, 0.1667),
            vec3(0.1667, 0.6666, 0.1667),
            vec3(0.1667, 0.1667, 0.6666)
        );
        for (int guard = 0; guard < 160 && primIdx != PRIM_END; ++guard) {
            ScenePrim pr = primPool.prims[primIdx];
            uint nextIdx = pr.meta.x;
            uint e4 = (pr.meta.y >> 24u) & 15u;
            if (e4 > 0u) {
                vec3 p0, e1, e2, tintSrgb;
                vec2 uv0, uv1, uv2;
                float primEmit;
                bool primTrans;
                uint dummyNext;
                unpackScenePrim(pr, p0, e1, e2, uv0, uv1, uv2, tintSrgb, primEmit, primTrans, dummyNext);
                for (int s = 0; s < 4; ++s) {
                    vec3 b = baryPts[s];
                    vec2 sUV = uv0 * b.x + uv1 * b.y + uv2 * b.z;
                    vec4 tex = textureLod(colortex14, sUV, 0.0);
                    if (tex.a < 0.1) continue;
                    vec3 albLin = srgbToLinear(tex.rgb * tintSrgb);
                    float maxAlb = maxOf(albLin);
                    bool authored;
                    float emitStr = lightTexelEmission(sUV, specSize, primEmit * (primTrans ? tex.a : 1.0), albLin, authored);
                    if (hasAuthored && !authored) continue;
                    if (authored && !hasAuthored) {
                        // A material-authored emitter supersedes block-level estimates collected so far.
                        hasAuthored = true;
                        sumW = bestStr = 0.0;
                        sumCol = sumPos = vec3(0.0);
                    }
                    float w = emitStr * maxAlb * maxAlb;
                    vec3 sPos = vec3(v) + clamp(p0 + e1 * b.y + e2 * b.z, vec3(0.05), vec3(0.95));
                    sumCol += albLin * w;
                    sumPos += sPos * w;
                    sumW   += w;
                    bestStr = max(bestStr, emitStr);
                }
            }
            primIdx = nextIdx;
        }
        vec3 bestCol = (sumW > 0.0) ? (sumCol / sumW) : vec3(0.0);
        vec3 bestPos = (sumW > 0.0) ? (sumPos / sumW) : (vec3(v) + 0.5);
        lightList.lights[idx].posAndPower  = vec4(bestPos, bestStr);
        lightList.lights[idx].colorAndMeta = vec4(bestCol, 0.18);
        binLight(v, sv.primHead, idx, bestStr, bestCol);
    }
}
