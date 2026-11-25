#version 430 compatibility

// Pass: Resolve exact glowing color & Track-2 flame centroid for deduplicated emissive voxels
// registered by shadow.gsh, and bin them into the 5x3x5 Local Macro-Cell Grid (<0.01ms total!).
#include "/lib/SceneGeo.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(64, 1, 1); // 64 * 64 = 4,096 = MAX_SCENE_LIGHTS

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
        SurfaceVoxel sv = surfacePool.surfaces[slot];
        ivec2 atlasTexSize = max(textureSize(colortex14, 0), ivec2(1));
        vec2  invAtlasSize = 1.0 / vec2(atlasTexSize);

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
            float emitStr = 0.0;
            vec3 emitCol = evalEmitterColor(albLin, blockEmit, 0.0, emitStr);
            vec3 finalPos = vec3(v) + 0.5;
            lightList.lights[idx].posAndPower  = vec4(finalPos, emitStr);
            lightList.lights[idx].colorAndMeta = vec4(emitCol, 0.5);
            if (emitStr > 1e-4) {
                ivec3 lc = clamp(ivec3(floor(finalPos * (1.0 / 32.0))), ivec3(0), ivec3(LIGHT_CELL_XZ - 1, LIGHT_CELL_Y - 1, LIGHT_CELL_XZ - 1));
                uint cIdx = uint(lc.x + lc.z * LIGHT_CELL_XZ + lc.y * (LIGHT_CELL_XZ * LIGHT_CELL_XZ));
                uint cSlot = atomicAdd(lightList.cellCount[cIdx], 1u);
                if (cSlot < LIGHT_CELL_CAP) {
                    lightList.cellLights[cIdx * LIGHT_CELL_CAP + cSlot] = idx;
                }
            }
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
                    if (maxAlb < 0.25) continue;
                    float emitStr = 0.0;
                    vec3 emitCol = evalEmitterColor(albLin, primEmit, 0.0, emitStr);
                    float w = emitStr * maxAlb * maxAlb;
                    vec3 sPos = vec3(v) + clamp(p0 + e1 * b.y + e2 * b.z, vec3(0.05), vec3(0.95));
                    sumCol += emitCol * w;
                    sumPos += sPos * w;
                    sumW   += w;
                    bestStr = max(bestStr, emitStr);
                }
            }
            primIdx = nextIdx;
        }
        vec3 bestCol = (sumW > 1e-5) ? (sumCol / sumW) : vec3(1.0, 0.55, 0.11);
        vec3 bestPos = (sumW > 1e-5) ? (sumPos / sumW) : (vec3(v) + 0.5);
        lightList.lights[idx].posAndPower  = vec4(bestPos, bestStr);
        lightList.lights[idx].colorAndMeta = vec4(bestCol, 0.18);
        if (bestStr > 1e-4) {
            ivec3 lc = clamp(ivec3(floor(bestPos * (1.0 / 32.0))), ivec3(0), ivec3(LIGHT_CELL_XZ - 1, LIGHT_CELL_Y - 1, LIGHT_CELL_XZ - 1));
            uint cIdx = uint(lc.x + lc.z * LIGHT_CELL_XZ + lc.y * (LIGHT_CELL_XZ * LIGHT_CELL_XZ));
            uint cSlot = atomicAdd(lightList.cellCount[cIdx], 1u);
            if (cSlot < LIGHT_CELL_CAP) {
                lightList.cellLights[cIdx * LIGHT_CELL_CAP + cSlot] = idx;
            }
        }
    }
}
