#version 430 compatibility

// Shadow rendering and voxelization. Full block faces use Track 1 descriptors.
// Irregular blocks and entities register their triangles in the cells they overlap.
#define STAGE_VOXEL_GRID_WRITE
#include "/lib/SceneGeo.glsl"

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform int renderStage;
uniform sampler2D gtexture;

in vec3  vVoxelPos[3];
in vec2  vTexCoord[3];
in vec4  vColor[3];
in float vSkyLight[3];
in float vEmissive[3];
in vec4  vShadowClip[3];
flat in int vIsWater[3];

out vec2 gTexCoord;
out vec4 gColor;
flat out int gNeedTex;
flat out int gIsWater;

void writeTintAndSky(uint slot, vec3 avgColor, float avgSky, bool isTinted, uint emissive4) {
    uint packedTint = packTintAndSky(isTinted ? avgColor : vec3(1.0), avgSky);
    if (isTinted && emissive4 == 0u) {
        surfacePool.surfaces[slot].tintAndSky = packedTint;
    } else {
        atomicMax(surfacePool.surfaces[slot].tintAndSky, packedTint);
    }
}

void registerEmissiveVoxel(uint slot, ivec3 vCoord, uint emissive4) {
    if (emissive4 == 0u) return;
    // Bit 28 of tintAndSky deduplicates 1 LightList entry per emitting voxel (100% race-free!).
    if ((atomicOr(surfacePool.surfaces[slot].tintAndSky, 0x10000000u) & 0x10000000u) == 0u) {
        uint lIdx = atomicAdd(lightList.lightCount, 1u);
        if (lIdx < MAX_SCENE_LIGHTS) {
            lightList.lights[lIdx].posAndPower  = vec4(vec3(vCoord) + 0.5, float(emissive4) * (1.0 / 15.0));
            lightList.lights[lIdx].colorAndMeta = vec4(1.0, 1.0, 1.0, 0.5);
        }
    }
}

#ifdef ENTITY_REFLECTIONS_ENABLED
// Save the texture bound for this draw before the shadow pass moves on to another model.
bool captureEntityTexture(out vec2 uv[3], out uint texturePatch) {
    ivec2 sourceSize = textureSize(gtexture, 0);
    if (any(lessThanEqual(sourceSize, ivec2(0)))) return false;

    vec2 minPx = min(vTexCoord[0], min(vTexCoord[1], vTexCoord[2])) * vec2(sourceSize);
    vec2 maxPx = max(vTexCoord[0], max(vTexCoord[1], vTexCoord[2])) * vec2(sourceSize);
    ivec2 lo = clamp(ivec2(floor(minPx)), ivec2(0), sourceSize - 1);
    ivec2 hi = clamp(ivec2(ceil(maxPx)) - 1, lo, sourceSize - 1);
    ivec2 span = hi - lo + 1;
    ivec2 size = min(span, ivec2(64));
    uint count = uint(size.x * size.y);
    uint offset = atomicAdd(entityPool.texelCount, count);
    if (offset > ENTITY_TEXEL_CAPACITY - count) return false;

    float alpha = (vColor[0].a + vColor[1].a + vColor[2].a) * (1.0 / 3.0);
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            ivec2 texel = lo + ((2 * ivec2(x, y) + 1) * span) / (2 * size);
            vec4 rgba = texelFetch(gtexture, texel, 0);
            rgba.a *= alpha;
            entityPool.texels[offset + uint(x + y * size.x)] = packUnorm4x8(rgba);
        }
    }
    for (int i = 0; i < 3; ++i) uv[i] = (vTexCoord[i] * vec2(sourceSize) - vec2(lo)) / vec2(span);
    // 20-bit texel offset followed by the two 6-bit extents minus one.
    texturePatch = offset | (uint(size.x - 1) << 20u) | (uint(size.y - 1) << 26u);
    return true;
}

bool registerEntityTriangle(ivec3 cell, uint slot, vec3 local0, vec3 e1, vec3 e2,
                            vec2 uv[3], vec3 tint, uint sky, uint texturePatch) {
    uint localIndex = atomicAdd(primPool.entityPrimCount, 1u);
    if (localIndex >= MAX_ENTITY_PRIMITIVES) return false;

    uint index = MAX_PRIMITIVES + localIndex;
    uint next = atomicExchange(entityPool.heads[slot], index);
    ScenePrim prim = packScenePrim(local0, e1, e2, uv[0], uv[1], uv[2], tint, sky, false, false, next);
    prim.meta.w = texturePatch;
    primPool.prims[index] = prim;

    // Record each occupied cell once for both traversal and next frame's clear.
    if (next == PRIM_END) {
        entityPool.occupiedSlots[localIndex] = slot;
        uint brick = entityNodeIndex(cell >> 2, BRICK_RES_XZ);
        uint super = entityNodeIndex(cell >> 4, SUPER_RES_XZ);
        atomicOr(entityPool.brickBits[brick >> 5u], 1u << (brick & 31u));
        atomicOr(entityPool.superBits[super >> 5u], 1u << (super & 31u));
    }
    return true;
}
#endif

void voxelizeTriangle(bool isWater, bool isEntity) {
    vec3 vp0 = vVoxelPos[0];
    vec3 vp1 = vVoxelPos[1];
    vec3 vp2 = vVoxelPos[2];

    // Fast AABB early-out for triangles strictly outside the 160x96x160 voxel grid (80..128 blocks away):
    // skips cross/normalize/log2/atomics on ~60% of shadow-pass geometry!
    vec3 rawMin = min(vp0, min(vp1, vp2));
    vec3 rawMax = max(vp0, max(vp1, vp2));
    if (any(lessThan(rawMax, vec3(0.0))) || any(greaterThanEqual(rawMin, vec3(VOXEL_RES_XZ, VOXEL_RES_Y, VOXEL_RES_XZ)))) {
        return;
    }

    vec3 e1 = vp1 - vp0;
    vec3 e2 = vp2 - vp0;
    vec3 rawNormal = cross(e1, e2);
    float area2 = length(rawNormal);
    if (area2 < 1e-5) return;
    vec3 faceNormal = rawNormal / area2;

    bool isTranslucent = (renderStage == MC_RENDER_STAGE_TERRAIN_TRANSLUCENT);
    uint emissive4 = uint(clamp(round(max(vEmissive[0], max(vEmissive[1], vEmissive[2]))), 0.0, 15.0));
    vec3 avgColor = (vColor[0].rgb + vColor[1].rgb + vColor[2].rgb) * (1.0 / 3.0);
    float avgSky  = (vSkyLight[0] + vSkyLight[1] + vSkyLight[2]) * (1.0 / 3.0);
    bool isTinted = minOf(avgColor) < 0.99;

    // ---- Classify: Track 1 (Axis-Aligned Full-Cube Outer Face) vs Track 2 (Non-Cube Primitive) ----
    vec3 centroidPos = (vp0 + vp1 + vp2) * (1.0 / 3.0);
    ivec3 vCoord = ivec3(floor(centroidPos - faceNormal * 0.004));
    bool isTrack1 = false;
    uint faceIdx  = normalToFaceIndex(faceNormal);
    vec3 localSum = vec3(0.0);

    // Keep water on Track 2 even for flat faces so its material tag and biome tint are per triangle.
    if (!isWater && !isEntity && inVoxelBounds(vCoord) && maxOf(abs(faceNormal)) > 0.999) {
        vec3 local0 = vp0 - vec3(vCoord);
        vec3 local1 = vp1 - vec3(vCoord);
        vec3 local2 = vp2 - vec3(vCoord);

        vec3 nAbs = abs(faceNormal);
        float expectedBoundary = dot(faceNormal, vec3(1.0)) > 0.0 ? 1.0 : 0.0;
        float d0 = abs(dot(local0, nAbs) - expectedBoundary);
        float d1 = abs(dot(local1, nAbs) - expectedBoundary);
        float d2 = abs(dot(local2, nAbs) - expectedBoundary);

        vec3 bMin = min(local0, min(local1, local2));
        vec3 bMax = max(local0, max(local1, local2));
        vec3 span = bMax - bMin;

        bool onPlane = max(d0, max(d1, d2)) < 0.005;
        bool fullSpan = all(lessThan(abs((span + nAbs) - vec3(1.0)), vec3(0.006)))
                     && all(greaterThanEqual(bMin, vec3(-0.005)))
                     && all(lessThanEqual(bMax, vec3(1.005)));
        isTrack1 = onPlane && fullSpan;
        localSum = local0 + local1 + local2;
    }

    if (isTrack1) {
        vec3 T, B;
        faceBasis(faceIdx, T, B);
        // Track-1 Quad Deduplication: every full-cube face splits into 2 triangles on [0,1]^2;
        // pick strictly 1 of the 2 triangles -> 50% fewer imageStore + contended same-address atomics!
        float sSum = dot(localSum, abs(T));
        float tSum = dot(localSum, abs(B));
        float stSum = sSum + tSum;
        if (stSum > 3.5 || (stSum >= 2.5 && sSum < tSum)) {
            return;
        }

        float ds1 = dot(e1, T), dt1 = dot(e1, B);
        float ds2 = dot(e2, T), dt2 = dot(e2, B);
        float det = ds1 * dt2 - dt1 * ds2;

        if (abs(det) > 1e-4) {
            uint slot = voxelLinearIndex(vCoord);
            imageStore(voxelGridImg, vCoord, uvec4(1u));
            writeTintAndSky(slot, avgColor, avgSky, isTinted, emissive4);
            registerEmissiveVoxel(slot, vCoord, emissive4);

            vec2 duv1 = vTexCoord[1] - vTexCoord[0];
            vec2 duv2 = vTexCoord[2] - vTexCoord[0];
            float invDet = 1.0 / det;
            vec2 dUVds = (duv1 * dt2 - duv2 * dt1) * invDet;
            vec2 dUVdt = (duv2 * ds1 - duv1 * ds2) * invDet;

            vec2 atlasRes = max(vec2(textureSize(gtexture, 0)), vec2(1.0));
            float tilePix = max(max(abs(dUVds.x), abs(dUVdt.x)) * atlasRes.x,
                                max(abs(dUVds.y), abs(dUVdt.y)) * atlasRes.y);
            uint sizeExp = uint(clamp(round(log2(max(tilePix, 1.0))), 0.0, 15.0));

            vec2 minUV = min(vTexCoord[0], min(vTexCoord[1], vTexCoord[2]));
            uvec2 tileCoord = uvec2(clamp(round(minUV * 512.0), vec2(0.0), vec2(511.0)));

            bool swapUV = abs(dUVds.y) > abs(dUVds.x);
            bool flipU  = swapUV ? (dUVdt.x < 0.0) : (dUVds.x < 0.0);
            bool flipV  = swapUV ? (dUVds.y < 0.0) : (dUVdt.y < 0.0);
            uint uvXform = (uint(swapUV) << 2u) | (uint(flipU) << 1u) | uint(flipV);

            uint desc = packFaceDesc(tileCoord, sizeExp, uvXform, isTinted, isTranslucent, emissive4);
            atomicMax(surfacePool.surfaces[slot].face[faceIdx], desc);
            return;
        }
    }

    // ---- Track 2: Non-Cube / Cross-Voxel Primitives (Exact AABB Registration) ----
    // Inset AABB by 1e-4 so single-voxel quads touching [0, 1] (like ladders!) register ONLY
    // in their true owning voxel instead of spilling into 4 neighbors (4x less primPool load!),
    // while genuinely cross-voxel geometry still registers in every voxel it overlaps.
    // Entity surfaces may lie exactly on a cell boundary and have no owning block.
    vec3 triMin = isEntity ? rawMin - vec3(1e-4) : rawMin - faceNormal * 0.002 + vec3(1e-4);
    vec3 triMax = isEntity ? rawMax + vec3(1e-4) : rawMax - faceNormal * 0.002 - vec3(1e-4);
    triMax = max(triMax, triMin);
    ivec3 gridMax = ivec3(VOXEL_RES_XZ - 1, VOXEL_RES_Y - 1, VOXEL_RES_XZ - 1);
    ivec3 cellMin = max(ivec3(floor(triMin)), ivec3(0));
    ivec3 cellMax = min(ivec3(floor(triMax)), gridMax);
    if (any(greaterThan(cellMin, cellMax))) return;

    if (!isEntity) cellMax = min(cellMax, cellMin + ivec3(1));
    float planeLimit = dot(abs(faceNormal), vec3(0.5)) + 0.01;
    if (maxOf(abs(e1)) > 1.99 || maxOf(abs(e2)) > 1.99) return;

    #ifdef ENTITY_REFLECTIONS_ENABLED
        vec2 entityUV[3];
        uint texturePatch = 0u;
        uint entitySky = 0u;
        if (isEntity) {
            if (!captureEntityTexture(entityUV, texturePatch)) return;
            entitySky = uint(round(saturate(avgSky) * 15.0));
        }
    #endif

    for (int vz = cellMin.z; vz <= cellMax.z; ++vz) {
        for (int vy = cellMin.y; vy <= cellMax.y; ++vy) {
            for (int vx = cellMin.x; vx <= cellMax.x; ++vx) {
                ivec3 c = ivec3(vx, vy, vz);
                vec3 voxelCenter = vec3(c) + 0.5;
                if (abs(dot(voxelCenter - vp0, faceNormal)) > planeLimit) continue;

                vec3 local0 = vp0 - vec3(c);
                if (maxOf(abs(local0)) > 1.99) continue;

                uint slot = voxelLinearIndex(c);
                #ifdef ENTITY_REFLECTIONS_ENABLED
                    if (isEntity) {
                        if (!registerEntityTriangle(c, slot, local0, e1, e2, entityUV, avgColor, entitySky, texturePatch)) return;
                        continue;
                    }
                #endif
                imageStore(voxelGridImg, c, uvec4(1u));
                if (!isWater) {
                    writeTintAndSky(slot, avgColor, avgSky, isTinted, emissive4);
                    registerEmissiveVoxel(slot, c, emissive4);
                }

                uint pIdx = atomicAdd(primPool.primCount, 1u);
                if (pIdx >= MAX_PRIMITIVES) return;

                uint prevHead = atomicExchange(surfacePool.surfaces[slot].primHead, pIdx);
                primPool.prims[pIdx] = packScenePrim(
                    local0, e1, e2,
                    vTexCoord[0], vTexCoord[1], vTexCoord[2],
                    avgColor, emissive4, isTranslucent, isWater, prevHead
                );
            }
        }
    }
}

void main() {
    bool isTerrain = (renderStage == MC_RENDER_STAGE_TERRAIN_SOLID)
                  || (renderStage == MC_RENDER_STAGE_TERRAIN_CUTOUT)
                  || (renderStage == MC_RENDER_STAGE_TERRAIN_CUTOUT_MIPPED)
                  || (renderStage == MC_RENDER_STAGE_TERRAIN_TRANSLUCENT);
    bool isWater = renderStage == MC_RENDER_STAGE_TERRAIN_TRANSLUCENT && vIsWater[0] != 0;
    if (isTerrain) {
        voxelizeTriangle(isWater, false);
    }
    #ifdef ENTITY_REFLECTIONS_ENABLED
        else if (renderStage == MC_RENDER_STAGE_ENTITIES) {
            voxelizeTriangle(false, true);
        }
    #endif

    vec3 cMin = min(vShadowClip[0].xyz, min(vShadowClip[1].xyz, vShadowClip[2].xyz));
    vec3 cMax = max(vShadowClip[0].xyz, max(vShadowClip[1].xyz, vShadowClip[2].xyz));
    if (cMax.x < -1.05 || cMin.x > 1.05 || cMax.y < -1.05 || cMin.y > 1.05) {
        return;
    }

    // Keep both front-facing and back-facing geometry in the shadow map (matching Iris's
    // double-sided shadow pass) so 1-block-thick walls, ceilings, stairs, slabs, and
    // corners are 100% airtight with zero light leaks.
    int needTex = (renderStage != MC_RENDER_STAGE_TERRAIN_SOLID) ? 1 : 0;
    for (int i = 0; i < 3; ++i) {
        gl_Position = vShadowClip[i];
        gTexCoord   = vTexCoord[i];
        gColor      = vColor[i];
        gNeedTex    = needTex;
        gIsWater    = int(isWater);
        EmitVertex();
    }
    EndPrimitive();
}
