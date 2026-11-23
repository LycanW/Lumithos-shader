#ifndef LIB_RAYTRACE_GLSL
#define LIB_RAYTRACE_GLSL

#include "/lib/SceneGeo.glsl"

struct TraceHit {
    bool  hit;
    bool  exitedGrid;
    bool  fromTrack2;
    float dist;
    vec3  worldPos;
    vec3  normal;
    vec3  albedo;
    float roughness;
    float f0;
    float emission;
    float emitFluxScale;
    float skyLight;
};

TraceHit makeMissHit() {
    TraceHit h;
    h.hit = false;
    h.exitedGrid = false;
    h.fromTrack2 = false;
    h.dist = 1e6;
    h.worldPos = vec3(0.0);
    h.normal = vec3(0.0, 1.0, 0.0);
    h.albedo = vec3(0.0);
    h.roughness = 1.0;
    h.f0 = 0.04;
    h.emission = 0.0;
    h.emitFluxScale = 1.0;
    h.skyLight = 1.0;
    return h;
}

// Hybrid 64-Way Hierarchical DDA (16x16x16 -> 4x4x4 -> 1x1x1)
// Track 1: O(1) Full-Cube 6-Face Descriptor with exact 8-way UV rotation/mirroring
// Track 2: Exact Triangle Intersection for Non-Cube / Custom / Mod Geometry
TraceHit traceScene(vec3 worldOrigin, vec3 rayDir, int maxSteps, float atlasLod) {
    TraceHit result = makeMissHit();

    vec3 vOrigin = worldToVoxelPos(worldOrigin);
    vec3 safeDir = vec3(
        abs(rayDir.x) < 1e-6 ? (rayDir.x < 0.0 ? -1e-6 : 1e-6) : rayDir.x,
        abs(rayDir.y) < 1e-6 ? (rayDir.y < 0.0 ? -1e-6 : 1e-6) : rayDir.y,
        abs(rayDir.z) < 1e-6 ? (rayDir.z < 0.0 ? -1e-6 : 1e-6) : rayDir.z
    );
    vec3 invDir = 1.0 / safeDir;
    ivec3 stepSign = ivec3(greaterThan(safeDir, vec3(0.0))) * 2 - 1;
    ivec3 stepPos  = ivec3(greaterThan(safeDir, vec3(0.0)));
    ivec2 atlasTexSize = max(textureSize(colortex14, 0), ivec2(1));
    ivec2 specTexSize  = max(textureSize(colortex15, 0), ivec2(1));
    vec2  invAtlasSize = 1.0 / vec2(atlasTexSize);

    // Entry face index for each axis when stepping into a voxel:
    // +X step enters -X face (1), -X step enters +X face (0), etc.
    ivec3 enterFace = ivec3(
        stepSign.x > 0 ? 1 : 0,
        stepSign.y > 0 ? 3 : 2,
        stepSign.z > 0 ? 5 : 4
    );

    float tCur = 0.0;
    ivec3 vCoord = ivec3(floor(vOrigin));
    int lastAxis = -1;

    // If starting outside grid, intersect grid AABB first.
    if (!inVoxelBounds(vCoord)) {
        vec3 boxMax = vec3(VOXEL_RES_XZ, VOXEL_RES_Y, VOXEL_RES_XZ);
        vec3 t0 = (vec3(0.0) - vOrigin) * invDir;
        vec3 t1 = (boxMax - vOrigin) * invDir;
        vec3 tMinV = min(t0, t1);
        vec3 tMaxV = max(t0, t1);
        float tEnter = maxOf(tMinV);
        float tExit  = minOf(tMaxV);
        if (tExit < max(tEnter, 0.0)) {
            result.exitedGrid = true;
            return result;
        }
        tCur = max(tEnter, 0.0) + 1e-4;
        vCoord = ivec3(floor(vOrigin + safeDir * tCur));
        if (!inVoxelBounds(vCoord)) {
            result.exitedGrid = true;
            return result;
        }
        lastAxis = (tMinV.x >= tMinV.y && tMinV.x >= tMinV.z) ? 0 : (tMinV.y >= tMinV.z ? 1 : 2);
    }

    for (int stepIdx = 0; stepIdx < maxSteps; ++stepIdx) {
        if (!inVoxelBounds(vCoord)) {
            result.exitedGrid = true;
            break;
        }

        // ---- Level 2: 16x16x16 Super-Brick Skip (1.1 KB, L1 resident) ----
        int shift = 0;
        if (texelFetch(superOccTex, vCoord >> 4, 0).r == 0u) {
            shift = 4;
        }
        // ---- Level 1: 4x4x4 Brick Skip (73 KB, L2 resident) ----
        else if (texelFetch(brickOccTex, vCoord >> 2, 0).r == 0u) {
            shift = 2;
        }
        // ---- Level 0: Exact Voxel Cell ----
        else {
            if (texelFetch(voxelGridTex, vCoord, 0).r != 0u) {
                uint slot = voxelLinearIndex(vCoord);
                SurfaceVoxel sv = surfacePool.surfaces[slot];

                // Compute voxel exit distance to bound primitive tests cleanly.
                vec3 voxelExitPlane = (vec3(vCoord + stepPos) - vOrigin) * invDir;
                float tVoxelExit = minOf(voxelExitPlane);
                float bestT = tVoxelExit + 1e-3;
                bool foundHit = false;

                // --- Track 1: O(1) Full-Cube 6-Face Fast Path ---
                if (lastAxis >= 0) {
                    uint faceIdx = uint(enterFace[lastAxis]);
                    uint faceDesc = sv.face[faceIdx];
                    // If this pure full-cube voxel (primHead == PRIM_END) had its entered face
                    // culled by Sodium shadow backface culling or internal grazing entry, fall back
                    // to the opposite face or any populated face so rays NEVER leak through solid walls!
                    if (faceDesc == 0u && sv.primHead == PRIM_END) {
                        faceDesc = sv.face[faceIdx ^ 1u];
                        if (faceDesc == 0u) {
                            for (int k = 0; k < 6; ++k) {
                                if (sv.face[k] != 0u) { faceDesc = sv.face[k]; break; }
                            }
                        }
                    }
                    if (faceDesc != 0u) {
                        vec3 localHit = (vOrigin + safeDir * tCur) - vec3(vCoord);
                        bool isTinted, isTranslucent;
                        float blockEmit;
                        vec2 atlasUV = decodeFaceAtlasUV(faceDesc, faceIdx, localHit, invAtlasSize, isTinted, isTranslucent, blockEmit);
                        vec4 tex = textureLod(colortex14, atlasUV, atlasLod);
                        if (tex.a >= 0.1) {
                            vec3 tintSrgb = isTinted ? unpackTint(sv.tintAndSky) : vec3(1.0);
                            ivec2 spx = clamp(ivec2(atlasUV * vec2(specTexSize)), ivec2(0), specTexSize - 1);
                            vec4 specTex = texelFetch(colortex15, spx, 0);
                            float rough = 0.85, f0 = 0.04, pbrEmit = 0.0;
                            decodeLabPBR(specTex, rough, f0, pbrEmit);

                            bestT = tCur;
                            foundHit = true;
                            result.hit = true;
                            result.fromTrack2 = false;
                            result.dist = tCur;
                            result.worldPos = voxelToWorldPos(vOrigin + safeDir * tCur);
                            result.normal = faceIndexToNormal(faceIdx);
                            result.albedo = srgbToLinear(tex.rgb * tintSrgb);
                            result.roughness = rough;
                            result.f0 = f0;
                            result.emission = max(blockEmit, pbrEmit);
                            result.emitFluxScale = 1.0;
                            result.skyLight = unpackSky(sv.tintAndSky);
                        }
                    }
                }

                // --- Track 2: Exact Non-Cube Primitives (Stairs, Slabs, Torches, Mod Models) ---
                uint primIdx = sv.primHead;
                if (primIdx != PRIM_END) {
                    vec3 localOrig = vOrigin - vec3(vCoord);
                    for (int guard = 0; guard < 48 && primIdx != PRIM_END; ++guard) {
                        vec3 p0, e1, e2, tintSrgb;
                        vec2 uv0, uv1, uv2;
                        float primEmit;
                        bool primTrans;
                        uint nextIdx;
                        unpackScenePrim(primPool.prims[primIdx], p0, e1, e2, uv0, uv1, uv2, tintSrgb, primEmit, primTrans, nextIdx);
                        primIdx = nextIdx;

                        vec3 pvec = cross(safeDir, e2);
                        float det = dot(e1, pvec);
                        if (abs(det) < 1e-6) continue;
                        float invDet = 1.0 / det;

                        vec3 tvec = localOrig - p0;
                        float u = dot(tvec, pvec) * invDet;
                        if (u < -1e-4 || u > 1.0001) continue;

                        vec3 qvec = cross(tvec, e1);
                        float v = dot(safeDir, qvec) * invDet;
                        if (v < -1e-4 || u + v > 1.0001) continue;

                        float t = dot(e2, qvec) * invDet;
                        if (t <= 1e-4 || t >= bestT) continue;

                        vec2 hitUV = uv0 * (1.0 - u - v) + uv1 * u + uv2 * v;
                        vec4 tex = textureLod(colortex14, hitUV, atlasLod);
                        if (tex.a < 0.1) continue;

                        vec3 rawCross = cross(e1, e2);
                        float triArea = 0.5 * length(rawCross);
                        vec3 triN = rawCross / max(2.0 * triArea, 1e-6);
                        if (dot(triN, safeDir) > 0.0) triN = -triN;

                        ivec2 spx = clamp(ivec2(hitUV * vec2(specTexSize)), ivec2(0), specTexSize - 1);
                        vec4 specTex = texelFetch(colortex15, spx, 0);
                        float rough = 0.85, f0 = 0.04, pbrEmit = 0.0;
                        decodeLabPBR(specTex, rough, f0, pbrEmit);

                        bestT = t;
                        foundHit = true;
                        result.hit = true;
                        result.fromTrack2 = true;
                        result.dist = t;
                        result.worldPos = voxelToWorldPos(vOrigin + safeDir * t);
                        result.normal = triN;
                        result.albedo = srgbToLinear(tex.rgb * tintSrgb);
                        result.roughness = rough;
                        result.f0 = f0;
                        result.emission = max(primEmit, pbrEmit);
                        // Small-emitter solid-angle flux compensation (torches, lanterns, candles, end rods)
                        // so a level-14 torch emits realistic ~12-14 block illumination instead of 1/64th of a full block!
                        result.emitFluxScale = clamp(0.25 / max(triArea, 0.006), 1.0, 14.0);
                        result.skyLight = unpackSky(sv.tintAndSky);
                    }
                }

                if (foundHit) return result;
            }
        }

        // Advance cleanly to next node boundary at hierarchy level `shift`.
        int nodeSize = 1 << shift;
        ivec3 nodeMin = (vCoord >> shift) << shift;
        ivec3 nodeExit = nodeMin + stepPos * nodeSize;
        vec3 tPlane = (vec3(nodeExit) - vOrigin) * invDir;

        float tNext;
        if (tPlane.x <= tPlane.y && tPlane.x <= tPlane.z) {
            tNext = tPlane.x;
            lastAxis = 0;
        } else if (tPlane.y <= tPlane.z) {
            tNext = tPlane.y;
            lastAxis = 1;
        } else {
            tNext = tPlane.z;
            lastAxis = 2;
        }

        tCur = max(tCur, tNext);
        vec3 nextPos = vOrigin + safeDir * tCur;
        // Clamp non-stepping axes inside the exited node [nodeMin, nodeMin + nodeSize - 1].
        // At shift == 0 (1x1x1 voxel), this enforces strict 6-connected DDA stepping so rays
        // aiming at 90-degree concave corners NEVER jump diagonally past the wall/ceiling seam!
        ivec3 nextCoord = clamp(ivec3(floor(nextPos)), nodeMin, nodeMin + ivec3(nodeSize - 1));
        nextCoord[lastAxis] = nodeExit[lastAxis] + (stepSign[lastAxis] < 0 ? -1 : 0);
        vCoord = nextCoord;
    }

    if (!inVoxelBounds(vCoord)) {
        result.exitedGrid = true;
    } else {
        result.worldPos = voxelToWorldPos(vOrigin + safeDir * tCur);
        result.dist = tCur;
    }
    return result;
}

#endif
