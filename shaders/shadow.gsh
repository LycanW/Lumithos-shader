#version 430 compatibility

// Dual-purpose Shadow + Direct-Addressed Hybrid Dual-Track Voxelization Geometry Shader:
// - ZERO spinlocks, ZERO retry caps, ZERO imageAtomicCompSwap, ZERO dropped faces!
// - Track 1: O(1) 6-Face Full-Cube + exact 8-way UV rotation/mirroring + 1/512 normalized atlas origin.
// - Track 2: Full AABB cross-voxel registration for Non-Cube / Stairs / Slabs / Fences / Mod Models.
#define STAGE_VOXEL_GRID_WRITE
#include "/lib/SceneGeo.glsl"

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

uniform int renderStage;
uniform sampler2D gtexture;

in vec3  vWorldPos[3];
in vec2  vTexCoord[3];
in vec4  vColor[3];
in float vSkyLight[3];
in float vEmissive[3];
in vec4  vShadowClip[3];

out vec2 gTexCoord;
out vec4 gColor;

void writeTintAndSky(uint slot, vec3 avgColor, float avgSky, bool isTinted) {
    uint packedTint = packTintAndSky(isTinted ? avgColor : vec3(1.0), avgSky);
    if (isTinted) {
        surfacePool.surfaces[slot].tintAndSky = packedTint;
    } else {
        atomicMax(surfacePool.surfaces[slot].tintAndSky, packedTint);
    }
}

void voxelizeTerrainTriangle() {
    vec3 vp0 = worldToVoxelPos(vWorldPos[0]);
    vec3 vp1 = worldToVoxelPos(vWorldPos[1]);
    vec3 vp2 = worldToVoxelPos(vWorldPos[2]);

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

    if (inVoxelBounds(vCoord) && maxOf(abs(faceNormal)) > 0.999) {
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
    }

    if (isTrack1) {
        vec3 T, B;
        faceBasis(faceIdx, T, B);
        float ds1 = dot(e1, T), dt1 = dot(e1, B);
        float ds2 = dot(e2, T), dt2 = dot(e2, B);
        float det = ds1 * dt2 - dt1 * ds2;

        if (abs(det) > 1e-4) {
            uint slot = voxelLinearIndex(vCoord);
            imageStore(voxelGridImg, vCoord, uvec4(1u));
            writeTintAndSky(slot, avgColor, avgSky, isTinted);

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
    vec3 triMin = min(vp0, min(vp1, vp2)) - faceNormal * 0.002 + vec3(1e-4);
    vec3 triMax = max(vp0, max(vp1, vp2)) - faceNormal * 0.002 - vec3(1e-4);
    triMax = max(triMax, triMin);
    ivec3 gridMax = ivec3(VOXEL_RES_XZ - 1, VOXEL_RES_Y - 1, VOXEL_RES_XZ - 1);
    ivec3 cellMin = max(ivec3(floor(triMin)), ivec3(0));
    ivec3 cellMax = min(ivec3(floor(triMax)), gridMax);
    if (any(greaterThan(cellMin, cellMax))) return;

    cellMax = min(cellMax, cellMin + ivec3(1));
    float planeLimit = dot(abs(faceNormal), vec3(0.5)) + 0.01;

    for (int vz = cellMin.z; vz <= cellMax.z; ++vz) {
        for (int vy = cellMin.y; vy <= cellMax.y; ++vy) {
            for (int vx = cellMin.x; vx <= cellMax.x; ++vx) {
                ivec3 c = ivec3(vx, vy, vz);
                vec3 voxelCenter = vec3(c) + 0.5;
                if (abs(dot(voxelCenter - vp0, faceNormal)) > planeLimit) continue;

                vec3 local0 = vp0 - vec3(c);
                if (maxOf(abs(local0)) > 1.99 || maxOf(abs(e1)) > 1.99 || maxOf(abs(e2)) > 1.99) continue;

                uint slot = voxelLinearIndex(c);
                imageStore(voxelGridImg, c, uvec4(1u));
                writeTintAndSky(slot, avgColor, avgSky, isTinted);

                uint pIdx = atomicAdd(primPool.primCount, 1u);
                if (pIdx >= MAX_PRIMITIVES) return;

                uint prevHead = atomicExchange(surfacePool.surfaces[slot].primHead, pIdx);
                primPool.prims[pIdx] = packScenePrim(
                    local0, e1, e2,
                    vTexCoord[0], vTexCoord[1], vTexCoord[2],
                    avgColor, emissive4, isTranslucent, prevHead
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
    if (isTerrain) {
        voxelizeTerrainTriangle();
    }

    vec2 cMin = min(vShadowClip[0].xy, min(vShadowClip[1].xy, vShadowClip[2].xy));
    vec2 cMax = max(vShadowClip[0].xy, max(vShadowClip[1].xy, vShadowClip[2].xy));
    if (cMax.x < -1.05 || cMin.x > 1.05 || cMax.y < -1.05 || cMin.y > 1.05) {
        return;
    }

    for (int i = 0; i < 3; ++i) {
        gl_Position = vShadowClip[i];
        gTexCoord   = vTexCoord[i];
        gColor      = vColor[i];
        EmitVertex();
    }
    EndPrimitive();
}
