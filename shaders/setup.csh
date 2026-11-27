#version 430 compatibility

// One-time initialization of direct-addressed SurfaceVoxel pool, voxelGridImg,
// O(1) 3-Cascade Radiance Cache clipmap, and worklist buffers.
#define STAGE_VOXEL_GRID_WRITE
#define STAGE_BRICK_WRITE
#define STAGE_SUPER_WRITE
#include "/lib/Resources.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
const ivec3 workGroups = ivec3(20, 12, 20); // exact 160 x 96 x 160 = 2,457,600 threads

void main() {
    ivec3 v = ivec3(gl_GlobalInvocationID.xyz);
    if (!inVoxelBounds(v)) return;

    imageStore(voxelGridImg, v, uvec4(0u));
    if (all(lessThan(v, ivec3(BRICK_RES_XZ, BRICK_RES_Y, BRICK_RES_XZ)))) {
        imageStore(brickOccImg, v, uvec4(0u));
    }
    if (all(lessThan(v, ivec3(SUPER_RES_XZ, SUPER_RES_Y, SUPER_RES_XZ)))) {
        imageStore(superOccImg, v, uvec4(0u));
    }
    uint slot = voxelLinearIndex(v);
    for (int f = 0; f < 6; ++f) surfacePool.surfaces[slot].face[f] = 0u;
    surfacePool.surfaces[slot].tintAndSky = 0u;
    surfacePool.surfaces[slot].primHead   = PRIM_END;

    #ifdef ENTITY_REFLECTIONS_ENABLED
        entityPool.heads[slot] = PRIM_END;
        if (slot < MAX_ENTITY_PRIMITIVES) entityPool.occupiedSlots[slot] = TOTAL_VOXELS;
        if (slot < ENTITY_BRICK_WORDS) entityPool.brickBits[slot] = 0u;
        if (slot < ENTITY_SUPER_WORDS) entityPool.superBits[slot] = 0u;
    #endif

    if (slot < SC_TOTAL_SLOTS) {
        scTable.entries[slot] = SurfaceCacheEntry(0u, 0u, uvec2(0u));
    }
    if (slot == 0u) {
        primPool.primCount = 0u;
        lightList.lightCount = 0u;
        #ifdef ENTITY_REFLECTIONS_ENABLED
            primPool.entityPrimCount = 0u;
            entityPool.texelCount = 0u;
        #endif
    }
}
