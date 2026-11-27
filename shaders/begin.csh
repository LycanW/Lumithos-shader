#version 430 compatibility

// Per-frame hierarchical sparse clear:
// Uses last frame's L1-resident superOccTex (600B) + L2-resident brickOccTex (38KB) so ~95% of
// empty air/underground workgroups exit in 1 cached texture read without touching voxelGridImg!
#define STAGE_VOXEL_GRID_WRITE
#include "/lib/Resources.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
const ivec3 workGroups = ivec3(20, 12, 20); // exact 160 x 96 x 160

void main() {
    ivec3 v = ivec3(gl_GlobalInvocationID.xyz);
    if (!inVoxelBounds(v)) return;
    uint slot = voxelLinearIndex(v);

    if (v == ivec3(0)) {
        primPool.primCount = 0u;
        lightList.lightCount = 0u;
        #ifdef ENTITY_REFLECTIONS_ENABLED
            primPool.entityPrimCount = 0u;
            entityPool.texelCount = 0u;
        #endif
    }
    #ifdef ENTITY_REFLECTIONS_ENABLED
        if (slot < MAX_ENTITY_PRIMITIVES) {
            uint occupied = entityPool.occupiedSlots[slot];
            if (occupied != TOTAL_VOXELS) {
                entityPool.heads[occupied] = PRIM_END;
                entityPool.occupiedSlots[slot] = TOTAL_VOXELS;
            }
        }
        if (slot < ENTITY_BRICK_WORDS) entityPool.brickBits[slot] = 0u;
        if (slot < ENTITY_SUPER_WORDS) entityPool.superBits[slot] = 0u;
    #endif
    if (v.y == 0 && v.z == 0 && v.x < int(LIGHT_CELL_TOTAL)) {
        lightList.cellCount[v.x] = 0u;
    }

    // 100% uniform across the entire 8x8x8 workgroup (half of a 16x16x16 super-brick)!
    if (texelFetch(superOccTex, v >> 4, 0).r == 0u) return;
    // 4x4x4 brick early-out before any uncached voxelGridImg load.
    if (texelFetch(brickOccTex, v >> 2, 0).r == 0u) return;

    if (imageLoad(voxelGridImg, v).r != 0u) {
        imageStore(voxelGridImg, v, uvec4(0u));
        for (int f = 0; f < 6; ++f) surfacePool.surfaces[slot].face[f] = 0u;
        surfacePool.surfaces[slot].tintAndSky = 0u;
        surfacePool.surfaces[slot].primHead   = PRIM_END;
    }
}
