#version 430 compatibility

// One-time initialization of direct-addressed SurfaceVoxel pool, voxelGridImg,
// O(1) 3-Cascade Radiance Cache clipmap, and worklist buffers.
#define STAGE_VOXEL_GRID_WRITE
#include "/lib/Resources.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
const ivec3 workGroups = ivec3(20, 12, 20); // exact 160 x 96 x 160 = 2,457,600 threads

void main() {
    ivec3 v = ivec3(gl_GlobalInvocationID.xyz);
    if (!inVoxelBounds(v)) return;

    imageStore(voxelGridImg, v, uvec4(0u));
    uint slot = voxelLinearIndex(v);
    for (int f = 0; f < 6; ++f) surfacePool.surfaces[slot].face[f] = 0u;
    surfacePool.surfaces[slot].tintAndSky = 0u;
    surfacePool.surfaces[slot].primHead   = PRIM_END;

    if (slot < RC_TOTAL_CELLS) {
        rcClipmap.cells[slot] = RCProbe(0u, 0u, uvec2(0u));
    }
    if (slot < 2u * RC_DEDUP_WORDS) {
        rcWork.data[slot] = 0u;
    }
    if (slot == 0u) {
        primPool.primCount = 0u;
        lightList.lightCount = 0u;
        rcWork.count[0] = 0u;
        rcWork.count[1] = 0u;
    }
}
