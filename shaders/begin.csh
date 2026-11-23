#version 430 compatibility

// Per-frame sparse clear: only resets SurfaceVoxel entries that were actually occupied
// last frame (~1-3% of the grid -> <0.03 ms on GPU!), plus resets Track 2 primCount.
#define STAGE_VOXEL_GRID_WRITE
#include "/lib/Resources.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
const ivec3 workGroups = ivec3(20, 12, 20); // exact 160 x 96 x 160

void main() {
    ivec3 v = ivec3(gl_GlobalInvocationID.xyz);
    if (!inVoxelBounds(v)) return;

    if (v == ivec3(0)) {
        primPool.primCount = 0u;
    }

    if (imageLoad(voxelGridImg, v).r != 0u) {
        imageStore(voxelGridImg, v, uvec4(0u));
        uint slot = voxelLinearIndex(v);
        for (int f = 0; f < 6; ++f) surfacePool.surfaces[slot].face[f] = 0u;
        surfacePool.surfaces[slot].tintAndSky = 0u;
        surfacePool.surfaces[slot].primHead   = PRIM_END;
    }
}
