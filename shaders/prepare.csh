#version 430 compatibility

// Pass: Build Level-1 4x4x4 brick occupancy (40 x 24 x 40) from voxelGridTex.
#define STAGE_BRICK_WRITE
#include "/lib/Resources.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
const ivec3 workGroups = ivec3(10, 6, 10); // exact 40 x 24 x 40 bricks

void main() {
    ivec3 brick = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(brick, ivec3(BRICK_RES_XZ, BRICK_RES_Y, BRICK_RES_XZ)))) return;

    ivec3 base = brick << 2;
    uint occ = 0u;
    for (int z = 0; z < 4 && occ == 0u; ++z) {
        for (int y = 0; y < 4 && occ == 0u; ++y) {
            for (int x = 0; x < 4; ++x) {
                if (texelFetch(voxelGridTex, base + ivec3(x, y, z), 0).r != 0u) {
                    occ = 1u;
                    break;
                }
            }
        }
    }
    imageStore(brickOccImg, brick, uvec4(occ));
}
