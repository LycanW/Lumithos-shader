#version 430 compatibility

// Pass: Build Level-2 16x16x16 super-brick occupancy (10 x 6 x 10) from brickOccTex.
#define STAGE_SUPER_WRITE
#include "/lib/Resources.glsl"

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
const ivec3 workGroups = ivec3(3, 2, 3); // covers 10 x 6 x 10 super-bricks

void main() {
    ivec3 sb = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(sb, ivec3(SUPER_RES_XZ, SUPER_RES_Y, SUPER_RES_XZ)))) return;

    ivec3 base = sb << 2;
    uint occ = 0u;
    for (int z = 0; z < 4 && occ == 0u; ++z) {
        for (int y = 0; y < 4 && occ == 0u; ++y) {
            for (int x = 0; x < 4; ++x) {
                if (texelFetch(brickOccTex, base + ivec3(x, y, z), 0).r != 0u) {
                    occ = 1u;
                    break;
                }
            }
        }
    }
    imageStore(superOccImg, sb, uvec4(occ));
}
