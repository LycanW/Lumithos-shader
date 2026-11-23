#version 430 compatibility

// Pass: Clear current frame's active-set dedup bitmask & worklist counter (12,288 uints = ~0.005 ms!).
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(192, 1, 1); // 192 * 64 = 12,288 exact RC_DEDUP_WORDS

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= RC_DEDUP_WORDS) return;

    uint currSide = uint(frameCounter) & 1u;
    rcWork.data[currSide * RC_DEDUP_WORDS + idx] = 0u;
    if (idx == 0u) {
        rcWork.count[currSide] = 0u;
    }
}
