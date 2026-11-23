#version 430 compatibility

// Pass: Publish TAA result to persistent history (taaHistImg) and final output (colorimg0)
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D taaHistImg;
layout(rgba16f) uniform writeonly image2D colorimg0;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    vec4 col = texelFetch(scratchTex, px, 0);
    imageStore(taaHistImg, px, col);
    imageStore(colorimg0,  px, col);
}
