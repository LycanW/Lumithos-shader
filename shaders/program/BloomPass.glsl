#include "/lib/Bloom.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

#if BLOOM_STAGE == 0
    const vec2 workGroupsRender = vec2(0.5, 0.5);
    uniform sampler2D taaHistTex;
#elif BLOOM_STAGE == 1
    const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif BLOOM_STAGE == 2
    const vec2 workGroupsRender = vec2(0.125, 0.125);
#elif BLOOM_STAGE == 3
    const vec2 workGroupsRender = vec2(0.0625, 0.0625);
#elif BLOOM_STAGE == 6
    const vec2 workGroupsRender = vec2(0.0625, 0.0625);
#elif BLOOM_STAGE == 7
    const vec2 workGroupsRender = vec2(0.125, 0.125);
#elif BLOOM_STAGE == 8
    const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif BLOOM_STAGE == 9
    const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif

#if BLOOM_STAGE != 0
    uniform sampler2D scratchTex;
#endif
layout(rgba16f) uniform writeonly image2D scratchImg;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());

    #if BLOOM_STAGE == 0
        BloomRect dst = getBloomRect(0, false, res);
        if (any(greaterThanEqual(px, dst.size))) return;
        ivec2 taaOff = ivec2((frameCounter & 1) * res.x, 0);
        // Karis-weighted 4x4 B-spline prefilter eliminates single-pixel specular bloom fireflies!
        vec3 col = bloomDownsample4x4(taaHistTex, px, taaOff, res, true);
        imageStore(scratchImg, dst.origin + px, roundHalf(vec4(col, 1.0)));

    #elif BLOOM_STAGE == 1 || BLOOM_STAGE == 2 || BLOOM_STAGE == 3
        BloomRect src = getBloomRect(BLOOM_STAGE - 1, false, res);
        BloomRect dst = getBloomRect(BLOOM_STAGE,     false, res);
        if (any(greaterThanEqual(px, dst.size))) return;
        vec3 col = bloomDownsample4x4(scratchTex, px, src.origin, src.size, false);
        imageStore(scratchImg, dst.origin + px, roundHalf(vec4(col, 1.0)));

    #elif BLOOM_STAGE == 6
        // Upsample L4 (tile 0) + L3 (tile 0) -> U3 (tile 1)
        BloomRect coarse = getBloomRect(4, false, res);
        BloomRect fine   = getBloomRect(3, false, res);
        BloomRect dst    = getBloomRect(3, true,  res);
        if (any(greaterThanEqual(px, dst.size))) return;
        vec3 up   = bloomUpsample3x3(scratchTex, px, dst.size, coarse.origin, coarse.size);
        vec3 base = max(texelFetch(scratchTex, fine.origin + px, 0).rgb, vec3(0.0));
        imageStore(scratchImg, dst.origin + px, roundHalf(vec4(mix(base, up, 0.50), 1.0)));

    #elif BLOOM_STAGE == 7
        // Upsample U3 (tile 1) + L2 (tile 0) -> U2 (tile 1)
        BloomRect coarse = getBloomRect(3, true,  res);
        BloomRect fine   = getBloomRect(2, false, res);
        BloomRect dst    = getBloomRect(2, true,  res);
        if (any(greaterThanEqual(px, dst.size))) return;
        vec3 up   = bloomUpsample3x3(scratchTex, px, dst.size, coarse.origin, coarse.size);
        vec3 base = max(texelFetch(scratchTex, fine.origin + px, 0).rgb, vec3(0.0));
        imageStore(scratchImg, dst.origin + px, roundHalf(vec4(mix(base, up, 0.52), 1.0)));

    #elif BLOOM_STAGE == 8
        // Upsample U2 (tile 1) + L1 (tile 0) -> U1 (tile 1)
        BloomRect coarse = getBloomRect(2, true,  res);
        BloomRect fine   = getBloomRect(1, false, res);
        BloomRect dst    = getBloomRect(1, true,  res);
        if (any(greaterThanEqual(px, dst.size))) return;
        vec3 up   = bloomUpsample3x3(scratchTex, px, dst.size, coarse.origin, coarse.size);
        vec3 base = max(texelFetch(scratchTex, fine.origin + px, 0).rgb, vec3(0.0));
        imageStore(scratchImg, dst.origin + px, roundHalf(vec4(mix(base, up, 0.54), 1.0)));

    #elif BLOOM_STAGE == 9
        // Upsample U1 (tile 1) + L0 (tile 0) -> U0 (tile 1)
        BloomRect coarse = getBloomRect(1, true,  res);
        BloomRect fine   = getBloomRect(0, false, res);
        BloomRect dst    = getBloomRect(0, true,  res);
        if (any(greaterThanEqual(px, dst.size))) return;
        vec3 up   = bloomUpsample3x3(scratchTex, px, dst.size, coarse.origin, coarse.size);
        vec3 base = max(texelFetch(scratchTex, fine.origin + px, 0).rgb, vec3(0.0));
        imageStore(scratchImg, dst.origin + px, roundHalf(vec4(mix(base, up, 0.56), 1.0)));
    #endif
}
