#version 430 compatibility

/*
    const int colortex0Format = RGBA16F;
    const int colortex1Format = RG32UI;
    const int colortex2Format = R32UI;
    const int colortex3Format = RGBA16F;
    const int colortex4Format = RGBA16F;
    const int colortex5Format = RGBA16F;

    const bool colortex0Clear = true;
    const bool colortex1Clear = false;
    const bool colortex2Clear = false;
    const bool colortex3Clear = false;
    const bool colortex4Clear = false;
    const bool colortex5Clear = false;
*/

#include "/lib/Tonemap.glsl"

uniform sampler2D taaHistTex;
uniform sampler2D scratchTex;
uniform sampler2D exposureTex;

in vec2 vUV;
layout(location = 0) out vec4 fragColor;

vec3 sampleBloomBilinear(ivec2 px, ivec2 res) {
    BloomRect u0 = getBloomRect(0, true, res);
    vec2 uv = (vec2(px) + 0.5) / vec2(res);
    vec2 pos = uv * vec2(u0.size) - 0.5;
    ivec2 base = ivec2(floor(pos));
    vec2 f = pos - vec2(base);
    ivec2 maxP = u0.size - 1;

    vec3 c00 = max(texelFetch(scratchTex, u0.origin + clamp(base + ivec2(0, 0), ivec2(0), maxP), 0).rgb, vec3(0.0));
    vec3 c10 = max(texelFetch(scratchTex, u0.origin + clamp(base + ivec2(1, 0), ivec2(0), maxP), 0).rgb, vec3(0.0));
    vec3 c01 = max(texelFetch(scratchTex, u0.origin + clamp(base + ivec2(0, 1), ivec2(0), maxP), 0).rgb, vec3(0.0));
    vec3 c11 = max(texelFetch(scratchTex, u0.origin + clamp(base + ivec2(1, 1), ivec2(0), maxP), 0).rgb, vec3(0.0));
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

void main() {
    ivec2 res = ivec2(screenSize());
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), res - 1);
    ivec2 currOff = ivec2((frameCounter & 1) * res.x, 0);
    vec3 hdr = max(texelFetch(taaHistTex, px + currOff, 0).rgb, vec3(0.0));
    if (any(isnan(hdr)) || any(isinf(hdr))) hdr = vec3(0.0);

    #if DEBUG_VIEW == 5
        vec3 ldr = hdr;
    #elif DEBUG_VIEW != 0
        vec3 ldr = applyTonemap(hdr * float(EXPOSURE));
    #else
        #ifdef BLOOM_ENABLED
            vec3 bloom = sampleBloomBilinear(px, res);
            hdr = mix(hdr, bloom, float(BLOOM_STRENGTH));
        #endif

        float expMul = float(EXPOSURE);
        #ifdef AUTO_EXPOSURE
            float autoExp = texelFetch(exposureTex, ivec2(0), 0).g;
            if (autoExp > 0.05 && !isnan(autoExp) && !isinf(autoExp)) {
                expMul *= autoExp;
            }
        #endif

        vec3 ldr = applyTonemap(hdr * expMul);
    #endif

    fragColor = vec4(linearToSrgb(ldr), 1.0);
}
