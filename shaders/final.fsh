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

#include "/lib/Resources.glsl"

uniform sampler2D taaHistTex;

in vec2 vUV;
layout(location = 0) out vec4 fragColor;

// ACES Filmic Tonemap (Narkowicz 2015)
vec3 tonemapACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

void main() {
    ivec2 res = ivec2(screenSize());
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), res - 1);
    ivec2 currOff = ivec2((frameCounter & 1) * res.x, 0);
    vec3 hdr = max(texelFetch(taaHistTex, px + currOff, 0).rgb, vec3(0.0));
    #if DEBUG_VIEW == 5
        vec3 ldr = hdr;
    #else
        vec3 ldr = tonemapACES(hdr * float(EXPOSURE));
    #endif
    fragColor = vec4(linearToSrgb(ldr), 1.0);
}
