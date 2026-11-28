#ifndef LIB_CAMERA_OUTPUT_GLSL
#define LIB_CAMERA_OUTPUT_GLSL

#include "/lib/Bloom.glsl"
#include "/lib/Camera.glsl"
#include "/lib/CameraWhiteBalance.glsl"
#include "/lib/Tonemap.glsl"

const uint CAMERA_VIEWFINDER_BINS = 64u;
const int CAMERA_VIEWFINDER_HISTOGRAM_ROW = 1;

vec3 sampleCameraHdr(sampler2D sceneTex, ivec2 px, ivec2 res) {
    ivec2 currOff = ivec2((frameCounter & 1) * res.x, 0);
    vec3 hdr = max(texelFetch(sceneTex, px + currOff, 0).rgb, vec3(0.0));
    return any(isnan(hdr)) || any(isinf(hdr)) ? vec3(0.0) : hdr;
}

vec3 sampleBloomBilinear(sampler2D bloomTex, ivec2 px, ivec2 res) {
    BloomRect u0 = getBloomRect(0, true, res);
    vec2 uv = (vec2(px) + 0.5) / vec2(res);
    vec2 pos = uv * vec2(u0.size) - 0.5;
    ivec2 base = ivec2(floor(pos));
    vec2 f = pos - vec2(base);
    ivec2 maxP = u0.size - 1;

    vec3 c00 = max(texelFetch(bloomTex, u0.origin + clamp(base + ivec2(0, 0), ivec2(0), maxP), 0).rgb, vec3(0.0));
    vec3 c10 = max(texelFetch(bloomTex, u0.origin + clamp(base + ivec2(1, 0), ivec2(0), maxP), 0).rgb, vec3(0.0));
    vec3 c01 = max(texelFetch(bloomTex, u0.origin + clamp(base + ivec2(0, 1), ivec2(0), maxP), 0).rgb, vec3(0.0));
    vec3 c11 = max(texelFetch(bloomTex, u0.origin + clamp(base + ivec2(1, 1), ivec2(0), maxP), 0).rgb, vec3(0.0));
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

vec3 cameraDisplayColor(vec3 hdr, sampler2D bloomTex, ivec2 px, ivec2 res, float exposure, vec3 whiteBalanceScale) {
    #ifdef BLOOM_ENABLED
        hdr = mix(hdr, sampleBloomBilinear(bloomTex, px, res), float(BLOOM_STRENGTH));
    #endif
    return linearToSrgb(applyTonemap(applyCameraWhiteBalance(hdr, whiteBalanceScale) * exposure));
}

#endif
