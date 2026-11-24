#version 430 compatibility

// Single-workgroup pass:
// 1) Downsample Bloom L3 -> L4 (resolution-independent stride loop).
// 2) Center-weighted log2-luminance Auto Exposure with smooth asymmetric temporal eye adaptation.
#include "/lib/Tonemap.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(1, 1, 1);

uniform sampler2D scratchTex;
layout(rgba16f) uniform writeonly image2D scratchImg;
layout(rgba32f) uniform image2D exposureImg;

shared float sLogSum[256];
shared float sWSum[256];

void main() {
    ivec2 res = ivec2(screenSize());
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    uint tid = gl_LocalInvocationIndex;

    BloomRect l3 = getBloomRect(3, false, res);
    BloomRect l4 = getBloomRect(4, false, res);

    // 1) Downsample L3 -> L4 (tile 0)
    #ifdef BLOOM_ENABLED
        for (int y = lid.y; y < l4.size.y; y += 16) {
            for (int x = lid.x; x < l4.size.x; x += 16) {
                vec3 col = bloomDownsample4x4(scratchTex, ivec2(x, y), l3.origin, l3.size, false);
                imageStore(scratchImg, l4.origin + ivec2(x, y), roundHalf(vec4(col, 1.0)));
            }
        }
    #endif

    // 2) Center-weighted log-luminance reduction over L3 (prefiltered, firefly-free!)
    float localLog = 0.0;
    float localW   = 0.0;
    #ifdef AUTO_EXPOSURE
        vec2 invL3 = 1.0 / vec2(max(l3.size, ivec2(1)));
        for (int y = lid.y; y < l3.size.y; y += 16) {
            for (int x = lid.x; x < l3.size.x; x += 16) {
                vec3 c = max(texelFetch(scratchTex, l3.origin + ivec2(x, y), 0).rgb, vec3(0.0));
                if (any(isnan(c)) || any(isinf(c))) continue;
                float lum = clamp(luma(c), 0.015, 16.0);
                vec2 ndc = (vec2(x, y) + 0.5) * invL3 * 2.0 - 1.0;
                float w = exp(-dot(ndc, ndc) * 1.1);
                localLog += log2(lum) * w;
                localW   += w;
            }
        }
    #endif

    sLogSum[tid] = localLog;
    sWSum[tid]   = localW;
    barrier();

    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sLogSum[tid] += sLogSum[tid + s];
            sWSum[tid]   += sWSum[tid + s];
        }
        barrier();
    }

    if (tid == 0u) {
        #ifdef AUTO_EXPOSURE
            float avgLog = (sWSum[0] > 1e-5) ? (sLogSum[0] / sWSum[0]) : -1.5;
            float geoMean = exp2(avgLog);
            const float KEY_VALUE = 0.34;
            float targetExp = clamp(KEY_VALUE / max(geoMean, 1e-4), 0.45, 2.20);

            vec4 prev = imageLoad(exposureImg, ivec2(0));
            float prevExp = prev.g;
            float dt = clamp(frameTime, 0.001, 0.1);
            float adaptedExp;
            if (frameCounter <= 2 || prevExp <= 0.05 || isnan(prevExp) || isinf(prevExp)) {
                adaptedExp = targetExp;
            } else {
                // Fast bright adaptation (stepping into sunlight), gentle dark adaptation (entering caves).
                float tau = (targetExp < prevExp) ? 0.45 : 1.25;
                float k = 1.0 - exp(-dt / tau);
                adaptedExp = mix(prevExp, targetExp, k);
            }
            imageStore(exposureImg, ivec2(0), vec4(avgLog, adaptedExp, targetExp, 1.0));
        #else
            imageStore(exposureImg, ivec2(0), vec4(0.0, 1.0, 1.0, 1.0));
        #endif
    }
}
