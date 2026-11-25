#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#ifdef ATROUS_WRITE_COLORTEX3
    layout(rgba16f) uniform writeonly image2D colorimg3;
#elif defined(ATROUS_WRITE_COLORTEX4)
    layout(rgba16f) uniform writeonly image2D colorimg4;
#else
    layout(rgba16f) uniform writeonly image2D scratchImg;
#endif

#if ATROUS_PASS == 0 && !defined(ATROUS_SPECULAR)
    layout(rgba16f) uniform image2D diffHistImg;
#endif

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        #ifdef ATROUS_WRITE_COLORTEX3
            imageStore(colorimg3, px, vec4(0.0));
        #elif defined(ATROUS_WRITE_COLORTEX4)
            imageStore(colorimg4, px, vec4(0.0));
        #else
            imageStore(scratchImg, px + ivec2(ATROUS_DST_TILE * res.x, 0), vec4(0.0));
        #endif
        return;
    }

    float rough, f0Raw;
    bool isHand;
    readGBufferSpecMask(px, rough, f0Raw, isHand);
    #ifdef ATROUS_SPECULAR
        bool isSpec = true;
        if (!hasSpecularTrace(rough, f0Raw, isHand)) {
            #ifdef ATROUS_WRITE_COLORTEX4
                imageStore(colorimg4, px, vec4(0.0));
            #else
                imageStore(scratchImg, px + ivec2(ATROUS_DST_TILE * res.x, 0), vec4(0.0));
            #endif
            return;
        }
    #else
        bool isSpec = false;
    #endif

    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    ivec2 srcOff = ivec2(ATROUS_SRC_TILE * res.x, 0);
    vec4 center = texelFetch(scratchTex, px + srcOff, 0);

    // Pure deterministic mirrors (< 0.08, matching deferred2.csh) bypass spatial blur completely!
    if (isSpec && rough < 0.08) {
        #ifdef ATROUS_WRITE_COLORTEX4
            imageStore(colorimg4, px, roundHalf(center));
        #else
            imageStore(scratchImg, px + ivec2(ATROUS_DST_TILE * res.x, 0), roundHalf(center));
        #endif
        return;
    }

    float centerLuma = luma(center.rgb);

    // 3x3 variance smoothing on pass 0 only.
    float stdDev = center.a;
    #if ATROUS_PASS == 0
        float varSum = 0.0;
        float varW = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                ivec2 q = clamp(px + ivec2(x, y), ivec2(0), res - 1);
                float kw = (x == 0 ? 0.5 : 0.25) * (y == 0 ? 0.5 : 0.25);
                float qs = texelFetch(scratchTex, q + srcOff, 0).a;
                varSum += kw * qs * qs;
                varW += kw;
            }
        }
        stdDev = sqrt(max(varSum / max(varW, 1e-6), 1e-10));
    #endif

    // Uniform A-Trous stride across the entire screen (never widen stride selectively on low specAge,
    // which otherwise draws a visible rectangular screen-window boundary when rotating the camera!).
    int stride = 1 << ATROUS_PASS;

    float phiLuma = isSpec
        ? max(4.5 * stdDev, 0.008)
        : max(6.5 * stdDev, 0.015 * max(centerLuma, 0.20));
    vec2 centerNdc = ((vec2(px) + 0.5) * texelSize()) * 2.0 - 1.0;
    #ifdef TAA_ENABLED
        centerNdc -= taaJitter;
    #endif
    vec3 centerView = fastScreenToViewNdc(centerNdc, depth);
    vec3 centerViewGeoN = normalize(mat3(gbufferModelView) * geoNormal);

    const float k[2] = float[2](1.0, 0.5);

    vec3 sumCol = center.rgb;
    float sumVar = center.a * center.a;
    float sumW = 1.0;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 q = px + ivec2(x, y) * stride;
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;

            float wGeo = atrousWeight(centerView, centerViewGeoN, geoNormal, shadeNormal, q, isSpec, rough);
            if (wGeo <= 1e-5) continue;

            vec4 tap = texelFetch(scratchTex, q + srcOff, 0);
            float wLuma = exp(-abs(centerLuma - luma(tap.rgb)) / phiLuma);
            float w = k[abs(x)] * k[abs(y)] * wGeo * wLuma;

            sumCol += w * tap.rgb;
            sumVar += w * w * (tap.a * tap.a);
            sumW   += w;
        }
    }

    vec4 outVal = vec4(sumCol / sumW, sqrt(max(sumVar, 0.0)) / sumW);

    // Disocclusion-only FeedbackTap = 0 on pass 0 for Diffuse GI only:
    // Specular history stays 100% pure temporal (zero spatial write-back -> mathematically zero recursive smear!).
    #if ATROUS_PASS == 0 && !defined(ATROUS_SPECULAR)
        int currSide = frameCounter & 1;
        ivec2 hPx = px + ivec2((currSide * 2) * res.x, 0);
        float age = imageLoad(diffHistImg, hPx).a;
        float fb = smoothstep(8.0, 1.0, age) * 0.40;
        if (fb > 0.01) {
            imageStore(diffHistImg, hPx, roundHalf(vec4(mix(center.rgb, outVal.rgb, fb), age)));
        }
    #endif

    #ifdef ATROUS_WRITE_COLORTEX3
        imageStore(colorimg3, px, roundHalf(outVal));
    #elif defined(ATROUS_WRITE_COLORTEX4)
        imageStore(colorimg4, px, roundHalf(outVal));
    #else
        imageStore(scratchImg, px + ivec2(ATROUS_DST_TILE * res.x, 0), roundHalf(outVal));
    #endif
}
