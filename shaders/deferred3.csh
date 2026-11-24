#version 430 compatibility

// Pass: Diffuse SVGF Temporal Accumulation (Ping-Pong Race-Free + 3x3 RCRS Despike + Variance-Aware Response)
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D diffHistImg;
layout(rg32f)   uniform writeonly image2D diffMomImg;
layout(rgba16f) uniform writeonly image2D scratchImg;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    int currSide = frameCounter & 1;
    int prevSide = currSide ^ 1;
    ivec2 prevSlowOff = ivec2((prevSide * 2) * res.x, 0);
    ivec2 prevFastOff = ivec2((prevSide * 2 + 1) * res.x, 0);
    ivec2 currSlowOff = ivec2((currSide * 2) * res.x, 0);
    ivec2 currFastOff = ivec2((currSide * 2 + 1) * res.x, 0);
    ivec2 prevMomOff  = ivec2(prevSide * res.x, 0);
    ivec2 currMomOff  = ivec2(currSide * res.x, 0);

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        imageStore(scratchImg, px, vec4(0.0));
        imageStore(diffHistImg, px + currSlowOff, vec4(0.0));
        imageStore(diffHistImg, px + currFastOff, vec4(0.0));
        imageStore(diffMomImg,  px + currMomOff,  vec4(0.0));
        return;
    }

    float rough;
    bool isHand;
    readGBufferRoughHand(px, rough, isHand);
    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);
    float expectedPrevDepth = -(gbufferPreviousModelView * vec4(worldPos + cameraVelocity, 1.0)).z;

    #if DEBUG_VIEW == 1 || DIFFUSE_TRACE_RES == 2
        #define DIFFUSE_MODE 2
    #elif DIFFUSE_TRACE_RES == 1
        #define DIFFUSE_MODE 1
    #else
        #define DIFFUSE_MODE 0
    #endif

    #if DIFFUSE_MODE == 0
        ivec2 qOff = diffuseQuarterOffset(frameCounter);
        bool isTraced = ((px & 1) == qOff);
    #elif DIFFUSE_MODE == 1
        bool isTraced = (((px.x ^ px.y ^ frameCounter) & 1) == 0);
    #else
        bool isTraced = true;
    #endif

    // Bilinear geometric reprojection against prevGuideTex (exact integer normal decode).
    vec3 oldSlow = vec3(0.0);
    vec4 oldFast = vec4(0.0);
    vec2 oldMom  = vec2(0.0);
    float oldAge = 0.0;
    float support = 0.0;

    if (frameCounter > 1 && !isHand) {
        vec2 prevUV = reprojectScreenUV(uv, worldPos);
        if (all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
            vec2 pCoord = prevUV * vec2(res) - 0.5;
            ivec2 base = ivec2(floor(pCoord));
            vec2 f = pCoord - vec2(base);

            for (int j = 0; j < 4; ++j) {
                ivec2 tap = base + ivec2(j & 1, j >> 1);
                if (any(lessThan(tap, ivec2(0))) || any(greaterThanEqual(tap, res))) continue;
                float w = ((j & 1) == 0 ? 1.0 - f.x : f.x) * ((j >> 1) == 0 ? 1.0 - f.y : f.y);
                if (w <= 1e-5) continue;

                vec4 pg = texelFetch(prevGuideTex, tap, 0);
                float pDepth = pg.w;
                if (pDepth <= 0.0) continue;

                vec3 pGeoN   = unpackNormal16(uint(round(pg.x)));
                vec3 pShadeN = unpackNormal16(uint(round(pg.y)));
                if (dot(geoNormal, pGeoN) < 0.90) continue;
                if (dot(shadeNormal, pShadeN) < 0.45) continue;
                if (abs(expectedPrevDepth - pDepth) > max(0.08, 0.05 * max(expectedPrevDepth, 0.1))) continue;

                vec4 hs = texelFetch(diffHistTex, tap + prevSlowOff, 0);
                vec4 hf = texelFetch(diffHistTex, tap + prevFastOff, 0);
                vec2 hm = texelFetch(diffMomTex,  tap + prevMomOff,  0).rg;

                oldSlow += w * hs.rgb;
                oldAge  += w * hs.a;
                oldFast += w * hf;
                oldMom  += w * hm;
                support += w;
            }
        }
    }

    if (support > 1e-4) {
        oldSlow /= support;
        oldAge  /= support;
        oldFast /= support;
        oldMom  /= support;
    }

    // Untraced checkerboard pixel: reuse exact reprojected history (zero blur on static/moving surfaces!)
    // or cross-fill from the 4 adjacent traced pixels when disoccluded!
    if (!isTraced) {
        if (support > 1e-4) {
            imageStore(diffHistImg, px + currSlowOff, roundHalf(vec4(oldSlow, oldAge)));
            imageStore(diffHistImg, px + currFastOff, roundHalf(oldFast));
            imageStore(diffMomImg,  px + currMomOff,  vec4(oldMom, 0.0, 0.0));
            imageStore(scratchImg,  px,               roundHalf(vec4(oldSlow, max(oldAge, 1.0))));
        } else {
            #if DIFFUSE_MODE == 0
                vec3 filled = quarterBilinearFill(colortex3, px, res, qOff, geoNormal, shadeNormal, depth);
            #else
                vec3 fMin, fMax;
                bool fHas;
                vec3 filled = checkerboardCrossFill(colortex3, px, res, geoNormal, shadeNormal, depth, false, 0.0, fMin, fMax, fHas).rgb;
            #endif
            float fl = luma(filled);
            imageStore(diffHistImg, px + currSlowOff, roundHalf(vec4(filled, 0.5)));
            imageStore(diffHistImg, px + currFastOff, roundHalf(vec4(fl, fl, 0.5, 0.0)));
            imageStore(diffMomImg,  px + currMomOff,  vec4(fl, fl * fl, 0.0, 0.0));
            imageStore(scratchImg,  px,               roundHalf(vec4(filled, 1.0)));
        }
        return;
    }

    vec3 raw = max(texelFetch(colortex3, px, 0).rgb, vec3(0.0));
    if (any(isnan(raw)) || any(isinf(raw))) raw = vec3(0.0);
    // Preserve distant small-emitter hits while suppressing extreme multi-light fireflies.
    #if DIFFUSE_MODE == 0
        vec3 curr = despikeQuarter(colortex3, px, res, raw, geoNormal, shadeNormal, depth, 6.0, 0.8);
    #elif DIFFUSE_MODE == 1
        vec3 curr = despikeCheckerboard(colortex3, px, res, raw, geoNormal, shadeNormal, depth, 6.0, 0.8);
    #else
        vec3 curr = despike3x3(colortex3, px, res, raw, geoNormal, shadeNormal, depth, 6.0, 0.8);
    #endif
    float currLuma = luma(curr);

    float age = min(DIFFUSE_MAX_HISTORY, oldAge + 1.0);
    float alpha = 1.0 / max(age, 1.0);

    // Variance-aware responsive history (stable on dark/multi-lantern scenes, fast on real light changes).
    float fastAlpha = 1.0 / min(age, 12.0);
    float fastSafeLuma = (oldAge > 8.0) ? min(currLuma, max(oldFast.y * 4.0 + 0.5, 1.5)) : currLuma;
    float fastLuma  = mix(oldFast.x, fastSafeLuma, fastAlpha);
    float slowLuma  = mix(oldFast.y, currLuma, alpha);

    vec2 moments = mix(oldMom, vec2(fastSafeLuma, fastSafeLuma * fastSafeLuma), alpha);
    float stdDev = sqrt(max(moments.y - moments.x * moments.x, 0.0));

    float noiseFloor = 3.5 * stdDev * inversesqrt(12.0) + 0.035;
    float relChange = abs(fastLuma - slowLuma) / max(max(fastLuma, slowLuma), noiseFloor);
    float response = smoothstep(0.45, 0.75, relChange);

    if (response > 0.01 && age > 12.0) {
        float targetAlpha = mix(alpha, 0.20, response);
        alpha = max(alpha, targetAlpha);
        age = 1.0 / alpha;
        if (response > 0.65) slowLuma = fastLuma;
    }

    // Soft temporal firefly guard on settled pixels prevents isolated 20x spikes from pulsing.
    vec3 safeCurr = curr;
    if (oldAge >= 12.0 && response < 0.25) {
        float maxSettledLuma = max(luma(oldSlow) * 12.0 + 2.5, 4.0);
        if (currLuma > maxSettledLuma) {
            safeCurr *= maxSettledLuma / max(currLuma, 1e-6);
        }
    }

    vec3 integrated = mix(oldSlow, safeCurr, alpha);

    imageStore(diffHistImg, px + currSlowOff, roundHalf(vec4(integrated, age)));
    imageStore(diffHistImg, px + currFastOff, roundHalf(vec4(fastLuma, slowLuma, age, 0.0)));
    imageStore(diffMomImg,  px + currMomOff,  vec4(moments, 0.0, 0.0));
    imageStore(scratchImg,  px,               roundHalf(vec4(integrated, age)));
}
