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

            // Thin-geometry (1-3px iron bars / chains / latticework) 3x3 jitter-aligned fallback:
            // Only executes when 2x2 bilinear misses completely (zero overhead on normal surfaces).
            if (support <= 1e-4) {
                vec2 alignedPos = (prevUV + 0.5 * (getPrevTaaJitter() - taaJitter)) * vec2(res) - 0.5;
                ivec2 centerPx = ivec2(round(alignedPos));
                float bestDistSq = 1e9;
                for (int y = -1; y <= 1; ++y) {
                    for (int x = -1; x <= 1; ++x) {
                        ivec2 tap = centerPx + ivec2(x, y);
                        if (any(lessThan(tap, ivec2(0))) || any(greaterThanEqual(tap, res))) continue;
                        vec4 pg = texelFetch(prevGuideTex, tap, 0);
                        float pDepth = pg.w;
                        if (pDepth <= 0.0) continue;
                        vec3 pGeoN   = unpackNormal16(uint(round(pg.x)));
                        vec3 pShadeN = unpackNormal16(uint(round(pg.y)));
                        if (dot(geoNormal, pGeoN) < 0.90 || dot(shadeNormal, pShadeN) < 0.45) continue;
                        if (abs(expectedPrevDepth - pDepth) > max(0.08, 0.05 * max(expectedPrevDepth, 0.1))) continue;
                        vec2 d = vec2(tap) - alignedPos;
                        float d2 = dot(d, d);
                        if (d2 < bestDistSq) {
                            bestDistSq = d2;
                            vec4 hs = texelFetch(diffHistTex, tap + prevSlowOff, 0);
                            oldSlow = hs.rgb;
                            oldAge  = hs.a;
                            oldFast = texelFetch(diffHistTex, tap + prevFastOff, 0);
                            oldMom  = texelFetch(diffMomTex,  tap + prevMomOff,  0).rg;
                            support = 1.0;
                        }
                    }
                }
            }
        }
    }

    if (support > 1e-4) {
        oldSlow /= support;
        oldAge  /= support;
        oldFast /= support;
        oldMom  /= support;
    }

    // Untraced pixel: reuse exact reprojected history (zero spatial blur on static/moving surfaces!)
    // or geometry-aware fill from surrounding traced pixels when disoccluded!
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
                vec3 filled = checkerboardCrossFill(colortex3, px, res, geoNormal, shadeNormal, depth, false, 0.0).rgb;
            #endif
            float fl = luma(filled);
            imageStore(diffHistImg, px + currSlowOff, roundHalf(vec4(filled, 0.5)));
            imageStore(diffHistImg, px + currFastOff, roundHalf(vec4(filled, 0.5)));
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
        const float maxHist    = min(DIFFUSE_MAX_HISTORY, 48.0);
        const float fastWindow = 6.0;
    #elif DIFFUSE_MODE == 1
        vec3 curr = despikeCheckerboard(colortex3, px, res, raw, geoNormal, shadeNormal, depth, 6.0, 0.8);
        const float maxHist    = min(DIFFUSE_MAX_HISTORY, 64.0);
        const float fastWindow = 8.0;
    #else
        vec3 curr = despike3x3(colortex3, px, res, raw, geoNormal, shadeNormal, depth, 6.0, 0.8);
        const float maxHist    = DIFFUSE_MAX_HISTORY;
        const float fastWindow = 12.0;
    #endif
    float currLuma = luma(curr);

    float age = min(maxHist, oldAge + 1.0);
    float alpha = 1.0 / max(age, 1.0);

    // Full-RGB dual-EMA dynamic-lighting detector (stored in currFastOff, zero extra VRAM):
    // Rock-solid (3.5-sigma) on static multi-light rooms while responding cleanly to broken/placed lights!
    float fastAge = min(fastWindow, oldFast.a + 1.0);
    float fastAlpha = 1.0 / max(fastAge, 1.0);
    vec3 fastRGB = (support > 1e-4) ? mix(oldFast.rgb, curr, fastAlpha) : curr;

    float momAlpha = max(alpha, 1.0 / 24.0);
    vec2 moments = (support > 1e-4)
        ? mix(oldMom, vec2(currLuma, currLuma * currLuma), momAlpha)
        : vec2(currLuma, currLuma * currLuma);
    float stdDev = sqrt(max(moments.y - moments.x * moments.x, 0.0));

    float response = 0.0;
    if (support > 1e-4 && oldAge > fastWindow) {
        // Exact EMA standard error = sigma / sqrt(2 * fastWindow - 1).
        float fastErr = stdDev * inversesqrt(max(2.0 * fastWindow - 1.0, 1.0));
        float lumDiff = abs(luma(fastRGB) - luma(oldSlow));
        float rgbDiff = length(fastRGB - oldSlow) * 0.57735027;
        float change  = max(lumDiff, rgbDiff);
        float noiseFloor = 3.5 * fastErr + 0.035;
        float relChange = change / max(max(luma(fastRGB), luma(oldSlow)), noiseFloor);
        float sigRatio  = change / max(noiseFloor, 1e-4);
        response = smoothstep(0.40, 0.75, relChange) * smoothstep(1.0, 2.0, sigRatio);
    }

    if (response > 0.01) {
        float targetAlpha = mix(alpha, 0.45, response);
        alpha = max(alpha, targetAlpha);
        age = 1.0 / alpha;
    }

    // Soft temporal firefly guard activates only when steady-state (never suppresses real new lights!).
    vec3 safeCurr = curr;
    if (support > 1e-4 && oldAge >= 4.0 && response < 0.15) {
        float refLuma = max(luma(oldSlow), luma(fastRGB));
        float maxSettledLuma = max(refLuma * 5.0 + 0.6, 1.2);
        if (currLuma > maxSettledLuma) {
            safeCurr *= maxSettledLuma / max(currLuma, 1e-6);
        }
    }

    // During strong confirmed lighting transitions, pull oldSlow toward the already-filtered fastRGB
    // so stale screen-space history clears cleanly across the entire room with zero screen-window residual!
    vec3 baseSlow = (response > 0.35) ? mix(oldSlow, fastRGB, smoothstep(0.35, 0.85, response) * 0.75) : oldSlow;
    vec3 integrated = (support > 1e-4) ? mix(baseSlow, safeCurr, alpha) : safeCurr;

    imageStore(diffHistImg, px + currSlowOff, roundHalf(vec4(integrated, age)));
    imageStore(diffHistImg, px + currFastOff, roundHalf(vec4(fastRGB, fastAge)));
    imageStore(diffMomImg,  px + currMomOff,  vec4(moments, 0.0, 0.0));
    imageStore(scratchImg,  px,               roundHalf(vec4(integrated, age)));
}
