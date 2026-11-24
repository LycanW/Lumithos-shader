#version 430 compatibility

// Pass: Physically-Based Specular SVGF Temporal Accumulation
// 1) Dual reprojection: stabilized specular-dominant virtual reflection reprojection with surface fallback.
// 2) Exact 3D point-to-plane verification against prevGuideTex (preserves valid virtual parallax on tilted floors/walls).
// 3) Cumulative GGX lobe angular drift bound: N * dTheta <= 1.5 * roughness^2 (FPS-independent, zero roughness cliffs).
// 4) Statistical 9-sample (m1 ± 2.5*sigma) variance clamp during parallax to kill occluded-lantern ghosts without dropping samples!
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D specHistImg;
layout(rg32f)   uniform writeonly image2D specMomImg;
layout(rgba16f) uniform writeonly image2D scratchImg;

vec3 prevViewFromGuide(ivec2 tap, ivec2 res, float pLinDepth, vec2 prevJitter) {
    vec2 ndc = ((vec2(tap) + 0.5) / vec2(res)) * 2.0 - 1.0 - prevJitter;
    float fx = gbufferPreviousProjection[0][0];
    float fy = gbufferPreviousProjection[1][1];
    return vec3((ndc.x / fx) * pLinDepth, (ndc.y / fy) * pLinDepth, -pLinDepth);
}

bool validPrevSpecTap(ivec2 tap, ivec2 res, ivec2 prevOff, vec2 prevJitter, vec3 prevSurfView, vec3 prevViewGeoN,
                      vec3 geoNormal, vec3 shadeNormal, float roughness, float planeTol,
                      out vec4 outHist, out vec2 outMom) {
    outHist = vec4(0.0);
    outMom  = vec2(0.0);
    if (any(lessThan(tap, ivec2(0))) || any(greaterThanEqual(tap, res))) return false;

    vec4 pg = texelFetch(prevGuideTex, tap, 0);
    float pDepth = pg.w;
    if (pDepth <= 0.0) return false;

    vec3 pGeoN = unpackNormal16(uint(round(pg.x)));
    if (dot(geoNormal, pGeoN) < 0.90) return false;
    vec3 pShadeN = unpackNormal16(uint(round(pg.y)));
    if (dot(shadeNormal, pShadeN) < 0.65) return false;
    if (abs(roughness - pg.z) > max(0.05, 0.22 * max(roughness, pg.z))) return false;

    vec3 pTapView = prevViewFromGuide(tap, res, pDepth, prevJitter);
    if (abs(dot(pTapView - prevSurfView, prevViewGeoN)) > planeTol) return false;

    vec4 hs = texelFetch(specHistTex, tap + prevOff, 0);
    // Reject non-reflective / skipped pixels (which store age == 0) so matte zero-values
    // NEVER bleed into adjacent rough reflective surfaces across material boundaries!
    if (hs.a <= 0.01) return false;

    outHist = hs;
    outMom  = texelFetch(specMomTex, tap + prevOff, 0).rg;
    return true;
}

float fetchPrevSpecHistory(vec2 prevUV, ivec2 res, ivec2 prevOff, vec2 prevJitter, vec3 prevSurfView, vec3 prevViewGeoN,
                           vec3 geoNormal, vec3 shadeNormal, float roughness, float planeTol, bool allowThinFallback,
                           out vec3 outSlow, out vec2 outMom, out float outAge) {
    outSlow = vec3(0.0);
    outMom  = vec2(0.0);
    outAge  = 0.0;
    if (any(lessThan(prevUV, vec2(0.0))) || any(greaterThanEqual(prevUV, vec2(1.0)))) return 0.0;

    vec2 pCoord = prevUV * vec2(res) - 0.5;
    ivec2 base = ivec2(floor(pCoord));
    vec2 f = pCoord - vec2(base);
    float support = 0.0;

    for (int j = 0; j < 4; ++j) {
        ivec2 tap = base + ivec2(j & 1, j >> 1);
        float w = ((j & 1) == 0 ? 1.0 - f.x : f.x) * ((j >> 1) == 0 ? 1.0 - f.y : f.y);
        if (w <= 1e-5) continue;

        vec4 hs;
        vec2 hm;
        if (!validPrevSpecTap(tap, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                              geoNormal, shadeNormal, roughness, planeTol, hs, hm)) continue;

        outSlow += w * hs.rgb;
        outAge  += w * hs.a;
        outMom  += w * hm;
        support += w;
    }

    if (support > 1e-4) {
        outSlow /= support;
        outAge  /= support;
        outMom  /= support;
        return support;
    }

    // Thin-geometry (1-3px iron bars / chains / latticework) fallback:
    // Only runs when 2x2 bilinear misses completely (0ms on normal surfaces); searches a 3x3
    // neighborhood around the exact jitter-aligned previous pixel to recover subpixel-shifted thin bars!
    if (allowThinFallback) {
        vec2 alignedUV = prevUV + 0.5 * (prevJitter - taaJitter);
        vec2 alignedPos = alignedUV * vec2(res) - 0.5;
        ivec2 centerPx = ivec2(round(alignedPos));
        float bestDistSq = 1e9;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                ivec2 tap = centerPx + ivec2(x, y);
                vec4 hs;
                vec2 hm;
                if (!validPrevSpecTap(tap, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                                      geoNormal, shadeNormal, roughness, planeTol, hs, hm)) continue;
                vec2 d = vec2(tap) - alignedPos;
                float d2 = dot(d, d);
                if (d2 < bestDistSq) {
                    bestDistSq = d2;
                    outSlow = hs.rgb;
                    outAge  = hs.a;
                    outMom  = hm;
                    support = 1.0;
                }
            }
        }
    }
    return support;
}

// Gather local 9-sample (same-parity traced) RGB moments and stabilized hit distance in one pass.
// Uses rank-2 (second-max) outlier clamping so an isolated single-pixel HDR firefly cannot blow up
// a newly disoccluded 3x3 strip, while genuine multi-hit glossy reflections stay 100% unbiased!
void gatherTracedSpecStats(ivec2 px, ivec2 res, bool isTraced, bool useChecker, vec3 centerCol, float centerDist,
                           vec3 geoNormal, vec3 shadeNormal, float centerDepth,
                           out vec3 outM1, out vec3 outSigma, out float outHitDist, out int outCount) {
    vec3 cols[9];
    float lumas[9];
    float maxN = 0.0;
    float secondMax = 0.0;
    float distSum = 0.0;
    outCount = 0;

    if (isTraced) {
        float cl = luma(centerCol);
        cols[0] = centerCol;
        lumas[0] = cl;
        maxN = cl;
        distSum = min(centerDist, 64.0);
        outCount = 1;
    }

    const ivec2 cbOffs[8] = ivec2[8](
        ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1),
        ivec2(-2,  0), ivec2(2,  0), ivec2( 0,-2), ivec2(0, 2)
    );
    const ivec2 axOffs[4] = ivec2[4](
        ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1)
    );
    const ivec2 fullOffs[8] = ivec2[8](
        ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
        ivec2(-1,  0),               ivec2(1,  0),
        ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1)
    );

    int nOffs = isTraced ? 8 : 4;
    for (int i = 0; i < nOffs; ++i) {
        ivec2 q = px + (isTraced ? (useChecker ? cbOffs[i] : fullOffs[i]) : axOffs[i]);
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;
        float qd = texelFetch(depthtex1, q, 0).r;
        if (qd >= 0.999999 || abs(qd - centerDepth) > 0.025 * max(centerDepth, 0.01)) continue;
        vec3 qGeoN, qShadeN;
        readGBufferNormals(q, qGeoN, qShadeN);
        if (dot(geoNormal, qGeoN) < 0.92 || dot(shadeNormal, qShadeN) < 0.70) continue;

        vec4 qt = texelFetch(colortex4, q, 0);
        if (qt.a < 0.0) continue;
        vec3 qc = max(qt.rgb, vec3(0.0));
        if (any(isnan(qc)) || any(isinf(qc))) continue;

        float ql = luma(qc);
        cols[outCount] = qc;
        lumas[outCount] = ql;
        if (ql > maxN) { secondMax = maxN; maxN = ql; }
        else if (ql > secondMax) { secondMax = ql; }
        distSum += min(qt.a, 64.0);
        ++outCount;
    }

    if (outCount > 0) {
        float fireflyLim = (outCount >= 3) ? (secondMax * 3.5 + 0.25) : 1e6;
        outM1 = vec3(0.0);
        vec3 m2 = vec3(0.0);
        for (int i = 0; i < outCount; ++i) {
            vec3 c = cols[i];
            if (lumas[i] > fireflyLim) {
                c *= fireflyLim / max(lumas[i], 1e-6);
            }
            outM1 += c;
            m2 += c * c;
        }
        float invN = 1.0 / float(outCount);
        outM1 *= invN;
        m2 *= invN;
        outSigma = sqrt(max(m2 - outM1 * outM1, vec3(0.0)));
        outHitDist = distSum * invN;
    } else {
        outM1 = centerCol;
        outSigma = vec3(0.0);
        outHitDist = min(centerDist, 64.0);
    }
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    int currSide = frameCounter & 1;
    int prevSide = currSide ^ 1;
    ivec2 prevOff = ivec2(prevSide * res.x, 0);
    ivec2 currOff = ivec2(currSide * res.x, 0);

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        imageStore(scratchImg, px, vec4(0.0));
        imageStore(specHistImg, px + currOff, vec4(0.0));
        imageStore(specMomImg,  px + currOff, vec4(0.0));
        return;
    }

    float roughness, f0Raw;
    bool isHand;
    readGBufferSpecMask(px, roughness, f0Raw, isHand);
    if (!hasSpecularTrace(roughness, f0Raw, isHand)) {
        imageStore(scratchImg, px, vec4(0.0));
        imageStore(specHistImg, px + currOff, vec4(0.0));
        imageStore(specMomImg,  px + currOff, vec4(0.0));
        return;
    }

    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec4 rawTap = texelFetch(colortex4, px, 0);
    bool isTraced = (rawTap.a >= 0.0);
    vec3 raw = isTraced ? max(rawTap.rgb, vec3(0.0)) : vec3(0.0);
    if (any(isnan(raw)) || any(isinf(raw))) raw = vec3(0.0);

    // Pure FLAT deterministic mirrors bypass temporal accumulation completely for zero ghosting,
    // and preserve hitDist in .a so TAA can reproject at exact planar virtual parallax!
    bool isFlatMirror = (roughness < 0.08) && (dot(shadeNormal, geoNormal) > 0.998);
    if (isFlatMirror) {
        imageStore(specHistImg, px + currOff, roundHalf(vec4(raw, 1.0)));
        imageStore(specMomImg,  px + currOff, vec4(luma(raw), luma(raw) * luma(raw), 0.0, 0.0));
        imageStore(scratchImg,  px,           roundHalf(vec4(raw, rawTap.a)));
        return;
    }

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);

    #ifdef SPECULAR_CHECKERBOARD
        bool useChecker = (roughness >= 0.16);
    #else
        bool useChecker = false;
    #endif

    vec3 fillMin = vec3(0.0);
    vec3 fillMax = vec3(0.0);
    bool fillHasBounds = false;
    vec4 filledTap = vec4(0.0, 0.0, 0.0, 64.0);
    vec3 curr = vec3(0.0);

    if (isTraced) {
        float roughT = smoothstep(0.08, 0.35, roughness);
        if (useChecker) {
            curr = despikeCheckerboard(colortex4, px, res, raw, geoNormal, shadeNormal, depth,
                                       mix(5.5, 3.0, roughT), mix(0.22, 0.05, roughT));
        } else {
            curr = despike3x3(colortex4, px, res, raw, geoNormal, shadeNormal, depth,
                              mix(5.5, 3.0, roughT), mix(0.22, 0.05, roughT));
        }
    } else {
        filledTap = checkerboardCrossFill(colortex4, px, res, geoNormal, shadeNormal, depth, true, roughness, fillMin, fillMax, fillHasBounds);
    }
    float currLuma = luma(curr);

    // Gather stabilized hitDist + rank-2 outlier-safe 9-sample moments (m1, sigma) for anti-ghosting & disocclusion.
    vec3 locM1, locSigma;
    float smoothHitDist;
    int locCount;
    gatherTracedSpecStats(px, res, isTraced, useChecker, curr, isTraced ? rawTap.a : filledTap.a,
                          geoNormal, shadeNormal, depth, locM1, locSigma, smoothHitDist, locCount);

    // Previous-frame surface plane for exact 3D plane-distance verification.
    vec3 prevWorldPos = worldPos + cameraVelocity;
    vec3 prevSurfView = (gbufferPreviousModelView * vec4(prevWorldPos, 1.0)).xyz;
    vec3 prevViewGeoN = normalize(mat3(gbufferPreviousModelView) * geoNormal);
    float planeTol = max(0.06, 0.025 * max(-prevSurfView.z, 0.1));

    // Planar virtual reflection reprojection is ONLY geometrically valid on flat, smooth mirrors
    // (dot(shadeNormal, geoNormal) ≈ 1 and roughness < 0.25). On normal-mapped/bumpy surfaces (e.g.
    // striped iron blocks) or rough surfaces, the reflection is anchored to per-texel shadeNormal ->
    // use exact surface reprojection (surfPrevUV) so history NEVER slides across blocks as a double image!
    vec3 eyeToSurf = mat3(gbufferModelViewInverse) * viewPos;
    vec3 viewRay   = normalize(eyeToSurf);
    float flatness = smoothstep(0.985, 0.998, dot(shadeNormal, geoNormal));
    float domFactor = smoothstep(0.25, 0.08, roughness) * flatness;

    vec3 oldSlow = vec3(0.0);
    vec2 oldMom  = vec2(0.0);
    float oldAge = 0.0;
    float support = 0.0;

    if (frameCounter > 1) {
        vec2 prevJitter = getPrevTaaJitter();
        if (domFactor > 0.01) {
            vec3 virtualPos = worldPos + viewRay * (smoothHitDist * domFactor);
            vec2 virtPrevUV = reprojectScreenUV(uv, virtualPos);
            support = fetchPrevSpecHistory(virtPrevUV, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                                           geoNormal, shadeNormal, roughness, planeTol, false, oldSlow, oldMom, oldAge);
        }
        if (support <= 1e-4) {
            vec2 surfPrevUV = reprojectScreenUV(uv, worldPos);
            support = fetchPrevSpecHistory(surfPrevUV, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                                           geoNormal, shadeNormal, roughness, planeTol, true, oldSlow, oldMom, oldAge);
        }
    }

    // Cumulative GGX reflected-lobe angular drift bound: N * dTheta_per_sample <= 3.5 * roughness^2.
    // Automatically scales with FPS, camera distance, roughness, and checkerboard sample interval
    // without wiping out history on close-up iron bars / railings!
    vec3 currV = -viewRay;
    vec3 prevEyePos = -transpose(mat3(gbufferPreviousModelView)) * gbufferPreviousModelView[3].xyz;
    vec3 prevV = normalize(prevEyePos - prevWorldPos);
    float cosParallax = saturate(dot(currV, prevV));
    float sinParallax = sqrt(max(0.0, 1.0 - cosParallax * cosParallax));
    float sampleInterval = useChecker ? 2.0 : 1.0;
    float dThetaPerSample = sinParallax * sampleInterval;
    float ggxAlpha = max(roughness * roughness, 0.006);
    float maxHist = clamp((3.5 * ggxAlpha) / max(dThetaPerSample, 1e-5), 2.5, SPECULAR_MAX_HISTORY);
    oldAge = min(oldAge, maxHist);

    // Statistical temporal-vs-spatial consistency guard (camera-motion-independent):
    // Compares reprojected history against the current 9-sample spatial mean (locM1) using BOTH
    // spatial stdDev (locSigma) and accumulated temporal stdDev (prevTempSigma).
    // - Static noisy GGX reflections (high prevTempSigma) remain 100% inside the envelope (zero sample drop!).
    // - Moving/sliding reflections & dynamic toggling lights (e.g. redstone repeaters/lamps) fall outside
    //   the envelope even when camera parallax is zero -> clamped immediately with proportional age reset!
    if (support > 1e-4 && locCount >= 4) {
        float prevTempSigma = sqrt(max(oldMom.y - oldMom.x * oldMom.x, 0.0));
        float parallaxTighten = mix(1.0, 0.65, saturate(dThetaPerSample / ggxAlpha));
        vec3 clampExtent = max(max(locSigma * 2.0, vec3(prevTempSigma * 1.5)), vec3(0.04)) * parallaxTighten;

        vec3 clamped = clamp(oldSlow, max(locM1 - clampExtent, vec3(0.0)), locM1 + clampExtent);
        vec3 excess = abs(oldSlow - clamped);
        float excessRatio = maxOf(excess / max(clampExtent, vec3(0.04)));

        if (excessRatio > 0.01) {
            float pullW = smoothstep(0.0, 1.5, excessRatio);
            oldSlow = mix(oldSlow, clamped, pullW);
            oldAge  = mix(oldAge, min(oldAge, 3.0), pullW);
            float cl = luma(oldSlow);
            float sl = luma(locSigma);
            oldMom = mix(oldMom, vec2(cl, cl * cl + sl * sl), pullW);
        }
    }

    if (!isTraced) {
        if (support > 1e-4) {
            float tempStdDev = sqrt(max(oldMom.y - oldMom.x * oldMom.x, 0.0));
            float spatStdDev = (locCount >= 2) ? luma(locSigma) : max(0.25 * luma(oldSlow), 0.05);
            float stdDev = (oldAge < 4.0) ? max(tempStdDev, spatStdDev * (2.0 / max(oldAge, 1.0))) : tempStdDev;
            imageStore(specHistImg, px + currOff, roundHalf(vec4(oldSlow, oldAge)));
            imageStore(specMomImg,  px + currOff, vec4(oldMom, 0.0, 0.0));
            imageStore(scratchImg,  px,           roundHalf(vec4(oldSlow, stdDev)));
        } else {
            vec3 initCol = (locCount >= 2) ? locM1 : filledTap.rgb;
            float fl = luma(initCol);
            float stdDev = (locCount >= 2) ? max(luma(locSigma) * 2.0, 0.15 * fl) : max(0.25 * fl, 0.05);
            imageStore(specHistImg, px + currOff, roundHalf(vec4(initCol, 0.5)));
            imageStore(specMomImg,  px + currOff, vec4(fl, fl * fl, 0.0, 0.0));
            imageStore(scratchImg,  px,           roundHalf(vec4(initCol, stdDev)));
        }
        return;
    }

    float age = min(maxHist, oldAge + 1.0);
    float alpha = 1.0 / max(age, 1.0);
    vec3 safeCurr = curr;
    if (support > 1e-4 && oldAge >= 2.0) {
        float refLuma = max(luma(oldSlow), luma(locM1));
        float maxSpecLuma = max(refLuma * 4.5 + 0.4, 1.0);
        if (currLuma > maxSpecLuma) {
            safeCurr *= maxSpecLuma / max(currLuma, 1e-6);
        }
    }

    // Textbook SVGF disocclusion initialization (Schied 2017, Sec. 4.2) using unbiased outlier-safe local mean:
    // Preserves true glossy highlight brightness while rejecting isolated single-pixel HDR fireflies!
    vec3 integrated;
    vec2 moments;
    if (support <= 1e-4) {
        integrated = (locCount >= 3) ? locM1 : safeCurr;
        float initLuma = luma(integrated);
        moments = vec2(initLuma, initLuma * initLuma);
    } else {
        float safeLuma = luma(safeCurr);
        moments = mix(oldMom, vec2(safeLuma, safeLuma * safeLuma), alpha);
        integrated = mix(oldSlow, safeCurr, alpha);
    }

    float tempStdDev = sqrt(max(moments.y - moments.x * moments.x, 0.0));
    float spatStdDev = (locCount >= 3) ? luma(locSigma) : max(0.25 * luma(integrated), 0.05);
    float stdDev = (age < 4.0) ? max(tempStdDev, spatStdDev * (2.0 / max(age, 1.0))) : tempStdDev;

    imageStore(specHistImg, px + currOff, roundHalf(vec4(integrated, age)));
    imageStore(specMomImg,  px + currOff, vec4(moments, 0.0, 0.0));
    imageStore(scratchImg,  px,           roundHalf(vec4(integrated, stdDev)));
}
