#version 430 compatibility

// Pass: Physically-Based Specular SVGF Temporal Accumulation
// 1) Roughness-agnostic planar virtual reflection reprojection using outlier-proof median hitDist.
// 2) Exact 3D point-to-plane + shading-normal + reflection-space hitDist disocclusion validation against prevGuideTex/specMomTex.
// 3) Geometric multi-depth parallax pixel-error bound (zero roughness cliffs, zero magic color-clamp hacks!).
// 4) Dual-EMA dynamic-reflection response via specMomImg (meanLuma, stdDev, prevHitDist, fastLuma in rgba16f).
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D specHistImg;
layout(rgba16f) uniform writeonly image2D specMomImg;
layout(rgba16f) uniform writeonly image2D scratchImg;

vec3 prevViewFromGuide(ivec2 tap, ivec2 res, float pLinDepth, vec2 prevJitter) {
    vec2 ndc = ((vec2(tap) + 0.5) / vec2(res)) * 2.0 - 1.0 - prevJitter;
    float fx = gbufferPreviousProjection[0][0];
    float fy = gbufferPreviousProjection[1][1];
    return vec3((ndc.x / fx) * pLinDepth, (ndc.y / fy) * pLinDepth, -pLinDepth);
}

bool validPrevSpecTap(ivec2 tap, ivec2 res, ivec2 prevOff, vec2 prevJitter, vec3 prevSurfView, vec3 prevViewGeoN,
                      vec3 geoNormal, vec3 shadeNormal, float planeTol, float minShadeDot,
                      bool checkHitDist, float currHitDist, float hitDistTol,
                      out vec4 outHist, out vec4 outMom) {
    outHist = vec4(0.0);
    outMom  = vec4(0.0);
    if (any(lessThan(tap, ivec2(0))) || any(greaterThanEqual(tap, res))) return false;

    vec4 pg = texelFetch(prevGuideTex, tap, 0);
    float pDepth = pg.w;
    if (pDepth <= 0.0) return false;

    vec3 pGeoN = unpackNormal16(uint(round(pg.x)));
    if (dot(geoNormal, pGeoN) < 0.90) return false;
    vec3 pShadeN = unpackNormal16(uint(round(pg.y)));
    if (dot(shadeNormal, pShadeN) < minShadeDot) return false;

    vec3 pTapView = prevViewFromGuide(tap, res, pDepth, prevJitter);
    if (abs(dot(pTapView - prevSurfView, prevViewGeoN)) > planeTol) return false;

    vec4 hs = texelFetch(specHistTex, tap + prevOff, 0);
    // specHistTex stores demodulated incoming radiance (specEnv) with hs.a > 0 iff reflective:
    // never reject on per-texel roughness variation (which broke virtual tracking inside a single block!).
    if (hs.a <= 0.01) return false;

    vec4 hm = texelFetch(specMomTex, tap + prevOff, 0);
    // Reflection-space disocclusion check: when virtual parallax moves across a foreground/background
    // reflected edge (e.g., pillar/door vs far wall on the same floor plane), reject mismatched hitDist!
    if (checkHitDist && hm.b > 0.05 && currHitDist > 0.05) {
        if (abs(hm.b - currHitDist) > hitDistTol) return false;
    }

    outHist = hs;
    outMom  = hm;
    return true;
}

float fetchPrevSpecHistory(vec2 prevUV, ivec2 res, ivec2 prevOff, vec2 prevJitter, vec3 prevSurfView, vec3 prevViewGeoN,
                           vec3 geoNormal, vec3 shadeNormal, float planeTol, float minShadeDot,
                           bool checkHitDist, float currHitDist, float hitDistTol, bool allowThinFallback,
                           out vec3 outSlow, out vec4 outMom, out float outAge) {
    outSlow = vec3(0.0);
    outMom  = vec4(0.0);
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
        vec4 hm;
        if (!validPrevSpecTap(tap, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                              geoNormal, shadeNormal, planeTol, minShadeDot,
                              checkHitDist, currHitDist, hitDistTol, hs, hm)) continue;

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
    // Only runs when 2x2 bilinear misses completely (0ms on normal surfaces).
    if (allowThinFallback) {
        vec2 alignedUV = prevUV + 0.5 * (prevJitter - taaJitter);
        vec2 alignedPos = alignedUV * vec2(res) - 0.5;
        ivec2 centerPx = ivec2(round(alignedPos));
        float bestDistSq = 1e9;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                ivec2 tap = centerPx + ivec2(x, y);
                vec4 hs;
                vec4 hm;
                if (!validPrevSpecTap(tap, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                                      geoNormal, shadeNormal, planeTol, minShadeDot,
                                      checkHitDist, currHitDist, hitDistTol, hs, hm)) continue;
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

// Gather local 9-sample (same-parity traced) RGB moments, exact outlier-proof MEDIAN hit distance,
// and interquartile hit-distance spread (exact multi-depth parallax ambiguity for wide GGX lobes!).
void gatherTracedSpecStats(ivec2 px, ivec2 res, bool isTraced, bool useChecker, vec3 centerCol, float centerDist,
                           vec3 geoNormal, vec3 shadeNormal, float centerDepth,
                           out vec3 outM1, out vec3 outSigma, out float outHitDist, out float outDistSpread, out int outCount) {
    vec3 cols[9];
    float lumas[9];
    float dists[9];
    float maxN = 0.0;
    float secondMax = 0.0;
    int distCount = 0;
    outCount = 0;

    float centerLinZ = linearizeDepth(centerDepth);
    float depthTol = max(0.08, 0.05 * centerLinZ);

    if (isTraced) {
        float cl = luma(centerCol);
        cols[0] = centerCol;
        lumas[0] = cl;
        maxN = cl;
        if (centerDist > 0.02) {
            dists[0] = clamp(centerDist, 0.05, 128.0);
            distCount = 1;
        }
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
        if (qd >= 0.999999 || abs(linearizeDepth(qd) - centerLinZ) > depthTol) continue;
        vec3 qGeoN, qShadeN;
        readGBufferNormals(q, qGeoN, qShadeN);
        if (dot(geoNormal, qGeoN) < 0.92 || dot(shadeNormal, qShadeN) < 0.75) continue;

        vec4 qt = texelFetch(colortex4, q, 0);
        if (qt.a < 0.0) continue;
        vec3 qc = max(qt.rgb, vec3(0.0));
        if (any(isnan(qc)) || any(isinf(qc))) continue;

        float ql = luma(qc);
        cols[outCount] = qc;
        lumas[outCount] = ql;
        if (ql > maxN) { secondMax = maxN; maxN = ql; }
        else if (ql > secondMax) { secondMax = ql; }
        if (qt.a > 0.02) {
            dists[distCount++] = clamp(qt.a, 0.05, 128.0);
        }
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

        // Exact register insertion sort on <=9 hit distances -> immune to both short grazing bevels
        // AND distant sky/window escapes (unlike arithmetic or harmonic means)!
        if (distCount > 0) {
            for (int i = 1; i < distCount; ++i) {
                float key = dists[i];
                int j = i - 1;
                while (j >= 0 && dists[j] > key) {
                    dists[j + 1] = dists[j];
                    --j;
                }
                dists[j + 1] = key;
            }
            int mid = distCount >> 1;
            outHitDist = ((distCount & 1) == 1) ? dists[mid] : 0.5 * (dists[mid - 1] + dists[mid]);
            int qLo = distCount >> 2;
            int qHi = distCount - 1 - qLo;
            outDistSpread = max(dists[qHi] - dists[qLo], 0.0);
        } else {
            outHitDist = clamp(centerDist, 0.0, 128.0);
            outDistSpread = 0.0;
        }
    } else {
        outM1 = centerCol;
        outSigma = vec3(0.0);
        outHitDist = clamp(centerDist, 0.0, 128.0);
        outDistSpread = 0.0;
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
    // and publish exact hitDist into specMomImg.b + scratchImg.a for glossy neighbors & TAA!
    bool isFlatMirror = (roughness < 0.08) && (dot(shadeNormal, geoNormal) > 0.998);
    if (isFlatMirror) {
        float rl = luma(raw);
        float hd = clamp(rawTap.a, 0.0, 128.0);
        imageStore(specHistImg, px + currOff, roundHalf(vec4(raw, 1.0)));
        imageStore(specMomImg,  px + currOff, roundHalf(vec4(rl, 0.0, hd, rl)));
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
        filledTap = checkerboardCrossFill(colortex4, px, res, geoNormal, shadeNormal, depth, true, roughness);
    }
    float currLuma = luma(curr);

    // Gather outlier-proof median hitDist + interquartile spread + 9-sample spatial moments.
    vec3 locM1, locSigma;
    float smoothHitDist, distSpread;
    int locCount;
    gatherTracedSpecStats(px, res, isTraced, useChecker, curr, isTraced ? rawTap.a : filledTap.a,
                          geoNormal, shadeNormal, depth, locM1, locSigma, smoothHitDist, distSpread, locCount);

    // Previous-frame surface plane for exact 3D plane-distance verification.
    vec3 prevWorldPos = worldPos + cameraVelocity;
    vec3 prevSurfView = (gbufferPreviousModelView * vec4(prevWorldPos, 1.0)).xyz;
    vec3 prevViewGeoN = normalize(mat3(gbufferPreviousModelView) * geoNormal);
    float planeTol = max(0.06, 0.025 * max(-prevSurfView.z, 0.1));

    vec3 eyeToSurf = mat3(gbufferModelViewInverse) * viewPos;
    float camDist  = length(eyeToSurf);
    vec3 viewRay   = eyeToSurf / max(camDist, 1e-4);
    bool canVirtual = (smoothHitDist > 0.05);

    vec3 oldSlow = vec3(0.0);
    vec4 oldMom  = vec4(0.0); // (meanLuma, stdDev, prevHitDist, fastLuma)
    float oldAge = 0.0;
    float support = 0.0;
    bool usedVirtual = false;
    float parallaxPx = 0.0;

    if (frameCounter > 1) {
        vec2 prevJitter = getPrevTaaJitter();
        vec2 surfPrevUV = reprojectScreenUV(uv, worldPos);

        if (canVirtual) {
            vec3 virtualPos = worldPos + viewRay * smoothHitDist;
            vec2 virtPrevUV = reprojectScreenUV(uv, virtualPos);
            if (surfPrevUV.x >= 0.0 && virtPrevUV.x >= 0.0) {
                parallaxPx = length((virtPrevUV - surfPrevUV) * vec2(res));
            }
            // During real reflection parallax (>0.35px), enforce physical reflection-space hitDist
            // continuity to detect foreground/background disocclusions on the exact same floor plane!
            // Also require shadeNormal similarity (>=0.85) so tilted tile bevel grooves never bleed onto flat tiles.
            bool checkHitDist = (parallaxPx > 0.35);
            float hitDistTol = max(0.50, 0.45 * smoothHitDist) + 1.5 * distSpread + length(cameraVelocity) * 2.0;
            support = fetchPrevSpecHistory(virtPrevUV, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                                           geoNormal, shadeNormal, planeTol, 0.85,
                                           checkHitDist, smoothHitDist, hitDistTol, false, oldSlow, oldMom, oldAge);
            usedVirtual = (support > 1e-4);
        }

        if (!usedVirtual) {
            vec3 surfSlow;
            vec4 surfMom;
            float surfAge;
            float surfSup = fetchPrevSpecHistory(surfPrevUV, res, prevOff, prevJitter, prevSurfView, prevViewGeoN,
                                                 geoNormal, shadeNormal, planeTol, 0.80,
                                                 false, smoothHitDist, 0.0, true, surfSlow, surfMom, surfAge);
            if (surfSup > 1e-4) {
                // If virtual parallax shifted the reflection (>0.25px) and virtual lookup missed
                // (e.g., reflection-space disocclusion or edge of reflector), reset cleanly instead
                // of dragging surface-locked history across the floor!
                bool reflDisoccluded = canVirtual && (parallaxPx > 0.25);
                if (!reflDisoccluded) {
                    oldSlow = surfSlow; oldMom = surfMom; oldAge = surfAge; support = surfSup;
                }
            }
        }
    }

    // Exact geometric multi-depth parallax pixel-error bound (roughness-agnostic, FPS-independent!):
    // - Static camera (parallaxPx == 0, sinParallax == 0): full SPECULAR_MAX_HISTORY (zero noise!).
    // - Moving camera on single-depth reflection (distSpread ≈ 0): virtual reprojection is exact -> full smooth history!
    // - Moving camera on multi-depth reflection (e.g. door + wall + ceiling inside one GGX cone):
    //   near and far layers slide at different screen velocities -> cap history by exact pixel mismatch!
    vec3 currV = -viewRay;
    vec3 prevEyePos = -transpose(mat3(gbufferPreviousModelView)) * gbufferPreviousModelView[3].xyz;
    vec3 prevV = normalize(prevEyePos - prevWorldPos);
    float cosParallax = saturate(dot(currV, prevV));
    float sinParallax = sqrt(max(0.0, 1.0 - cosParallax * cosParallax));
    float sampleInterval = useChecker ? 2.0 : 1.0;

    float spreadRatio = clamp(distSpread / max(smoothHitDist, 0.25), 0.0, 2.0);
    float parallaxErrPx = parallaxPx * sampleInterval * (0.12 + 0.85 * spreadRatio);
    float viewDriftErr  = sinParallax * sampleInterval * 12.0;
    float totalMotionErr = max(parallaxErrPx, viewDriftErr);
    float maxHist = clamp(1.5 / max(totalMotionErr, 1e-4), 1.5, SPECULAR_MAX_HISTORY);
    oldAge = min(oldAge, maxHist);

    // Dual-EMA dynamic-reflection response (3.5-sigma guard keeps static multi-light reflections rock-solid
    // while adapting smoothly to dynamic light changes and opened doors!).
    float oldMeanL = oldMom.r;
    float oldStd   = oldMom.g;
    float oldHitD  = (oldMom.b > 0.02) ? oldMom.b : smoothHitDist;
    float oldFastL = oldMom.a;

    float sampleLuma = isTraced ? currLuma : ((locCount >= 2) ? luma(locM1) : luma(filledTap.rgb));
    const float specFastWindow = 8.0;
    float fastAlpha = 1.0 / min(max(oldAge + 1.0, 1.0), specFastWindow);
    float fastLuma  = (support > 1e-4) ? mix(oldFastL, sampleLuma, fastAlpha) : sampleLuma;

    if (support > 1e-4 && oldAge > specFastWindow) {
        float fastErr = oldStd * inversesqrt(2.0 * specFastWindow - 1.0);
        float diff = abs(fastLuma - oldMeanL);
        float noiseFloor = 3.5 * fastErr + 0.06;
        float relChange = diff / max(max(fastLuma, oldMeanL), noiseFloor);
        float sigRatio  = diff / max(noiseFloor, 1e-4);
        float resp = smoothstep(0.45, 0.80, relChange) * smoothstep(1.0, 2.0, sigRatio);
        if (resp > 0.01) {
            oldAge = mix(oldAge, min(oldAge, 3.0), resp);
        }
    }

    if (!isTraced) {
        if (support > 1e-4) {
            vec3 outCol = oldSlow;
            float outA  = oldAge;
            vec4 outMom = vec4(oldMeanL, oldStd, oldHitD, fastLuma);
            if ((parallaxPx > 0.15 || sinParallax > 1e-4) && locCount >= 2) {
                float uAge   = min(maxHist, oldAge + 1.0);
                float uAlpha = 1.0 / max(uAge, 1.0);
                outCol = mix(oldSlow, locM1, uAlpha);
                outA   = uAge;
                float uMean = mix(oldMeanL, luma(locM1), uAlpha);
                float uVar  = mix(oldStd * oldStd, (luma(locM1) - uMean) * (luma(locM1) - uMean), max(uAlpha, 0.15));
                float uHitD = mix(oldHitD, smoothHitDist, max(uAlpha, 0.25));
                outMom = vec4(uMean, sqrt(max(uVar, 0.0)), clamp(uHitD, 0.0, 128.0), fastLuma);
            }
            float spatStdDev = (locCount >= 2) ? luma(locSigma) : max(0.25 * luma(outCol), 0.05);
            float stdDev = mix(spatStdDev, outMom.g, smoothstep(1.0, 6.0, outA));
            imageStore(specHistImg, px + currOff, roundHalf(vec4(outCol, outA)));
            imageStore(specMomImg,  px + currOff, roundHalf(outMom));
            imageStore(scratchImg,  px,           roundHalf(vec4(outCol, stdDev)));
        } else {
            vec3 initCol = (locCount >= 2) ? locM1 : filledTap.rgb;
            float fl = luma(initCol);
            float stdDev = (locCount >= 2) ? max(luma(locSigma), 0.10 * fl) : max(0.20 * fl, 0.04);
            imageStore(specHistImg, px + currOff, roundHalf(vec4(initCol, 0.5)));
            imageStore(specMomImg,  px + currOff, roundHalf(vec4(fl, stdDev, clamp(smoothHitDist, 0.0, 128.0), fl)));
            imageStore(scratchImg,  px,           roundHalf(vec4(initCol, stdDev)));
        }
        return;
    }

    float age = min(maxHist, oldAge + 1.0);
    float alpha = 1.0 / max(age, 1.0);
    vec3 safeCurr = curr;
    if (support > 1e-4 && oldAge >= 3.0) {
        float refLuma = max(max(luma(oldSlow), luma(locM1)), fastLuma);
        float maxSpecLuma = max(refLuma * 4.5 + 0.4, 1.0);
        if (currLuma > maxSpecLuma) {
            safeCurr *= maxSpecLuma / max(currLuma, 1e-6);
        }
    }

    vec3 integrated;
    vec4 outMom;
    if (support <= 1e-4) {
        // Blend lightly with locM1 only on rough disoccluded pixels to suppress 1-SPP sparkle
        // without creating a pre-blurred rectangular screen-window border during camera rotation.
        integrated = (roughness >= 0.16 && locCount >= 3) ? mix(safeCurr, locM1, 0.5) : safeCurr;
        float initLuma = luma(integrated);
        float initStd  = (locCount >= 3) ? luma(locSigma) : max(0.20 * initLuma, 0.04);
        outMom = vec4(initLuma, initStd, clamp(smoothHitDist, 0.0, 128.0), initLuma);
    } else {
        float safeLuma = luma(safeCurr);
        integrated = mix(oldSlow, safeCurr, alpha);
        float newMean = mix(oldMeanL, safeLuma, alpha);
        float varAlpha = max(alpha, 1.0 / 16.0);
        float newVar  = mix(oldStd * oldStd, (safeLuma - newMean) * (safeLuma - newMean), varAlpha);
        float newHitD = mix(oldHitD, smoothHitDist, max(alpha, 0.25));
        outMom = vec4(newMean, sqrt(max(newVar, 0.0)), clamp(newHitD, 0.0, 128.0), fastLuma);
    }

    float spatStdDev = (locCount >= 3) ? luma(locSigma) : max(0.20 * luma(integrated), 0.04);
    float stdDev = mix(spatStdDev, outMom.g, smoothstep(1.0, 6.0, age));

    imageStore(specHistImg, px + currOff, roundHalf(vec4(integrated, age)));
    imageStore(specMomImg,  px + currOff, roundHalf(outMom));
    imageStore(scratchImg,  px,           roundHalf(vec4(integrated, stdDev)));
}
