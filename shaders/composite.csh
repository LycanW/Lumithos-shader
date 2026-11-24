#version 430 compatibility

// Pass: Clean YCoCg Ellipsoid TAA with 3x3 Closest-Depth Velocity Dilation, Exact Catmull-Rom,
// Planar Mirror Virtual Reprojection, and Geometry-Aware Disocclusion Reset.
// Ping-pong taaHistImg (2x1 tiles) lets final.fsh read directly — zero copy pass!
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D taaHistTex;
layout(rgba16f) uniform writeonly image2D taaHistImg;

const float TAA_MAX_FRAMES  = 16.0;
const float TAA_CLIP_GAMMA  = 1.5;
const float TAA_ANTIFLICKER = 0.25;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    int currSide = frameCounter & 1;
    int prevSide = currSide ^ 1;
    ivec2 currOff = ivec2(currSide * res.x, 0);
    ivec2 prevOff = ivec2(prevSide * res.x, 0);

    vec3 curr = max(texelFetch(colortex0, px, 0).rgb, vec3(0.0));
    if (any(isnan(curr)) || any(isinf(curr))) curr = vec3(0.0);

    #if !defined(TAA_ENABLED) || DEBUG_VIEW == 1 || DEBUG_VIEW == 5
        imageStore(taaHistImg, px + currOff, roundHalf(vec4(curr, 1.0)));
        return;
    #endif

    float centerDepth = texelFetch(depthtex1, px, 0).r;
    float centerRough = 1.0;
    bool centerHand = false;
    vec3 centerGeoN = vec3(0.0, 1.0, 0.0);
    vec3 centerShadeN = vec3(0.0, 1.0, 0.0);

    if (centerDepth < 0.999999) {
        readGBufferRoughHand(px, centerRough, centerHand);
        // First-person hand has no world motion vector and swings rapidly -> bypass TAA cleanly (zero hand ghosting!).
        if (centerHand) {
            imageStore(taaHistImg, px + currOff, roundHalf(vec4(curr, 1.0)));
            return;
        }
        readGBufferNormals(px, centerGeoN, centerShadeN);
    }

    // 3x3 closest-depth dilation keeps thin foreground silhouettes sharp.
    float bestDepth = centerDepth;
    ivec2 bestPx = px;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 q = clamp(px + ivec2(x, y), ivec2(0), res - 1);
            float d = texelFetch(depthtex1, q, 0).r;
            if (d < bestDepth) {
                bestDepth = d;
                bestPx = q;
            }
        }
    }

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec2 bestUV = (vec2(bestPx) + 0.5) * texelSize();
    vec2 motion = vec2(0.0);
    bool validMotion = false;
    bool isMirrorInterior = false;
    vec3 centerWorld = vec3(0.0);

    if (bestDepth >= 0.999999) {
        vec3 skyDir = normalize(mat3(gbufferModelViewInverse) * screenToView(vec3(uv, 1.0)));
        vec4 currClip = gbufferProjection * vec4(mat3(gbufferModelView) * skyDir, 1.0);
        vec4 prevClip = gbufferPreviousProjection * vec4(mat3(gbufferPreviousModelView) * skyDir, 1.0);
        if (currClip.w > 0.0 && prevClip.w > 0.0) {
            vec2 currUnjit = (currClip.xy / currClip.w) * 0.5 + 0.5;
            vec2 prevUnjit = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
            motion = prevUnjit - currUnjit;
            validMotion = true;
        }
    } else {
        vec3 centerView = (centerDepth < 0.999999) ? screenToView(vec3(uv, centerDepth)) : vec3(0.0);
        centerWorld = (centerDepth < 0.999999) ? viewToWorld(centerView) : vec3(0.0);

        vec3 reprojTarget = viewToWorld(screenToView(vec3(bestUV, bestDepth)));
        vec2 refUV = bestUV;

        // Exact planar virtual reflection reprojection for flat mirrors (using actual bobbed eye ray).
        if (centerDepth < 0.999999 && abs(centerDepth - bestDepth) < 0.002
            && centerRough < 0.08 && dot(centerShadeN, centerGeoN) > 0.998) {
            float hitDist = texelFetch(colortex4, px, 0).a;
            if (hitDist > 0.05 && hitDist < 256.0) {
                vec3 eyeRayDir = normalize(mat3(gbufferModelViewInverse) * centerView);
                reprojTarget = centerWorld + eyeRayDir * hitDist;
                refUV = uv;
                isMirrorInterior = true;
            }
        }

        vec2 prevAtRef = reprojectScreenUV(refUV, reprojTarget);
        if (prevAtRef.x >= 0.0) {
            motion = prevAtRef - refUV;
            validMotion = true;
        }
    }

    vec2 prevUV = uv + motion;
    vec3 result = curr;
    float outFrames = 1.0;

    // Geometry-guided disocclusion check using jitter-aligned prevGuideTex:
    // - Mirror interior: reject if previous pixel was outside this mirror plane (prevents sky bleed across mirror edges).
    // - Solid surface interior (bestPx == px): check 2x2 around jitter-aligned prevGuideUV so thin iron bars
    //   never falsely reset from subpixel jitter while placed/broken blocks still reset immediately!
    if (validMotion && centerDepth < 0.999999 && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
        vec2 alignedPrevUV = prevUV + 0.5 * (getPrevTaaJitter() - taaJitter);
        if (isMirrorInterior) {
            ivec2 pGuidePx = clamp(ivec2(alignedPrevUV * vec2(res)), ivec2(0), res - 1);
            vec4 pg = texelFetch(prevGuideTex, pGuidePx, 0);
            if (pg.w <= 0.0 || pg.z >= 0.08 || dot(centerGeoN, unpackNormal16(uint(round(pg.x)))) < 0.96) {
                validMotion = false;
            }
        } else if (bestPx == px) {
            float expectedPrevDepth = -(gbufferPreviousModelView * vec4(centerWorld + cameraVelocity, 1.0)).z;
            float depthTol = max(0.25, 0.15 * max(expectedPrevDepth, 0.1));
            ivec2 gBase = ivec2(floor(alignedPrevUV * vec2(res) - 0.5));
            bool foundMatch = false;
            for (int j = 0; j < 4; ++j) {
                ivec2 gp = clamp(gBase + ivec2(j & 1, j >> 1), ivec2(0), res - 1);
                vec4 pg = texelFetch(prevGuideTex, gp, 0);
                if (pg.w > 0.0
                    && abs(expectedPrevDepth - pg.w) <= depthTol
                    && dot(centerGeoN, unpackNormal16(uint(round(pg.x)))) >= 0.80) {
                    foundMatch = true;
                    break;
                }
            }
            if (!foundMatch) {
                validMotion = false;
            }
        }
    }

    if (frameCounter > 1 && validMotion && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
        vec4 prevTap = fetchHistoryFast4(taaHistTex, prevUV, res, prevOff);
        vec3 prev = prevTap.rgb;
        float prevFrames = prevTap.a;
        if (!any(isnan(prev)) && !any(isinf(prev))) {
            // 3x3 linear HDR YCoCg moments + strict 100% ellipsoid clipping.
            vec3 currYcc = rgbToYCoCg(curr);
            vec3 m1 = currYcc;
            vec3 m2 = currYcc * currYcc;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    if (x == 0 && y == 0) continue;
                    ivec2 q = clamp(px + ivec2(x, y), ivec2(0), res - 1);
                    vec3 s = rgbToYCoCg(max(texelFetch(colortex0, q, 0).rgb, vec3(0.0)));
                    m1 += s;
                    m2 += s * s;
                }
            }
            m1 *= (1.0 / 9.0);
            m2 *= (1.0 / 9.0);
            vec3 clipStdDev = max(sqrt(max(m2 - m1 * m1, vec3(0.0))) * TAA_CLIP_GAMMA, vec3(1e-4));

            vec3 prevYcc = rgbToYCoCg(prev);
            vec3 delta = prevYcc - m1;
            vec3 unitD = delta / clipStdDev;
            float unitLenSq = dot(unitD, unitD);
            if (unitLenSq > 1.0) {
                prevYcc = m1 + delta * inversesqrt(unitLenSq);
            }

            // Subpixel sharpening during fractional-pixel resampling (mathematically 0.0 when static).
            vec2 fOff = fract(prevUV * vec2(res)) - 0.5;
            prevYcc = mix(prevYcc, currYcc, dot(fOff, fOff) * 0.5);

            float currLum = max(currYcc.x, 0.0);
            float prevLum = max(prevYcc.x, 0.0);
            float temporalContrast = saturate(abs(currLum - prevLum) / max(max(currLum, prevLum), 1e-4));

            outFrames = min(prevFrames + 1.0, TAA_MAX_FRAMES);
            float blendWeight = outFrames * (1.0 + temporalContrast * temporalContrast * TAA_ANTIFLICKER);

            // Perceptual YCoCg blend.
            vec3 prevP = prevYcc / (1.0 + max(prevYcc.x, 0.0));
            vec3 currP = currYcc / (1.0 + max(currYcc.x, 0.0));
            vec3 outP  = mix(prevP, currP, 1.0 / max(blendWeight, 1.0));
            vec3 outYcc = outP / max(1.0 - max(outP.x, 0.0), 1e-4);
            result = max(yCoCgToRgb(outYcc), vec3(0.0));
        }
    }

    imageStore(taaHistImg, px + currOff, roundHalf(vec4(result, outFrames)));
}
