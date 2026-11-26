#version 430 compatibility

// Pass: Clean YCoCg Ellipsoid TAA with 3x3 Closest-Depth Velocity Dilation, Exact Catmull-Rom,
// Planar Mirror Virtual Reprojection, and Geometry-Aware Disocclusion Reset.
// Ping-pong taaHistImg (2x1 tiles) lets final.fsh read directly — zero copy pass!
#include "/lib/Reprojection.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/Water.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D taaHistTex;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
layout(rgba16f) uniform writeonly image2D taaHistImg;

const float TAA_MAX_FRAMES  = 32.0;
const float TAA_CLIP_GAMMA  = 2.0;
const float TAA_ANTIFLICKER = 0.25;

// Resolve translucent layers over the opaque scene: bg(refracted) * T + L,
// plus true submerged underwater volume fog when camera is inside water (without double-fogging Snell's window!).
vec3 fetchSceneColor(ivec2 q, ivec2 res, vec3 uwLightDir, vec3 uwDirectCol, vec3 uwSkyUp) {
    vec3 bg = texelFetch(colortex0, q, 0).rgb;
    #if DEBUG_VIEW == 0
    vec3 T  = texelFetch(colortex7, q, 0).rgb;
    vec3 L  = texelFetch(colortex6, q, 0).rgb;
    if (minOf(T) < 0.9999 || maxOf(L) > 0.0) {
        vec2 off = texelFetch(colortex8, q, 0).rg;
        if (dot(off, off) > 1e-10) {
            ivec2 rq = clamp(q + ivec2(round(off * vec2(res))), ivec2(0), res - 1);
            float surfDepth = texelFetch(depthtex0, q, 0).r;
            if (texelFetch(depthtex1, rq, 0).r > surfDepth && texelFetch(colortex7, rq, 0).r < 0.9999) {
                bg = texelFetch(colortex0, rq, 0).rgb;
            }
        }
        bg = bg * T + L;
    } else if (isEyeInWater == 1) {
        vec2 qUV = (vec2(q) + 0.5) * texelSize();
        float d1 = texelFetch(depthtex1, q, 0).r;
        vec3 worldDir;
        float waterDist;
        if (d1 >= 0.999999) {
            worldDir = normalize(mat3(gbufferModelViewInverse) * screenToView(vec3(qUV, 1.0)));
            waterDist = 48.0;
        } else {
            vec3 vp = screenToView(vec3(qUV, d1));
            waterDist = length(vp);
            worldDir = normalize(mat3(gbufferModelViewInverse) * vp);
        }
        vec3 Tvol, inscatter;
        evalWaterVolume(waterDist, worldDir, vec3(0.0, 1.0, 0.0), uwLightDir, uwDirectCol,
                        vec3(1.0), uwSkyUp, 1.0, vec3(0.20, 0.55, 0.85), true, Tvol, inscatter);
        bg = bg * Tvol + inscatter;
    }
    #endif
    bg = max(bg, vec3(0.0));
    if (any(isnan(bg)) || any(isinf(bg))) bg = vec3(0.0);
    return bg;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    int currSide = frameCounter & 1;
    int prevSide = currSide ^ 1;
    ivec2 currOff = ivec2(currSide * res.x, 0);
    ivec2 prevOff = ivec2(prevSide * res.x, 0);

    vec3 uwLightDir  = (isEyeInWater == 1) ? getShadowLightDir() : vec3(0.0, 1.0, 0.0);
    vec3 uwDirectCol = (isEyeInWater == 1) ? getDirectLightColor() : vec3(0.0);
    vec3 uwSkyUp     = (isEyeInWater == 1) ? evalSky(vec3(0.0, 1.0, 0.0), false) : vec3(0.0);

    vec3 curr = fetchSceneColor(px, res, uwLightDir, uwDirectCol, uwSkyUp);

    #if !defined(TAA_ENABLED) || DEBUG_VIEW == 1 || DEBUG_VIEW == 5 || DEBUG_VIEW == 6
        imageStore(taaHistImg, px + currOff, roundHalf(vec4(curr, 1.0)));
        return;
    #endif

    float centerDepth = texelFetch(depthtex1, px, 0).r;
    float centerDepth0 = texelFetch(depthtex0, px, 0).r;
    bool isTranslucent = centerDepth0 < centerDepth - 1e-7;
    float centerRough = 1.0, centerF0 = 0.0;
    float frameLimit = TAA_MAX_FRAMES;
    bool centerHand = false;
    vec3 centerGeoN = vec3(0.0, 1.0, 0.0);
    vec3 centerShadeN = vec3(0.0, 1.0, 0.0);

    if (centerDepth < 0.999999) {
        readGBufferSpecMask(px, centerRough, centerF0, centerHand);
        // First-person hand has no world motion vector and swings rapidly -> bypass TAA cleanly (zero hand ghosting!).
        if (centerHand && !isTranslucent) {
            imageStore(taaHistImg, px + currOff, roundHalf(vec4(curr, 1.0)));
            return;
        }
        readGBufferNormals(px, centerGeoN, centerShadeN);
    }

    // 5-tap diagonal closest-depth dilation (including translucent surfaces) keeps thin silhouettes sharp.
    float bestDepth = centerDepth0;
    ivec2 bestPx = px;
    const ivec2 diagOffs[4] = ivec2[4](ivec2(1, 1), ivec2(-1, 1), ivec2(1, -1), ivec2(-1, -1));
    for (int i = 0; i < 4; ++i) {
        ivec2 q = clamp(px + diagOffs[i], ivec2(0), res - 1);
        float d = texelFetch(depthtex0, q, 0).r;
        if (d < bestDepth) {
            bestDepth = d;
            bestPx = q;
        }
    }

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec2 bestUV = (vec2(bestPx) + 0.5) * texelSize();
    vec2 motion = vec2(0.0);
    bool validMotion = false;

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
        vec3 centerWorld = (centerDepth < 0.999999) ? viewToWorld(centerView) : vec3(0.0);

        vec3 reprojTarget = viewToWorld(screenToView(vec3(bestUV, bestDepth)));
        vec2 refUV = bestUV;

        // Exact planar virtual reflection reprojection for flat mirrors (using actual bobbed eye ray).
        if (!isTranslucent && centerDepth < 0.999999 && abs(centerDepth - bestDepth) < 0.002
            && centerRough < 0.08 && dot(centerShadeN, centerGeoN) > 0.998) {
            float hitDist = texelFetch(colortex4, px, 0).a;
            if (hitDist > 0.05 && hitDist < 256.0) {
                vec3 eyeRayDir = normalize(mat3(gbufferModelViewInverse) * centerView);
                reprojTarget = centerWorld + eyeRayDir * hitDist;
                refUV = uv;
            }
        }

        #ifdef SPECULAR_REFLECTIONS
        if (!isTranslucent && isLabPBRMetal(centerF0) && centerRough >= 0.08 && any(notEqual(cameraVelocity, vec3(0.0)))) {
            vec3 v = mat3(gbufferModelView) * cameraVelocity;
            vec2 offset = v.xy - v.z * centerView.xy / centerView.z;
            vec2 focalPx = 0.5 * vec2(res) * vec2(gbufferProjection[0][0], gbufferProjection[1][1]);
            float parallaxPx = length(focalPx * offset) / max(-centerView.z, 1e-4);
            float footprintPx = max(1.0, 0.25 * focalPx.y * centerRough * centerRough);
            frameLimit = min(frameLimit, 1.0 + footprintPx / max(parallaxPx, 1e-4));
        }
        #endif

        vec2 prevAtRef = reprojectScreenUV(refUV, reprojTarget);
        if (prevAtRef.x >= 0.0) {
            motion = prevAtRef - refUV;
            validMotion = true;
        }
    }

    vec2 prevUV = uv + motion;
    vec3 result = curr;
    float outFrames = 1.0;

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
                    vec3 s = rgbToYCoCg(fetchSceneColor(q, res, uwLightDir, uwDirectCol, uwSkyUp));
                    m1 += s;
                    m2 += s * s;
                }
            }
            m1 *= (1.0 / 9.0);
            m2 *= (1.0 / 9.0);
            vec3 prevYcc = rgbToYCoCg(prev);
            vec3 delta = prevYcc - m1;
            float clipFloor = 1e-4;
            bool limitClippedHistory = frameLimit < TAA_MAX_FRAMES;
            #ifdef SPECULAR_REFLECTIONS
            if (!isTranslucent && isLabPBRMetal(centerF0) && centerRough >= 0.16
                && all(equal(cameraVelocity, vec3(0.0))) && length(motion * vec2(res)) < 0.001) {
                // A flat, noisy patch can have almost zero spatial variance despite changing ray samples.
                float noise = 0.12 * sqrt(max(m1.x, 0.01));
                float deviation = maxOf(abs(delta)) / noise;
                clipFloor = max(clipFloor, noise * (1.0 - smoothstep(1.0, 3.0, deviation)));
                limitClippedHistory = deviation > 3.0;
            }
            #endif
            vec3 clipStdDev = max(sqrt(max(m2 - m1 * m1, vec3(0.0))) * TAA_CLIP_GAMMA, vec3(clipFloor));
            vec3 unitD = delta / clipStdDev;
            float unitLenSq = dot(unitD, unitD);
            if (unitLenSq > 1.0) {
                float retained = inversesqrt(unitLenSq);
                prevYcc = m1 + delta * retained;
                if (limitClippedHistory)
                    frameLimit = min(frameLimit, 1.0 / max(1.0 - retained, 1e-5));
            }

            // Subpixel sharpening during fractional-pixel resampling (mathematically 0.0 when static).
            vec2 fOff = fract(prevUV * vec2(res)) - 0.5;
            prevYcc = mix(prevYcc, currYcc, dot(fOff, fOff) * 0.5);

            float currLum = max(currYcc.x, 0.0);
            float prevLum = max(prevYcc.x, 0.0);
            float temporalContrast = saturate(abs(currLum - prevLum) / max(max(currLum, prevLum), 1e-4));

            outFrames = min(prevFrames + 1.0, frameLimit);
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
