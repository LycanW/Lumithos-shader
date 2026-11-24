#version 430 compatibility

// Pass: Karis-Tonemapped YCoCg TAA with 3x3 Closest-Depth Velocity Dilation & Exact Catmull-Rom
// Ping-pong taaHistImg (2x1 tiles) lets final.fsh read directly — eliminates composite1.csh copy pass!
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D taaHistTex;
layout(rgba16f) uniform writeonly image2D taaHistImg;

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

    // 3x3 closest-depth dilation keeps thin foreground silhouettes (beacon beams, edges) razor-sharp.
    float centerDepth = texelFetch(depthtex1, px, 0).r;
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
    vec3 mirrorGeoN = vec3(0.0, 1.0, 0.0);

    if (centerDepth < 0.56) {
        // First-person hand stays locked to camera (never reproject with world cameraVelocity!).
        motion = vec2(0.0);
        validMotion = true;
    } else if (bestDepth >= 0.999999) {
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
        vec3 worldPos = viewToWorld(screenToView(vec3(bestUV, bestDepth)));
        vec3 reprojTarget = worldPos;
        vec2 refUV = bestUV;

        // For interior pixels of smooth mirrors (check depth continuity rather than `bestPx == px`,
        // since tilted floors always have a closer neighbor!), reproject at exact planar virtual
        // reflection position `centerWorld + normalize(centerWorld) * hitDist`.
        if (centerDepth < 0.999999 && abs(centerDepth - bestDepth) < 0.002) {
            float rough;
            bool isHand;
            readGBufferRoughHand(px, rough, isHand);
            if (rough < 0.08 && !isHand) {
                vec3 geoN, shadeN;
                readGBufferNormals(px, geoN, shadeN);
                if (dot(shadeN, geoN) > 0.998) {
                    float hitDist = texelFetch(colortex4, px, 0).a;
                    if (hitDist > 0.05 && hitDist < 256.0) {
                        vec3 centerView  = screenToView(vec3(uv, centerDepth));
                        vec3 centerWorld = viewToWorld(centerView);
                        vec3 eyeRayDir   = normalize(mat3(gbufferModelViewInverse) * centerView);
                        reprojTarget = centerWorld + eyeRayDir * hitDist;
                        refUV = uv;
                        isMirrorInterior = true;
                        mirrorGeoN = geoN;
                    }
                }
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

    // Mirror-boundary disocclusion guard: if virtual reflection reprojection lands outside
    // the mirror surface (e.g. background sky or a different wall in prevGuideTex), reject history
    // immediately so outside sky/objects NEVER bleed across mirror edges when walking!
    if (validMotion && isMirrorInterior && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
        ivec2 pGuidePx = clamp(ivec2(prevUV * vec2(res)), ivec2(0), res - 1);
        vec4 pg = texelFetch(prevGuideTex, pGuidePx, 0);
        if (pg.w <= 0.0 || pg.z >= 0.08 || dot(mirrorGeoN, unpackNormal16(uint(round(pg.x)))) < 0.96) {
            validMotion = false;
        }
    }

    if (frameCounter > 1 && validMotion && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
        vec4 prevTap = fetchHistoryFast4(taaHistTex, prevUV, res, prevOff);
        vec3 prev = prevTap.rgb;
        float prevFrames = prevTap.a;
        if (!any(isnan(prev)) && !any(isinf(prev))) {
            // Linear HDR YCoCg 3x3 Moments + Ellipsoid Clipping + 32-frame Accumulation:
            // Computes moments in linear HDR YCoCg so bright sky/edge outliers keep variance ellipsoid open,
            // then blends in perceptual YCoCg space (ycc / (1 + Y)) with temporal-contrast anti-flicker!
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
            vec3 sigma = sqrt(max(m2 - m1 * m1, vec3(0.0)));

            float speedPx = length(motion * vec2(res));
            float camSpeed = length(cameraVelocity);
            float moveT = saturate(speedPx * 0.35);
            // Flat mirrors are noise-free deterministic rays: keep silky 12-frame AA when standing still,
            // but respond aggressively to walking / rotation so mirror reflections NEVER ghost!
            float mirrorMoveT = saturate(max(speedPx * 1.2, camSpeed * 25.0));

            // Ellipsoid intersection clipping (no hard [bMin, bMax] box chop that destroys subpixel holes!).
            float aggression = isMirrorInterior
                ? mix(1.85, 0.85, mirrorMoveT)
                : mix(2.2, 1.15, moveT);
            vec3 clipStdDev = max(sigma * aggression, vec3(1e-4));

            vec3 prevYcc = rgbToYCoCg(prev);
            vec3 delta = prevYcc - m1;
            vec3 unitD = delta / clipStdDev;
            float unitLenSq = dot(unitD, unitD);
            vec3 clippedYcc = (unitLenSq > 1.0) ? (m1 + delta * inversesqrt(unitLenSq)) : prevYcc;

            // Static soft-clipping prevents 1-frame snaps on subpixel leaf holes / grazing edges;
            // moving cameras & moving mirrors snap straight to 1.0 hard clipping!
            float effMove = isMirrorInterior ? mirrorMoveT : smoothstep(0.02, 0.25, speedPx);
            float clipStrength = mix(0.20, 1.0, effMove);
            prevYcc = mix(prevYcc, clippedYcc, clipStrength);

            // Subpixel sharpening during fractional-pixel motion (exact 0.0 when static!).
            vec2 fOff = fract(prevUV * vec2(res)) - 0.5;
            float subpixSqr = dot(fOff, fOff); // 0.0 at pixel center, up to 0.5 at corner
            prevYcc = mix(prevYcc, currYcc, subpixSqr * 0.5 * smoothstep(0.02, 0.35, speedPx));

            float currLum = max(currYcc.x, 0.0);
            float prevLum = max(prevYcc.x, 0.0);
            float temporalContrast = saturate(abs(currLum - prevLum) / max(max(currLum, prevLum), 1e-4));

            // Frame-count accumulation up to 32 frames when static (rock-solid!), decaying briskly on motion.
            float maxFrames = isMirrorInterior
                ? mix(12.0, 1.5, mirrorMoveT)
                : mix(32.0, 3.2, moveT);
            outFrames = min(prevFrames + 1.0, maxFrames);
            float antiFlicker = isMirrorInterior
                ? mix(0.6, 0.0, mirrorMoveT)
                : mix(1.5, 0.25, moveT);
            float blendWeight = outFrames * (1.0 + temporalContrast * temporalContrast * antiFlicker);

            vec3 prevP = prevYcc / (1.0 + max(prevYcc.x, 0.0));
            vec3 currP = currYcc / (1.0 + max(currYcc.x, 0.0));
            vec3 outP  = mix(prevP, currP, 1.0 / max(blendWeight, 1.0));
            vec3 outYcc = outP / max(1.0 - max(outP.x, 0.0), 1e-4);
            result = max(yCoCgToRgb(outYcc), vec3(0.0));
        }
    }

    imageStore(taaHistImg, px + currOff, roundHalf(vec4(result, outFrames)));
}
