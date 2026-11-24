#version 430 compatibility

// Pass: Karis-Tonemapped YCoCg TAA with 3x3 Closest-Depth Velocity Dilation & Exact Catmull-Rom
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform sampler2D taaHistTex;
layout(rgba16f) uniform writeonly image2D scratchImg;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    vec3 curr = max(texelFetch(colortex0, px, 0).rgb, vec3(0.0));
    if (any(isnan(curr)) || any(isinf(curr))) curr = vec3(0.0);

    #if !defined(TAA_ENABLED) || DEBUG_VIEW == 1 || DEBUG_VIEW == 5
        imageStore(scratchImg, px, roundHalf(vec4(curr, 1.0)));
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

    if (bestDepth >= 0.999999) {
        vec3 skyDir = normalize(viewToWorld(screenToView(vec3(uv, 1.0))));
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
                        vec3 centerWorld = viewToWorld(screenToView(vec3(uv, centerDepth)));
                        reprojTarget = centerWorld + normalize(centerWorld) * hitDist;
                        refUV = uv;
                        isMirrorInterior = true;
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

    if (frameCounter > 1 && validMotion && all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
        vec3 prev = fetchHistoryFast(taaHistTex, prevUV, res);
        if (!any(isnan(prev)) && !any(isinf(prev))) {
            // 3x3 Karis-tonemapped YCoCg variance + min/max clipping (prevents HDR bright-edge flicker).
            vec3 m1 = vec3(0.0);
            vec3 m2 = vec3(0.0);
            vec3 bMin = vec3( 1e6);
            vec3 bMax = vec3(-1e6);
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    ivec2 q = clamp(px + ivec2(x, y), ivec2(0), res - 1);
                    vec3 s = rgbToYCoCg(karisForward(max(texelFetch(colortex0, q, 0).rgb, vec3(0.0))));
                    m1 += s;
                    m2 += s * s;
                    bMin = min(bMin, s);
                    bMax = max(bMax, s);
                }
            }
            m1 /= 9.0;
            m2 /= 9.0;
            vec3 sigma = sqrt(max(m2 - m1 * m1, vec3(0.0)));
            float speedPx = length(motion * vec2(res));
            // Measure subpixel resampling misalignment (0 at exact texel center, 1 at half-texel):
            // prevents fractional-pixel drift from compounding blur across frames!
            vec2 fracPrev = abs(fract(prevUV * vec2(res) - 0.5) - 0.5);
            float subpix = 1.0 - max(fracPrev.x, fracPrev.y) * 2.0;
            float moveT = saturate(speedPx * 0.35);

            float gamma = isMirrorInterior
                ? mix(1.05, 0.72, moveT)
                : mix(1.35, 0.82, moveT);
            vec3 clipMin = max(bMin, m1 - gamma * sigma);
            vec3 clipMax = min(bMax, m1 + gamma * sigma);

            vec3 currK = karisForward(curr);
            vec3 prevYcc = clipAABB(clipMin, clipMax, rgbToYCoCg(karisForward(prev)));
            vec3 prevK = yCoCgToRgb(prevYcc);

            float alpha = isMirrorInterior
                ? mix(0.12, 0.50, moveT)
                : mix(0.09, 0.34, moveT) + 0.08 * subpix * smoothstep(0.05, 0.6, speedPx);
            result = karisInverse(mix(prevK, currK, saturate(alpha)));
        }
    }

    imageStore(scratchImg, px, roundHalf(vec4(max(result, vec3(0.0)), 1.0)));
}
