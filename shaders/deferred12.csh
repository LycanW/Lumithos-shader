#version 430 compatibility

// Pass: Specular SVGF Temporal Accumulation (Ping-Pong Race-Free + Roughness-Aware Despike)
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D specHistImg;
layout(rg32f)   uniform writeonly image2D specMomImg;
layout(rgba16f) uniform writeonly image2D scratchImg;

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

    GBufferMat mat = readGBuffer(px);
    if (mat.roughness > SPECULAR_ROUGH_CUTOFF || mat.isHand) {
        imageStore(scratchImg, px, vec4(0.0));
        imageStore(specHistImg, px + currOff, vec4(0.0));
        imageStore(specMomImg,  px + currOff, vec4(0.0));
        return;
    }

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);
    float expectedPrevDepth = -(gbufferPreviousModelView * vec4(worldPos + cameraVelocity, 1.0)).z;

    vec4 rawTap = texelFetch(colortex4, px, 0);
    vec3 raw = max(rawTap.rgb, vec3(0.0));
    if (any(isnan(raw)) || any(isinf(raw))) raw = vec3(0.0);

    // Pure FLAT deterministic mirrors (roughness < 0.08 & no normal-map bump) have zero sampling or
    // normal-map TAA-jitter noise: bypass temporal accumulation completely for zero ghosting,
    // and preserve hitDist in .a so TAA can reproject at exact planar virtual parallax!
    // Bumpy normal-mapped glossy surfaces run gentle temporal stabilization so normal-map facets don't sparkle.
    bool isFlatMirror = (mat.roughness < 0.08) && (dot(mat.shadeNormal, mat.geoNormal) > 0.998);
    if (isFlatMirror) {
        imageStore(specHistImg, px + currOff, roundHalf(vec4(raw, 1.0)));
        imageStore(specMomImg,  px + currOff, vec4(luma(raw), luma(raw) * luma(raw), 0.0, 0.0));
        imageStore(scratchImg,  px,           roundHalf(vec4(raw, rawTap.a)));
        return;
    }

    // Roughness-scaled 3x3 firefly despike for ALL stochastic GGX surfaces (0.08..0.50):
    // Prevents isolated 1-SPP lantern hits on polished glossy walls/floors from spawning glowing blotches!
    float roughT = smoothstep(0.08, 0.35, mat.roughness);
    vec3 curr = despike3x3(colortex4, px, res, raw, mat.geoNormal, mat.shadeNormal, depth,
                           mix(5.5, 2.8, roughT), mix(0.22, 0.04, roughT));
    float currLuma = luma(curr);

    // Virtual reflection depth reprojection for smooth mirrors, surface reprojection for rough.
    vec3 viewDir = normalize(worldPos);
    vec3 reprojPos = mix(worldPos + viewDir * min(rawTap.a, 64.0), worldPos, smoothstep(0.12, 0.35, mat.roughness));

    vec3 oldSlow = vec3(0.0);
    vec2 oldMom  = vec2(0.0);
    float oldAge = 0.0;
    float support = 0.0;
    float speedPx = 0.0;

    if (frameCounter > 1) {
        vec2 prevUV = reprojectScreenUV(uv, reprojPos);
        if (all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThan(prevUV, vec2(1.0)))) {
            speedPx = length((prevUV - uv) * vec2(res));
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

                vec3 pGeoN = unpackNormal16(uint(round(pg.x)));
                if (dot(mat.geoNormal, pGeoN) < 0.94) continue;
                if (abs(expectedPrevDepth - pDepth) > max(0.08, 0.05 * max(expectedPrevDepth, 0.1))) continue;
                if (abs(mat.roughness - pg.z) > max(0.06, 0.25 * max(mat.roughness, pg.z))) continue;

                vec4 hs = texelFetch(specHistTex, tap + prevOff, 0);
                vec2 hm = texelFetch(specMomTex,  tap + prevOff, 0).rg;
                oldSlow += w * hs.rgb;
                oldAge  += w * hs.a;
                oldMom  += w * hm;
                support += w;
            }
        }
    }

    if (support > 1e-4) {
        oldSlow /= support;
        oldAge  /= support;
        oldMom  /= support;

        // Motion-gated 3x3 neighborhood clamp:
        // Only clamp converged glossy history during actual camera/reflection motion so stationary
        // 1-SPP glossy reflections converge rock-solid without flickering or popping bright spots!
        float motionClampW = smoothstep(0.20, 1.75, speedPx);
        if (motionClampW > 0.01) {
            vec3 nMin = curr;
            vec3 nMax = curr;
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    if (x == 0 && y == 0) continue;
                    ivec2 q = clamp(px + ivec2(x, y), ivec2(0), res - 1);
                    vec3 s = max(texelFetch(colortex4, q, 0).rgb, vec3(0.0));
                    nMin = min(nMin, s);
                    nMax = max(nMax, s);
                }
            }
            vec3 margin = max((nMax - nMin) * 0.5, vec3(0.05));
            vec3 clampedHist = clamp(oldSlow, max(nMin - margin, vec3(0.0)), nMax + margin);
            oldSlow = mix(oldSlow, clampedHist, motionClampW);
        }
    }

    float moveDecay = exp(-length(cameraVelocity) * (mat.roughness < 0.20 ? 5.0 : 2.0));
    oldAge *= moveDecay;

    // Soft temporal firefly guard on settled glossy pixels suppresses stray 1-SPP lantern spikes.
    vec3 safeCurr = curr;
    if (support > 1e-4 && oldAge >= 8.0 && speedPx < 0.75) {
        float maxSpecLuma = max(luma(oldSlow) * 6.0 + 0.6, 1.5);
        if (currLuma > maxSpecLuma) {
            safeCurr *= maxSpecLuma / max(currLuma, 1e-6);
        }
    }

    float maxHist = (mat.roughness < 0.08) ? 10.0 : SPECULAR_MAX_HISTORY;
    float age = min(maxHist, oldAge + 1.0);
    float alpha = 1.0 / max(age, 1.0);
    float safeLuma = luma(safeCurr);
    vec2 moments = mix(oldMom, vec2(safeLuma, safeLuma * safeLuma), alpha);
    vec3 integrated = mix(oldSlow, safeCurr, alpha);
    float stdDev = sqrt(max(moments.y - moments.x * moments.x, 0.0));

    imageStore(specHistImg, px + currOff, roundHalf(vec4(integrated, age)));
    imageStore(specMomImg,  px + currOff, vec4(moments, 0.0, 0.0));
    imageStore(scratchImg,  px,           roundHalf(vec4(integrated, stdDev)));
}
