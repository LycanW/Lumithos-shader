#version 430 compatibility

// Pass: Spatial 3x3 Leave-One-Out Variance Estimation + RCRS Clamp + Perceptual Tonemap
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D scratchImg;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    ivec2 dstPx = px + ivec2(res.x, 0);
    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        imageStore(scratchImg, dstPx, vec4(0.0));
        return;
    }

    vec4 centerData = texelFetch(scratchTex, px, 0);
    vec3 rad = max(centerData.rgb, vec3(0.0));
    float age = max(centerData.a, 1.0);

    uint g1 = texelFetch(colortex2, px, 0).r;
    vec3 centerGeoN = unpackNormal16(g1 & 65535u);

    // Leave-One-Out geometry-aware 3x3 spatial moment estimation (strict linear view-space depth check
    // prevents foreground door/wall silhouettes from bleeding across window cutouts into background rooms!).
    float centerLinZ = linearizeDepth(depth);
    float depthTol = max(0.10, 0.06 * centerLinZ);
    vec2 moments = vec2(0.0);
    float wSum = 0.0;
    vec2 dMoments = vec2(0.0);
    float dWSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 q = clamp(px + ivec2(x, y), ivec2(0), res - 1);
            float qd = texelFetch(depthtex1, q, 0).r;
            if (qd >= 0.999999) continue;

            if (abs(linearizeDepth(qd) - centerLinZ) > depthTol) continue;
            float ql = luma(max(texelFetch(scratchTex, q, 0).rgb, vec3(0.0)));
            dMoments += vec2(ql, ql * ql);
            dWSum += 1.0;

            uint qg1 = texelFetch(colortex2, q, 0).r;
            vec3 qGeoN = unpackNormal16(qg1 & 65535u);
            if (dot(centerGeoN, qGeoN) < 0.85) continue;

            moments += vec2(ql, ql * ql);
            wSum += 1.0;
        }
    }
    float cLuma = luma(rad);
    if (wSum >= 3.0) {
        moments /= wSum;
    } else if (dWSum >= 2.0) {
        moments = dMoments / dWSum;
        wSum = max(dWSum, 3.0);
    } else if (wSum > 0.5) {
        moments /= wSum;
    } else {
        moments = vec2(cLuma, cLuma * cLuma);
    }

    float sigma = sqrt(max(moments.y - moments.x * moments.x, 0.0));
    sigma += (moments.x + 0.025) / max(exp2(min(age * 0.45, 10.0)) - 1.0, 0.25);

    if (wSum >= 3.0) {
        float margin = max(2.25 * sigma, 0.02 + 0.15 * moments.x);
        float clampedLuma = clamp(cLuma, max(moments.x - margin, 0.0), moments.x + margin);
        if (cLuma > 1e-5) {
            rad *= clampedLuma / cLuma;
        }
    }

    float tonemapDenom = 1.0 + 0.5 * luma(rad);
    rad /= tonemapDenom;
    float sigmaScaled = max(sigma / tonemapDenom, 0.004);

    imageStore(scratchImg, dstPx, roundHalf(vec4(rad, sigmaScaled)));
}
