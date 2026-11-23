#version 430 compatibility

// Pass: Spatial 3x3 Variance Estimation + RCRS Clamp + Perceptual Tonemap
// Inspired by Revoxelation (VarianceEstimate) & iterationRP (DiffuseVarianceEstimation).
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

    // Geometry-aware 3x3 (stride 2) spatial moment estimation captures residual blotch variance.
    vec2 moments = vec2(0.0);
    float wSum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 q = clamp(px + ivec2(x, y) * 2, ivec2(0), res - 1);
            float qd = texelFetch(depthtex1, q, 0).r;
            if (qd >= 0.999999) continue;
            uint qg1 = texelFetch(colortex2, q, 0).r;
            vec3 qGeoN = unpackNormal16(qg1 & 65535u);
            if (dot(centerGeoN, qGeoN) < 0.85) continue;

            float ql = luma(max(texelFetch(scratchTex, q, 0).rgb, vec3(0.0)));
            moments += vec2(ql, ql * ql);
            wSum += 1.0;
        }
    }
    moments /= max(wSum, 1.0);

    float sigma = sqrt(max(moments.y - moments.x * moments.x, 0.0));
    // Boost spatial filter width on newly disoccluded / low-history pixels.
    sigma += (moments.x + 0.025) / max(exp2(min(age * 0.45, 10.0)) - 1.0, 0.25);

    // Gentle RCRS variance clamp removes isolated bright/dark outliers before large-radius filtering.
    float cLuma = luma(rad);
    float margin = max(2.5 * sigma, 0.015);
    float clampedLuma = clamp(cLuma, max(moments.x - margin, 0.0), moments.x + margin);
    if (cLuma > 1e-5) {
        rad *= clampedLuma / cLuma;
    }

    // Perceptual compression before spatial filtering (inverted at end of pass 7).
    float tonemapDenom = 1.0 + 0.5 * luma(rad);
    rad /= tonemapDenom;
    float sigmaScaled = max(sigma / tonemapDenom, 0.004);

    imageStore(scratchImg, dstPx, roundHalf(vec4(rad, sigmaScaled)));
}
