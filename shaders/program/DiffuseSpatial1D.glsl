// 1D Separable 5-Tap Edge-Avoiding Filter (4 passes = 16 taps, 2x fewer full-screen passes & barriers!)
// Covers +-32..48 pixels across 4 orthogonal/diagonal passes with grazing-angle foreshortening and blue-noise jitter.
#include "/lib/SVGF.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#ifdef DIFFUSE_1D_WRITE_FINAL
    layout(rgba16f) uniform writeonly image2D colorimg3;
#else
    layout(rgba16f) uniform writeonly image2D scratchImg;
#endif

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        #ifdef DIFFUSE_1D_WRITE_FINAL
            imageStore(colorimg3, px, vec4(0.0));
        #else
            imageStore(scratchImg, px + ivec2(DIFFUSE_1D_DST_TILE * res.x, 0), vec4(0.0));
        #endif
        return;
    }

    ivec2 srcOff = ivec2(DIFFUSE_1D_SRC_TILE * res.x, 0);
    vec4 center = texelFetch(scratchTex, px + srcOff, 0);
    float centerLuma = luma(center.rgb);
    float centerSigma = max(center.a, 0.004);

    // Fetch ONLY colortex2 (normals) + depthtex1 — zero colortex1 albedo bandwidth!
    vec3 centerGeoN, centerShadeN;
    readGBufferNormals(px, centerGeoN, centerShadeN);

    vec2 tSize = texelSize();
    vec2 centerNdc = ((vec2(px) + 0.5) * tSize) * 2.0 - 1.0;
    #ifdef TAA_ENABLED
        centerNdc -= taaJitter;
    #endif
    vec3 viewPos = fastScreenToViewNdc(centerNdc, depth);
    vec3 viewGeoN = normalize(mat3(gbufferModelView) * centerGeoN);

    // Perspective-projected foreshortening shrinks steps along grazing surfaces.
    float absNdotV = abs(dot(viewGeoN, normalize(-viewPos)));
    vec2 axis = DIFFUSE_1D_AXIS;
    vec2 offsetScale = 1.0 - abs(viewGeoN.xy) * (1.0 - absNdotV);

    // Spatiotemporal blue-noise jitter breaks fixed-grid ringing and A-Trous waves.
    float jitter = stbn2D(px, frameCounter + DIFFUSE_1D_PASS * 7, 11u).x;
    ivec2 baseStep = ivec2(round(axis * offsetScale * (0.75 + 0.50 * jitter)));
    if (baseStep == ivec2(0)) {
        baseStep = ivec2(sign(axis));
    }
    vec2 stepNdc = vec2(baseStep) * (tSize * 2.0);

    float planeTol = max(0.035, 0.008 * (-viewPos.z));
    float lumaWeightScale = (DIFFUSE_1D_VAR_SCALE > 0.0)
        ? (DIFFUSE_1D_VAR_SCALE / max(centerSigma * 5.0 + 0.015, 0.015))
        : 0.0;

    vec4 sum = center;
    float sumW = 1.0;

    // 4 neighbor taps: +1, -1 (weight 0.85), +2, -2 (weight 0.55) along the axis!
    const int mults[4] = int[4](1, -1, 2, -2);
    const float kW[4]  = float[4](0.85, 0.85, 0.55, 0.55);

    for (int i = 0; i < 4; ++i) {
        int m = mults[i];
        ivec2 q = px + baseStep * m;
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;

        float qd = texelFetch(depthtex1, q, 0).r;
        if (qd >= 0.999999) continue;

        vec3 qGeoN, qShadeN;
        readGBufferNormals(q, qGeoN, qShadeN);
        if (dot(centerGeoN, qGeoN) < 0.90) continue;

        vec3 qViewPos = fastScreenToViewNdc(centerNdc + stepNdc * float(m), qd);
        float planeDist = abs(dot(qViewPos - viewPos, viewGeoN));
        if (planeDist > planeTol) continue;

        float sDot = max(dot(centerShadeN, qShadeN), 0.0);
        float s2 = sDot * sDot;
        float s4 = s2 * s2;
        float wNorm = s4 * s4; // exact sDot^8 without exp2/log2!
        if (wNorm < 1e-4) continue;

        vec4 tap = texelFetch(scratchTex, q + srcOff, 0);
        float wLuma = (lumaWeightScale > 0.0)
            ? exp(-abs(luma(tap.rgb) - centerLuma) * lumaWeightScale)
            : 1.0;

        float w = kW[i] * wNorm * wLuma;
        sum  += tap * w;
        sumW += w;
    }

    vec4 outVal = sum / sumW;

    #ifdef DIFFUSE_1D_WRITE_FINAL
        // Invert perceptual compression back to linear HDR irradiance.
        float outLuma = luma(max(outVal.rgb, vec3(0.0)));
        vec3 linearRad = max(outVal.rgb, vec3(0.0)) / max(1.0 - 0.5 * outLuma, 0.05);
        imageStore(colorimg3, px, roundHalf(vec4(linearRad, 1.0)));
    #else
        imageStore(scratchImg, px + ivec2(DIFFUSE_1D_DST_TILE * res.x, 0), roundHalf(outVal));
    #endif
}
