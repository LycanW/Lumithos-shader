// 1D Separable 3-Tap Rotated-Axis Edge-Avoiding Filter (Sundial / iterationRP / Revoxelation hybrid)
// Only 2 neighbor fetches per pass (16 total across 8 passes -> 2x faster than 4x 3x3 A-Trous!)
// yet reaches +-48 pixels with grazing-angle foreshortening and jittered steps to wipe out blotches!
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
    uint g1 = texelFetch(colortex2, px, 0).r;
    vec3 centerGeoN   = unpackNormal16(g1 & 65535u);
    vec3 centerShadeN = unpackNormal16(g1 >> 16u);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos = screenToView(vec3(uv, depth));
    vec3 viewGeoN = normalize(mat3(gbufferModelView) * centerGeoN);

    // Perspective-projected foreshortening shrinks steps along grazing surfaces.
    float absNdotV = abs(dot(viewGeoN, normalize(-viewPos)));
    vec2 axis = DIFFUSE_1D_AXIS;
    vec2 offsetScale = 1.0 - abs(viewGeoN.xy) * (1.0 - absNdotV);

    // Spatiotemporal jitter breaks fixed-grid ringing and A-Trous waves.
    float jitter = stbn2D(px, frameCounter + DIFFUSE_1D_PASS * 7, 11u).x;
    ivec2 tapOffset = ivec2(round(axis * offsetScale * (0.70 + 0.60 * jitter)));
    if (tapOffset == ivec2(0)) {
        tapOffset = ivec2(sign(axis));
    }

    float planeTol = max(0.035, 0.008 * (-viewPos.z));
    float lumaWeightScale = (DIFFUSE_1D_VAR_SCALE > 0.0)
        ? (DIFFUSE_1D_VAR_SCALE / max(centerSigma * 5.0 + 0.015, 0.015))
        : 0.0;

    vec4 sum = center;
    float sumW = 1.0;

    for (int i = 0; i < 2; ++i) {
        ivec2 q = px + tapOffset;
        tapOffset = -tapOffset;
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, res))) continue;

        float qd = texelFetch(depthtex1, q, 0).r;
        if (qd >= 0.999999) continue;

        uint qg1 = texelFetch(colortex2, q, 0).r;
        vec3 qGeoN = unpackNormal16(qg1 & 65535u);
        if (dot(centerGeoN, qGeoN) < 0.90) continue;

        vec2 qUv = (vec2(q) + 0.5) * texelSize();
        vec3 qViewPos = screenToView(vec3(qUv, qd));
        float planeDist = abs(dot(qViewPos - viewPos, viewGeoN));
        if (planeDist > planeTol) continue;

        vec3 qShadeN = unpackNormal16(qg1 >> 16u);
        float wNorm = pow(max(dot(centerShadeN, qShadeN), 0.0), 8.0);
        if (wNorm < 1e-4) continue;

        vec4 tap = texelFetch(scratchTex, q + srcOff, 0);
        float wLuma = (lumaWeightScale > 0.0)
            ? exp(-abs(luma(tap.rgb) - centerLuma) * lumaWeightScale)
            : 1.0;

        float w = wNorm * wLuma;
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
