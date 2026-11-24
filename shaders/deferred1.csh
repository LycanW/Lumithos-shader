#version 430 compatibility

// Pass: Dedicated Lightweight 1-SPP Direct Light NEE (RIS + Stained-Glass Visibility Ray)
// Separated from deferred.csh to eliminate GRF register spilling on Xe-HPG; early-outs in ~0.00ms when no lights!
#include "/lib/Raytrace.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
#if DEBUG_VIEW == 1 || DIFFUSE_TRACE_RES == 2
    #define DIFFUSE_MODE 2
    const vec2 workGroupsRender = vec2(1.0, 1.0);
#elif DIFFUSE_TRACE_RES == 1
    #define DIFFUSE_MODE 1
    const vec2 workGroupsRender = vec2(0.5, 1.0);
#else
    #define DIFFUSE_MODE 0
    const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif

layout(rgba16f) uniform image2D colorimg3;

void main() {
    if (lightList.lightCount == 0u) return;

    ivec2 tid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    #if DIFFUSE_MODE == 0
        int f4 = frameCounter & 3;
        ivec2 qOff = ivec2(((f4 >> 1) ^ (f4 & 1)) & 1, f4 & 1);
        ivec2 px = tid * 2 + qOff;
        ivec2 bnPx = tid;
        int bnSeq = (frameCounter >> 2) + f4 * 197;
    #elif DIFFUSE_MODE == 1
        ivec2 px = ivec2(tid.x * 2 + ((tid.y ^ frameCounter) & 1), tid.y);
        ivec2 bnPx = tid;
        int bnSeq = (frameCounter >> 1) + (frameCounter & 1) * 197;
    #else
        ivec2 px = tid;
        ivec2 bnPx = px;
        int bnSeq = frameCounter;
    #endif
    if (any(greaterThanEqual(px, res))) return;

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) return;

    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);
    vec3 eyeToSurf = mat3(gbufferModelViewInverse) * viewPos;
    float camDist = length(eyeToSurf);
    vec3 toCam = -eyeToSurf / max(camDist, 1e-4);
    vec3 rayOrig = makeCornerSafeRayOrigin(worldPos, geoNormal, toCam, camDist);

    vec3 neeDir = vec3(0.0, 1.0, 0.0);
    vec3 neeRad = sampleDirectLightsNEE(rayOrig, shadeNormal, false, 16, stbn2D(bnPx, bnSeq, 8u), neeDir);
    if (any(isnan(neeRad)) || any(isinf(neeRad)) || maxOf(neeRad) <= 1e-5) return;

    vec4 prev = imageLoad(colorimg3, px);
    imageStore(colorimg3, px, vec4(min(prev.rgb + min(neeRad, vec3(16.0)), vec3(24.0)), prev.a));
}
