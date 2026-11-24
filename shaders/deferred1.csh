#version 430 compatibility

// Pass: Dedicated Lightweight 1-SPP Direct Light NEE (RIS + Stained-Glass Visibility Ray)
// Separated from deferred.csh to eliminate GRF register spilling on Xe-HPG; early-outs in ~0.00ms when no lights!
#include "/lib/Raytrace.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform image2D colorimg3;

void main() {
    uint lCount = lightList.lightCount;
    if (lCount == 0u || lCount > MAX_SCENE_LIGHTS) return;

    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) return;

    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 worldPos = viewToWorld(screenToView(vec3(uv, depth)));
    float camDist = length(worldPos);
    vec3 toCam = -worldPos / max(camDist, 1e-4);
    vec3 rayOrig = makeCornerSafeRayOrigin(worldPos, geoNormal, toCam, camDist);

    vec3 neeDir = vec3(0.0, 1.0, 0.0);
    vec3 neeRad = sampleDirectLightsNEE(rayOrig, shadeNormal, false, 8, stbn2D(px, frameCounter, 8u), neeDir);
    if (any(isnan(neeRad)) || any(isinf(neeRad)) || maxOf(neeRad) <= 1e-5) return;

    vec4 prev = imageLoad(colorimg3, px);
    imageStore(colorimg3, px, vec4(min(prev.rgb + neeRad, vec3(28.0)), prev.a));
}
