#version 430 compatibility
#include "/lib/LightFilter.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
layout(rgba32f) uniform writeonly image2D lightGuideImg;

layout(rgba16f) uniform writeonly image2D diffuseWorkImg;

// The upstream input accepts a full or horizontal checkerboard field. For 1/4
// tracing, fill the missing row from the four true samples across that row.
vec4 diffuseInput(ivec2 p, uint normals, float z) {
    if (z <= 0.0 || !diffuseField(p)) return vec4(0,0,0,-1);
    if (diffuseTraced(p)) return texelFetch(colortex3,p,0);
    vec3 face = unpackNormal16(normals & 65535u), n = unpackNormal16(normals >> 16u);
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int y = -1; y <= 1; y += 2) for (int x = -1; x <= 1; x += 2) {
        ivec2 q = p + ivec2(x,y);
        if (!inFrame(q)) continue;
        vec4 light = texelFetch(colortex3,q,0);
        if (light.a <= 0.0) continue;
        float nq = linearizeDepth(texelFetch(depthtex1,q,0).r);
        if (abs(nq-z) > max(0.1, 0.07*z)) continue;
        vec3 qf, qn;
        readGBufferNormals(q,qf,qn);
        float cosine = dot(n,qn);
        if (dot(face,qf) < 0.9 || cosine < 0.45) continue;
        float w = pow(cosine,4.0);
        sum += w*light;
        total += w;
    }
    return total > 0.0 ? sum/total : vec4(0,0,0,-1);
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (!inFrame(p)) return;
    float rough, f0;
    bool hand;
    readGBufferSpecMask(p, rough, f0, hand);
    float depth = texelFetch(depthtex1, p, 0).r;
    uint normals = texelFetch(colortex2, p, 0).r;
    float z = hand || depth >= 0.999999 ? 0.0 : linearizeDepth(depth);
    float gloss = hasSpecularTrace(rough, f0, hand) && !usesMirrorDirection(rough) ? rough : -1.0;
    imageStore(diffuseWorkImg, p, diffuseInput(p,normals,z));
    imageStore(lightGuideImg, p, vec4(float(normals & 65535u), float(normals >> 16u), gloss, z));
}
