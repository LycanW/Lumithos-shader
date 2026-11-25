#version 430 compatibility

#include "/lib/Resources.glsl"

in vec2 mc_Entity;
in vec4 at_midBlock;
in vec4 at_tangent;

out vec2  vTexCoord;
out vec4  vColor;
out vec3  vGeoNormal;
out vec4  vTangent;
out vec3  vWorldPos;
out float vSkyLight;
out float vBlockEmit;
flat out int vMatId;

void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    vec4 clipPos = gl_ProjectionMatrix * viewPos;
    #ifdef TAA_ENABLED
        clipPos.xy += clipPos.w * taaJitter;
    #endif
    gl_Position = clipPos;

    vWorldPos  = (gbufferModelViewInverse * viewPos).xyz;
    vGeoNormal = normalize(mat3(gbufferModelViewInverse) * normalize(gl_NormalMatrix * gl_Normal));
    vTangent = vec4(mat3(gbufferModelViewInverse) * (gl_NormalMatrix * at_tangent.xyz), at_tangent.w);
    vTexCoord  = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vColor     = gl_Color;
    vSkyLight  = saturate((gl_MultiTexCoord1.y - 8.0) / 232.0);

    int id = int(mc_Entity.x + 0.5);
    vMatId = (id == 10001) ? 1 : ((id == 10003) ? 2 : 0);
    // On fluid faces Iris supplies the host block's emission (e.g. waterlogged amethyst).
    vBlockEmit = vMatId == 1 ? 0.0 : saturate(at_midBlock.w / 15.0);
}
