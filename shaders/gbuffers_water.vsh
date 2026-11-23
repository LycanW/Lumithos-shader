#version 430 compatibility

#include "/lib/Resources.glsl"

out vec2  vTexCoord;
out vec4  vColor;
out vec3  vGeoNormal;
out vec3  vWorldPos;
out float vSkyLight;

void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    vec4 clipPos = gl_ProjectionMatrix * viewPos;
    #ifdef TAA_ENABLED
        clipPos.xy += clipPos.w * taaJitter;
    #endif
    gl_Position = clipPos;

    vWorldPos  = (gbufferModelViewInverse * viewPos).xyz;
    vGeoNormal = normalize(mat3(gbufferModelViewInverse) * normalize(gl_NormalMatrix * gl_Normal));
    vTexCoord  = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vColor     = gl_Color;
    vSkyLight  = saturate((gl_MultiTexCoord1.y - 8.0) / 232.0);
}
