#version 430 compatibility

#include "/lib/Resources.glsl"

in vec4 at_midBlock;

out vec3  vVoxelPos;
out vec2  vTexCoord;
out vec4  vColor;
out float vSkyLight;
out float vEmissive;
out vec4  vShadowClip;

void main() {
    vec3 shadowViewPos = (gl_ModelViewMatrix * gl_Vertex).xyz;
    vec3 worldPos = (shadowModelViewInverse * vec4(shadowViewPos, 1.0)).xyz;
    vVoxelPos = worldToVoxelPos(worldPos);

    vec4 clipPos = gl_ProjectionMatrix * vec4(shadowViewPos, 1.0);
    clipPos.xy = distortShadowXY(clipPos.xy);
    clipPos.z *= 0.5;
    vShadowClip = clipPos;
    gl_Position = clipPos;

    vTexCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vColor    = gl_Color;
    vSkyLight = saturate((gl_MultiTexCoord1.y - 8.0) / 232.0);
    vEmissive = clamp(at_midBlock.w, 0.0, 15.0);
}
