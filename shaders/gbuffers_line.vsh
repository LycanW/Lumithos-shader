#version 430 compatibility

#include "/lib/Config.glsl"

uniform vec2 taaJitter;
uniform float viewWidth;
uniform float viewHeight;
in float LineWidth;
out vec4 vColor;

void main() {
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    viewPos.xyz *= 255.0 / 256.0;
    vec4 clipPos = gl_ProjectionMatrix * viewPos;
    vec4 clipDir = gl_ProjectionMatrix * vec4(mat3(gl_ModelViewMatrix) * gl_Normal, 0.0);

    // The line vertices come in pairs. Move each copy to one side of the segment.
    vec2 viewport = vec2(viewWidth, viewHeight);
    vec2 axis = (clipDir.xy * clipPos.w - clipPos.xy * clipDir.w) * viewport;
    vec2 across = normalize(vec2(-axis.y, axis.x));
    if (across.x < 0.0) across = -across;
    float side = (gl_VertexID & 1) == 0 ? 1.0 : -1.0;
    clipPos.xy += clipPos.w * across * (side * LineWidth) / viewport;

    #ifdef TAA_ENABLED
        clipPos.xy += clipPos.w * taaJitter;
    #endif
    gl_Position = clipPos;
    vColor = gl_Color;
}
