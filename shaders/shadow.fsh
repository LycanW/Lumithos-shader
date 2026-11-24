#version 430 compatibility
#include "/lib/Common.glsl"
uniform sampler2D gtexture;

in vec2 gTexCoord;
in vec4 gColor;
flat in int gNeedTex;
flat in int gIsWater;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 shadowColorOut;

void main() {
    if (gIsWater != 0) {
        // Alpha zero tags water. Glass/cutout fragments below 0.1 are discarded below.
        shadowColorOut = vec4(exp(-waterExtinction(gColor.rgb)), 0.0);
        return;
    }
    // Opaque TERRAIN_SOLID never discards and never needs shadowcolor0 tint -> skip atlas fetch!
    if (gNeedTex == 0) {
        shadowColorOut = vec4(1.0);
        return;
    }
    vec4 albedo = texture(gtexture, gTexCoord) * gColor;
    if (albedo.a < 0.1) discard;
    shadowColorOut = albedo;
}
