#version 430 compatibility
uniform sampler2D gtexture;

in vec2 gTexCoord;
in vec4 gColor;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 shadowColorOut;

void main() {
    vec4 albedo = texture(gtexture, gTexCoord) * gColor;
    if (albedo.a < 0.1) discard;
    shadowColorOut = albedo;
}
