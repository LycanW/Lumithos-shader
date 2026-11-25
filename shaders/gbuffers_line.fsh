#version 430 compatibility

#include "/lib/Common.glsl"

in vec4 vColor;

/* RENDERTARGETS: 6,7 */
layout(location = 0) out vec4 radianceOut;
layout(location = 1) out vec4 transmitOut;

void main() {
    radianceOut = vec4(srgbToLinear(vColor.rgb) * vColor.a, vColor.a);
    transmitOut = vec4(vec3(1.0 - vColor.a), 1.0);
}
