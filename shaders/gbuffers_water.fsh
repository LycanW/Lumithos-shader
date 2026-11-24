#version 430 compatibility

// Minimal translucent forward pass (water, stained glass, ice) blended directly onto colortex0
// after deferred lighting completes, before composite TAA.
#include "/lib/SkyShadow.glsl"

uniform sampler2D gtexture;

in vec2  vTexCoord;
in vec4  vColor;
in vec3  vGeoNormal;
in vec3  vWorldPos;
in float vSkyLight;

/* RENDERTARGETS: 0 */
layout(location = 0) out vec4 colorOut;

void main() {
    vec4 tex = texture(gtexture, vTexCoord) * vColor;
    if (tex.a < 0.01) discard;

    vec3 N = normalize(vGeoNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(-vWorldPos);
    vec3 R = reflect(-V, N);

    vec3 albedo = srgbToLinear(tex.rgb);
    vec3 lightDir = getShadowLightDir();
    float NdotL = max(dot(N, lightDir), 0.0);
    vec3 shadowVis = (NdotL > 0.0) ? sampleShadowFast(vWorldPos, N) : vec3(0.0);

    vec3 direct = albedo * INV_PI * getDirectLightColor() * (NdotL * shadowVis);
    vec3 ambient = albedo * evalSky(N, false) * (0.25 + 0.75 * vSkyLight);
    vec3 skyRefl = evalSky(R, true) * (0.15 + 0.85 * vSkyLight);

    float fresnel = fresnelSchlick(vec3(0.04), max(dot(N, V), 0.0)).x;
    vec3 col = mix(direct + ambient, skyRefl, fresnel);
    float outAlpha = clamp(mix(tex.a, 1.0, fresnel * 0.75), 0.05, 0.98);

    colorOut = vec4(max(col, vec3(0.0)), outAlpha);
}
