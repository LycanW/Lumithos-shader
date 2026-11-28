#ifndef LIB_CAMERA_WHITE_BALANCE_GLSL
#define LIB_CAMERA_WHITE_BALANCE_GLSL

#include "/lib/Config.glsl"

// Bradford cone response for linear Rec.709 with a D65 reference white.
const mat3 CAMERA_RGB_TO_XYZ = mat3(
    vec3(0.41239080, 0.21263901, 0.01933082),
    vec3(0.35758434, 0.71516868, 0.11919478),
    vec3(0.18048079, 0.07219232, 0.95053215)
);
const mat3 CAMERA_XYZ_TO_CONE = mat3(
    vec3( 0.8951, -0.7502,  0.0389),
    vec3( 0.2664,  1.7135, -0.0685),
    vec3(-0.1614,  0.0367,  1.0296)
);
const mat3 CAMERA_RGB_TO_CONE = CAMERA_XYZ_TO_CONE * CAMERA_RGB_TO_XYZ;
// Precomputed inverse of CAMERA_RGB_TO_CONE.
const mat3 CAMERA_CONE_TO_RGB = mat3(
    vec3( 2.53844799, -0.14600038, -0.04228838),
    vec3(-1.29348257,  1.11662242, -0.07159007),
    vec3(-0.04024325, -0.02232853,  1.02250731)
);
const vec3 CAMERA_D65_CONE = CAMERA_RGB_TO_CONE * vec3(1.0);
const ivec2 CAMERA_WHITE_BALANCE_TEXEL = ivec2(2, 0);

vec3 cameraWhiteBalanceScale(sampler2D cameraState) {
    #ifdef CAMERA_AUTO_WHITE_BALANCE
        vec4 state = texelFetch(cameraState, CAMERA_WHITE_BALANCE_TEXEL, 0);
        return state.w > 0.0 ? state.rgb : vec3(1.0);
    #else
        return vec3(1.0);
    #endif
}

vec3 applyCameraWhiteBalance(vec3 hdr, vec3 automaticScale) {
    #ifdef CAMERA_AUTO_WHITE_BALANCE
        return CAMERA_CONE_TO_RGB * (automaticScale * (CAMERA_RGB_TO_CONE * hdr));
    #elif CAMERA_COLOR_TEMPERATURE == 6500 && CAMERA_COLOR_TINT == 0
        return hdr;
    #else
        // Krystek's CIE 1960 Planckian locus approximation.
        const vec2 a = vec2(0.860117757, 0.317398726);
        const vec2 b = vec2(1.54118254e-4, 4.22806245e-5);
        const vec2 c = vec2(1.28641212e-7, 4.20481691e-8);
        const vec2 d = vec2(8.42420235e-4, -2.89741816e-5);
        const vec2 e = vec2(7.08145163e-7, 1.61456053e-7);
        const float temperature = float(CAMERA_COLOR_TEMPERATURE);
        const vec2 denominator = 1.0 + temperature * (d + e * temperature);
        const vec2 locus = (a + temperature * (b + c * temperature)) / denominator;
        const vec2 referenceLocus = (a + 6500.0 * (b + c * 6500.0)) / (1.0 + 6500.0 * (d + e * 6500.0));
        const vec2 d65 = vec2(4.0 * 0.3127, 6.0 * 0.3290) / (3.0 - 2.0 * 0.3127 + 12.0 * 0.3290);

        // Anchor 6500 K to D65 and shift tint perpendicular to the local locus.
        const vec2 tangent = (b + 2.0 * c * temperature - locus * (d + 2.0 * e * temperature)) / denominator;
        const vec2 tintDirection = normalize(vec2(tangent.y, -tangent.x));
        const vec2 white = d65 + (locus - referenceLocus) + (0.00025 * float(CAMERA_COLOR_TINT)) * tintDirection;
        const vec3 sourceXYZ = vec3(3.0 * white.x, 2.0 * white.y, 4.0 - white.x - 10.0 * white.y) / (2.0 * white.y);

        const vec3 scale = CAMERA_D65_CONE / (CAMERA_XYZ_TO_CONE * sourceXYZ);
        const mat3 diagonal = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
        const mat3 adaptation = CAMERA_CONE_TO_RGB * diagonal * CAMERA_RGB_TO_CONE;
        return adaptation * hdr;
    #endif
}

#endif
