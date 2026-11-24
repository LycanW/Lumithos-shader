#version 430 compatibility

// Pass: Hillaire 2020 Transmittance LUT (256 x 64)
// Integrates exact spectral optical depth through Rayleigh + Mie + 25km Ozone atmosphere.
#define STAGE_ATMO_TRANS_WRITE
#include "/lib/Atmosphere.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const ivec3 workGroups = ivec3(32, 8, 1); // exact 256 x 64

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(px, ATMO_TRANS_RES))) return;

    vec2 uv = (vec2(px) + 0.5) / vec2(ATMO_TRANS_RES);
    float r, mu;
    atmoTransUVToParams(uv, r, mu);

    float distTop = atmoRaySphereDist(r, mu, ATMO_TOP_R);
    if (distTop <= 0.0) {
        imageStore(atmoTransImg, px, vec4(1.0));
        return;
    }

    const int STEPS = 40;
    float dt = distTop / float(STEPS);
    vec3 opticalDepth = vec3(0.0);

    for (int i = 0; i < STEPS; ++i) {
        float t = (float(i) + 0.5) * dt;
        float ri = sqrt(max(r * r + t * t + 2.0 * r * t * mu, ATMO_BOTTOM_R * ATMO_BOTTOM_R));
        vec3 sR, sM, ext;
        atmoSampleMedium(ri - ATMO_BOTTOM_R, rainStrength, sR, sM, ext);
        opticalDepth += ext * dt;
    }

    vec3 trans = exp(-opticalDepth);
    imageStore(atmoTransImg, px, vec4(trans, 1.0));
}
