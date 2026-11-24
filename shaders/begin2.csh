#version 430 compatibility

// Pass: Hillaire 2020 Isotropic Multi-Scattering LUT (32 x 32)
// 1 workgroup per texel (64 Fibonacci sphere directions reduced in shared memory) -> ~0.015ms!
#define STAGE_ATMO_MULTI_WRITE
#include "/lib/Atmosphere.glsl"

layout(local_size_x = 64, local_size_y = 1) in;
const ivec3 workGroups = ivec3(32, 32, 1); // 1 workgroup per (32 x 32) texel

shared vec3 sL2nd[64];
shared vec3 sFms[64];

void main() {
    ivec2 px = ivec2(gl_WorkGroupID.xy);
    uint tid = gl_LocalInvocationID.x;

    vec2 uv = (vec2(px) + 0.5) / vec2(ATMO_MULTI_RES);
    float sunMu = clamp(uv.x * 2.0 - 1.0, -1.0, 1.0);
    float r = mix(ATMO_BOTTOM_R + 0.05, ATMO_TOP_R - 0.05, uv.y);

    vec3 pos = vec3(0.0, r, 0.0);
    vec3 sunDir = vec3(sqrt(max(1.0 - sunMu * sunMu, 0.0)), sunMu, 0.0);

    // Uniform 64-direction Fibonacci sphere quadrature (exact 4*PI solid angle integration).
    const float GOLDEN_RATIO = 1.61803398875;
    float z = 1.0 - (2.0 * float(tid) + 1.0) / 64.0;
    float rad = sqrt(max(1.0 - z * z, 0.0));
    float phi = TWO_PI * fract(float(tid) / GOLDEN_RATIO);
    vec3 rayDir = vec3(rad * cos(phi), z, rad * sin(phi));

    float distTop = atmoRaySphereDist3D(pos, rayDir, ATMO_TOP_R);
    float distGround = atmoRaySphereDist3D(pos, rayDir, ATMO_BOTTOM_R);
    bool hitGround = (distGround >= 0.0);
    float tMax = hitGround ? distGround : distTop;

    vec3 lum = vec3(0.0);
    vec3 fms = vec3(0.0);
    vec3 throughput = vec3(1.0);

    if (tMax > 0.0) {
        const int STEPS = 20;
        float dt = tMax / float(STEPS);
        const float isoPhase = 0.25 * INV_PI;

        for (int i = 0; i < STEPS; ++i) {
            float t = (float(i) + 0.5) * dt;
            vec3 p = pos + rayDir * t;
            float ri = length(p);
            vec3 up = p / max(ri, 1e-4);

            vec3 sR, sM, ext;
            atmoSampleMedium(ri - ATMO_BOTTOM_R, rainStrength, sR, sM, ext);
            vec3 sTotal = sR + sM;

            vec3 sampleTrans = exp(-ext * dt);
            vec3 invExt = 1.0 / max(ext, vec3(1e-7));
            vec3 integ = (1.0 - sampleTrans) * invExt;

            float sMu = dot(up, sunDir);
            if (!atmoHitsGround(ri, sMu)) {
                vec3 sunTrans = sampleAtmoTransmittance(ri, sMu);
                lum += throughput * (sTotal * isoPhase) * sunTrans * integ;
            }
            // f_ms integrates (sTotal * p_iso) over 4*PI solid angle -> 4*PI * p_iso = 1.0!
            fms += throughput * sTotal * integ;
            throughput *= sampleTrans;
        }

        // Account for sunlight bouncing off the planetary surface back into the atmosphere.
        if (hitGround) {
            vec3 gp = pos + rayDir * distGround;
            vec3 gUp = normalize(gp);
            float gMu = max(dot(gUp, sunDir), 0.0);
            if (gMu > 0.0) {
                vec3 gSunTrans = sampleAtmoTransmittance(ATMO_BOTTOM_R + 0.01, gMu);
                lum += throughput * gSunTrans * (ATMO_GROUND_ALBEDO * INV_PI * gMu);
            }
        }
    }

    // Solid angle weight per Fibonacci sample = 4*PI / 64; isotropic average divides by 4*PI -> 1 / 64.
    sL2nd[tid] = lum * (1.0 / 64.0);
    sFms[tid]  = fms * (1.0 / 64.0);
    barrier();

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            sL2nd[tid] += sL2nd[tid + stride];
            sFms[tid]  += sFms[tid + stride];
        }
        barrier();
    }

    if (tid == 0u) {
        // Infinite geometric series sum: Psi_ms = L_2nd / (1 - f_ms).
        vec3 psiMs = sL2nd[0] / max(vec3(1.0) - sFms[0], vec3(0.05));
        imageStore(atmoMultiImg, px, vec4(max(psiMs, vec3(0.0)), 1.0));
    }
}
