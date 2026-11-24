#version 430 compatibility

// Pass: Hillaire 2020 Non-Linear Horizon-Focused Sky-View LUT (192 x 108)
// Integrates Sun + Moon single & multi-scattering through Rayleigh + Mie + Ozone + Rain Overcast.
#define STAGE_ATMO_SKY_WRITE
#include "/lib/Atmosphere.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const ivec3 workGroups = ivec3(24, 14, 1); // covers 192 x 112 >= 192 x 108

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(px, ATMO_SKY_RES))) return;

    vec2 uv = (vec2(px) + 0.5) / vec2(ATMO_SKY_RES);
    vec3 sunDir = getSunDir();
    vec3 moonDir = -sunDir;
    vec3 rayDir = atmoSkyUVToDir(uv, sunDir);

    float camR = atmoCameraRadius();
    vec3 pos = vec3(0.0, camR, 0.0);

    // Clamp below-horizon integration smoothly to the horizon tangent (no hard planetary ground-sphere
    // dark cutoff band below y = 0!) so chunk edges and floating worlds blend seamlessly into the sky.
    vec3 traceDir = (rayDir.y < 0.0) ? normalize(vec3(rayDir.x, 0.0, rayDir.z)) : rayDir;
    float tMax = atmoRaySphereDist3D(pos, traceDir, ATMO_TOP_R);

    vec3 skyLum = vec3(0.0);
    if (tMax > 0.0) {
        float cosSun  = dot(traceDir, sunDir);
        float cosMoon = dot(traceDir, moonDir);
        float pR_sun  = atmoRayleighPhase(cosSun);
        float pM_sun  = atmoMiePhase(cosSun, rainStrength);
        float pR_moon = atmoRayleighPhase(cosMoon);
        float pM_moon = atmoMiePhase(cosMoon, rainStrength);

        const int STEPS = 32;
        vec3 throughput = vec3(1.0);
        float prevT = 0.0;

        for (int i = 0; i < STEPS; ++i) {
            // Quadratic step spacing concentrates resolution near the observer and low-altitude Mie/Ozone layers!
            float u1 = float(i + 1) / float(STEPS);
            float t1 = tMax * (u1 * u1);
            float dt = max(t1 - prevT, 1e-5);
            float tMid = 0.5 * (prevT + t1);
            prevT = t1;

            vec3 p = pos + traceDir * tMid;
            float ri = length(p);
            vec3 up = p / max(ri, 1e-4);

            vec3 sR, sM, ext;
            atmoSampleMedium(ri - ATMO_BOTTOM_R, rainStrength, sR, sM, ext);
            vec3 sTotal = sR + sM;

            vec3 sampleTrans = exp(-ext * dt);
            vec3 integ = (1.0 - sampleTrans) / max(ext, vec3(1e-7));

            // 1) Sunlight: single scattering + Hillaire isotropic multiple scattering.
            float sMu = dot(up, sunDir);
            vec3 sunSingle = vec3(0.0);
            if (!atmoHitsGround(ri, sMu)) {
                vec3 sunTrans = sampleAtmoTransmittance(ri, sMu);
                sunSingle = (sR * pR_sun + sM * pM_sun) * sunTrans;
            }
            vec3 sunMulti = sTotal * sampleAtmoMultiScatter(ri, sMu);
            vec3 inScatter = (sunSingle + sunMulti) * ATMO_SUN_IRRADIANCE;

            // 2) Moonlight: physical cool-silver night atmosphere + subtle moon halo.
            float mMu = dot(up, moonDir);
            vec3 moonSingle = vec3(0.0);
            if (!atmoHitsGround(ri, mMu)) {
                vec3 moonTrans = sampleAtmoTransmittance(ri, mMu);
                moonSingle = (sR * pR_moon + sM * pM_moon) * moonTrans;
            }
            vec3 moonMulti = sTotal * sampleAtmoMultiScatter(ri, mMu);
            inScatter += (moonSingle + moonMulti) * ATMO_MOON_IRRADIANCE;

            skyLum += throughput * inScatter * integ;
            throughput *= sampleTrans;
        }

        // Subtle dark-sky airglow floor at midnight so deep moonless skies stay naturally luminous (never 0.000).
        float zenithFactor = saturate(rayDir.y * 0.5 + 0.5);
        vec3 airglow = mix(vec3(0.0045, 0.0070, 0.0135), vec3(0.0020, 0.0038, 0.0095), zenithFactor);
        skyLum += airglow * throughput;

        // Gentle soft darkening only when looking far downward into the void (zero seam at horizon!).
        if (rayDir.y < 0.0) {
            skyLum *= mix(1.0, 0.50, smoothstep(0.0, -0.85, rayDir.y));
        }
    }

    // Rainy overcast sky coupling: softly veils the clear sky with cool moody cloud-deck scattering.
    if (rainStrength > 1e-3) {
        float dayFactor = smoothstep(-0.08, 0.18, sunDir.y);
        vec3 overcastZenith = mix(vec3(0.006, 0.009, 0.014), vec3(0.16, 0.19, 0.24), dayFactor);
        vec3 overcastHoriz  = mix(vec3(0.010, 0.014, 0.020), vec3(0.28, 0.31, 0.36), dayFactor);
        float h = pow(1.0 - saturate(rayDir.y), 3.0);
        vec3 overcast = mix(overcastZenith, overcastHoriz, h);
        skyLum = mix(skyLum, overcast, rainStrength * 0.62);
    }

    skyLum = max(skyLum, vec3(0.0));
    if (any(isnan(skyLum)) || any(isinf(skyLum))) skyLum = vec3(0.0);
    imageStore(atmoSkyImg, px, vec4(skyLum, 1.0));
}
