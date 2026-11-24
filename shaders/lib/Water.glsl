#ifndef LIB_WATER_GLSL
#define LIB_WATER_GLSL

#include "/lib/Resources.glsl"

// ============================================================================
// Physical Water Optics: Anisotropic Swells, Wave Parallax, Chromatic Caustics,
// Crystal-Clear Turquoise Beer-Lambert Attenuation & Directional Subsurface Scattering
// ============================================================================

const float WATER_IOR = 1.333;
const vec3  WATER_SCATTER_TINT = vec3(0.10, 0.62, 1.00);

float waterHash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth quintic-interpolated value noise (C2 continuous -> zero facet creases in specular glints!).
float waterNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = waterHash12(i);
    float b = waterHash12(i + vec2(1.0, 0.0));
    float c = waterHash12(i + vec2(0.0, 1.0));
    float d = waterHash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float waterCurveTop(float x) {
    float inv = 1.0 - x;
    return 1.0 - inv * inv;
}

// Golden-angle rotation matrix for non-repeating multi-octave wave directions.
const mat2 WATER_ROT_GA = mat2(-0.73736888, 0.67549029, -0.67549029, -0.73736888);

// Multi-octave anisotropic wind-modulated wave heightfield in [0, 1]:
// - Macro wind envelope creates organic calm + rippled patches across lakes and oceans.
// - Two primary directional swells (one sharp-crested) drive parallax & macro reflections.
// - Three domain-warped capillary octaves add crisp micro-ripple sparkle when detail == true.
float evalWaterHeight(vec3 absPos, bool detail, bool useWindMod) {
    float t = frameTimeCounter;
    vec2 drift = vec2(0.38, 0.14) * t;
    vec2 p = absPos.xz - absPos.y * 0.1;

    float windEnv = useWindMod ? waterNoise(absPos.xz * 0.08 + drift * 0.05) : 0.65;
    vec2 wc = p * 0.42;

    float h = 0.0;
    h += waterNoise(wc * vec2(1.0, 0.60) + drift + windEnv * 1.4) * 16.0;
    wc = WATER_ROT_GA * (wc * 1.95);
    h += waterCurveTop(waterNoise(wc * vec2(1.0, 0.35) - drift * 1.15 - windEnv * 1.1)) * 12.0;

    if (detail) {
        float warp = h * 0.065;
        vec2 dc = wc - drift * 0.25;
        dc = WATER_ROT_GA * (dc * 1.75);
        h += waterNoise(dc * vec2(1.0, 0.45) + drift * 1.3 + warp) * 2.2;
        dc = WATER_ROT_GA * (dc * 1.45);
        h += waterNoise(dc * vec2(1.0, 0.45) - drift * 1.5 + warp) * 1.0;
        dc = WATER_ROT_GA * (dc * 1.40);
        h += waterNoise(dc * vec2(1.0, 0.65) + drift * 1.7 + warp) * 1.1;
    }

    float normH = 1.0 - h * (1.0 / 32.3);
    if (useWindMod) {
        normH *= (saturate(windEnv * 1.8 - 0.5) * 0.65 + 0.35);
    }
    return normH;
}

// Relief parallax march over the 2-swell water surface for real 3D stereo wave depth.
vec3 evalWaterParallax(vec3 absPos, vec3 viewDirWorld, float NdotU, float noise, inout float waterThickness) {
    #ifdef WATER_PARALLAX
        if (NdotU > 0.995) {
            const float waveRange = 0.38;
            const int STEPS = 10;
            const int REFINES = 4;

            // viewDirWorld points from eye -> surface (negative Y when looking down onto water).
            vec3 dir = viewDirWorld / (abs(viewDirWorld.y) + 0.14);
            dir.y = -1.0;
            vec3 stepDir = dir * (waveRange / float(STEPS));
            stepDir.xz *= 0.85 * float(WAVE_STRENGTH);

            vec3 basePos = absPos - stepDir * 4.0;
            vec3 stepVec = vec3(0.0);
            float h = evalWaterHeight(basePos, false, true);

            for (int i = 0; i < STEPS; ++i) {
                stepVec = stepDir * (float(i) + noise);
                h = evalWaterHeight(basePos + stepVec, false, true);
                if (h * waveRange > stepVec.y + waveRange) break;
            }
            for (int i = 0; i < REFINES; ++i) {
                stepDir *= 0.5;
                float sgn = (stepVec.y + waveRange > h * waveRange) ? 1.0 : -1.0;
                stepVec += stepDir * sgn;
                h = evalWaterHeight(basePos + stepVec, false, true);
            }
            waterThickness = max(waterThickness - (stepVec.y + waveRange * 0.5), 0.0);
            return basePos + stepVec;
        }
    #endif
    return absPos;
}

vec3 evalWaterNormal(vec3 absPos, vec3 geoN, float camDist) {
    if (abs(geoN.y) < 0.5) return geoN;
    const float eps = 0.09;
    float h0 = evalWaterHeight(absPos, true, true);
    float hx = evalWaterHeight(absPos + vec3(eps, 0.0, 0.0), true, true);
    float hz = evalWaterHeight(absPos + vec3(0.0, 0.0, eps), true, true);

    float strength = 2.6 * float(WAVE_STRENGTH);
    vec3 n = normalize(vec3((h0 - hx) * strength, 1.0, (h0 - hz) * strength));
    // Smoothly attenuate high-frequency ripples at far grazing distances to prevent shimmering.
    float distFade = smoothstep(28.0, 128.0, camDist);
    return sign(geoN.y) * normalize(mix(n, vec3(0.0, 1.0, 0.0), distFade * 0.85));
}

// Compress low wave-reflected elevations smoothly above the mesh horizon so Fresnel & raytrace agree.
vec3 stabilizeWaterNormal(vec3 N, vec3 geoN, vec3 V) {
    vec3 R = reflect(-V, N);
    float elev = dot(R, geoN);
    const float minElev = 0.002;
    float shoulder = max(0.45 * dot(V, geoN), 2.0 * minElev);
    if (elev >= shoulder) return N;
    float span = shoulder - minElev;
    float corr = minElev + span * span / (span + shoulder - elev);
    R = normalize(R + (corr - elev) * geoN);
    return normalize(V + R);
}

// Exact water-to-air Fresnel including Total Internal Reflection outside Snell's window (~48.6 deg).
float evalSnellInternalFresnel(float NdotV) {
    float sinT2 = (WATER_IOR * WATER_IOR) * max(1.0 - NdotV * NdotV, 0.0);
    if (sinT2 >= 1.0) return 1.0;
    float cosT = sqrt(max(1.0 - sinT2, 0.0));
    float rs = (WATER_IOR * NdotV - cosT) / max(WATER_IOR * NdotV + cosT, 1e-5);
    float rp = (NdotV - WATER_IOR * cosT) / max(NdotV + WATER_IOR * cosT, 1e-5);
    return saturate(0.5 * (rs * rs + rp * rp));
}

// Energy-conserving analytical wave-curvature caustics projected along sunlight onto submerged geometry,
// with subtle RGB chromatic dispersion. Returns a multiplicative sunlight modulation centered near 1.0.
vec3 evalWaterCaustics(vec3 seabedAbsPos, vec3 lightDir, float waterPath) {
    #ifndef WATER_CAUSTICS
        return vec3(1.0);
    #endif
    float depthBelow = waterPath * max(lightDir.y, 0.0);
    if (lightDir.y <= 0.05 || depthBelow <= 0.02) return vec3(1.0);

    float depthMask = smoothstep(0.02, 0.45, depthBelow) * exp(-depthBelow * 0.065);
    if (depthMask <= 1e-3) return vec3(1.0);

    vec2 projXZ = seabedAbsPos.xz + lightDir.xz * waterPath;
    const float e = 0.14;
    float focusScale = clamp(depthBelow * 0.45, 0.15, 1.40);
    vec2 disp = normalize(lightDir.xz + vec2(1e-4)) * (0.045 * focusScale);

    vec3 caustics;
    for (int ch = 0; ch < 3; ++ch) {
        vec2 off = disp * float(ch - 1);
        vec3 cp = vec3((projXZ.x + off.x) * 1.55, 0.0, (projXZ.y + off.y) * 1.55);
        float hc = evalWaterHeight(cp, false, false);
        float hx = evalWaterHeight(cp + vec3(e, 0.0, 0.0), false, false)
                 + evalWaterHeight(cp - vec3(e, 0.0, 0.0), false, false);
        float hz = evalWaterHeight(cp + vec3(0.0, 0.0, e), false, false)
                 + evalWaterHeight(cp - vec3(0.0, 0.0, e), false, false);
        // Positive curvature troughs focus refracted rays into crisp bright caustic filaments!
        float lap = (hx + hz - 4.0 * hc);
        float filament = pow(saturate(lap * 6.8 + 0.12), 2.8) * 2.15;
        caustics[ch] = 0.68 + filament;
    }
    return mix(vec3(1.0), caustics, depthMask);
}

// Crystal-clear Beer-Lambert volume attenuation + sunlit directional turquoise subsurface scattering.
// worldDir points from eye -> surface/pixel.
void evalWaterVolume(float waterDist, vec3 worldDir, vec3 waveN, vec3 lightDir,
                     vec3 directCol, vec3 sunVis, vec3 skyAmbient, float skyLight,
                     vec3 biomeSrgb, bool underwaterCam,
                     out vec3 outTransmittance, out vec3 outInscatter) {
    float d = max(waterDist, 0.0);
    vec3 sigmaT = waterExtinction(biomeSrgb);
    outTransmittance = exp(-sigmaT * d);

    vec3 biomeLin = srgbToLinear(clamp(biomeSrgb, vec3(0.04), vec3(1.0)));
    vec3 biomeChroma = biomeLin / max(maxOf(biomeLin), 1e-3);
    vec3 scatterTint = mix(WATER_SCATTER_TINT, WATER_SCATTER_TINT * biomeChroma, 0.35);

    float VdotL = dot(worldDir, lightDir);
    if (!underwaterCam) {
        // Above-water view: sunlit wave-crest subsurface translucency + deep ocean sapphire ambient.
        float viewCos = saturate(dot(-worldDir, waveN));
        float slopeCrest = saturate((1.0 - abs(waveN.y)) * 8.0);
        float forwardScatter = 0.25 + 1.85 * pow(saturate(VdotL * 0.5 + 0.5), 3.0);
        float crestGlow = (0.35 + 0.65 * slopeCrest) * (0.45 + 0.55 * (1.0 - viewCos)) * forwardScatter;

        vec3 sunTerm = directCol * sunVis * (crestGlow * 0.14);
        vec3 skyTerm = skyAmbient * (0.30 + 0.70 * skyLight) * 0.24;
        outInscatter = (sunTerm + skyTerm) * scatterTint * (1.0 - exp2(-d * 0.22));
    } else {
        // Submerged underwater camera fog: luminous sunlit forward scatter + clear blue-teal depth falloff.
        float sunPhase = 1.0 / (max(-VdotL, -0.8) * 3.5 + 4.8);
        vec3 sunTerm = directCol * sunVis * (sunPhase * 0.42 * skyLight);
        vec3 skyTerm = skyAmbient * (0.40 + 0.60 * saturate(worldDir.y * 0.5 + 0.5)) * (0.25 + 0.75 * skyLight) * 0.34;
        outInscatter = (sunTerm + skyTerm) * scatterTint * (1.0 - exp2(-d * 0.075));
    }
}

#endif
