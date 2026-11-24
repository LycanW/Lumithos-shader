#ifndef LIB_ATMOSPHERE_GLSL
#define LIB_ATMOSPHERE_GLSL

#include "/lib/Resources.glsl"

// ============================================================================
// Physically-Based Spherical Atmosphere (Hillaire 2020):
// - 3-Media Optical Model: Rayleigh + Cornette-Shanks Mie + 25km Ozone Band
// - Real-Time Precomputed LUTs: Transmittance (256x64), Multi-Scattering (32x32),
//   Non-Linear Horizon-Focused Sky-View (192x108)
// - Physical Sun Disc (limb darkening + exact transmittance), Lunar Disc,
//   Atmospheric-Extincted Stars + Milky Way, Weather Coupling & Aerial Perspective
// ============================================================================

const float ATMO_BOTTOM_R = 6360.0; // Earth radius (km)
const float ATMO_TOP_R    = 6460.0; // Top of atmosphere (km, 100 km thick)
const float ATMO_H_RAYLEIGH = 8.0;  // Rayleigh scale height (km)
const float ATMO_H_MIE      = 1.2;  // Mie aerosol scale height (km)

const vec3 ATMO_RAYLEIGH_SCATTER = vec3(5.802e-3, 13.558e-3, 33.100e-3);
const vec3 ATMO_OZONE_ABSORB     = vec3(0.650e-3,  1.881e-3,  0.085e-3);
const vec3 ATMO_GROUND_ALBEDO    = vec3(0.18, 0.20, 0.16);

// Radiometric solar/lunar top-of-atmosphere irradiance calibrated to Lumithos HDR exposure.
const vec3 ATMO_SUN_IRRADIANCE  = vec3(1.00, 0.96, 0.91) * 6.0;
const vec3 ATMO_MOON_IRRADIANCE = vec3(0.62, 0.78, 1.00) * 0.11;

const ivec2 ATMO_TRANS_RES = ivec2(256, 64);
const ivec2 ATMO_MULTI_RES = ivec2(32, 32);
const ivec2 ATMO_SKY_RES   = ivec2(192, 108);

float atmoCameraRadius() {
    // Anchor sea level (Y=62) at 0.25 km above ATMO_BOTTOM_R to avoid horizon grazing singularities.
    float altKm = max(0.15, 0.25 + (cameraPosition.y - 62.0) * 0.002);
    return min(ATMO_BOTTOM_R + altKm, ATMO_TOP_R - 0.5);
}

// Ray-sphere intersection from origin at radius r along direction with zenith cosine mu = dot(up, dir).
// Returns distance to nearest positive intersection, or -1.0 if missed.
float atmoRaySphereDist(float r, float mu, float sphereR) {
    float disc = r * r * (mu * mu - 1.0) + sphereR * sphereR;
    if (disc < 0.0) return -1.0;
    float s = sqrt(max(disc, 0.0));
    float t0 = -r * mu - s;
    float t1 = -r * mu + s;
    if (t0 >= 0.0) return t0;
    return (t1 >= 0.0) ? t1 : -1.0;
}

float atmoRaySphereDist3D(vec3 pos, vec3 dir, float sphereR) {
    float b = dot(pos, dir);
    float c = dot(pos, pos) - sphereR * sphereR;
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float s = sqrt(max(disc, 0.0));
    float t0 = -b - s;
    float t1 = -b + s;
    if (t0 >= 0.0) return t0;
    return (t1 >= 0.0) ? t1 : -1.0;
}

bool atmoHitsGround(float r, float mu) {
    return (mu < 0.0) && ((r * r * (mu * mu - 1.0) + ATMO_BOTTOM_R * ATMO_BOTTOM_R) >= 0.0);
}

// Evaluate local scattering & extinction coefficients (km^-1) at altitude altKm above ATMO_BOTTOM_R.
void atmoSampleMedium(float altKm, float rain,
                      out vec3 scatterR, out vec3 scatterM, out vec3 extinction) {
    float h = max(altKm, 0.0);
    float dR = exp(-h / ATMO_H_RAYLEIGH);
    float dM = exp(-h / ATMO_H_MIE);
    float dO = max(0.0, 1.0 - abs(h - 25.0) / 15.0);

    float mieBoost = mix(1.0, 4.2, rain);
    vec3 mieScatBase = vec3(3.996e-3) * mieBoost;
    vec3 mieAbsBase  = vec3(mix(4.400e-3, 1.600e-3, rain)) * mieBoost;

    scatterR   = ATMO_RAYLEIGH_SCATTER * dR;
    scatterM   = mieScatBase * dM;
    extinction = scatterR + (mieScatBase + mieAbsBase) * dM + ATMO_OZONE_ABSORB * dO;
}

float atmoRayleighPhase(float cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

float atmoHGPhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4);
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(denom));
}

// Dual-lobe Mie phase: crisp forward solar aureole + subtle anti-solar horizon backscatter.
float atmoMiePhase(float cosTheta, float rain) {
    float gFwd = mix(0.80, 0.58, rain);
    return 0.86 * atmoHGPhase(cosTheta, gFwd) + 0.14 * atmoHGPhase(cosTheta, -0.32);
}

// ---- Exact Hillaire 2020 Horizon-Focused Transmittance LUT Mapping ----
vec2 atmoTransParamsToUV(float r, float mu) {
    float H = sqrt(max(ATMO_TOP_R * ATMO_TOP_R - ATMO_BOTTOM_R * ATMO_BOTTOM_R, 1e-4));
    float rho = sqrt(max(r * r - ATMO_BOTTOM_R * ATMO_BOTTOM_R, 0.0));
    float disc = r * r * (mu * mu - 1.0) + ATMO_TOP_R * ATMO_TOP_R;
    float d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    float dMin = ATMO_TOP_R - r;
    float dMax = rho + H;
    float u = (d - dMin) / max(dMax - dMin, 1e-4);
    float v = rho / H;
    return clamp(vec2(u, v), vec2(0.0), vec2(1.0));
}

void atmoTransUVToParams(vec2 uv, out float r, out float mu) {
    float H = sqrt(max(ATMO_TOP_R * ATMO_TOP_R - ATMO_BOTTOM_R * ATMO_BOTTOM_R, 1e-4));
    float rho = H * uv.y;
    r = sqrt(rho * rho + ATMO_BOTTOM_R * ATMO_BOTTOM_R);
    float dMin = ATMO_TOP_R - r;
    float dMax = rho + H;
    float d = dMin + uv.x * (dMax - dMin);
    mu = (d <= 1e-4) ? 1.0 : ((H * H - rho * rho - d * d) / max(2.0 * r * d, 1e-4));
    mu = clamp(mu, -1.0, 1.0);
}

vec2 atmoClampTexelUV(vec2 uv, ivec2 res) {
    vec2 fRes = vec2(res);
    return (clamp(uv, vec2(0.0), vec2(1.0)) * (fRes - 1.0) + 0.5) / fRes;
}

#ifndef STAGE_ATMO_TRANS_WRITE
vec3 sampleAtmoTransmittance(float r, float mu) {
    vec2 uv = atmoClampTexelUV(atmoTransParamsToUV(r, mu), ATMO_TRANS_RES);
    return max(textureLod(atmoTransTex, uv, 0.0).rgb, vec3(0.0));
}
#endif

#ifndef STAGE_ATMO_MULTI_WRITE
vec3 sampleAtmoMultiScatter(float r, float sunMu) {
    vec2 uv = vec2(0.5 + 0.5 * clamp(sunMu, -1.0, 1.0),
                   saturate((r - ATMO_BOTTOM_R) / (ATMO_TOP_R - ATMO_BOTTOM_R)));
    uv = atmoClampTexelUV(uv, ATMO_MULTI_RES);
    return max(textureLod(atmoMultiTex, uv, 0.0).rgb, vec3(0.0));
}
#endif

// ---- Sun-Symmetric Non-Linear Horizon-Focused Sky-View LUT Mapping ----
// Azimuth symmetry phi in [0, PI] doubles angular resolution and removes any +-PI seam!
vec2 atmoDirToSkyUV(vec3 dir, vec3 sunDir) {
    float elev = asin(clamp(dir.y, -1.0, 1.0));
    float s = (elev >= 0.0) ? 1.0 : -1.0;
    float v = 0.5 + 0.5 * s * sqrt(abs(elev) * (2.0 / PI));

    vec2 sunXZ = sunDir.xz;
    float sunXZLen = length(sunXZ);
    vec2 sunH = (sunXZLen > 1e-4) ? (sunXZ / sunXZLen) : vec2(1.0, 0.0);

    vec2 dirXZ = dir.xz;
    float dirXZLen = length(dirXZ);
    float cosPhi = (dirXZLen > 1e-4) ? clamp(dot(dirXZ / dirXZLen, sunH), -1.0, 1.0) : 1.0;
    float u = acos(cosPhi) * INV_PI;
    return vec2(u, v);
}

vec3 atmoSkyUVToDir(vec2 uv, vec3 sunDir) {
    float vSign = (uv.y >= 0.5) ? 1.0 : -1.0;
    float vRel = (uv.y - 0.5) * 2.0;
    float elev = vSign * (vRel * vRel) * HALF_PI;

    float cosE = cos(elev);
    float sinE = sin(elev);
    float phi = clamp(uv.x, 0.0, 1.0) * PI;

    vec2 sunXZ = sunDir.xz;
    float sunXZLen = length(sunXZ);
    vec2 fwd = (sunXZLen > 1e-4) ? (sunXZ / sunXZLen) : vec2(1.0, 0.0);
    vec2 right = vec2(-fwd.y, fwd.x);
    vec2 horiz = fwd * cos(phi) + right * sin(phi);
    return normalize(vec3(horiz.x * cosE, sinE, horiz.y * cosE));
}

// ---- Procedural Stars + Subtle Galactic Band & Lunar Surface Detail ----
float atmoHash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

vec3 atmoHash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

float atmoValueNoise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = atmoHash31(i + vec3(0, 0, 0));
    float n100 = atmoHash31(i + vec3(1, 0, 0));
    float n010 = atmoHash31(i + vec3(0, 1, 0));
    float n110 = atmoHash31(i + vec3(1, 1, 0));
    float n001 = atmoHash31(i + vec3(0, 0, 1));
    float n101 = atmoHash31(i + vec3(1, 0, 1));
    float n011 = atmoHash31(i + vec3(0, 1, 1));
    float n111 = atmoHash31(i + vec3(1, 1, 1));
    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}

vec3 evalStarfield(vec3 dir, vec3 sunDir) {
    float starScale = float(STAR_INTENSITY);
    if (starScale <= 0.0 || dir.y <= 0.005) return vec3(0.0);

    // Stars emerge smoothly during nautical/astronomical twilight as sun dips below horizon.
    float nightMask = smoothstep(0.03, -0.14, sunDir.y) * (1.0 - rainStrength);
    if (nightMask <= 1e-3) return vec3(0.0);

    // Rotate star sphere slowly with worldTime so constellations drift naturally across the night sky.
    float rot = float(worldTime) * (TWO_PI / 24000.0);
    float cr = cos(rot), sr = sin(rot);
    vec3 sDir = vec3(dir.x * cr - dir.y * sr, dir.x * sr + dir.y * cr, dir.z);

    vec3 stars = vec3(0.0);
    // Layer 1: bright landmark stars with realistic O/B/A/G/K blackbody color tints.
    vec3 p1 = sDir * 135.0;
    vec3 c1 = floor(p1);
    vec3 h1 = atmoHash33(c1);
    vec3 off1 = fract(p1) - (0.18 + 0.64 * h1);
    float d1 = dot(off1, off1);
    if (h1.x > 0.965) {
        float lum = pow(saturate(1.0 - d1 * 28.0), 4.0) * pow((h1.x - 0.965) / 0.035, 1.5);
        float twinkle = 0.85 + 0.15 * sin(frameTimeCounter * (2.5 + h1.y * 4.0) + h1.z * TWO_PI);
        vec3 tint = mix(vec3(0.68, 0.82, 1.0), vec3(1.0, 0.82, 0.58), h1.y);
        stars += tint * (lum * twinkle * 0.42);
    }

    // Layer 2: denser faint starfield.
    vec3 p2 = sDir * 280.0;
    vec3 c2 = floor(p2);
    vec3 h2 = atmoHash33(c2 + 19.7);
    vec3 off2 = fract(p2) - (0.20 + 0.60 * h2);
    float d2 = dot(off2, off2);
    if (h2.x > 0.955) {
        float lum = pow(saturate(1.0 - d2 * 24.0), 3.5) * ((h2.x - 0.955) / 0.045);
        vec3 tint = mix(vec3(0.75, 0.86, 1.0), vec3(1.0, 0.92, 0.80), h2.z);
        stars += tint * (lum * 0.14);
    }

    // Subtle Milky Way galactic band tilted across the celestial sphere.
    float galPlane = dot(sDir, normalize(vec3(0.32, 0.24, 0.91)));
    float band = exp(-galPlane * galPlane * 16.0);
    if (band > 0.02) {
        float n1 = atmoValueNoise3D(sDir * 6.0);
        float n2 = atmoValueNoise3D(sDir * 14.0 + n1 * 1.2);
        float dust = smoothstep(0.28, 0.78, n1 * 0.6 + n2 * 0.4);
        stars += vec3(0.010, 0.014, 0.024) * (band * dust);
    }

    return stars * (nightMask * starScale);
}

#if !defined(STAGE_ATMO_TRANS_WRITE) && !defined(STAGE_ATMO_MULTI_WRITE) && !defined(STAGE_ATMO_SKY_WRITE)

// Physical top-of-atmosphere solar/lunar irradiance attenuated by exact atmospheric transmittance LUT!
vec3 getDirectLightColor() {
    vec3 sunDir = getSunDir();
    float camR = atmoCameraRadius();
    float rainAtten = 1.0 - 0.84 * rainStrength;

    if (!isNight()) {
        float sunMu = sunDir.y;
        float horizonFade = smoothstep(-0.01, 0.045, sunMu);
        if (horizonFade <= 1e-4) return vec3(0.0);
        vec3 trans = sampleAtmoTransmittance(camR, max(sunMu, 0.0));
        return ATMO_SUN_IRRADIANCE * trans * (horizonFade * rainAtten);
    } else {
        vec3 moonDir = -sunDir;
        float moonMu = moonDir.y;
        float horizonFade = smoothstep(-0.01, 0.06, moonMu);
        if (horizonFade <= 1e-4) return vec3(0.0);
        vec3 trans = sampleAtmoTransmittance(camR, max(moonMu, 0.0));
        return ATMO_MOON_IRRADIANCE * trans * (horizonFade * rainAtten);
    }
}

// Physical Sky-View LUT evaluation + optional Sun Disc (limb darkening), Textured Moon Disc, and Stars.
vec3 evalSky(vec3 dir, bool includeDisc) {
    vec3 d = normalize(dir);
    vec3 sunDir = getSunDir();
    vec2 uv = atmoClampTexelUV(atmoDirToSkyUV(d, sunDir), ATMO_SKY_RES);
    vec3 sky = max(textureLod(atmoSkyTex, uv, 0.0).rgb, vec3(0.0));

    if (includeDisc) {
        float camR = atmoCameraRadius();
        float horizonMask = smoothstep(-0.018, 0.012, d.y);
        float clearMask = 1.0 - rainStrength;

        if (horizonMask > 1e-4 && clearMask > 1e-4) {
            vec3 viewTrans = sampleAtmoTransmittance(camR, max(d.y, 0.0));

            // 1) Physical Sun Disc with realistic limb darkening + exact atmospheric reddening.
            const float sunCosRadius = 0.99952; // ~1.78 deg angular radius (visually balanced for MC + Bloom)
            float sunCos = dot(d, sunDir);
            if (sunCos > sunCosRadius - 0.0004) {
                float discEdge = smoothstep(sunCosRadius - 0.0002, sunCosRadius + 0.0001, sunCos);
                float r2 = saturate((1.0 - sunCos) / (1.0 - sunCosRadius));
                float limb = 0.45 + 0.55 * sqrt(max(1.0 - r2, 0.0));
                sky += ATMO_SUN_IRRADIANCE * viewTrans * (discEdge * limb * 3.6 * horizonMask * clearMask);
            }

            // 2) Textured Lunar Disc with gentle procedural maria shading.
            vec3 moonDir = -sunDir;
            const float moonCosRadius = 0.99958;
            float moonCos = dot(d, moonDir);
            if (moonCos > moonCosRadius - 0.0004) {
                float discEdge = smoothstep(moonCosRadius - 0.0002, moonCosRadius + 0.0001, moonCos);
                vec3 moonUp = abs(moonDir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
                vec3 mT = normalize(cross(moonUp, moonDir));
                vec3 mB = cross(moonDir, mT);
                vec2 mUV = vec2(dot(d, mT), dot(d, mB)) / sqrt(max(1.0 - moonCosRadius * moonCosRadius, 1e-6));
                float maria = 0.72 + 0.28 * atmoValueNoise3D(vec3(mUV * 3.2, 1.7));
                float limb = 0.65 + 0.35 * sqrt(max(1.0 - dot(mUV, mUV), 0.0));
                sky += vec3(1.25, 1.32, 1.45) * viewTrans * (discEdge * maria * limb * horizonMask * clearMask);
            }

            // 3) Atmospheric-extincted procedural starfield & subtle Milky Way.
            vec3 stars = evalStarfield(d, sunDir);
            if (maxOf(stars) > 0.0) {
                sky += stars * viewTrans * horizonMask;
            }
        }
    }

    return sky;
}

// Physical Aerial Perspective (Distant Atmospheric Haze + Seamless Horizon Fade):
// - Analytically integrates height-dependent Rayleigh + Mie extinction along the view ray.
// - In-scattering samples the live Sky-View LUT along viewRay so distant terrain melts
//   100% seamlessly into the physical horizon at noon, sunset, blue hour, night, and rain!
void evalAerialPerspective(vec3 worldPos, vec3 rayDir, float skyLight,
                           out vec3 outTransmittance, out vec3 outInscatter) {
    #ifndef AERIAL_PERSPECTIVE
        outTransmittance = vec3(1.0);
        outInscatter = vec3(0.0);
        return;
    #endif

    float dist = length(worldPos);
    if (dist <= 2.0) {
        outTransmittance = vec3(1.0);
        outInscatter = vec3(0.0);
        return;
    }

    // Scale world meters to visual-physical atmospheric km path (clear up close, rich depth at 100..400 blocks).
    float density = float(FOG_DENSITY);
    float camY = cameraPosition.y;
    float surfY = camY + worldPos.y;
    float avgAltBlocks = max(0.5 * (camY + surfY) - 62.0, 0.0);
    float heightFactor = exp(-avgAltBlocks * (1.0 / 120.0));

    float opticalKm = max(dist - 8.0, 0.0) * (0.0115 * density) * (0.35 + 0.65 * heightFactor);
    float mieBoost = mix(1.0, 3.8, rainStrength);
    vec3 sigmaR = ATMO_RAYLEIGH_SCATTER;
    vec3 sigmaM = vec3(4.2e-3) * mieBoost;
    vec3 totalExt = sigmaR + sigmaM;

    // Suppress near-field fog inside enclosed caves/interiors while preserving true far-horizon blend.
    float indoorMask = mix(smoothstep(0.08, 0.55, skyLight), 1.0, smoothstep(48.0, 140.0, dist));
    vec3 T = exp(-totalExt * (opticalKm * indoorMask));

    // Smooth far-plane chunk-border dissolution so unloaded render edge never shows a hard cliff.
    float borderFade = smoothstep(far * 0.72, far * 0.99, dist) * indoorMask;
    T *= (1.0 - borderFade);

    // Sample live Sky-View LUT slightly lifted toward horizon so downward-angled mountains still
    // inherit authentic horizon haze rather than dark ground bounce.
    vec3 fogDir = normalize(vec3(rayDir.x, max(rayDir.y, 0.015), rayDir.z));
    vec3 horizonCol = evalSky(fogDir, false);

    // Subtle forward Mie golden glow when looking toward low sun across distant terrain.
    vec3 sunDir = getSunDir();
    float VdotL = dot(rayDir, sunDir);
    float sunDay = smoothstep(-0.04, 0.14, sunDir.y) * (1.0 - rainStrength);
    float forwardHalo = pow(saturate(VdotL), 6.0) * sunDay * 0.35;
    vec3 fogRadiance = horizonCol * (1.0 + forwardHalo);

    outTransmittance = saturate(T);
    outInscatter     = max(fogRadiance * (1.0 - outTransmittance), vec3(0.0));
}

#endif

#endif
