#ifndef LIB_COMMON_GLSL
#define LIB_COMMON_GLSL

#define PI      3.14159265359
#define TWO_PI  6.28318530718
#define HALF_PI 1.57079632679
#define INV_PI  0.31830988618

float rcp(float x) { return 1.0 / x; }
float saturate(float x) { return clamp(x, 0.0, 1.0); }
vec2  saturate(vec2 x)  { return clamp(x, 0.0, 1.0); }
vec3  saturate(vec3 x)  { return clamp(x, 0.0, 1.0); }
float maxOf(vec3 v)     { return max(v.x, max(v.y, v.z)); }
float minOf(vec3 v)     { return min(v.x, min(v.y, v.z)); }
float luma(vec3 c)      { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

vec3 srgbToLinear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 linearToSrgb(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); }

// Coefficients per block. Biome tint affects absorption; scattering is achromatic.
const float WATER_SCATTERING = 0.007;

vec3 waterExtinction(vec3 biomeColor) {
    vec3 absorption = vec3(0.033, 0.011, 0.004) - 0.04 * log(clamp(biomeColor, vec3(0.08), vec3(1.0)));
    return absorption + WATER_SCATTERING;
}

// ---- Stained-Glass / Translucent Light Tinting & Colored Emitters ----
// Physical translucent filter transmittance: keeps white glass clear, black/tinted glass dark,
// and delivers rich chromatic tinting for single- or multi-layer stained glass.
vec3 evalGlassTransmittance(vec3 srgbCol, float alpha) {
    vec3 lin = srgbToLinear(saturate(srgbCol));
    float maxC = maxOf(lin);
    vec3 chroma = (maxC > 1e-4) ? (lin / maxC) : vec3(0.0);
    vec3 deepTint = pow(saturate(chroma), vec3(1.35)) * mix(0.05, 0.96, pow(saturate(maxC), 0.42));
    float tintStrength = saturate(alpha * 2.2);
    return mix(vec3(1.0), deepTint, tintStrength);
}

// Vibrant hue-accurate emissive color for colored light sources (torches, soul fire, redstone,
// froglights, sea lanterns, amethyst, crying obsidian, modded colored lamps).
// Masks dark non-emissive parts (lantern iron cages, torch sticks, ore stone) on vanilla emitters
// while leaving LabPBR per-texel emission maps untouched.
vec3 evalEmitterColor(vec3 albedoLinear, float blockEmit, float pbrEmit, out float emitStrength) {
    float maxC = maxOf(albedoLinear);
    float mask = (pbrEmit > 0.0) ? 1.0 : smoothstep(0.12, 0.42, maxC);
    emitStrength = max(blockEmit * mask, pbrEmit);
    if (emitStrength <= 1e-4 || maxC <= 1e-4) {
        emitStrength = 0.0;
        return vec3(0.0);
    }
    vec3 chroma = saturate(albedoLinear / maxC);
    vec3 richChroma = pow(chroma, vec3(1.55));
    float lumBase = max(luma(albedoLinear), 0.15 * maxC);
    float lumRich = max(luma(richChroma), 0.12);
    float energy = mix(maxC, lumBase / lumRich, 0.35);
    return richChroma * clamp(energy, 0.25, 1.35);
}

vec3 evalDirectEmitterColor(vec3 albedoLinear) {
    float maxC = maxOf(albedoLinear);
    if (maxC <= 1e-4) return vec3(0.0);
    vec3 chroma = saturate(albedoLinear / maxC);
    vec3 richChroma = pow(chroma, vec3(1.40));
    float lumBase = max(luma(albedoLinear), 0.15 * maxC);
    float lumRich = max(luma(richChroma), 0.12);
    float energy = mix(maxC, lumBase / lumRich, 0.35);
    return richChroma * clamp(energy, 0.25, 1.35);
}

// ---- Fast PCG Hash & RNG ----
uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint initRng(ivec2 pixel, uint frame, uint salt) {
    return pcgHash(uint(pixel.x) + uint(pixel.y) * 4099u + frame * 16777619u + salt * 2654435761u);
}

float nextFloat(inout uint rng) {
    rng = pcgHash(rng);
    return float(rng >> 8u) * (1.0 / 16777216.0);
}

vec2 nextVec2(inout uint rng) {
    return vec2(nextFloat(rng), nextFloat(rng));
}

// ---- True Spatiotemporal Blue Noise (Heitz 2019: 16x16 Bayer-Ranked 2D Sobol + Stride-1 Temporal R2) ----
// - Zero 4x4 white-noise tile seams: exact recursive 16x16 blue-noise stratification (33x less LF energy!).
// - Stride-1 R2 temporal rotation (never frame<<4!) preserves exact spatial blue noise on every frame
//   while giving every single pixel optimal O(1/N) quasi-Monte-Carlo convergence over time!
uint bayer256(ivec2 p) {
    uint x = uint(p.x) & 15u;
    uint y = uint(p.y) & 15u;
    uint z = x ^ y;
    uint r = ((z & 1u) << 7u) | ((y & 1u) << 6u)
           | ((z & 2u) << 4u) | ((y & 2u) << 3u)
           | ((z & 4u) << 1u) | ((y & 4u) << 0u)
           | ((z & 8u) >> 2u) | ((y & 8u) >> 3u);
    return r;
}

uint sobolDim1_8(uint k) {
    uint v = 0u;
    if ((k &   1u) != 0u) v ^= 0x80u;
    if ((k &   2u) != 0u) v ^= 0xC0u;
    if ((k &   4u) != 0u) v ^= 0xA0u;
    if ((k &   8u) != 0u) v ^= 0xF0u;
    if ((k &  16u) != 0u) v ^= 0x88u;
    if ((k &  32u) != 0u) v ^= 0xCCu;
    if ((k &  64u) != 0u) v ^= 0xAAu;
    if ((k & 128u) != 0u) v ^= 0xFFu;
    return v;
}

vec2 stbn2D(ivec2 px, int frame, uint salt) {
    // Rotate pixel grid lightly by salt so independent dimensions never correlate.
    ivec2 sp = px + ivec2(int((salt * 5u) & 15u), int((salt * 11u) & 15u));
    uint idx = bayer256(sp);
    uint k   = bitfieldReverse(idx) >> 24u;
    vec2 bn  = vec2(float(idx) + 0.5, float(sobolDim1_8(k)) + 0.5) * (1.0 / 256.0);
    // Stride-1 R2 temporal sequence + salt phase shift (constant across screen -> preserves spatial blue noise!)
    float t  = float(frame & 2047) + float(salt) * 17.0;
    return fract(bn + vec2(0.7548776662, 0.5698402909) * t);
}

// ---- 3D Spatiotemporal Blue Noise (4x4x4 Recursive 3D Bayer-Sobol + Stride-1 Temporal R2) ----
// Guarantees face-adjacent 3D Radiance Cache cells sample opposite hemispheres (0.000000 4x4x4 variance!),
// turning 1-SPP Monte Carlo probe errors strictly into high-frequency blue noise that cancels out cleanly!
uint bayer3D64(ivec3 c) {
    uint x = uint(c.x) & 3u;
    uint y = uint(c.y) & 3u;
    uint z = uint(c.z) & 3u;
    uint xy = x ^ y;
    uint xyz = xy ^ z;
    return ((xyz & 1u) << 5u) | ((xy & 1u) << 4u) | ((z & 1u) << 3u)
         | (((xyz >> 1u) & 1u) << 2u) | (((xy >> 1u) & 1u) << 1u) | ((z >> 1u) & 1u);
}

vec3 stbn3D(ivec3 cell, uint seq, uint salt) {
    ivec3 sc = cell + ivec3(int(salt & 3u), int((salt * 3u) & 3u), int((salt * 5u) & 3u));
    uint idx = bayer3D64(sc);
    uint k   = bitfieldReverse(idx) >> 26u;
    uint sy  = 0u;
    if ((k &  1u) != 0u) sy ^= 0x20u;
    if ((k &  2u) != 0u) sy ^= 0x30u;
    if ((k &  4u) != 0u) sy ^= 0x28u;
    if ((k &  8u) != 0u) sy ^= 0x3Cu;
    if ((k & 16u) != 0u) sy ^= 0x22u;
    if ((k & 32u) != 0u) sy ^= 0x33u;
    uint sz  = 0u;
    if ((k &  1u) != 0u) sz ^= 0x20u;
    if ((k &  2u) != 0u) sz ^= 0x10u;
    if ((k &  4u) != 0u) sz ^= 0x38u;
    if ((k &  8u) != 0u) sz ^= 0x24u;
    if ((k & 16u) != 0u) sz ^= 0x12u;
    if ((k & 32u) != 0u) sz ^= 0x29u;
    vec3 bn  = (vec3(float(idx), float(sy), float(sz)) + 0.5) * (1.0 / 64.0);
    float t  = float(seq & 2047u) + float(salt) * 13.0;
    return fract(bn + vec3(0.8191725134, 0.6710436067, 0.5497004779) * t);
}

// ---- Orthonormal Basis (Duff et al. 2017) ----
mat3 buildTBN(vec3 n) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    vec3 t = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    vec3 bi = vec3(b, s + n.y * n.y * a, -n.y);
    return mat3(t, bi, n);
}

// ---- Octahedral Normal Encoding ----
vec2 octEncode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-8);
    if (n.z < 0.0) {
        vec2 s = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return n.xy * 0.5 + 0.5;
}

vec3 octDecode(vec2 e) {
    vec2 f = e * 2.0 - 1.0;
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (n.z < 0.0) {
        vec2 s = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * s;
    }
    return normalize(n);
}

// Use EVEN quantization scale 254.0 (0..254) so exact center 0.5 maps to integer 127.0 instead of 127.5!
// With odd 255.0, all 6 axis-aligned Minecraft faces (±X, ±Y, ±Z) land dead-center on the 127.5 rounding
// cliff, causing ±1e-7 camera-matrix float noise to snap flat mirror normals by ±0.45° (±10px mirror jump!).
uint packNormal16(vec3 n) {
    uvec2 q = uvec2(round(saturate(octEncode(n)) * 254.0));
    return q.x | (q.y << 8u);
}

vec3 unpackNormal16(uint word) {
    return octDecode(vec2(word & 255u, (word >> 8u) & 255u) * (1.0 / 254.0));
}

// ---- YCoCg Color Transform ----
vec3 rgbToYCoCg(vec3 rgb) {
    float y  = dot(rgb, vec3(0.25, 0.5, 0.25));
    float co = 0.5 * (rgb.r - rgb.b);
    float cg = 0.5 * rgb.g - 0.25 * (rgb.r + rgb.b);
    return vec3(y, co, cg);
}

vec3 yCoCgToRgb(vec3 ycc) {
    float tmp = ycc.x - ycc.z;
    return max(vec3(tmp + ycc.y, ycc.x + ycc.z, tmp - ycc.y), vec3(0.0));
}

// ---- Sampling Helpers ----
vec3 sampleCosineHemisphere(vec3 n, vec2 u) {
    float phi = TWO_PI * u.x;
    float cosTheta = sqrt(max(1.0 - u.y, 0.0));
    float sinTheta = sqrt(max(u.y, 0.0));
    return buildTBN(n) * vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

vec3 sampleVMF(vec3 mu, float kappa, vec2 u) {
    float eps = max(u.x, 1e-6);
    float cosTheta = clamp(1.0 + log(eps + (1.0 - eps) * exp(-2.0 * kappa)) / kappa, -1.0, 1.0);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = TWO_PI * u.y;
    return buildTBN(mu) * vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float pdfVMF(vec3 w, vec3 mu, float kappa) {
    if (kappa < 1e-3) return 1.0 / (4.0 * PI);
    float denom = TWO_PI * (1.0 - exp(-2.0 * kappa));
    return (kappa / max(denom, 1e-6)) * exp(kappa * (dot(w, mu) - 1.0));
}

// ---- GGX Specular VNDF Sampling (Heitz 2018) & Physical Smith GGX BRDF ----
vec3 sampleGGXVNDF(vec3 viewDir, vec3 normal, float roughness, vec2 u, out vec3 outH) {
    float alpha = max(roughness * roughness, 0.002);
    mat3 tbn = buildTBN(normal);
    vec3 Ve = transpose(tbn) * viewDir;
    vec3 Vh = normalize(vec3(alpha * Ve.x, alpha * Ve.y, max(Ve.z, 1e-4)));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 1e-6 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = TWO_PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    vec3 Ne = normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
    outH = tbn * Ne;
    return reflect(-viewDir, outH);
}

// Height-correlated Smith GGX visibility: G2(V, L) / (4 * NdotV * NdotL)
float visSmithGGX(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float vTerm = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float lTerm = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(vTerm + lTerm, 1e-5);
}

// Exact Heitz 2018 VNDF estimator geometric weight: G2(V, L) / G1(V)
float smithGGX_G2OverG1(float NdotV, float NdotL, float alpha) {
    float a2 = alpha * alpha;
    float vTerm = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float lTerm = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return saturate((NdotV * NdotL + vTerm) / max(vTerm + lTerm, 1e-5));
}

// Analytical GGX split-sum DFG LUT (Lazarov 2013 / Karis):
// Used for unbiased SVGF specular demodulation/remodulation and diffuse energy conservation.
vec2 envBRDFScaleBias(float NdotV, float roughness) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return clamp(vec2(-1.04, 1.04) * a004 + r.zw, vec2(0.0), vec2(1.0));
}

vec3 evalEnvBRDF(vec3 F0, float NdotV, float roughness) {
    vec2 ab = envBRDFScaleBias(NdotV, roughness);
    return F0 * ab.x + vec3(ab.y);
}

vec3 fresnelSchlick(vec3 F0, float cosTheta) {
    float f = pow(saturate(1.0 - cosTheta), 5.0);
    return F0 + (1.0 - F0) * f;
}

// ---- Half4x16 Helpers ----
uvec2 packHalf4x16(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}

vec4 unpackHalf4x16(uvec2 v) {
    return vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
}

// ---- Explicit Round-to-Nearest-Even FP16 ----
vec4 roundHalf(vec4 v) {
    uvec4 b = floatBitsToUint(v);
    vec4 r = uintBitsToFloat((b + uvec4(0x00000fffu) + ((b >> 13u) & uvec4(1u))) & uvec4(0xffffe000u));
    r = mix(r, roundEven(v * 16777216.0) * 0.000000059604644775390625, lessThan(abs(v), vec4(0.00006103515625)));
    return mix(r, v, greaterThanEqual(b & uvec4(0x7fffffffu), uvec4(0x7f800000u)));
}

#endif
