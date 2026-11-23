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

// Analytical 4x4 Bayer-stratified R2 spatiotemporal 2D sequence:
// Steps full integer R2 indices 0..15 across each 4x4 tile so every tile covers [0,1)^2 uniformly.
vec2 stbn2D(ivec2 px, int frame, uint salt) {
    ivec2 b = px & 3;
    int bayer16 = (((b.x ^ b.y) & 1) << 3) | ((b.y & 1) << 2) | ((((b.x ^ b.y) >> 1) & 1) << 1) | ((b.y >> 1) & 1);
    uint tileSeed = pcgHash(uint(px.x >> 2) * 374761393u + uint(px.y >> 2) * 668265263u + salt * 2246822519u);
    vec2 tileShift = vec2(float(tileSeed & 65535u), float(tileSeed >> 16u)) * (1.0 / 65536.0);
    float idx = float(((frame & 255) << 4) + bayer16) + 1.0;
    return fract(vec2(0.7548776662, 0.5698402909) * idx + tileShift);
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

uint packNormal16(vec3 n) {
    uvec2 q = uvec2(round(saturate(octEncode(n)) * 255.0));
    return q.x | (q.y << 8u);
}

vec3 unpackNormal16(uint word) {
    return octDecode(vec2(word & 255u, (word >> 8u) & 255u) * (1.0 / 255.0));
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

// ---- GGX Specular VNDF Sampling (Heitz 2018) ----
vec3 sampleGGXVNDF(vec3 viewDir, vec3 normal, float roughness, vec2 u) {
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
    vec3 H = tbn * Ne;
    return reflect(-viewDir, H);
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
