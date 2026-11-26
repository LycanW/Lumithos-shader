#ifndef LIB_LIGHT_FILTER_GLSL
#define LIB_LIGHT_FILTER_GLSL

#include "/lib/Resources.glsl"

uniform sampler2D lightGuideTex;
uniform sampler2D diffuseWorkTex;
uniform sampler2D diffuseHistoryTex;
#ifdef SPECULAR_REFLECTIONS
uniform sampler2D glossWorkTex;
uniform sampler2D glossHistoryTex;
#endif

struct FilterSurface {
    vec3 face, normal;
    float rough, z;
};

bool inFrame(ivec2 p) {
    return all(greaterThanEqual(p, ivec2(0))) && all(lessThan(p, ivec2(screenSize())));
}

FilterSurface filterSurface(ivec2 p) {
    vec4 g = texelFetch(lightGuideTex, p, 0);
    return FilterSurface(unpackNormal16(uint(g.x)), unpackNormal16(uint(g.y)), g.z, g.w);
}

// Four full-resolution tiles: slow/intermediate, fast, prepared sample, spatial result.
ivec2 workPixel(ivec2 p, int tile) {
    return p + ivec2(tile & 1, tile >> 1) * ivec2(screenSize());
}

bool diffuseTraced(ivec2 p) {
    #if DIFFUSE_TRACE_RES == 0
        int y = frameCounter & 1;
        return all(equal(p & 1, ivec2(((frameCounter >> 1) ^ y) & 1, y)));
    #elif DIFFUSE_TRACE_RES == 1
        return ((p.x ^ p.y ^ frameCounter) & 1) == 0;
    #else
        return true;
    #endif
}

bool glossTraced(ivec2 p) {
    #ifdef SPECULAR_CHECKERBOARD
        return ((p.x ^ p.y ^ frameCounter) & 1) == 0;
    #else
        return true;
    #endif
}

// Dot the difference between the two perspective view positions with the receiver plane normal.
struct FilterPlane { float incidence; vec2 pixelSlope; };
FilterPlane filterPlane(ivec2 p, vec3 face) {
    vec2 ndc = (vec2(p) + 0.5) * (2.0 * texelSize()) - 1.0;
    #ifdef TAA_ENABLED
        ndc -= taaJitter;
    #endif
    vec3 n = mat3(gbufferModelView) * face;
    vec2 slope = n.xy / vec2(gbufferProjection[0][0], gbufferProjection[1][1]);
    return FilterPlane(dot(ndc + gbufferProjection[2].xy, slope) - n.z, 2.0 * slope * texelSize());
}

float surfaceWeight(FilterSurface a, FilterSurface b, FilterPlane plane, ivec2 delta) {
    if (b.z <= 0.0 || dot(a.face, b.face) < 0.90) return 0.0;
    float gap = (b.z - a.z) * plane.incidence + b.z * dot(vec2(delta), plane.pixelSlope);
    float dz = max(0.008, a.z * texelSize().y * 2.0);
    return exp(-abs(gap) / dz) * pow(max(dot(a.normal, b.normal), 0.0), 8.0);
}

float glossWeight(float a, float b, float cosine) {
    if (a < 0.0 || b < 0.0) return 0.0;
    float width = max(a*a, 0.01);
    return exp(-abs(a-b) / width - max(1.0-cosine, 0.0) / (width*width));
}

// Fraction of the optical path beyond the reflector, bounded even for sky hits.
float reflectionFraction(float distance, float eyeDistance) {
    if (distance >= 65504.0) return 1.0;
    return distance / max(distance + eyeDistance, 1e-5);
}

// Checkerboard reconstruction and temporal history have different clocks.
bool diffuseField(ivec2 p) {
    #if DIFFUSE_TRACE_RES == 0
        return ((p.x ^ p.y ^ (frameCounter >> 1)) & 1) == 0;
    #else
        return diffuseTraced(p);
    #endif
}

float nextHistoryAge(float previousAge, float coverage) {
    return clamp((previousAge + 1.0) * sqrt(clamp(coverage, 0.0, 1.0)), 1.0, 31.0);
}

float temporalGain(float age, float cap, float observation) {
    return observation / clamp(age, 1.0, cap + 1.0);
}

float signalVariance(float mean, float secondMoment) {
    return max(secondMoment - mean * mean, 0.0);
}

float dominantFraction(float roughness, float cosine) {
    float headOn = 0.298475 * log(39.4115 - 39.0029 * roughness);
    return clamp(mix(headOn, 1.0, pow(1.0 - clamp(cosine,0.0,1.0), 10.8649)), 0.0, 1.0);
}

// The virtual image of a reflector has axial magnification 1 / (1 + 2*k*d).
float virtualImageDistance(float hit, float curvature, float incidence, float eyeDist, float noV, float dominant) {
    float inverseScale = 1.0 + 2.0 * curvature * incidence * hit;
    if (abs(inverseScale) < 1e-5) inverseScale = inverseScale < 0.0 ? -1e-5 : 1e-5;
    float silhouette = 1.0 + eyeDist * (1.0 - noV) * max(curvature,0.0);
    return dominant * hit / (inverseScale * silhouette);
}

vec3 clampResponsive(vec3 slow, vec3 fast, vec3 mean, vec3 sigma) {
    vec3 lo = min(fast, mean - 2.0*sigma), hi = max(fast, mean + 2.0*sigma);
    return clamp(slow, lo, hi);
}

float resetFraction(float slowLuma, float localLuma, float spatialSigma, float noiseSigma) {
    float band = 4.5 * spatialSigma + 0.5 * noiseSigma;
    return 0.5 * max(abs(slowLuma-localLuma)-band, 0.0) / (max(slowLuma, localLuma) + band + 1e-6);
}

#endif
