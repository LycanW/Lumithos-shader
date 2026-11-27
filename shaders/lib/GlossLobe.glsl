#ifndef LIB_GLOSS_LOBE_GLSL
#define LIB_GLOSS_LOBE_GLSL

// Fit two Gaussian radial CDFs to the small-angle, near-normal GGX CDF t^2 / (1 + t^2).
// Widths are standard deviations in units of 2 * perceptualRoughness^2.
const float GLOSS_CORE_WIDTH = 0.634;
const float GLOSS_TAIL_WIDTH = 2.004;
const float GLOSS_TAIL_MASS = 0.339;

mat3 glossLobeBasis(vec3 mirror, vec3 view) {
    vec3 across = cross(mirror,-view);
    float len2 = dot(across,across);
    across = len2 > 1e-8 ? across * inversesqrt(len2) : buildTBN(mirror)[0];
    return mat3(across,cross(mirror,across),mirror);
}

vec3 sampleGlossCore(vec3 view, vec3 normal, float roughness, vec2 noise) {
    vec3 mirror = reflect(-view, normal);
    mat3 basis = glossLobeBasis(mirror, view);
    float angle = TWO_PI * noise.x;
    float grazing = saturate(dot(normal, view));
    // Truncated Gaussian in the tangent plane, compressed across the plane of incidence.
    float radial = GLOSS_CORE_WIDTH * sqrt(-2.0 * log(1.0 - 0.95606 * noise.y));
    vec3 offset = (cos(angle) * grazing) * basis[0] + sin(angle) * basis[1];
    return normalize(mirror + (2.0 * roughness * roughness * radial) * offset);
}

// Frame-local view-space hit points. The working atlas reuses this tile for fast history in deferred5.
ivec2 glossHitPixel(ivec2 p) {
    return p + ivec2(int(screenSize().x),0);
}

#endif
