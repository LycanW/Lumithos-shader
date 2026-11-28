#version 430 compatibility

#include "/lib/Atmosphere.glsl"
#include "/lib/Raytrace.glsl"

layout(local_size_x = 64) in;
const ivec3 workGroups = ivec3(1, 1, 1);

uniform ivec2 eyeBrightness;
layout(rgba16f) uniform writeonly image2D handSkyIrradianceImg;

shared vec3 skyDirections[64];
shared vec3 skyRadiance[64];

void main() {
    uint index = gl_LocalInvocationIndex;
    if (eyeBrightness.y <= 0) {
        if (index < 6u) imageStore(handSkyIrradianceImg, ivec2(index, 0), vec4(0.0));
        return;
    }

    // Fixed equal-area directions in the upper hemisphere. All hand pixels share
    // the same visibility estimate, independent of the viewmodel animation.
    float y = (float(index) + 0.5) / float(gl_WorkGroupSize.x);
    float azimuth = float(index) * 2.39996323;
    float radius = sqrt(1.0 - y * y);
    vec3 direction = vec3(radius * cos(azimuth), y, radius * sin(azimuth));
    TraceHit hit = traceScene(vec3(0.0), direction, DIFFUSE_MAX_STEPS, 0.0, true);
    skyDirections[index] = direction;
    skyRadiance[index] = hit.exitedGrid ? evalSky(direction, false) * hit.throughput : vec3(0.0);
    barrier();

    if (index < 6u) {
        vec3 axis = vec3(0.0);
        axis[index / 2u] = (index & 1u) == 0u ? 1.0 : -1.0;
        vec3 irradiance = vec3(0.0);
        for (uint ray = 0u; ray < gl_WorkGroupSize.x; ++ray) {
            irradiance += skyRadiance[ray] * max(dot(axis, skyDirections[ray]), 0.0);
        }
        // 2*pi is the sampled solid angle. Lambertian normalization divides by pi.
        irradiance *= 2.0 / float(gl_WorkGroupSize.x);
        imageStore(handSkyIrradianceImg, ivec2(index, 0), vec4(irradiance, 0.0));
    }
}
