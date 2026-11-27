#ifndef LIB_SPECULAR_TRACE_GLSL
#define LIB_SPECULAR_TRACE_GLSL

#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/RadianceCache.glsl"

vec3 sampleReflectionDirection(vec3 viewDir, vec3 shadeNormal, float roughness, ivec2 samplePx, int sampleIndex, out vec3 H) {
    H = shadeNormal;
    if (usesMirrorDirection(roughness)) return reflect(-viewDir, shadeNormal);
    return sampleGGXVNDF(viewDir, shadeNormal, roughness, sampleBlueNoiseRG(samplePx, sampleIndex), H);
}

vec3 traceVoxelReflection(vec3 rayOrig, vec3 rayDir, float roughness, ivec2 bnPx, int smoothSteps, bool includeSkyDisc, out float hitDist) {
    vec3 radiance = vec3(0.0);
    hitDist = 0.0; // No measured geometry distance for a cache fallback.
    // Full-res atlas LOD 0 for razor-sharp mirror reflections of 6-face & rotated block textures!
    float specLod = roughness < 0.25 ? 0.0 : 1.0;
    int maxSpecSteps = roughness < 0.22 ? smoothSteps : 48;
    TraceHit hit = traceScene(rayOrig, rayDir, maxSpecSteps, specLod, true);
    if (hit.hit) {
        hitDist = hit.dist;
        float emission = coveredHitEmission(hit, rayOrig);
        radiance += hit.albedo * (emission * EMISSIVE_SCALE);

        vec3 lightDir = getShadowLightDir();
        float hitNdotL = max(dot(hit.normal, lightDir), 0.0);
        vec3 direct = vec3(0.0);
        if (hitNdotL > 0.0) {
            vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
            direct = getDirectLightColor() * (hitNdotL * vis);
        }

        vec3 skyAmb = evalSky(hit.normal, false) * (0.15 + 0.85 * hit.skyLight);
        vec3 bounce = (roughness < 0.25)
            ? queryRadianceCacheSmooth(hit.worldPos, hit.normal, skyAmb, 0u, stbn2D(bnPx, frameCounter, 6u))
            : queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, stbn2D(bnPx, frameCounter, 6u));
        radiance += secondaryHitLighting(hit, rayDir, lightDir, direct, bounce, usesMirrorDirection(roughness));
        radiance *= hit.throughput;
    } else if (hit.exitedGrid) {
        hitDist = 65504.0; // Distinguish a sky ray from a missing hit in the reflection filter.
        radiance = evalSky(rayDir, includeSkyDisc) * hit.throughput;
    } else {
        radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false) * hit.throughput;
    }
    return radiance;
}

#endif
