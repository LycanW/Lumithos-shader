#ifndef LIB_SPECULAR_TRACE_GLSL
#define LIB_SPECULAR_TRACE_GLSL

#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/RadianceCache.glsl"
#include "/lib/GlossLobe.glsl"

vec3 sampleReflectionDirection(vec3 viewDir, vec3 shadeNormal, float roughness, ivec2 samplePx, int sampleIndex, out vec3 H) {
    H = shadeNormal;
    if (usesMirrorDirection(roughness)) return reflect(-viewDir, shadeNormal);
    return sampleGGXVNDF(viewDir, shadeNormal, roughness, sampleBlueNoiseRG(samplePx, sampleIndex), H);
}

float specularBounceShare(float roughness, float f0) {
    return isLabPBRMetal(f0) ? 1.0 : 1.0 - smoothstep(0.50, SPECULAR_ROUGH_CUTOFF, roughness);
}

bool hasReflectionVoxel(ivec3 voxel) {
    return inVoxelBounds(voxel) && texelFetch(voxelGridTex, voxel, 0).r != 0u;
}

vec3 traceVoxelReflection(vec3 rayOrig, vec3 rayDir, float roughness, ivec2 bnPx, int smoothSteps, bool includeSkyDisc, out float hitDist) {
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    hitDist = 0.0; // No measured geometry distance for a cache fallback.
    vec3 lightDir = getShadowLightDir();
    for (int bounce = 0; bounce < SPECULAR_BOUNCES; ++bounce) {
        float specLod = roughness < 0.25 ? 0.0 : 1.0;
        int maxSpecSteps = roughness < 0.22 ? smoothSteps : 48;
        TraceHit hit = traceScene(rayOrig, rayDir, maxSpecSteps, specLod, true);
        throughput *= hit.throughput;
        // The spatial filter reconstructs the first hit position along the original ray.
        if (bounce == 0) hitDist = hit.hit ? hit.dist : (hit.exitedGrid ? 65504.0 : 0.0);
        if (!hit.hit) {
            // Sun and moon visibility at a surface hit is evaluated by the direct-light term.
            vec3 incoming = hit.exitedGrid
                ? evalSky(rayDir, bounce == 0 && includeSkyDisc)
                : queryRadianceCache(hit.worldPos, -rayDir, 0u, false);
            radiance += throughput * incoming;
            break;
        }
        if (hit.dist <= 0.0) break;

        float emission = coveredHitEmission(hit, rayOrig);
        radiance += throughput * hit.albedo * (emission * EMISSIVE_SCALE);

        float hitNdotL = max(dot(hit.normal, lightDir), 0.0);
        vec3 direct = vec3(0.0);
        if (hitNdotL > 0.0) {
            vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
            direct = getDirectLightColor() * (hitNdotL * vis);
        }

        #if SPECULAR_BOUNCES > 1
            vec3 view = -rayDir;
            float noV = dot(hit.normal, view);
            float share = bounce + 1 < SPECULAR_BOUNCES && noV > 1e-4
                ? specularBounceShare(hit.roughness, hit.f0) : 0.0;
        #else
            const float share = 0.0;
        #endif
        bool metal = isLabPBRMetal(hit.f0);
        vec3 cached = vec3(0.0);
        // The diffuse cache can close a broad lobe. A mirror needs directional incident radiance.
        if (!metal || (share <= 0.0 && !usesMirrorDirection(hit.roughness))) {
            vec2 noise = stbn2D(bnPx, frameCounter, 6u);
            if (roughness < 0.25) {
                vec3 skyAmb = evalSky(hit.normal, false) * (0.15 + 0.85 * hit.skyLight);
                cached = queryRadianceCacheSmooth(hit.worldPos, hit.normal, skyAmb, 0u, noise);
            } else {
                cached = queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, noise);
            }
        }
        vec3 local = secondaryHitLighting(hit, rayDir, lightDir, direct, cached, usesMirrorDirection(roughness));
        if (share <= 0.0) {
            radiance += throughput * local;
            break;
        }

        #if SPECULAR_BOUNCES > 1
            vec3 f0 = evalLabPBRF0(hit.f0, hit.albedo);
            vec3 envWeight = evalEnvBRDF(f0, noV, hit.roughness);
            radiance += throughput * local * (metal ? vec3(1.0) : vec3(1.0) - share * envWeight);

            vec3 nextDir = reflect(rayDir, hit.normal);
            vec3 nextWeight = fresnelSchlick(f0, noV);
            if (!usesMirrorDirection(hit.roughness)) {
                vec2 sampleUV = stbn2D(bnPx, frameCounter, 19u + uint(bounce));
                #if SPECULAR_LOBE_MODE == 0
                    vec3 h;
                    nextDir = sampleGGXVNDF(view, hit.normal, hit.roughness, sampleUV, h);
                    float noL = dot(hit.normal, nextDir);
                    if (noL <= 1e-4) break;
                    nextWeight = fresnelSchlick(f0, saturate(dot(view, h))) *
                        smithGGX_G2OverG1(noV, noL, max(hit.roughness * hit.roughness, 0.002));
                #else
                    nextDir = sampleGlossCore(view, hit.normal, hit.roughness, sampleUV);
                    if (dot(hit.normal, nextDir) <= 1e-4) break;
                    nextWeight = envWeight;
                #endif
            }
            throughput *= share * nextWeight;
            rayOrig = makeCornerSafeRayOrigin(hit.worldPos, hit.normal, view, hit.dist);
            rayDir = nextDir;
            roughness = max(roughness, hit.roughness);
            smoothSteps = SPECULAR_MAX_STEPS;
        #endif
    }
    return radiance;
}

#endif
