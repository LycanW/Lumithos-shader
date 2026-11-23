#version 430 compatibility

// Pass: Specular Reflection (1 SPP GGX VNDF + 16-step SSR + Hybrid 6-Face/Non-Cube Raytrace)
#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/ScreenTrace.glsl"
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D colorimg4;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    #ifndef SPECULAR_REFLECTIONS
        imageStore(colorimg4, px, vec4(0.0));
        return;
    #endif

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        imageStore(colorimg4, px, vec4(0.0));
        return;
    }

    GBufferMat mat = readGBuffer(px);
    if (mat.roughness > SPECULAR_ROUGH_CUTOFF || mat.isHand) {
        imageStore(colorimg4, px, vec4(0.0));
        return;
    }

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos = screenToView(vec3(uv, depth));
    // Pure mirrors trace along unjittered pixel centers so thin 1px cutout geometry (ladders, fences,
    // iron bars) never shimmers or blinks across alpha gaps when TAA jitter oscillates!
    if (mat.roughness < 0.08) {
        vec4 vpRaw = gbufferProjectionInverse * vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
        viewPos = vpRaw.xyz / vpRaw.w;
    }
    vec3 worldPos = viewToWorld(viewPos);
    vec3 viewDir  = normalize(-worldPos); // toward camera in world space
    uint rng = initRng(px, uint(frameCounter), 2u);

    vec3 rayDir;
    if (mat.roughness < 0.08) {
        rayDir = reflect(-viewDir, mat.shadeNormal);
    } else {
        rayDir = sampleGGXVNDF(viewDir, mat.shadeNormal, mat.roughness, stbn2D(px, frameCounter, 2u));
    }

    if (dot(rayDir, mat.geoNormal) <= 0.0) {
        rayDir = reflect(-viewDir, mat.geoNormal);
    }

    vec3 radiance = vec3(0.0);
    float hitDist = 128.0;
    bool ssrHit = false;

    #ifdef SPECULAR_SSR
        // Smooth/mirror surfaces (perceptual roughness < 0.25) bypass SSR and trace 100% exact 3D geometry
        // so mirrors never bleed background objects behind them.
        if (frameCounter > 1 && mat.roughness >= 0.25) {
            vec2 prevHitUV;
            float ssDist;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * mat.geoNormal) * 0.04, rayDir, 24, 0.65, nextFloat(rng), 0.25, prevHitUV, ssDist)) {
                ivec2 hpx = clamp(ivec2(prevHitUV * screenSize()), ivec2(0), res - 1);
                vec3 prevLit = texelFetch(colortex5, hpx, 0).rgb;
                if (!any(isnan(prevLit)) && !any(isinf(prevLit))) {
                    radiance = min(max(prevLit, vec3(0.0)), vec3(32.0));
                    hitDist = ssDist;
                    ssrHit = true;
                }
            }
        }
    #endif

    if (!ssrHit) {
        float camDist = length(worldPos);
        vec3 rayOrig = worldPos + mat.geoNormal * (0.018 + 0.001 * camDist) + viewDir * (0.012 + 0.0005 * camDist);
        // Full-res atlas LOD 0 for razor-sharp mirror reflections of 6-face & rotated block textures!
        float specLod = mat.roughness < 0.25 ? 0.0 : 1.0;
        int maxSpecSteps = mat.roughness < 0.22 ? SPECULAR_MAX_STEPS : 64;
        TraceHit hit = traceScene(rayOrig, rayDir, maxSpecSteps, specLod);
        if (hit.hit) {
            hitDist = hit.dist;
            radiance += hit.albedo * (hit.emission * EMISSIVE_SCALE);

            vec3 lightDir = getShadowLightDir();
            float NdotL = max(dot(hit.normal, lightDir), 0.0);
            if (NdotL > 0.0) {
                float vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
                radiance += hit.albedo * INV_PI * getDirectLightColor() * (NdotL * vis);
            }

            vec3 skyAmb = evalSky(hit.normal, false) * (0.15 + 0.85 * hit.skyLight);
            vec3 bounce = (mat.roughness < 0.25)
                ? queryRadianceCacheSmooth(hit.worldPos, hit.normal, skyAmb, 0u, nextVec2(rng))
                : queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, nextVec2(rng));
            radiance += hit.albedo * bounce * 0.85;
        } else if (hit.exitedGrid) {
            radiance = evalSky(rayDir, true);
        } else {
            radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false);
        }
    }

    radiance = max(radiance, vec3(0.0));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);

    imageStore(colorimg4, px, vec4(radiance, hitDist));
}
