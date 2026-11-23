#version 430 compatibility

// Pass: Diffuse GI (1 SPP + Primary L1 SH vMF Path Guiding + 8-step SSRT + Hybrid 64-Way Trace)
#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/ScreenTrace.glsl"
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D colorimg3;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        imageStore(colorimg3, px, vec4(0.0));
        return;
    }

    GBufferMat mat = readGBuffer(px);
    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);
    uint rng = initRng(px, uint(frameCounter), 1u);

    // Query L1 SH Radiance Cache for primary-hit vMF path guiding (1/16 pixels allocate).
    vec3 guideDir = mat.shadeNormal;
    float guideKappa = 0.0;
    float guideProb = 0.0;
    bool allowAlloc = !mat.isHand
                   && (((px.x ^ frameCounter) & 3) == 0)
                   && (((px.y ^ (frameCounter >> 2)) & 3) == 0);
    bool useGuide = !mat.isHand && queryRCPathGuide(worldPos, mat.geoNormal, allowAlloc, nextVec2(rng), guideDir, guideKappa, guideProb);

    vec2 uDir = stbn2D(px, frameCounter, 0u);
    vec3 rayDir;
    if (useGuide && nextFloat(rng) < guideProb) {
        rayDir = sampleVMF(guideDir, guideKappa, uDir);
    } else {
        rayDir = sampleCosineHemisphere(mat.shadeNormal, uDir);
    }

    // Unbiased hemisphere rejection: samples landing below the geometric or shading
    // hemisphere contribute 0.0 (never resampled, preserving exact 1.0 Monte Carlo energy!).
    float cosN = dot(rayDir, mat.shadeNormal);
    if (dot(rayDir, mat.geoNormal) <= 1e-4 || cosN <= 1e-4) {
        imageStore(colorimg3, px, vec4(0.0, 0.0, 0.0, 64.0));
        return;
    }

    float pdfCos = cosN * INV_PI;
    float pdfMix = useGuide ? mix(pdfCos, pdfVMF(rayDir, guideDir, guideKappa), guideProb) : pdfCos;
    // Exact diffuse irradiance weight (cosN / PI) / pdfMix, bounded <= ~3.6.
    float weight = pdfCos / max(pdfMix, 1e-5);

    vec3 radiance = vec3(0.0);
    float hitDist = 64.0;
    bool ssrtHit = false;

    #ifdef DIFFUSE_SSRT
        if (!mat.isHand && frameCounter > 1) {
            vec2 prevHitUV;
            float ssDist;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * mat.geoNormal) * 0.04, rayDir, 8, 0.18, nextFloat(rng), 0.25, prevHitUV, ssDist)) {
                ivec2 hpx = clamp(ivec2(prevHitUV * screenSize()), ivec2(0), res - 1);
                vec3 prevLit = texelFetch(colortex5, hpx, 0).rgb;
                if (!any(isnan(prevLit)) && !any(isinf(prevLit))) {
                    radiance = min(max(prevLit, vec3(0.0)), vec3(16.0));
                    hitDist = ssDist;
                    ssrtHit = true;
                }
            }
        }
    #endif

    if (!ssrtHit) {
        float camDist = length(worldPos);
        vec3 toCam = -worldPos / max(camDist, 1e-4);
        // Pull ray origin slightly toward camera + normal so concave 90-degree corners
        // never poke into the perpendicular wall voxel and leak outdoor daylight.
        vec3 rayOrig = worldPos + mat.geoNormal * (0.018 + 0.001 * camDist) + toCam * (0.012 + 0.0005 * camDist);
        TraceHit hit = traceScene(rayOrig, rayDir, DIFFUSE_MAX_STEPS, 1.0);
        if (hit.hit) {
            hitDist = hit.dist;
            radiance += hit.albedo * (hit.emission * EMISSIVE_GI_SCALE * hit.emitFluxScale);

            vec3 lightDir = getShadowLightDir();
            float NdotL = max(dot(hit.normal, lightDir), 0.0);
            if (NdotL > 0.0) {
                float vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
                radiance += hit.albedo * INV_PI * getDirectLightColor() * (NdotL * vis);
            }

            vec3 bounce = queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, nextVec2(rng));
            radiance += hit.albedo * bounce * 0.85;
        } else if (hit.exitedGrid) {
            radiance = evalSky(rayDir, false);
        } else {
            // Ray ran out of steps while still inside an indoor volume: query cache instead of fake sky!
            radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false);
        }
    }

    radiance = min(max(radiance * min(weight, 3.0), vec3(0.0)), vec3(64.0));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);

    imageStore(colorimg3, px, vec4(radiance, hitDist));
}
