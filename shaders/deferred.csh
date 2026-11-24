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
    bool useGuide = !mat.isHand && queryRCPathGuide(worldPos, mat.geoNormal, allowAlloc, stbn2D(px, frameCounter, 3u), guideDir, guideKappa, guideProb);

    vec2 uDir = stbn2D(px, frameCounter, 0u);
    vec3 rayDir;
    if (useGuide && stbn2D(px, frameCounter, 4u).x < guideProb) {
        rayDir = sampleVMF(guideDir, guideKappa, uDir);
    } else {
        rayDir = sampleCosineHemisphere(mat.shadeNormal, uDir);
    }

    // Unbiased hemisphere rejection on indirect bounce only (NEE direct lights always evaluate!).
    float cosN = dot(rayDir, mat.shadeNormal);
    bool validIndirect = (dot(rayDir, mat.geoNormal) > 1e-4 && cosN > 1e-4);

    float pdfCos = max(cosN, 0.0) * INV_PI;
    float pdfMix = useGuide ? mix(pdfCos, pdfVMF(rayDir, guideDir, guideKappa), guideProb) : pdfCos;
    // Exact diffuse irradiance weight (cosN / PI) / pdfMix, bounded <= ~3.0.
    float weight = validIndirect ? (pdfCos / max(pdfMix, 1e-5)) : 0.0;

    float camDist = length(worldPos);
    vec3 toCam = -worldPos / max(camDist, 1e-4);
    vec3 rayOrig = makeCornerSafeRayOrigin(worldPos, mat.geoNormal, toCam, camDist);

    bool useNEE = (lightList.lightCount > 0u && lightList.lightCount <= MAX_SCENE_LIGHTS);

    // Direct 1-SPP RIS Next-Event Estimation from Emissive Voxel LightList (8 candidates -> fast L2 loop!).
    vec3 neeDir = vec3(0.0, 1.0, 0.0);
    vec3 neeRad = useNEE ? sampleDirectLightsNEE(rayOrig, mat.shadeNormal, false, 8, stbn2D(px, frameCounter, 8u), neeDir) : vec3(0.0);
    if (any(isnan(neeRad)) || any(isinf(neeRad))) neeRad = vec3(0.0);

    vec3 radiance = vec3(0.0);
    float hitDist = 64.0;
    bool ssrtHit = false;

    #ifdef DIFFUSE_SSRT
        if (validIndirect && !mat.isHand && frameCounter > 1) {
            vec2 prevHitUV;
            float ssDist;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * mat.geoNormal) * 0.04, rayDir, 8, 0.18, nextFloat(rng), 0.25, prevHitUV, ssDist)) {
                ivec2 hpx = clamp(ivec2(prevHitUV * screenSize()), ivec2(0), res - 1);
                vec4 prevTap = texelFetch(colortex5, hpx, 0);
                // Keep DIFFUSE_SSRT active on 99.9% of surfaces; only skip emissive pixels when NEE is active!
                if ((!useNEE || prevTap.a < 0.01) && !any(isnan(prevTap.rgb)) && !any(isinf(prevTap.rgb))) {
                    radiance = min(max(prevTap.rgb, vec3(0.0)), vec3(16.0));
                    hitDist = ssDist;
                    ssrtHit = true;
                }
            }
        }
    #endif

    if (validIndirect && !ssrtHit) {
        TraceHit hit = traceScene(rayOrig, rayDir, DIFFUSE_MAX_STEPS, 1.0, true);
        if (hit.hit) {
            hitDist = hit.dist;
            if (!useNEE || !hit.isBlockEmit) {
                radiance += hit.emitColor * (hit.emission * EMISSIVE_GI_SCALE * hit.emitFluxScale);
            }

            vec3 lightDir = getShadowLightDir();
            float NdotL = max(dot(hit.normal, lightDir), 0.0);
            if (NdotL > 0.0) {
                vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
                radiance += hit.albedo * INV_PI * getDirectLightColor() * (NdotL * vis);
            }

            vec3 bounce = queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, stbn2D(px, frameCounter, 5u));
            radiance += hit.albedo * bounce * 0.85;
            radiance *= hit.throughput;
        } else if (hit.exitedGrid) {
            radiance = evalSky(rayDir, false) * hit.throughput;
        } else {
            // Ray ran out of steps while still inside an indoor volume: query cache instead of fake sky!
            radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false) * hit.throughput;
        }
    }

    radiance = min(max(radiance * min(weight, 3.0) + neeRad, vec3(0.0)), vec3(28.0));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);

    imageStore(colorimg3, px, vec4(radiance, hitDist));
}
