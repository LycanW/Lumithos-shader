#version 430 compatibility

// Pass: Pure Indirect Diffuse GI (1 SPP + L1 SH vMF Path Guiding + Hybrid 64-Way Trace)
// Direct block-light NEE runs in dedicated lightweight deferred1.csh to eliminate GRF register spilling!
#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#ifdef DIFFUSE_SSRT
    #include "/lib/ScreenTrace.glsl"
#endif
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
#if DEBUG_VIEW == 1 || DIFFUSE_TRACE_RES == 2
    #define DIFFUSE_MODE 2
    const vec2 workGroupsRender = vec2(1.0, 1.0);
#elif DIFFUSE_TRACE_RES == 1
    #define DIFFUSE_MODE 1
    const vec2 workGroupsRender = vec2(0.5, 1.0);
#else
    #define DIFFUSE_MODE 0
    const vec2 workGroupsRender = vec2(0.5, 0.5);
#endif

layout(rgba16f) uniform writeonly image2D colorimg3;

void main() {
    ivec2 tid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    #if DIFFUSE_MODE == 0
        int f4 = frameCounter & 3;
        ivec2 qOff = ivec2(((f4 >> 1) ^ (f4 & 1)) & 1, f4 & 1);
        ivec2 px = tid * 2 + qOff;
        ivec2 bnPx = tid;
        int bnSeq = (frameCounter >> 2) + f4 * 197;
    #elif DIFFUSE_MODE == 1
        ivec2 px = ivec2(tid.x * 2 + ((tid.y ^ frameCounter) & 1), tid.y);
        ivec2 bnPx = tid;
        int bnSeq = (frameCounter >> 1) + (frameCounter & 1) * 197;
    #else
        ivec2 px = tid;
        ivec2 bnPx = px;
        int bnSeq = frameCounter;
    #endif
    if (any(greaterThanEqual(px, res))) return;

    float depth = texelFetch(depthtex1, px, 0).r;
    if (depth >= 0.999999) {
        imageStore(colorimg3, px, vec4(0.0));
        return;
    }

    float rough;
    bool isHand;
    readGBufferRoughHand(px, rough, isHand);
    if (isHand) {
        imageStore(colorimg3, px, vec4(0.0));
        return;
    }
    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);

    // Register each visible pixel's exact cache cell once per 16 frames.
    // Lower frame bits have already selected the traced pixel in each tile.
    #if DIFFUSE_MODE == 0
        bool registerSurface = ((tid.x ^ (frameCounter >> 2)) & 1) == 0 && ((tid.y ^ (frameCounter >> 3)) & 1) == 0;
    #elif DIFFUSE_MODE == 1
        bool registerSurface = ((tid.x ^ (frameCounter >> 1)) & 3) == 0 && ((tid.y ^ (frameCounter >> 3)) & 1) == 0;
    #else
        bool registerSurface = ((px.x ^ frameCounter) & 3) == 0 && ((px.y ^ (frameCounter >> 2)) & 3) == 0;
    #endif

    vec3 guideDir = shadeNormal;
    float guideKappa = 0.0;
    float guideProb = 0.0;
    bool useGuide = queryRCPathGuide(worldPos, geoNormal, registerSurface, stbn2D(bnPx, bnSeq, 3u), guideDir, guideKappa, guideProb);

    vec2 uDir = sampleBlueNoiseRG(bnPx + ivec2(64, 0), bnSeq);
    vec3 rayDir;
    if (useGuide && stbn2D(bnPx, bnSeq, 4u).x < guideProb) {
        rayDir = sampleVMF(guideDir, guideKappa, uDir);
    } else {
        rayDir = sampleCosineHemisphere(shadeNormal, uDir);
    }

    float cosN = dot(rayDir, shadeNormal);
    if (dot(rayDir, geoNormal) <= 1e-4 || cosN <= 1e-4) {
        imageStore(colorimg3, px, vec4(0.0, 0.0, 0.0, 64.0));
        return;
    }

    float pdfCos = cosN * INV_PI;
    float pdfMix = useGuide ? mix(pdfCos, pdfVMF(rayDir, guideDir, guideKappa), guideProb) : pdfCos;
    float weight = pdfCos / max(pdfMix, 1e-5);

    vec3 radiance = vec3(0.0);
    float hitDist = 64.0;
    bool ssrtHit = false;

    #ifdef DIFFUSE_SSRT
        if (!isHand && frameCounter > 1) {
            vec2 prevHitUV;
            float ssDist;
            ivec3 hitVoxel;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * geoNormal) * 0.04, rayDir, 8, stbn2D(bnPx, bnSeq, 9u).x, 0.25, prevHitUV, ssDist, hitVoxel)) {
                ivec2 hpx = clamp(ivec2(prevHitUV * screenSize()), ivec2(0), res - 1);
                vec4 prevTap = texelFetch(colortex5, hpx, 0);
                if ((prevTap.a <= 0.0 || !directLightCovers(hitVoxel, worldPos, shadeNormal))
                    && !any(isnan(prevTap.rgb)) && !any(isinf(prevTap.rgb))) {
                    radiance = min(max(prevTap.rgb, vec3(0.0)), vec3(16.0));
                    hitDist = ssDist;
                    ssrtHit = true;
                }
            }
        }
    #endif

    if (!ssrtHit) {
        vec3 eyeToSurf = mat3(gbufferModelViewInverse) * viewPos;
        float camDist = length(eyeToSurf);
        vec3 toCam = -eyeToSurf / max(camDist, 1e-4);
        vec3 rayOrig = makeCornerSafeRayOrigin(worldPos, geoNormal, toCam, camDist);
        TraceHit hit = traceScene(rayOrig, rayDir, DIFFUSE_MAX_STEPS, 1.0, true);
        if (hit.hit) {
            hitDist = hit.dist;
            radiance += indirectHitEmission(hit, rayOrig, shadeNormal);

            vec3 lightDir = getShadowLightDir();
            float NdotL = max(dot(hit.normal, lightDir), 0.0);
            vec3 direct = vec3(0.0);
            if (NdotL > 0.0) {
                vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
                direct = getDirectLightColor() * (NdotL * vis);
            }

            vec3 bounce = queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, stbn2D(bnPx, bnSeq, 5u));
            radiance += secondaryHitLighting(hit, rayDir, lightDir, direct, bounce, false);
            radiance *= hit.throughput;
        } else if (hit.exitedGrid) {
            radiance = evalSky(rayDir, false) * hit.throughput;
        } else {
            radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false) * hit.throughput;
        }
    }

    radiance = min(max(radiance * min(weight, 2.0), vec3(0.0)), vec3(16.0));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);

    imageStore(colorimg3, px, vec4(radiance, hitDist));
}
