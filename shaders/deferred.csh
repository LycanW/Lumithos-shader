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
    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);

    // Query L1 SH Radiance Cache for primary-hit vMF path guiding (1/16 screen pixels allocate).
    vec3 guideDir = shadeNormal;
    float guideKappa = 0.0;
    float guideProb = 0.0;
    #if DIFFUSE_MODE == 0
        bool allowAlloc = !isHand
                       && (((tid.x ^ frameCounter) & 1) == 0)
                       && (((tid.y ^ (frameCounter >> 1)) & 1) == 0);
    #elif DIFFUSE_MODE == 1
        bool allowAlloc = !isHand
                       && (((tid.x ^ frameCounter) & 3) == 0)
                       && (((tid.y ^ (frameCounter >> 2)) & 1) == 0);
    #else
        bool allowAlloc = !isHand
                       && (((px.x ^ frameCounter) & 3) == 0)
                       && (((px.y ^ (frameCounter >> 2)) & 3) == 0);
    #endif
    bool useGuide = !isHand && queryRCPathGuide(worldPos, geoNormal, allowAlloc, stbn2D(bnPx, bnSeq, 3u), guideDir, guideKappa, guideProb);

    vec2 uDir = stbn2D(bnPx, bnSeq, 0u);
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

    bool useNEE = (lightList.lightCount > 0u && lightList.lightCount <= MAX_SCENE_LIGHTS);

    vec3 radiance = vec3(0.0);
    float hitDist = 64.0;
    bool ssrtHit = false;

    #ifdef DIFFUSE_SSRT
        if (!isHand && frameCounter > 1) {
            vec2 prevHitUV;
            float ssDist;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * geoNormal) * 0.04, rayDir, 8, 0.18, stbn2D(bnPx, bnSeq, 9u).x, 0.25, prevHitUV, ssDist)) {
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

    if (!ssrtHit) {
        vec3 eyeToSurf = mat3(gbufferModelViewInverse) * viewPos;
        float camDist = length(eyeToSurf);
        vec3 toCam = -eyeToSurf / max(camDist, 1e-4);
        vec3 rayOrig = makeCornerSafeRayOrigin(worldPos, geoNormal, toCam, camDist);
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

            vec3 bounce = queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, stbn2D(bnPx, bnSeq, 5u));
            radiance += hit.albedo * bounce * 0.85;
            radiance *= hit.throughput;
        } else if (hit.exitedGrid) {
            radiance = evalSky(rayDir, false) * hit.throughput;
        } else {
            radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false) * hit.throughput;
        }
    }

    radiance = min(max(radiance * min(weight, 3.0), vec3(0.0)), vec3(28.0));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);

    imageStore(colorimg3, px, vec4(radiance, hitDist));
}
