#version 430 compatibility

// Pass: Specular Reflection (1 SPP GGX VNDF + 24-step SSR + Hybrid 6-Face/Non-Cube Raytrace)
#define TRACE_HIGH_PRIM_GUARD
#include "/lib/Raytrace.glsl"
#include "/lib/SkyShadow.glsl"
#include "/lib/ScreenTrace.glsl"
#include "/lib/RadianceCache.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
#ifdef SPECULAR_CHECKERBOARD
const vec2 workGroupsRender = vec2(0.5, 1.0);
#else
const vec2 workGroupsRender = vec2(1.0, 1.0);
#endif

layout(rgba16f) uniform writeonly image2D colorimg4;

vec4 traceSpecularPixel(ivec2 px, ivec2 bnPx, ivec2 res, float depth, float roughness, float f0Raw) {
    vec3 geoNormal, shadeNormal;
    readGBufferNormals(px, geoNormal, shadeNormal);

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    // Always unjitter with screenToView() to match depthtex1's TAA-jittered rasterization,
    // and compute viewDir from the actual view-bobbed camera eye (gbufferModelViewInverse[3].xyz),
    // never un-bobbed vec3(0.0), so mirror reflections stay 100% optically stationary while walking!
    vec3 viewPos   = screenToView(vec3(uv, depth));
    vec3 worldPos  = viewToWorld(viewPos);
    vec3 eyeToSurf = mat3(gbufferModelViewInverse) * viewPos;
    float camDist  = length(eyeToSurf);
    vec3 viewDir   = -eyeToSurf / max(camDist, 1e-4); // toward actual bobbed camera eye in world space

    vec3 H = shadeNormal;
    vec3 rayDir;
    if (roughness < 0.08) {
        rayDir = reflect(-viewDir, shadeNormal);
    } else {
        rayDir = sampleGGXVNDF(viewDir, shadeNormal, roughness, stbn2D(bnPx, frameCounter, 2u), H);
    }

    // Physical Heitz VNDF microfacet horizon test: rays reflected below the surface are self-occluded
    // (weight = 0), NEVER replaced with a fake mirror reflection!
    float NdotV = max(dot(shadeNormal, viewDir), 1e-4);
    float NdotL = dot(shadeNormal, rayDir);
    if (dot(rayDir, geoNormal) <= 1e-4 || NdotL <= 1e-4) {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }

    vec3 brdfRatio = vec3(1.0);
    if (roughness >= 0.08) {
        float alpha = max(roughness * roughness, 0.002);
        float VdotH = saturate(dot(viewDir, H));
        float g2OverG1 = smithGGX_G2OverG1(NdotV, NdotL, alpha);
        vec3 albedoLin = (f0Raw >= (237.5 / 255.0)) ? srgbToLinear(unpackUnorm4x8(texelFetch(colortex1, px, 0).r).rgb) : vec3(0.0);
        vec3 F0_rgb = evalLabPBRF0(f0Raw, albedoLin);
        vec3 F_vh = fresnelSchlick(F0_rgb, VdotH);
        vec3 envBrdf = evalEnvBRDF(F0_rgb, NdotV, roughness);
        brdfRatio = clamp((F_vh * g2OverG1) / max(envBrdf, vec3(1e-3)), vec3(0.0), vec3(4.0));
    }

    vec3 radiance = vec3(0.0);
    float hitDist = 128.0;
    bool ssrHit = false;

    #ifdef SPECULAR_SSR
        // Smooth/mirror surfaces (perceptual roughness < 0.25) bypass SSR and trace 100% exact 3D geometry
        // so mirrors never bleed background objects behind them.
        if (frameCounter > 1 && roughness >= 0.25) {
            vec2 prevHitUV;
            float ssDist;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * geoNormal) * 0.04, rayDir, 24, 0.65, stbn2D(bnPx, frameCounter, 10u).x, 0.25, prevHitUV, ssDist)) {
                ivec2 hpx = clamp(ivec2(prevHitUV * screenSize()), ivec2(0), res - 1);
                vec3 prevLit = texelFetch(colortex5, hpx, 0).rgb;
                if (!any(isnan(prevLit)) && !any(isinf(prevLit))) {
                    radiance = min(max(prevLit, vec3(0.0)), vec3(12.0));
                    hitDist = ssDist;
                    ssrHit = true;
                }
            }
        }
    #endif

    if (!ssrHit) {
        vec3 rayOrig = makeCornerSafeRayOrigin(worldPos, geoNormal, viewDir, camDist);
        // Full-res atlas LOD 0 for razor-sharp mirror reflections of 6-face & rotated block textures!
        float specLod = roughness < 0.25 ? 0.0 : 1.0;
        int maxSpecSteps = roughness < 0.22 ? SPECULAR_MAX_STEPS : 48;
        TraceHit hit = traceScene(rayOrig, rayDir, maxSpecSteps, specLod, true);
        if (hit.hit) {
            hitDist = hit.dist;
            radiance += hit.emitColor * (hit.emission * EMISSIVE_SCALE);

            vec3 lightDir = getShadowLightDir();
            float hitNdotL = max(dot(hit.normal, lightDir), 0.0);
            if (hitNdotL > 0.0) {
                vec3 vis = sampleShadowFastRay(hit.worldPos, hit.normal, rayDir);
                radiance += hit.albedo * INV_PI * getDirectLightColor() * (hitNdotL * vis);
            }

            vec3 skyAmb = evalSky(hit.normal, false) * (0.15 + 0.85 * hit.skyLight);
            vec3 bounce = (roughness < 0.25)
                ? queryRadianceCacheSmooth(hit.worldPos, hit.normal, skyAmb, 0u, stbn2D(bnPx, frameCounter, 6u))
                : queryRadianceCacheStochastic(hit.worldPos, hit.normal, 0u, stbn2D(bnPx, frameCounter, 6u));
            radiance += hit.albedo * bounce * 0.85;
            radiance *= hit.throughput;
        } else if (hit.exitedGrid) {
            radiance = evalSky(rayDir, true) * hit.throughput;
        } else {
            radiance = queryRadianceCache(hit.worldPos, -rayDir, 0u, false) * hit.throughput;
        }
    }

    radiance *= brdfRatio;
    float specClamp = (roughness < 0.12) ? 32.0 : 12.0;
    radiance = min(max(radiance, vec3(0.0)), vec3(specClamp));
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);

    return vec4(radiance, hitDist);
}

void main() {
    ivec2 tid = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());

    #ifdef SPECULAR_CHECKERBOARD
        if (tid.y >= res.y) return;
        int parity = (tid.y ^ frameCounter) & 1;
        ivec2 pxA = ivec2(tid.x * 2 + parity, tid.y);
        ivec2 pxB = ivec2(tid.x * 2 + (parity ^ 1), tid.y);

        #ifndef SPECULAR_REFLECTIONS
            if (pxA.x < res.x) imageStore(colorimg4, pxA, vec4(0.0, 0.0, 0.0, -1.0));
            if (pxB.x < res.x) imageStore(colorimg4, pxB, vec4(0.0, 0.0, 0.0, -1.0));
            return;
        #endif

        // Single call-site loop prevents compiler double-inlining of traceScene (keeps GRF registers & I-cache tight!):
        // - slot 0 (active checkerboard pixel): always traces within specular cutoff (zero SIMD warp divergence!).
        // - slot 1 (companion pixel): traces at 100% FULL RESOLUTION ONLY for smooth/flat mirrors (< 0.16);
        //   rough glossy pixels (>= 0.16) exit uniformly with -1.0 sentinel for deferred9.csh!
        for (int slot = 0; slot < 2; ++slot) {
            ivec2 px = (slot == 0) ? pxA : pxB;
            if (px.x >= res.x) continue;

            float depth = texelFetch(depthtex1, px, 0).r;
            if (depth >= 0.999999) {
                imageStore(colorimg4, px, vec4(0.0, 0.0, 0.0, -1.0));
                continue;
            }

            float roughness, f0Raw;
            bool isHand;
            readGBufferSpecMask(px, roughness, f0Raw, isHand);
            if (!hasSpecularTrace(roughness, f0Raw, isHand)) {
                imageStore(colorimg4, px, vec4(0.0, 0.0, 0.0, -1.0));
                continue;
            }

            if (slot == 1 && roughness >= 0.16) {
                imageStore(colorimg4, px, vec4(0.0, 0.0, 0.0, -1.0));
                continue;
            }

            ivec2 bnPx = (slot == 0) ? tid : px;
            imageStore(colorimg4, px, traceSpecularPixel(px, bnPx, res, depth, roughness, f0Raw));
        }
    #else
        ivec2 px = tid;
        if (any(greaterThanEqual(px, res))) return;

        #ifndef SPECULAR_REFLECTIONS
            imageStore(colorimg4, px, vec4(0.0, 0.0, 0.0, -1.0));
            return;
        #endif

        float depth = texelFetch(depthtex1, px, 0).r;
        if (depth >= 0.999999) {
            imageStore(colorimg4, px, vec4(0.0, 0.0, 0.0, -1.0));
            return;
        }

        float roughness, f0Raw;
        bool isHand;
        readGBufferSpecMask(px, roughness, f0Raw, isHand);
        if (!hasSpecularTrace(roughness, f0Raw, isHand)) {
            imageStore(colorimg4, px, vec4(0.0, 0.0, 0.0, -1.0));
            return;
        }

        imageStore(colorimg4, px, traceSpecularPixel(px, px, res, depth, roughness, f0Raw));
    #endif
}
