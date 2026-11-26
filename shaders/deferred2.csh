#version 430 compatibility

// Pass: Specular Reflection (1 SPP GGX VNDF + 24-step frustum-bounded SSR + Hybrid 6-Face/Non-Cube Raytrace)
#define TRACE_HIGH_PRIM_GUARD
#define TRACE_REFLECTION_QUADS
#include "/lib/SpecularTrace.glsl"
#include "/lib/ScreenTrace.glsl"

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

    // Grazing normal-map horizon fix: prevent steep normal-mapped tile bevels from facing away
    // from a grazing camera viewDir (which would reflect underground into pitch-black tile grooves!).
    float rawNdotV = dot(shadeNormal, viewDir);
    if (rawNdotV < 0.05) {
        shadeNormal = normalize(shadeNormal + viewDir * (0.05 - rawNdotV));
    }

    vec3 H;
    int sampleIndex = frameCounter;
    #ifdef SPECULAR_CHECKERBOARD
        // GGX pixels trace every other frame; full-rate mirrors do not sample blue noise.
        sampleIndex >>= 1;
    #endif
    vec3 rayDir = sampleReflectionDirection(viewDir, shadeNormal, roughness, roughness >= 0.16 ? bnPx : px, sampleIndex, H);

    float NdotV = max(dot(shadeNormal, viewDir), 1e-4);
    float NdotL = dot(shadeNormal, rayDir);
    if (NdotL <= 1e-4) {
        return vec4(0.0, 0.0, 0.0, 0.0);
    }

    // Normal-mapped bevel/bump reflection dipping below the macro plane:
    // lift smoothly onto the grazing macro horizon instead of returning pitch-black 0.0!
    float GdotL = dot(rayDir, geoNormal);
    if (GdotL <= 0.02) {
        rayDir = normalize(rayDir + geoNormal * (0.025 - GdotL));
        H = normalize(viewDir + rayDir);
        NdotL = max(dot(shadeNormal, rayDir), 1e-4);
    }

    vec3 brdfRatio = vec3(1.0);
    if (!usesMirrorDirection(roughness)) {
        float alpha = max(roughness * roughness, 0.002);
        float VdotH = saturate(dot(viewDir, H));
        float g2OverG1 = smithGGX_G2OverG1(NdotV, NdotL, alpha);
        vec3 albedoLin = isLabPBRMetal(f0Raw) ? srgbToLinear(unpackUnorm4x8(texelFetch(colortex1, px, 0).r).rgb) : vec3(0.0);
        vec3 F0_rgb = evalLabPBRF0(f0Raw, albedoLin);
        vec3 F_vh = fresnelSchlick(F0_rgb, VdotH);
        vec3 envBrdf = evalEnvBRDF(F0_rgb, NdotV, roughness);
        brdfRatio = clamp((F_vh * g2OverG1) / max(envBrdf, vec3(1e-3)), vec3(0.0), vec3(4.0));
    }

    vec3 radiance = vec3(0.0);
    float hitDist;
    bool ssrHit = false;

    #ifdef SPECULAR_SSR
        // Smooth/mirror surfaces (perceptual roughness < 0.25) bypass SSR and trace 100% exact 3D geometry
        // so mirrors never bleed background objects behind them.
        if (frameCounter > 1 && roughness >= 0.25) {
            vec2 prevHitUV;
            float ssDist;
            if (screenSpaceTrace(viewPos + (mat3(gbufferModelView) * geoNormal) * 0.04, rayDir, 24, stbn2D(bnPx, frameCounter, 10u).x, 0.25, prevHitUV, ssDist)) {
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
        radiance = traceVoxelReflection(rayOrig, rayDir, roughness, bnPx, SPECULAR_MAX_STEPS, true, hitDist);
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
        // - slot 1: trace deterministic mirrors at full resolution. RELAX takes checkerboarded rough reflections.
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

            if (slot == 1 && !usesMirrorDirection(roughness)) {
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
