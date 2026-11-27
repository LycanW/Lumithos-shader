#version 430 compatibility

// Pass: Combine Direct Sunlight + Emission + Diffuse GI + Specular GI + Sky + Debug Views,
// and publish prevGuideImg and local surface lighting for screen-space hits.
#include "/lib/SkyShadow.glsl"
#include "/lib/RadianceCache.glsl"
#include "/lib/Water.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D colorimg0;
layout(rgba16f) uniform writeonly image2D colorimg5;
layout(rgba32f) uniform writeonly image2D prevGuideImg;

vec3 evalHeldLightColor(int itemId) {
    if (itemId == 20001) return vec3(0.38, 0.94, 1.00); // Soul fire
    if (itemId == 20002) return vec3(0.74, 0.95, 1.00); // Sea lantern / beacon
    if (itemId == 20003) return vec3(1.00, 0.24, 0.20); // Redstone
    if (itemId == 20004) return vec3(1.00, 0.91, 0.78); // End rod / glowstone
    if (itemId == 20005) return vec3(0.78, 0.40, 1.00); // Amethyst / arcane
    if (itemId == 20006) return vec3(0.52, 1.00, 0.60); // Verdant froglight
    return vec3(1.00, 0.55, 0.11);                      // Torch / lantern / lava default (matches placed flame!)
}

// Perspective-exact clip-space depth contact shadow for held-item dynamic lights:
// - Interpolates homogeneous clip coordinates (sampleClip.z == exact linear view depth) to avoid
//   non-linear NDC depth self-intersection on front-facing close surfaces.
// - Uses pixel-world-size grazing bias, near-plane ray clamping, cubic contact-concentrated steps,
//   and smooth front-penumbra / back-thickness occlusion (ignoring the hand holding the light).
float traceHeldLightShadow(vec3 viewPos, vec3 viewGeoN, vec3 lightViewPos, vec3 viewLightDir, bool isOriginHand, float noise) {
    float viewZ = -viewPos.z;
    vec2 projScale = vec2(gbufferProjection[0][0], gbufferProjection[1][1]);
    vec2 invRes = texelSize();
    float pixelWorldSize = 2.0 * viewZ * max(invRes.x / max(abs(projScale.x), 1e-4), invRes.y / max(abs(projScale.y), 1e-4));
    float geoNdotL = saturate(dot(viewGeoN, viewLightDir));

    vec3 startView = viewPos
        + viewGeoN * min(pixelWorldSize * (0.9 + 1.0 / max(geoNdotL, 0.15)), 0.08)
        + viewLightDir * min(pixelWorldSize * 1.0, 0.04);

    vec3 endView = lightViewPos;
    if (-endView.z < 0.15) {
        float tClamp = saturate((startView.z + 0.15) / min(startView.z - lightViewPos.z, -1e-4));
        endView = mix(startView, lightViewPos, tClamp);
    }
    float traceDist = length(endView - startView);
    if (traceDist <= 1e-3) return 1.0;

    vec3 startClip = vec3(projScale * startView.xy + gbufferProjection[2].xy * startView.z, -startView.z);
    vec3 endClip   = vec3(projScale * endView.xy   + gbufferProjection[2].xy * endView.z,   -endView.z);
    vec3 rayClip   = endClip - startClip;

    const int STEPS = 16;
    const float invSamples = 1.0 / float(STEPS);
    vec2 res = screenSize();
    float maxOcc = 0.0;

    for (int i = 0; i < STEPS; ++i) {
        float u = (float(i) + noise) * invSamples;
        float t = u * (0.10 + u * (0.30 + 0.60 * u));

        vec3 sampleClip = startClip + rayClip * t;
        if (sampleClip.z <= 0.15) break;

        vec2 sampleUV = (sampleClip.xy / sampleClip.z) * 0.5 + 0.5;
        #ifdef TAA_ENABLED
            sampleUV += taaJitter * 0.5;
        #endif
        if (any(lessThan(sampleUV, vec2(0.001))) || any(greaterThanEqual(sampleUV, vec2(0.999)))) break;

        ivec2 q = ivec2(sampleUV * res);
        float sd = texelFetch(depthtex1, q, 0).r;
        if (sd >= 0.999999) continue;

        float sampleViewZ = linearizeDepth(sd);
        float depthDiff = sampleClip.z - sampleViewZ;
        if (depthDiff > 0.0) {
            uint w1 = texelFetch(colortex1, q, 0).g;
            bool isHandSample = ((w1 >> 24u) & 1u) != 0u;
            // Ignore the hand holding this light source while keeping the opposite hand as a caster.
            if (isHandSample && sampleClip.x * lightViewPos.x > 0.0) continue;

            float travelDist = traceDist * t;
            float stepZ = abs(rayClip.z) * (0.10 + u * (0.60 + 1.80 * u)) * invSamples;
            float penumbra;
            float thickness;
            if (isOriginHand) {
                penumbra  = max(pixelWorldSize * 2.0, 0.006 + travelDist * 0.05);
                thickness = clamp(0.10 + travelDist * 0.50 + stepZ * 1.5, 0.10, 0.55);
            } else if (isHandSample) {
                penumbra  = max(pixelWorldSize * 2.0, 0.015 + travelDist * 0.06);
                thickness = clamp(0.20 + travelDist * 0.50 + stepZ * 1.5, 0.20, 0.80);
            } else {
                penumbra  = max(pixelWorldSize * 2.5, 0.008 + travelDist * 0.07);
                thickness = clamp(0.15 + travelDist * 0.65 + stepZ * 2.2, 0.15, 2.40);
            }

            float frontOcc = smoothstep(0.0, penumbra, depthDiff);
            float backFade = 1.0 - smoothstep(thickness * 0.55, thickness, depthDiff);
            maxOcc = max(maxOcc, frontOcc * backFade);
            if (maxOcc > 0.99) break;
        }
    }

    return 1.0 - 0.82 * maxOcc;
}

void evalHeldHandLight(int lightVal, int itemId, vec3 handViewPos, GBufferMat mat, vec3 viewPos, vec3 viewDir,
                       vec3 F0_rgb, bool isMetal, float NdotV, vec3 viewGeoN, float dither,
                       inout vec3 outDiffuse, inout vec3 outSpec) {
    if (lightVal <= 0) return;

    bool isSameHand = mat.isHand && (viewPos.x * handViewPos.x > 0.0);
    vec3 effectiveLightPos;
    if (isSameHand) {
        effectiveLightPos = vec3(0.0);
    } else if (mat.isHand) {
        effectiveLightPos = handViewPos;
    } else {
        effectiveLightPos = handViewPos * smoothstep(0.05, 0.35, -viewPos.z);
    }

    vec3 toLightView = effectiveLightPos - viewPos;
    float dist = length(toLightView);
    float maxRange = 4.0 + float(lightVal) * 0.95;
    if (dist >= maxRange) return;

    vec3 viewLightDir = toLightView / max(dist, 1e-4);
    vec3 L = normalize(mat3(gbufferModelViewInverse) * viewLightDir);
    float NdotL = max(dot(mat.shadeNormal, L), 0.0);
    float geoHorizon = smoothstep(0.0, 0.08, dot(mat.geoNormal, L));
    float backNL = (mat.isFoliage && !mat.isHand) ? max(dot(-mat.geoNormal, L), 0.0) : 0.0;
    if ((NdotL <= 0.0 || geoHorizon <= 0.0) && backNL <= 0.0) return;

    float rNorm = dist / maxRange;
    float r2 = rNorm * rNorm;
    float window = saturate(1.0 - r2 * r2);
    float atten = (window * window) / (dist * dist + 0.35);
    float power = (float(lightVal) / 15.0) * 5.2;
    vec3 lightRad = evalHeldLightColor(itemId) * (power * atten);

    float vis = 0.0;
    if (NdotL > 0.0 && geoHorizon > 0.0) {
        float shadow = 1.0;
        if (!isSameHand) {
            shadow = traceHeldLightShadow(viewPos, viewGeoN, effectiveLightPos, viewLightDir, mat.isHand, dither);
            float shadowDistFade = 1.0 - smoothstep(maxRange * 0.65, maxRange, dist);
            shadow = mix(1.0, shadow, shadowDistFade);
        }
        vis = shadow * geoHorizon;
    }

    if (vis > 1e-4) {
        vec3 H = normalize(L + viewDir);
        float NdotH = max(dot(mat.shadeNormal, H), 0.0);
        float VdotH = saturate(dot(viewDir, H));
        vec3 F = fresnelSchlick(F0_rgb, VdotH);
        vec3 kD = isMetal ? vec3(0.0) : (1.0 - F);

        outDiffuse += mat.albedo * INV_PI * kD * lightRad * (NdotL * vis);
        outSpec    += F * directGGXLobe(NdotV, NdotL, NdotH, mat.roughness) * lightRad * (NdotL * vis);
    }

    if (backNL > 0.0) {
        outDiffuse += mat.albedo * mat.albedo * INV_PI * lightRad * (mat.sss * backNL * 0.45);
    }
}

// Perspective-exact clip-space Screen-Space Sun Shadow (Contact + Distant Auto-Extension):
// - Inside shadowDistance (shadowCoverage ~ 1): tight contact shadow plugs thin-geometry bias leaks.
// - Beyond shadowDistance (shadowCoverage -> 0): ray length automatically extends to 12..80 blocks so
//   distant mountains, builds, and forests cast continuous directional sun shadows with zero range ring!
float traceSunScreenSpaceShadow(vec3 viewPos, vec3 viewGeoN, vec3 viewLightDir, float shadowCoverage, bool isOriginHand, bool isFoliage, float noise) {
    if (isOriginHand) return 1.0;

    float viewZ = -viewPos.z;
    vec2 projScale = vec2(gbufferProjection[0][0], gbufferProjection[1][1]);
    vec2 invRes = texelSize();
    float pixelWorldSize = 2.0 * viewZ * max(invRes.x / max(abs(projScale.x), 1e-4), invRes.y / max(abs(projScale.y), 1e-4));
    float geoNdotL = saturate(dot(viewGeoN, viewLightDir));

    float nearLen = isFoliage ? clamp(0.22 + sqrt(viewZ) * 0.06, 0.22, 0.85) : clamp(0.16 + sqrt(viewZ) * 0.07, 0.16, 1.10);
    float farLen  = clamp(viewZ * 0.35, 12.0, 84.0);
    float rayLen  = mix(farLen, nearLen, shadowCoverage);

    vec3 startView = viewPos
        + viewGeoN * min(pixelWorldSize * (1.2 + 1.2 / max(geoNdotL, 0.15)), 0.14 + 0.0035 * viewZ)
        + viewLightDir * min(pixelWorldSize * 1.2, 0.08 + 0.002 * viewZ);

    vec3 endView = startView + viewLightDir * rayLen;
    if (-endView.z < 0.15) {
        float tClamp = saturate((startView.z + 0.15) / min(startView.z - endView.z, -1e-4));
        endView = mix(startView, endView, tClamp);
    }
    float traceDist = length(endView - startView);
    if (traceDist <= 1e-3) return 1.0;

    vec3 startClip = vec3(projScale * startView.xy + gbufferProjection[2].xy * startView.z, -startView.z);
    vec3 endClip   = vec3(projScale * endView.xy   + gbufferProjection[2].xy * endView.z,   -endView.z);
    vec3 rayClip   = endClip - startClip;

    const int STEPS = 12;
    const float invSamples = 1.0 / float(STEPS);
    vec2 res = screenSize();
    float maxOcc = 0.0;

    for (int i = 0; i < STEPS; ++i) {
        float u = (float(i) + noise) * invSamples;
        float t = u * (0.12 + u * (0.33 + 0.55 * u));

        vec3 sampleClip = startClip + rayClip * t;
        if (sampleClip.z <= 0.15) break;

        vec2 sampleUV = (sampleClip.xy / sampleClip.z) * 0.5 + 0.5;
        #ifdef TAA_ENABLED
            sampleUV += taaJitter * 0.5;
        #endif
        if (any(lessThan(sampleUV, vec2(0.001))) || any(greaterThanEqual(sampleUV, vec2(0.999)))) break;

        ivec2 q = ivec2(sampleUV * res);
        float sd = texelFetch(depthtex1, q, 0).r;
        if (sd >= 0.999999) continue;

        float sampleViewZ = linearizeDepth(sd);
        float depthDiff = sampleClip.z - sampleViewZ;
        if (depthDiff > 0.0) {
            uint w1 = texelFetch(colortex1, q, 0).g;
            if (((w1 >> 24u) & 1u) != 0u) continue; // never let first-person hand shadow the world

            float travelDist = traceDist * t;
            float stepZ = abs(rayClip.z) * (0.12 + u * (0.66 + 1.65 * u)) * invSamples;
            float penumbra  = max(pixelWorldSize * 2.2, 0.010 + travelDist * 0.055);
            float maxThick  = mix(max(3.5, viewZ * 0.12), 1.2, shadowCoverage);
            float thickness = clamp(0.14 + travelDist * 0.45 + stepZ * 2.0, 0.14, maxThick);

            float frontOcc = smoothstep(0.0, penumbra, depthDiff);
            float backFade = 1.0 - smoothstep(thickness * 0.55, thickness, depthDiff);
            maxOcc = max(maxOcc, frontOcc * backFade);
            if (maxOcc > 0.99) break;
        }
    }

    float occStrength = isFoliage ? 0.80 : 1.0;
    return 1.0 - occStrength * maxOcc;
}

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    float depth = texelFetch(depthtex1, px, 0).r;

    if (depth >= 0.999999) {
        vec3 skyDir = normalize(mat3(gbufferModelViewInverse) * screenToView(vec3(uv, 1.0)));
        vec3 sky = evalSky(skyDir, true);
        imageStore(colorimg0, px, vec4(sky, 1.0));
        imageStore(colorimg5, px, vec4(sky, 1.0));
        imageStore(prevGuideImg, px, vec4(0.0));
        return;
    }

    GBufferMat mat = readGBuffer(px);
    // gbuffers_hand does NOT apply TAA jitter, so never subtract taaJitter when unprojecting hand pixels!
    vec3 viewPos;
    if (mat.isHand) {
        vec4 vp = gbufferProjectionInverse * vec4(vec3(uv, depth) * 2.0 - 1.0, 1.0);
        viewPos = vp.xyz / vp.w;
    } else {
        viewPos = screenToView(vec3(uv, depth));
    }
    vec3 worldPos = viewToWorld(viewPos);
    vec3 viewDir  = normalize(mat3(gbufferModelViewInverse) * (-viewPos));

    // The low 16 bits store the face normal. Bit 16 carries the exact reflection
    // class for the next temporal pass and is ignored by the normal decoder.
    uint surfaceGuide = packNormal16(mat.geoNormal);
    #ifdef SPECULAR_REFLECTIONS
        if (hasSpecularTrace(mat.roughness, mat.f0, mat.isHand) && !usesMirrorDirection(mat.roughness))
            surfaceGuide |= 65536u;
    #endif
    imageStore(prevGuideImg, px, vec4(
        float(surfaceGuide),
        float(packNormal16(mat.shadeNormal)),
        mat.roughness,
        -viewPos.z
    ));

    // Direct Sunlight / Moonlight:
    // - Inside shadowDistance: Contact-Hardening Blue-Noise PCSS + tight screen-space contact shadow.
    // - Transition & beyond shadowDistance: smooth spherical handoff into distant screen-space sun
    //   shadows * skyLight occlusion (eliminates visible shadow range circle all the way to horizon!).
    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();
    float NdotL = max(dot(mat.shadeNormal, lightDir), 0.0);
    float geoHorizon = smoothstep(0.0, 0.08, dot(mat.geoNormal, lightDir));
    vec3 shadowVis = vec3(0.0);
    if (NdotL > 0.0 && geoHorizon > 0.0 && maxOf(directCol) > 1e-4) {
        vec2 shNoise = stbn2D(px, mat.isHand ? 0 : frameCounter, 7u);
        float shadowCoverage = 0.0;
        float waterPath = 0.0;
        vec3 smVis = sampleShadowPCSS(worldPos, mat.geoNormal, shNoise.x, shadowCoverage, waterPath);

        float ssSun = 1.0;
        if (!mat.isHand && (shadowCoverage < 0.99 || maxOf(smVis) > 0.01)) {
            vec3 viewGeoN = normalize(mat3(gbufferModelView) * mat.geoNormal);
            vec3 viewLightDir = normalize(mat3(gbufferModelView) * lightDir);
            ssSun = traceSunScreenSpaceShadow(viewPos, viewGeoN, viewLightDir, shadowCoverage, mat.isHand, mat.isFoliage, shNoise.y);
        }

        vec3 distantVis = vec3(smoothstep(0.42, 0.94, mat.skyLight) * ssSun);
        vec3 nearVis    = smVis * ssSun;
        shadowVis = mix(distantVis, nearVis, shadowCoverage) * geoHorizon;

        // Projected chromatic underwater caustics on sunlit submerged geometry (both above-water & underwater views).
        if (!mat.isHand && waterPath > 0.02 && maxOf(shadowVis) > 1e-4) {
            vec3 absPos = worldPos + cameraPositionFract + vec3(cameraPositionInt % 4096);
            vec3 caustics = evalWaterCaustics(absPos, lightDir, waterPath);
            shadowVis *= mix(vec3(1.0), caustics, shadowCoverage);
        }
    }
    vec3 directDiffuse = mat.albedo * INV_PI * directCol * (NdotL * shadowVis);

    // Physical LabPBR 1.3 subsurface scattering via Beer-Lambert shadow-map optical thickness:
    // - Thin foliage/plants (leaves, grass, vines) transmit lush dappled backlight through canopies.
    // - Solid mineral/wax/ice SSS (quartz, marble, snow, candles) stays strictly airtight through
    //   1-block-thick walls (zero corridor light leaks & zero grazing stripes!) while glowing softly at thin edges.
    vec3 sssLight = vec3(0.0);
    if (mat.sss > 0.0 && !mat.isHand && maxOf(directCol) > 1e-4) {
        float backNL = dot(-mat.geoNormal, lightDir);
        if (backNL > 0.0) {
            float sssNoise = stbn2D(px, frameCounter, 8u).x;
            vec3 transVis = evalShadowSSS(worldPos, mat.geoNormal, lightDir, mat.isFoliage, sssNoise);
            float mu = dot(-viewDir, lightDir);
            const float g = 0.55;
            float hg = (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * mu, 1e-4), 1.5));
            float phase = 0.35 * INV_PI + 0.65 * hg;
            float backWeight = mat.isFoliage ? saturate(backNL + 0.25) : saturate(backNL);
            sssLight = mat.albedo * mat.albedo * directCol * transVis * (mat.sss * phase * backWeight);
        }
    }

    // Direct Specular Highlight (Cook-Torrance GGX with Height-Correlated Smith Visibility & LabPBR RGB F0)
    // Match deferred2.csh's grazing normal-map horizon fix so steep bevels keep consistent NdotV/Fresnel!
    float rawNdotV = dot(mat.shadeNormal, viewDir);
    if (rawNdotV < 0.05) {
        mat.shadeNormal = normalize(mat.shadeNormal + viewDir * (0.05 - rawNdotV));
    }
    vec3 F0_rgb = evalLabPBRF0(mat.f0, mat.albedo);
    bool isMetal = isLabPBRMetal(mat.f0);
    float NdotV = max(dot(mat.shadeNormal, viewDir), 1e-4);
    vec3 directSpec = vec3(0.0);
    vec3 kD_direct = isMetal ? vec3(0.0) : (1.0 - evalEnvBRDF(F0_rgb, NdotV, mat.roughness));
    if (NdotL > 0.0 && maxOf(shadowVis) > 0.0) {
        vec3 H = normalize(lightDir + viewDir);
        float NdotH = max(dot(mat.shadeNormal, H), 0.0);
        float VdotH = saturate(dot(viewDir, H));
        vec3 F = fresnelSchlick(F0_rgb, VdotH);
        directSpec = F * directGGXLobe(NdotV, NdotL, NdotH, mat.roughness) * directCol * (NdotL * shadowVis);
        if (!isMetal) kD_direct = 1.0 - F;
    }

    // First-person hand bypasses temporal accumulation (RELAX and TAA) for zero swing ghosting,
    // so shade hand indirect GI directly from smooth 2x2-bilinear L1 SH Radiance Cache (0% noise/jitter!).
    vec3 diffuseGI;
    if (mat.isHand) {
        vec3 skyAmb = evalSky(mat.shadeNormal, false) * (0.15 + 0.85 * mat.skyLight);
        diffuseGI = queryRadianceCacheSmooth(worldPos, mat.shadeNormal, skyAmb, 0u, vec2(0.5));
    } else {
        diffuseGI = max(texelFetch(colortex3, px, 0).rgb, vec3(0.0));
    }
    vec3 specularGI = max(texelFetch(colortex4, px, 0).rgb, vec3(0.0));
    vec3 emissive   = (mat.emission > 0.0) ? (mat.albedo * (mat.emission * EMISSIVE_SCALE)) : vec3(0.0);

    // Apply the preintegrated BRDF strength to the filtered reflection lighting.
    // Metals use the traced signal at all roughnesses. Rough dielectrics blend toward diffuse GI.
    vec3 envBrdf = evalEnvBRDF(F0_rgb, NdotV, mat.roughness);
    vec3 kD_env  = isMetal ? vec3(0.0) : (1.0 - envBrdf);
    float specTraceFade = isMetal ? 1.0 : smoothstep(SPECULAR_ROUGH_CUTOFF, 0.50, mat.roughness);
    vec3 specEnv = mat.isHand ? diffuseGI : mix(diffuseGI, specularGI, specTraceFade);

    // LabPBR material AO (normals.b) on indirect diffuse + Lagarde specular occlusion on indirect specular.
    float specOcc = saturate(pow(NdotV + mat.ao, exp2(-16.0 * mat.roughness - 1.0)) - 1.0 + mat.ao);

    // Handheld dynamic lighting (main hand + off-hand) with perspective-exact screen-space contact shadows.
    // Uniform early-out costs 0.00ms whenever neither hand holds an emissive item!
    vec3 heldDiffuse = vec3(0.0);
    vec3 heldSpec    = vec3(0.0);
    if (heldBlockLightValue > 0 || heldBlockLightValue2 > 0) {
        vec3 viewGeoN = normalize(mat3(gbufferModelView) * mat.geoNormal);
        float heldDither = stbn2D(px, mat.isHand ? 0 : frameCounter, 14u).x;
        evalHeldHandLight(heldBlockLightValue,  heldItemId,  vec3( 0.45, 0.0, 0.0), mat, viewPos, viewDir,
                          F0_rgb, isMetal, NdotV, viewGeoN, heldDither, heldDiffuse, heldSpec);
        evalHeldHandLight(heldBlockLightValue2, heldItemId2, vec3(-0.45, 0.0, 0.0), mat, viewPos, viewDir,
                          F0_rgb, isMetal, NdotV, viewGeoN, fract(heldDither + 0.5), heldDiffuse, heldSpec);
    }

    vec3 diffuseLit = directDiffuse * kD_direct + sssLight * kD_direct + heldDiffuse + (diffuseGI * mat.albedo) * (kD_env * mat.ao) + emissive;
    vec3 finalLit   = diffuseLit + directSpec + heldSpec + specEnv * envBrdf * specOcc;

    // A screen hit sees the metal from another direction. Store its local, angularly averaged response.
    vec3 localHit = diffuseLit;
    if (isMetal) {
        // 2/3 is the mean cosine over a cosine-weighted hemisphere.
        localHit += evalEnvBRDF(F0_rgb, 2.0 / 3.0, mat.roughness) * diffuseGI * mat.ao;
        localHit += evalEnvBRDF(F0_rgb, NdotL, mat.roughness) * (INV_PI * NdotL) * directCol * shadowVis;
    }
    imageStore(colorimg5, px, vec4(localHit, mat.emission));

    // Physical Aerial Perspective (distant Rayleigh+Mie haze & seamless horizon dissolve).
    if (!mat.isHand && isEyeInWater == 0) {
        vec3 apTrans, apInscatter;
        evalAerialPerspective(worldPos, -viewDir, mat.skyLight, apTrans, apInscatter);
        finalLit = finalLit * apTrans + apInscatter;
    }

    // Optional Debug Visualizations
    #if DEBUG_VIEW == 1 || DEBUG_VIEW == 2
        finalLit = diffuseGI;
    #elif DEBUG_VIEW == 3
        finalLit = specularGI;
    #elif DEBUG_VIEW == 4
        finalLit = queryRadianceCache(worldPos, mat.geoNormal, 0u, false);
    #elif DEBUG_VIEW == 5
        ivec3 vCoord = ivec3(floor(worldToVoxelPos(worldPos - mat.geoNormal * 0.05)));
        vec3 dbg = vec3(0.05, 0.05, 0.15);
        if (inVoxelBounds(vCoord)) {
            if (texelFetch(voxelGridTex, vCoord, 0).r != 0u) {
                SurfaceVoxel sv = readSurface(voxelLinearIndex(vCoord));
                bool hasT1 = (sv.face[0] | sv.face[1] | sv.face[2] | sv.face[3] | sv.face[4] | sv.face[5]) != 0u;
                bool hasT2 = (sv.primHead != PRIM_END);
                dbg = vec3(hasT2 ? 1.0 : 0.0, hasT1 ? 1.0 : 0.0, 0.0);
            }
        }
        finalLit = dbg;
    #elif DEBUG_VIEW == 6
        finalLit = saturate(vec3(
            luma(directDiffuse * kD_direct + directSpec) * 2.5,
            luma(specEnv * envBrdf * specOcc) * 4.0,
            luma((diffuseGI * mat.albedo) * (kD_env * mat.ao) + sssLight * kD_direct) * 2.5
        ));
    #endif

    imageStore(colorimg0, px, vec4(finalLit, 1.0));
}
