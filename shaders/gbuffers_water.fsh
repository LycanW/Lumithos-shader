#version 430 compatibility

// Physically-Based Translucent Forward Pass (Water / Glass / Ice / Stained Glass / Other Translucents)
// Writes separable layers resolved in composite TAA so opaque speculars behind glass are preserved:
//   colortex6.rgb = reflected + in-scattered + emitted radiance, .a = layer opacity (1 - luma(T))
//   colortex7.rgb = per-channel transmittance (multiplicative blend, clear = 1)
//   colortex8.rg  = refraction screen-UV offset (nearest layer)
// - Water: multi-octave anisotropic swells + wave relief parallax, crisp liquid Fresnel & Snell's window,
//   crystal-clear turquoise Beer-Lambert volume attenuation + sunlit wave-crest subsurface scattering,
//   sparkling GGX sun glint, and screen-space refraction.
// - Glass/Ice: smooth dielectric Fresnel, colored transmittance, diffuse frame/frost lighting.
// - Reflections: current-frame SSR when enabled, with voxel PT for off-screen geometry.
#define TRACE_HIGH_PRIM_GUARD
#include "/lib/SpecularTrace.glsl"
#include "/lib/MaterialMaps.glsl"
#include "/lib/Water.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;
uniform sampler2D prevGuideTex;
layout(rgba16f) uniform readonly image2D colorimg0;

in vec2  vTexCoord;
in vec4  vColor;
in vec3  vGeoNormal;
in vec4  vTangent;
in vec3  vWorldPos;
in float vSkyLight;
in float vBlockEmit;
flat in int vMatId;

/* RENDERTARGETS: 6,7,8 */
layout(location = 0) out vec4 radianceOut;
layout(location = 1) out vec4 transmitOut;
layout(location = 2) out vec4 refractOut;

// Current-frame SSR reads opaque depth and the lit scene after aerial perspective.
// - Reconstructs the view-space surface normal from neighboring depth samples.
// - Enforces front-to-back crossing + 6-step binary refinement + 3D plane distance + collinearity check;
//   any miss falls back immediately to exact 3D voxel path-traced reflection!
float traceTranslucentSSR(vec3 viewPos, vec3 viewR, float dither, out vec3 hitCol) {
    hitCol = vec3(0.0);
    float maxT = 2.0 * far;
    if (viewR.z > 1e-5) maxT = min(maxT, (-near * 1.01 - viewPos.z) / viewR.z);
    if (viewR.z < -1e-5) maxT = min(maxT, (-far - viewPos.z) / viewR.z);
    if (maxT <= 0.1) return 0.0;

    vec3 s0 = viewToScreen(viewPos);
    if (any(lessThan(s0.xy, vec2(0.0))) || any(greaterThanEqual(s0.xy, vec2(1.0)))) return 0.0;
    vec3 s1 = viewToScreen(viewPos + viewR * maxT);
    vec3 delta = s1 - s0;
    vec2 toEdge = mix(s0.xy, 1.0 - s0.xy, greaterThan(delta.xy, vec2(0.0)));
    vec2 edgeT = toEdge / max(abs(delta.xy), vec2(1e-8));
    delta *= min(1.0, min(edgeT.x, edgeT.y));

    const int STEPS = 28;
    vec3 stepV = delta / float(STEPS);
    vec2 res = screenSize();
    ivec2 maxPx = ivec2(res) - 1;
    vec3 prevP = s0;
    float prevRayZ = -viewPos.z;

    for (int i = 0; i < STEPS; ++i) {
        float stepIndex = (i == STEPS - 1) ? float(STEPS) : float(i) + 0.5 + 0.5 * dither;
        vec3 p = s0 + stepV * stepIndex;
        if (any(lessThan(p.xy, vec2(0.0))) || any(greaterThanEqual(p.xy, vec2(1.0))) || p.z <= 0.0 || p.z >= 1.0) return 0.0;
        ivec2 px = clamp(ivec2(p.xy * res), ivec2(0), maxPx);
        float sd = texelFetch(depthtex1, px, 0).r;
        float rayZ = linearizeDepth(p.z);
        if (sd >= 0.999999 || p.z <= sd) { prevP = p; prevRayZ = rayZ; continue; }

        // Skip ultra-near first-person hand depth slab without needing G-buffer integer samplers.
        if (sd < 0.56) { prevP = p; prevRayZ = rayZ; continue; }

        float sceneZ = linearizeDepth(sd);
        float stepDZ = rayZ - prevRayZ;
        float diffZ0 = rayZ - sceneZ;
        // Step-bracket crossing test only (zero artificial surface thickness!).
        float stepBracket = max(0.10, abs(stepDZ) * 1.15);

        if (prevRayZ <= sceneZ + 0.03 && diffZ0 > 0.0 && diffZ0 < stepBracket) {
            vec3 a = prevP;
            vec3 b = p;
            for (int k = 0; k < 6; ++k) {
                vec3 m = 0.5 * (a + b);
                ivec2 mpx = clamp(ivec2(m.xy * res), ivec2(0), maxPx);
                float md = texelFetch(depthtex1, mpx, 0).r;
                if (md < 0.999999 && m.z > md) b = m; else a = m;
            }
            ivec2 bpx = clamp(ivec2(b.xy * res), ivec2(0), maxPx);
            float bd = texelFetch(depthtex1, bpx, 0).r;
            if (bd >= 0.56 && bd < 0.999999) {
                vec3 bRayView = screenToView(b);
                vec3 bHitView = screenToView(vec3(b.xy, bd));

                // Reconstruct local 3D surface normal from neighboring depthtex1 taps (zero G-buffer dependency).
                ivec2 rxPx = clamp(bpx + ivec2(bpx.x < maxPx.x ? 1 : -1, 0), ivec2(0), maxPx);
                ivec2 ryPx = clamp(bpx + ivec2(0, bpx.y < maxPx.y ? 1 : -1), ivec2(0), maxPx);
                float dX = texelFetch(depthtex1, rxPx, 0).r;
                float dY = texelFetch(depthtex1, ryPx, 0).r;
                vec3 vX = screenToView(vec3((vec2(rxPx) + 0.5) * texelSize(), dX));
                vec3 vY = screenToView(vec3((vec2(ryPx) + 0.5) * texelSize(), dY));
                vec3 rawN = cross(vX - bHitView, vY - bHitView);
                float nLen = length(rawN);
                vec3 hitViewN = (nLen > 1e-6) ? (rawN / nLen) : normalize(-bHitView);
                if (dot(hitViewN, -bHitView) < 0.0) hitViewN = -hitViewN;

                if (dot(hitViewN, viewR) < -0.02) {
                    float planeDist = abs(dot(bRayView - bHitView, hitViewN));
                    float planeTol  = max(0.025, 0.006 * (-bHitView.z));
                    if (planeDist <= planeTol) {
                        vec3 toHit = bHitView - viewPos;
                        float d = length(toHit);
                        if (d > 0.08 && dot(toHit / max(d, 1e-5), viewR) > 0.985) {
                            // The opaque resolve has already published the reflection class in this frame's guide.
                            vec4 guide = texelFetch(prevGuideTex, bpx, 0);
                            #if SPECULAR_BOUNCES > 1
                                bool needsDirectionalTrace = guide.z < SPECULAR_ROUGH_CUTOFF || (uint(guide.x) & 65536u) != 0u;
                            #else
                                bool needsDirectionalTrace = usesMirrorDirection(guide.z);
                            #endif
                            if (guide.w > 0.0 && needsDirectionalTrace) {
                                vec3 hitWorld = viewToWorld(bHitView - hitViewN * 0.004);
                                if (hasReflectionVoxel(ivec3(floor(worldToVoxelPos(hitWorld))))) return 0.0;
                            }
                            // colortex5 marks opaque emitters whose emission is baked into the lit scene.
                            if (beyondEmitterRange(d) && texelFetch(colortex5, bpx, 0).a > 0.0) return 0.0;
                            hitCol = max(imageLoad(colorimg0, bpx).rgb, vec3(0.0));
                            vec2 edge = min(b.xy, 1.0 - b.xy);
                            if (!any(isnan(hitCol)) && !any(isinf(hitCol))) {
                                return smoothstep(0.0, 0.03, min(edge.x, edge.y));
                            }
                        }
                    }
                }
            }
        }
        prevP = p;
        prevRayZ = rayZ;
    }
    return 0.0;
}

float screenEdgeFade(vec2 uv) {
    vec2 edge = min(uv, 1.0 - uv);
    return smoothstep(0.0, 0.08, min(edge.x, edge.y));
}

vec3 translucentReflection(vec3 V, vec3 geoN, vec3 N, vec3 F0, float roughness,
                           float traceWeight, vec3 ambient, ivec2 px, vec3 viewPos,
                           bool useOverrideFresnel, vec3 overrideFresnel) {
    vec3 envWeight = useOverrideFresnel ? overrideFresnel : evalEnvBRDF(F0, saturate(dot(N, V)), roughness);
    vec3 env = envWeight * ambient;
    #ifdef SPECULAR_REFLECTIONS
        if (traceWeight <= 0.0) return env;
        vec3 H;
        vec3 R = sampleReflectionDirection(V, N, roughness, px, frameCounter, H);
        float NdotL = dot(N, R);
        if (min(dot(geoN, R), NdotL) <= 1e-4) return env * (1.0 - traceWeight);
        vec3 screenColor = vec3(0.0);
        float screenWeight = 0.0;
        #ifdef SPECULAR_SSR
            if (vMatId == 1 || roughness >= 0.25) {
                screenWeight = traceTranslucentSSR(viewPos + mat3(gbufferModelView) * geoN * 0.04,
                    mat3(gbufferModelView) * R, stbn2D(px, frameCounter, 12u).x, screenColor);
            }
        #endif
        vec3 radiance = screenColor;
        if (screenWeight < 1.0) {
            float hitDist;
            vec3 origin = makeCornerSafeRayOrigin(vWorldPos, geoN, V, length(viewPos));
            // A grazing ray can traverse a whole row of occupied water cells without hitting land.
            int smoothSteps = vMatId == 1 ? 2 * VOXEL_RES_XZ + VOXEL_RES_Y : SPECULAR_MAX_STEPS;
            vec3 voxelColor = traceVoxelReflection(origin, R, roughness, px, smoothSteps, false, hitDist);
            radiance = mix(voxelColor, screenColor, screenWeight);
        }
        vec3 weight = useOverrideFresnel ? overrideFresnel : fresnelSchlick(F0, saturate(dot(V, H)));
        if (roughness >= 0.08) {
            weight *= smithGGX_G2OverG1(max(dot(N, V), 1e-4), NdotL, max(roughness * roughness, 0.002));
        }
        return mix(env, clamp(radiance, vec3(0.0), vec3(roughness < 0.12 ? 32.0 : 12.0)) * weight, traceWeight);
    #else
        return env;
    #endif
}

void main() {
    bool isWater = (vMatId == 1);
    // Water uses biome tint and procedural waves; its animated base sprite supplies no coverage or color.
    vec4 tex = isWater ? vec4(vColor.rgb, 1.0) : texture(gtexture, vTexCoord) * vColor;
    if (vMatId == 0 && tex.a < 0.01) discard;

    ivec2 fragPx = ivec2(gl_FragCoord.xy);
    vec2 res = screenSize();
    vec2 uv = gl_FragCoord.xy / res;

    vec3 geoN = normalize(vGeoNormal);
    if (!gl_FrontFacing) geoN = -geoN;
    vec3 eyePos = gbufferModelViewInverse[3].xyz;
    vec3 V = normalize(eyePos - vWorldPos);
    if (dot(geoN, V) < 0.0) geoN = -geoN;
    vec3 viewPos = (gbufferModelView * vec4(vWorldPos, 1.0)).xyz;
    float camDist = length(viewPos);

    // ---- Opaque background thickness along the view ray ----
    float opaqueDepth = texelFetch(depthtex1, fragPx, 0).r;
    float thickness = 64.0;
    if (opaqueDepth < 0.999999) {
        vec3 opaqueView = screenToView(vec3(uv, opaqueDepth));
        thickness = max(distance(opaqueView, viewPos), 0.0);
    }

    // ---- Material parameters ----
    bool viewFromInside = isWater && (isEyeInWater == 1);
    float f0Raw = isWater ? 0.02 : 0.04;
    float roughness = isWater ? 0.018 : ((vMatId == 2 && tex.a > 0.55 && tex.a < 0.9) ? 0.08 : 0.02);
    vec3 albedo = srgbToLinear(tex.rgb);
    float pbrEmission = 0.0;
    vec3 N;
    float ao = 1.0;
    float waterPath = viewFromInside ? min(camDist, 64.0) : min(thickness, 64.0);

    if (isWater) {
        vec3 absPos = vWorldPos + cameraPositionFract + vec3(cameraPositionInt % 4096);
        vec3 wavePos = evalWaterParallax(absPos, -V, geoN.y, stbn2D(fragPx, frameCounter, 15u).x, waterPath);
        vec3 waveN = evalWaterNormal(wavePos, geoN, camDist);
        N = stabilizeWaterNormal(waveN, geoN, V);
    } else {
        vec4 normTex = texture(normals, vTexCoord);
        N = applyLabPBRNormal(geoN, vTangent, normTex.rg);
        ao = normTex.b > 0.01 ? normTex.b : 1.0;
        vec4 specTex = sampleLabPBRSpec(specular, vTexCoord);
        if (maxOf(specTex.rgb) > 0.0 || (specTex.a > 0.0 && specTex.a < 1.0)) {
            float porosity, sss;
            decodeLabPBR(specTex, roughness, f0Raw, porosity, sss, pbrEmission);
        }
    }
    if (!isWater && dot(N, V) < 0.02) N = normalize(N + V * (0.02 - dot(N, V)));
    float NdotV = saturate(dot(N, V));
    vec3 F0 = evalLabPBRF0(f0Raw, albedo);
    bool metal = isLabPBRMetal(f0Raw);

    vec3 surfaceFresnel = viewFromInside
        ? vec3(evalSnellInternalFresnel(NdotV))
        : fresnelSchlick(F0, NdotV);
    vec3 transmittedFraction = metal ? vec3(0.0) : (1.0 - surfaceFresnel);

    // ---- Lighting environment ----
    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();
    vec3 sunVis = (maxOf(directCol) > 1e-4 && (dot(geoN, lightDir) > 0.0 || viewFromInside))
        ? sampleShadowPCSS(vWorldPos, viewFromInside ? -geoN : geoN, stbn2D(fragPx, frameCounter, 13u).x)
        : vec3(0.0);

    float skyFactor = smoothstep(0.35, 0.95, vSkyLight);
    vec3 rcUp = queryRadianceCache(vWorldPos + geoN * 0.3, geoN, 0u, false);
    vec3 skyUp = evalSky(vec3(0.0, 1.0, 0.0), false) * (0.15 + 0.85 * vSkyLight);
    vec3 skyDirAmb = evalSky(geoN, false) * (0.15 + 0.85 * vSkyLight);
    vec3 ambient = (luma(rcUp) > 1e-5) ? mix(rcUp, max(rcUp, skyDirAmb), skyFactor) : skyDirAmb;

    float traceWeight = metal ? 1.0 : 1.0 - smoothstep(0.50, SPECULAR_ROUGH_CUTOFF, roughness);
    vec3 reflected = translucentReflection(V, geoN, N, F0, roughness, traceWeight, ambient, fragPx, viewPos,
                                           viewFromInside, surfaceFresnel);

    // ---- GGX sun glint (Height-Correlated Smith) ----
    vec3 sunSpec = vec3(0.0);
    float NdotL = dot(N, lightDir);
    if (!viewFromInside && NdotL > 0.0 && maxOf(sunVis) > 0.0) {
        vec3 H = normalize(lightDir + V);
        float NdotH = saturate(dot(N, H));
        float alpha = max(roughness * roughness, isWater ? 0.0008 : 0.002);
        float a2 = alpha * alpha;
        float dDenom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = min(a2 / max(PI * dDenom * dDenom, 1e-6), isWater ? 4000.0 : 2000.0);
        float Vis = visSmithGGX(max(NdotV, 1e-4), NdotL, alpha);
        vec3 F = fresnelSchlick(F0, saturate(dot(V, H)));
        sunSpec = directCol * sunVis * (F * D * Vis * NdotL);
    }

    vec3 L = vec3(0.0);
    vec3 T = vec3(1.0);
    vec2 refrOffset = vec2(0.0);

    if (isWater) {
        vec3 Tvol, inscatter;
        evalWaterVolume(waterPath, -V, N, lightDir, directCol, sunVis, skyUp, vSkyLight,
                        vColor.rgb, viewFromInside, Tvol, inscatter);

        vec3 viewN = mat3(gbufferModelView) * (N - geoN);
        if (!viewFromInside) {
            // Above-water: seabed transmits through volume -> crosses surface -> reflection adds on top.
            L = transmittedFraction * inscatter + reflected + sunSpec;
            T = transmittedFraction * Tvol;

            float refrStrength = min(thickness, 3.0) * 0.55 / max(-viewPos.z, 1.0);
            refrOffset = viewN.xy * refrStrength * screenEdgeFade(uv);
        } else {
            // Submerged looking up: Snell's window + TIR travel through waterPath meters to the eye.
            L = reflected * Tvol + inscatter;
            T = transmittedFraction * Tvol;

            float refrStrength = (1.0 - surfaceFresnel.x) * 0.16;
            refrOffset = viewN.xy * refrStrength * screenEdgeFade(uv);
        }
    } else {
        // Glass / Ice / Stained Glass / generic translucents.
        float coverage = tex.a;
        float diffuseOpacity = smoothstep(0.62, 0.98, coverage);
        vec3 glassT = evalGlassTransmittance(tex.rgb, coverage);

        vec3 surfaceLit = albedo * (INV_PI * directCol * sunVis * max(NdotL, 0.0) + ambient * ao);
        L = transmittedFraction * diffuseOpacity * surfaceLit + reflected + sunSpec;
        T = transmittedFraction * (1.0 - diffuseOpacity) * glassT;
    }

    float emission = surfaceEmission(pbrEmission, vBlockEmit * tex.a, f0Raw, albedo);
    L += albedo * (emission * EMISSIVE_SCALE);
    L = max(L, vec3(0.0));
    T = clamp(T, vec3(0.0), vec3(1.0));

    // Exact algebraically non-double-fogging Aerial Perspective for distant oceans & translucents:
    // since opaque background `bg` already carries T_ap + L_ap, writing L' = L*T_ap + L_ap*(1 - T)
    // yields bg_fogged * T + L' == (bg * T + L) * T_ap + L_ap identically!
    if (!viewFromInside && isEyeInWater == 0) {
        vec3 apTrans, apInscatter;
        evalAerialPerspective(vWorldPos, -V, vSkyLight, apTrans, apInscatter);
        L = L * apTrans + apInscatter * (1.0 - T);
    }
    if (any(isnan(L)) || any(isinf(L))) L = vec3(0.0);

    radianceOut = vec4(L, saturate(1.0 - luma(T)));
    transmitOut = vec4(T, 1.0);
    refractOut  = vec4(refrOffset, 0.0, 1.0);
}
