#version 430 compatibility

// Physically-Based Translucent Forward Pass (Water / Glass / Ice / Stained Glass / Other Translucents)
// Writes separable layers resolved in composite TAA so opaque speculars behind glass are preserved:
//   colortex6.rgb = reflected + in-scattered + emitted radiance, .a = layer opacity (1 - luma(T))
//   colortex7.rgb = per-channel transmittance (multiplicative blend, clear = 1)
//   colortex8.rg  = refraction screen-UV offset (nearest layer)
// - Water: two-sided procedural world-space wave normals, Fresnel reflections,
//   Beer-Lambert biome-tinted absorption + single-scatter in-scattering, GGX sun glint, refraction.
// - Glass/Ice: smooth dielectric Fresnel, colored transmittance, diffuse frame/frost lighting.
// - Reflections: current-frame SSR when enabled, with voxel PT for off-screen geometry.
#define TRACE_HIGH_PRIM_GUARD
#include "/lib/SpecularTrace.glsl"
#include "/lib/MaterialMaps.glsl"

uniform sampler2D gtexture;
uniform sampler2D normals;
uniform sampler2D specular;

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

float linZ(float depth) {
    float ndcZ = depth * 2.0 - 1.0;
    return -gbufferProjectionInverse[3][2] / max(gbufferProjectionInverse[2][3] * ndcZ + gbufferProjectionInverse[3][3], 1e-6);
}

// ---- Procedural water wave height (sum of directional Gerstner-like sines + value noise detail) ----
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float waveHeight(vec2 p, float t) {
    float h = 0.0;
    h += 0.050 * sin(dot(p, vec2( 0.80,  0.60)) * 1.10 + t * 1.30);
    h += 0.035 * sin(dot(p, vec2(-0.55,  0.83)) * 1.70 + t * 1.75);
    h += 0.020 * sin(dot(p, vec2( 0.25, -0.97)) * 2.90 + t * 2.40);
    h += 0.010 * (valueNoise(p * 1.6 + vec2(t * 0.35, t * 0.22)) - 0.5);
    h += 0.005 * (valueNoise(p * 2.1 - vec2(t * 0.55, t * 0.41)) - 0.5);
    return h;
}

vec3 waterWaveNormal(vec3 worldPos, vec3 geoN) {
    if (abs(geoN.y) < 0.5) return geoN;
    vec2 p = worldPos.xz + cameraPositionFract.xz + vec2(cameraPositionInt.xz % 4096);
    float t = frameTimeCounter;
    const float e = 0.06;
    float h0 = waveHeight(p, t);
    float hx = waveHeight(p + vec2(e, 0.0), t);
    float hz = waveHeight(p + vec2(0.0, e), t);
    vec3 n = normalize(vec3(-(hx - h0) / e, 1.0, -(hz - h0) / e));
    // Fade wave detail with distance to avoid sub-pixel normal aliasing.
    float dist = length(worldPos);
    return sign(geoN.y) * normalize(mix(n, vec3(0.0, 1.0, 0.0), smoothstep(24.0, 96.0, dist)));
}

// The mesh stays flat while its shading normal ripples. Compress low reflection elevations
// smoothly above the mesh horizon, then use the corresponding normal for both Fresnel and tracing.
vec3 waterReflectionNormal(vec3 N, vec3 geoN, vec3 V) {
    vec3 R = reflect(-V, N);
    float elevation = dot(R, geoN);
    const float minElevation = 0.001;
    float shoulder = max(0.5 * dot(V, geoN), 2.0 * minElevation);
    if (elevation >= shoulder) return N;
    float span = shoulder - minElevation;
    float corrected = minElevation + span * span / (span + shoulder - elevation);
    R = normalize(R + (corrected - elevation) * geoN);
    return normalize(V + R);
}

// Confidence falls to zero at the edge so PT can take over before the screen hit disappears.
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
    vec3 prevP = s0;
    for (int i = 0; i < STEPS; ++i) {
        float stepIndex = (i == STEPS - 1) ? float(STEPS) : float(i) + 0.5 + 0.5 * dither;
        vec3 p = s0 + stepV * stepIndex;
        if (any(lessThan(p.xy, vec2(0.0))) || any(greaterThanEqual(p.xy, vec2(1.0))) || p.z <= 0.0) return 0.0;
        float sd = texelFetch(depthtex1, ivec2(p.xy * res), 0).r;
        if (sd >= 0.999999 || p.z <= sd) { prevP = p; continue; }

        // Binary refinement between prevP (in front) and p (behind).
        vec3 a = prevP;
        vec3 b = p;
        for (int k = 0; k < 8; ++k) {
            vec3 m = 0.5 * (a + b);
            float md = texelFetch(depthtex1, ivec2(m.xy * res), 0).r;
            if (m.z > md) b = m; else a = m;
        }
        float bd = texelFetch(depthtex1, ivec2(b.xy * res), 0).r;
        float diffZ = linZ(b.z) - linZ(bd);
        if (bd < 0.999999 && diffZ > -0.05 && diffZ < 0.35 + 0.04 * linZ(bd)) {
            hitCol = max(texelFetch(colortex5, ivec2(b.xy * res), 0).rgb, vec3(0.0));
            vec2 edge = min(b.xy, 1.0 - b.xy);
            if (any(isnan(hitCol)) || any(isinf(hitCol))) return 0.0;
            return smoothstep(0.0, 0.03, min(edge.x, edge.y));
        }
        return 0.0;
    }
    return 0.0;
}

float screenEdgeFade(vec2 uv) {
    vec2 edge = min(uv, 1.0 - uv);
    return smoothstep(0.0, 0.08, min(edge.x, edge.y));
}

vec3 translucentReflection(vec3 V, vec3 geoN, vec3 N, vec3 F0, float roughness,
                           float traceWeight, vec3 ambient, ivec2 px, vec3 viewPos) {
    vec3 env = evalEnvBRDF(F0, saturate(dot(N, V)), roughness) * ambient;
    #ifdef SPECULAR_REFLECTIONS
        if (traceWeight <= 0.0) return env;
        vec3 H;
        vec3 R = sampleReflectionDirection(V, N, roughness, px, H);
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
        vec3 weight = fresnelSchlick(F0, saturate(dot(V, H)));
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
    float roughness = isWater ? 0.08 : ((vMatId == 2 && tex.a > 0.55 && tex.a < 0.9) ? 0.08 : 0.02);
    vec3 albedo = srgbToLinear(tex.rgb);
    float pbrEmission = 0.0;
    vec3 N;
    float ao = 1.0;
    if (isWater) {
        N = waterReflectionNormal(waterWaveNormal(vWorldPos, geoN), geoN, V);
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

    // Use a gradual grazing reflection on both sides so the shore stays visible from below.
    vec3 transmittedFraction = metal ? vec3(0.0) : 1.0 - fresnelSchlick(F0, NdotV);

    // ---- Lighting environment ----
    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();
    vec3 sunVis = (maxOf(directCol) > 1e-4 && dot(geoN, lightDir) > 0.0)
        ? sampleShadowPCSS(vWorldPos, geoN, stbn2D(fragPx, frameCounter, 13u).x)
        : vec3(0.0);

    float skyFactor = smoothstep(0.35, 0.95, vSkyLight);
    vec3 rcUp = queryRadianceCache(vWorldPos + geoN * 0.3, geoN, 0u, false);
    vec3 skyUp = evalSky(geoN, false) * (0.15 + 0.85 * vSkyLight);
    vec3 ambient = (luma(rcUp) > 1e-5) ? mix(rcUp, max(rcUp, skyUp), skyFactor) : skyUp;

    float traceWeight = metal ? 1.0 : 1.0 - smoothstep(0.50, SPECULAR_ROUGH_CUTOFF, roughness);
    vec3 reflected = translucentReflection(V, geoN, N, F0, roughness, traceWeight, ambient, fragPx, viewPos);

    // ---- GGX sun glint (Height-Correlated Smith) ----
    vec3 sunSpec = vec3(0.0);
    float NdotL = dot(N, lightDir);
    if (NdotL > 0.0 && maxOf(sunVis) > 0.0) {
        vec3 H = normalize(lightDir + V);
        float NdotH = saturate(dot(N, H));
        float alpha = max(roughness * roughness, 0.002);
        float a2 = alpha * alpha;
        float dDenom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = min(a2 / max(PI * dDenom * dDenom, 1e-6), 2000.0);
        float Vis = visSmithGGX(max(NdotV, 1e-4), NdotL, alpha);
        vec3 F = fresnelSchlick(F0, saturate(dot(V, H)));
        sunSpec = directCol * sunVis * (F * D * Vis * NdotL);
    }

    vec3 L = vec3(0.0);
    vec3 T = vec3(1.0);
    vec2 refrOffset = vec2(0.0);

    if (isWater) {
        // Biome-tinted Beer-Lambert absorption + single-scatter in-scattering.
        vec3 sigmaT = waterExtinction(vColor.rgb);
        float pathLen = viewFromInside ? 0.0 : min(thickness, 64.0);
        vec3 Tvol = exp(-sigmaT * pathLen);
        // Isotropic single scattering distributes directional irradiance over 4*pi steradians.
        vec3 lightInWater = ambient + directCol * sunVis * (0.25 * INV_PI);
        vec3 inscatter = (1.0 - Tvol) * (WATER_SCATTERING / sigmaT) * lightInWater;

        L = transmittedFraction * inscatter;
        T = transmittedFraction * Tvol;

        // Screen-space refraction offset (scaled by water depth, faded at screen borders).
        if (!viewFromInside) {
            vec3 viewN = mat3(gbufferModelView) * (N - geoN);
            float refrStrength = min(thickness, 3.0) * 0.55 / max(-viewPos.z, 1.0);
            refrOffset = viewN.xy * refrStrength * screenEdgeFade(uv);
        }
    } else {
        // Glass / Ice / Stained Glass / generic translucents.
        float coverage = tex.a;
        float diffuseOpacity = smoothstep(0.62, 0.98, coverage);
        vec3 glassT = evalGlassTransmittance(tex.rgb, coverage);

        vec3 surfaceLit = albedo * (INV_PI * directCol * sunVis * max(NdotL, 0.0) + ambient * ao);
        L = transmittedFraction * diffuseOpacity * surfaceLit;
        T = transmittedFraction * (1.0 - diffuseOpacity) * glassT;
    }

    float emission = max(pbrEmission, vBlockEmit * tex.a);
    L += reflected + sunSpec + evalDirectEmitterColor(albedo) * (emission * EMISSIVE_SCALE);
    L = max(L, vec3(0.0));
    T = clamp(T, vec3(0.0), vec3(1.0));
    if (any(isnan(L)) || any(isinf(L))) L = vec3(0.0);

    radianceOut = vec4(L, saturate(1.0 - luma(T)));
    transmitOut = vec4(T, 1.0);
    refractOut  = vec4(refrOffset, 0.0, 1.0);
}
