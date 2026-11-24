#version 430 compatibility

// Pass: Combine Direct Sunlight + Emission + Diffuse GI + Specular GI + Sky + Debug Views,
// and publish prevGuideImg (exact float32 integers!) + diffuse-safe colortex5 (lit history).
#include "/lib/SkyShadow.glsl"
#if DEBUG_VIEW == 4
    #include "/lib/RadianceCache.glsl"
#endif

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

layout(rgba16f) uniform writeonly image2D colorimg0;
layout(rgba16f) uniform writeonly image2D colorimg5;
layout(rgba32f) uniform writeonly image2D prevGuideImg;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 res = ivec2(screenSize());
    if (any(greaterThanEqual(px, res))) return;

    vec2 uv = (vec2(px) + 0.5) * texelSize();
    float depth = texelFetch(depthtex1, px, 0).r;

    if (depth >= 0.999999) {
        vec3 skyDir = normalize(viewToWorld(screenToView(vec3(uv, 1.0))));
        vec3 sky = evalSky(skyDir, true);
        imageStore(colorimg0, px, vec4(sky, 1.0));
        imageStore(colorimg5, px, vec4(sky, 1.0));
        imageStore(prevGuideImg, px, vec4(0.0));
        return;
    }

    GBufferMat mat = readGBuffer(px);
    vec3 viewPos  = screenToView(vec3(uv, depth));
    vec3 worldPos = viewToWorld(viewPos);
    vec3 viewDir  = normalize(-worldPos);

    // Publish exact float32 integer-encoded 16-bit normals (never denormal uintBitsToFloat!).
    imageStore(prevGuideImg, px, vec4(
        float(packNormal16(mat.geoNormal)),
        float(packNormal16(mat.shadeNormal)),
        mat.roughness,
        -viewPos.z
    ));

    // Direct Sunlight / Moonlight (4-tap blue-noise-rotated colored PCF on Shadow Map)
    vec3 lightDir = getShadowLightDir();
    vec3 directCol = getDirectLightColor();
    float NdotL = max(dot(mat.shadeNormal, lightDir), 0.0);
    vec3 shadowVis = (NdotL > 0.0 && dot(mat.geoNormal, lightDir) > 0.0 && maxOf(directCol) > 1e-4)
        ? sampleShadowPCF(worldPos, mat.geoNormal, stbn2D(px, frameCounter, 7u).x)
        : vec3(0.0);
    vec3 directDiffuse = mat.albedo * INV_PI * directCol * (NdotL * shadowVis);

    // Direct Specular Highlight (GGX with LabPBR RGB F0 & physical NDF cap)
    vec3 F0_rgb = evalLabPBRF0(mat.f0, mat.albedo);
    bool isMetal = isLabPBRMetal(mat.f0);
    vec3 directSpec = vec3(0.0);
    if (NdotL > 0.0 && maxOf(shadowVis) > 0.0) {
        vec3 H = normalize(lightDir + viewDir);
        float NdotH = max(dot(mat.shadeNormal, H), 0.0);
        float alpha = max(mat.roughness * mat.roughness, 0.015);
        float a2 = alpha * alpha;
        float dDenom = NdotH * NdotH * (a2 - 1.0) + 1.0;
        float D = min(a2 / max(PI * dDenom * dDenom, 1e-5), 128.0);
        vec3 F = fresnelSchlick(F0_rgb, max(dot(H, viewDir), 0.0));
        directSpec = F * D * 0.25 * directCol * (NdotL * shadowVis);
    }

    vec3 diffuseGI  = max(texelFetch(colortex3, px, 0).rgb, vec3(0.0));
    vec3 specularGI = max(texelFetch(colortex4, px, 0).rgb, vec3(0.0));
    vec3 emissive   = (mat.emission > 0.0) ? (evalDirectEmitterColor(mat.albedo) * (mat.emission * EMISSIVE_SCALE)) : vec3(0.0);

    vec3 F_env = fresnelSchlick(F0_rgb, max(dot(mat.shadeNormal, viewDir), 0.0));
    float specMask = (mat.roughness <= SPECULAR_ROUGH_CUTOFF && !mat.isHand) ? 1.0 : 0.0;
    vec3 kD = isMetal ? vec3(0.0) : (1.0 - F_env * specMask);

    vec3 diffuseLit = directDiffuse * kD + (diffuseGI * mat.albedo) * kD + emissive;
    vec3 finalLit   = diffuseLit + directSpec + specularGI * F_env * specMask;

    // Store pure diffuse+emissive radiance in colortex5.rgb, and emission flag in .a so
    // DIFFUSE_SSRT can stay enabled alongside NEE without double-counting block lamps!
    imageStore(colorimg5, px, vec4(diffuseLit, mat.emission));

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
                SurfaceVoxel sv = surfacePool.surfaces[voxelLinearIndex(vCoord)];
                bool hasT1 = (sv.face[0] | sv.face[1] | sv.face[2] | sv.face[3] | sv.face[4] | sv.face[5]) != 0u;
                bool hasT2 = (sv.primHead != PRIM_END);
                dbg = vec3(hasT2 ? 1.0 : 0.0, hasT1 ? 1.0 : 0.0, 0.0);
            }
        }
        finalLit = dbg;
    #endif

    imageStore(colorimg0, px, vec4(finalLit, 1.0));
}
