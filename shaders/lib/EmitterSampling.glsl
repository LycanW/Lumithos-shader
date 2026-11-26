#ifndef LIB_EMITTER_SAMPLING_GLSL
#define LIB_EMITTER_SAMPLING_GLSL

#include "/lib/Raytrace.glsl"

struct EmitterSample {
    vec3 position;
    vec3 normal;
    vec2 uv;
    vec3 tint;
    float blockEmission;
    float inverseAreaPdf;
    bool translucent;
    bool triangle;
};

EmitterSample sampleEmitterSurface(uint index, ivec3 voxel, vec3 receiver, vec2 random) {
    EmitterSample samplePoint;
    samplePoint.inverseAreaPdf = 0.0;
    uvec2 shape = emitterGeometry.shapes[index];
    if (shape.x == 0u) return samplePoint;
    vec3 voxelOrigin = vec3(voxel);
    samplePoint.triangle = (shape.x & EMITTER_CUBE) == 0u;

    if (samplePoint.triangle) {
        float area = uintBitsToFloat(shape.y);
        float draw = random.x * area;
        uint first = index * EMITTER_TRIANGLE_LIMIT;
        uint lo = 0u;
        uint hi = shape.x - 1u;
        float lower = 0.0;
        while (lo < hi) {
            uint mid = (lo + hi) >> 1u;
            float upper = uintBitsToFloat(emitterGeometry.triangles[first + mid].y);
            if (draw < upper) {
                hi = mid;
            } else {
                lo = mid + 1u;
                lower = upper;
            }
        }

        uvec2 entry = emitterGeometry.triangles[first + lo];
        float triangleArea = uintBitsToFloat(entry.y) - lower;
        if (triangleArea <= 0.0) return samplePoint;
        vec3 p0, e1, e2;
        vec2 uv0, uv1, uv2;
        uint nextIndex;
        unpackScenePrim(primPool.prims[entry.x], p0, e1, e2, uv0, uv1, uv2, samplePoint.tint,
                        samplePoint.blockEmission, samplePoint.translucent, nextIndex);
        // Reuse the residual within the selected CDF interval as a uniform draw.
        float root = sqrt(saturate((draw - lower) / triangleArea));
        float u = root * random.y;
        float v = root - u;
        vec3 local = p0 + e1 * u + e2 * v;
        // A spanning triangle can be registered in multiple voxels. Each voxel samples
        // its part of the triangle by rejecting points outside that voxel.
        vec3 normal = normalize(cross(e1, e2));
        if (!inEmitterCell(local, normal)) return samplePoint;

        samplePoint.position = voxelOrigin + local;
        samplePoint.normal = normal;
        samplePoint.uv = uv0 * (1.0 - root) + uv1 * u + uv2 * v;
        samplePoint.inverseAreaPdf = area;
    } else {
        uint faces = frontEmitterFaces(shape.x, receiver - voxelOrigin);
        int count = bitCount(faces);
        if (count == 0) return samplePoint;
        float faceDraw = random.x * float(count);
        uint skip = min(uint(faceDraw), uint(count - 1));
        for (uint i = 0u; i < skip; ++i) faces &= faces - 1u;

        uint face = uint(findLSB(faces));
        vec3 normal = faceIndexToNormal(face);
        vec3 tangent, bitangent;
        faceBasis(face, tangent, bitangent);
        float u = fract(faceDraw);
        float v = random.y;
        vec3 local = 0.5 + normal * 0.5 + tangent * (u - 0.5) + bitangent * (v - 0.5);

        SurfaceVoxel surface = readSurface(voxelLinearIndex(voxel));
        vec2 invAtlas = 1.0 / vec2(max(textureSize(colortex14, 0), ivec2(1)));
        bool tinted;
        samplePoint.uv = decodeFaceAtlasUV(surface.face[face], face, local, invAtlas, tinted,
                                           samplePoint.translucent, samplePoint.blockEmission);
        samplePoint.position = voxelOrigin + local;
        samplePoint.normal = normal;
        samplePoint.tint = tinted ? unpackTint(surface.tintAndSky) : vec3(1.0);
        samplePoint.inverseAreaPdf = float(count); // Every selected voxel face has unit area.
    }
    return samplePoint;
}

// Score a small subset of the existing local light grid, then evaluate one
// point on the selected emitter's geometry.
vec3 sampleDirectLightsNEE(vec3 worldPos, vec3 normal, int maxCandidates, vec2 lightRandom, vec2 surfaceRandom,
                           out vec3 outLightDir) {
    outLightDir = vec3(0.0, 1.0, 0.0);
    if (lightList.lightCount == 0u) return vec3(0.0);

    vec3 vPos = worldToVoxelPos(worldPos);
    if (!inVoxelBounds(ivec3(floor(vPos)))) return vec3(0.0);
    ivec3 c0 = ivec3(floor((vPos - 16.0) * (1.0 / 32.0)));

    uint cIdxs[8];
    uint cCounts[8];
    uint localTotal = 0u;
    int nCells = 0;

    for (int dz = 0; dz < 2; ++dz) {
        int cz = c0.z + dz;
        if (cz < 0 || cz >= LIGHT_CELL_XZ) continue;
        for (int dy = 0; dy < 2; ++dy) {
            int cy = c0.y + dy;
            if (cy < 0 || cy >= LIGHT_CELL_Y) continue;
            for (int dx = 0; dx < 2; ++dx) {
                int cx = c0.x + dx;
                if (cx < 0 || cx >= LIGHT_CELL_XZ) continue;
                uint ci = uint(cx + cz * LIGHT_CELL_XZ + cy * (LIGHT_CELL_XZ * LIGHT_CELL_XZ));
                uint rawCnt = lightList.cellCount[ci];
                if (rawCnt > 0u) {
                    uint cnt = min(rawCnt, LIGHT_CELL_CAP);
                    cIdxs[nCells]   = ci;
                    cCounts[nCells] = cnt;
                    localTotal += cnt;
                    ++nCells;
                }
            }
        }
    }
    if (localTotal == 0u) return vec3(0.0);

    uint nCand = min(localTotal, uint(maxCandidates));
    float stepF = float(localTotal) / float(nCand);
    float startF = lightRandom.x * stepF;

    float wSum = 0.0;
    float selW = 0.0;
    float selectedShare = 0.0;
    uint selected = 0u;
    float pick = lightRandom.y;
    float extent = emitterHorizonExtent(normal);

    for (uint i = 0u; i < nCand; ++i) {
        uint target = uint(startF + float(i) * stepF) % localTotal;
        uint idx = 0u;
        uint rem = target;
        for (int c = 0; c < nCells; ++c) {
            if (rem < cCounts[c]) {
                idx = lightList.cellLights[cIdxs[c] * LIGHT_CELL_CAP + rem];
                break;
            }
            rem -= cCounts[c];
        }

        SceneLight light = lightList.lights[idx];
        vec3 delta = light.posAndPower.xyz - vPos;
        float dist2 = dot(delta, delta);
        float share = localLightWindow(dist2, light.posAndPower.w);
        if (share <= 0.0) continue;

        vec3 center = floor(light.posAndPower.xyz) + 0.5 - vPos;
        float projected = dot(center, normal);
        if (projected + extent <= 0.0) continue;
        float cosine = saturate(max(dot(delta, normal), 0.25 * (projected + extent)) *
                                inversesqrt(max(dist2, 1e-4)));
        float w = luma(light.colorAndMeta.rgb) * (light.posAndPower.w * share * cosine / (dist2 + 0.35));
        if (w <= 0.0) continue;

        // A uniform draw remains uniform within the interval it selected.
        // Remap it so the next reservoir decision can reuse the blue-noise draw.
        float total = wSum + w;
        float draw = pick * total;
        if (draw < w) {
            selected = idx;
            selectedShare = share;
            selW = w;
            pick = draw / w;
        } else {
            pick = (draw - w) / wSum;
        }
        pick = min(pick, 0.99999994);
        wSum = total;
    }
    if (selW <= 0.0) return vec3(0.0);

    ivec3 voxel = ivec3(floor(lightList.lights[selected].posAndPower.xyz));
    EmitterSample point = sampleEmitterSurface(selected, voxel, vPos, surfaceRandom);
    if (point.inverseAreaPdf <= 0.0) return vec3(0.0);
    vec3 delta = point.position - vPos;
    float dist2 = dot(delta, delta);
    if (dist2 <= 1e-8) return vec3(0.0);
    float distanceToLight = sqrt(dist2);
    vec3 direction = delta / distanceToLight;
    float NoL = dot(normal, direction);
    float projected = dot(point.normal, -direction);
    // The non-cube path intersection shades both sides of packed triangles.
    float emitterCosine = point.triangle ? abs(projected) : projected;
    if (min(NoL, emitterCosine) <= 0.0) return vec3(0.0);

    vec4 tex = textureLod(colortex14, point.uv, 1.0);
    if (tex.a < 0.1) return vec3(0.0);
    vec3 albedo = srgbToLinear(tex.rgb * point.tint);
    ivec2 specSize = textureSize(colortex15, 0);
    ivec2 specPixel = clamp(ivec2(point.uv * vec2(specSize)), ivec2(0), specSize - 1);
    vec2 spec = texelFetch(colortex15, specPixel, 0).ga;
    float coverage = point.translucent ? tex.a : 1.0;
    float emission = surfaceEmission(decodeLabPBREmission(spec.y), point.blockEmission * coverage, spec.x, albedo);
    if (emission <= 0.0 || maxOf(albedo) <= 0.0) return vec3(0.0);

    vec3 visibility = traceVisibility(vPos, direction, distanceToLight);
    if (maxOf(visibility) <= 1e-4) return vec3(0.0);
    outLightDir = direction;

    float inversePdf = (stepF * wSum / selW) * point.inverseAreaPdf;
    float geometry = NoL * emitterCosine * INV_PI / dist2;
    // Diffuse emitter hits integrate the complementary share of this range window.
    return albedo * visibility * (emission * EMISSIVE_SCALE * geometry * inversePdf * selectedShare);
}

#endif
