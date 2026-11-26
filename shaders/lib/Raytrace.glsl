#ifndef LIB_RAYTRACE_GLSL
#define LIB_RAYTRACE_GLSL

#include "/lib/SceneGeo.glsl"

struct TraceHit {
    bool  hit;
    bool  exitedGrid;
    bool  fromTrack2;
    float dist;
    vec3  worldPos;
    vec3  normal;
    vec3  albedo;
    vec3  throughput;
    float roughness;
    float f0;
    float pbrEmission;
    float blockEmission;
    float coverage;
    ivec3 voxel;
    float skyLight;
};

TraceHit makeMissHit() {
    TraceHit h;
    h.hit = false;
    h.exitedGrid = false;
    h.fromTrack2 = false;
    h.dist = 1e6;
    h.worldPos = vec3(0.0);
    h.normal = vec3(0.0, 1.0, 0.0);
    h.albedo = vec3(0.0);
    h.throughput = vec3(1.0);
    h.roughness = 1.0;
    h.f0 = 0.04;
    h.pbrEmission = 0.0;
    h.blockEmission = 0.0;
    h.coverage = 1.0;
    h.voxel = ivec3(-1);
    h.skyLight = 1.0;
    return h;
}

void readHitMaterial(inout TraceHit hit, vec2 uv, ivec2 specSize, vec3 albedoSrgb, float blockEmit, float coverage) {
    ivec2 px = clamp(ivec2(uv * vec2(specSize)), ivec2(0), specSize - 1);
    vec4 spec = texelFetch(colortex15, px, 0);
    hit.albedo = srgbToLinear(albedoSrgb);
    hit.roughness = 1.0 - spec.r;
    hit.f0 = spec.g;
    hit.pbrEmission = decodeLabPBREmission(spec.a);
    hit.blockEmission = blockEmit;
    hit.coverage = coverage;
}

vec3 secondaryHitLighting(TraceHit hit, vec3 rayDir, vec3 lightDir, vec3 direct, vec3 bounce, bool mirrorPath) {
    if (!isLabPBRMetal(hit.f0)) return hit.albedo * (direct * INV_PI + bounce * 0.85);
    float nv = max(-dot(hit.normal, rayDir), 1e-4);
    vec3 F0 = evalLabPBRF0(hit.f0, hit.albedo);
    // The surface cache supplies the low-frequency incident light for this last bounce.
    vec3 reflected = evalEnvBRDF(F0, nv, hit.roughness) * bounce;
    if (maxOf(direct) > 0.0) {
        float nl = max(dot(hit.normal, lightDir), 0.0);
        vec3 response;
        if (mirrorPath) {
            vec3 h = normalize(lightDir - rayDir);
            vec3 F = fresnelSchlick(F0, saturate(dot(-rayDir, h)));
            response = F * directGGXLobe(nv, nl, saturate(dot(hit.normal, h)), hit.roughness);
        } else {
            // Last stochastic bounce: distribute the GGX directional-hemispherical reflectance over outgoing directions.
            response = evalEnvBRDF(F0, nl, hit.roughness) * INV_PI;
        }
        reflected += direct * response;
    }
    return reflected;
}

float sceneLightRange(float strength) {
    return clamp(strength * 16.0, 4.0, 16.0);
}

// NEE owns emission only when this emitter is actually selectable from the receiver.
bool directLightCovers(ivec3 voxel, vec3 receiver, vec3 normal) {
    if (!inVoxelBounds(voxel)) return false;
    ivec3 cell = voxel / 32;
    uint id = uint(cell.x + LIGHT_CELL_XZ * (cell.z + LIGHT_CELL_XZ * cell.y));
    uint slot = surfacePool.surfaces[voxelLinearIndex(voxel)].primHead >> 21u;
    if (slot == 0u) return false;
    vec4 lamp = lightList.lights[lightList.cellLights[id * LIGHT_CELL_CAP + slot - 1u]].posAndPower;
    vec3 delta = lamp.xyz - worldToVoxelPos(receiver);
    float reach = sceneLightRange(lamp.w);
    float dist2 = dot(delta, delta);
    return dist2 < reach * reach && dot(delta, normal) > 1e-4 * sqrt(max(dist2, 1e-6));
}

vec3 indirectHitEmission(TraceHit hit, vec3 receiver, vec3 normal) {
    float emission = surfaceEmission(hit.pbrEmission, hit.blockEmission * hit.coverage, hit.f0, hit.albedo);
    if (emission <= 0.0 || (hit.blockEmission > 0.0 && directLightCovers(hit.voxel, receiver, normal))) return vec3(0.0);
    return hit.albedo * (emission * EMISSIVE_SCALE);
}

// Structural helper: returns true iff voxel `vCoord` is an opaque Track-1 full cube.
bool isOpaqueTrack1Voxel(ivec3 vCoord) {
    if (!inVoxelBounds(vCoord)) return false;
    if (texelFetch(voxelGridTex, vCoord, 0).r == 0u) return false;
    SurfaceVoxel sv = readSurface(voxelLinearIndex(vCoord));
    if (sv.primHead != PRIM_END) return false;
    for (int f = 0; f < 6; ++f) {
        if (sv.face[f] != 0u && ((sv.face[f] >> 26u) & 1u) == 0u) return true;
    }
    return false;
}

// Structural corner-safe ray origin:
// At high resolutions (1080p/2K/4K), depth quantization + TAA subpixel jitter near a 3-way concave
// corner can place `worldPos` 1-3 mm inside a perpendicular solid wall or ceiling voxel.
// Pulling along `toCam` guarantees the ray ALWAYS starts inside the open indoor air voxel!
vec3 makeCornerSafeRayOrigin(vec3 worldPos, vec3 geoNormal, vec3 toCam, float camDist) {
    vec3 orig = worldPos + geoNormal * (0.018 + 0.001 * camDist) + toCam * (0.014 + 0.0006 * camDist);
    ivec3 vc = ivec3(floor(worldToVoxelPos(orig)));
    if (isOpaqueTrack1Voxel(vc)) {
        orig = worldPos + geoNormal * 0.045 + toCam * 0.065;
        vc = ivec3(floor(worldToVoxelPos(orig)));
        if (isOpaqueTrack1Voxel(vc)) {
            orig = worldPos + toCam * 0.16;
        }
    }
    return orig;
}

// Hybrid 64-Way Hierarchical DDA (16x16x16 -> 4x4x4 -> 1x1x1) with Stained-Glass Transmission
// Track 1: O(1) Full-Cube 6-Face Descriptor with exact 8-way UV rotation/mirroring
// Track 2: Exact Triangle Intersection for Non-Cube / Custom / Mod Geometry
TraceHit traceScene(vec3 worldOrigin, vec3 rayDir, int maxSteps, float atlasLod, bool passTranslucent) {
    TraceHit result = makeMissHit();
    vec3 throughput = vec3(1.0);

    vec3 vOrigin = worldToVoxelPos(worldOrigin);
    vec3 safeDir = vec3(
        abs(rayDir.x) < 1e-6 ? (rayDir.x < 0.0 ? -1e-6 : 1e-6) : rayDir.x,
        abs(rayDir.y) < 1e-6 ? (rayDir.y < 0.0 ? -1e-6 : 1e-6) : rayDir.y,
        abs(rayDir.z) < 1e-6 ? (rayDir.z < 0.0 ? -1e-6 : 1e-6) : rayDir.z
    );
    vec3 invDir = 1.0 / safeDir;
    ivec3 stepSign = ivec3(greaterThan(safeDir, vec3(0.0))) * 2 - 1;
    ivec3 stepPos  = ivec3(greaterThan(safeDir, vec3(0.0)));
    ivec2 atlasTexSize = max(textureSize(colortex14, 0), ivec2(1));
    ivec2 specTexSize  = max(textureSize(colortex15, 0), ivec2(1));
    vec2  invAtlasSize = 1.0 / vec2(atlasTexSize);

    // Entry face index for each axis when stepping into a voxel:
    // +X step enters -X face (1), -X step enters +X face (0), etc.
    ivec3 enterFace = ivec3(
        stepSign.x > 0 ? 1 : 0,
        stepSign.y > 0 ? 3 : 2,
        stepSign.z > 0 ? 5 : 4
    );

    float tCur = 0.0;
    ivec3 vCoord = ivec3(floor(vOrigin));
    int lastAxis = -1;

    // If starting outside grid, intersect grid AABB first.
    if (!inVoxelBounds(vCoord)) {
        vec3 boxMax = vec3(VOXEL_RES_XZ, VOXEL_RES_Y, VOXEL_RES_XZ);
        vec3 t0 = (vec3(0.0) - vOrigin) * invDir;
        vec3 t1 = (boxMax - vOrigin) * invDir;
        vec3 tMinV = min(t0, t1);
        vec3 tMaxV = max(t0, t1);
        float tEnter = maxOf(tMinV);
        float tExit  = minOf(tMaxV);
        if (tExit < max(tEnter, 0.0)) {
            result.exitedGrid = true;
            return result;
        }
        tCur = max(tEnter, 0.0) + 1e-4;
        vCoord = ivec3(floor(vOrigin + safeDir * tCur));
        if (!inVoxelBounds(vCoord)) {
            result.exitedGrid = true;
            return result;
        }
        lastAxis = (tMinV.x >= tMinV.y && tMinV.x >= tMinV.z) ? 0 : (tMinV.y >= tMinV.z ? 1 : 2);
    } else if (isOpaqueTrack1Voxel(vCoord)) {
        // Belt-and-suspenders structural guard: if numerical precision still places a ray origin
        // inside an opaque solid wall/ceiling cube at step 0, NEVER let it walk out the outer face
        // into outdoor sky — return an immediate blocked hit!
        result.hit = true;
        result.dist = 0.0;
        result.worldPos = worldOrigin;
        result.normal = -safeDir;
        result.albedo = vec3(0.0);
        result.skyLight = 0.0;
        return result;
    }

    for (int stepIdx = 0; stepIdx < maxSteps; ++stepIdx) {
        if (!inVoxelBounds(vCoord)) {
            result.exitedGrid = true;
            break;
        }

        // ---- Level 2: 16x16x16 Super-Brick Skip (1.1 KB, L1 resident) ----
        int shift = 0;
        if (texelFetch(superOccTex, vCoord >> 4, 0).r == 0u) {
            shift = 4;
        }
        // ---- Level 1: 4x4x4 Brick Skip (73 KB, L2 resident) ----
        else if (texelFetch(brickOccTex, vCoord >> 2, 0).r == 0u) {
            shift = 2;
        }
        // ---- Level 0: Exact Voxel Cell ----
        else {
            if (texelFetch(voxelGridTex, vCoord, 0).r != 0u) {
                uint slot = voxelLinearIndex(vCoord);
                SurfaceVoxel sv = readSurface(slot);

                // Compute voxel exit distance to bound primitive tests cleanly.
                vec3 voxelExitPlane = (vec3(vCoord + stepPos) - vOrigin) * invDir;
                float tVoxelExit = minOf(voxelExitPlane);
                float bestT = tVoxelExit + 1e-3;
                bool foundHit = false;

                // --- Track 1: O(1) Full-Cube 6-Face Fast Path ---
                if (lastAxis >= 0) {
                    uint faceIdx = uint(enterFace[lastAxis]);
                    uint faceDesc = sv.face[faceIdx];
                    // If this pure full-cube voxel (primHead == PRIM_END) had its entered face
                    // culled by Sodium shadow backface culling or internal grazing entry, fall back
                    // to the opposite face or any populated face so rays NEVER leak through solid walls!
                    if (faceDesc == 0u && sv.primHead == PRIM_END) {
                        faceDesc = sv.face[faceIdx ^ 1u];
                        if (faceDesc == 0u) {
                            for (int k = 0; k < 6; ++k) {
                                if (sv.face[k] != 0u) { faceDesc = sv.face[k]; break; }
                            }
                        }
                    }
                    if (faceDesc != 0u) {
                        vec3 localHit = (vOrigin + safeDir * tCur) - vec3(vCoord);
                        bool isTinted, isTranslucent;
                        float blockEmit;
                        vec2 atlasUV = decodeFaceAtlasUV(faceDesc, faceIdx, localHit, invAtlasSize, isTinted, isTranslucent, blockEmit);
                        vec4 tex = textureLod(colortex14, atlasUV, atlasLod);
                        if (tex.a >= 0.1) {
                            vec3 tintSrgb = isTinted ? unpackTint(sv.tintAndSky) : vec3(1.0);
                            if (passTranslucent && isTranslucent && blockEmit <= 0.0 && tex.a < 0.98) {
                                // Colored glass / ice / translucent block: filter light and keep tracing!
                                throughput *= evalGlassTransmittance(tex.rgb * tintSrgb, tex.a);
                            } else {
                                readHitMaterial(result, atlasUV, specTexSize, tex.rgb * tintSrgb, blockEmit,
                                                isTranslucent ? tex.a : 1.0);

                                bestT = tCur;
                                foundHit = true;
                                result.hit = true;
                                result.fromTrack2 = false;
                                result.dist = tCur;
                                result.worldPos = voxelToWorldPos(vOrigin + safeDir * tCur);
                                result.normal = faceIndexToNormal(faceIdx);
                                result.skyLight = unpackSky(sv.tintAndSky);
                            }
                        }
                    }
                }

                // --- Track 2: Exact Non-Cube Primitives (Lazy 24B Geo Unpack + Post-Loop Shading) ---
                uint primIdx = sv.primHead;
                if (primIdx != PRIM_END) {
                    vec3 localOrig = vOrigin - vec3(vCoord);
                    bool hitTrans = false;
                    float bestTransT = bestT;
                    vec3 transFilter = vec3(1.0);
                    bool foundT2 = false;
                    vec3 bestE1 = vec3(0.0);
                    vec3 bestE2 = vec3(0.0);
                    vec2 bestHitUV = vec2(0.0);
                    vec4 bestTex = vec4(0.0);
                    vec3 bestTintSrgb = vec3(1.0);
                    float bestPrimEmit = 0.0;
                    #ifdef TRACE_HIGH_PRIM_GUARD
                        const int maxPrims = 256;
                    #else
                        const int maxPrims = 48;
                    #endif
                    for (int guard = 0; guard < maxPrims && primIdx != PRIM_END; ++guard) {
                        uint curIdx = primIdx;
                        vec3 p0, e1, e2;
                        uint nextIdx;
                        unpackScenePrimGeo(curIdx, p0, e1, e2, nextIdx);
#ifdef TRACE_REFLECTION_QUADS
                        uint link = primPool.prims[curIdx].meta.z;
                        primIdx = link & PRIM_END;
                        if ((link & 0x80000000u) != 0u && guard + 1 < maxPrims) {
                            uint bounds = primPool.prims[curIdx].meta.w;
                            uvec3 lo = uvec3(bounds, bounds >> 7u, bounds >> 14u) & 127u;
                            uvec3 hi = uvec3(bounds >> 21u, (bounds >> 28u) | ((link >> 21u) << 4u), link >> 24u) & 127u;
                            // Test in grid space so the exact triangle path keeps its own local-origin precision.
                            // Expand the bounds for quantization and the barycentric edge tolerance.
                            vec3 t0 = ((vec3(vCoord) + vec3(lo) * (1.0 / 31.0) - 2.002) - vOrigin) * invDir;
                            vec3 t1 = ((vec3(vCoord) + vec3(hi) * (1.0 / 31.0) - 1.998) - vOrigin) * invDir;
                            float enter = maxOf(min(t0, t1)), leave = minOf(max(t0, t1));
                            if (leave < 0.0 || enter > bestT + 0.002 || enter > leave) {
                                ++guard; // The original list budget counts both triangles.
                                continue;
                            }
                            primIdx = nextIdx;
                        }
#else
                        primIdx = nextIdx;
#endif

                        vec3 pvec = cross(safeDir, e2);
                        float det = dot(e1, pvec);
                        if (abs(det) < 1e-7) continue;
                        float invDet = 1.0 / det;

                        vec3 tvec = localOrig - p0;
                        float u = dot(tvec, pvec) * invDet;
                        if (u < -1e-4 || u > 1.0001) continue;

                        vec3 qvec = cross(tvec, e1);
                        float v = dot(safeDir, qvec) * invDet;
                        if (v < -1e-4 || u + v > 1.0001) continue;

                        float t = dot(e2, qvec) * invDet;
                        if (t <= 1e-4 || t >= bestT) continue;

                        // Defer UV/tint/material decode until ray actually intersects this triangle!
                        vec2 uv0, uv1, uv2;
                        vec3 tintSrgb;
                        float primEmit;
                        bool primTrans, primWater;
                        unpackScenePrimShade(curIdx, uv0, uv1, uv2, tintSrgb, primEmit, primTrans, primWater);

                        vec2 hitUV = uv0 * (1.0 - u - v) + uv1 * u + uv2 * v;
                        vec4 tex = primWater ? vec4(1.0) : textureLod(colortex14, hitUV, atlasLod);
                        if (!primWater && tex.a < 0.1) continue;

                        if (passTranslucent && (primWater || (primTrans && primEmit <= 0.0 && tex.a < 0.98))) {
                            // Secondary rays store interfaces only. Approximate one block of water per crossing.
                            if (t < bestTransT) {
                                bestTransT = t;
                                hitTrans = true;
                                transFilter = primWater ? exp(-waterExtinction(tintSrgb)) : evalGlassTransmittance(tex.rgb * tintSrgb, tex.a);
                            }
                            continue;
                        }

                        bestT = t;
                        foundT2 = true;
                        bestE1 = e1;
                        bestE2 = e2;
                        bestHitUV = hitUV;
                        bestTex = vec4(tex.rgb, primTrans ? tex.a : 1.0);
                        bestTintSrgb = tintSrgb;
                        bestPrimEmit = primEmit;
                    }
                    if (foundT2) {
                        vec3 rawCross = cross(bestE1, bestE2);
                        float triArea = 0.5 * length(rawCross);
                        vec3 triN = rawCross / max(2.0 * triArea, 1e-6);
                        if (dot(triN, safeDir) > 0.0) triN = -triN;

                        readHitMaterial(result, bestHitUV, specTexSize, bestTex.rgb * bestTintSrgb, bestPrimEmit, bestTex.a);

                        foundHit = true;
                        result.hit = true;
                        result.fromTrack2 = true;
                        result.dist = bestT;
                        result.worldPos = voxelToWorldPos(vOrigin + safeDir * bestT);
                        result.normal = triN;
                        result.skyLight = unpackSky(sv.tintAndSky);
                    }
                    if (hitTrans && bestTransT < bestT) {
                        throughput *= transFilter;
                    }
                }

                if (foundHit) {
                    result.voxel = vCoord;
                    result.throughput = throughput;
                    return result;
                }
                if (maxOf(throughput) < 0.015) {
                    result.throughput = vec3(0.0);
                    return result;
                }
            }
        }

        // Advance cleanly to next node boundary at hierarchy level `shift`.
        int nodeSize = 1 << shift;
        ivec3 nodeMin = (vCoord >> shift) << shift;
        ivec3 nodeExit = nodeMin + stepPos * nodeSize;
        vec3 tPlane = (vec3(nodeExit) - vOrigin) * invDir;

        float tNext;
        if (tPlane.x <= tPlane.y && tPlane.x <= tPlane.z) {
            tNext = tPlane.x;
            lastAxis = 0;
        } else if (tPlane.y <= tPlane.z) {
            tNext = tPlane.y;
            lastAxis = 1;
        } else {
            tNext = tPlane.z;
            lastAxis = 2;
        }

        tCur = max(tCur, tNext);
        vec3 nextPos = vOrigin + safeDir * tCur;
        // Clamp non-stepping axes inside the exited node [nodeMin, nodeMin + nodeSize - 1].
        // At shift == 0 (1x1x1 voxel), this enforces strict 6-connected DDA stepping so rays
        // aiming at 90-degree concave corners NEVER jump diagonally past the wall/ceiling seam!
        ivec3 nextCoord = clamp(ivec3(floor(nextPos)), nodeMin, nodeMin + ivec3(nodeSize - 1));
        nextCoord[lastAxis] = nodeExit[lastAxis] + (stepSign[lastAxis] < 0 ? -1 : 0);
        vCoord = nextCoord;
    }

    result.throughput = throughput;
    if (!inVoxelBounds(vCoord)) {
        result.exitedGrid = true;
    } else {
        result.worldPos = voxelToWorldPos(vOrigin + safeDir * tCur);
        result.dist = tCur;
    }
    return result;
}

TraceHit traceScene(vec3 worldOrigin, vec3 rayDir, int maxSteps, float atlasLod) {
    return traceScene(worldOrigin, rayDir, maxSteps, atlasLod, true);
}

// Fast 6-connected hierarchical shadow/visibility ray with stained-glass colored transmission.
// Returns vec3(0.0) if blocked by opaque geometry before reaching targetVoxel, or accumulated
// colored glass transmittance if visible.
vec3 traceVisibility(vec3 worldOrigin, vec3 targetVoxelPos, ivec3 targetVoxel, int maxSteps) {
    vec3 vOrigin = worldToVoxelPos(worldOrigin);
    ivec3 vCoord = ivec3(floor(vOrigin));
    if (all(equal(vCoord, targetVoxel))) return vec3(1.0);

    vec3 toTarget = targetVoxelPos - vOrigin;
    float targetDist = length(toTarget);
    if (targetDist < 0.05) return vec3(1.0);
    vec3 rayDir = toTarget / targetDist;

    vec3 safeDir = vec3(
        abs(rayDir.x) < 1e-6 ? (rayDir.x < 0.0 ? -1e-6 : 1e-6) : rayDir.x,
        abs(rayDir.y) < 1e-6 ? (rayDir.y < 0.0 ? -1e-6 : 1e-6) : rayDir.y,
        abs(rayDir.z) < 1e-6 ? (rayDir.z < 0.0 ? -1e-6 : 1e-6) : rayDir.z
    );
    vec3 invDir = 1.0 / safeDir;
    ivec3 stepSign = ivec3(greaterThan(safeDir, vec3(0.0))) * 2 - 1;
    ivec3 stepPos  = ivec3(greaterThan(safeDir, vec3(0.0)));
    ivec2 atlasTexSize = max(textureSize(colortex14, 0), ivec2(1));
    vec2  invAtlasSize = 1.0 / vec2(atlasTexSize);
    ivec3 enterFace = ivec3(
        stepSign.x > 0 ? 1 : 0,
        stepSign.y > 0 ? 3 : 2,
        stepSign.z > 0 ? 5 : 4
    );

    float tCur = 0.0;
    int lastAxis = -1;
    vec3 throughput = vec3(1.0);

    for (int stepIdx = 0; stepIdx < maxSteps; ++stepIdx) {
        if (!inVoxelBounds(vCoord) || tCur >= targetDist) return throughput;
        if (all(equal(vCoord, targetVoxel))) return throughput;

        int shift = 0;
        if (texelFetch(superOccTex, vCoord >> 4, 0).r == 0u) {
            shift = 4;
        } else if (texelFetch(brickOccTex, vCoord >> 2, 0).r == 0u) {
            shift = 2;
        } else if (texelFetch(voxelGridTex, vCoord, 0).r != 0u) {
            uint slot = voxelLinearIndex(vCoord);
            SurfaceVoxel sv = readSurface(slot);

            if (lastAxis >= 0) {
                uint faceIdx = uint(enterFace[lastAxis]);
                uint faceDesc = sv.face[faceIdx];
                if (faceDesc == 0u && sv.primHead == PRIM_END) {
                    faceDesc = sv.face[faceIdx ^ 1u];
                    if (faceDesc == 0u) {
                        for (int k = 0; k < 6; ++k) {
                            if (sv.face[k] != 0u) { faceDesc = sv.face[k]; break; }
                        }
                    }
                }
                if (faceDesc != 0u) {
                    vec3 localHit = (vOrigin + safeDir * tCur) - vec3(vCoord);
                    bool isTinted, isTranslucent;
                    float blockEmit;
                    vec2 atlasUV = decodeFaceAtlasUV(faceDesc, faceIdx, localHit, invAtlasSize, isTinted, isTranslucent, blockEmit);
                    vec4 tex = textureLod(colortex14, atlasUV, 1.0);
                    if (tex.a >= 0.1) {
                        if (isTranslucent && blockEmit <= 0.0 && tex.a < 0.98) {
                            vec3 tintSrgb = isTinted ? unpackTint(sv.tintAndSky) : vec3(1.0);
                            throughput *= evalGlassTransmittance(tex.rgb * tintSrgb, tex.a);
                            if (maxOf(throughput) < 0.015) return vec3(0.0);
                        } else {
                            return vec3(0.0);
                        }
                    }
                }
            }

            uint primIdx = sv.primHead;
            if (primIdx != PRIM_END) {
                vec3 voxelExitPlane = (vec3(vCoord + stepPos) - vOrigin) * invDir;
                float tLimit = min(minOf(voxelExitPlane) + 1e-3, targetDist);
                float bestTransT = tLimit;
                vec3 localOrig = vOrigin - vec3(vCoord);
                bool hitTrans = false;
                vec3 transFilter = vec3(1.0);
                for (int guard = 0; guard < 32 && primIdx != PRIM_END; ++guard) {
                    uint curIdx = primIdx;
                    vec3 p0, e1, e2;
                    uint nextIdx;
                    unpackScenePrimGeo(curIdx, p0, e1, e2, nextIdx);
                    primIdx = nextIdx;

                    vec3 pvec = cross(safeDir, e2);
                    float det = dot(e1, pvec);
                    if (abs(det) < 1e-7) continue;
                    float invDet = 1.0 / det;

                    vec3 tvec = localOrig - p0;
                    float u = dot(tvec, pvec) * invDet;
                    if (u < -1e-4 || u > 1.0001) continue;

                    vec3 qvec = cross(tvec, e1);
                    float v = dot(safeDir, qvec) * invDet;
                    if (v < -1e-4 || u + v > 1.0001) continue;

                    float t = dot(e2, qvec) * invDet;
                    if (t <= 1e-4 || t >= tLimit) continue;

                    vec2 uv0, uv1, uv2;
                    vec3 tintSrgb;
                    float primEmit;
                    bool primTrans, primWater;
                    unpackScenePrimShade(curIdx, uv0, uv1, uv2, tintSrgb, primEmit, primTrans, primWater);

                    vec2 hitUV = uv0 * (1.0 - u - v) + uv1 * u + uv2 * v;
                    vec4 tex = primWater ? vec4(1.0) : textureLod(colortex14, hitUV, 1.0);
                    if (!primWater && tex.a < 0.1) continue;

                    if (primWater || (primTrans && primEmit <= 0.0 && tex.a < 0.98)) {
                        if (t < bestTransT) {
                            hitTrans = true;
                            bestTransT = t;
                            transFilter = primWater ? exp(-waterExtinction(tintSrgb)) : evalGlassTransmittance(tex.rgb * tintSrgb, tex.a);
                        }
                    } else {
                        return vec3(0.0);
                    }
                }
                if (hitTrans) {
                    throughput *= transFilter;
                    if (maxOf(throughput) < 0.015) return vec3(0.0);
                }
            }
        }

        int nodeSize = 1 << shift;
        ivec3 nodeMin = (vCoord >> shift) << shift;
        ivec3 nodeExit = nodeMin + stepPos * nodeSize;
        vec3 tPlane = (vec3(nodeExit) - vOrigin) * invDir;

        float tNext;
        if (tPlane.x <= tPlane.y && tPlane.x <= tPlane.z) {
            tNext = tPlane.x;
            lastAxis = 0;
        } else if (tPlane.y <= tPlane.z) {
            tNext = tPlane.y;
            lastAxis = 1;
        } else {
            tNext = tPlane.z;
            lastAxis = 2;
        }

        tCur = max(tCur, tNext);
        if (tCur >= targetDist) return throughput;
        vec3 nextPos = vOrigin + safeDir * tCur;
        ivec3 nextCoord = clamp(ivec3(floor(nextPos)), nodeMin, nodeMin + ivec3(nodeSize - 1));
        nextCoord[lastAxis] = nodeExit[lastAxis] + (stepSign[lastAxis] < 0 ? -1 : 0);
        vCoord = nextCoord;
    }

    return throughput;
}

// 1-SPP Local-Grid RIS (Weighted Reservoir Sampling) Next-Event Estimation:
// Queries ONLY the 2x2x2 local 32^3 macro-cells covering [vPos-16, vPos+16] (100% exact for Minecraft's
// 16-block light radius!), eliminating global stepF = 150x fireflies in giant multi-chandelier halls!
vec3 sampleDirectLightsNEE(vec3 worldPos, vec3 normal, bool isSphere, int maxCandidates, vec2 bn, out vec3 outLightDir) {
    outLightDir = vec3(0.0, 1.0, 0.0);
    if (lightList.lightCount == 0u) return vec3(0.0);

    vec3 vPos = worldToVoxelPos(worldPos);
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
    float startF = bn.x * stepF;

    float wSum = 0.0;
    float selW = 0.0;
    vec3 selContrib = vec3(0.0);
    vec3 selDir = vec3(0.0, 1.0, 0.0);
    vec3 selTargetVoxPos = vec3(0.0);

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

        SceneLight L = lightList.lights[idx];
        vec3 lWorld = voxelToWorldPos(L.posAndPower.xyz);
        vec3 toL = lWorld - worldPos;
        float dist2 = dot(toL, toL);
        float emitStr = L.posAndPower.w;
        float maxRange = sceneLightRange(emitStr);
        if (dist2 >= maxRange * maxRange) continue;

        float dist = sqrt(max(dist2, 1e-6));
        vec3 lDir = toL / dist;
        // Analytical spherical average of max(N.L, 0) over S^2 is exactly 0.25.
        float geom = isSphere ? 0.25 : max(dot(normal, lDir), 0.0);
        if (geom <= 1e-4) continue;

        float rRatio = dist / maxRange;
        float r2 = rRatio * rRatio;
        float win = saturate(1.0 - r2 * r2);
        win *= win;
        float atten = win / (dist2 + 0.35);

        // Projected-area calibration for the non-cube and full-cube light proxies.
        float neePower = (L.colorAndMeta.a < 0.3) ? 1.55 : 1.35;
        vec3 contrib = L.colorAndMeta.rgb * (emitStr * neePower * atten * geom);
        float w = luma(contrib);
        if (w <= 1e-6) continue;

        wSum += w;
        float uPick = fract(bn.y + 0.6180339887 * float(i));
        if (uPick * wSum <= w) {
            selW = w;
            selContrib = contrib;
            selDir = lDir;
            selTargetVoxPos = L.posAndPower.xyz;
        }
    }

    if (wSum <= 1e-6 || selW <= 1e-6) return vec3(0.0);

    int visSteps = isSphere ? 36 : 24;
    vec3 vis = traceVisibility(worldPos, selTargetVoxPos, ivec3(floor(selTargetVoxPos)), visSteps);
    if (maxOf(vis) <= 1e-4) return vec3(0.0);

    outLightDir = selDir;
    return selContrib * ((stepF * wSum / selW) * vis);
}

#endif
