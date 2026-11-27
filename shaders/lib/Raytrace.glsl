#ifndef LIB_RAYTRACE_GLSL
#define LIB_RAYTRACE_GLSL

#include "/lib/EmitterGeometry.glsl"

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
    #ifdef ENTITY_REFLECTIONS_ENABLED
        float heldEmission;
    #endif
    #ifdef ENTITY_REFLECTION_HISTORY
        uint entityPose;
    #endif
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
    #ifdef ENTITY_REFLECTIONS_ENABLED
        h.heldEmission = 0.0;
    #endif
    #ifdef ENTITY_REFLECTION_HISTORY
        h.entityPose = 0u;
    #endif
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

float localLightWindow(float dist2, float strength) {
    float reach = clamp(strength * 16.0, 4.0, 16.0);
    float r2 = dist2 / (reach * reach);
    float window = max(1.0 - r2 * r2, 0.0);
    return window * window;
}

float emitterHorizonExtent(vec3 normal) {
    return 0.5 * dot(abs(normal), vec3(1.0));
}

float sampledEmitterShare(ivec3 voxel, vec3 receiver, vec3 normal, out uvec2 shape) {
    shape = uvec2(0u);
    if (!inVoxelBounds(voxel) || !inVoxelBounds(ivec3(floor(receiver)))) return 0.0;
    if (dot(vec3(voxel) + 0.5 - receiver, normal) + emitterHorizonExtent(normal) <= 0.0) return 0.0;

    uint slot = surfacePool.surfaces[voxelLinearIndex(voxel)].primHead >> 21u;
    if (slot == 0u || slot > LIGHT_CELL_CAP) return 0.0;
    ivec3 cell = voxel / 32;
    uint id = uint(cell.x + LIGHT_CELL_XZ * (cell.z + LIGHT_CELL_XZ * cell.y));
    uint index = lightList.cellLights[id * LIGHT_CELL_CAP + slot - 1u];
    shape = emitterGeometry.shapes[index];
    if ((shape.x & EMITTER_CUBE) != 0u) {
        shape.x = EMITTER_CUBE | frontEmitterFaces(shape.x, receiver - vec3(voxel));
    }
    if ((shape.x & ~EMITTER_CUBE) == 0u) return 0.0;

    vec4 lamp = lightList.lights[index].posAndPower;
    vec3 delta = lamp.xyz - receiver;
    return localLightWindow(dot(delta, delta), lamp.w);
}

bool directLightCovers(ivec3 voxel, vec3 receiver, vec3 normal) {
    uvec2 shape;
    return sampledEmitterShare(voxel, worldToVoxelPos(receiver), normal, shape) > 0.0;
}

float coveredHitEmission(TraceHit hit, vec3 receiver) {
    float emission = surfaceEmission(hit.pbrEmission, hit.blockEmission * hit.coverage, hit.f0, hit.albedo);
    #if EMITTER_RANGE > 0
        if (emission > 0.0 && beyondEmitterRange(hit.dist)) {
            vec3 faceNormal = hit.fromTrack2 ? vec3(0.0) : hit.normal;
            if (!emitterCoversCell(hit.voxel, faceNormal, worldToVoxelPos(receiver))) return 0.0;
        }
    #endif
    return emission;
}

vec3 indirectHitEmission(TraceHit hit, vec3 receiver, vec3 normal) {
    float emission = coveredHitEmission(hit, receiver);
    if (emission <= 0.0) return vec3(0.0);

    if (hit.blockEmission > 0.0) {
        uvec2 shape;
        float share = sampledEmitterShare(hit.voxel, worldToVoxelPos(receiver), normal, shape);
        if (share > 0.0) {
            bool cube = (shape.x & EMITTER_CUBE) != 0u;
            if (hit.fromTrack2 && !cube) {
                // The light table covers the primitive prefix searched by diffuse rays.
                vec3 local = worldToVoxelPos(hit.worldPos) - vec3(hit.voxel);
                if (inEmitterCell(local, hit.normal)) emission *= 1.0 - share;
            } else if (!hit.fromTrack2 && cube && (shape.x & (1u << normalToFaceIndex(hit.normal))) != 0u) {
                emission *= 1.0 - share;
            }
        }
    }
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
#ifdef ENTITY_REFLECTIONS_ENABLED
bool entitySuperOccupied(ivec3 voxel) {
    uint node = entityNodeIndex(voxel >> 4, SUPER_RES_XZ);
    return (entityPool.superBits[node >> 5u] & (1u << (node & 31u))) != 0u;
}

bool entityBrickOccupied(ivec3 voxel) {
    uint node = entityNodeIndex(voxel >> 2, BRICK_RES_XZ);
    return (entityPool.brickBits[node >> 5u] & (1u << (node & 31u))) != 0u;
}

vec4 sampleEntityPatch(uint texturePatch, vec2 uv) {
    ivec2 size = ivec2(uvec2(texturePatch >> 20u, texturePatch >> 26u) & 63u) + 1;
    ivec2 texel = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
    uint offset = texturePatch & 0xfffffu;
    return unpackUnorm4x8(entityPool.texels[offset + uint(texel.x + texel.y * size.x)]);
}
#endif

TraceHit traceScene(vec3 worldOrigin, vec3 rayDir, int maxSteps, float atlasLod, bool passTranslucent, bool includeEntities) {
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
        if (texelFetch(superOccTex, vCoord >> 4, 0).r == 0u
            #ifdef ENTITY_REFLECTIONS_ENABLED
                && !(includeEntities && entitySuperOccupied(vCoord))
            #endif
        ) {
            shift = 4;
        }
        // ---- Level 1: 4x4x4 Brick Skip (73 KB, L2 resident) ----
        else if (texelFetch(brickOccTex, vCoord >> 2, 0).r == 0u
            #ifdef ENTITY_REFLECTIONS_ENABLED
                && !(includeEntities && entityBrickOccupied(vCoord))
            #endif
        ) {
            shift = 2;
        }
        // ---- Level 0: Exact Voxel Cell ----
        else {
            uint slot = voxelLinearIndex(vCoord);
            #ifdef ENTITY_REFLECTIONS_ENABLED
                uint entityHead = includeEntities ? entityPool.heads[slot] : PRIM_END;
            #else
                const uint entityHead = PRIM_END;
            #endif
            if (texelFetch(voxelGridTex, vCoord, 0).r != 0u || entityHead != PRIM_END) {
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
                uint nextEntity = foundHit ? PRIM_END : entityHead;
                if (primIdx != PRIM_END || nextEntity != PRIM_END) {
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
                    #ifdef ENTITY_REFLECTIONS_ENABLED
                        uint bestEntityIdx = PRIM_END;
                    #endif
                    #ifdef TRACE_HIGH_PRIM_GUARD
                        const int maxPrims = 256;
                    #else
                        const int maxPrims = 48;
                    #endif
                    // Each list has its own traversal budget.
                    for (int guard = 0; ; ++guard) {
                        if (guard >= maxPrims || primIdx == PRIM_END) {
                            if (nextEntity == PRIM_END) break;
                            primIdx = nextEntity;
                            nextEntity = PRIM_END;
                            guard = 0;
                        }
                        uint curIdx = primIdx;
                        #ifdef ENTITY_REFLECTIONS_ENABLED
                            bool entity = curIdx >= MAX_PRIMITIVES;
                        #endif
                        vec3 p0, e1, e2;
                        uint nextIdx;
                        unpackScenePrimGeo(curIdx, p0, e1, e2, nextIdx);
                        primIdx = nextIdx;
#ifdef TRACE_REFLECTION_QUADS
                        #ifdef ENTITY_REFLECTIONS_ENABLED
                            if (!entity)
                        #endif
                        {
                            uint link = primPool.prims[curIdx].meta.z;
                            primIdx = link & PRIM_END;
                            if ((link & 0x80000000u) != 0u && guard + 1 < maxPrims) {
                                uint bounds = primPool.prims[curIdx].meta.w;
                                uvec3 lo = uvec3(bounds, bounds >> 7u, bounds >> 14u) & 127u;
                                uvec3 hi = uvec3(bounds >> 21u, (bounds >> 28u) | ((link >> 21u) << 4u), link >> 24u) & 127u;
                                // Test in grid space, expanding for quantization and barycentric edge tolerance.
                                vec3 t0 = ((vec3(vCoord) + vec3(lo) * (1.0 / 31.0) - 2.002) - vOrigin) * invDir;
                                vec3 t1 = ((vec3(vCoord) + vec3(hi) * (1.0 / 31.0) - 1.998) - vOrigin) * invDir;
                                float enter = maxOf(min(t0, t1)), leave = minOf(max(t0, t1));
                                if (leave < 0.0 || enter > bestT + 0.002 || enter > leave) {
                                    ++guard; // The original list budget counts both triangles.
                                    continue;
                                }
                                primIdx = nextIdx;
                            }
                        }
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
                        #ifdef ENTITY_REFLECTIONS_ENABLED
                            if (entity && t < tCur - 1e-3) continue;
                        #endif

                        // Defer UV/tint/material decode until ray actually intersects this triangle!
                        vec2 uv0, uv1, uv2;
                        vec3 tintSrgb;
                        float primEmit;
                        bool primTrans, primWater;
                        unpackScenePrimShade(curIdx, uv0, uv1, uv2, tintSrgb, primEmit, primTrans, primWater);

                        vec2 hitUV = uv0 * (1.0 - u - v) + uv1 * u + uv2 * v;
                        vec4 tex;
                        float alphaCutoff = 0.1;
                        #ifdef ENTITY_REFLECTIONS_ENABLED
                            if (entity) {
                                tex = sampleEntityPatch(primPool.prims[curIdx].meta.w, hitUV);
                                alphaCutoff = 0.5 / 255.0;
                            } else
                        #endif
                        tex = primWater ? vec4(1.0) : textureLod(colortex14, hitUV, atlasLod);
                        if (!primWater && tex.a < alphaCutoff) continue;

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
                        #ifdef ENTITY_REFLECTIONS_ENABLED
                            bestEntityIdx = entity ? curIdx : PRIM_END;
                            if (entity) bestTex.a = tex.a;
                        #endif
                    }
                    if (foundT2) {
                        vec3 rawCross = cross(bestE1, bestE2);
                        float triArea = 0.5 * length(rawCross);
                        vec3 triN = rawCross / max(2.0 * triArea, 1e-6);
                        if (dot(triN, safeDir) > 0.0) triN = -triN;

                        #ifdef ENTITY_REFLECTIONS_ENABLED
                            if (bestEntityIdx != PRIM_END) {
                                result.albedo = srgbToLinear(bestTex.rgb * bestTintSrgb);
                                result.skyLight = bestPrimEmit; // Entity primitives store sky light in this field.
                                result.heldEmission = (round(bestTex.a * 255.0) - 1.0) / 254.0;
                                #ifdef ENTITY_REFLECTION_HISTORY
                                    result.entityPose = primPool.prims[bestEntityIdx].meta.z;
                                #endif
                            } else
                        #endif
                        {
                            readHitMaterial(result, bestHitUV, specTexSize, bestTex.rgb * bestTintSrgb, bestPrimEmit, bestTex.a);
                            result.skyLight = unpackSky(sv.tintAndSky);
                        }

                        foundHit = true;
                        result.hit = true;
                        result.fromTrack2 = true;
                        result.dist = bestT;
                        result.worldPos = voxelToWorldPos(vOrigin + safeDir * bestT);
                        result.normal = triN;
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

TraceHit traceScene(vec3 worldOrigin, vec3 rayDir, int maxSteps, float atlasLod, bool passTranslucent) {
    return traceScene(worldOrigin, rayDir, maxSteps, atlasLod, passTranslucent, false);
}

TraceHit traceScene(vec3 worldOrigin, vec3 rayDir, int maxSteps, float atlasLod) {
    return traceScene(worldOrigin, rayDir, maxSteps, atlasLod, true);
}

// Fast 6-connected hierarchical shadow/visibility ray with stained-glass colored transmission.
// Trace the open segment to the sampled emitter surface, including blockers inside its voxel.
vec3 traceVisibility(vec3 vOrigin, vec3 rayDir, float distanceToLight) {
    ivec3 vCoord = ivec3(floor(vOrigin));
    // Exclude the sampled surface itself from the intersection tests at the ray endpoint.
    float targetDist = distanceToLight - max(2e-5, distanceToLight * 1.2e-6);

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

    // Each iteration crosses at least one plane. A local light's proxy is at
    // most 16 blocks away, and its sampled surface lies in the proxy's cell.
    const int maxSteps = 34;
    for (int stepIdx = 0; stepIdx < maxSteps; ++stepIdx) {
        if (!inVoxelBounds(vCoord)) return vec3(0.0);
        if (tCur >= targetDist) return throughput;

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
                    vec3 localHit = (vOrigin + rayDir * tCur) - vec3(vCoord);
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
                for (int guard = 0; guard < 48 && primIdx != PRIM_END; ++guard) {
                    uint curIdx = primIdx;
                    vec3 p0, e1, e2;
                    uint nextIdx;
                    unpackScenePrimGeo(curIdx, p0, e1, e2, nextIdx);
                    primIdx = nextIdx;

                    vec3 pvec = cross(rayDir, e2);
                    float det = dot(e1, pvec);
                    if (abs(det) < 1e-7) continue;
                    float invDet = 1.0 / det;

                    vec3 tvec = localOrig - p0;
                    float u = dot(tvec, pvec) * invDet;
                    if (u < -1e-4 || u > 1.0001) continue;

                    vec3 qvec = cross(tvec, e1);
                    float v = dot(rayDir, qvec) * invDet;
                    if (v < -1e-4 || u + v > 1.0001) continue;

                    float t = dot(e2, qvec) * invDet;
                    if (t <= 1e-4 || t < tCur - 1e-3 || t >= tLimit) continue;

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
        vec3 nextPos = vOrigin + rayDir * tCur;
        ivec3 nextCoord = clamp(ivec3(floor(nextPos)), nodeMin, nodeMin + ivec3(nodeSize - 1));
        nextCoord[lastAxis] = nodeExit[lastAxis] + (stepSign[lastAxis] < 0 ? -1 : 0);
        vCoord = nextCoord;
    }

    return vec3(0.0);
}

#endif
