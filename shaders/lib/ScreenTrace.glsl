#ifndef LIB_SCREEN_TRACE_GLSL
#define LIB_SCREEN_TRACE_GLSL

#include "/lib/SVGF.glsl"

// Zero-thickness 3D plane-exact screen-space raytrace (SSR / SSRT) with seamless PT fallback:
// - Rejects first-person hand pixels in depthtex1.
// - Triggers binary refinement ONLY on genuine front-to-back depth crossings within the current step
//   (no artificial depth-slab thickness -> rays passing behind foreground walls continue cleanly!).
// - Validates the 6-step refined hit directly against the pixel's 3D geometric surface plane
//   (abs(dot(bRayView - bHitView, hitViewGeoN)) <= subpixel tolerance) + front-facing normal + strict collinearity;
//   any miss falls back immediately to exact 3D voxel path-traced reflection!
bool screenSpaceTrace(vec3 viewOrigin, vec3 worldDir, int steps, float maxScreenLen, float dither, float minWorldDist, out vec2 prevHitUV, out float hitDist) {
    prevHitUV = vec2(-1.0);
    hitDist = 0.0;

    vec3 viewDir = normalize(mat3(gbufferModelView) * worldDir);
    if (viewDir.z > 0.92) return false; // pointing back toward camera

    vec3 rayStart = viewToScreen(viewOrigin);
    float maxViewT = viewDir.z > 0.0 ? min(48.0, -(viewOrigin.z + near * 1.2) / max(viewDir.z, 1e-4)) : 48.0;
    if (maxViewT <= minWorldDist) return false;

    vec3 rayEnd = viewToScreen(viewOrigin + viewDir * maxViewT);
    vec3 rayDelta = rayEnd - rayStart;

    vec2 screenLen = abs(rayDelta.xy);
    float maxComp = max(screenLen.x, screenLen.y);
    if (maxComp < 0.002) return false;
    if (maxComp > maxScreenLen) {
        rayDelta *= maxScreenLen / maxComp;
    }

    vec3 rayStep = rayDelta / float(steps);
    vec2 res = screenSize();
    ivec2 maxPx = ivec2(res) - 1;

    vec3 prevP = rayStart;
    float prevRayZ = -viewOrigin.z;

    for (int i = 0; i < steps; ++i) {
        vec3 p = rayStart + rayStep * (float(i) + dither + 0.5);
        if (any(lessThan(p.xy, vec2(0.002))) || any(greaterThanEqual(p.xy, vec2(0.998))) || p.z <= 0.0 || p.z >= 1.0) {
            return false;
        }
        ivec2 px = clamp(ivec2(p.xy * res), ivec2(0), maxPx);
        float sceneDepth = texelFetch(depthtex1, px, 0).r;
        float rayLinZ = linearizeDepth(p.z);

        // Fast monotonic NDC early-out before any crossing checks.
        if (sceneDepth >= 0.999999 || p.z <= sceneDepth) {
            prevP = p;
            prevRayZ = rayLinZ;
            continue;
        }

        // Never let the first-person hand or held item act as a world reflector/occluder.
        uint w1 = texelFetch(colortex1, px, 0).g;
        if (((w1 >> 24u) & 1u) != 0u) {
            prevP = p;
            prevRayZ = rayLinZ;
            continue;
        }

        float sceneLinZ = linearizeDepth(sceneDepth);
        float stepDZ = rayLinZ - prevRayZ;
        float diffZ  = rayLinZ - sceneLinZ;
        // Step-bracket crossing test only (zero artificial surface thickness!).
        float stepBracket = max(0.10, abs(stepDZ) * 1.15);

        if (prevRayZ <= sceneLinZ + 0.03 && diffZ > 0.0 && diffZ < stepBracket) {
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
            uint bw1 = texelFetch(colortex1, bpx, 0).g;
            if (bd < 0.999999 && (((bw1 >> 24u) & 1u) == 0u)) {
                vec3 hitGeoN, hitShadeN;
                readGBufferNormals(bpx, hitGeoN, hitShadeN);
                if (dot(hitGeoN, worldDir) < -0.02) {
                    vec3 bRayView = screenToView(b);
                    vec3 bHitView = screenToView(vec3(b.xy, bd));
                    vec3 hitViewGeoN = normalize(mat3(gbufferModelView) * hitGeoN);
                    float planeDist = abs(dot(bRayView - bHitView, hitViewGeoN));
                    float planeTol  = max(0.025, 0.006 * (-bHitView.z));

                    // Exact zero-thickness 3D surface plane intersection + strict ray collinearity!
                    if (planeDist <= planeTol) {
                        vec3 toHit = bHitView - viewOrigin;
                        float d = length(toHit);
                        if (d >= minWorldDist && dot(toHit / max(d, 1e-5), viewDir) > 0.985) {
                            vec3 hitWorld = viewToWorld(bHitView);
                            vec2 puv = reprojectScreenUV(b.xy, hitWorld);
                            if (all(greaterThanEqual(puv, vec2(0.002))) && all(lessThan(puv, vec2(0.998)))) {
                                ivec2 prevPx = clamp(ivec2(puv * res), ivec2(0), maxPx);
                                vec4 pg = texelFetch(prevGuideTex, prevPx, 0);
                                float expectedPrevZ = -(gbufferPreviousModelView * vec4(hitWorld + cameraVelocity, 1.0)).z;
                                if (pg.w > 0.0 && abs(expectedPrevZ - pg.w) < max(0.15, 0.08 * max(expectedPrevZ, 0.1))) {
                                    vec3 pGeoN = unpackNormal16(uint(round(pg.x)));
                                    if (dot(hitGeoN, pGeoN) > 0.85) {
                                        prevHitUV = puv;
                                        hitDist = d;
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        prevP = p;
        prevRayZ = rayLinZ;
    }
    return false;
}

#endif
