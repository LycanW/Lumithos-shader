#ifndef LIB_SCREEN_TRACE_GLSL
#define LIB_SCREEN_TRACE_GLSL

#include "/lib/SVGF.glsl"

// Reconstruct linear view-space depth (-viewPos.z > 0) from NDC depth [0, 1].
float linearizeDepth(float depth) {
    float ndcZ = depth * 2.0 - 1.0;
    vec4 v = gbufferProjectionInverse * vec4(0.0, 0.0, ndcZ, 1.0);
    return -v.z / max(v.w, 1e-6);
}

// Linear-Z verified screen-space raytrace (SSRT for diffuse, SSR for specular).
// Uses view-space linear thickness (never non-linear depth slabs) and reprojects
// hit coordinates into previous frame UV so colortex5 stays rock-solid during motion.
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

    for (int i = 0; i < steps; ++i) {
        vec3 p = rayStart + rayStep * (float(i) + dither + 0.5);
        if (any(lessThan(p.xy, vec2(0.002))) || any(greaterThanEqual(p.xy, vec2(0.998))) || p.z <= 0.0 || p.z >= 1.0) {
            return false;
        }
        ivec2 px = ivec2(p.xy * res);
        float sceneDepth = texelFetch(depthtex1, px, 0).r;
        if (sceneDepth >= 0.999999) continue;

        float rayLinZ   = linearizeDepth(p.z);
        float sceneLinZ = linearizeDepth(sceneDepth);
        float diffZ     = rayLinZ - sceneLinZ;
        float thickness = 0.18 + 0.035 * sceneLinZ;

        if (diffZ > 0.02 && diffZ < thickness) {
            GBufferMat hitMat = readGBuffer(px);
            if (dot(hitMat.geoNormal, worldDir) < 0.0) {
                vec3 hitView = screenToView(vec3(p.xy, sceneDepth));
                float d = distance(hitView, viewOrigin);
                if (d >= minWorldDist) {
                    vec3 hitWorld = viewToWorld(hitView);
                    vec2 puv = reprojectScreenUV(p.xy, hitWorld);
                    if (all(greaterThanEqual(puv, vec2(0.002))) && all(lessThan(puv, vec2(0.998)))) {
                        // Validate against previous camera's prevGuideTex so disoccluded walls
                        // never read stale foreground occluders from colortex5!
                        ivec2 prevPx = clamp(ivec2(puv * res), ivec2(0), ivec2(res) - 1);
                        vec4 pg = texelFetch(prevGuideTex, prevPx, 0);
                        float expectedPrevZ = -(gbufferPreviousModelView * vec4(hitWorld + cameraVelocity, 1.0)).z;
                        if (pg.w > 0.0 && abs(expectedPrevZ - pg.w) < max(0.15, 0.08 * max(expectedPrevZ, 0.1))) {
                            vec3 pGeoN = unpackNormal16(uint(round(pg.x)));
                            if (dot(hitMat.geoNormal, pGeoN) > 0.85) {
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
    return false;
}

#endif
