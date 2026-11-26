#ifndef LIB_REPROJECTION_GLSL
#define LIB_REPROJECTION_GLSL

#include "/lib/Resources.glsl"

// Reproject current world position to previous unjittered screen UV.
vec2 reprojectWorldToPrevUV(vec3 worldPos) {
    vec3 prevWorld = worldPos + cameraVelocity;
    vec4 prevClip = gbufferPreviousProjection * (gbufferPreviousModelView * vec4(prevWorld, 1.0));
    if (prevClip.w <= 0.0) return vec2(-1.0);
    return (prevClip.xy / prevClip.w) * 0.5 + 0.5;
}

// Jitter-compensated screen-space reprojection:
// Computes exact unjittered motion (prevUnjitUV - currUnjitUV) so static pixels
// stay locked to currUV with zero subpixel jitter or drift.
vec2 reprojectScreenUV(vec2 currUV, vec3 worldPos) {
    vec2 prevUnjitUV = reprojectWorldToPrevUV(worldPos);
    if (prevUnjitUV.x < 0.0) return vec2(-1.0);
    vec4 currClip = gbufferProjection * (gbufferModelView * vec4(worldPos, 1.0));
    if (currClip.w <= 0.0) return vec2(-1.0);
    vec2 currUnjitUV = (currClip.xy / currClip.w) * 0.5 + 0.5;
    return currUV + (prevUnjitUV - currUnjitUV);
}

// Filter-independent 12-tap cornerless Catmull-Rom bicubic history fetch via texelFetch:
// - 1-tap exact fast-path when static (zero overhead).
// - Tile-clamped (`tileOff`) for race-free ping-pong TAA history without cross-tile bleed!
// - Returns RGB (Catmull-Rom) + A (non-ringing bilinear accumulated frame count).
vec4 fetchHistoryFast4(sampler2D tex, vec2 uv, ivec2 res, ivec2 tileOff) {
    vec2 pos = uv * vec2(res) - 0.5;
    ivec2 base = ivec2(floor(pos));
    vec2 f = pos - vec2(base);

    if (abs(f.x) < 1e-4 && abs(f.y) < 1e-4) {
        ivec2 p = clamp(base, ivec2(0), res - 1) + tileOff;
        vec4 tap = texelFetch(tex, p, 0);
        return vec4(max(tap.rgb, vec3(0.0)), max(tap.a, 0.0));
    }

    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 = 0.5 * (f3 - f2);

    ivec2 maxPx = res - 1;
    int x0 = clamp(base.x - 1, 0, maxPx.x) + tileOff.x;
    int x1 = clamp(base.x,     0, maxPx.x) + tileOff.x;
    int x2 = clamp(base.x + 1, 0, maxPx.x) + tileOff.x;
    int x3 = clamp(base.x + 2, 0, maxPx.x) + tileOff.x;
    int y0 = clamp(base.y - 1, 0, maxPx.y) + tileOff.y;
    int y1 = clamp(base.y,     0, maxPx.y) + tileOff.y;
    int y2 = clamp(base.y + 1, 0, maxPx.y) + tileOff.y;
    int y3 = clamp(base.y + 2, 0, maxPx.y) + tileOff.y;

    // 12-tap cornerless Catmull-Rom for RGB + exact bilinear of center 2x2 for frame count A.
    vec4 t11 = texelFetch(tex, ivec2(x1, y1), 0);
    vec4 t21 = texelFetch(tex, ivec2(x2, y1), 0);
    vec4 t12 = texelFetch(tex, ivec2(x1, y2), 0);
    vec4 t22 = texelFetch(tex, ivec2(x2, y2), 0);

    vec3 sum = vec3(0.0);
    sum += max(texelFetch(tex, ivec2(x1, y0), 0).rgb, vec3(0.0)) * (w1.x * w0.y);
    sum += max(texelFetch(tex, ivec2(x2, y0), 0).rgb, vec3(0.0)) * (w2.x * w0.y);

    sum += max(texelFetch(tex, ivec2(x0, y1), 0).rgb, vec3(0.0)) * (w0.x * w1.y);
    sum += max(t11.rgb, vec3(0.0)) * (w1.x * w1.y);
    sum += max(t21.rgb, vec3(0.0)) * (w2.x * w1.y);
    sum += max(texelFetch(tex, ivec2(x3, y1), 0).rgb, vec3(0.0)) * (w3.x * w1.y);

    sum += max(texelFetch(tex, ivec2(x0, y2), 0).rgb, vec3(0.0)) * (w0.x * w2.y);
    sum += max(t12.rgb, vec3(0.0)) * (w1.x * w2.y);
    sum += max(t22.rgb, vec3(0.0)) * (w2.x * w2.y);
    sum += max(texelFetch(tex, ivec2(x3, y2), 0).rgb, vec3(0.0)) * (w3.x * w2.y);

    sum += max(texelFetch(tex, ivec2(x1, y3), 0).rgb, vec3(0.0)) * (w1.x * w3.y);
    sum += max(texelFetch(tex, ivec2(x2, y3), 0).rgb, vec3(0.0)) * (w2.x * w3.y);

    float wNorm = 1.0 - (w0.x + w3.x) * (w0.y + w3.y);
    vec3 catmull = sum / max(wNorm, 1e-4);
    // Anti-ringing clamp against the center 2x2 neighborhood eliminates bicubic overshoot halos on sharp edges!
    vec3 c11 = max(t11.rgb, vec3(0.0));
    vec3 c21 = max(t21.rgb, vec3(0.0));
    vec3 c12 = max(t12.rgb, vec3(0.0));
    vec3 c22 = max(t22.rgb, vec3(0.0));
    vec3 minC = min(min(c11, c21), min(c12, c22));
    vec3 maxC = max(max(c11, c21), max(c12, c22));
    float aBilin = mix(mix(t11.a, t21.a, f.x), mix(t12.a, t22.a, f.x), f.y);
    return vec4(clamp(catmull, minC, maxC), max(aBilin, 0.0));
}

#endif
