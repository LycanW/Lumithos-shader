#ifndef LIB_CAMERA_VIEWFINDER_GLSL
#define LIB_CAMERA_VIEWFINDER_GLSL

#include "/lib/CameraOutput.glsl"

uniform sampler2D cameraHistogramTex;

vec3 drawCameraViewfinder(vec3 color, vec2 pixel, ivec2 res) {
    #if defined(AUTO_EXPOSURE) && (CAMERA_METERING_MODE == 1 || CAMERA_METERING_MODE == 2)
        vec2 radii = cameraMeterRadii(res);
        float edge = abs(length((pixel - vec2(res) * 0.5) / radii) - 1.0) * min(radii.x, radii.y);
        color = mix(color, vec3(0.0), 0.62 * (1.0 - smoothstep(1.4, 2.7, edge)));
        color = mix(color, vec3(1.0), 0.82 * (1.0 - smoothstep(0.4, 1.2, edge)));
    #endif

    float scale = float(min(res.x, res.y)) / 1000.0;
    vec2 size = vec2(196.0, 82.0) * scale;
    vec2 origin = vec2(float(res.x) - size.x - 18.0 * scale, 22.0 * scale);
    vec2 local = pixel - origin;
    if (any(lessThan(local, vec2(0.0))) || any(greaterThanEqual(local, size))) return color;

    color = mix(color, vec3(0.025), 0.76);
    float padding = 5.0 * scale;
    vec2 plot = (local - padding) / (size - 2.0 * padding);
    if (any(lessThan(plot, vec2(0.0))) || any(greaterThanEqual(plot, vec2(1.0)))) return color;

    int bin = int(plot.x * float(CAMERA_VIEWFINDER_BINS));
    float height = texelFetch(cameraHistogramTex, ivec2(bin, 0), 0).r;
    float baseline = 1.0 / (size.y - 2.0 * padding);
    return mix(color, vec3(0.93), plot.y < height ? 0.9 : (plot.y < baseline ? 0.35 : 0.0));
}

#endif
