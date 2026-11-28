#ifndef LIB_CAMERA_GLSL
#define LIB_CAMERA_GLSL

#include "/lib/Resources.glsl"

const uint CAMERA_METER_BINS = 128u;
const float CAMERA_METER_MIN_LOG = -16.0;
const float CAMERA_METER_BIN_WIDTH = 24.0 / float(CAMERA_METER_BINS);
const float CAMERA_METER_KEY = 0.18;

#if CAMERA_METERING_MODE == 1 || CAMERA_METERING_MODE == 2
vec2 cameraMeterRadii(ivec2 res) {
    #if CAMERA_METERING_MODE == 2
        const float frameFraction = 0.25;
    #else
        const float frameFraction = 0.025;
    #endif
    vec2 frameSize = vec2(res);
    vec2 halfFrame = 0.5 * frameSize;
    float radiusSquared = frameFraction * frameSize.x * frameSize.y * INV_PI;
    float radius = sqrt(radiusSquared);
    // Fit to extreme aspect ratios while keeping the area by extending the other axis.
    return min(halfFrame, radiusSquared / min(vec2(radius), halfFrame.yx));
}
#endif

bool validCameraExposure(float exposure) {
    return exposure > 0.0 && !isnan(exposure) && !isinf(exposure);
}

uvec2 cameraMeterSample(vec3 hdr, uint weight) {
    if (any(isnan(hdr)) || any(isinf(hdr))) return uvec2(0u);
    float logLuminance = log2(max(luma(max(hdr, vec3(0.0))), exp2(CAMERA_METER_MIN_LOG)));
    float bin = (logLuminance - CAMERA_METER_MIN_LOG) / CAMERA_METER_BIN_WIDTH;
    return uvec2(uint(clamp(bin, 0.0, float(CAMERA_METER_BINS - 1u))), weight);
}

vec4 adaptCameraExposure(float meteredLogLuminance, float previousExposure) {
    float targetEV = clamp(log2(CAMERA_METER_KEY * float(EXPOSURE)) - meteredLogLuminance,
                           float(CAMERA_MIN_EV), float(CAMERA_MAX_EV));
    float exposureEV = targetEV;
    if (frameCounter > 2 && validCameraExposure(previousExposure)) {
        float previousEV = clamp(log2(previousExposure), float(CAMERA_MIN_EV), float(CAMERA_MAX_EV));
        float tau = targetEV < previousEV ? float(CAMERA_ADAPT_BRIGHT) : float(CAMERA_ADAPT_DARK);
        float dt = clamp(frameTime, 0.0, 0.1);
        exposureEV = mix(previousEV, targetEV, 1.0 - exp(-dt / tau));
    }
    return vec4(meteredLogLuminance, exp2(exposureEV), exp2(targetEV), 1.0);
}

float cameraExposure(sampler2D exposureState) {
    #ifdef AUTO_EXPOSURE
        float automatic = texelFetch(exposureState, ivec2(0), 0).g;
        if (validCameraExposure(automatic)) return automatic;
        return clamp(float(EXPOSURE), exp2(float(CAMERA_MIN_EV)), exp2(float(CAMERA_MAX_EV)));
    #else
        return float(EXPOSURE);
    #endif
}

#endif
