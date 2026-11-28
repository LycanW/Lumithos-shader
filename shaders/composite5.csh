#version 430 compatibility

// Single-workgroup pass:
// 1) Downsample Bloom L3 -> L4 (resolution-independent stride loop).
// 2) Meter scene HDR and update temporal auto exposure.
#include "/lib/Bloom.glsl"
#include "/lib/Camera.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(1, 1, 1);

uniform sampler2D scratchTex;
layout(rgba16f) uniform writeonly image2D scratchImg;
layout(rgba32f) uniform image2D exposureImg;

#ifdef AUTO_EXPOSURE
#if CAMERA_METERING_MODE != 0
uniform sampler2D taaHistTex;
#endif
#if CAMERA_METERING_MODE == 3
const int CAMERA_METER_ZONES_PER_AXIS = 4;
const int CAMERA_METER_ZONE_SAMPLES_PER_AXIS = 8;
const uint CAMERA_METER_ZONES = uint(CAMERA_METER_ZONES_PER_AXIS * CAMERA_METER_ZONES_PER_AXIS);
const uint CAMERA_METER_FRAME_HISTOGRAM = CAMERA_METER_ZONES * CAMERA_METER_BINS;
const uint CAMERA_METER_HISTOGRAM_SIZE = CAMERA_METER_FRAME_HISTOGRAM + CAMERA_METER_BINS;
shared vec2 meterZoneReadings[CAMERA_METER_ZONES];
#else
const uint CAMERA_METER_HISTOGRAM_SIZE = CAMERA_METER_BINS;
#endif
shared uint meterHistogram[CAMERA_METER_HISTOGRAM_SIZE];

void accumulateMeterSample(uint firstBin, vec3 hdr, uint weight) {
    uvec2 sampleMeter = cameraMeterSample(hdr, weight);
    if (sampleMeter.y > 0u) atomicAdd(meterHistogram[firstBin + sampleMeter.x], sampleMeter.y);
}

// Return the trimmed mean in log luminance and the total valid sample weight.
vec2 meterHistogramReading(uint firstBin, vec2 percentileRange) {
    uint totalWeight = 0u;
    for (uint bin = 0u; bin < CAMERA_METER_BINS; ++bin) totalWeight += meterHistogram[firstBin + bin];
    if (totalWeight == 0u) return vec2(log2(CAMERA_METER_KEY), 0.0);

    vec2 bounds = float(totalWeight) * percentileRange;
    uint prefix = 0u;
    float sum = 0.0;
    for (uint bin = 0u; bin < CAMERA_METER_BINS; ++bin) {
        uint end = prefix + meterHistogram[firstBin + bin];
        float weight = max(min(float(end), bounds.y) - max(float(prefix), bounds.x), 0.0);
        float logLuminance = CAMERA_METER_MIN_LOG + (float(bin) + 0.5) * CAMERA_METER_BIN_WIDTH;
        sum += weight * logLuminance;
        prefix = end;
    }
    return vec2(sum / (bounds.y - bounds.x), float(totalWeight));
}

#if CAMERA_METERING_MODE == 3
float meterMultiZoneLogLuminance() {
    vec2 weightedSum = vec2(0.0);
    for (uint zone = 0u; zone < CAMERA_METER_ZONES; ++zone) {
        vec2 reading = meterZoneReadings[zone];
        weightedSum += vec2(reading.x, 1.0) * reading.y;
    }
    if (weightedSum.y == 0.0) return log2(CAMERA_METER_KEY);
    float sceneLogLuminance = weightedSum.x / weightedSum.y;

    // Hold broad highlights near linear 4.0 before exposure compensation.
    // Averaging a percentile band reacts smoothly to area and ignores the top 2%.
    const float highlightCeilingEV = 2.0;
    float highlightLogLuminance = meterHistogramReading(CAMERA_METER_FRAME_HISTOGRAM, vec2(0.90, 0.98)).x;
    return max(sceneLogLuminance, highlightLogLuminance + log2(CAMERA_METER_KEY) - highlightCeilingEV);
}
#endif
#endif

void main() {
    ivec2 res = ivec2(screenSize());
    ivec2 lid = ivec2(gl_LocalInvocationID.xy);
    uint tid = gl_LocalInvocationIndex;

    #if defined(BLOOM_ENABLED) || (defined(AUTO_EXPOSURE) && CAMERA_METERING_MODE == 0)
        BloomRect l3 = getBloomRect(3, false, res);
    #endif

    // 1) Downsample L3 -> L4 (tile 0)
    #ifdef BLOOM_ENABLED
        BloomRect l4 = getBloomRect(4, false, res);
        for (int y = lid.y; y < l4.size.y; y += 16) {
            for (int x = lid.x; x < l4.size.x; x += 16) {
                vec3 col = bloomDownsample4x4(scratchTex, ivec2(x, y), l3.origin, l3.size, false);
                imageStore(scratchImg, l4.origin + ivec2(x, y), roundHalf(vec4(col, 1.0)));
            }
        }
    #endif

    #ifdef AUTO_EXPOSURE
        for (uint bin = tid; bin < CAMERA_METER_HISTOGRAM_SIZE; bin += gl_WorkGroupSize.x * gl_WorkGroupSize.y) {
            meterHistogram[bin] = 0u;
        }
        barrier();

        #if CAMERA_METERING_MODE != 0
            ivec2 currOff = ivec2((frameCounter & 1) * res.x, 0);
        #endif

        #if CAMERA_METERING_MODE == 3
            const int gridSize = CAMERA_METER_ZONES_PER_AXIS * CAMERA_METER_ZONE_SAMPLES_PER_AXIS;
            for (int y = lid.y; y < gridSize; y += int(gl_WorkGroupSize.y)) {
                for (int x = lid.x; x < gridSize; x += int(gl_WorkGroupSize.x)) {
                    ivec2 cell = ivec2(x, y);
                    ivec2 zone = cell / CAMERA_METER_ZONE_SAMPLES_PER_AXIS;
                    uint firstBin = uint(zone.x + zone.y * CAMERA_METER_ZONES_PER_AXIS) * CAMERA_METER_BINS;
                    ivec2 px = ivec2((vec2(cell) + 0.5) * vec2(res) / float(gridSize));
                    accumulateMeterSample(firstBin, texelFetch(taaHistTex, px + currOff, 0).rgb, 1u);
                }
            }
        #elif CAMERA_METERING_MODE != 0
            #if CAMERA_METERING_MODE == 2
                const int gridSize = 32;
            #else
                const int gridSize = 16;
            #endif
            vec2 radii = cameraMeterRadii(res);
            for (int y = lid.y; y < gridSize; y += int(gl_WorkGroupSize.y)) {
                for (int x = lid.x; x < gridSize; x += int(gl_WorkGroupSize.x)) {
                    vec2 offset = (vec2(x, y) + 0.5) * (2.0 / float(gridSize)) - 1.0;
                    if (dot(offset, offset) <= 1.0) {
                        ivec2 px = clamp(ivec2(vec2(res) * 0.5 + offset * radii), ivec2(0), res - 1);
                        accumulateMeterSample(0u, texelFetch(taaHistTex, px + currOff, 0).rgb, 1u);
                    }
                }
            }
        #else
            vec2 invL3 = 1.0 / vec2(max(l3.size, ivec2(1)));
            for (int y = lid.y; y < l3.size.y; y += 16) {
                for (int x = lid.x; x < l3.size.x; x += 16) {
                    vec3 hdr = texelFetch(scratchTex, l3.origin + ivec2(x, y), 0).rgb;
                    vec2 ndc = (vec2(x, y) + 0.5) * invL3 * 2.0 - 1.0;
                    uint weight = uint(round(256.0 * exp(-dot(ndc, ndc) * 1.1)));
                    accumulateMeterSample(0u, hdr, weight);
                }
            }
        #endif
        barrier();

        #if CAMERA_METERING_MODE == 3
            if (tid < CAMERA_METER_BINS) {
                uint frameWeight = 0u;
                for (uint zone = 0u; zone < CAMERA_METER_ZONES; ++zone) {
                    frameWeight += meterHistogram[zone * CAMERA_METER_BINS + tid];
                }
                meterHistogram[CAMERA_METER_FRAME_HISTOGRAM + tid] = frameWeight;
            }
            if (tid < CAMERA_METER_ZONES) {
                meterZoneReadings[tid] = meterHistogramReading(tid * CAMERA_METER_BINS, vec2(0.25, 0.90));
            }
            barrier();
        #endif

        if (tid == 0u) {
            #if CAMERA_METERING_MODE == 3
                float meteredLogLuminance = meterMultiZoneLogLuminance();
            #else
                float meteredLogLuminance = meterHistogramReading(0u, vec2(0.25, 0.90)).x;
            #endif
            float previousExposure = imageLoad(exposureImg, ivec2(0)).g;
            vec4 exposure = adaptCameraExposure(meteredLogLuminance, previousExposure);
            imageStore(exposureImg, ivec2(0), exposure);
        }
    #endif
}
