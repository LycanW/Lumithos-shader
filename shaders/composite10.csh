#version 430 compatibility

#include "/lib/CameraOutput.glsl"

layout(local_size_x = 16, local_size_y = 16) in;
const ivec3 workGroups = ivec3(1, 1, 1);

uniform sampler2D taaHistTex, scratchTex, exposureTex;
layout(rgba32f) uniform writeonly image2D exposureImg;

shared uint histogram[CAMERA_VIEWFINDER_BINS];
shared uint peakCount;

void main() {
    uint tid = gl_LocalInvocationIndex;
    if (tid < CAMERA_VIEWFINDER_BINS) histogram[tid] = 0u;
    barrier();

    // Uniform full-frame sampling of the final image before the viewfinder overlay.
    ivec2 res = ivec2(screenSize());
    ivec2 grid = min(res, ivec2(64, 36));
    uint stride = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    float exposure = cameraExposure(exposureTex);
    vec3 whiteBalanceScale = cameraWhiteBalanceScale(exposureTex);
    for (uint i = tid; i < uint(grid.x * grid.y); i += stride) {
        uvec2 cell = uvec2(i % uint(grid.x), i / uint(grid.x));
        ivec2 px = ivec2((vec2(cell) + 0.5) * vec2(res) / vec2(grid));
        vec3 rgb = cameraDisplayColor(sampleCameraHdr(taaHistTex, px, res), scratchTex, px, res, exposure, whiteBalanceScale);
        uint bin = min(uint(saturate(luma(rgb)) * float(CAMERA_VIEWFINDER_BINS)), CAMERA_VIEWFINDER_BINS - 1u);
        atomicAdd(histogram[bin], 1u);
    }
    barrier();

    if (tid == 0u) {
        peakCount = 1u;
        for (uint bin = 0u; bin < CAMERA_VIEWFINDER_BINS; ++bin) peakCount = max(peakCount, histogram[bin]);
    }
    barrier();

    if (tid < CAMERA_VIEWFINDER_BINS) {
        float height = float(histogram[tid]) / float(peakCount);
        imageStore(exposureImg, ivec2(int(tid), CAMERA_VIEWFINDER_HISTOGRAM_ROW), vec4(height, 0.0, 0.0, 1.0));
    }
}
