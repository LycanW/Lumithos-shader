#version 430 compatibility

/*
    const int colortex0Format = RGBA16F;
    const int colortex1Format = RG32UI;
    const int colortex2Format = R32UI;
    const int colortex3Format = RGBA16F;
    const int colortex4Format = RGBA16F;
    const int colortex5Format = RGBA16F;

    const bool colortex0Clear = true;
    const bool colortex1Clear = false;
    const bool colortex2Clear = false;
    const bool colortex3Clear = false;
    const bool colortex4Clear = false;
    const bool colortex5Clear = false;
*/

#include "/lib/CameraOutput.glsl"
#if defined(CAMERA_VIEWFINDER) && DEBUG_VIEW == 0
    #include "/lib/CameraViewfinder.glsl"
#endif

uniform sampler2D taaHistTex;
uniform sampler2D scratchTex;
uniform sampler2D exposureTex;

in vec2 vUV;
layout(location = 0) out vec4 fragColor;

void main() {
    ivec2 res = ivec2(screenSize());
    ivec2 px = clamp(ivec2(gl_FragCoord.xy), ivec2(0), res - 1);
    vec3 hdr = sampleCameraHdr(taaHistTex, px, res);

    #if DEBUG_VIEW == 5 || DEBUG_VIEW == 6
        vec3 display = linearToSrgb(hdr);
    #elif DEBUG_VIEW != 0
        vec3 display = linearToSrgb(applyTonemap(hdr * float(EXPOSURE)));
    #else
        vec3 display = cameraDisplayColor(hdr, scratchTex, px, res,
                                         cameraExposure(exposureTex), cameraWhiteBalanceScale(exposureTex));
        #ifdef CAMERA_VIEWFINDER
            display = drawCameraViewfinder(display, vec2(px) + 0.5, res, exposureTex);
        #endif
    #endif

    fragColor = vec4(display, 1.0);
}
