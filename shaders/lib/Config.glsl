#ifndef LIB_CONFIG_GLSL
#define LIB_CONFIG_GLSL

// ---- Direct-Addressed Voxel Grid & 64-Way Occupancy Hierarchy ----
#define VOXEL_RES_XZ 160
#define VOXEL_RES_Y  96
#define BRICK_RES_XZ 40
#define BRICK_RES_Y  24
#define SUPER_RES_XZ 10
#define SUPER_RES_Y  6
#define TOTAL_VOXELS 2457600u

#define MAX_PRIMITIVES     1048576u
#define PRIM_END           0xffffffffu

// ---- O(1) 3-Cascade Toroidal 3D Clipmap Radiance Cache ----
#define RC_XZ              64
#define RC_Y               32
#define RC_CASCADES        3
#define RC_CASCADE_CELLS   131072u
#define RC_TOTAL_CELLS     393216u
#define RC_DEDUP_WORDS     12288u
#define RC_WORKLIST_CAP    8192u

// ---- Feature Toggles ----
#define DIFFUSE_SSRT
#define SPECULAR_SSR
#define PATH_GUIDING
#define SPECULAR_REFLECTIONS
#define TAA_ENABLED
#define EXPOSURE 1.0 // [0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0]
#define DEBUG_VIEW 0 // [0 1 2 3 4 5]

// ---- Lighting & Tracing Tuning ----
#define DIFFUSE_MAX_STEPS     80
#define SPECULAR_MAX_STEPS    96
#define RC_MAX_STEPS          80
#define RC_UPDATE_PERIOD      4u
#define RC_WARMUP_UPDATES     12u
#define RC_MAX_ACCUM          128.0
#define DIFFUSE_MAX_HISTORY   96.0
#define SPECULAR_MAX_HISTORY  64.0
#define EMISSIVE_SCALE        3.5
#define EMISSIVE_GI_SCALE     10.0
#define SPECULAR_ROUGH_CUTOFF 0.50

// ---- Shadow Map Constants ----
const int   shadowMapResolution = 2048;
const float shadowDistance      = 128.0;
const float shadowIntervalSize  = 2.0;
const float sunPathRotation     = -35.0;
const float ambientOcclusionLevel = 0.0;
const bool  shadowtex0Nearest   = true;
const bool  shadowtex1Nearest   = true;

/*
    const int colortex0Format = RGBA16F;  // lit scene / final
    const int colortex1Format = RG32UI;   // G-Buffer 0: albedoRGB8+sky8, pbr+flags
    const int colortex2Format = R32UI;    // G-Buffer 1: geoNormal16 + shadeNormal16
    const int colortex3Format = RGBA16F;  // Diffuse GI (raw -> denoised)
    const int colortex4Format = RGBA16F;  // Specular GI (raw -> denoised)
    const int colortex5Format = RGBA16F;  // Previous frame lit color (for SSRT/SSR reuse)

    const bool colortex0Clear = true;
    const bool colortex1Clear = false;
    const bool colortex2Clear = false;
    const bool colortex3Clear = false;
    const bool colortex4Clear = false;
    const bool colortex5Clear = false;
*/

#endif
