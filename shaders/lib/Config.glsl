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
#define PRIM_END           0x001fffffu // 21 low bits, above every valid primitive index
#define MAX_SCENE_LIGHTS   4096u
#define LIGHT_CELL_XZ      5
#define LIGHT_CELL_Y       3
#define LIGHT_CELL_TOTAL   75u
#define LIGHT_CELL_CAP     256u

// ---- World-Space 4-Way Set-Associative Surface Hash Cache (262,144 directed patches = 4.0 MB) ----
#define SC_NUM_BUCKETS      65536u
#define SC_BUCKET_SIZE      4u
#define SC_TOTAL_SLOTS      262144u
#define SC_ACTIVE_WINDOW    4u
#define SC_STALE_FRAMES     24u
#define SC_WARMUP_SAMPLES   8u

// ---- Feature Toggles ----
// Diffuse tracing resolution: 0 = 1/4 (2x2 Bayer), 1 = 1/2 (Checkerboard), 2 = 1/1 (Full-Res)
#define DIFFUSE_TRACE_RES 0 // [0 1 2]
// 0 = 31-frame limit, 1 = 64-frame limit, 2 = frame-rate-scaled limits.
#define DIFFUSE_HISTORY_MODE 1 // [0 1 2]
#define SPECULAR_CHECKERBOARD
// 0 = GGX VNDF, 2 = fitted core and spatially reused tail.
#define SPECULAR_LOBE_MODE 2 // [0 2]
#define SPECULAR_BOUNCES 1 // [1 2 3 4]
//#define DIFFUSE_SSRT
#define SPECULAR_SSR
#define PATH_GUIDING
#define SPECULAR_REFLECTIONS
#define TAA_ENABLED
// Tonemapper: 0 = AgX (Default), 1 = GT7, 2 = PBR Neutral, 3 = ACES
#define TONEMAP_MODE 0 // [0 1 2 3]
#define AUTO_EXPOSURE
#define BLOOM_ENABLED
#define BLOOM_STRENGTH 0.06 // [0.02 0.04 0.06 0.08 0.12 0.16]
#define EXPOSURE 1.0 // [0.25 0.5 0.75 1.0 1.25 1.5 2.0 3.0]
#define WATER_PARALLAX
#define WATER_CAUSTICS
#define WAVE_STRENGTH 1.0 // [0.25 0.5 0.75 1.0 1.25 1.5 2.0]
#define AERIAL_PERSPECTIVE
#define FOG_DENSITY 1.0 // [0.25 0.5 0.75 1.0 1.5 2.0]
#define STAR_INTENSITY 1.0 // [0.0 0.5 1.0 1.5 2.0]
#define DEBUG_VIEW 0 // [0 1 2 3 4 5 6]

// ---- Lighting & Tracing Tuning ----
#define DIFFUSE_MAX_STEPS     64
#define SPECULAR_MAX_STEPS    96
#define RC_MAX_STEPS          64
#define EMITTER_RANGE         40 // Source coverage for diffuse and specular hits in blocks. Zero disables the limit.
#define SC_MAX_ACCUM_NEAR     48.0
#define SC_MAX_ACCUM_MID      72.0
#define SC_MAX_ACCUM_FAR      96.0
#define EMISSIVE_SCALE        3.5
#define SPECULAR_ROUGH_CUTOFF 0.65

// ---- Shadow Map Constants ----
const int   shadowMapResolution = 2048; // [1024 2048 3072 4096]
const float shadowDistance      = 256.0; // [128.0 192.0 256.0 320.0]
const float voxelDistance       = 88.0; // Unculled 360-deg radius for 160x96x160 voxel grid (reversed culling)
const float shadowIntervalSize  = 2.0;
const float sunPathRotation     = -35.0;
const float ambientOcclusionLevel = 0.0;
const bool  shadowtex0Nearest   = true;
const bool  shadowtex1Nearest   = true;
const bool  shadowcolor0Nearest = true;

/*
    const int colortex0Format = RGBA16F;  // lit scene / final
    const int colortex1Format = RG32UI;   // G-Buffer 0: albedoRGB8+sky8, pbr+flags+sss+ao
    const int colortex2Format = R32UI;    // G-Buffer 1: geoNormal16 + shadeNormal16
    const int colortex3Format = RGBA16F;  // Diffuse GI (raw -> denoised)
    const int colortex4Format = RGBA16F;  // Specular GI (raw -> denoised)
    const int colortex5Format = RGBA16F;  // Previous frame lit color (for SSRT/SSR reuse)
    const int colortex6Format = RGBA16F;  // Translucent in-scattered + reflected radiance (rgb), layer opacity (a)
    const int colortex7Format = RGBA16F;  // Translucent per-channel transmittance (multiplicative, clear = 1)
    const int colortex8Format = RG16F;    // Translucent refraction screen-UV offset

    const bool colortex0Clear = true;
    const bool colortex1Clear = false;
    const bool colortex2Clear = false;
    const bool colortex3Clear = false;
    const bool colortex4Clear = false;
    const bool colortex5Clear = false;
    const bool colortex6Clear = true;
    const bool colortex7Clear = true;
    const bool colortex8Clear = true;
    const vec4 colortex6ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
    const vec4 colortex7ClearColor = vec4(1.0, 1.0, 1.0, 1.0);
    const vec4 colortex8ClearColor = vec4(0.0, 0.0, 0.0, 0.0);
*/

#endif
