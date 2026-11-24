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
#define MAX_SCENE_LIGHTS   4096u
#define LIGHT_CELL_XZ      5
#define LIGHT_CELL_Y       3
#define LIGHT_CELL_TOTAL   75u
#define LIGHT_CELL_CAP     256u

// ---- O(1) 3-Cascade Toroidal 3D Clipmap Radiance Cache ----
#define RC_XZ              64
#define RC_Y               32
#define RC_CASCADES        3
#define RC_CASCADE_CELLS   131072u
#define RC_TOTAL_CELLS     393216u
#define RC_DEDUP_WORDS     12288u
#define RC_WORKLIST_CAP    8192u

// ---- Feature Toggles ----
// Diffuse tracing resolution: 0 = 1/4 (2x2 Bayer), 1 = 1/2 (Checkerboard), 2 = 1/1 (Full-Res)
#define DIFFUSE_TRACE_RES 0 // [0 1 2]
#define SPECULAR_CHECKERBOARD
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
#define DEBUG_VIEW 0 // [0 1 2 3 4 5 6]

// ---- Lighting & Tracing Tuning ----
#define DIFFUSE_MAX_STEPS     64
#define SPECULAR_MAX_STEPS    96
#define RC_MAX_STEPS          64
#define RC_UPDATE_PERIOD      4u
#define RC_WARMUP_UPDATES     12u
#define RC_MAX_ACCUM          128.0
#define DIFFUSE_MAX_HISTORY   96.0
#define SPECULAR_MAX_HISTORY  32.0
#define EMISSIVE_SCALE        3.5
#define EMISSIVE_GI_SCALE     10.0
#define SPECULAR_ROUGH_CUTOFF 0.65

// ---- Shadow Map Constants ----
const int   shadowMapResolution = 2048;
const float shadowDistance      = 128.0;
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
