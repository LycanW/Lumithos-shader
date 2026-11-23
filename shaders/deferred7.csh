#version 430 compatibility
#define DIFFUSE_1D_PASS 3
#define DIFFUSE_1D_AXIS vec2(0.0, 32.0)
#define DIFFUSE_1D_VAR_SCALE 0.30
#define DIFFUSE_1D_SRC_TILE 0
#define DIFFUSE_1D_DST_TILE 1
#include "/program/DiffuseSpatial1D.glsl"
