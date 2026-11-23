#version 430 compatibility
#define DIFFUSE_1D_PASS 2
#define DIFFUSE_1D_AXIS vec2(32.0, 0.0)
#define DIFFUSE_1D_VAR_SCALE 0.30
#define DIFFUSE_1D_SRC_TILE 1
#define DIFFUSE_1D_DST_TILE 0
#include "/program/DiffuseSpatial1D.glsl"
