#version 430 compatibility
#define DIFFUSE_1D_PASS 3
#define DIFFUSE_1D_AXIS vec2(4.0, -4.0)
#define DIFFUSE_1D_VAR_SCALE 0.55
#define DIFFUSE_1D_SRC_TILE 0
#define DIFFUSE_1D_WRITE_FINAL
#include "/program/DiffuseSpatial1D.glsl"
