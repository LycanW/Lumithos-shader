#ifndef LIB_EMITTER_GEOMETRY_GLSL
#define LIB_EMITTER_GEOMETRY_GLSL

#include "/lib/SceneGeo.glsl"

const int EMITTER_TRIANGLE_LIMIT = 160;
const uint EMITTER_CUBE = 0x80000000u;
const float EMITTER_CELL_EPSILON = 1e-4;

layout(std430, binding = 4) buffer EmitterGeometry {
    uvec2 shapes[MAX_SCENE_LIGHTS]; // Cube face mask, or triangle count and total area.
    uvec2 triangles[MAX_SCENE_LIGHTS * EMITTER_TRIANGLE_LIMIT]; // Primitive index and cumulative area.
} emitterGeometry;

bool beyondEmitterRange(float distance) {
    return EMITTER_RANGE > 0 && distance > float(EMITTER_RANGE);
}

#if EMITTER_RANGE > 0
bool emitterCoversCell(ivec3 voxel, vec3 faceNormal, vec3 receiver) {
    // A full cube emits from a face. Irregular emitters use the owning voxel as their bounds.
    vec3 sourceMin = vec3(voxel) + max(faceNormal, vec3(0.0));
    vec3 sourceMax = vec3(voxel) + 1.0 + min(faceNormal, vec3(0.0));
    vec3 receiverMin = floor(receiver);
    vec3 gap = max(max(sourceMin - (receiverMin + 1.0), receiverMin - sourceMax), vec3(0.0));
    return dot(gap, gap) <= float(EMITTER_RANGE * EMITTER_RANGE);
}
#endif

bool inEmitterCell(vec3 localPosition, vec3 normal) {
    // Spanning triangles belong to one cell at each sampled point. Allow packed
    // position error on grid-aligned planes, which are registered to one side.
    vec3 eps = vec3(greaterThan(abs(normal), vec3(0.99999))) * EMITTER_CELL_EPSILON;
    return all(greaterThanEqual(localPosition, -eps)) &&
           all(lessThan(localPosition, vec3(1.0) + eps));
}

uint frontEmitterFaces(uint faceMask, vec3 receiverLocal) {
    uint facing = 0u;
    if (receiverLocal.x > 1.0) facing |= 1u;
    if (receiverLocal.x < 0.0) facing |= 2u;
    if (receiverLocal.y > 1.0) facing |= 4u;
    if (receiverLocal.y < 0.0) facing |= 8u;
    if (receiverLocal.z > 1.0) facing |= 16u;
    if (receiverLocal.z < 0.0) facing |= 32u;
    return facing & faceMask;
}

#endif
