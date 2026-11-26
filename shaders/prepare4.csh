#version 430 compatibility
#include "/lib/Resources.glsl"

layout(local_size_x = 128) in;
const ivec3 workGroups = ivec3(8192, 1, 1); // MAX_PRIMITIVES / 128

ivec2 unpackSigned(uint v) {
    return ivec2(bitfieldExtract(int(v), 0, 16), bitfieldExtract(int(v), 16, 16));
}
ivec2 unpackUV16(uint v) { return ivec2(v & 65535u, v >> 16u); }

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= min(primPool.primCount, MAX_PRIMITIVES)) return;
    uint j = primPool.prims[i].meta.x;
    if (j == PRIM_END || primPool.prims[i].meta.y != primPool.prims[j].meta.y) return;

    // Recognize (A,B,C), (C,D,A) from their actual packed vertices and UVs.
    uvec4 u = primPool.prims[i].uv, v = primPool.prims[j].uv;
    if (u.y != v.w || u.w != v.y || unpackUV16(u.y) + unpackUV16(u.w) != unpackUV16(u.z) + unpackUV16(v.z)) return;
    uvec4 a = primPool.prims[i].geo, b = primPool.prims[j].geo;
    ivec2 az = unpackSigned(a.y), bz = unpackSigned(b.y);
    ivec3 origin = ivec3(unpackSigned(a.x), az.x);
    ivec3 edge = ivec3(az.y, unpackSigned(a.z));
    ivec3 diagonal = ivec3(unpackSigned(a.w), unpackSigned(u.x).x);
    if (origin + diagonal != ivec3(unpackSigned(b.x), bz.x)
        || edge != -ivec3(bz.y, unpackSigned(b.z))
        || diagonal != -ivec3(unpackSigned(b.w), unpackSigned(v.x).x)) return;

    vec3 low = vec3(origin + min(min(ivec3(0), edge), min(diagonal, diagonal - edge))) * (2.0 / 32767.0);
    vec3 high = vec3(origin + max(max(ivec3(0), edge), max(diagonal, diagonal - edge))) * (2.0 / 32767.0);
    if (any(lessThan(low, vec3(-2.0))) || any(greaterThan(high, vec3(2.0)))) return;

    // Seven bits per bound, rounded outwards; split high.y across the two spare words.
    uvec3 lo = uvec3(floor((low + 2.0) * 31.0)), hi = uvec3(ceil((high + 2.0) * 31.0));
    uint upper = ((hi.y >> 4u) | (hi.z << 3u)) << 21u;
    primPool.prims[i].meta.z = 0x80000000u | upper | primPool.prims[j].meta.x;
    primPool.prims[i].meta.w = lo.x | (lo.y << 7u) | (lo.z << 14u) | (hi.x << 21u) | (hi.y << 28u);
}
