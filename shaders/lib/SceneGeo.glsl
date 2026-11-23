#ifndef LIB_SCENE_GEO_GLSL
#define LIB_SCENE_GEO_GLSL

#include "/lib/Resources.glsl"

// Canonical 6-face indexing:
// 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
uint normalToFaceIndex(vec3 n) {
    vec3 an = abs(n);
    if (an.x >= an.y && an.x >= an.z) return n.x > 0.0 ? 0u : 1u;
    if (an.y >= an.z)                 return n.y > 0.0 ? 2u : 3u;
    return n.z > 0.0 ? 4u : 5u;
}

vec3 faceIndexToNormal(uint faceIdx) {
    const vec3 normals[6] = vec3[6](
        vec3( 1.0,  0.0,  0.0),
        vec3(-1.0,  0.0,  0.0),
        vec3( 0.0,  1.0,  0.0),
        vec3( 0.0, -1.0,  0.0),
        vec3( 0.0,  0.0,  1.0),
        vec3( 0.0,  0.0, -1.0)
    );
    return normals[faceIdx];
}

void faceBasis(uint faceIdx, out vec3 tangent, out vec3 bitangent) {
    const vec3 T[6] = vec3[6](
        vec3( 0.0,  0.0, -1.0), // +X
        vec3( 0.0,  0.0,  1.0), // -X
        vec3( 1.0,  0.0,  0.0), // +Y
        vec3( 1.0,  0.0,  0.0), // -Y
        vec3( 1.0,  0.0,  0.0), // +Z
        vec3(-1.0,  0.0,  0.0)  // -Z
    );
    const vec3 B[6] = vec3[6](
        vec3( 0.0, -1.0,  0.0), // +X
        vec3( 0.0, -1.0,  0.0), // -X
        vec3( 0.0,  0.0,  1.0), // +Y
        vec3( 0.0,  0.0, -1.0), // -Y
        vec3( 0.0, -1.0,  0.0), // +Z
        vec3( 0.0, -1.0,  0.0)  // -Z
    );
    tangent = T[faceIdx];
    bitangent = B[faceIdx];
}

// Compute canonical (s, t) in [0, 1]^2 from voxel-local hit point p in [0, 1]^3.
vec2 faceLocalST(uint faceIdx, vec3 p) {
    if (faceIdx == 0u) return vec2(1.0 - p.z, 1.0 - p.y); // +X
    if (faceIdx == 1u) return vec2(      p.z, 1.0 - p.y); // -X
    if (faceIdx == 2u) return vec2(      p.x,       p.z); // +Y
    if (faceIdx == 3u) return vec2(      p.x, 1.0 - p.z); // -Y
    if (faceIdx == 4u) return vec2(      p.x, 1.0 - p.y); // +Z
    return                    vec2(1.0 - p.x, 1.0 - p.y); // -Z
}

// Pack Track 1 face descriptor (32 bits):
// [8:0]   tileX (0..511)
// [17:9]  tileY (0..511)
// [21:18] sizeExp (log2 tile pixels, 0..15)
// [24:22] uvTransform (swapUV<<2 | flipU<<1 | flipV)
// [25]    isTinted
// [26]    isTranslucent
// [30:27] emissiveLevel (0..15)
// [31]    valid (1)
uint packFaceDesc(uvec2 tileCoord, uint sizeExp, uint uvTransform, bool isTinted, bool isTranslucent, uint emissiveLevel) {
    return (tileCoord.x & 511u)
         | ((tileCoord.y & 511u) << 9u)
         | ((sizeExp & 15u) << 18u)
         | ((uvTransform & 7u) << 22u)
         | (uint(isTinted) << 25u)
         | (uint(isTranslucent) << 26u)
         | ((emissiveLevel & 15u) << 27u)
         | 0x80000000u;
}

// Decode Track 1 face descriptor into exact atlas UV + material metadata.
vec2 decodeFaceAtlasUV(uint desc, uint faceIdx, vec3 localPos, vec2 invAtlasSize, out bool isTinted, out bool isTranslucent, out float emissive) {
    uvec2 tileCoord = uvec2(desc & 511u, (desc >> 9u) & 511u);
    uint  sizeExp   = (desc >> 18u) & 15u;
    uint  uvXform   = (desc >> 22u) & 7u;
    isTinted        = ((desc >> 25u) & 1u) != 0u;
    isTranslucent   = ((desc >> 26u) & 1u) != 0u;
    emissive        = float((desc >> 27u) & 15u) * (1.0 / 15.0);

    vec2 st = saturate(faceLocalST(faceIdx, localPos));
    bool swapUV = (uvXform & 4u) != 0u;
    bool flipU  = (uvXform & 2u) != 0u;
    bool flipV  = (uvXform & 1u) != 0u;

    vec2 uv = swapUV ? vec2(st.y, st.x) : st;
    if (flipU) uv.x = 1.0 - uv.x;
    if (flipV) uv.y = 1.0 - uv.y;

    // Slight inset avoids atlas neighbor bleeding at tile borders.
    uv = uv * 0.996 + 0.002;
    vec2 tileOriginUV = vec2(tileCoord) * (1.0 / 512.0);
    vec2 tileExtentUV = vec2(float(1u << sizeExp)) * invAtlasSize;
    return tileOriginUV + uv * tileExtentUV;
}

uint packTintAndSky(vec3 tintSrgb, float skyLight) {
    uint rgb = packUnorm4x8(vec4(saturate(tintSrgb), 0.0)) & 0x00ffffffu;
    uint sky4 = uint(round(saturate(skyLight) * 15.0));
    return rgb | (sky4 << 24u);
}

vec3 unpackTint(uint tintAndSky) {
    return unpackUnorm4x8(tintAndSky).rgb;
}

float unpackSky(uint tintAndSky) {
    return float((tintAndSky >> 24u) & 15u) * (1.0 / 15.0);
}

// Pack & unpack Track 2 compact non-cube triangle (48 bytes).
ScenePrim packScenePrim(vec3 p0, vec3 e1, vec3 e2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 tintSrgb, uint emissive4, bool isTranslucent, uint nextIdx) {
    ScenePrim pr;
    pr.geo.x  = packSnorm2x16(clamp(p0.xy * 0.5, -1.0, 1.0));
    pr.geo.y  = packSnorm2x16(clamp(vec2(p0.z, e1.x) * 0.5, -1.0, 1.0));
    pr.geo.z  = packSnorm2x16(clamp(e1.yz * 0.5, -1.0, 1.0));
    pr.geo.w  = packSnorm2x16(clamp(e2.xy * 0.5, -1.0, 1.0));
    pr.uv.x   = packSnorm2x16(clamp(vec2(e2.z * 0.5, 0.0), -1.0, 1.0));
    pr.uv.y   = packUnorm2x16(saturate(uv0));
    pr.uv.z   = packUnorm2x16(saturate(uv1));
    pr.uv.w   = packUnorm2x16(saturate(uv2));
    uint rgb  = packUnorm4x8(vec4(saturate(tintSrgb), 0.0)) & 0x00ffffffu;
    pr.meta   = uvec4(nextIdx, rgb | ((emissive4 & 15u) << 24u) | (uint(isTranslucent) << 28u), 0u, 0u);
    return pr;
}

void unpackScenePrim(ScenePrim pr, out vec3 p0, out vec3 e1, out vec3 e2, out vec2 uv0, out vec2 uv1, out vec2 uv2, out vec3 tintSrgb, out float emissive, out bool isTranslucent, out uint nextIdx) {
    vec2 g0 = unpackSnorm2x16(pr.geo.x) * 2.0;
    vec2 g1 = unpackSnorm2x16(pr.geo.y) * 2.0;
    vec2 g2 = unpackSnorm2x16(pr.geo.z) * 2.0;
    vec2 g3 = unpackSnorm2x16(pr.geo.w) * 2.0;
    vec2 g4 = unpackSnorm2x16(pr.uv.x)  * 2.0;

    p0  = vec3(g0.x, g0.y, g1.x);
    e1  = vec3(g1.y, g2.x, g2.y);
    e2  = vec3(g3.x, g3.y, g4.x);
    uv0 = unpackUnorm2x16(pr.uv.y);
    uv1 = unpackUnorm2x16(pr.uv.z);
    uv2 = unpackUnorm2x16(pr.uv.w);

    nextIdx       = pr.meta.x;
    tintSrgb      = unpackUnorm4x8(pr.meta.y).rgb;
    emissive      = float((pr.meta.y >> 24u) & 15u) * (1.0 / 15.0);
    isTranslucent = ((pr.meta.y >> 28u) & 1u) != 0u;
}

#endif
