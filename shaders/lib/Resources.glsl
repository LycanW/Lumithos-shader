#ifndef LIB_RESOURCES_GLSL
#define LIB_RESOURCES_GLSL

#include "/lib/Config.glsl"
#include "/lib/Common.glsl"

// ---- Standard Iris Uniforms ----
uniform int   frameCounter;
uniform float frameTime;
uniform float frameTimeCounter;
uniform float viewWidth;
uniform float viewHeight;
uniform float near;
uniform float far;
uniform float rainStrength;
uniform float wetness;
uniform float sunAngle;
uniform float shadowAngle;
uniform int   worldTime;
uniform ivec2 atlasSize;
uniform ivec3 cameraPositionInt;
uniform ivec3 previousCameraPositionInt;
uniform vec3  cameraPosition;
uniform vec3  cameraPositionFract;
uniform vec3  previousCameraPositionFract;
uniform vec3  cameraVelocity;
uniform vec2  taaJitter;
uniform vec3  sunPosition;
uniform vec3  moonPosition;
uniform vec3  shadowLightPosition;
uniform mat4  gbufferModelView;
uniform mat4  gbufferModelViewInverse;
uniform mat4  gbufferProjection;
uniform mat4  gbufferProjectionInverse;
uniform mat4  gbufferPreviousModelView;
uniform mat4  gbufferPreviousProjection;
uniform mat4  shadowModelView;
uniform mat4  shadowModelViewInverse;
uniform mat4  shadowProjection;
uniform mat4  shadowProjectionInverse;

// ---- Standard Samplers ----
uniform sampler2D  colortex0;
uniform usampler2D colortex1;
uniform usampler2D colortex2;
uniform sampler2D  colortex3;
uniform sampler2D  colortex4;
uniform sampler2D  colortex5;
uniform sampler2D  depthtex0;
uniform sampler2D  depthtex1;
uniform sampler2D  shadowtex0;
uniform sampler2D  shadowtex1;
uniform sampler2D  shadowcolor0;
uniform sampler2D  colortex14; // blocks atlas
uniform sampler2D  colortex15; // blocks_s specular atlas

// ---- Voxel Grid & 64-Way Occupancy Hierarchy ----
#ifdef STAGE_VOXEL_GRID_WRITE
    layout(r8ui) uniform uimage3D voxelGridImg;
#else
    uniform usampler3D voxelGridTex;
#endif

#ifdef STAGE_BRICK_WRITE
    layout(r8ui) uniform uimage3D brickOccImg;
#else
    uniform usampler3D brickOccTex;
#endif

#ifdef STAGE_SUPER_WRITE
    layout(r8ui) uniform uimage3D superOccImg;
#else
    uniform usampler3D superOccTex;
#endif

// ---- Track 1: Direct-Addressed 6-Face Full-Cube Surface Pool (32 bytes/voxel, ZERO spinlocks!) ----
struct SurfaceVoxel {
    uint face[6];    // +X, -X, +Y, -Y, +Z, -Z
    uint tintAndSky; // rgb8 tint | sky4 << 24 | flags4 << 28
    uint primHead;   // Track 2 linked list head (PRIM_END = none)
};

layout(std430, binding = 0) coherent buffer SurfacePoolBuffer {
    SurfaceVoxel surfaces[];
} surfacePool;

uint voxelLinearIndex(ivec3 v) {
    return uint(v.x) + uint(v.z) * uint(VOXEL_RES_XZ) + uint(v.y) * uint(VOXEL_RES_XZ * VOXEL_RES_XZ);
}

// ---- Track 2: Non-Cube Compact Primitive Pool (48 bytes/prim) ----
struct ScenePrim {
    uvec4 geo;  // x: packSnorm2x16(p0.xy*0.5), y: packSnorm2x16(p0.z*0.5, e1.x*0.5), z: packSnorm2x16(e1.yz*0.5), w: packSnorm2x16(e2.xy*0.5)
    uvec4 uv;   // x: packSnorm2x16(e2.z*0.5, 0), y: packUnorm2x16(uv0), z: packUnorm2x16(uv1), w: packUnorm2x16(uv2)
    uvec4 meta; // x: next, y: tintRgb8 | (emissive4 << 24) | (flags4 << 28), z: pad, w: pad
};

layout(std430, binding = 1) coherent buffer PrimPoolBuffer {
    uint primCount;
    uint pad0;
    uint pad1;
    uint pad2;
    ScenePrim prims[];
} primPool;

// ---- O(1) 3-Cascade Toroidal 3D Clipmap Radiance Cache (16 bytes/probe = 1x 128-bit load!) ----
// word0: worldKey (worldCell.x:10 | y:10 | z:10 | cascade:2, XOR 1 -> 0 = empty)
// word1: normalized Co(10b signed) | Cg(10b signed) | accumFrames(12b)
// word2..3: packHalf4x16(L0_Y, L1_x, L1_y, L1_z)
struct RCProbe {
    uint  worldKey;
    uint  chromaFrames;
    uvec2 shLuma;
};

layout(std430, binding = 2) buffer RadianceCacheClipmap {
    RCProbe cells[];
} rcClipmap;

// Ping-pong active-set dedup bitmask + compacted probe worklist (unsized SSBO -> instant driver link!)
// data[0 .. 2*RC_DEDUP_WORDS - 1]         = dedup bitmask (2 sides)
// data[2*RC_DEDUP_WORDS .. +2*RC_WORKLIST_CAP - 1] = active worklist entries (2 sides)
layout(std430, binding = 3) coherent buffer RadianceCacheWorklist {
    uint count[2];
    uint pad0;
    uint pad1;
    uint data[];
} rcWork;

// ---- Compact Emissive Voxel LightList for 1-SPP RIS Next-Event Estimation (64 KB, L2-resident!) ----
struct SceneLight {
    vec4 posAndPower;  // xyz: voxel-space emitter center, w: block emission level (0..1)
    vec4 colorAndMeta; // rgb: linear rich emitter color, w: packed emitter voxel flags/radius
};

layout(std430, binding = 4) coherent buffer LightListBuffer {
    uint lightCount;
    uint pad0;
    uint pad1;
    uint pad2;
    SceneLight lights[];
} lightList;

// ---- Coordinate Space Helpers ----
vec2 screenSize() { return vec2(viewWidth, viewHeight); }
vec2 texelSize()  { return vec2(1.0 / viewWidth, 1.0 / viewHeight); }

vec3 screenToView(vec3 screenPos) {
    vec3 ndc = screenPos * 2.0 - 1.0;
    #ifdef TAA_ENABLED
        ndc.xy -= taaJitter;
    #endif
    vec4 vp = gbufferProjectionInverse * vec4(ndc, 1.0);
    return vp.xyz / vp.w;
}

vec3 viewToScreen(vec3 viewPos) {
    vec4 cp = gbufferProjection * vec4(viewPos, 1.0);
    vec3 ndc = cp.xyz / cp.w;
    #ifdef TAA_ENABLED
        ndc.xy += taaJitter;
    #endif
    return ndc * 0.5 + 0.5;
}

vec3 viewToWorld(vec3 viewPos) {
    return (gbufferModelViewInverse * vec4(viewPos, 1.0)).xyz;
}

vec3 worldToView(vec3 worldPos) {
    return (gbufferModelView * vec4(worldPos, 1.0)).xyz;
}

vec3 worldToVoxelPos(vec3 worldPos) {
    return worldPos + cameraPositionFract + vec3(VOXEL_RES_XZ * 0.5, VOXEL_RES_Y * 0.5, VOXEL_RES_XZ * 0.5);
}

vec3 voxelToWorldPos(vec3 voxelPos) {
    return voxelPos - cameraPositionFract - vec3(VOXEL_RES_XZ * 0.5, VOXEL_RES_Y * 0.5, VOXEL_RES_XZ * 0.5);
}

bool inVoxelBounds(ivec3 v) {
    return all(greaterThanEqual(v, ivec3(0))) && all(lessThan(v, ivec3(VOXEL_RES_XZ, VOXEL_RES_Y, VOXEL_RES_XZ)));
}

// ---- Shadow Space Distortion (standard radial focus near player) ----
float shadowDistortFactor(vec2 xy) {
    return 0.85 * length(xy) + 0.15;
}

vec2 distortShadowXY(vec2 xy) {
    return xy / shadowDistortFactor(xy);
}

vec3 worldToShadowScreen(vec3 worldPos) {
    vec4 sp = shadowProjection * (shadowModelView * vec4(worldPos, 1.0));
    sp.xy = distortShadowXY(sp.xy);
    sp.z *= 0.5;
    return sp.xyz * 0.5 + 0.5;
}

// ---- Sun / Sky Directions ----
vec3 getShadowLightDir() {
    return normalize(mat3(gbufferModelViewInverse) * shadowLightPosition);
}

vec3 getSunDir() {
    return normalize(mat3(gbufferModelViewInverse) * sunPosition);
}

bool isNight() {
    return sunAngle > 0.5;
}

// ---- Material Packing / Unpacking (G-Buffer) ----
struct GBufferMat {
    vec3  albedo;
    vec3  geoNormal;
    vec3  shadeNormal;
    float roughness;
    float f0;
    float emission;
    float skyLight;
    bool  isHand;
};

uvec2 packGBuffer0(vec3 albedoSrgb, float roughness, float f0, float emission, float skyLight, bool isHand) {
    uint w0 = packUnorm4x8(vec4(saturate(albedoSrgb), saturate(skyLight)));
    uint r8 = uint(round(saturate(roughness) * 255.0));
    uint f8 = uint(round(saturate(f0) * 255.0));
    uint e8 = uint(round(saturate(emission) * 255.0));
    uint fl = isHand ? 1u : 0u;
    uint w1 = r8 | (f8 << 8u) | (e8 << 16u) | (fl << 24u);
    return uvec2(w0, w1);
}

uint packGBuffer1(vec3 geoNormal, vec3 shadeNormal) {
    return packNormal16(geoNormal) | (packNormal16(shadeNormal) << 16u);
}

GBufferMat readGBuffer(ivec2 pixel) {
    uvec2 g0 = texelFetch(colortex1, pixel, 0).rg;
    uint  g1 = texelFetch(colortex2, pixel, 0).r;
    vec4  c0 = unpackUnorm4x8(g0.x);

    GBufferMat m;
    m.albedo      = srgbToLinear(c0.rgb);
    m.skyLight    = c0.a;
    m.roughness   = float(g0.y & 255u) * (1.0 / 255.0);
    m.f0          = float((g0.y >> 8u) & 255u) * (1.0 / 255.0);
    m.emission    = float((g0.y >> 16u) & 255u) * (1.0 / 255.0);
    m.isHand      = ((g0.y >> 24u) & 1u) != 0u;
    m.geoNormal   = unpackNormal16(g1 & 65535u);
    m.shadeNormal = unpackNormal16(g1 >> 16u);
    return m;
}

// ---- LabPBR 1.3 Specular Decode ----
// Stores perceptual roughness (1 - smoothness) so GGX squares it ONCE (alpha = r^2),
// and preserves raw f0 byte (0..255) so hardcoded/albedo-tinted metals keep full RGB F0.
void decodeLabPBR(vec4 specTex, out float perceptualRoughness, out float f0Raw, out float emission) {
    perceptualRoughness = saturate(1.0 - specTex.r);
    f0Raw = saturate(specTex.g);
    emission = (specTex.a > (0.5 / 255.0) && specTex.a < (254.5 / 255.0)) ? specTex.a : 0.0;
}

bool isLabPBRMetal(float f0Raw) {
    return f0Raw >= (229.5 / 255.0);
}

vec3 evalLabPBRF0(float f0Raw, vec3 albedoLinear) {
    int code = int(round(saturate(f0Raw) * 255.0));
    if (code < 230) {
        return vec3(max(f0Raw, 0.04));
    }
    if (code == 230) return vec3(0.562, 0.565, 0.578); // Iron
    if (code == 231) return vec3(1.000, 0.766, 0.336); // Gold
    if (code == 232) return vec3(0.913, 0.921, 0.925); // Aluminum
    if (code == 233) return vec3(0.550, 0.556, 0.554); // Chrome
    if (code == 234) return vec3(0.955, 0.637, 0.538); // Copper
    if (code == 235) return vec3(0.632, 0.626, 0.641); // Lead
    if (code == 236) return vec3(0.672, 0.637, 0.585); // Platinum
    if (code == 237) return vec3(0.972, 0.960, 0.915); // Silver
    return albedoLinear;                               // 238..255: Albedo-tinted metal
}

#endif
