# Lumithos-shader

A real-time hybrid path-tracing shaderpack for Minecraft (Iris + Sodium, OpenGL 4.6).

## Architecture Overview

- **Dual-Track Hybrid Voxel Scene (`shadow.gsh`, `lib/SceneGeo.glsl`, `lib/Raytrace.glsl`)**
  - **Track 1**: Lock-free direct-addressed 6-face full-cube descriptor with exact 8-way UV rotation/mirroring.
  - **Track 2**: Compact non-cube triangle primitive pool for stairs, slabs, torches, ladders, fences, and custom/mod models.
  - **64-Way Hierarchical DDA**: 16×16×16 super-brick → 4×4×4 brick → strictly 6-connected 1×1×1 voxel traversal.
- **3-Cascade Toroidal 3D Clipmap Radiance Cache (`lib/RadianceCache.glsl`, `prepare3.csh`)**
  - O(1) L1 SH + YCoCg probe lookup, deduplicated active-set worklist, and vMF path guiding (`deferred.csh`).
- **Diffuse & Specular Denoising + TAA (`deferred2.csh`–`deferred14.csh`, `composite.csh`)**
  - Variance-aware SVGF temporal accumulation and separable / à-trous spatial filtering.
  - Deterministic zero-ghosting pure mirror reflections with virtual-depth parallax TAA reprojection.
- **LabPBR 1.3 Material Support (`program/GBufferOpaque.glsl`, `lib/Resources.glsl`)**
  - Perceptual roughness, metallic RGB F0, normal maps, and physically scaled block/PBR emission.
