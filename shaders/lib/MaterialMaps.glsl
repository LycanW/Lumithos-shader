#ifndef LIB_MATERIAL_MAPS_GLSL
#define LIB_MATERIAL_MAPS_GLSL

// Filter smoothness but preserve discrete metal IDs and emission codes.
vec4 sampleLabPBRSpec(sampler2D atlas, vec2 uv) {
    ivec2 size = textureSize(atlas, 0);
    ivec2 px = clamp(ivec2(fract(uv) * vec2(size)), ivec2(0), size - 1);
    vec4 material = texelFetch(atlas, px, 0);
    material.r = texture(atlas, uv).r;
    return material;
}

vec3 applyLabPBRNormal(vec3 baseNormal, vec4 tangent, vec2 normalRG) {
    vec2 xy = normalRG * 2.0 - 1.0;
    vec3 T = tangent.xyz - baseNormal * dot(tangent.xyz, baseNormal);
    if (dot(xy, xy) <= 1e-4 || dot(T, T) <= 1e-8) return baseNormal;
    T = normalize(T);
    vec3 B = cross(baseNormal, T) * (tangent.w < 0.0 ? -1.0 : 1.0);
    return normalize(T * xy.x + B * xy.y + baseNormal * sqrt(max(0.0, 1.0 - dot(xy, xy))));
}

#endif
