#include "/lib/Resources.glsl"

#ifdef VSH
    in vec4 at_tangent;
    #ifdef STAGE_TERRAIN
        in vec4 at_midBlock;
    #endif

    out vec2 vTexCoord;
    out vec3 vColor;
    out vec3 vGeoNormal;
    out vec4 vTangent;
    out float vSkyLight;
    out float vBlockEmit;

    void main() {
        vec4 clipPos = ftransform();
        #ifdef TAA_ENABLED
            #ifndef STAGE_HAND
                clipPos.xy += clipPos.w * taaJitter;
            #endif
        #endif
        gl_Position = clipPos;

        vTexCoord  = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
        vColor     = gl_Color.rgb;
        vGeoNormal = normalize(mat3(gbufferModelViewInverse) * normalize(gl_NormalMatrix * gl_Normal));
        vTangent   = vec4(normalize(mat3(gbufferModelViewInverse) * normalize(gl_NormalMatrix * at_tangent.xyz)), at_tangent.w);
        vSkyLight  = saturate((gl_MultiTexCoord1.y - 8.0) / 232.0);
        #ifdef STAGE_TERRAIN
            vBlockEmit = saturate(at_midBlock.w / 15.0);
        #else
            vBlockEmit = 0.0;
        #endif
    }
#endif

#ifdef FSH
    uniform sampler2D gtexture;
    uniform sampler2D normals;
    uniform sampler2D specular;
    uniform float alphaTestRef = 0.1;

    in vec2 vTexCoord;
    in vec3 vColor;
    in vec3 vGeoNormal;
    in vec4 vTangent;
    in float vSkyLight;
    in float vBlockEmit;

    /* RENDERTARGETS: 1,2 */
    layout(location = 0) out uvec2 gbuf0Out;
    layout(location = 1) out uint  gbuf1Out;

    void main() {
        #ifdef STAGE_BASIC
            vec4 tex = vec4(1.0);
        #else
            vec4 tex = texture(gtexture, vTexCoord);
        #endif

        if (tex.a < alphaTestRef) discard;

        vec3 albedoSrgb = tex.rgb * vColor;
        vec3 geoNormal = normalize(vGeoNormal);
        if (!gl_FrontFacing) geoNormal = -geoNormal;

        vec3 shadeNormal = geoNormal;
        #if defined(MC_NORMAL_MAP) && !defined(STAGE_BASIC)
            vec2 nXY = texture(normals, vTexCoord).rg * 2.0 - 1.0;
            float nLenSq = dot(nXY, nXY);
            if (nLenSq > 1e-4) {
                vec3 tNorm = vec3(nXY, sqrt(max(0.0, 1.0 - nLenSq)));
                vec3 T = normalize(vTangent.xyz - geoNormal * dot(vTangent.xyz, geoNormal));
                vec3 B = cross(geoNormal, T) * (vTangent.w < 0.0 ? -1.0 : 1.0);
                shadeNormal = normalize(T * tNorm.x + B * tNorm.y + geoNormal * tNorm.z);
            }
        #endif

        float roughness = 0.92;
        float f0 = 0.04;
        float emission = vBlockEmit;
        #if defined(MC_SPECULAR_MAP) && !defined(STAGE_BASIC)
            ivec2 sRes = max(textureSize(specular, 0), ivec2(1));
            ivec2 sPx  = clamp(ivec2(fract(vTexCoord) * vec2(sRes)), ivec2(0), sRes - 1);
            vec4 specTex = texelFetch(specular, sPx, 0);
            float pbrEmit = 0.0;
            decodeLabPBR(specTex, roughness, f0, pbrEmit);
            emission = max(emission, pbrEmit);
        #endif

        bool isHand = false;
        #ifdef STAGE_HAND
            isHand = true;
        #endif

        gbuf0Out = packGBuffer0(albedoSrgb, roughness, f0, emission, vSkyLight, isHand);
        gbuf1Out = packGBuffer1(geoNormal, shadeNormal);
    }
#endif
