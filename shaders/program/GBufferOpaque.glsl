#include "/lib/Resources.glsl"

#ifdef VSH
    in vec4 at_tangent;
    #ifdef STAGE_TERRAIN
        in vec4 at_midBlock;
        in vec2 mc_Entity;
    #endif

    out vec2 vTexCoord;
    out vec3 vColor;
    out vec3 vGeoNormal;
    out vec4 vTangent;
    out float vSkyLight;
    out float vBlockEmit;
    out float vIsFoliage;

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
            vIsFoliage = (abs(mc_Entity.x - 10002.0) < 0.5) ? 1.0 : 0.0;
        #else
            vBlockEmit = 0.0;
            vIsFoliage = 0.0;
        #endif
    }
#endif

#ifdef FSH
    #include "/lib/MaterialMaps.glsl"
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
    in float vIsFoliage;

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
        // Snap strictly axis-aligned block faces to exact canonical axes to eliminate
        // camera-rotation floating-point noise from (gbufferModelViewInverse * gl_NormalMatrix).
        vec3 absGeoN = abs(geoNormal);
        if (maxOf(absGeoN) > 0.9999) {
            geoNormal = (absGeoN.x >= absGeoN.y && absGeoN.x >= absGeoN.z)
                ? vec3(geoNormal.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0)
                : ((absGeoN.y >= absGeoN.z)
                    ? vec3(0.0, geoNormal.y >= 0.0 ? 1.0 : -1.0, 0.0)
                    : vec3(0.0, 0.0, geoNormal.z >= 0.0 ? 1.0 : -1.0));
        }

        vec3 shadeNormal = geoNormal;
        float matAO = 1.0;
        #if defined(MC_NORMAL_MAP) && !defined(STAGE_BASIC)
            vec4 normTex = texture(normals, vTexCoord);
            shadeNormal = applyLabPBRNormal(geoNormal, vTangent, normTex.rg);
            // LabPBR 1.3 normals.b encodes micro-surface AO (guard near-zero empty channels).
            if (normTex.b > 0.01) {
                matAO = saturate(normTex.b);
            }
        #endif

        float roughness = 0.92;
        float f0 = 0.04;
        float porosity = 0.0;
        float sss = 0.0;
        float pbrEmit = 0.0;
        #if defined(MC_SPECULAR_MAP) && !defined(STAGE_BASIC)
            vec4 specTex = sampleLabPBRSpec(specular, vTexCoord);
            decodeLabPBR(specTex, roughness, f0, porosity, sss, pbrEmit);
        #endif

        // Automatic foliage/plant SSS fallback when resource pack doesn't encode specular.b SSS.
        if (vIsFoliage > 0.5 && sss < 0.01) {
            sss = 0.65;
        }

        bool isHand = false;
        #ifdef STAGE_HAND
            isHand = true;
        #endif

        // LabPBR 1.3 rain wetness & porosity: porous sky-exposed blocks darken albedo and smooth roughness.
        if (!isHand && wetness > 1e-3 && !isLabPBRMetal(f0)) {
            float wet = wetness * smoothstep(0.80, 0.96, vSkyLight) * smoothstep(0.15, 0.70, geoNormal.y);
            if (wet > 1e-3) {
                vec3 linAlb = srgbToLinear(albedoSrgb);
                linAlb *= mix(1.0, 0.48, wet * porosity);
                albedoSrgb = linearToSrgb(linAlb);
                roughness = mix(roughness, 0.06, wet * (1.0 - 0.55 * porosity));
            }
        }

        float maxAlb = maxOf(srgbToLinear(albedoSrgb));
        float emitMask = (pbrEmit > 0.0) ? 1.0 : smoothstep(0.12, 0.42, maxAlb);
        float emission = max(vBlockEmit * emitMask, pbrEmit);

        gbuf0Out = packGBuffer0(albedoSrgb, roughness, f0, emission, vSkyLight, sss, matAO, isHand);
        gbuf1Out = packGBuffer1(geoNormal, shadeNormal);
    }
#endif
