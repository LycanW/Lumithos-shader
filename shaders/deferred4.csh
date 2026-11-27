#version 430 compatibility
#include "/lib/LightFilter.glsl"
#include "/lib/GlossLobe.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
layout(rgba16f) uniform writeonly image2D diffuseWorkImg;
#ifdef SPECULAR_REFLECTIONS
layout(rgba16f) uniform writeonly image2D glossWorkImg;
#endif

vec4 prepareLight(ivec2 p, FilterSurface s, FilterPlane plane, sampler2D source, bool glossy, vec3 viewPos) {
    if (s.z <= 0.0 || (glossy && s.rough < 0.0)) return vec4(0,0,0,-1);
    vec4 center = texelFetch(source,p,0);
    if (center.a < 0.0) {
        vec4 sum = vec4(0.0);
        float weight = 0.0;
        for (int x=-1; x<=1; x+=2) {
            ivec2 q = p + ivec2(x,0);
            if (!inFrame(q)) continue;
            vec4 v = texelFetch(source,q,0);
            FilterSurface n = filterSurface(q);
            float w = surfaceWeight(s,n,plane,q-p);
            if (glossy) w *= glossWeight(s.rough,n.rough,dot(s.normal,n.normal));
            if (v.a < 0.0) continue;
            sum += w*v;
            weight += w;
        }
        if (weight <= 1e-5) return vec4(0,0,0,-1);
        center = sum/weight;
    }
    float pixelSize = 2.0*s.z / (gbufferProjection[1][1]*screenSize().y);
    float frustum = pixelSize * min(screenSize().x,screenSize().y);
    float d = center.a == 0.0 ? 1.0 : center.a;
    float radius = 30.0 * min(1.0,d/frustum);
    #if defined(SPECULAR_REFLECTIONS) && SPECULAR_LOBE_MODE != 0
        bool reuseTail = glossy && center.a > 0.0 && center.a < 65504.0;
        mat3 viewToLobe;
        vec2 invTailWidth;
        vec3 faceView;
    #endif
    if (glossy) {
        vec3 toEye = normalize(-(mat3(gbufferModelViewInverse)*viewPos));
        float cosine = abs(dot(s.normal,toEye));
        float dominant = dominantFraction(s.rough,cosine);
        float noD = abs(dot(s.normal,normalize(mix(s.normal,reflect(-toEye,s.normal),dominant))));
        float curve = pow(s.rough,0.25) * (1.0-exp2(-200.0*s.rough*s.rough));
        float lobe = d * noD * s.rough*s.rough / (pixelSize * (1.0 + d*dominant/s.z));
        radius = min(lobe, 50.0 * curve * min(1.0,d*noD/frustum));
        #if defined(SPECULAR_REFLECTIONS) && SPECULAR_LOBE_MODE != 0
            if (reuseTail) {
                vec3 normal = s.normal;
                float noV = dot(normal,toEye);
                if (noV < 0.05) normal = normalize(normal + toEye * (0.05 - noV));
                noV = max(dot(normal,toEye),0.05);
                float tailWidth = 2.0 * GLOSS_TAIL_WIDTH * s.rough * s.rough;
                invTailWidth = 1.0 / max(tailWidth * vec2(noV,1.0),vec2(1e-4));
                viewToLobe = transpose(mat3(gbufferModelView) * glossLobeBasis(reflect(-toEye,normal),toEye));
                faceView = mat3(gbufferModelView) * s.face;
                // Reuse the existing taps across a wider footprint of the virtual reflection.
                radius = min(5.5 * GLOSS_TAIL_WIDTH * lobe, 160.0 * curve * min(1.0,d*noD/frustum));
            }
        #endif
    }
    if (center.a == 0.0) radius = max(radius,1.0);
    vec4 sum = center;
    float total=1.0, nearest=center.a > 0.0 ? center.a : 65504.0;
    bool haveHit = center.a > 0.0;
    // Eight reconstructions of existing texture values on a rotating equal-area disk.
    float phase = 6.2831853 * fract(float(frameCounter)*0.618034 + dot(vec2(p & 3),vec2(0.25,0.0625)));
    for (int i=0; i<8; ++i) {
        float radial = (float(i)+0.5)/8.0;
        float angle = phase + float(i)*2.39996323;
        ivec2 q = p + ivec2(round(radius*sqrt(radial)*vec2(cos(angle),sin(angle))));
        if (!(glossy ? glossTraced(q) : diffuseField(q))) q.x += (i & 1)*2-1;
        if (!inFrame(q) || all(equal(q,p))) continue;
        vec4 v = texelFetch(source,q,0);
        if (v.a < 0.0) continue;
        FilterSurface n = filterSurface(q);
        float w = surfaceWeight(s,n,plane,q-p);
        if (glossy) w *= glossWeight(s.rough,n.rough,dot(s.normal,n.normal));
        #if defined(SPECULAR_REFLECTIONS) && SPECULAR_LOBE_MODE != 0
            if (reuseTail) {
                if (w <= 1e-5 || v.a <= 0.0 || v.a >= 65504.0) continue;
                vec3 toHit = texelFetch(glossWorkTex,glossHitPixel(q),0).xyz - viewPos;
                vec3 angular = viewToLobe * toHit;
                if (angular.z <= 1e-4 || dot(faceView,toHit) <= 0.0) continue;
                vec2 offset = angular.xy * invTailWidth / angular.z;
                float distanceRatio = min(center.a,v.a) / max(center.a,v.a);
                w *= exp(-0.5 * dot(offset,offset)) * pow(distanceRatio,4.0);
            } else
        #endif
        {
            w *= exp(-2.0*radial);
        }
        if (glossy && w>0.01 && v.a>0.0) { nearest=min(nearest,v.a); haveHit=true; }
        #if defined(SPECULAR_REFLECTIONS) && SPECULAR_LOBE_MODE != 0
            if (!reuseTail)
        #endif
        if (center.a>0.0) w *= mix(0.2,1.0,exp(-abs(center.a-v.a)/max(0.01,center.a/9.0)));
        sum += w*v;
        total += w;
    }
    vec4 value = sum/total;
    #if defined(SPECULAR_REFLECTIONS) && SPECULAR_LOBE_MODE != 0
        if (reuseTail) value.rgb = mix(center.rgb,value.rgb,GLOSS_TAIL_MASS);
    #endif
    if (glossy) value.a = haveHit ? nearest : 0.0;
    return value;
}

void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if (!inFrame(p)) return;
    FilterSurface s=filterSurface(p);
    FilterPlane plane=filterPlane(p,s.face);
    vec3 viewPos = screenToView(vec3((vec2(p)+0.5)*texelSize(),texelFetch(depthtex1,p,0).r));
    imageStore(diffuseWorkImg,workPixel(p,2),prepareLight(p,s,plane,diffuseWorkTex,false,viewPos));
    #ifdef SPECULAR_REFLECTIONS
        imageStore(glossWorkImg,workPixel(p,2),prepareLight(p,s,plane,colortex4,true,viewPos));
    #endif
}
