#include "/lib/LightFilter.glsl"
layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

#ifndef GLOSS_FILTER_STEP
    #define GLOSS_FILTER_STEP FILTER_STEP
#endif

#ifdef FILTER_RESOLVE
layout(rgba32f) uniform writeonly image2D exposureImg;
layout(rgba16f) uniform writeonly image2D colorimg3, colorimg4;
#else
layout(rgba16f) uniform writeonly image2D diffuseWorkImg;
#ifdef SPECULAR_REFLECTIONS
layout(rgba16f) uniform writeonly image2D glossWorkImg;
#endif
#endif

#if FILTER_STEP == 1
vec4 firstVariance(ivec2 p, FilterSurface s, FilterPlane plane, sampler2D work, bool glossy, float age, vec4 center, out float localVariance) {
    ivec2 src=workPixel(p,FILTER_SOURCE);
    localVariance=max(center.a,0.0);
        if (age < 3.0) {
            vec3 sum = vec3(0.0);
            float total = 0.0, second = 0.0;
            for (int y=-2;y<=2;++y) for (int x=-2;x<=2;++x) {
                ivec2 delta=ivec2(x,y),q=p+delta;
                if (!inFrame(q)) continue;
                vec4 v=texelFetch(work,src+delta,0);
                if(v.a<0.0) continue;
                FilterSurface n=filterSurface(q);
                float w=surfaceWeight(s,n,plane,delta);
                if(glossy && n.rough<0.0) continue;
                sum+=w*v.rgb;second+=w*(v.a+pow(luma(v.rgb),2.0));total+=w;
            }
            if(total<=1e-5) return center;
            vec3 mean=sum/total;
            return vec4(mean,signalVariance(luma(mean),second/total)*max(1.0,4.0/(age+1.0)));
        }
        float sumVar=0.0, weightVar=0.0;
        for (int y=-1;y<=1;++y) for (int x=-1;x<=1;++x) {
            ivec2 q=clamp(p+ivec2(x,y),ivec2(0),ivec2(screenSize())-1);
            float v=texelFetch(work,workPixel(q,FILTER_SOURCE),0).a;
            if(v<0.0) continue;
            float w=float((2-abs(x))*(2-abs(y)));
            sumVar+=w*v;weightVar+=w;
        }
        localVariance=sumVar/max(weightVar,1e-5);

    return center;
}
#endif

struct WaveletSum { vec4 weighted; float weight, centerY, sensitivity; };
WaveletSum startSum(vec4 center,float variance,float sensitivity) {
    float w=center.a>=0.0?4.0:0.0;
    return WaveletSum(vec4(w*center.rgb,w*w*max(center.a,0.0)),w,luma(center.rgb),
                      center.a>=0.0?sensitivity/max(1e-4,sqrt(variance)):0.0);
}
void addTap(inout WaveletSum result,vec4 tap,float weight) {
    if(tap.a<0.0) return;
    float w=weight*exp(-result.sensitivity*abs(luma(tap.rgb)-result.centerY));
    result.weight+=w;
    result.weighted+=vec4(w*tap.rgb,w*w*tap.a);
}
#ifdef SPECULAR_REFLECTIONS
void addGlossTap(inout WaveletSum result,ivec2 tap,FilterSurface s,FilterSurface neighbor,float weight,float confidence) {
    float cosine=mix(dot(s.normal,neighbor.normal),1.0,0.3*(1.0-confidence));
    addTap(result,texelFetch(glossWorkTex,tap,0),weight*glossWeight(s.rough,neighbor.rough,cosine));
}
#endif
vec4 resolveSum(WaveletSum result,vec4 fallback) {
    float w=result.weight;
    return w>1e-4 ? result.weighted/vec4(vec3(w),w*w) : fallback;
}
void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if (!inFrame(p)) return;
    FilterSurface s=filterSurface(p);
    FilterPlane plane=filterPlane(p,s.face);
    ivec2 src=workPixel(p,FILTER_SOURCE);
    vec4 diffuse=texelFetch(diffuseWorkTex,src,0);
    float dv=max(diffuse.a,0.0);
    #ifdef SPECULAR_REFLECTIONS
        vec4 gloss=texelFetch(glossWorkTex,src,0);
        float sv=max(gloss.a,0.0);
    #endif
    bool filterNeighbors=s.z>0.0;
    #if FILTER_STEP == 1
        float age=texelFetch(diffuseHistoryTex,p,0).a;
        if (filterNeighbors) {
            diffuse=firstVariance(p,s,plane,diffuseWorkTex,false,age,diffuse,dv);
            #ifdef SPECULAR_REFLECTIONS
                if(s.rough>=0.0) gloss=firstVariance(p,s,plane,glossWorkTex,true,age,gloss,sv);
            #endif
        }
        filterNeighbors=filterNeighbors && age>=3.0;
    #endif
    if(filterNeighbors) {
        WaveletSum d=startSum(diffuse,dv,0.5);
        #ifdef SPECULAR_REFLECTIONS
            float confidence=texelFetch(glossHistoryTex,workPixel(p,2+(frameCounter&1)),0).z;
            WaveletSum g=startSum(gloss,sv,confidence);
        #endif
        for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
            if(x==0 && y==0) continue;
            ivec2 offset=ivec2(x,y);
            float kernel=float((2-abs(x))*(2-abs(y)));
            ivec2 delta=offset*FILTER_STEP,q=p+delta;
            if(inFrame(q)) {
                FilterSurface neighbor=filterSurface(q);
                float w=kernel*surfaceWeight(s,neighbor,plane,delta);
                addTap(d,texelFetch(diffuseWorkTex,src+delta,0),w);
                #if defined(SPECULAR_REFLECTIONS) && FILTER_STEP == GLOSS_FILTER_STEP
                    if(s.rough>=0.0) addGlossTap(g,src+delta,s,neighbor,w,confidence);
                #endif
            }
            #if defined(SPECULAR_REFLECTIONS) && FILTER_STEP != GLOSS_FILTER_STEP
                ivec2 glossDelta=offset*GLOSS_FILTER_STEP,glossPixel=p+glossDelta;
                if(s.rough>=0.0 && inFrame(glossPixel)) {
                    FilterSurface neighbor=filterSurface(glossPixel);
                    float w=kernel*surfaceWeight(s,neighbor,plane,glossDelta);
                    addGlossTap(g,src+glossDelta,s,neighbor,w,confidence);
                }
            #endif
        }
        diffuse=resolveSum(d,diffuse);
        #ifdef SPECULAR_REFLECTIONS
            if(s.rough>=0.0) gloss=resolveSum(g,gloss);
        #endif
    }
    #ifdef FILTER_RESOLVE
        imageStore(colorimg3, p, vec4(diffuse.rgb,1.0));
        #ifdef SPECULAR_REFLECTIONS
            // Mirrors bypass filtering. TAA consumes the original optical distance.
            vec4 raw = texelFetch(colortex4,p,0);
            float distance = raw.a < 0.0 ? 0.0 : (raw.a == 0.0 || raw.a >= 65504.0 ? 128.0 : raw.a);
            imageStore(colorimg4,p,vec4(s.rough < 0.0 ? raw.rgb : gloss.rgb,s.rough < 0.0 ? distance : 0.0));
        #endif
        if (all(equal(p,ivec2(0))))
            imageStore(exposureImg,ivec2(1,0),vec4(float(frameCounter+1), screenSize(),0.0));
    #else
        imageStore(diffuseWorkImg,workPixel(p,FILTER_DEST),diffuse);
        #ifdef SPECULAR_REFLECTIONS
            imageStore(glossWorkImg,workPixel(p,FILTER_DEST),gloss);
        #endif
    #endif
}
