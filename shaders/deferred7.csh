#version 430 compatibility
#include "/lib/LightFilter.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
layout(rgba16f) uniform writeonly image2D diffuseWorkImg, diffuseHistoryImg;
#ifdef SPECULAR_REFLECTIONS
layout(rgba16f) uniform writeonly image2D glossWorkImg, glossHistoryImg;
#endif

// Shared 12 x 12 neighborhood for the 5 x 5 clamping statistics of an 8 x 8 group.
#ifdef SPECULAR_REFLECTIONS
const int filterSignals=2;
#else
const int filterSignals=1;
#endif
shared vec4 localRaw[filterSignals][144];
shared vec3 localResponsive[filterSignals][144];
void cacheNeighborhood(sampler2D work,int signalIndex) {
    for(int i=int(gl_LocalInvocationIndex);i<144;i+=64) {
        ivec2 q=ivec2(gl_WorkGroupID.xy)*8+ivec2(i%12,i/12)-2;
        q=clamp(q,ivec2(0),ivec2(screenSize())-1);
        localRaw[signalIndex][i]=texelFetch(work,workPixel(q,2),0);
        localResponsive[signalIndex][i]=rgbToYCoCg(texelFetch(work,workPixel(q,3),0).rgb);
    }
}

void adjustHistory(ivec2 p,FilterSurface s,sampler2D work,bool glossy,out vec4 slow,out vec4 fast,out vec4 seed) {
    slow=texelFetch(work,p,0); fast=texelFetch(work,workPixel(p,3),0);
    seed=vec4(0,0,0,-1);
    if(s.z<=0.0 || (glossy && s.rough<0.0)) return;
    vec3 mean=vec3(0),m2=vec3(0),rawMean=vec3(0);
    float rawM2=0.0,n=0.0;
    for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
        ivec2 rel=ivec2(gl_LocalInvocationID.xy)+ivec2(x+2,y+2);
        int slot=rel.y*12+rel.x, channel=glossy?1:0;
        vec4 raw=localRaw[channel][slot];
        if(raw.a<0.0) continue;
        vec3 v=localResponsive[channel][slot];
        mean+=v;m2+=v*v;
        rawMean+=raw.rgb;float yRaw=luma(raw.rgb);rawM2+=yRaw*yRaw;
        n+=1.0;
    }
    if(n<=0.0) return;
    mean/=n;rawMean/=n;
    vec3 sigma=sqrt(max(vec3(0),m2/n-mean*mean));
    vec3 old=slow.rgb, responsive=fast.rgb;
    vec3 oldYCC=rgbToYCoCg(old), fastYCC=rgbToYCoCg(responsive);
    vec3 bounded=clampResponsive(oldYCC,fastYCC,mean,sigma);
    bool young=slow.a<=4.0;
    float clipFactor=abs(fastYCC.x-oldYCC.x)>1e-6 ? clamp((bounded.x-oldYCC.x)/(fastYCC.x-oldYCC.x),0.0,1.0) : 0.0;
    slow.rgb=young?responsive:yCoCgToRgb(bounded);
    // Displace both estimates toward the neighborhood's current energy; bound
    // the displacement by the energy still separating it from the fast estimate.
    vec3 residual=rawMean-responsive;
    float remaining=luma(abs(residual));
    float acceleration=(glossy?0.99:3.0)*luma(abs(responsive-old))*(young?0.0:clipFactor);
    vec3 change=residual*min(1.0,acceleration/max(remaining,1e-6));
    slow.rgb+=change;fast.rgb+=change;
    vec4 center=texelFetch(work,workPixel(p,2),0);
    float noiseSigma=sqrt(signalVariance(luma(rawMean),rawM2/n));
    float reset=center.a>=0.0?resetFraction(luma(old),luma(rawMean),sigma.x,noiseSigma):0.0;
    slow.rgb=max(mix(slow.rgb,center.rgb,reset),vec3(0));
    fast.rgb=max(mix(fast.rgb,center.rgb,reset),vec3(0));
    float variance=signalVariance(luma(old),fast.a);
    fast.a=variance+pow(luma(slow.rgb),2.0);
    if(slow.a<=0.0) return;
    seed=vec4(slow.rgb,variance);
}

void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    cacheNeighborhood(diffuseWorkTex,0);
    #ifdef SPECULAR_REFLECTIONS
        cacheNeighborhood(glossWorkTex,1);
    #endif
    barrier();
    if(!inFrame(p)) return;
    FilterSurface s=filterSurface(p);
    vec4 slow,fast,seed;
    adjustHistory(p,s,diffuseWorkTex,false,slow,fast,seed);
    imageStore(diffuseHistoryImg,p,slow);imageStore(diffuseHistoryImg,workPixel(p,1),fast);
    imageStore(diffuseWorkImg,workPixel(p,1),seed);
    #ifdef SPECULAR_REFLECTIONS
        adjustHistory(p,s,glossWorkTex,true,slow,fast,seed);
        imageStore(glossHistoryImg,p,slow);imageStore(glossHistoryImg,workPixel(p,1),fast);
        imageStore(glossWorkImg,workPixel(p,1),seed);
    #endif
}
