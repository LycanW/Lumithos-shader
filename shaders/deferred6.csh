#version 430 compatibility
#include "/lib/LightFilter.glsl"

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
layout(rgba16f) uniform writeonly image2D diffuseWorkImg;
#ifdef SPECULAR_REFLECTIONS
layout(rgba16f) uniform writeonly image2D glossWorkImg;
#endif

vec4 repairShortHistory(ivec2 p,FilterSurface s,FilterPlane plane,sampler2D work,bool glossy) {
    vec4 slow=texelFetch(work,p,0),fast=texelFetch(work,workPixel(p,1),0);
    if(s.z<=0.0 || (glossy && s.rough<0.0) || slow.a>4.0) return fast;
    int stride=int(round(14.0/(1.0+max(slow.a,1.0))));
    vec3 total=slow.rgb;
    float sum=slow.a>0.0?1.0:0.0;
    for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
        if(x==0 && y==0) continue;
        ivec2 offset=ivec2(x,y)*stride,q=p+offset;
        if(!inFrame(q)) continue;
        vec4 sampleValue=texelFetch(work,q,0);
        if(sampleValue.a<=0.0) continue;
        FilterSurface n=filterSurface(q);
        float w=surfaceWeight(s,n,plane,offset);
        if(glossy) w*=glossWeight(s.rough,n.rough,dot(s.normal,n.normal));
        total+=w*sampleValue.rgb;sum+=w;
    }
    return vec4(sum>0.0?total/sum:fast.rgb,fast.a);
}

void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(!inFrame(p)) return;
    FilterSurface s=filterSurface(p);
    FilterPlane plane=filterPlane(p,s.face);
    imageStore(diffuseWorkImg,workPixel(p,3),repairShortHistory(p,s,plane,diffuseWorkTex,false));
    #ifdef SPECULAR_REFLECTIONS
        imageStore(glossWorkImg,workPixel(p,3),repairShortHistory(p,s,plane,glossWorkTex,true));
    #endif
}
