#version 430 compatibility
#include "/lib/LightFilter.glsl"
#ifdef ENTITY_REFLECTION_HISTORY
    #include "/lib/GlossLobe.glsl"
#endif
#if DIFFUSE_HISTORY_MODE == 2
uniform float diffuseHistoryFrameTime;
#endif

layout(local_size_x = 8, local_size_y = 8) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);
uniform sampler2D prevGuideTex, exposureTex;
layout(rgba16f) uniform writeonly image2D diffuseWorkImg;
#ifdef SPECULAR_REFLECTIONS
layout(rgba16f) uniform writeonly image2D glossWorkImg, glossHistoryImg;
#endif

struct Footprint { ivec2 origin; vec4 weights, glossWeights; };
struct History { vec4 slow, fast; vec2 distances; float support; };

vec2 projectPrevious(vec3 view,vec2 jitter) {
    vec4 clip=gbufferPreviousProjection*vec4(view,1.0);
    return clip.w>0.0 ? (0.5+0.5*(clip.xy/clip.w+jitter))*screenSize()-0.5 : vec2(-3.0);
}

Footprint previousFootprint(vec2 pos,vec3 receiver,FilterSurface s,vec2 jitter) {
    Footprint b=Footprint(ivec2(floor(pos)),vec4(0.0),vec4(0.0));
    vec3 n=mat3(gbufferPreviousModelView)*s.face;
    vec2 f=fract(pos);
    for(int i=0;i<4;++i) {
        ivec2 k=ivec2(i&1,i>>1), q=b.origin+k;
        if(!inFrame(q)) continue;
        vec4 prev=texelFetch(prevGuideTex,q,0);
        if(prev.w<=0.0 || dot(s.face,unpackNormal16(uint(prev.x)))<0.9) continue;
        #ifdef SPECULAR_REFLECTIONS
            if ((uint(prev.x) >= 65536u) != (s.rough >= 0.0)) continue;
        #endif
        vec2 ndc=(vec2(q)+0.5)*(2.0*texelSize())-1.0-jitter;
        vec3 ray=vec3((ndc+gbufferPreviousProjection[2].xy)/vec2(gbufferPreviousProjection[0][0],gbufferPreviousProjection[1][1]),-1.0);
        float gap=abs(dot(prev.w*ray-receiver,n));
        if(gap>max(0.008,-receiver.z*(0.01+2.0*texelSize().y))) continue;
        float cosine=dot(s.normal,unpackNormal16(uint(prev.y)));
        if(cosine<0.0) continue;
        vec2 axis=mix(1.0-f,f,vec2(k));
        b.weights[i]=axis.x*axis.y;
        #ifdef SPECULAR_REFLECTIONS
            b.glossWeights[i]=axis.x*axis.y*glossWeight(s.rough,prev.z,cosine);
        #endif
    }
    return b;
}

History lookup(sampler2D source,Footprint b,bool glossy) {
    History h=History(vec4(0),vec4(0),vec2(0),0.0);
    for(int i=0;i<4;++i) {
        float w=glossy?b.glossWeights[i]:b.weights[i];
        if(w<=0.0) continue;
        ivec2 q=b.origin+ivec2(i&1,i>>1);
        vec4 slow=texelFetch(source,q,0);
        if(slow.a<=0.0) continue;
        h.slow+=w*slow;
        h.fast+=w*texelFetch(source,workPixel(q,1),0);
        if(glossy) h.distances+=w*texelFetch(source,workPixel(q,3-(frameCounter&1)),0).xy;
        h.support+=w;
    }
    if(h.support>1e-5) {
        h.slow/=h.support; h.fast/=h.support; h.distances/=h.support;
    }
    return h;
}

void updateMeans(History h, vec4 sampleValue, float age, float observation, float confidence, vec2 maxAges,
                 out vec4 slow, out vec4 fast) {
    if(sampleValue.a<0.0) {slow=h.slow;fast=h.fast;return;}
    vec2 weights=h.support>0.0 ? max(vec2(1.0-confidence),1.0/min(vec2(age),maxAges))*observation : vec2(1.0);
    float y=luma(sampleValue.rgb);
    slow=vec4(mix(h.slow.rgb,sampleValue.rgb,weights.x),age);
    fast=vec4(mix(h.fast.rgb,sampleValue.rgb,weights.y),mix(h.fast.a,y*y,weights.x));
}

void main() {
    ivec2 p=ivec2(gl_GlobalInvocationID.xy);
    if(!inFrame(p)) return;
    FilterSurface s=filterSurface(p);
    vec3 view=screenToView(vec3((vec2(p)+0.5)*texelSize(),texelFetch(depthtex1,p,0).r));
    vec3 worldRay=mat3(gbufferModelViewInverse)*view;
    vec3 receiver=(gbufferPreviousModelView*vec4(viewToWorld(view)+cameraVelocity,1)).xyz;
    vec2 jitter=vec2(0);float jitterDelta=0.0;
    #ifdef TAA_ENABLED
        jitter=(fract(float(max(frameCounter-1,0))/vec2(1.3247179572,1.7548776662)+0.5)*2.0-1.0)*texelSize();
        vec2 shift=abs(taaJitter-jitter)*screenSize()*0.5;
        jitterDelta=max(shift.x,shift.y);
    #endif
    bool valid=s.z>0.0 && frameCounter>0 && all(equal(texelFetch(exposureTex,ivec2(1,0),0).xyz,vec3(float(frameCounter),screenSize())));
    vec2 surfacePixel=projectPrevious(receiver,jitter);
    Footprint surface=Footprint(ivec2(0),vec4(0),vec4(0));
    if(valid) surface=previousFootprint(surfacePixel,receiver,s,jitter);
    History diffuse=lookup(diffuseHistoryTex,surface,false);
    vec3 toEye=normalize(-worldRay), oldEye=normalize(-receiver);
    float noV=abs(dot(toEye,s.normal));
    float noVprev=abs(dot(mat3(gbufferPreviousModelView)*s.normal,oldEye));
    float stretch=min(1.0,pow((noVprev+0.001)/(noV+0.001),4.0));
    const vec2 legacyLimits=vec2(31.0,7.0);
    vec2 diffuseLimits=legacyLimits;
    #if DIFFUSE_HISTORY_MODE > 0
        diffuseLimits.x=64.0;
    #endif
    #if DIFFUSE_HISTORY_MODE == 2
        float relativeFps=1.0/(60.0*max(diffuseHistoryFrameTime,0.001));
        vec2 scaledFrames=round((diffuseLimits-1.0)*relativeFps);
        diffuseLimits=1.0+clamp(scaledFrames,vec2(1.0,0.0),vec2(254.0));
    #endif
    float coverage=diffuse.support*mix(0.1,1.0,stretch);
    float age=nextHistoryAge(diffuse.slow.a,coverage,diffuseLimits.x);
    float cadence=15.0*clamp(0.01666/max(frameTime,0.0001),0.25,4.0);
    float inferred=mix(1.0/(1.0+cadence),0.5,jitterDelta);
    float obs=age<=1.0 || diffuseField(p) ? 1.0 : inferred;
    vec4 slow,fast;
    updateMeans(diffuse,texelFetch(diffuseWorkTex,workPixel(p,2),0),age,obs,1.0,diffuseLimits,slow,fast);
    imageStore(diffuseWorkImg,p,slow); imageStore(diffuseWorkImg,workPixel(p,1),fast);
    #ifdef SPECULAR_REFLECTIONS
        vec4 tracking=vec4(0.0);
        slow=vec4(0.0);fast=vec4(0.0);
        if(s.z>0.0 && s.rough>=0.0) {
            float glossAge=nextHistoryAge(min(diffuse.slow.a,legacyLimits.x),coverage,legacyLimits.x);
            vec4 value=texelFetch(glossWorkTex,workPixel(p,2),0);
            History real=lookup(glossHistoryTex,surface,true);
            obs=glossAge<=1.0 || glossTraced(p) ? 1.0 : inferred;
            float dominant=dominantFraction(s.rough,noV);
            vec3 D=normalize(mix(s.normal,reflect(-toEye,s.normal),dominant));
            float distance=max(value.a,0.0);
            float curvature=0.0;
            vec3 direction=mat3(gbufferPreviousModelView)*worldRay;
            vec2 parallax=surfacePixel-projectPrevious(direction,jitter);
            if(valid && dot(parallax,parallax)>0.01) {
                ivec2 delta=ivec2(round(normalize(parallax)));
                if(inFrame(p+delta)) {
                    FilterSurface n=filterSurface(p+delta);
                    vec3 adjacent=screenToView(vec3((vec2(p+delta)+0.5)*texelSize(),texelFetch(depthtex1,p+delta,0).r));
                    vec3 tangent=mat3(gbufferModelViewInverse)*(adjacent-view);
                    if(dot(s.face,n.face)>0.9 && abs(dot(tangent,s.face))<max(0.008,0.003*s.z))
                        curvature=dot(n.normal-s.normal,tangent)/max(dot(tangent,tangent),1e-8);
                }
            }
            float imageDistance=virtualImageDistance(distance,curvature,abs(dot(D,s.normal)),length(view),noV,dominant);
            vec2 virtualPixel=projectPrevious(receiver+normalize(direction)*imageDistance,jitter);
            History virt=real;
            if(valid && length(virtualPixel-surfacePixel)>0.01) virt=lookup(glossHistoryTex,previousFootprint(virtualPixel,receiver,s,jitter),true);
            float aperture=max(0.005,s.rough*s.rough);
            float surfaceConfidence=real.support*exp(-max(0.0,1.0-dot(normalize(-direction),oldEye))/(aperture*aperture));
            float oldDistance=virtualImageDistance(virt.distances.x,curvature,abs(dot(D,s.normal)),length(view),noV,dominant);
            vec2 oldHitPixel=projectPrevious(receiver+normalize(direction)*oldDistance,jitter);
            float lobePixels=max(0.25,0.5*screenSize().y*gbufferProjection[1][1]*aperture*reflectionFraction(distance,length(view)));
            float historyConsistency=abs(reflectionFraction(distance,length(view))-reflectionFraction(virt.distances.y,length(view)));
            float virtualConfidence=virt.support*exp(-historyConsistency/max(0.1,aperture))*(1.0-smoothstep(0.0,lobePixels+0.25,length(oldHitPixel-virtualPixel)));
            float blend=dominant*min(1.0,virtualConfidence/max(surfaceConfidence,1e-5));
            vec4 vSlow,vFast;
            updateMeans(real,value,glossAge,obs,surfaceConfidence,legacyLimits,slow,fast);
            updateMeans(virt,value,glossAge,obs,virtualConfidence,legacyLimits,vSlow,vFast);
            slow=mix(slow,vSlow,blend);fast=mix(fast,vFast,blend);
            float confidence=mix(surfaceConfidence,virtualConfidence,blend);
            float tracked=mix(real.distances.y,virt.distances.y,blend);
            tracking=vec4(distance,mix(tracked,distance,max(0.1,1.0-confidence)),confidence,0.0);
        }
        #ifdef ENTITY_REFLECTION_HISTORY
            else if (s.z > 0.0 && texelFetch(colortex4, p, 0).a >= 0.0) {
                tracking.xy = texelFetch(glossWorkTex, glossHitPixel(p), 0).xy;
                tracking.z = any(notEqual(tracking.xy, vec2(0.0))) ? -1.0 : 0.0;
            }
        #endif
        imageStore(glossHistoryImg,workPixel(p,2+(frameCounter&1)),tracking);
        imageStore(glossWorkImg,p,slow);imageStore(glossWorkImg,workPixel(p,1),fast);
    #endif
}
