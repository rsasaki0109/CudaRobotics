#pragma once

// Internal normal-equation assembly. Every valid correspondence contributes
// the same 30 terms; block partials avoid contended global atomic additions.
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "cuda_check.cuh"

namespace kiss_reduction {

struct Equation { float value[30]; };

__device__ inline Equation contribution(const float* pw,const float* q,
                                         const float* nq,const float* d2,
                                         int i,int n,float k2) {
    Equation e{};
    if(i>=n || d2[i]>1e29f) return e;
    const float px=pw[3*i],py=pw[3*i+1],pz=pw[3*i+2];
    const float nx=nq[3*i],ny=nq[3*i+1],nz=nq[3*i+2];
    const float rs=nx*(px-q[3*i])+ny*(py-q[3*i+1])+nz*(pz-q[3*i+2]);
    const float gm=k2/(k2+d2[i]),w=gm*gm;
    const float j[18]={1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float jp[6];
    for(int a=0;a<6;++a) jp[a]=nx*j[a]+ny*j[6+a]+nz*j[12+a];
    int c=0;
    for(int a=0;a<6;++a) for(int b=a;b<6;++b) e.value[c++]=w*jp[a]*jp[b];
    for(int a=0;a<6;++a) e.value[21+a]=w*jp[a]*rs;
    e.value[27]=w*rs*rs; e.value[28]=w; e.value[29]=1.f;
    return e;
}

struct Add {
    __device__ Equation operator()(const Equation& a,const Equation& b) const {
        Equation sum;
        #pragma unroll
        for(int c=0;c<30;++c) sum.value[c]=a.value[c]+b.value[c];
        return sum;
    }
};

static __global__ void atomic_sum(const float* pw,const float* q,const float* nq,
                                  const float* d2,int n,float k2,float* hg) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || d2[i]>1e29f) return;
    const Equation e=contribution(pw,q,nq,d2,i,n,k2);
    for(int c=0;c<30;++c) atomicAdd(hg+c,e.value[c]);
}

static __global__ void block_sum(const float* pw,const float* q,const float* nq,
                                 const float* d2,int n,float k2,float* partials) {
    using Reduce=cub::BlockReduce<Equation,256>;
    __shared__ typename Reduce::TempStorage storage;
    const int i=blockIdx.x*256+threadIdx.x;
    const Equation sum=Reduce(storage).Reduce(contribution(pw,q,nq,d2,i,n,k2),Add{});
    if(threadIdx.x==0) for(int c=0;c<30;++c) partials[blockIdx.x*30+c]=sum.value[c];
}

static __global__ void finish_sum(const float* partials,int blocks,float* hg) {
    const int c=threadIdx.x;
    if(c>=30) return;
    float sum=0.f;
    for(int b=0;b<blocks;++b) sum+=partials[b*30+c];
    hg[c]=sum;
}

inline void launch(const float* pw,const float* q,const float* nq,const float* d2,
                   int n,float k2,float* hg,float* partials,bool use_atomic) {
    const int blocks=(n+255)/256;
    if(use_atomic) {
        CUDA_CHECK(cudaMemset(hg,0,30*sizeof(float)));
        if(blocks) atomic_sum<<<blocks,256>>>(pw,q,nq,d2,n,k2,hg);
    } else {
        if(blocks) block_sum<<<blocks,256>>>(pw,q,nq,d2,n,k2,partials);
        finish_sum<<<1,32>>>(partials,blocks,hg);
    }
}

} // namespace kiss_reduction
