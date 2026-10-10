#include "kiss_icp_reduction.cuh"
#include "kiss_icp_order.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool check(int n,bool all_invalid) {
    const int capacity=n ? n : 1;
    std::vector<float> p(3*capacity),q(3*capacity),normal(3*capacity),distance(capacity);
    std::mt19937 random(8142+n);
    std::uniform_real_distribution<float> coord(-8.f,8.f),delta(-.15f,.15f);
    double expected[30]={},magnitude[30]={};
    int count=0;
    for(int i=0;i<n;++i) {
        float length=0.f,d2=0.f;
        for(int a=0;a<3;++a) {
            p[3*i+a]=coord(random); q[3*i+a]=p[3*i+a]+delta(random);
            normal[3*i+a]=coord(random); length+=normal[3*i+a]*normal[3*i+a];
            const float d=p[3*i+a]-q[3*i+a]; d2+=d*d;
        }
        for(int a=0;a<3;++a) normal[3*i+a]/=std::sqrt(length);
        distance[i]=(all_invalid || i%7==0) ? 1e30f : d2;
        if(distance[i]>1e29f) continue;
        ++count;
        const double x=p[3*i],y=p[3*i+1],z=p[3*i+2];
        const double nx=normal[3*i],ny=normal[3*i+1],nz=normal[3*i+2];
        const double jacobian[6]={nx,ny,nz,y*nz-z*ny,z*nx-x*nz,x*ny-y*nx};
        double residual=0.;
        for(int a=0;a<3;++a) residual+=static_cast<double>(normal[3*i+a])*(static_cast<double>(p[3*i+a])-q[3*i+a]);
        const double gm=.25/(.25+distance[i]),weight=gm*gm;
        double terms[30]={}; int c=0;
        for(int a=0;a<6;++a) for(int b=a;b<6;++b) terms[c++]=weight*jacobian[a]*jacobian[b];
        for(int a=0;a<6;++a) terms[21+a]=weight*jacobian[a]*residual;
        terms[27]=weight*residual*residual; terms[28]=weight; terms[29]=1.;
        for(int a=0;a<30;++a) { expected[a]+=terms[a]; magnitude[a]+=std::fabs(terms[a]); }
    }
    float *dp=nullptr,*dq=nullptr,*dn=nullptr,*dd=nullptr,*hg=nullptr,*partials=nullptr;
    CUDA_CHECK(cudaMalloc(&dp,p.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&dq,q.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dn,normal.size()*sizeof(float))); CUDA_CHECK(cudaMalloc(&dd,distance.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hg,30*sizeof(float))); CUDA_CHECK(cudaMalloc(&partials,((capacity+255)/256)*30*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dp,p.data(),p.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dn,normal.data(),normal.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd,distance.data(),distance.size()*sizeof(float),cudaMemcpyHostToDevice));
    bool ok=true;
    for(bool atomic:{false,true}) {
        float output[30],repeat[30];
        kiss_reduction::launch(dp,dq,dn,dd,n,.25f,hg,partials,atomic);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(output,hg,sizeof(output),cudaMemcpyDeviceToHost));
        for(int c=0;c<30;++c) {
            const double tolerance=(atomic ? 2e-4 : 1e-5)*magnitude[c]+2e-4;
            if(!std::isfinite(output[c]) || std::fabs(output[c]-expected[c])>tolerance) {
                std::fprintf(stderr,"normal equation mismatch n=%d atomic=%d term=%d actual=%g expected=%g\n",n,atomic,c,output[c],expected[c]);
                ok=false;
            }
        }
        if(output[29]!=count) ok=false;
        if(!atomic) {
            kiss_reduction::launch(dp,dq,dn,dd,n,.25f,hg,partials,false);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(repeat,hg,sizeof(repeat),cudaMemcpyDeviceToHost));
            if(std::memcmp(output,repeat,sizeof(output))) ok=false;
        }
    }
    cudaFree(dp); cudaFree(dq); cudaFree(dn); cudaFree(dd); cudaFree(hg); cudaFree(partials);
    return ok;
}

static bool check_gather() {
    constexpr int n=257;
    std::vector<int> order(n);
    std::vector<float> points(3*n),expected(3*n),actual(3*n);
    std::mt19937 random(47);
    for(int i=0;i<n;++i) {
        order[i]=i;
        for(int a=0;a<3;++a) points[3*i+a]=static_cast<float>(i*7+a)/13.f;
    }
    points[0]=-0.f;
    std::shuffle(order.begin(),order.end(),random);
    for(int i=0;i<n;++i) for(int a=0;a<3;++a) expected[3*i+a]=points[3*order[i]+a];
    float *input=nullptr,*output=nullptr; int* indices=nullptr;
    CUDA_CHECK(cudaMalloc(&input,points.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output,actual.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&indices,order.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(input,points.data(),points.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(indices,order.data(),order.size()*sizeof(int),cudaMemcpyHostToDevice));
    kiss_order::gather<<<2,256>>>(input,indices,n,output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaFree(input); cudaFree(output); cudaFree(indices);
    return !std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float));
}

int main() {
    if(!check_gather()) return 2;
    for(int n:{0,1,31,255,256,257,4099,131071})
        if(!check(n,false) || !check(n,true)) return 1;
    std::puts("KISS-ICP normal equations match the double-precision CPU reference: PASS");
    return 0;
}
