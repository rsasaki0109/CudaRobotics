// Compare host and device aggregation of final MPPI rollout diagnostics.
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}while(0)
struct Diagnostics{float minimum,sum;int valid;};
__global__ void count_valid(const float*c,int*v,float limit,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&c[i]<limit)atomicAdd(v,1);}
__global__ void pack(const float*m,const float*s,const int*v,Diagnostics*d){if(!threadIdx.x&&!blockIdx.x)*d={*m,*s,*v};}

int main(){
  constexpr int threads=256,iters=200; constexpr float collision=1.0e6f;
  std::printf("K,host_ms,gpu_ms,speedup,min_error,mean_error,valid_error\n");
  for(int K:{2048,8192,16384,65536}){
    std::vector<float> costs(K);for(int i=0;i<K;++i)costs[i]=(i%17==0)?collision+float(i%5):5.0f+0.01f*(i%101);
    float *dc,*dm,*ds;int*dv;Diagnostics*dd;void*temp;size_t a=0,b=0;
    CUDA_CHECK(cudaMalloc(&dc,K*sizeof(float)));CUDA_CHECK(cudaMalloc(&dm,sizeof(float)));CUDA_CHECK(cudaMalloc(&ds,sizeof(float)));CUDA_CHECK(cudaMalloc(&dv,sizeof(int)));CUDA_CHECK(cudaMalloc(&dd,sizeof(Diagnostics)));CUDA_CHECK(cudaMemcpy(dc,costs.data(),K*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cub::DeviceReduce::Min(nullptr,a,dc,dm,K));CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,b,dc,ds,K));size_t bytes=std::max(a,b);CUDA_CHECK(cudaMalloc(&temp,bytes));
    float host_min=0;double host_sum=0;int host_valid=0;auto h0=std::chrono::steady_clock::now();
    for(int it=0;it<iters;++it){CUDA_CHECK(cudaMemcpy(costs.data(),dc,K*sizeof(float),cudaMemcpyDeviceToHost));host_min=*std::min_element(costs.begin(),costs.end());host_sum=0;host_valid=0;for(float c:costs){host_sum+=c;host_valid+=c<collision;}}
    auto h1=std::chrono::steady_clock::now();cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
    for(int it=0;it<iters;++it){cub::DeviceReduce::Min(temp,bytes,dc,dm,K);cub::DeviceReduce::Sum(temp,bytes,dc,ds,K);cudaMemset(dv,0,sizeof(int));count_valid<<<(K+threads-1)/threads,threads>>>(dc,dv,collision,K);pack<<<1,1>>>(dm,ds,dv,dd);}
    cudaEventRecord(e);cudaEventSynchronize(e);float total=0;cudaEventElapsedTime(&total,s,e);Diagnostics got;cudaMemcpy(&got,dd,sizeof(got),cudaMemcpyDeviceToHost);
    double host_ms=std::chrono::duration<double,std::milli>(h1-h0).count()/iters,gpu_ms=total/iters;
    std::printf("%d,%.6f,%.6f,%.2f,%.9g,%.9g,%d\n",K,host_ms,gpu_ms,host_ms/gpu_ms,std::fabs(host_min-got.minimum),std::fabs(float(host_sum/K)-got.sum/K),std::abs(host_valid-got.valid));
    cudaFree(temp);cudaFree(dc);cudaFree(dm);cudaFree(ds);cudaFree(dv);cudaFree(dd);cudaEventDestroy(s);cudaEventDestroy(e);
  }
}
