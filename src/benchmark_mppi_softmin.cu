// Compare the legacy host softmin pipeline with a device-only CUB pipeline.
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}while(0)

__global__ void make_weights(const float * costs, float * weights, const float * minimum, float lambda, int n)
{ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) weights[i]=expf(-(costs[i]-*minimum)/lambda); }
__global__ void normalize(float * weights, const float * sum, int n)
{ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) weights[i]/=*sum; }

int main()
{
  constexpr float lambda=0.12f; constexpr int threads=256, iterations=200;
  std::printf("K,host_pipeline_ms,gpu_pipeline_ms,speedup,max_abs_error\n");
  for(int K:{2048,8192,16384,65536}){
    std::vector<float> costs(K), host_weights(K), gpu_weights(K);
    for(int i=0;i<K;++i) costs[i]=5.0f+0.08f*std::sin(i*0.017f)+0.00001f*(i%97);
    float *d_costs,*d_weights,*d_min,*d_sum; void *temp=nullptr; size_t min_bytes=0,sum_bytes=0;
    CUDA_CHECK(cudaMalloc(&d_costs,K*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_weights,K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min,sizeof(float))); CUDA_CHECK(cudaMalloc(&d_sum,sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_costs,costs.data(),K*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cub::DeviceReduce::Min(nullptr,min_bytes,d_costs,d_min,K));
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr,sum_bytes,d_weights,d_sum,K));
    size_t bytes=std::max(min_bytes,sum_bytes); CUDA_CHECK(cudaMalloc(&temp,bytes));
    auto host_start=std::chrono::steady_clock::now();
    for(int it=0;it<iterations;++it){
      CUDA_CHECK(cudaMemcpy(costs.data(),d_costs,K*sizeof(float),cudaMemcpyDeviceToHost));
      float minimum=*std::min_element(costs.begin(),costs.end()); double sum=0.0;
      for(int i=0;i<K;++i){host_weights[i]=std::exp(-(costs[i]-minimum)/lambda);sum+=host_weights[i];}
      float inv=static_cast<float>(1.0/sum); for(float &w:host_weights)w*=inv;
      CUDA_CHECK(cudaMemcpy(d_weights,host_weights.data(),K*sizeof(float),cudaMemcpyHostToDevice));
    }
    auto host_stop=std::chrono::steady_clock::now();
    cudaEvent_t start,stop; CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for(int it=0;it<iterations;++it){
      CUDA_CHECK(cub::DeviceReduce::Min(temp,bytes,d_costs,d_min,K));
      make_weights<<<(K+threads-1)/threads,threads>>>(d_costs,d_weights,d_min,lambda,K);
      CUDA_CHECK(cub::DeviceReduce::Sum(temp,bytes,d_weights,d_sum,K));
      normalize<<<(K+threads-1)/threads,threads>>>(d_weights,d_sum,K);
    }
    CUDA_CHECK(cudaEventRecord(stop));CUDA_CHECK(cudaEventSynchronize(stop)); float gpu_total=0;CUDA_CHECK(cudaEventElapsedTime(&gpu_total,start,stop));
    CUDA_CHECK(cudaMemcpy(gpu_weights.data(),d_weights,K*sizeof(float),cudaMemcpyDeviceToHost));
    double host_ms=std::chrono::duration<double,std::milli>(host_stop-host_start).count()/iterations;
    float gpu_ms=gpu_total/iterations,max_error=0;for(int i=0;i<K;++i)max_error=std::max(max_error,std::fabs(host_weights[i]-gpu_weights[i]));
    std::printf("%d,%.6f,%.6f,%.2f,%.9g\n",K,host_ms,gpu_ms,host_ms/gpu_ms,max_error);
    cudaFree(temp);cudaFree(d_costs);cudaFree(d_weights);cudaFree(d_min);cudaFree(d_sum);cudaEventDestroy(start);cudaEventDestroy(stop);
  }
}
