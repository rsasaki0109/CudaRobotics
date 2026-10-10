#include "cudarobotics/kiss_icp_gpu.hpp"
#include "kiss_icp_normal_cache.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace cudarobotics;

static bool run(bool ties,int k=12) {
    KissIcpConfig config;
    config.max_scan_points=2048; config.max_map_points=4096; config.hash_capacity=8192;
    config.map_voxel_size=.08f; config.scan_voxel_size=.06f;
    config.map_radius=3.f; config.threshold_min=.3f; config.threshold_max=.6f;
    config.normal_neighbors=k;
    config.normal_update=KissIcpNormalUpdate::Validate;
    KissIcpConfig reference=config;
    reference.normal_update=KissIcpNormalUpdate::Full;
    KissIcpOdometry cached(config),full(reference);
    std::vector<float> world;
    std::mt19937 random(571);
    std::uniform_real_distribution<float> coordinate(-2.4f,2.4f);
    for(int i=0;i<600;++i) {
        if(ties) world.insert(world.end(),{(i%20-10)*.2f,(i/20-15)*.2f,0.f});
        else world.insert(world.end(),{coordinate(random),coordinate(random),coordinate(random)});
    }
    bool saw_reuse=false;
    for(int frame=0;frame<50;++frame) {
        if(frame==30) {
            KissIcpPose initial; initial.t[0]=1.f; initial.t[1]=-.5f;
            cached.reset(initial); full.reset(initial);
            if(cached.timing().normal_reused_points || cached.timing().normal_recomputed_points) return false;
        }
        auto scan=world;
        const float shift=frame<5 ? 0.f : .02f*(frame%30);
        for(size_t i=0;i<scan.size();i+=3) scan[i]-=shift;
        // New voxel representatives both near supports and outside them.
        if(frame>=10 && frame<25) scan.insert(scan.end(),{.037f+frame*.01f,.047f,.071f,2.37f,2.11f,.63f});
        const auto a=cached.register_scan(scan),b=full.register_scan(scan);
        if(std::memcmp(&a.pose,&b.pose,sizeof(KissIcpPose)) || a.map_points!=b.map_points ||
           a.alignment.inliers!=b.alignment.inliers || a.alignment.rmse!=b.alignment.rmse ||
           a.alignment.threshold!=b.alignment.threshold || a.alignment.iterations!=b.alignment.iterations) {
            std::fprintf(stderr,"incremental/full streaming mismatch ties=%d frame=%d\n",ties,frame); return false;
        }
        saw_reuse|=cached.timing().normal_reused_points>0;
    }
    if(!ties && !saw_reuse) { std::fprintf(stderr,"stationary random cloud never reused normals\n"); return false; }
    return true;
}

__global__ void invalidation(kiss_spatial::View index,const float* added,int n,
        const float* queries,const float* radius2,int count,int* missed) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const float x=queries[3*i],y=queries[3*i+1],z=queries[3*i+2];
    bool expected=false;
    for(int j=0;j<n;++j) {
        const float dx=added[3*j]-x,dy=added[3*j+1]-y,dz=added[3*j+2]-z;
        expected|=dx*dx+dy*dy+dz*dz<=radius2[i];
    }
    const bool candidate=kiss_normal_cache::addition_near(index,added,x,y,z,radius2[i]);
    if(expected && !candidate) atomicAdd(missed,1);
}

static bool boundaries() {
    std::vector<float> added,queries,radius2;
    for(int x=-5;x<=5;++x) for(float side:{-1.f,1.f})
        added.insert(added.end(),{x*.5f,side*.5f,0.f});
    for(int x=-5;x<=5;++x) for(float delta:{-1e-6f,0.f,1e-6f}) {
        queries.insert(queries.end(),{x*.5f+delta,0.f,0.f});radius2.push_back(.25f);
    }
    std::mt19937 random(423);
    std::uniform_real_distribution<float> coordinate(-3.f,3.f);
    for(int i=0;i<257;++i) {
        queries.insert(queries.end(),{coordinate(random),coordinate(random),coordinate(random)});
        radius2.push_back(i%5==0 ? 100.f : .2f*(i%10));
    }
    float *dp=nullptr,*dq=nullptr,*dr=nullptr; int* missed=nullptr;
    CUDA_CHECK(cudaMalloc(&dp,added.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dq,queries.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dr,radius2.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&missed,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dp,added.data(),added.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq,queries.data(),queries.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dr,radius2.data(),radius2.size()*sizeof(float),cudaMemcpyHostToDevice));
    bool ok=true;
    kiss_spatial::Index index(static_cast<int>(added.size()/3),128);
    for(float cell:{.5f,1.05f,8.4f}) {
        auto v=index.build(dp,static_cast<int>(added.size()/3),cell);
        CUDA_CHECK(cudaMemset(missed,0,sizeof(int)));
        invalidation<<<(radius2.size()+127)/128,128>>>(v,dp,static_cast<int>(added.size()/3),dq,dr,
            static_cast<int>(radius2.size()),missed);
        int errors=0; CUDA_CHECK(cudaMemcpy(&errors,missed,sizeof(int),cudaMemcpyDeviceToHost));
        if(errors) { std::fprintf(stderr,"missed %d normal invalidations at cell=%g\n",errors,cell); ok=false; }
    }
    cudaFree(dp);cudaFree(dq);cudaFree(dr);cudaFree(missed);
    return ok;
}

static bool sparse() {
    KissIcpConfig config; config.max_scan_points=32;config.max_map_points=64;config.hash_capacity=128;
    config.normal_update=KissIcpNormalUpdate::Validate; config.normal_neighbors=20;
    config.map_voxel_size=.05f; config.scan_voxel_size=.03f;
    std::vector<float> scan;
    for(int i=0;i<12;++i) scan.insert(scan.end(),{i*.13f,(i%3)*.23f,(i%5)*.17f});
    KissIcpOdometry odometry(config);
    for(int i=0;i<4;++i) odometry.register_scan(scan);
    return odometry.timing().normal_reused_points==0;
}

int main() {
    if(!run(false) || !run(false,1) || !run(false,20) || !run(true,20) || !boundaries() || !sparse()) return 1;
    // Validation cannot claim to compare unsupported reference paths.
    KissIcpConfig invalid; invalid.normal_update=KissIcpNormalUpdate::Validate;
    invalid.map_backend=KissIcpMapBackend::Unordered;
    if(validate_kiss_icp_config(invalid).empty()) return 2;
    std::puts("Incremental map normals and poses match full recomputation: PASS");
    return 0;
}
