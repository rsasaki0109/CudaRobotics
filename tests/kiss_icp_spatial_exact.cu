#include "kiss_icp_spatial.cuh"
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>
#include <vector>

__global__ void query(kiss_spatial::View index,const float* map,int n,int k,int* neighbors,
                       const float* queries,int nq,float gate2,int* nearest,float* distances,kiss_spatial::View coarse) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) kiss_spatial::knn(index,map,n,i,k,neighbors+i*k,coarse);
    if(i<nq) nearest[i]=kiss_spatial::nearest(index,map,queries[3*i],queries[3*i+1],queries[3*i+2],gate2,distances[i]);
}
bool check(const std::vector<float>& points,int k,float cell,float gate,bool hierarchical=true) {
    const int n=points.size()/3;
    int capacity=2; while(capacity<n*2) capacity*=2;
    kiss_spatial::Index index(n,capacity);
    kiss_spatial::Index coarse(n,capacity);
    float *map,*queries,*distances; int *neighbors,*nearest;
    auto q=points;
    for(int i=0;i<n;++i) { q[3*i]+=.17f; q[3*i+1]+=.31f; q[3*i+2]-=.11f; }
    q.insert(q.end(),{0.f,0.f,0.f,-1.f,-1.f,-1.f,1000.f,1000.f,1000.f,.5f,0.f,0.f,-.5f,0.f,0.f});
    const int nq=q.size()/3;
    CUDA_CHECK(cudaMalloc(&map,points.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&queries,q.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&neighbors,n*k*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&nearest,nq*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&distances,nq*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(map,points.data(),points.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(queries,q.data(),q.size()*sizeof(float),cudaMemcpyHostToDevice));
    const auto v=index.build(map,n,cell);
    const auto cv=hierarchical ? coarse.build(map,n,cell*8.f) : kiss_spatial::View{};
    query<<<(nq+127)/128,128>>>(v,map,n,k,neighbors,queries,nq,gate*gate,nearest,distances,cv);
    std::vector<int> ids(n*k), nn(nq); std::vector<float> ds(nq);
    CUDA_CHECK(cudaMemcpy(ids.data(),neighbors,ids.size()*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nn.data(),nearest,nn.size()*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ds.data(),distances,ds.size()*sizeof(float),cudaMemcpyDeviceToHost));
    bool ok=true;
    for(int i=0;i<nq;++i) {
        std::vector<std::pair<float,int>> sorted;
        for(int j=0;j<n;++j) {
            float x=points[3*j]-q[3*i],y=points[3*j+1]-q[3*i+1],z=points[3*j+2]-q[3*i+2];
            sorted.emplace_back(x*x+y*y+z*z,j);
        }
        std::sort(sorted.begin(),sorted.end());
        int expected=sorted[0].first<gate*gate ? sorted[0].second : -1;
        if(nn[i]!=expected) { std::printf("nearest mismatch i=%d expected=%d got=%d\n",i,expected,nn[i]); ok=false; break; }
        if(i<n) {
            // kNN queries are map points, unlike the offset correspondence queries.
            sorted.clear();
            for(int j=0;j<n;++j) {
                const float x=points[3*j]-points[3*i],y=points[3*j+1]-points[3*i+1],z=points[3*j+2]-points[3*i+2];
                sorted.emplace_back(x*x+y*y+z*z,j);
            }
            std::sort(sorted.begin(),sorted.end());
            sorted.erase(std::remove_if(sorted.begin(),sorted.end(),[&](const std::pair<float,int>& a){return a.second==i;}),sorted.end());
            for(int a=0;a<k;++a) {
                expected=a<static_cast<int>(sorted.size()) ? sorted[a].second : -1;
                if(ids[i*k+a]!=expected) {std::printf("kNN mismatch i=%d rank=%d expected=%d got=%d\n",i,a,expected,ids[i*k+a]);ok=false;break;}
            }
        }
    }
    cudaFree(map);cudaFree(queries);cudaFree(neighbors);cudaFree(nearest);cudaFree(distances);
    return ok;
}
int main() {
    std::vector<float> grid;
    for(int x=-4;x<=4;++x) for(int y=-4;y<=4;++y) for(int z=-2;z<=2;++z)
        grid.insert(grid.end(),{x*.25f,y*.25f,z*.25f});
    // Equal-distance ties, duplicate points, exact gate boundaries, negative
    // cells, partial blocks and sparse fallback are all compared exhaustively.
    grid.insert(grid.end(),{0.f,0.f,0.f,80.f,80.f,80.f,-90.f,-90.f,-90.f});
    for(float cell:{.5f,1.f,3.f}) for(int k:{1,12,20})
        if(!check(grid,k,cell,1.f))return 1;
    if(!check({-20.f,0.f,0.f,0.f,0.f,0.f,20.f,0.f,0.f},20,.5f,.5f))return 2;
    std::mt19937 rng(42);std::uniform_real_distribution<float> random(-20.f,20.f);
    std::vector<float> cloud;
    for(int i=0;i<769;++i) cloud.insert(cloud.end(),{random(rng),random(rng),random(rng)});
    for(float cell:{.5f,1.05f,3.f}) if(!check(cloud,12,cell,3.f))return 3;
    if(!check(cloud,12,.5f,3.f,false))return 4;
    std::printf("Exact spatial kNN and radius-gated NN match exhaustive reference: PASS\n");
    return 0;
}
