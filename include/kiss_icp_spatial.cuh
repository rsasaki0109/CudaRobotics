#pragma once

// Internal, exact spatial queries for KISS-ICP. Buckets hold contiguous point
// indices; neither the point cloud nor the correspondence gate is reduced.
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "cuda_check.cuh"

namespace kiss_spatial {

constexpr unsigned long long EMPTY = ~0ull;
__host__ __device__ inline unsigned long long key(int x, int y, int z) {
    return ((static_cast<unsigned long long>(x) & 0x1fffffull) << 42) |
           ((static_cast<unsigned long long>(y) & 0x1fffffull) << 21) |
           (static_cast<unsigned long long>(z) & 0x1fffffull);
}
__device__ inline unsigned long long mix(unsigned long long x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
struct View {
    const unsigned long long* keys;
    const int *counts, *offsets, *indices;
    int capacity;
    float cell, inv;
};
__device__ inline int query_point(View v,int rank,bool cell_order) {
    return cell_order ? v.indices[rank] : rank;
}
__device__ inline int slot(View v, int x, int y, int z) {
    const auto k = key(x, y, z);
    int s = static_cast<int>(mix(k) & (v.capacity - 1));
    for (int p = 0; p < v.capacity; ++p) {
        const auto found = v.keys[s];
        if (found == k) return s;
        if (found == EMPTY) return -1;
        s = (s + 1) & (v.capacity - 1);
    }
    return -1;
}
static __global__ void count_cells(const float* points, int n, float inv,
                                   unsigned long long* keys, int* counts,
                                   int* point_slots, int capacity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto k = key(static_cast<int>(floorf(points[3*i]*inv)),
                       static_cast<int>(floorf(points[3*i+1]*inv)),
                       static_cast<int>(floorf(points[3*i+2]*inv)));
    int s = static_cast<int>(mix(k) & (capacity - 1));
    for (int p = 0; p < capacity; ++p) {
        const auto found = atomicCAS(keys + s, EMPTY, k);
        if (found == EMPTY || found == k) {
            point_slots[i] = s; atomicAdd(counts + s, 1); return;
        }
        s = (s + 1) & (capacity - 1);
    }
}
static __global__ void scatter_indices(const int* slots, int n, const int* offsets,
                                       int* cursors, int* indices) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int s = slots[i];
        indices[offsets[s] + atomicAdd(cursors + s, 1)] = i;
    }
}

class Index {
public:
    Index(int maximum_points, int capacity) : capacity_(capacity) {
        if (maximum_points < 1 || capacity < maximum_points ||
            (capacity & (capacity - 1))) throw std::invalid_argument("invalid spatial index capacity");
        try {
            CUDA_CHECK(cudaMalloc(&keys_, capacity * sizeof(unsigned long long)));
            CUDA_CHECK(cudaMalloc(&counts_, capacity * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&offsets_, capacity * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&cursors_, capacity * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&slots_, maximum_points * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&indices_, maximum_points * sizeof(int)));
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temporary_bytes_, counts_, offsets_, capacity));
            CUDA_CHECK(cudaMalloc(&temporary_, temporary_bytes_));
        } catch (...) { release(); throw; }
    }
    ~Index() { release(); }
    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;
    View build(const float* points, int n, float cell) {
        CUDA_CHECK(cudaMemset(keys_, 0xff, capacity_ * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(counts_, 0, capacity_ * sizeof(int)));
        count_cells<<<(n+255)/256,256>>>(points,n,1.f/cell,keys_,counts_,slots_,capacity_);
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temporary_,temporary_bytes_,counts_,offsets_,capacity_));
        CUDA_CHECK(cudaMemset(cursors_, 0, capacity_ * sizeof(int)));
        scatter_indices<<<(n+255)/256,256>>>(slots_,n,offsets_,cursors_,indices_);
        CUDA_CHECK(cudaGetLastError());
        return {keys_,counts_,offsets_,indices_,capacity_,cell,1.f/cell};
    }
private:
    void release() noexcept {
        cudaFree(keys_); cudaFree(counts_); cudaFree(offsets_); cudaFree(cursors_);
        cudaFree(slots_); cudaFree(indices_); cudaFree(temporary_);
    }
    int capacity_;
    unsigned long long* keys_ = nullptr;
    int *counts_ = nullptr, *offsets_ = nullptr, *cursors_ = nullptr;
    int *slots_ = nullptr, *indices_ = nullptr;
    void* temporary_ = nullptr;
    size_t temporary_bytes_ = 0;
};

// Inflate bounding boxes slightly to keep floating-point lower bounds
// conservative near voxel boundaries. This changes pruning, not distances.
__device__ inline float slack(float x, float y, float z, float cell) {
    return 8.e-6f * (fabsf(x) + fabsf(y) + fabsf(z) + cell + 1.f);
}
__device__ inline float cell_distance2(float x,float y,float z,int a,int b,int c,float cell,float e) {
    const float dx = fmaxf(0.f,fmaxf(a*cell-e-x,x-((a+1)*cell+e)));
    const float dy = fmaxf(0.f,fmaxf(b*cell-e-y,y-((b+1)*cell+e)));
    const float dz = fmaxf(0.f,fmaxf(c*cell-e-z,z-((c+1)*cell+e)));
    return dx*dx + dy*dy + dz*dz;
}
__device__ inline float outside_distance(float x,float y,float z,int a,int b,int c,int r,float cell,float e) {
    float d = fminf(x-(a-r)*cell,(a+r+1)*cell-x);
    d = fminf(d,fminf(y-(b-r)*cell,(b+r+1)*cell-y));
    d = fminf(d,fminf(z-(c-r)*cell,(c+r+1)*cell-z));
    return fmaxf(0.f,d-e);
}
__device__ inline void insert(float d,int j,float* distances,int* indices,int k) {
    if (d > distances[k-1] || (d == distances[k-1] && j >= indices[k-1])) return;
    int p = k-1;
    while (p > 0 && (distances[p-1] > d || (distances[p-1] == d && indices[p-1] > j))) {
        distances[p] = distances[p-1]; indices[p] = indices[p-1]; --p;
    }
    distances[p] = d; indices[p] = j;
}
__device__ inline bool search_knn(View v,const float* points,int i,int k,int* ids,float* distances) {
    for (int t=0;t<k;++t) { distances[t]=1e30f; ids[t]=-1; }
    const float x=points[3*i],y=points[3*i+1],z=points[3*i+2];
    const int a=static_cast<int>(floorf(x*v.inv)),b=static_cast<int>(floorf(y*v.inv)),c=static_cast<int>(floorf(z*v.inv));
    const float e=slack(x,y,z,v.cell);
    for (int r=0;r<=4;++r) {
        for (int dz=-r;dz<=r;++dz) for (int dy=-r;dy<=r;++dy) for (int dx=-r;dx<=r;++dx) {
            if (r && abs(dx)<r && abs(dy)<r && abs(dz)<r) continue;
            if (cell_distance2(x,y,z,a+dx,b+dy,c+dz,v.cell,e) > distances[k-1]) continue;
            const int s=slot(v,a+dx,b+dy,c+dz);
            if (s<0) continue;
            for (int p=v.offsets[s];p<v.offsets[s]+v.counts[s];++p) {
                const int j=v.indices[p]; if(j==i) continue;
                const float ex=points[3*j]-x,ey=points[3*j+1]-y,ez=points[3*j+2]-z;
                insert(ex*ex+ey*ey+ez*ez,j,distances,ids,k);
            }
        }
        const float d=outside_distance(x,y,z,a,b,c,r,v.cell,e);
        if (ids[k-1]>=0 && d*d > distances[k-1]) return true;
    }
    return false;
}
// Exact global kNN excluding self. A coarse index handles sparse fine-grid
// queries before the exhaustive fallback. Each search restarts its top-k;
// no point is counted twice and no radius cutoff changes the neighbours.
__device__ inline void knn(View v,const float* points,int n,int i,int k,int* ids,View coarse={}) {
    float distances[20];
    if(search_knn(v,points,i,k,ids,distances)) return;
    if(coarse.keys && search_knn(coarse,points,i,k,ids,distances)) return;
    const float x=points[3*i],y=points[3*i+1],z=points[3*i+2];
    // Restart: a point must not appear twice when the exhaustive path revisits it.
    for (int t=0;t<k;++t) { distances[t]=1e30f; ids[t]=-1; }
    for (int j=0;j<n;++j) if(j!=i) {
        const float ex=points[3*j]-x,ey=points[3*j+1]-y,ez=points[3*j+2]-z;
        insert(ex*ex+ey*ey+ez*ez,j,distances,ids,k);
    }
}

__device__ inline int nearest(View v,const float* points,float x,float y,float z,float gate2,float& best) {
    const int a=static_cast<int>(floorf(x*v.inv)),b=static_cast<int>(floorf(y*v.inv)),c=static_cast<int>(floorf(z*v.inv));
    const float e=slack(x,y,z,v.cell);
    const int radius=static_cast<int>(ceilf((sqrtf(gate2)+e)*v.inv));
    best=gate2; int id=-1;
    for (int r=0;r<=radius;++r) {
        for(int dz=-r;dz<=r;++dz) for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx) {
            if(r && abs(dx)<r && abs(dy)<r && abs(dz)<r) continue;
            if(cell_distance2(x,y,z,a+dx,b+dy,c+dz,v.cell,e)>best) continue;
            const int s=slot(v,a+dx,b+dy,c+dz); if(s<0) continue;
            for(int p=v.offsets[s];p<v.offsets[s]+v.counts[s];++p) {
                const int j=v.indices[p];
                const float ex=points[3*j]-x,ey=points[3*j+1]-y,ez=points[3*j+2]-z;
                const float d=ex*ex+ey*ey+ez*ez;
                if(d<best || (id>=0 && d==best && j<id)) { best=d; id=j; }
            }
        }
        const float d=outside_distance(x,y,z,a,b,c,r,v.cell,e);
        if(d*d>best) break;
    }
    return id;
}

} // namespace kiss_spatial
