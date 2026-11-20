#pragma once

#include "kiss_icp_spatial.cuh"
#include "kiss_icp_order.cuh"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace kiss_normal_cache {

struct View {
    const int *to_previous, *to_current, *previous_neighbors;
    const float *previous_normals, *previous_radii;
    int* neighbors;
    float* radii;
    int* reused;
    kiss_spatial::View additions;
};

// Conservative invalidation: any newly retained point at or inside the last
// support distance requires recomputation. Large searches also recompute.
__device__ inline bool addition_near(kiss_spatial::View v,const float* points,
                                     float x,float y,float z,float radius2) {
    if(!v.keys) return false;
    const float e=kiss_spatial::slack(x,y,z,v.cell);
    const int r=static_cast<int>(ceilf((sqrtf(radius2)+e)*v.inv));
    if(r>4) return true;
    const int a=static_cast<int>(floorf(x*v.inv));
    const int b=static_cast<int>(floorf(y*v.inv));
    const int c=static_cast<int>(floorf(z*v.inv));
    for(int dz=-r;dz<=r;++dz) for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx) {
        if(kiss_spatial::cell_distance2(x,y,z,a+dx,b+dy,c+dz,v.cell,e)>radius2) continue;
        const int s=kiss_spatial::slot(v,a+dx,b+dy,c+dz);
        if(s<0) continue;
        for(int p=v.offsets[s];p<v.offsets[s]+v.counts[s];++p) {
            const int j=v.indices[p];
            const float ex=points[3*j]-x,ey=points[3*j+1]-y,ez=points[3*j+2]-z;
            if(ex*ex+ey*ey+ez*ez<=radius2) return true;
        }
    }
    return false;
}

__device__ inline bool reuse(View v,const float* map,const float* additions,int i,int k,int* ids) {
    const int old=v.to_previous[i];
    if(old<0 || v.previous_radii[old]<0.f) return false;
    for(int a=0;a<k;++a) {
        const int previous=v.previous_neighbors[old*k+a];
        if(previous<0 || v.to_current[previous]<0) return false;
        ids[a]=v.to_current[previous];
    }
    return !addition_near(v.additions,additions,map[3*i],map[3*i+1],map[3*i+2],v.previous_radii[old]);
}

// Disable reuse for internal ties and ties at the k/k+1 boundary. Reordering
// point IDs can otherwise change support or PCA summation order without motion.
__device__ inline float support_radius(const int* ids,const float* distances,int k) {
    if(ids[k-1]<0) return -1.f;
    for(int a=1;a<=k;++a)
        if(ids[a]>=0 && distances[a]==distances[a-1]) return -1.f;
    return distances[k-1];
}

class Cache {
public:
    Cache(int capacity,int k,int hash_capacity,bool validate) {
        to_previous_.reserve(capacity); to_current_.reserve(capacity); added_.reserve(capacity);
        try {
            CUDA_CHECK(cudaMalloc(&d_to_previous_,capacity*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_to_current_,capacity*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_added_,capacity*sizeof(int)));
            CUDA_CHECK(cudaMalloc(&added_points_,capacity*3*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&previous_normals_,capacity*3*sizeof(float)));
            for(int b=0;b<2;++b) {
                CUDA_CHECK(cudaMalloc(&neighbors_[b],static_cast<size_t>(capacity)*k*sizeof(int)));
                CUDA_CHECK(cudaMalloc(&radii_[b],capacity*sizeof(float)));
            }
            CUDA_CHECK(cudaMalloc(&reused_,sizeof(int)));
            if(validate) {
                CUDA_CHECK(cudaMalloc(&validation_,capacity*3*sizeof(float)));
                CUDA_CHECK(cudaMalloc(&mismatches_,sizeof(int)));
            }
            delta_.reset(new kiss_spatial::Index(capacity,hash_capacity));
        } catch (...) { release(); throw; }
    }
    ~Cache() { release(); }
    Cache(const Cache&)=delete;
    Cache& operator=(const Cache&)=delete;
    void reset() { previous_count_=0; }

    View prepare(const float* map,float*& normals,const std::vector<int>& order,
                 const std::vector<int>& previous,float cell) {
        to_previous_.resize(order.size());
        to_current_.assign(previous_count_,-1); added_.clear();
        for(size_t i=0;i<order.size();++i) {
            const int old=previous[order[i]];
            if(old>=previous_count_) throw std::logic_error("invalid normal cache predecessor");
            to_previous_[i]=old;
            if(old>=0) to_current_[old]=static_cast<int>(i);
            else added_.push_back(static_cast<int>(i));
        }
        CUDA_CHECK(cudaMemcpy(d_to_previous_,to_previous_.data(),order.size()*sizeof(int),cudaMemcpyHostToDevice));
        if(previous_count_)
            CUDA_CHECK(cudaMemcpy(d_to_current_,to_current_.data(),previous_count_*sizeof(int),cudaMemcpyHostToDevice));
        kiss_spatial::View additions{};
        if(!added_.empty()) {
            CUDA_CHECK(cudaMemcpy(d_added_,added_.data(),added_.size()*sizeof(int),cudaMemcpyHostToDevice));
            kiss_order::gather<<<(added_.size()+255)/256,256>>>(map,d_added_,static_cast<int>(added_.size()),added_points_);
            additions=delta_->build(added_points_,static_cast<int>(added_.size()),cell);
        }
        std::swap(normals,previous_normals_);
        std::swap(neighbors_[0],neighbors_[1]); std::swap(radii_[0],radii_[1]);
        CUDA_CHECK(cudaMemset(reused_,0,sizeof(int)));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        return {d_to_previous_,d_to_current_,neighbors_[0],previous_normals_,radii_[0],
                neighbors_[1],radii_[1],reused_,additions};
    }
    void commit(const std::vector<int>& order,std::vector<int>& previous) {
        for(size_t i=0;i<order.size();++i) previous[order[i]]=static_cast<int>(i);
        previous_count_=static_cast<int>(order.size());
    }
    int reused() const {
        int count=0; CUDA_CHECK(cudaMemcpy(&count,reused_,sizeof(int),cudaMemcpyDeviceToHost)); return count;
    }
    const float* additions() const { return added_points_; }
    float* validation() const { return validation_; }
    int* mismatches() const { return mismatches_; }
private:
    void release() noexcept {
        cudaFree(d_to_previous_); cudaFree(d_to_current_); cudaFree(d_added_); cudaFree(added_points_);
        cudaFree(previous_normals_); cudaFree(reused_); cudaFree(validation_); cudaFree(mismatches_);
        for(int b=0;b<2;++b) { cudaFree(neighbors_[b]); cudaFree(radii_[b]); }
    }
    int previous_count_=0;
    std::vector<int> to_previous_,to_current_,added_;
    int *d_to_previous_=nullptr,*d_to_current_=nullptr,*d_added_=nullptr,*reused_=nullptr,*mismatches_=nullptr;
    int* neighbors_[2]={nullptr,nullptr};
    float *added_points_=nullptr,*previous_normals_=nullptr,*validation_=nullptr;
    float* radii_[2]={nullptr,nullptr};
    std::unique_ptr<kiss_spatial::Index> delta_;
};

} // namespace kiss_normal_cache
