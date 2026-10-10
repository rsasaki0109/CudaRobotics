#pragma once

// One retained point per voxel, with exact radius eviction and dense storage.
// Existing points keep their coordinates. Removal fills a hole with the last
// point, so point order is not the unordered-map reference's iteration order.
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace kiss_host_map {

inline int64_t key(float x,float y,float z,float cell) {
    const float inv=1.f/cell;
    const int64_t a=static_cast<int64_t>(std::floor(x*inv));
    const int64_t b=static_cast<int64_t>(std::floor(y*inv));
    const int64_t c=static_cast<int64_t>(std::floor(z*inv));
    return ((a&0x1fffff)<<42)^((b&0x1fffff)<<21)^(c&0x1fffff);
}

class Dense {
public:
    Dense(size_t capacity,float cell,std::vector<float>& points)
        : capacity_(capacity),cell_(cell),points_(points) {
        slots_.reserve(capacity); points_.reserve(capacity*3);
    }
    void clear() { slots_.clear(); points_.clear(); }
    void export_order(std::vector<int>& order) const {
        order.clear(); order.reserve(slots_.size());
        for(const auto& item:slots_) order.push_back(static_cast<int>(item.second));
    }
    void prune(const float* center,float radius) {
        const float radius2=radius*radius;
        for(size_t i=0;i<points_.size()/3;) {
            const float x=points_[3*i],y=points_[3*i+1],z=points_[3*i+2];
            const float dx=x-center[0],dy=y-center[1],dz=z-center[2];
            if(!(dx*dx+dy*dy+dz*dz>radius2)) { ++i; continue; }
            slots_.erase(key(x,y,z,cell_));
            const size_t last=points_.size()/3-1;
            if(i!=last) {
                for(int a=0;a<3;++a) points_[3*i+a]=points_[3*last+a];
                slots_.find(key(points_[3*i],points_[3*i+1],points_[3*i+2],cell_))->second=i;
            }
            points_.resize(points_.size()-3);
        }
    }
    void insert(const std::vector<float>& world,const float* center,float radius) {
        const float radius2=radius*radius;
        for(size_t i=0;i<world.size()/3;++i) {
            const float x=world[3*i],y=world[3*i+1],z=world[3*i+2];
            const float dx=x-center[0],dy=y-center[1],dz=z-center[2];
            if(dx*dx+dy*dy+dz*dz>radius2) continue;
            const int64_t voxel=key(x,y,z,cell_);
            if(slots_.find(voxel)!=slots_.end()) continue;
            if(slots_.size()>=capacity_) throw std::runtime_error("KISS-ICP local map capacity exceeded");
            slots_[voxel]=points_.size()/3;
            points_.push_back(x); points_.push_back(y); points_.push_back(z);
        }
    }
private:
    size_t capacity_;
    float cell_;
    std::vector<float>& points_;
    std::unordered_map<int64_t,size_t> slots_;
};

} // namespace kiss_host_map
