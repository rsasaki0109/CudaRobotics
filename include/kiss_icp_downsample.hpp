#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kiss_downsample {

// Input-order centroid sums and output container are unchanged. Adjacent points
// in the same voxel can reuse the previous value pointer without another lookup.
inline std::vector<float> sample(const std::vector<float>& points,float cell,bool cached) {
    std::unordered_map<int64_t,std::array<float,4>> voxels;
    voxels.reserve(points.size()/3);
    int64_t previous_key=0;
    std::array<float,4>* previous=nullptr;
    for(size_t i=0;i<points.size()/3;++i) {
        const float x=points[3*i],y=points[3*i+1],z=points[3*i+2];
        const int64_t a=static_cast<int64_t>(std::floor(x/cell));
        const int64_t b=static_cast<int64_t>(std::floor(y/cell));
        const int64_t c=static_cast<int64_t>(std::floor(z/cell));
        const int64_t key=((a&0x1fffff)<<42)^((b&0x1fffff)<<21)^(c&0x1fffff);
        if(!cached || !previous || key!=previous_key) {
            previous=&voxels[key]; previous_key=key;
        }
        auto& sum=*previous;
        sum[0]+=x; sum[1]+=y; sum[2]+=z; sum[3]+=1.f;
    }
    std::vector<float> output; output.reserve(voxels.size()*3);
    for(const auto& item:voxels) {
        const auto& sum=item.second;
        output.push_back(sum[0]/sum[3]); output.push_back(sum[1]/sum[3]); output.push_back(sum[2]/sum[3]);
    }
    return output;
}

} // namespace kiss_downsample
