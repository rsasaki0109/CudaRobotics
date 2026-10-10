#include "kiss_icp_downsample.hpp"
#include <cstdio>
#include <cstring>
#include <random>

static bool check(const std::vector<float>& points,float cell) {
    static kiss_downsample::Sampler sampler;
    const auto reference=kiss_downsample::sample(points,cell,false);
    const auto cached=kiss_downsample::sample(points,cell,true);
    const auto& pooled=sampler.sample(points,cell);
    return reference.size()==cached.size() && reference.size()==pooled.size() && (reference.empty() ||
        (!std::memcmp(reference.data(),cached.data(),reference.size()*sizeof(float)) &&
         !std::memcmp(reference.data(),pooled.data(),reference.size()*sizeof(float))));
}

int main() {
    if(!check({},.22f)) return 1;
    const auto centroid=kiss_downsample::sample({.01f,.01f,.01f,.03f,.03f,.03f},1.f,true);
    if(centroid.size()!=3 || std::fabs(centroid[0]-.02f)>1e-7f) return 2;
    for(float cell:{.05f,.22f,.35f,1.f}) {
        std::vector<float> boundaries;
        for(int i=-25;i<=25;++i) for(int direction=-1;direction<=1;++direction) {
            const float x=i*cell;
            const float shifted=direction==0 ? x : std::nextafter(x,direction<0 ? -INFINITY : INFINITY);
            for(int repeat=0;repeat<5;++repeat) boundaries.insert(boundaries.end(),{shifted,-shifted,cell});
        }
        if(!check(boundaries,cell)) return 3;
    }
    std::mt19937 random(9381);
    std::uniform_real_distribution<float> coord(-20.f,20.f);
    std::vector<float> cloud;
    for(int i=0;i<200000;++i) {
        if(i%4) {
            const size_t p=i%3 ? cloud.size()-3 : static_cast<size_t>(random()%(cloud.size()/3))*3;
            const float x=cloud[p],y=cloud[p+1],z=cloud[p+2];
            cloud.insert(cloud.end(),{x,y,z});
        } else cloud.insert(cloud.end(),{coord(random),coord(random),coord(random)});
    }
    for(float cell:{.05f,.22f,1.f}) if(!check(cloud,cell)) return 4;
    kiss_downsample::Sampler sampler;
    sampler.sample(cloud,.22f);
    const size_t allocations=sampler.upstream_allocations();
    for(int i=0;i<10;++i) {
        sampler.sample({},.22f);
        const auto& actual=sampler.sample(cloud,.22f);
        const auto expected=kiss_downsample::sample(cloud,.22f,true);
        if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)) ||
           allocations!=sampler.upstream_allocations()) return 5;
    }
    std::puts("KISS-ICP cached centroid values and order match the reference: PASS");
    return 0;
}
