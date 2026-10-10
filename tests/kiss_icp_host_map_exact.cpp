#include "kiss_icp_host_map.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <random>

using Point=std::array<float,3>;

static std::vector<Point> sorted(const std::vector<float>& xyz) {
    std::vector<Point> points;
    for(size_t i=0;i<xyz.size()/3;++i) points.push_back({xyz[3*i],xyz[3*i+1],xyz[3*i+2]});
    std::sort(points.begin(),points.end()); return points;
}

// Reference map membership/representatives, independent of dense slot movement.
static void reference(std::unordered_map<int64_t,Point>& map,const std::vector<float>& world,
                      const float* center,float radius,float cell) {
    auto inside=[&](const Point& p) {
        const float x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];
        return x*x+y*y+z*z<=radius*radius;
    };
    for(auto i=map.begin();i!=map.end();) if(!inside(i->second)) i=map.erase(i); else ++i;
    for(size_t i=0;i<world.size()/3;++i) {
        const Point p={world[3*i],world[3*i+1],world[3*i+2]};
        const int64_t voxel=kiss_host_map::key(p[0],p[1],p[2],cell);
        if(inside(p) && map.find(voxel)==map.end()) map[voxel]=p;
    }
}

int main() {
    std::vector<float> points;
    kiss_host_map::Dense small(4,1.f,points);
    float center[3]={0,0,0};
    small.insert({-1,0,0, .2f,0,0, 1,0,0, 2,0,0, .8f,0,0},center,2.f);
    if(sorted(points)!=std::vector<Point>{{-1,0,0},{.2f,0,0},{1,0,0},{2,0,0}}) return 1;
    center[0]=2;
    small.prune(center,2.f);
    small.insert({.9f,0,0, 4,0,0, -1,0,0},center,2.f);
    if(sorted(points)!=std::vector<Point>{{.2f,0,0},{1,0,0},{2,0,0},{4,0,0}}) return 2;
    center[0]=10; small.prune(center,1.f);
    if(!points.empty()) return 3;
    small.insert({10,0,0,11,0,0},center,1.f);
    small.clear();
    if(!points.empty()) return 4;
    small.insert({10,0,0},center,1.f);
    if(points.size()!=3) return 5;
    std::vector<float> bounded;
    kiss_host_map::Dense capacity(1,.1f,bounded);
    bool rejected=false;
    try { capacity.insert({10,0,0,11,0,0},center,2.f); }
    catch(const std::runtime_error&) { rejected=true; }
    if(!rejected || bounded.size()!=3) return 6;
    std::vector<int> after_capacity_error;
    capacity.export_order(after_capacity_error);
    if(after_capacity_error!=std::vector<int>{0}) return 9;

    std::vector<float> dense_points;
    kiss_host_map::Dense dense(20000,.35f,dense_points);
    std::unordered_map<int64_t,Point> expected;
    expected.reserve(20000);
    std::mt19937 random(7384);
    std::uniform_real_distribution<float> offset(-5.f,5.f);
    for(int frame=0;frame<120;++frame) {
        center[0]=-3.f+.08f*frame; center[1]=std::sin(.1f*frame); center[2]=.03f*frame;
        std::vector<float> world;
        for(int i=0;i<300;++i) {
            world.push_back(center[0]+offset(random)); world.push_back(center[1]+offset(random));
            world.push_back(center[2]+offset(random));
        }
        reference(expected,world,center,4.f,.35f);
        dense.prune(center,4.f); dense.insert(world,center,4.f);
        std::vector<Point> reference_points;
        for(const auto& item:expected) reference_points.push_back(item.second);
        std::sort(reference_points.begin(),reference_points.end());
        if(reference_points!=sorted(dense_points)) {
            std::fprintf(stderr,"map membership/representatives differ at frame %d\n",frame); return 7;
        }
        std::vector<int> order;
        dense.export_order(order);
        size_t rank=0;
        for(const auto& item:expected) {
            const size_t slot=static_cast<size_t>(order[rank++]);
            for(int a=0;a<3;++a) if(dense_points[3*slot+a]!=item.second[a]) {
                std::fprintf(stderr,"reference point order differs at frame %d\n",frame); return 8;
            }
        }
    }
    std::puts("KISS-ICP dense rolling map matches reference point membership: PASS");
    return 0;
}
