#include "kiss_icp_host_map.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
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

static bool pooled_exact() {
    std::vector<float> a,b;
    std::vector<int> ta,tb,oa,ob;
    kiss_host_map::Dense standard(20000,.35f,a,&ta),pooled(20000,.35f,b,&tb,true);
    std::mt19937 random(5891);
    std::uniform_real_distribution<float> offset(-4.f,4.f);
    size_t retained=0;
    for(int frame=0;frame<240;++frame) {
        if(frame==120) { standard.clear(); pooled.clear(); retained=pooled.pool_allocations(); }
        float center[3]={-.8f+.07f*(frame%120),std::sin(.1f*frame),0};
        std::vector<float> world;
        for(int i=0;i<400;++i) {
            const float x=center[0]+offset(random),y=center[1]+offset(random),z=offset(random);
            world.insert(world.end(),{x,y,z,x,y,z});  // Existing representatives stay unchanged.
        }
        world.insert(world.end(),{center[0]+4.f,center[1],0.f,center[0]-4.f,center[1],-0.f});
        standard.prune(center,4.f); pooled.prune(center,4.f);
        standard.insert(world,center,4.f); pooled.insert(world,center,4.f);
        standard.export_order(oa); pooled.export_order(ob);
        if(a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) ||
           ta!=tb || oa!=ob || standard.pool_bytes()!=0 || pooled.pool_bytes()==0) {
            std::fprintf(stderr,"pooled map point bits/order/tags differ at frame %d\n",frame); return false;
        }
        // Simulate committed cache ranks, then check they survive swap deletion.
        for(size_t i=0;i<oa.size();++i) ta[oa[i]]=tb[ob[i]]=static_cast<int>(i);
        if(frame==121 && pooled.pool_allocations()!=retained) return false;
    }
    // Repeated fills/evictions/reset must recycle nodes instead of accumulating
    // storage for every historical insertion. Also preserve partial capacity errors.
    std::vector<float> s,p;
    kiss_host_map::Dense small(4,1.f,s),small_pool(4,1.f,p,nullptr,true);
    float center[3]={0,0,0};
    size_t high_water=0;
    for(int cycle=0;cycle<100;++cycle) {
        small.clear(); small_pool.clear();
        for(auto* map:{&small,&small_pool}) {
            bool rejected=false;
            try { map->insert({-1,0,0,0,0,0,1,0,0,2,0,0,3,0,0},center,4.f); }
            catch(const std::runtime_error&) { rejected=true; }
            if(!rejected) return false;
        }
        small.export_order(oa); small_pool.export_order(ob);
        if(s!=p || oa!=ob) return false;
        if(!cycle) high_water=small_pool.pool_allocations();
        else if(high_water!=small_pool.pool_allocations()) return false;
        center[0]=20; small.prune(center,1.f); small_pool.prune(center,1.f);
        if(!s.empty() || !p.empty()) return false;
        center[0]=0;
    }
    return true;
}

int main() {
    if(!pooled_exact()) return 13;
    std::vector<float> tagged_points;
    std::vector<int> tags;
    kiss_host_map::Dense tagged(4,1.f,tagged_points,&tags);
    float tagged_center[3]={0,0,0};
    tagged.insert({-1,0,0,.2f,0,0,1,0,0,2,0,0},tagged_center,2.f);
    if(tags!=std::vector<int>(4,-1)) return 10;
    tags={0,1,2,3}; tagged_center[0]=2;
    tagged.prune(tagged_center,2.f);
    tagged.insert({4,0,0},tagged_center,2.f);
    for(size_t i=0;i<tags.size();++i) {
        const float x=tagged_points[3*i];
        const int expected=x==.2f ? 1 : x==1.f ? 2 : x==2.f ? 3 : -1;
        if(tags[i]!=expected) return 11;
    }
    tagged.clear();
    if(!tags.empty() || !tagged_points.empty()) return 12;
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
