#include "kiss_icp_downsample.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
template<class T> static T read(std::ifstream& file) {
    T v{};file.read(reinterpret_cast<char*>(&v),sizeof(v));
    if(!file) throw std::runtime_error("truncated sequence");return v;
}
struct Scan { unsigned frame;std::vector<float> xyz; };
static std::vector<Scan> scans(const std::string& path,unsigned samples) {
    std::ifstream file(path,std::ios::binary);
    char magic[8]{};file.read(magic,8);
    if(!file || std::memcmp(magic,"CRKICP1\0",8)) throw std::runtime_error("invalid sequence magic");
    const auto version=read<unsigned>(file),count=read<unsigned>(file);
    if((version!=1 && version!=2) || count<2 || count>100000) throw std::runtime_error("invalid sequence header");
    samples=std::min(samples,count);
    std::vector<Scan> output;
    for(unsigned frame=0,next=0;frame<count;++frame) {
        read<std::uint64_t>(file);for(int i=0;i<4;++i) read<float>(file);
        const auto n=read<unsigned>(file);
        if(n<30 || n>200000) throw std::runtime_error("invalid sequence point count");
        if(version==2) {read<float>(file);read<float>(file);}
        const bool selected=next<samples && frame==static_cast<std::uint64_t>(next)*(count-1)/(samples-1);
        if(selected) {
            Scan scan{frame,{}};scan.xyz.resize(static_cast<size_t>(n)*3);
            if(version==1) file.read(reinterpret_cast<char*>(scan.xyz.data()),n*3*sizeof(float));
            else for(unsigned i=0;i<n;++i) {
                for(int a=0;a<3;++a) scan.xyz[3*i+a]=read<float>(file);
                read<float>(file);
            }
            output.push_back(std::move(scan));++next;
        } else file.seekg(static_cast<std::streamoff>(n)*(version==2 ? 16 : 12),std::ios::cur);
        if(!file) throw std::runtime_error("truncated point payload");
    }
    return output;
}
struct Counts {size_t calls=0,bytes=0;};
template<class T> struct CountingAllocator {
    using value_type=T;
    Counts* counts=nullptr;
    CountingAllocator()=default;
    explicit CountingAllocator(Counts& c):counts(&c) {}
    template<class U> CountingAllocator(const CountingAllocator<U>& a):counts(a.counts) {}
    T* allocate(size_t n) {
        if(counts) {++counts->calls;counts->bytes+=n*sizeof(T);}
        return std::allocator<T>().allocate(n);
    }
    void deallocate(T* p,size_t n) {std::allocator<T>().deallocate(p,n);}
    template<class U> bool operator==(const CountingAllocator<U>& a) const {return counts==a.counts;}
    template<class U> bool operator!=(const CountingAllocator<U>& a) const {return counts!=a.counts;}
};
struct Profile {double keys,reserve,aggregate,emit,destroy;Counts allocations;size_t cells;};
static Profile profile(const std::vector<float>& points,float cell,const std::vector<float>& expected) {
    Profile p{};std::vector<int64_t> keys(points.size()/3);
    auto start=Clock::now();
    for(size_t i=0;i<keys.size();++i) {
        const int64_t a=static_cast<int64_t>(std::floor(points[3*i]/cell));
        const int64_t b=static_cast<int64_t>(std::floor(points[3*i+1]/cell));
        const int64_t c=static_cast<int64_t>(std::floor(points[3*i+2]/cell));
        keys[i]=((a&0x1fffff)<<42)^((b&0x1fffff)<<21)^(c&0x1fffff);
    }
    p.keys=elapsed(start);
    using Value=std::pair<const int64_t,std::array<float,4>>;
    using Map=std::unordered_map<int64_t,std::array<float,4>,std::hash<int64_t>,
                                 std::equal_to<int64_t>,CountingAllocator<Value>>;
    start=Clock::now();
    std::unique_ptr<Map> map(new Map(0,std::hash<int64_t>(),std::equal_to<int64_t>(),CountingAllocator<Value>(p.allocations)));
    map->reserve(keys.size());p.reserve=elapsed(start);
    start=Clock::now();std::array<float,4>* last=nullptr;int64_t previous=0;
    for(size_t i=0;i<keys.size();++i) {
        if(!last || previous!=keys[i]) {last=&(*map)[keys[i]];previous=keys[i];}
        (*last)[0]+=points[3*i];(*last)[1]+=points[3*i+1];(*last)[2]+=points[3*i+2];(*last)[3]+=1.f;
    }
    p.aggregate=elapsed(start);p.cells=map->size();
    start=Clock::now();std::vector<float> out;out.reserve(p.cells*3);
    for(const auto& item:*map) {
        const auto& sum=item.second;
        out.push_back(sum[0]/sum[3]);out.push_back(sum[1]/sum[3]);out.push_back(sum[2]/sum[3]);
    }
    p.emit=elapsed(start);
    start=Clock::now();map.reset();p.destroy=elapsed(start);
    if(out.size()!=expected.size() || (!out.empty() && std::memcmp(out.data(),expected.data(),out.size()*sizeof(float))))
        throw std::runtime_error("profile output mismatch");
    return p;
}
static bool equal(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)));
}
int main(int argc,char** argv) {
    try {
        std::string sequence,csv;unsigned samples=12,repeats=8;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(i+1>=argc) throw std::invalid_argument("option requires a value");
            std::string value=argv[++i];
            if(key=="--sequence") sequence=value;
            else if(key=="--csv") csv=value;
            else if(key=="--samples") samples=std::stoul(value);
            else if(key=="--repeats") repeats=std::stoul(value);
            else throw std::invalid_argument("unknown option");
        }
        if(sequence.empty() || csv.empty() || samples<2 || !repeats) throw std::invalid_argument("use --sequence PATH --csv PATH [--samples N>=2] [--repeats N>=1]");
        const auto input=scans(sequence,samples);
        std::ofstream output(csv);
        if(!output) throw std::runtime_error("cannot write CSV");
        output<<"frame,repeat,mode,points,voxels,total_ms,keys_ms,reserve_ms,aggregate_ms,emit_ms,destroy_ms,allocation_calls,allocation_bytes,arena_bytes,new_arena_slabs,byte_equal\n"<<std::setprecision(12);
        kiss_downsample::Sampler pooled(200000);
        for(const auto& scan:input) {
            const auto expected=kiss_downsample::sample(scan.xyz,.22f,true);
            if(!equal(expected,pooled.sample(scan.xyz,.22f))) throw std::runtime_error("warmup output mismatch");
            const auto p=profile(scan.xyz,.22f,expected);
            output<<scan.frame<<",0,profile,"<<scan.xyz.size()/3<<','<<p.cells<<",0,"<<p.keys<<','<<p.reserve<<','<<p.aggregate<<','<<p.emit<<','<<p.destroy<<','<<p.allocations.calls<<','<<p.allocations.bytes<<",0,0,1\n";
            for(unsigned repeat=0;repeat<repeats;++repeat) for(int order=0;order<2;++order) {
                const bool reuse=(order+(repeat%2))%2;
                const size_t before=pooled.upstream_allocations();
                auto start=Clock::now();double ms=0;bool matches=false;
                if(reuse) {const auto& out=pooled.sample(scan.xyz,.22f);ms=elapsed(start);matches=equal(expected,out);}
                else {const auto out=kiss_downsample::sample(scan.xyz,.22f,true);ms=elapsed(start);matches=equal(expected,out);}
                output<<scan.frame<<','<<repeat<<','<<(reuse ? "pooled" : "cached")<<','<<scan.xyz.size()/3<<','<<expected.size()/3<<','<<ms<<",0,0,0,0,0,0,0,"<<pooled.arena_bytes()<<','<<pooled.upstream_allocations()-before<<','<<matches<<'\n';
                if(!matches) throw std::runtime_error("timed output mismatch");
            }
        }
        std::printf("%zu raw real scans, %u repeats, all output floats/order match; arena=%zu bytes\n",input.size(),repeats,pooled.arena_bytes());
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"downsample benchmark: %s\n",e.what());return 1;}
}
