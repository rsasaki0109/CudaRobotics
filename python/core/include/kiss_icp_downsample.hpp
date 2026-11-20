#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

namespace kiss_downsample {

// Recreate the same unordered container each scan, but reuse its backing
// storage. Keeping reserve/insertion order preserves implementation-specific
// iteration order; retaining the old container's buckets would not do so.
class Arena {
    struct Block {
        void* data;
        size_t size,used;
        explicit Block(size_t bytes):data(::operator new(bytes)),size(bytes),used(0) {}
        ~Block() { ::operator delete(data); }
        Block(const Block&)=delete;
        Block& operator=(const Block&)=delete;
    };
public:
    Arena()=default;
    Arena(const Arena&)=delete;
    Arena& operator=(const Arena&)=delete;
    // All allocator clients must be destroyed before reusing their storage.
    void reset() { for(auto& b:blocks_) b->used=0; current_=0; }
    void* allocate(size_t bytes,size_t alignment) {
        for(;current_<blocks_.size();++current_) {
            auto& b=*blocks_[current_];
            const size_t aligned=(b.used+alignment-1)&~(alignment-1);
            if(aligned<=b.size && bytes<=b.size-aligned) {
                b.used=aligned+bytes;
                return static_cast<char*>(b.data)+aligned;
            }
        }
        constexpr size_t slab=1u<<20;
        if(bytes>std::numeric_limits<size_t>::max()-(slab-1)) throw std::bad_alloc();
        const size_t capacity=((bytes+slab-1)/slab)*slab;
        std::unique_ptr<Block> block(new Block(capacity));
        blocks_.push_back(std::move(block));
        return allocate(bytes,alignment);
    }
    size_t reserved_bytes() const {
        size_t bytes=0; for(const auto& b:blocks_) bytes+=b->size; return bytes;
    }
    size_t upstream_allocations() const { return blocks_.size(); }
private:
    std::vector<std::unique_ptr<Block>> blocks_;
    size_t current_=0;
};

template<class T> class Allocator {
public:
    using value_type=T;
    Arena* arena=nullptr;
    Allocator()=default;
    explicit Allocator(Arena& storage):arena(&storage) {}
    template<class U> Allocator(const Allocator<U>& other):arena(other.arena) {}
    T* allocate(size_t count) {
        static_assert(alignof(T)<=alignof(std::max_align_t),"over-aligned arena type");
        if(count>std::numeric_limits<size_t>::max()/sizeof(T)) throw std::bad_alloc();
        if(!arena) return std::allocator<T>().allocate(count);
        return static_cast<T*>(arena->allocate(count*sizeof(T),alignof(T)));
    }
    void deallocate(T* p,size_t count) {
        if(!arena) std::allocator<T>().deallocate(p,count);
    }
    template<class U> bool operator==(const Allocator<U>& other) const { return arena==other.arena; }
    template<class U> bool operator!=(const Allocator<U>& other) const { return arena!=other.arena; }
};

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

class Sampler {
public:
    explicit Sampler(size_t capacity=0) { output_.reserve(capacity*3); }
    const std::vector<float>& sample(const std::vector<float>& points,float cell) {
        arena_.reset(); output_.clear();
        using Value=std::pair<const int64_t,std::array<float,4>>;
        using Map=std::unordered_map<int64_t,std::array<float,4>,std::hash<int64_t>,
                                     std::equal_to<int64_t>,Allocator<Value>>;
        Map voxels(0,std::hash<int64_t>(),std::equal_to<int64_t>(),Allocator<Value>(arena_));
        voxels.reserve(points.size()/3);
        int64_t previous_key=0;
        std::array<float,4>* previous=nullptr;
        for(size_t i=0;i<points.size()/3;++i) {
            const float x=points[3*i],y=points[3*i+1],z=points[3*i+2];
            const int64_t a=static_cast<int64_t>(std::floor(x/cell));
            const int64_t b=static_cast<int64_t>(std::floor(y/cell));
            const int64_t c=static_cast<int64_t>(std::floor(z/cell));
            const int64_t key=((a&0x1fffff)<<42)^((b&0x1fffff)<<21)^(c&0x1fffff);
            if(!previous || key!=previous_key) { previous=&voxels[key]; previous_key=key; }
            auto& sum=*previous;
            sum[0]+=x; sum[1]+=y; sum[2]+=z; sum[3]+=1.f;
        }
        output_.reserve(voxels.size()*3);
        for(const auto& item:voxels) {
            const auto& sum=item.second;
            output_.push_back(sum[0]/sum[3]); output_.push_back(sum[1]/sum[3]); output_.push_back(sum[2]/sum[3]);
        }
        return output_;
    }
    size_t arena_bytes() const { return arena_.reserved_bytes(); }
    size_t upstream_allocations() const { return arena_.upstream_allocations(); }
private:
    Arena arena_;
    std::vector<float> output_;
};

} // namespace kiss_downsample
