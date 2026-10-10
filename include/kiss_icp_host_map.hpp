#pragma once

// One retained point per voxel, with exact radius eviction and dense storage.
// Existing points keep their coordinates. Removal fills a hole with the last
// point, so point order is not the unordered-map reference's iteration order.
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <utility>

namespace kiss_host_map {

// Persistent nodes cannot use a per-scan arena: erasing a voxel returns its
// storage to the matching size class, while live map nodes remain untouched.
// The unordered container still owns hashing, bucket layout and iteration order.
class NodePool {
    struct Free { Free* next; };
    struct Bin {
        size_t stride=0;
        Free* free=nullptr;
        char* next=nullptr;
        size_t remaining=0;
    };
    struct Delete { void operator()(void* p) const { ::operator delete(p); } };
public:
    NodePool()=default;
    NodePool(const NodePool&)=delete;
    NodePool& operator=(const NodePool&)=delete;
    void* allocate(size_t size,size_t alignment) {
        const size_t stride=slot_size(size,alignment);
        Bin* bin=nullptr;
        for(auto& candidate:bins_) {
            if(candidate.stride==stride) { bin=&candidate; break; }
            if(!candidate.stride) { candidate.stride=stride; bin=&candidate; break; }
        }
        if(!bin) throw std::bad_alloc();
        if(bin->free) {
            Free* p=bin->free; bin->free=p->next; return p;
        }
        if(!bin->remaining) {
            constexpr size_t slab=256u<<10;
            const size_t count=std::max(size_t(1),slab/stride);
            std::unique_ptr<void,Delete> block(::operator new(count*stride));
            char* data=static_cast<char*>(block.get());
            blocks_.push_back(std::move(block));
            bytes_+=count*stride;
            bin->next=data; bin->remaining=count;
        }
        void* p=bin->next; bin->next+=stride; --bin->remaining; return p;
    }
    void deallocate(void* p,size_t size,size_t alignment) noexcept {
        const size_t stride=slot_size(size,alignment);
        for(auto& bin:bins_) if(bin.stride==stride) {
            bin.free=new(p) Free{bin.free}; return;
        }
    }
    size_t reserved_bytes() const { return bytes_; }
    size_t upstream_allocations() const { return blocks_.size(); }
private:
    static size_t slot_size(size_t size,size_t alignment) noexcept {
        // Every slot starts on max_align_t, including nodes reused by another
        // allocator rebind with the same size but a different alignment.
        (void)alignment;
        const size_t a=alignof(std::max_align_t);
        return (std::max(size,sizeof(Free))+a-1)&~(a-1);
    }
    std::array<Bin,8> bins_{};
    std::vector<std::unique_ptr<void,Delete>> blocks_;
    size_t bytes_=0;
};

template<class T> class Allocator {
public:
    using value_type=T;
    NodePool* pool=nullptr;
    Allocator()=default;
    explicit Allocator(NodePool* p):pool(p) {}
    template<class U> Allocator(const Allocator<U>& other):pool(other.pool) {}
    T* allocate(size_t count) {
        static_assert(alignof(T)<=alignof(std::max_align_t),"over-aligned map node");
        if(count>std::numeric_limits<size_t>::max()/sizeof(T)) throw std::bad_alloc();
        // Bucket arrays keep the standard allocator. Only individual nodes
        // enter the pool; no implementation-specific node type is assumed.
        if(pool && count==1) return static_cast<T*>(pool->allocate(sizeof(T),alignof(T)));
        return std::allocator<T>().allocate(count);
    }
    void deallocate(T* p,size_t count) noexcept {
        if(pool && count==1) pool->deallocate(p,sizeof(T),alignof(T));
        else std::allocator<T>().deallocate(p,count);
    }
    template<class U> bool operator==(const Allocator<U>& other) const { return pool==other.pool; }
    template<class U> bool operator!=(const Allocator<U>& other) const { return pool!=other.pool; }
};

inline int64_t key(float x,float y,float z,float cell) {
    const float inv=1.f/cell;
    const int64_t a=static_cast<int64_t>(std::floor(x*inv));
    const int64_t b=static_cast<int64_t>(std::floor(y*inv));
    const int64_t c=static_cast<int64_t>(std::floor(z*inv));
    return ((a&0x1fffff)<<42)^((b&0x1fffff)<<21)^(c&0x1fffff);
}

class Dense {
public:
    Dense(size_t capacity,float cell,std::vector<float>& points,std::vector<int>* previous=nullptr,bool pooled=false)
        : capacity_(capacity),cell_(cell),points_(points),previous_(previous),cache_order_(pooled && supported_order()),
          slots_(0,std::hash<int64_t>(),std::equal_to<int64_t>(),Allocator<Value>(pooled ? &pool_ : nullptr)) {
        slots_.reserve(capacity); points_.reserve(capacity*3);
        if(previous_) previous_->reserve(capacity);
        if(cache_order_) { next_.reserve(capacity); before_.reserve(capacity); }
    }
    size_t pool_bytes() const { return pool_.reserved_bytes(); }
    size_t pool_allocations() const { return pool_.upstream_allocations(); }
    size_t order_bytes() const { return (next_.capacity()+before_.capacity())*sizeof(int); }
    void clear() {
        slots_.clear(); points_.clear(); if(previous_) previous_->clear();
        next_.clear(); before_.clear(); first_=last_=-1;
    }
    void export_order(std::vector<int>& order) const {
        if(cache_order_) {
            order.resize(slots_.size());
            size_t rank=0;
            int slot=first_;
            for(;slot>=0 && rank<order.size();slot=next_[slot]) {
                if(static_cast<size_t>(slot)>=next_.size()) throw std::runtime_error("invalid host-map order slot");
                order[rank++]=slot;
            }
            if(slot!=-1 || rank!=order.size()) throw std::runtime_error("incomplete host-map order chain");
            return;
        }
        order.clear(); order.reserve(slots_.size());
        for(const auto& item:slots_) order.push_back(static_cast<int>(item.second));
    }
    void prune(const float* center,float radius) {
        const float radius2=radius*radius;
        for(size_t i=0;i<points_.size()/3;) {
            const float x=points_[3*i],y=points_[3*i+1],z=points_[3*i+2];
            const float dx=x-center[0],dy=y-center[1],dz=z-center[2];
            if(!(dx*dx+dy*dy+dz*dz>radius2)) { ++i; continue; }
            if(cache_order_) unlink(static_cast<int>(i));
            slots_.erase(key(x,y,z,cell_));
            const size_t last=points_.size()/3-1;
            if(i!=last) {
                for(int a=0;a<3;++a) points_[3*i+a]=points_[3*last+a];
                slots_.find(key(points_[3*i],points_[3*i+1],points_[3*i+2],cell_))->second=i;
                if(previous_) (*previous_)[i]=(*previous_)[last];
                if(cache_order_) {
                    before_[i]=before_[last]; next_[i]=next_[last];
                    connect(before_[i],static_cast<int>(i),next_[i]);
                }
            }
            points_.resize(points_.size()-3);
            if(previous_) previous_->resize(previous_->size()-1);
            if(cache_order_) { next_.pop_back(); before_.pop_back(); }
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
            const size_t slot=points_.size()/3;
            if(cache_order_) {
                // The reserved table cannot rehash below capacity. Supported
                // libraries link just the new node into their global iteration
                // chain; its public successor tells us the exact insertion spot.
                auto item=slots_.emplace(voxel,slot).first;
                ++item;
                const int after=item==slots_.end() ? -1 : static_cast<int>(item->second);
                const int before=after<0 ? last_ : before_[after];
                before_.push_back(before); next_.push_back(after);
                connect(before,static_cast<int>(slot),after);
            } else slots_[voxel]=slot;
            points_.push_back(x); points_.push_back(y); points_.push_back(z);
            if(previous_) previous_->push_back(-1);
        }
    }
private:
    static constexpr bool supported_order() {
        // Insertion behaviour is implementation-specific, not promised by the
        // unordered-container standard. Unknown libraries retain iterator walks.
#if defined(_MSVC_STL_VERSION) || defined(__GLIBCXX__)
        return true;
#else
        return false;
#endif
    }
    void connect(int before,int slot,int after) {
        if(before<0) first_=slot; else next_[before]=slot;
        if(after<0) last_=slot; else before_[after]=slot;
    }
    void unlink(int slot) {
        const int before=before_[slot],after=next_[slot];
        if(before<0) first_=after; else next_[before]=after;
        if(after<0) last_=before; else before_[after]=before;
    }
    size_t capacity_;
    float cell_;
    std::vector<float>& points_;
    std::vector<int>* previous_;
    bool cache_order_;
    std::vector<int> next_,before_;
    int first_=-1,last_=-1;
    using Value=std::pair<const int64_t,size_t>;
    NodePool pool_;  // Must outlive the container and all rebound allocators.
    std::unordered_map<int64_t,size_t,std::hash<int64_t>,std::equal_to<int64_t>,Allocator<Value>> slots_;
};

} // namespace kiss_host_map
