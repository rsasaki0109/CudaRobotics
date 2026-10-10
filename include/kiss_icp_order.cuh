#pragma once
#include <cuda_runtime.h>

namespace kiss_order {

// Preserve the reference map's GPU point indices while the host keeps a dense
// vector. Exporting integer slots avoids packing coordinates on the CPU.
static __global__ void gather(const float* dense,const int* order,int n,float* ordered) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const int source=order[i];
    for(int a=0;a<3;++a) ordered[3*i+a]=dense[3*source+a];
}

} // namespace kiss_order
