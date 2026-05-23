// cuda_check.cuh
//
// CUDA_CHECK macro shared across cudabot CUDA demos.  Use it to wrap
// every CUDA runtime call so a failure aborts with file/line info.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)
#endif
