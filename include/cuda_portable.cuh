// Small portability shims so the demos build with both GCC and MSVC hosts.
#pragma once

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// Lookup table readable from host and device code. `__host__` is not a valid
// variable attribute (MSVC rejects it), so each compilation pass gets its own
// copy: a __device__ array in the device pass, a plain host array otherwise.
#ifdef __CUDA_ARCH__
#define CUDABOT_HD_TABLE __device__ static const
#else
#define CUDABOT_HD_TABLE static const
#endif

namespace cudabot {

__host__ __device__ inline int popcount32(unsigned int v)
{
#ifdef __CUDA_ARCH__
    return __popc(v);
#else
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return static_cast<int>((((v + (v >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24);
#endif
}

// Portable mkdir(path, 0755). Returns 0 on success, -1 otherwise (e.g. it exists).
inline int make_dir(const char* path)
{
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

}  // namespace cudabot
