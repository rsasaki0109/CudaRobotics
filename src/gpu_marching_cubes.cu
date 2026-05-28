// gpu_marching_cubes.cu
//
// GPU Marching Cubes — the standard iso-surface mesh extractor that turns a
// dense SDF / TSDF volume into a triangle mesh.  It is the natural successor to
// the TSDF demo (#130): TSDF fusion *builds* the volume, Marching Cubes
// *reads* it.
//
// The algorithm is the textbook one (Lorensen & Cline 1987, edge/tri tables
// from Bourke's public-domain layout): for every cube cell, classify each of
// its 8 corners as inside/outside the iso-surface, look the 8-bit configuration
// up in a 256-entry table, and emit up to 5 triangles whose vertices live on
// the cell's edges (linear interpolation of the SDF on each edge).  This is the
// canonical "one thread = one cell" GPU map.
//
// Correctness — deterministic by construction
// -------------------------------------------
// MC has no data-dependent branches that fork into different answers: every
// cell looks up the same fixed table and writes its triangles into a fixed
// slot (cell_idx * 5 + t).  With FMA contraction disabled (`--fmad=false`) the
// CPU and GPU vertex buffers are *bit-identical*.  This is the honest framing
// (in contrast to iLQR's bimodality caveat): the GPU is doing the same
// arithmetic in parallel, the win is throughput.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ------------------------------------------------------------------ constants
#define VOX_RES   128                   // voxels per axis (128^3 = 2.1M)
static const int   N_VOX     = VOX_RES * VOX_RES * VOX_RES;
static const int   N_CELLS_X = VOX_RES - 1;
static const int   N_CELLS   = N_CELLS_X * N_CELLS_X * N_CELLS_X;
static const int   MAX_TRI   = 5;       // MC emits at most 5 triangles per cell
static const int   SLOT_F    = MAX_TRI * 9;  // 5 tris * 3 verts * 3 coords

// volume axis-aligned bounds (metres)
static const float GMIN_X = -2.5f, GMIN_Y = -2.5f, GMIN_Z = -1.5f;
static const float GSPAN  = 5.0f;
static const float VOXSZ  = GSPAN / (VOX_RES - 1);   // grid spacing
static const float ISO    = 0.0f;

static const int   PANEL_W = 760;
static const int   PANEL_H = 600;

// --------------------------------------------------------------- scene SDF
// Same 3-sphere "snowman" used by the TSDF demo, so the meshes are directly
// recognisable side-by-side.
__host__ __device__ static inline float scene_sdf(float x, float y, float z) {
    auto sph = [](float x, float y, float z, float cx, float cy, float cz, float r) {
        float dx = x - cx, dy = y - cy, dz = z - cz;
        return sqrtf(dx*dx + dy*dy + dz*dz) - r;
    };
    float d = z - (-1.3f);                                           // ground
    float s1 = sph(x, y, z, 0.0f, 0.0f, -0.1f, 1.20f);                // body
    float s2 = sph(x, y, z, 0.0f, 0.0f,  1.45f, 0.80f);               // torso
    float s3 = sph(x, y, z, 0.0f, 0.0f,  2.55f, 0.52f);               // head
    d = fminf(d, s1); d = fminf(d, s2); d = fminf(d, s3);
    return d;
}

// ----------------------------------------------------- MC edge & triangle table
// 256-entry table from Paul Bourke's public-domain reference
// (http://paulbourke.net/geometry/polygonise/).  edge_table[i] is a 12-bit mask
// of which cube edges are crossed by the iso-surface for configuration `i`;
// tri_table[i] lists the edges of up to 5 triangles, terminated by -1.

__host__ __device__ static const int EDGE_TABLE[256] = {
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

__host__ __device__ static const int TRI_TABLE[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
    {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
    {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
    {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
    {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
    {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
    {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
    {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
    {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
    {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
    {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
    {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
    {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
    {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
    {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
    {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
    {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
    {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
    {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
    {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
    {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
    {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
    {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
    {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
    {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
    {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
    {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
    {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
    {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
    {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
    {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
    {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
    {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
    {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
    {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
    {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
    {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
    {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
    {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
    {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
    {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
    {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
    {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
    {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
    {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
    {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
    {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
    {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
    {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
    {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
    {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
    {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
    {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
    {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
    {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
    {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
    {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
    {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
    {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
    {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
    {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
    {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
    {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
    {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
    {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
    {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
    {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
    {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
    {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
    {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
    {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
    {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
    {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
    {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
    {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
    {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
    {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
    {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
    {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
    {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
    {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
    {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
    {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
    {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
    {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
    {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
    {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
    {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
    {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
    {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
    {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
    {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
    {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
    {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
    {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
    {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
    {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
    {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
    {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
    {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
    {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
    {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
    {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
    {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
    {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
    {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
    {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
    {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
    {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
    {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
    {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
    {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
    {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
    {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
    {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
    {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
    {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
    {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
    {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
    {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
    {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
    {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
    {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
    {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
    {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
    {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
    {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
    {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
    {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
    {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
    {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
    {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
    {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
    {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
    {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
    {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
    {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
    {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
    {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
    {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
    {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
    {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
    {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};

// ------------------------------------------------------------------ corner map
// For configuration bit `b`, the cube corner is `CORNER[b]` in unit-cube
// coordinates (0 or 1 along x, y, z).  This must match the edge encoding in
// EDGE_VTX below.
__host__ __device__ static const int CORNER_DX[8] = {0, 1, 1, 0, 0, 1, 1, 0};
__host__ __device__ static const int CORNER_DY[8] = {0, 0, 1, 1, 0, 0, 1, 1};
__host__ __device__ static const int CORNER_DZ[8] = {0, 0, 0, 0, 1, 1, 1, 1};

// Each edge connects two corners.  Edge 0 = v0-v1, edge 1 = v1-v2, ..., as in
// Bourke's reference.
__host__ __device__ static const int EDGE_VTX[12][2] = {
    {0,1},{1,2},{2,3},{3,0},
    {4,5},{5,6},{6,7},{7,4},
    {0,4},{1,5},{2,6},{3,7}
};

// ---------------------------------------------------------- shared MC routine
// One call processes one cell at (ci, cj, ck) and writes its triangles into a
// fixed slot (cell_idx * SLOT_F).  `out_n` records the actual triangle count
// for that cell.  Unused floats are filled with a NaN sentinel so the two
// volumes are byte-comparable.
__host__ __device__ static inline void mc_cell(
        int ci, int cj, int ck,
        const float* __restrict__ sdf,    // VOX_RES^3 floats
        float* __restrict__ slot_out,     // SLOT_F floats
        int*   __restrict__ ntri_out) {

    // Gather 8 corner SDF values and the 8-bit configuration.
    float corner_v[8];
    int   cube_idx = 0;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        int x = ci + CORNER_DX[c];
        int y = cj + CORNER_DY[c];
        int z = ck + CORNER_DZ[c];
        float v = sdf[(z * VOX_RES + y) * VOX_RES + x];
        corner_v[c] = v;
        if (v < ISO) cube_idx |= (1 << c);
    }

    // Fill all 45 slot floats with a NaN sentinel first.
    // (Bit pattern 0x7fc00000 = quiet NaN; both CPU and GPU read it identically.)
    const float NAN_F = nanf("");
    #pragma unroll
    for (int s = 0; s < SLOT_F; ++s) slot_out[s] = NAN_F;
    *ntri_out = 0;

    int em = EDGE_TABLE[cube_idx];
    if (em == 0) return;

    // For each of the 12 edges that is crossed, compute its iso-vertex once.
    // World-space placement: corner_pos = GMIN + (cell_idx + corner_d) * VOXSZ.
    float ex[12], ey[12], ez[12];
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        if (!(em & (1 << e))) continue;
        int a = EDGE_VTX[e][0], b = EDGE_VTX[e][1];
        float va = corner_v[a], vb = corner_v[b];
        float ax = GMIN_X + (ci + CORNER_DX[a]) * VOXSZ;
        float ay = GMIN_Y + (cj + CORNER_DY[a]) * VOXSZ;
        float az = GMIN_Z + (ck + CORNER_DZ[a]) * VOXSZ;
        float bx = GMIN_X + (ci + CORNER_DX[b]) * VOXSZ;
        float by = GMIN_Y + (cj + CORNER_DY[b]) * VOXSZ;
        float bz = GMIN_Z + (ck + CORNER_DZ[b]) * VOXSZ;
        float denom = vb - va;
        // Deterministic linear interpolation.  Both CPU and GPU use the same
        // arithmetic with --fmad=false, so the result is bit-identical.
        float t = (fabsf(denom) < 1e-12f) ? 0.5f : (ISO - va) / denom;
        ex[e] = ax + t * (bx - ax);
        ey[e] = ay + t * (by - ay);
        ez[e] = az + t * (bz - az);
    }

    int n = 0;
    #pragma unroll
    for (int i = 0; i < 16; i += 3) {
        int e0 = TRI_TABLE[cube_idx][i];
        if (e0 < 0) break;
        int e1 = TRI_TABLE[cube_idx][i + 1];
        int e2 = TRI_TABLE[cube_idx][i + 2];
        int base = n * 9;
        slot_out[base + 0] = ex[e0]; slot_out[base + 1] = ey[e0]; slot_out[base + 2] = ez[e0];
        slot_out[base + 3] = ex[e1]; slot_out[base + 4] = ey[e1]; slot_out[base + 5] = ez[e1];
        slot_out[base + 6] = ex[e2]; slot_out[base + 7] = ey[e2]; slot_out[base + 8] = ez[e2];
        ++n;
    }
    *ntri_out = n;
}

__global__ static void mc_kernel(const float* sdf, float* slots, int* ntri) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_CELLS) return;
    int ci = idx % N_CELLS_X;
    int cj = (idx / N_CELLS_X) % N_CELLS_X;
    int ck = idx / (N_CELLS_X * N_CELLS_X);
    mc_cell(ci, cj, ck, sdf, slots + idx * SLOT_F, ntri + idx);
}

static void mc_cpu(const float* sdf, float* slots, int* ntri) {
    for (int ck = 0; ck < N_CELLS_X; ++ck)
        for (int cj = 0; cj < N_CELLS_X; ++cj)
            for (int ci = 0; ci < N_CELLS_X; ++ci) {
                int idx = (ck * N_CELLS_X + cj) * N_CELLS_X + ci;
                mc_cell(ci, cj, ck, sdf, slots + idx * SLOT_F, ntri + idx);
            }
}

// ------------------------------------------------------------------ rendering
struct Cam { float yaw, pitch, dist; };
static cv::Point2i project(float x, float y, float z, Cam c, int W, int H) {
    float cx = std::cos(c.yaw) * x - std::sin(c.yaw) * y;
    float cy = std::sin(c.yaw) * x + std::cos(c.yaw) * y;
    float cz = z;
    float yy = std::cos(c.pitch) * cy - std::sin(c.pitch) * cz;
    float zz = std::sin(c.pitch) * cy + std::cos(c.pitch) * cz + c.dist;
    if (zz < 0.1f) zz = 0.1f;
    float f = 360.0f;
    int u = static_cast<int>(W * 0.5f + f * cx / zz);
    int v = static_cast<int>(H * 0.5f - f * yy / zz);
    return cv::Point2i(u, v);
}

struct Tri { float x[3], y[3], z[3]; float cz; cv::Scalar col; };

static void draw_mesh(cv::Mat& img, const std::vector<float>& slots,
                      const std::vector<int>& ntri, Cam cam) {
    std::vector<Tri> tris;
    tris.reserve(N_CELLS);
    for (int idx = 0; idx < N_CELLS; ++idx) {
        for (int t = 0; t < ntri[idx]; ++t) {
            const float* p = slots.data() + (idx * MAX_TRI + t) * 9;
            Tri tr;
            float cxs = 0.0f, cys = 0.0f, czs = 0.0f;
            for (int k = 0; k < 3; ++k) {
                tr.x[k] = p[k*3+0]; tr.y[k] = p[k*3+1]; tr.z[k] = p[k*3+2];
                cxs += tr.x[k]; cys += tr.y[k]; czs += tr.z[k];
            }
            cxs /= 3; cys /= 3; czs /= 3;
            float cam_y = std::sin(cam.yaw) * cxs + std::cos(cam.yaw) * cys;
            tr.cz = std::cos(cam.pitch) * cam_y - std::sin(cam.pitch) * czs;
            // shade by triangle normal (flat) — diffuse vs head-on light
            float ex1 = tr.x[1] - tr.x[0], ey1 = tr.y[1] - tr.y[0], ez1 = tr.z[1] - tr.z[0];
            float ex2 = tr.x[2] - tr.x[0], ey2 = tr.y[2] - tr.y[0], ez2 = tr.z[2] - tr.z[0];
            float nx = ey1*ez2 - ez1*ey2;
            float ny = ez1*ex2 - ex1*ez2;
            float nz = ex1*ey2 - ey1*ex2;
            float ln = std::sqrt(nx*nx + ny*ny + nz*nz) + 1e-9f;
            nx /= ln; ny /= ln; nz /= ln;
            // light from camera direction
            float lx = std::cos(cam.yaw),  ly = std::sin(cam.yaw),  lz = 0.3f;
            float lnrm = std::sqrt(lx*lx + ly*ly + lz*lz);
            lx/=lnrm; ly/=lnrm; lz/=lnrm;
            float d = nx*lx + ny*ly + nz*lz;
            if (d < 0) d = -d;
            d = 0.25f + 0.75f * d;
            tr.col = cv::Scalar(120.0f * d + 60.0f * (1-d),
                                180.0f * d + 80.0f * (1-d),
                                240.0f * d + 100.0f * (1-d));
            tris.push_back(tr);
        }
    }
    // painter's order: far triangles first
    std::sort(tris.begin(), tris.end(),
              [](const Tri& a, const Tri& b){ return a.cz > b.cz; });
    for (const Tri& tr : tris) {
        cv::Point2i q0 = project(tr.x[0], tr.y[0], tr.z[0], cam, img.cols, img.rows);
        cv::Point2i q1 = project(tr.x[1], tr.y[1], tr.z[1], cam, img.cols, img.rows);
        cv::Point2i q2 = project(tr.x[2], tr.y[2], tr.z[2], cam, img.cols, img.rows);
        std::vector<cv::Point> poly = {q0, q1, q2};
        cv::fillConvexPoly(img, poly, tr.col, cv::LINE_AA);
        cv::line(img, q0, q1, cv::Scalar(20,20,30), 1, cv::LINE_AA);
        cv::line(img, q1, q2, cv::Scalar(20,20,30), 1, cv::LINE_AA);
        cv::line(img, q2, q0, cv::Scalar(20,20,30), 1, cv::LINE_AA);
    }
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::printf("GPU Marching Cubes: %d^3 voxels (%d cells), iso=%.2f\n",
                VOX_RES, N_CELLS, ISO);

    // --- build SDF volume from analytic scene ------------------------------
    std::vector<float> sdf(N_VOX);
    for (int k = 0; k < VOX_RES; ++k)
        for (int j = 0; j < VOX_RES; ++j)
            for (int i = 0; i < VOX_RES; ++i) {
                float x = GMIN_X + i * VOXSZ;
                float y = GMIN_Y + j * VOXSZ;
                float z = GMIN_Z + k * VOXSZ;
                sdf[(k * VOX_RES + j) * VOX_RES + i] = scene_sdf(x, y, z);
            }

    // --- CPU MC (timed) -----------------------------------------------------
    std::vector<float> slots_cpu(N_CELLS * SLOT_F);
    std::vector<int>   ntri_cpu(N_CELLS, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    mc_cpu(sdf.data(), slots_cpu.data(), ntri_cpu.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU MC (timed) -----------------------------------------------------
    float *d_sdf, *d_slots;
    int   *d_ntri;
    CUDA_CHECK(cudaMalloc(&d_sdf,   sdf.size()       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_slots, slots_cpu.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ntri,  N_CELLS          * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sdf, sdf.data(), sdf.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    int block = 128, grid = (N_CELLS + block - 1) / block;
    mc_kernel<<<grid, block>>>(d_sdf, d_slots, d_ntri);   // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    mc_kernel<<<grid, block>>>(d_sdf, d_slots, d_ntri);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    std::vector<float> slots_gpu(N_CELLS * SLOT_F);
    std::vector<int>   ntri_gpu(N_CELLS);
    CUDA_CHECK(cudaMemcpy(slots_gpu.data(), d_slots,
                          slots_gpu.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ntri_gpu.data(), d_ntri,
                          N_CELLS * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // --- compare CPU vs GPU -------------------------------------------------
    long long ntri_total_cpu = 0, ntri_total_gpu = 0;
    for (int i = 0; i < N_CELLS; ++i) {
        ntri_total_cpu += ntri_cpu[i];
        ntri_total_gpu += ntri_gpu[i];
    }
    int n_ntri_diff = 0;
    for (int i = 0; i < N_CELLS; ++i)
        if (ntri_cpu[i] != ntri_gpu[i]) ++n_ntri_diff;

    double max_diff = 0.0;
    long long n_compared = 0;
    for (int idx = 0; idx < N_CELLS; ++idx) {
        int n = ntri_cpu[idx];
        for (int s = 0; s < n * 9; ++s) {
            float a = slots_cpu[idx * SLOT_F + s];
            float b = slots_gpu[idx * SLOT_F + s];
            double d = std::fabs((double)a - (double)b);
            if (d > max_diff) max_diff = d;
            ++n_compared;
        }
    }

    double speedup = cpu_ms / gpu_ms;
    std::printf("CPU MC %.1f ms,  GPU MC %.3f ms  -> %.0fx\n",
                cpu_ms, gpu_ms, speedup);
    std::printf("triangles: CPU %lld,  GPU %lld   (cells with mismatched n_tri: %d)\n",
                ntri_total_cpu, ntri_total_gpu, n_ntri_diff);
    std::printf("vertex max|diff| %.3e  over %lld floats\n",
                max_diff, n_compared);

    // --- animation: rotate the GPU mesh ------------------------------------
    if (system("mkdir -p tmp") != 0)
        std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_marching_cubes.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          12, cv::Size(PANEL_W, PANEL_H));

    const int N_FRAMES = 36;
    for (int f = 0; f < N_FRAMES; ++f) {
        float a = 2.0f * static_cast<float>(M_PI) * f / N_FRAMES;
        Cam cam{a, 0.35f, 13.0f};

        cv::Mat img(PANEL_H, PANEL_W, CV_8UC3, cv::Scalar(18, 18, 22));
        draw_mesh(img, slots_gpu, ntri_gpu, cam);

        cv::putText(img, "GPU Marching Cubes (one thread = one cell)",
                    cv::Point(12, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
        char l1[160], l2[160], l3[160];
        std::snprintf(l1, sizeof(l1),
                      "%d^3 voxels -> %lld triangles  (iso=%.2f)",
                      VOX_RES, ntri_total_gpu, ISO);
        std::snprintf(l2, sizeof(l2),
                      "extract %d cells:  CPU %.0f ms  vs  GPU %.2f ms  (%.0fx)",
                      N_CELLS, cpu_ms, gpu_ms, speedup);
        std::snprintf(l3, sizeof(l3),
                      "CPU/GPU vertex max|diff| %.1e  (deterministic MC, --fmad=false)",
                      max_diff);
        cv::putText(img, l1, cv::Point(12, PANEL_H - 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 220, 255), 1, cv::LINE_AA);
        cv::putText(img, l2, cv::Point(12, PANEL_H - 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 255, 200), 1, cv::LINE_AA);
        cv::putText(img, l3, cv::Point(12, PANEL_H - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        video.write(img);
    }
    video.release();

    cudabot::avi_to_gif("tmp/gpu_marching_cubes.avi",
                        "gif/gpu_marching_cubes.gif", 12, 760);
    std::printf("wrote gif/gpu_marching_cubes.gif\n");

    CUDA_CHECK(cudaFree(d_sdf));
    CUDA_CHECK(cudaFree(d_slots));
    CUDA_CHECK(cudaFree(d_ntri));
    return 0;
}
