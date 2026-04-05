/*************************************************************************
    Point Cloud Processing Benchmark - CPU vs GPU
    Generates synthetic room point cloud and benchmarks:
      - Voxel Grid Filter
      - Statistical Outlier Removal
      - Normal Estimation
      - GICP
      - RANSAC Plane Detection
 ************************************************************************/

#include "cuda_pointcloud.cuh"
#include "ply_loader.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cfloat>
#include <unordered_map>

using namespace cudabot;

// =====================================================================
// Room point cloud generation
// =====================================================================
// Room: 8m x 6m x 3m, sample points on walls, floor, ceiling

static void generate_room_pointcloud(std::vector<PointXYZ>& points, int n_target,
                                     float noise_sigma, float outlier_ratio,
                                     std::mt19937& rng) {
    points.clear();
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, noise_sigma);

    float W = 8.0f, H = 6.0f, D = 3.0f;

    // Surface areas: 2 walls WxD, 2 walls HxD, floor+ceiling WxH
    float area_wx = W * D;     // 2 of these
    float area_hx = H * D;     // 2 of these
    float area_wh = W * H;     // 2 of these (floor+ceiling)
    float total = 2 * area_wx + 2 * area_hx + 2 * area_wh;

    int n_surface = (int)(n_target * (1.0f - outlier_ratio));

    for (int i = 0; i < n_surface; i++) {
        float r = u01(rng) * total;
        float px, py, pz;

        if (r < area_wx) {
            // Wall y=0
            px = u01(rng) * W; py = 0; pz = u01(rng) * D;
        } else if (r < 2 * area_wx) {
            // Wall y=H
            px = u01(rng) * W; py = H; pz = u01(rng) * D;
        } else if (r < 2 * area_wx + area_hx) {
            // Wall x=0
            px = 0; py = u01(rng) * H; pz = u01(rng) * D;
        } else if (r < 2 * area_wx + 2 * area_hx) {
            // Wall x=W
            px = W; py = u01(rng) * H; pz = u01(rng) * D;
        } else if (r < 2 * area_wx + 2 * area_hx + area_wh) {
            // Floor z=0
            px = u01(rng) * W; py = u01(rng) * H; pz = 0;
        } else {
            // Ceiling z=D
            px = u01(rng) * W; py = u01(rng) * H; pz = D;
        }

        px += noise(rng);
        py += noise(rng);
        pz += noise(rng);

        points.push_back({px, py, pz});
    }

    // Add outliers
    int n_outliers = n_target - n_surface;
    std::uniform_real_distribution<float> ox(-2.0f, W + 2.0f);
    std::uniform_real_distribution<float> oy(-2.0f, H + 2.0f);
    std::uniform_real_distribution<float> oz(-2.0f, D + 2.0f);
    for (int i = 0; i < n_outliers; i++) {
        points.push_back({ox(rng), oy(rng), oz(rng)});
    }
}

// Transform point cloud: R (30 deg around Z) + t
static void transform_pointcloud(const std::vector<PointXYZ>& in,
                                 std::vector<PointXYZ>& out,
                                 float angle_deg, float tx, float ty, float tz) {
    float a = angle_deg * 3.14159265f / 180.0f;
    float ca = cosf(a), sa = sinf(a);
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        out[i].x = ca * in[i].x - sa * in[i].y + tx;
        out[i].y = sa * in[i].x + ca * in[i].y + ty;
        out[i].z = in[i].z + tz;
    }
}

// =====================================================================
// CPU reference implementations
// =====================================================================

static void cpu_voxel_grid_filter(const std::vector<PointXYZ>& in,
                                  std::vector<PointXYZ>& out, float leaf) {
    // Simple hash-based voxel grid
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    for (auto& p : in) {
        min_x = fminf(min_x, p.x); min_y = fminf(min_y, p.y); min_z = fminf(min_z, p.z);
    }

    struct VoxelData { double sx, sy, sz; int count; };
    std::unordered_map<long long, VoxelData> voxels;

    for (auto& p : in) {
        int ix = (int)floorf((p.x - min_x) / leaf);
        int iy = (int)floorf((p.y - min_y) / leaf);
        int iz = (int)floorf((p.z - min_z) / leaf);
        long long key = (long long)ix + (long long)iy * 100000LL + (long long)iz * 10000000000LL;
        auto& v = voxels[key];
        v.sx += p.x; v.sy += p.y; v.sz += p.z; v.count++;
    }

    out.clear();
    out.reserve(voxels.size());
    for (auto& kv : voxels) {
        auto& v = kv.second;
        float inv = 1.0f / v.count;
        out.push_back({(float)(v.sx * inv), (float)(v.sy * inv), (float)(v.sz * inv)});
    }
}

static void cpu_statistical_outlier_removal(const std::vector<PointXYZ>& in,
                                            std::vector<PointXYZ>& out,
                                            int k, float std_mul) {
    int n = (int)in.size();
    std::vector<float> mean_dists(n);

    for (int i = 0; i < n; i++) {
        // k-NN brute force
        std::vector<float> dists;
        dists.reserve(n);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            float dx = in[i].x - in[j].x;
            float dy = in[i].y - in[j].y;
            float dz = in[i].z - in[j].z;
            dists.push_back(sqrtf(dx*dx + dy*dy + dz*dz));
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        float sum = 0;
        for (int j = 0; j < k; j++) sum += dists[j];
        mean_dists[i] = sum / k;
    }

    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) { sum += mean_dists[i]; sum2 += (double)mean_dists[i] * mean_dists[i]; }
    float gmean = (float)(sum / n);
    float gstd = sqrtf((float)(sum2 / n - (double)gmean * gmean));
    float thresh = gmean + std_mul * gstd;

    out.clear();
    for (int i = 0; i < n; i++) {
        if (mean_dists[i] <= thresh) out.push_back(in[i]);
    }
}

static void cpu_estimate_normals(const std::vector<PointXYZ>& in,
                                 std::vector<float>& nx, std::vector<float>& ny, std::vector<float>& nz,
                                 int k) {
    int n = (int)in.size();
    nx.resize(n); ny.resize(n); nz.resize(n);

    for (int i = 0; i < n; i++) {
        // k-NN
        std::vector<std::pair<float, int>> dists;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            float dx = in[i].x - in[j].x;
            float dy = in[i].y - in[j].y;
            float dz = in[i].z - in[j].z;
            dists.push_back({dx*dx + dy*dy + dz*dz, j});
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        // Centroid
        float cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < k; j++) {
            int idx = dists[j].second;
            cx += in[idx].x; cy += in[idx].y; cz += in[idx].z;
        }
        float inv_k = 1.0f / k;
        cx *= inv_k; cy *= inv_k; cz *= inv_k;

        // Covariance
        float c00=0,c01=0,c02=0,c11=0,c12=0,c22=0;
        for (int j = 0; j < k; j++) {
            int idx = dists[j].second;
            float dx = in[idx].x - cx, dy = in[idx].y - cy, dz = in[idx].z - cz;
            c00 += dx*dx; c01 += dx*dy; c02 += dx*dz;
            c11 += dy*dy; c12 += dy*dz; c22 += dz*dz;
        }

        // Power iteration for smallest eigenvector (inverse iteration)
        // Simple: just use the cross-product method
        float p1 = c01*c01 + c02*c02 + c12*c12;
        float trace = c00 + c11 + c22;
        float q = trace / 3.0f;
        float b00 = c00 - q, b11 = c11 - q, b22 = c22 - q;
        float p2 = b00*b00 + b11*b11 + b22*b22 + 2*p1;
        float p = sqrtf(p2 / 6.0f);

        float eig0;
        if (p < 1e-12f) {
            eig0 = fminf(fminf(c00,c11),c22);
        } else {
            float inv_p = 1.0f/p;
            float cc00=b00*inv_p,cc01=c01*inv_p,cc02=c02*inv_p;
            float cc11=b11*inv_p,cc12=c12*inv_p,cc22=b22*inv_p;
            float detB = cc00*(cc11*cc22-cc12*cc12) - cc01*(cc01*cc22-cc12*cc02) + cc02*(cc01*cc12-cc11*cc02);
            float half = fmaxf(-1.0f, fminf(1.0f, detB*0.5f));
            float phi = acosf(half)/3.0f;
            eig0 = q + 2*p*cosf(phi + 2*3.14159265f/3.0f);
        }

        float m00=c00-eig0, m01=c01, m02=c02;
        float m10=c01, m11=c11-eig0, m12=c12;
        float m20=c02, m21=c12, m22=c22-eig0;

        float cx0=m01*m12-m02*m11, cy0=m02*m10-m00*m12, cz0=m00*m11-m01*m10; float l0=cx0*cx0+cy0*cy0+cz0*cz0;
        float cx1=m01*m22-m02*m21, cy1=m02*m20-m00*m22, cz1=m00*m21-m01*m20; float l1=cx1*cx1+cy1*cy1+cz1*cz1;
        float cx2=m11*m22-m12*m21, cy2=m12*m20-m10*m22, cz2=m10*m21-m11*m20; float l2=cx2*cx2+cy2*cy2+cz2*cz2;

        float rx,ry,rz;
        if (l0>=l1 && l0>=l2) { float inv=1.0f/sqrtf(l0+1e-20f); rx=cx0*inv; ry=cy0*inv; rz=cz0*inv; }
        else if (l1>=l0 && l1>=l2) { float inv=1.0f/sqrtf(l1+1e-20f); rx=cx1*inv; ry=cy1*inv; rz=cz1*inv; }
        else { float inv=1.0f/sqrtf(l2+1e-20f); rx=cx2*inv; ry=cy2*inv; rz=cz2*inv; }

        nx[i]=rx; ny[i]=ry; nz[i]=rz;
    }
}

static void cpu_ransac_plane(const std::vector<PointXYZ>& in, float* coeffs,
                             float dist_thresh, int max_iter) {
    int n = (int)in.size();
    std::mt19937 rng(42);
    int best_count = 0;
    float best[4] = {0,0,0,0};

    for (int iter = 0; iter < max_iter; iter++) {
        int i0 = rng() % n, i1 = rng() % n, i2 = rng() % n;
        while (i1 == i0) i1 = rng() % n;
        while (i2 == i0 || i2 == i1) i2 = rng() % n;

        float v1x = in[i1].x-in[i0].x, v1y = in[i1].y-in[i0].y, v1z = in[i1].z-in[i0].z;
        float v2x = in[i2].x-in[i0].x, v2y = in[i2].y-in[i0].y, v2z = in[i2].z-in[i0].z;
        float a = v1y*v2z - v1z*v2y;
        float b = v1z*v2x - v1x*v2z;
        float c = v1x*v2y - v1y*v2x;
        float norm = sqrtf(a*a+b*b+c*c);
        if (norm < 1e-10f) continue;
        a/=norm; b/=norm; c/=norm;
        float d = -(a*in[i0].x+b*in[i0].y+c*in[i0].z);

        int count = 0;
        for (int j = 0; j < n; j++) {
            if (fabsf(a*in[j].x+b*in[j].y+c*in[j].z+d) <= dist_thresh) count++;
        }
        if (count > best_count) { best_count = count; best[0]=a; best[1]=b; best[2]=c; best[3]=d; }
    }
    coeffs[0]=best[0]; coeffs[1]=best[1]; coeffs[2]=best[2]; coeffs[3]=best[3];
}

// CPU ICP (point-to-point, simplified)
static void cpu_icp(const std::vector<PointXYZ>& src, const std::vector<PointXYZ>& tgt,
                    float* R_out, float* t_out, int max_iter) {
    int n_src = (int)src.size(), n_tgt = (int)tgt.size();
    std::vector<PointXYZ> cur = src;

    for (int iter = 0; iter < max_iter; iter++) {
        // Find correspondences
        std::vector<int> corr(n_src);
        for (int i = 0; i < n_src; i++) {
            float best = 1e30f; int bi = 0;
            for (int j = 0; j < n_tgt; j++) {
                float dx = cur[i].x-tgt[j].x, dy = cur[i].y-tgt[j].y, dz = cur[i].z-tgt[j].z;
                float d2 = dx*dx+dy*dy+dz*dz;
                if (d2 < best) { best = d2; bi = j; }
            }
            corr[i] = bi;
        }

        // Centroids
        float cx_s=0,cy_s=0,cz_s=0, cx_t=0,cy_t=0,cz_t=0;
        for (int i = 0; i < n_src; i++) {
            cx_s+=cur[i].x; cy_s+=cur[i].y; cz_s+=cur[i].z;
            cx_t+=tgt[corr[i]].x; cy_t+=tgt[corr[i]].y; cz_t+=tgt[corr[i]].z;
        }
        float inv = 1.0f / n_src;
        cx_s*=inv; cy_s*=inv; cz_s*=inv;
        cx_t*=inv; cy_t*=inv; cz_t*=inv;

        // Cross-covariance W
        float W[9] = {0};
        for (int i = 0; i < n_src; i++) {
            float sx=cur[i].x-cx_s, sy=cur[i].y-cy_s, sz=cur[i].z-cz_s;
            float tx=tgt[corr[i]].x-cx_t, ty=tgt[corr[i]].y-cy_t, tz=tgt[corr[i]].z-cz_t;
            W[0]+=sx*tx; W[1]+=sx*ty; W[2]+=sx*tz;
            W[3]+=sy*tx; W[4]+=sy*ty; W[5]+=sy*tz;
            W[6]+=sz*tx; W[7]+=sz*ty; W[8]+=sz*tz;
        }

        // Simple rotation approximation (just used for timing, not accuracy)
        // Apply identity + small correction for benchmark purposes
        for (int i = 0; i < n_src; i++) {
            // Just apply centroid shift for rough convergence
            cur[i].x += (cx_t - cx_s) * 0.5f;
            cur[i].y += (cy_t - cy_s) * 0.5f;
            cur[i].z += (cz_t - cz_s) * 0.5f;
        }
    }

    // Output identity (this is just for timing)
    for (int i = 0; i < 9; i++) R_out[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    t_out[0] = t_out[1] = t_out[2] = 0;
}

// =====================================================================
// Timer helper
// =====================================================================

struct Timer {
    std::chrono::high_resolution_clock::time_point start_;
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// =====================================================================
// Main
// =====================================================================

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    printf("=============================================================\n");
    printf("  CudaPointCloud Benchmark: CPU vs GPU\n");
    printf("=============================================================\n\n");

    // Parse optional file inputs: --ply path.ply or --kitti path.bin
    std::vector<std::pair<std::string, std::vector<PointXYZ>>> file_clouds;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--ply" || arg == "--kitti" || arg == "--xyz") && i + 1 < argc) {
            std::string path = argv[++i];
            std::vector<PointXYZ> pts;
            bool ok = false;
            if (arg == "--ply") ok = load_ply(path, pts);
            else if (arg == "--kitti") ok = load_kitti_bin(path, pts);
            else if (arg == "--xyz") ok = load_xyz(path, pts);
            if (ok && !pts.empty()) file_clouds.push_back({path, std::move(pts)});
        }
    }

    std::vector<int> sizes = {2000, 5000, 10000, 20000};
    std::mt19937 rng(12345);

    // Print header
    printf("%-12s | %-18s | %12s | %12s | %8s\n",
           "Points", "Operation", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("-------------|--------------------|--------------|--------------|---------\n");

    for (int N : sizes) {
        std::vector<PointXYZ> cloud;
        generate_room_pointcloud(cloud, N, 0.01f, 0.05f, rng);

        // Reduce k for large point clouds to keep runtime reasonable
        int k_normal = 20;
        int k_stat = 20;
        int ransac_iters = 500;
        int gicp_iters = 10;

        // Upload to GPU
        CudaPointCloud gpu_cloud;
        gpu_cloud.upload(cloud);

        Timer timer;
        double cpu_ms, gpu_ms;

        // --- Voxel Grid Filter ---
        {
            std::vector<PointXYZ> cpu_out;
            timer.start();
            cpu_voxel_grid_filter(cloud, cpu_out, 0.1f);
            cpu_ms = timer.elapsed_ms();

            // Warm up
            auto tmp = voxel_grid_filter(gpu_cloud, 0.1f);

            timer.start();
            auto gpu_out = voxel_grid_filter(gpu_cloud, 0.1f);
            gpu_ms = timer.elapsed_ms();

            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n",
                   N, "VoxelGrid", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        }

        // --- Statistical Outlier Removal ---
        if (N <= 5000) {
            std::vector<PointXYZ> cpu_out;
            timer.start();
            cpu_statistical_outlier_removal(cloud, cpu_out, k_stat, 1.0f);
            cpu_ms = timer.elapsed_ms();

            auto tmp = statistical_outlier_removal(gpu_cloud, k_stat, 1.0f);

            timer.start();
            auto gpu_out = statistical_outlier_removal(gpu_cloud, k_stat, 1.0f);
            gpu_ms = timer.elapsed_ms();

            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n",
                   N, "StatisticalFilter", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        }

        // --- Normal Estimation ---
        if (N <= 5000) {
            std::vector<float> cpu_nx, cpu_ny, cpu_nz;
            timer.start();
            cpu_estimate_normals(cloud, cpu_nx, cpu_ny, cpu_nz, k_normal);
            cpu_ms = timer.elapsed_ms();

            float *d_nx, *d_ny, *d_nz;
            CUDA_CHECK(cudaMalloc(&d_nx, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_ny, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_nz, N * sizeof(float)));

            estimate_normals(gpu_cloud, d_nx, d_ny, d_nz, k_normal);

            timer.start();
            estimate_normals(gpu_cloud, d_nx, d_ny, d_nz, k_normal);
            gpu_ms = timer.elapsed_ms();

            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n",
                   N, "NormalEstimation", cpu_ms, gpu_ms, cpu_ms / gpu_ms);

            cudaFree(d_nx); cudaFree(d_ny); cudaFree(d_nz);
        }

        // --- RANSAC Plane ---
        {
            float cpu_coeffs[4], gpu_coeffs[4];
            timer.start();
            cpu_ransac_plane(cloud, cpu_coeffs, 0.05f, ransac_iters);
            cpu_ms = timer.elapsed_ms();

            ransac_plane(gpu_cloud, gpu_coeffs, 0.05f, ransac_iters);

            timer.start();
            ransac_plane(gpu_cloud, gpu_coeffs, 0.05f, ransac_iters);
            gpu_ms = timer.elapsed_ms();

            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n",
                   N, "RANSAC Plane", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        }

        // --- GICP ---
        if (N <= 10000) {
            std::vector<PointXYZ> src_cloud;
            transform_pointcloud(cloud, src_cloud, 30.0f, 1.0f, 0.5f, 0.2f);

            // Use subset for GICP to keep timing reasonable
            int gicp_n = std::min(N, 2000);
            std::vector<PointXYZ> tgt_sub(cloud.begin(), cloud.begin() + gicp_n);
            std::vector<PointXYZ> src_sub(src_cloud.begin(), src_cloud.begin() + gicp_n);

            float cpu_R[9], cpu_t[3];
            timer.start();
            cpu_icp(src_sub, tgt_sub, cpu_R, cpu_t, gicp_iters);
            cpu_ms = timer.elapsed_ms();

            CudaPointCloud gpu_src, gpu_tgt;
            gpu_src.upload(src_sub);
            gpu_tgt.upload(tgt_sub);

            float gpu_R[9], gpu_t[3];
            gicp_align(gpu_src, gpu_tgt, gpu_R, gpu_t, gicp_iters, 1e-4f);

            timer.start();
            gpu_src.upload(src_sub);
            gicp_align(gpu_src, gpu_tgt, gpu_R, gpu_t, gicp_iters, 1e-4f);
            gpu_ms = timer.elapsed_ms();

            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n",
                   gicp_n, "GICP", cpu_ms, gpu_ms, cpu_ms / gpu_ms);

            // Print GICP result
            printf("             |   R = [%6.3f %6.3f %6.3f]\n", gpu_R[0], gpu_R[1], gpu_R[2]);
            printf("             |       [%6.3f %6.3f %6.3f]\n", gpu_R[3], gpu_R[4], gpu_R[5]);
            printf("             |       [%6.3f %6.3f %6.3f]\n", gpu_R[6], gpu_R[7], gpu_R[8]);
            printf("             |   t = [%6.3f %6.3f %6.3f]\n", gpu_t[0], gpu_t[1], gpu_t[2]);
        }

        printf("-------------|--------------------|--------------|--------------|---------\n");
    }

    // =====================================================================
    // External file benchmarks (PLY, KITTI, XYZ)
    // =====================================================================
    for (size_t fi = 0; fi < file_clouds.size(); fi++) {
        const std::string& fpath = file_clouds[fi].first;
        const std::vector<PointXYZ>& cloud = file_clouds[fi].second;
        int N = (int)cloud.size();
        printf("\n=== External: %s (%d points) ===\n", fpath.c_str(), N);
        printf("%-12s | %-18s | %12s | %12s | %8s\n",
               "Points", "Operation", "CPU (ms)", "GPU (ms)", "Speedup");
        printf("-------------|--------------------|--------------|--------------|---------\n");

        CudaPointCloud gpu_cloud;
        gpu_cloud.upload(cloud);
        Timer timer;
        double cpu_ms, gpu_ms;

        // Voxel Grid
        {
            std::vector<PointXYZ> cpu_out;
            timer.start();
            cpu_voxel_grid_filter(cloud, cpu_out, 0.1f);
            cpu_ms = timer.elapsed_ms();
            auto tmp = voxel_grid_filter(gpu_cloud, 0.1f);
            timer.start();
            auto gpu_out = voxel_grid_filter(gpu_cloud, 0.1f);
            gpu_ms = timer.elapsed_ms();
            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n", N, "VoxelGrid", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        }

        // Normal Estimation (limit to 20K for CPU brute-force k-NN)
        if (N <= 20000) {
            int k = 20;
            std::vector<float> cpu_nx, cpu_ny, cpu_nz;
            timer.start();
            cpu_estimate_normals(cloud, cpu_nx, cpu_ny, cpu_nz, k);
            cpu_ms = timer.elapsed_ms();
            float *d_nx, *d_ny, *d_nz;
            CUDA_CHECK(cudaMalloc(&d_nx, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_ny, N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_nz, N * sizeof(float)));
            estimate_normals(gpu_cloud, d_nx, d_ny, d_nz, k);  // warmup
            timer.start();
            estimate_normals(gpu_cloud, d_nx, d_ny, d_nz, k);
            gpu_ms = timer.elapsed_ms();
            CUDA_CHECK(cudaFree(d_nx)); CUDA_CHECK(cudaFree(d_ny)); CUDA_CHECK(cudaFree(d_nz));
            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n", N, "NormalEstimation", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        }

        // RANSAC Plane
        {
            int iters = std::min(2000, std::max(500, N / 10));
            float cpu_plane[4];
            timer.start();
            cpu_ransac_plane(cloud, cpu_plane, 0.05f, iters);
            cpu_ms = timer.elapsed_ms();
            float plane[4];
            ransac_plane(gpu_cloud, plane, 0.05f, iters);  // warmup
            timer.start();
            ransac_plane(gpu_cloud, plane, 0.05f, iters);
            gpu_ms = timer.elapsed_ms();
            printf("%-12d | %-18s | %12.2f | %12.2f | %7.1fx\n", N, "RANSAC Plane", cpu_ms, gpu_ms, cpu_ms / gpu_ms);
        }

        printf("-------------|--------------------|--------------|--------------|---------\n");
    }

    printf("\nDone.\n");
    return 0;
}
