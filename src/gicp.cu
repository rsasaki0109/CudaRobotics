/*************************************************************************
    Generalized ICP (point-to-plane) - CUDA-parallelized
    Iteratively finds correspondences and solves for rigid transformation
    using point-to-plane error metric with SVD on host via Eigen.
 ************************************************************************/

#include "cuda_pointcloud.cuh"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>

namespace cudabot {

// Forward declaration
void estimate_normals(const CudaPointCloud& input, float* d_nx, float* d_ny, float* d_nz, int k);

// =====================================================================
// Kernels
// =====================================================================

// Find nearest neighbor in target for each source point
__global__ void gicp_find_correspondences_kernel(
    const float* src_x, const float* src_y, const float* src_z, int n_src,
    const float* tgt_x, const float* tgt_y, const float* tgt_z, int n_tgt,
    const float* tgt_nx, const float* tgt_ny, const float* tgt_nz,
    int* correspondences, float* distances)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src) return;

    float sx = src_x[i], sy = src_y[i], sz = src_z[i];
    float best_dist = 1e30f;
    int best_idx = 0;

    for (int j = 0; j < n_tgt; j++) {
        float dx = sx - tgt_x[j];
        float dy = sy - tgt_y[j];
        float dz = sz - tgt_z[j];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best_dist) {
            best_dist = d2;
            best_idx = j;
        }
    }

    correspondences[i] = best_idx;
    distances[i] = sqrtf(best_dist);
}

// Apply rigid transformation to source points: p' = R*p + t
__global__ void apply_transform_kernel(
    float* x, float* y, float* z, int n,
    float r00, float r01, float r02,
    float r10, float r11, float r12,
    float r20, float r21, float r22,
    float tx, float ty, float tz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = x[i], py = y[i], pz = z[i];
    x[i] = r00 * px + r01 * py + r02 * pz + tx;
    y[i] = r10 * px + r11 * py + r12 * pz + ty;
    z[i] = r20 * px + r21 * py + r22 * pz + tz;
}

// Compute point-to-plane JtJ and Jtr for least-squares
// Each thread contributes one correspondence to the 6x6 system
// We accumulate on host for simplicity
__global__ void compute_point_to_plane_kernel(
    const float* src_x, const float* src_y, const float* src_z,
    const float* tgt_x, const float* tgt_y, const float* tgt_z,
    const float* tgt_nx, const float* tgt_ny, const float* tgt_nz,
    const int* correspondences, const float* distances,
    int n_src, float max_dist,
    float* d_JtJ,  // [6*6] accumulated via atomicAdd
    float* d_Jtr,  // [6] accumulated via atomicAdd
    int* d_n_valid) // number of valid pairs
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src) return;

    if (distances[i] > max_dist) return;

    int j = correspondences[i];
    float sx = src_x[i], sy = src_y[i], sz = src_z[i];
    float tx = tgt_x[j], ty = tgt_y[j], tz = tgt_z[j];
    float nx = tgt_nx[j], ny = tgt_ny[j], nz = tgt_nz[j];

    // Point-to-plane residual: r = (s - t) . n
    float dx = sx - tx, dy = sy - ty, dz = sz - tz;
    float r = dx * nx + dy * ny + dz * nz;

    // Jacobian row for point-to-plane with small angle approximation:
    // d(r)/d(alpha, beta, gamma, tx, ty, tz)
    // where the transform is: p' = p + [gamma*py - beta*pz, alpha*pz - gamma*px, beta*px - alpha*py] + [tx,ty,tz]
    // J = [nz*sy - ny*sz, nx*sz - nz*sx, ny*sx - nx*sy, nx, ny, nz]
    float J[6];
    J[0] = nz * sy - ny * sz;
    J[1] = nx * sz - nz * sx;
    J[2] = ny * sx - nx * sy;
    J[3] = nx;
    J[4] = ny;
    J[5] = nz;

    // Accumulate JtJ and Jtr
    for (int a = 0; a < 6; a++) {
        for (int b = a; b < 6; b++) {
            atomicAdd(&d_JtJ[a * 6 + b], J[a] * J[b]);
        }
        atomicAdd(&d_Jtr[a], J[a] * r);
    }
    atomicAdd(d_n_valid, 1);
}

// =====================================================================
// Host function
// =====================================================================

void gicp_align(const CudaPointCloud& source, const CudaPointCloud& target,
                float* R_out, float* t_out, int max_iter, float tolerance) {
    int n_src = source.size();
    int n_tgt = target.size();
    if (n_src == 0 || n_tgt == 0) return;

    // Estimate normals for target
    float *d_tgt_nx, *d_tgt_ny, *d_tgt_nz;
    CUDA_CHECK(cudaMalloc(&d_tgt_nx, n_tgt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_ny, n_tgt * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tgt_nz, n_tgt * sizeof(float)));
    estimate_normals(target, d_tgt_nx, d_tgt_ny, d_tgt_nz, 20);

    // Create working copy of source
    CudaPointCloud src_work(source);

    // Allocate correspondence arrays
    int* d_corr;
    float* d_dists;
    CUDA_CHECK(cudaMalloc(&d_corr, n_src * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dists, n_src * sizeof(float)));

    // Allocate JtJ, Jtr
    float *d_JtJ, *d_Jtr;
    int* d_n_valid;
    CUDA_CHECK(cudaMalloc(&d_JtJ, 36 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Jtr, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_n_valid, sizeof(int)));

    // Cumulative transform
    Eigen::Matrix3f R_total = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_total = Eigen::Vector3f::Zero();

    int block = 256;
    int grid_src = (n_src + block - 1) / block;

    float prev_error = 1e30f;

    for (int iter = 0; iter < max_iter; iter++) {
        // Find correspondences
        gicp_find_correspondences_kernel<<<grid_src, block>>>(
            src_work.d_x(), src_work.d_y(), src_work.d_z(), n_src,
            target.d_x(), target.d_y(), target.d_z(), n_tgt,
            d_tgt_nx, d_tgt_ny, d_tgt_nz,
            d_corr, d_dists);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Compute mean distance for convergence check and max_dist threshold
        std::vector<float> h_dists(n_src);
        CUDA_CHECK(cudaMemcpy(h_dists.data(), d_dists, n_src * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_dist = 0;
        for (int i = 0; i < n_src; i++) mean_dist += h_dists[i];
        mean_dist /= n_src;

        float max_dist = mean_dist * 3.0f;  // reject outlier correspondences

        if (fabsf(prev_error - mean_dist) < tolerance) {
            break;
        }
        prev_error = mean_dist;

        // Reset accumulators
        CUDA_CHECK(cudaMemset(d_JtJ, 0, 36 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_Jtr, 0, 6 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_n_valid, 0, sizeof(int)));

        // Compute point-to-plane system
        compute_point_to_plane_kernel<<<grid_src, block>>>(
            src_work.d_x(), src_work.d_y(), src_work.d_z(),
            target.d_x(), target.d_y(), target.d_z(),
            d_tgt_nx, d_tgt_ny, d_tgt_nz,
            d_corr, d_dists, n_src, max_dist,
            d_JtJ, d_Jtr, d_n_valid);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download and solve on host
        float h_JtJ[36], h_Jtr[6];
        int h_n_valid;
        CUDA_CHECK(cudaMemcpy(h_JtJ, d_JtJ, 36 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Jtr, d_Jtr, 6 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_n_valid, d_n_valid, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_n_valid < 10) break;

        // Fill symmetric part
        Eigen::Matrix<float, 6, 6> JtJ;
        Eigen::Matrix<float, 6, 1> Jtr;
        for (int a = 0; a < 6; a++) {
            for (int b = 0; b < 6; b++) {
                if (a <= b) JtJ(a, b) = h_JtJ[a * 6 + b];
                else JtJ(a, b) = h_JtJ[b * 6 + a];
            }
            Jtr(a) = h_Jtr[a];
        }

        // Solve JtJ * x = -Jtr
        Eigen::Matrix<float, 6, 1> x = JtJ.ldlt().solve(-Jtr);

        float alpha = x(0), beta = x(1), gamma = x(2);
        float dtx = x(3), dty = x(4), dtz = x(5);

        // Small angle rotation matrix
        Eigen::Matrix3f dR;
        dR << 1, -gamma, beta,
              gamma, 1, -alpha,
              -beta, alpha, 1;

        // Re-orthogonalize via SVD
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(dR, Eigen::ComputeFullU | Eigen::ComputeFullV);
        dR = svd.matrixU() * svd.matrixV().transpose();

        Eigen::Vector3f dt(dtx, dty, dtz);

        // Accumulate
        t_total = dR * t_total + dt;
        R_total = dR * R_total;

        // Apply incremental transform to source
        apply_transform_kernel<<<grid_src, block>>>(
            src_work.d_x(), src_work.d_y(), src_work.d_z(), n_src,
            dR(0,0), dR(0,1), dR(0,2),
            dR(1,0), dR(1,1), dR(1,2),
            dR(2,0), dR(2,1), dR(2,2),
            dtx, dty, dtz);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Output R (row-major 3x3) and t (3x1)
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R_out[i * 3 + j] = R_total(i, j);
    t_out[0] = t_total(0);
    t_out[1] = t_total(1);
    t_out[2] = t_total(2);

    cudaFree(d_tgt_nx); cudaFree(d_tgt_ny); cudaFree(d_tgt_nz);
    cudaFree(d_corr); cudaFree(d_dists);
    cudaFree(d_JtJ); cudaFree(d_Jtr); cudaFree(d_n_valid);
}

} // namespace cudabot
