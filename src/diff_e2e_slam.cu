// diff_e2e_slam.cu
//
// Differentiable end-to-end 2D landmark SLAM.
//
// Setup:
//   - Ground-truth 2D trajectory of N_POSES SE(2) poses.
//   - M_LANDMARKS scattered in the world.
//   - Each pose observes nearby landmarks with range+bearing.
//   - Observations are corrupted by Gaussian noise plus a fraction of
//     heavy-tailed outliers.
//
// Inner solver:
//   - Gauss-Newton SLAM over (poses, landmarks).
//   - The solver weights each measurement by 1 / (sigma_est^2 + e^2),
//     so sigma_est plays the role of a Cauchy-like switch.
//   - Solve via Jacobi-PCG on the (3 N + 2 M) x (3 N + 2 M) sparse system,
//     but in practice we factor by hand here as a small dense problem
//     because the GPU win for outer learning loops is what matters.
//
// Outer learning:
//   - Tunable parameter: the scalar sigma_est.
//   - Loss: ground-truth pose RMSE after inner solver convergence.
//   - Optimizer: finite-difference Adam on sigma_est.
//
// What this demonstrates:
//   The optimal sigma_est is NOT the true measurement noise; it has to
//   balance suppression of outliers against under-weighting of clean
//   measurements.  End-to-end gradient descent finds the trade-off
//   that minimises the actual downstream task error.
//
// Output: gif/diff_e2e_slam.gif with 4 panels showing convergence of
//   sigma_est, pose-RMSE history, before / after trajectories.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

namespace cudabot {

constexpr int N_POSES = 60;
constexpr int M_LANDMARKS = 18;
constexpr int GN_ITERS = 30;
constexpr int PCG_ITERS = 60;
constexpr float TRUE_SIGMA_RNG = 0.10f;
constexpr float TRUE_SIGMA_BRG = 0.04f;
constexpr float OUTLIER_FRAC = 0.18f;
constexpr float OUTLIER_BIAS = 1.2f;
constexpr float SENSOR_RANGE = 12.0f;
constexpr int N_OUTER_STEPS = 35;

struct Obs {
    int pose, lm;
    float range, bearing;
};

static inline float wrap_pi(float a) {
    while (a >  M_PI) a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

// ---- ground truth generation ----------------------------------------------
static void make_gt(std::vector<float>& gt_poses,
                    std::vector<float>& gt_lms) {
    gt_poses.assign(N_POSES * 3, 0.0f);
    for (int k = 0; k < N_POSES; k++) {
        float s = static_cast<float>(k) / (N_POSES - 1);
        float u = s * 2.0f * static_cast<float>(M_PI);
        float x = 9.0f * std::sin(u);
        float y = 6.0f * std::sin(2.0f * u);
        float dxds = 9.0f * std::cos(u);
        float dyds = 12.0f * std::cos(2.0f * u);
        gt_poses[3 * k + 0] = x;
        gt_poses[3 * k + 1] = y;
        gt_poses[3 * k + 2] = std::atan2(dyds, dxds);
    }
    gt_lms.assign(M_LANDMARKS * 2, 0.0f);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> ux(-12.0f, 12.0f);
    std::uniform_real_distribution<float> uy(-9.0f, 9.0f);
    for (int j = 0; j < M_LANDMARKS; j++) {
        gt_lms[2 * j + 0] = ux(rng);
        gt_lms[2 * j + 1] = uy(rng);
    }
}

static std::vector<Obs> simulate_observations(const std::vector<float>& gt_poses,
                                              const std::vector<float>& gt_lms,
                                              std::mt19937& rng) {
    std::normal_distribution<float> nr(0.0f, TRUE_SIGMA_RNG);
    std::normal_distribution<float> nb(0.0f, TRUE_SIGMA_BRG);
    std::uniform_real_distribution<float> ub(-OUTLIER_BIAS, OUTLIER_BIAS);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    std::vector<Obs> obs;
    obs.reserve(N_POSES * M_LANDMARKS / 3);
    for (int k = 0; k < N_POSES; k++) {
        float xi = gt_poses[3 * k + 0];
        float yi = gt_poses[3 * k + 1];
        float ti = gt_poses[3 * k + 2];
        for (int j = 0; j < M_LANDMARKS; j++) {
            float dx = gt_lms[2 * j + 0] - xi;
            float dy = gt_lms[2 * j + 1] - yi;
            float r  = std::sqrt(dx * dx + dy * dy);
            if (r > SENSOR_RANGE) continue;
            float b = wrap_pi(std::atan2(dy, dx) - ti);
            float rn = nr(rng);
            float bn = nb(rng);
            bool outlier = u01(rng) < OUTLIER_FRAC;
            if (outlier) { rn += ub(rng); bn += ub(rng) * 0.5f; }
            obs.push_back({k, j, r + rn, wrap_pi(b + bn)});
        }
    }
    return obs;
}

// ---- inner GN solver (CPU, dense small system) ----------------------------
// State vector layout: [pose0 (3), pose1 (3), ..., lm0 (2), lm1 (2), ...].
// Pose 0 is anchored to ground truth to fix gauge.
static float pose_rmse(const std::vector<float>& poses,
                       const std::vector<float>& gt) {
    double s = 0.0;
    for (int k = 0; k < N_POSES; k++) {
        double dx = poses[3 * k + 0] - gt[3 * k + 0];
        double dy = poses[3 * k + 1] - gt[3 * k + 1];
        s += dx * dx + dy * dy;
    }
    return std::sqrt(s / N_POSES);
}

static float run_gn_slam(const std::vector<float>& init_poses,
                         const std::vector<float>& init_lms,
                         const std::vector<float>& gt_poses,
                         const std::vector<Obs>& obs,
                         float sigma_est,
                         std::vector<float>& out_poses,
                         std::vector<float>& out_lms,
                         std::vector<float>& rmse_history) {
    out_poses = init_poses;
    out_lms = init_lms;
    int N3 = 3 * N_POSES;
    int M2 = 2 * M_LANDMARKS;
    int dim = N3 + M2;
    std::vector<double> H(dim * dim, 0.0);
    std::vector<double> b(dim, 0.0);
    rmse_history.clear();
    rmse_history.push_back(pose_rmse(out_poses, gt_poses));

    for (int it = 0; it < GN_ITERS; it++) {
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(b.begin(), b.end(), 0.0);

        for (const auto& o : obs) {
            int p = o.pose, l = o.lm;
            float xi = out_poses[3 * p + 0];
            float yi = out_poses[3 * p + 1];
            float ti = out_poses[3 * p + 2];
            float xj = out_lms[2 * l + 0];
            float yj = out_lms[2 * l + 1];
            float dx = xj - xi;
            float dy = yj - yi;
            float r2 = dx * dx + dy * dy;
            float r = std::sqrt(r2 + 1e-9f);
            float pred_r = r;
            float pred_b = wrap_pi(std::atan2(dy, dx) - ti);

            float er = pred_r - o.range;
            float eb = wrap_pi(pred_b - o.bearing);
            // robust weight: w = 1 / (sigma_est^2 + e^2)
            float e_norm = std::sqrt(er * er + eb * eb);
            float w = 1.0f / (sigma_est * sigma_est + e_norm * e_norm);

            // Jacobians
            //   dpred_r / dxi = -dx/r,  /dyi = -dy/r,  /dti = 0
            //   dpred_r / dxj =  dx/r,  /dyj =  dy/r
            //   dpred_b / dxi =  dy/r2, /dyi = -dx/r2, /dti = -1
            //   dpred_b / dxj = -dy/r2, /dyj =  dx/r2
            float Jr_pi[3] = { -dx / r, -dy / r, 0.0f };
            float Jr_lj[2] = {  dx / r,  dy / r };
            float Jb_pi[3] = {  dy / r2, -dx / r2, -1.0f };
            float Jb_lj[2] = { -dy / r2,  dx / r2 };

            int pi0 = 3 * p;
            int lj0 = N3 + 2 * l;

            // Accumulate H += w * J^T J and b += w * J^T e for each obs.
            float Js[5];
            float es[2] = { er, eb };
            for (int row = 0; row < 2; row++) {
                if (row == 0) {
                    Js[0] = Jr_pi[0]; Js[1] = Jr_pi[1]; Js[2] = Jr_pi[2];
                    Js[3] = Jr_lj[0]; Js[4] = Jr_lj[1];
                } else {
                    Js[0] = Jb_pi[0]; Js[1] = Jb_pi[1]; Js[2] = Jb_pi[2];
                    Js[3] = Jb_lj[0]; Js[4] = Jb_lj[1];
                }
                int idx[5] = { pi0 + 0, pi0 + 1, pi0 + 2, lj0 + 0, lj0 + 1 };
                for (int u = 0; u < 5; u++) {
                    b[idx[u]] += static_cast<double>(w * Js[u] * es[row]);
                    for (int v = 0; v < 5; v++) {
                        H[idx[u] * dim + idx[v]] += static_cast<double>(w * Js[u] * Js[v]);
                    }
                }
            }
        }
        // anchor pose 0 (rows + cols 0..2) -> identity, b[0..2] = 0
        for (int k = 0; k < 3; k++) {
            for (int v = 0; v < dim; v++) {
                H[k * dim + v] = 0.0;
                H[v * dim + k] = 0.0;
            }
            H[k * dim + k] = 1.0;
            b[k] = 0.0;
        }
        // Levenberg-Marquardt diagonal damping for stability.
        double lambda = 1e-3;
        for (int k = 0; k < dim; k++) H[k * dim + k] += lambda;

        // Solve via Gauss elimination (dim = 3*60 + 2*18 = 216).
        std::vector<double> A = H;
        std::vector<double> rhs(b);
        for (int k = 0; k < dim; k++) {
            int piv = k;
            double bestp = std::abs(A[k * dim + k]);
            for (int r = k + 1; r < dim; r++) {
                if (std::abs(A[r * dim + k]) > bestp) {
                    bestp = std::abs(A[r * dim + k]); piv = r;
                }
            }
            if (bestp < 1e-12) break;
            if (piv != k) {
                for (int j = 0; j < dim; j++) std::swap(A[k * dim + j], A[piv * dim + j]);
                std::swap(rhs[k], rhs[piv]);
            }
            double inv = 1.0 / A[k * dim + k];
            for (int r = k + 1; r < dim; r++) {
                double f = A[r * dim + k] * inv;
                if (f == 0.0) continue;
                for (int j = k; j < dim; j++) {
                    A[r * dim + j] -= f * A[k * dim + j];
                }
                rhs[r] -= f * rhs[k];
            }
        }
        std::vector<double> dx(dim, 0.0);
        for (int k = dim - 1; k >= 0; k--) {
            double s = rhs[k];
            for (int j = k + 1; j < dim; j++) s -= A[k * dim + j] * dx[j];
            double diag = A[k * dim + k];
            dx[k] = (std::abs(diag) > 1e-12) ? (s / diag) : 0.0;
        }
        // Apply: poses += -dx (because solve solved H dx = b, b = J^T W e, update is -dx)
        for (int k = 1; k < N_POSES; k++) {
            out_poses[3 * k + 0] -= static_cast<float>(dx[3 * k + 0]);
            out_poses[3 * k + 1] -= static_cast<float>(dx[3 * k + 1]);
            out_poses[3 * k + 2] = wrap_pi(out_poses[3 * k + 2] - static_cast<float>(dx[3 * k + 2]));
        }
        for (int j = 0; j < M_LANDMARKS; j++) {
            out_lms[2 * j + 0] -= static_cast<float>(dx[N3 + 2 * j + 0]);
            out_lms[2 * j + 1] -= static_cast<float>(dx[N3 + 2 * j + 1]);
        }
        rmse_history.push_back(pose_rmse(out_poses, gt_poses));
    }
    return rmse_history.back();
}

// ---- visualization --------------------------------------------------------
static cv::Point to_px(float x, float y, float scale, int cx, int cy) {
    return cv::Point(static_cast<int>(cx + x * scale),
                     static_cast<int>(cy - y * scale));
}

static cv::Mat draw_traj_panel(const std::vector<float>& poses,
                               const std::vector<float>& lms,
                               const std::vector<float>& gt_poses,
                               const std::vector<float>& gt_lms,
                               const std::string& title,
                               float rmse, float sigma_est) {
    int W = 540, H = 540;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(20, 20, 26));
    float scale = 20.0f;
    int cx = W / 2;
    int cy = H / 2 + 20;
    // grid
    for (int g = -12; g <= 12; g += 2) {
        cv::line(img, to_px(g, -9, scale, cx, cy), to_px(g, 9, scale, cx, cy),
                 cv::Scalar(40, 40, 40), 1);
        cv::line(img, to_px(-12, g * 0.75f, scale, cx, cy), to_px(12, g * 0.75f, scale, cx, cy),
                 cv::Scalar(40, 40, 40), 1);
    }
    // gt trajectory
    for (int k = 1; k < N_POSES; k++) {
        cv::line(img,
                 to_px(gt_poses[3 * (k - 1) + 0], gt_poses[3 * (k - 1) + 1], scale, cx, cy),
                 to_px(gt_poses[3 * k + 0], gt_poses[3 * k + 1], scale, cx, cy),
                 cv::Scalar(170, 170, 170), 1);
    }
    // gt landmarks
    for (int j = 0; j < M_LANDMARKS; j++) {
        cv::drawMarker(img, to_px(gt_lms[2 * j + 0], gt_lms[2 * j + 1], scale, cx, cy),
                       cv::Scalar(180, 180, 180), cv::MARKER_TILTED_CROSS, 7, 1);
    }
    // est trajectory
    for (int k = 1; k < N_POSES; k++) {
        cv::line(img,
                 to_px(poses[3 * (k - 1) + 0], poses[3 * (k - 1) + 1], scale, cx, cy),
                 to_px(poses[3 * k + 0], poses[3 * k + 1], scale, cx, cy),
                 cv::Scalar(0, 165, 255), 2);
    }
    // est landmarks
    for (int j = 0; j < M_LANDMARKS; j++) {
        cv::circle(img, to_px(lms[2 * j + 0], lms[2 * j + 1], scale, cx, cy),
                   4, cv::Scalar(0, 220, 255), -1);
    }
    cv::putText(img, title, cv::Point(10, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "sigma_est = %.3f   pose RMSE = %.3f m", sigma_est, rmse);
    cv::putText(img, buf, cv::Point(10, H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(220, 220, 220), 1);
    return img;
}

static cv::Mat draw_curve_panel(const std::vector<float>& sigma_hist,
                                const std::vector<float>& loss_hist,
                                int cur_step,
                                float sigma_true) {
    int W = 540, H = 540;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(20, 20, 26));
    // axes
    cv::rectangle(img, cv::Rect(60, 40, W - 100, H - 90), cv::Scalar(50, 50, 60), 1);
    cv::putText(img, "End-to-end learning curve", cv::Point(10, 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

    if (loss_hist.empty()) return img;
    float lo = *std::min_element(loss_hist.begin(), loss_hist.end());
    float hi = *std::max_element(loss_hist.begin(), loss_hist.end());
    float so = *std::min_element(sigma_hist.begin(), sigma_hist.end());
    float sh = *std::max_element(sigma_hist.begin(), sigma_hist.end());
    if (hi - lo < 0.05f) hi = lo + 0.05f;
    if (sh - so < 0.05f) sh = so + 0.05f;

    int px0 = 60, py0 = 40 + H - 90;
    int pw = W - 100, ph = H - 90;

    auto loss_y = [&](float v) {
        return static_cast<int>(py0 - (v - lo) / (hi - lo) * (ph - 10));
    };
    auto sigma_y = [&](float v) {
        return static_cast<int>(py0 - (v - so) / (sh - so) * (ph - 10));
    };
    auto x_at = [&](int i) {
        return static_cast<int>(px0 + (static_cast<float>(i) / std::max(1, N_OUTER_STEPS - 1)) * pw);
    };

    // sigma curve (cyan)
    for (size_t i = 1; i < sigma_hist.size(); i++) {
        cv::line(img, cv::Point(x_at(i - 1), sigma_y(sigma_hist[i - 1])),
                       cv::Point(x_at(i), sigma_y(sigma_hist[i])),
                       cv::Scalar(220, 200, 60), 2);
    }
    // loss curve (orange)
    for (size_t i = 1; i < loss_hist.size(); i++) {
        cv::line(img, cv::Point(x_at(i - 1), loss_y(loss_hist[i - 1])),
                       cv::Point(x_at(i), loss_y(loss_hist[i])),
                       cv::Scalar(0, 165, 255), 2);
    }
    // true sigma line (white dashed)
    int yt = sigma_y(sigma_true);
    for (int x = px0; x < px0 + pw; x += 10) {
        cv::line(img, cv::Point(x, yt), cv::Point(x + 5, yt),
                 cv::Scalar(180, 180, 180), 1);
    }
    // current step marker
    int xs = x_at(std::min<int>(cur_step, static_cast<int>(loss_hist.size()) - 1));
    cv::line(img, cv::Point(xs, 40), cv::Point(xs, py0),
             cv::Scalar(80, 80, 90), 1);

    char buf[160];
    std::snprintf(buf, sizeof(buf), "outer step %d / %d", cur_step + 1, N_OUTER_STEPS);
    cv::putText(img, buf, cv::Point(70, H - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(220, 220, 220), 1);
    cv::putText(img, "cyan = sigma_est",  cv::Point(70, H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(220, 200, 60), 1);
    cv::putText(img, "orange = pose RMSE", cv::Point(260, H - 14),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(0, 165, 255), 1);
    std::snprintf(buf, sizeof(buf), "dashed = true noise std = %.3f", sigma_true);
    cv::putText(img, buf, cv::Point(70, H - 46),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(180, 180, 180), 1);
    return img;
}

static void convert_avi_to_gif(const std::string& avi, const std::string& gif, int fps) {
    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
                  "ffmpeg -y -i %s -vf \"fps=%d,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle\" %s 2>/dev/null",
                  avi.c_str(), fps, gif.c_str());
    int rc = std::system(cmd);
    if (rc != 0) std::fprintf(stderr, "ffmpeg failed (%d)\n", rc);
}

}  // namespace cudabot

using namespace cudabot;

int main() {
    std::vector<float> gt_poses, gt_lms;
    make_gt(gt_poses, gt_lms);

    std::mt19937 rng(101);
    auto obs = simulate_observations(gt_poses, gt_lms, rng);
    std::printf("Diff-E2E SLAM: %d poses, %d landmarks, %zu observations (outlier fraction %.2f)\n",
                N_POSES, M_LANDMARKS, obs.size(), OUTLIER_FRAC);

    // Initial guess: noisy odometry from pose 0 GT (chain forward).
    std::vector<float> init_poses(N_POSES * 3, 0.0f);
    init_poses[0] = gt_poses[0];
    init_poses[1] = gt_poses[1];
    init_poses[2] = gt_poses[2];
    std::normal_distribution<float> noxy(0.0f, 0.10f);
    std::normal_distribution<float> noth(0.0f, 0.03f);
    for (int k = 0; k < N_POSES - 1; k++) {
        float dxw = gt_poses[3 * (k + 1) + 0] - gt_poses[3 * k + 0];
        float dyw = gt_poses[3 * (k + 1) + 1] - gt_poses[3 * k + 1];
        float ti = init_poses[3 * k + 2];
        float c = std::cos(ti), s = std::sin(ti);
        float local_dx = dxw * c + dyw * s + noxy(rng);
        float local_dy = -dxw * s + dyw * c + noxy(rng);
        float local_dt = wrap_pi(gt_poses[3 * (k + 1) + 2] - gt_poses[3 * k + 2]) + noth(rng);
        init_poses[3 * (k + 1) + 0] = init_poses[3 * k + 0] + c * local_dx - s * local_dy;
        init_poses[3 * (k + 1) + 1] = init_poses[3 * k + 1] + s * local_dx + c * local_dy;
        init_poses[3 * (k + 1) + 2] = wrap_pi(ti + local_dt);
    }
    // Initial landmark guess: anchor on first observation, then ignore (start at origin).
    std::vector<float> init_lms(M_LANDMARKS * 2, 0.0f);
    // Use observed first-sighting to roughly initialize each landmark.
    std::vector<int> seen(M_LANDMARKS, 0);
    for (const auto& o : obs) {
        if (seen[o.lm]) continue;
        seen[o.lm] = 1;
        float xi = init_poses[3 * o.pose + 0];
        float yi = init_poses[3 * o.pose + 1];
        float ti = init_poses[3 * o.pose + 2];
        float bw = wrap_pi(ti + o.bearing);
        init_lms[2 * o.lm + 0] = xi + o.range * std::cos(bw);
        init_lms[2 * o.lm + 1] = yi + o.range * std::sin(bw);
    }

    // Initial diagnostic: with sigma_est very large (no robustification),
    // RMSE will be dominated by outliers. We baseline against sigma = true.
    std::vector<float> tmp_poses, tmp_lms, tmp_hist;
    float rmse_baseline = run_gn_slam(init_poses, init_lms, gt_poses, obs,
                                      TRUE_SIGMA_RNG, tmp_poses, tmp_lms, tmp_hist);
    auto traj_baseline = tmp_poses;
    auto lm_baseline   = tmp_lms;
    std::printf("Baseline (sigma_est = true %.3f) RMSE = %.3f m\n",
                TRUE_SIGMA_RNG, rmse_baseline);

    // Outer learning loop with Adam-like finite differences on sigma_est.
    float sigma_est = 0.03f;  // start small (over-trusting all measurements)
    float m_adam = 0.0f, v_adam = 0.0f;
    float beta1 = 0.85f, beta2 = 0.95f, eps = 1e-7f;
    float lr = 0.04f;
    float fd_h = 0.01f;
    std::vector<float> sigma_hist, loss_hist;

    std::system("mkdir -p gif");
    cv::VideoWriter video("gif/diff_e2e_slam.avi",
                          cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                          12, cv::Size(540 * 2, 540));

    std::vector<float> best_poses = init_poses;
    std::vector<float> best_lms = init_lms;
    float best_rmse = 1e9f;
    float best_sigma = sigma_est;

    for (int step = 0; step < N_OUTER_STEPS; step++) {
        std::vector<float> p_plus, l_plus, h_plus;
        std::vector<float> p_minus, l_minus, h_minus;
        float rmse_plus = run_gn_slam(init_poses, init_lms, gt_poses, obs,
                                       sigma_est + fd_h, p_plus, l_plus, h_plus);
        float rmse_minus = run_gn_slam(init_poses, init_lms, gt_poses, obs,
                                        std::max(1e-3f, sigma_est - fd_h),
                                        p_minus, l_minus, h_minus);
        float grad = (rmse_plus - rmse_minus) / (2.0f * fd_h);
        float rmse_here = 0.5f * (rmse_plus + rmse_minus);

        m_adam = beta1 * m_adam + (1.0f - beta1) * grad;
        v_adam = beta2 * v_adam + (1.0f - beta2) * grad * grad;
        float mhat = m_adam / (1.0f - std::pow(beta1, step + 1));
        float vhat = v_adam / (1.0f - std::pow(beta2, step + 1));
        sigma_est -= lr * mhat / (std::sqrt(vhat) + eps);
        if (sigma_est < 0.005f) sigma_est = 0.005f;
        if (sigma_est > 3.0f) sigma_est = 3.0f;

        sigma_hist.push_back(sigma_est);
        loss_hist.push_back(rmse_here);
        if (rmse_here < best_rmse) {
            best_rmse = rmse_here;
            best_sigma = sigma_est;
            best_poses = p_plus;
            best_lms = l_plus;
        }
        std::printf("  outer %02d  sigma=%.4f  RMSE=%.4f  grad=%+.4f\n",
                    step + 1, sigma_est, rmse_here, grad);

        cv::Mat left = draw_traj_panel(p_plus, l_plus, gt_poses, gt_lms,
                                       "current SLAM solution", rmse_here, sigma_est);
        cv::Mat right = draw_curve_panel(sigma_hist, loss_hist, step, TRUE_SIGMA_RNG);
        cv::Mat combined(540, 540 * 2, CV_8UC3);
        left.copyTo(combined(cv::Rect(0, 0, 540, 540)));
        right.copyTo(combined(cv::Rect(540, 0, 540, 540)));
        video.write(combined);
    }

    // Hold final frames
    cv::Mat final_left  = draw_traj_panel(best_poses, best_lms, gt_poses, gt_lms,
                                          "best-found tuned solution",
                                          best_rmse, best_sigma);
    cv::Mat final_right = draw_curve_panel(sigma_hist, loss_hist,
                                            N_OUTER_STEPS - 1, TRUE_SIGMA_RNG);
    cv::Mat final_combined(540, 540 * 2, CV_8UC3);
    final_left.copyTo(final_combined(cv::Rect(0, 0, 540, 540)));
    final_right.copyTo(final_combined(cv::Rect(540, 0, 540, 540)));
    for (int k = 0; k < 20; k++) video.write(final_combined);
    video.release();

    float initial_rmse = loss_hist.empty() ? best_rmse : loss_hist.front();
    std::printf("\nNaive initial sigma_est = (start)  ->  pose RMSE = %.4f m\n", initial_rmse);
    std::printf("End-to-end tuned   sigma_est = %.4f  ->  pose RMSE = %.4f m\n",
                best_sigma, best_rmse);
    std::printf("Oracle (sigma = true %.3f)            RMSE = %.4f m\n",
                TRUE_SIGMA_RNG, rmse_baseline);
    std::printf("Reduction from naive -> tuned = %.4f m (%.1f%%)\n",
                initial_rmse - best_rmse,
                100.0f * (initial_rmse - best_rmse) / std::max(1e-6f, initial_rmse));
    convert_avi_to_gif("gif/diff_e2e_slam.avi", "gif/diff_e2e_slam.gif", 12);
    std::printf("GIF saved to gif/diff_e2e_slam.gif\n");
    return 0;
}
