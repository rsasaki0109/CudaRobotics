// gpu_filterreg.cu
//
// GPU FilterReg demo + verification.  The reusable core lives in
// include/cudarobotics/filterreg_gpu.hpp + src/filterreg_gpu.cu.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cudarobotics/filterreg_gpu.hpp"
#include "cuda_video.h"

namespace cudabot {
namespace {

struct Mat3 { float m[9]; };
struct Pose { Mat3 R; float t[3]; };

static inline void mat3_vec(const Mat3 & R, const float * v, float * o)
{
  o[0] = R.m[0] * v[0] + R.m[1] * v[1] + R.m[2] * v[2];
  o[1] = R.m[3] * v[0] + R.m[4] * v[1] + R.m[5] * v[2];
  o[2] = R.m[6] * v[0] + R.m[7] * v[1] + R.m[8] * v[2];
}

static inline void pose_apply(const Pose & T, const float * y, float * p)
{
  mat3_vec(T.R, y, p);
  p[0] += T.t[0];
  p[1] += T.t[1];
  p[2] += T.t[2];
}

static inline Mat3 mat3_mul(const Mat3 & A, const Mat3 & B)
{
  Mat3 C;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float s = 0.f;
      for (int k = 0; k < 3; ++k) {
        s += A.m[i * 3 + k] * B.m[k * 3 + j];
      }
      C.m[i * 3 + j] = s;
    }
  }
  return C;
}

static inline Mat3 so3_exp(const float * w)
{
  float th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  Mat3 R;
  if (th < 1e-9f) {
    R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    return R;
  }
  float a = w[0] / th, b = w[1] / th, c = w[2] / th;
  float s = std::sin(th), co = std::cos(th), v = 1.f - co;
  R.m[0] = a * a * v + co;
  R.m[1] = a * b * v - c * s;
  R.m[2] = a * c * v + b * s;
  R.m[3] = a * b * v + c * s;
  R.m[4] = b * b * v + co;
  R.m[5] = b * c * v - a * s;
  R.m[6] = a * c * v - b * s;
  R.m[7] = b * c * v + a * s;
  R.m[8] = c * c * v + co;
  return R;
}

static Pose toPose(const cudarobotics::FilterRegResult & r)
{
  Pose T;
  for (int i = 0; i < 9; ++i) {
    T.R.m[i] = r.rotation[i];
  }
  for (int k = 0; k < 3; ++k) {
    T.t[k] = r.translation[k];
  }
  return T;
}

static std::vector<float> make_lumpy(int n, unsigned seed)
{
  std::vector<float> pts(n * 3);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uu(-1.f, 1.f), up(0.f, 6.2831853f);
  const float bumps[][5] = {
    {0.8f, 0.2f, 0.5f, 0.9f, 0.25f},
    {-0.3f, 0.9f, 0.2f, 0.7f, 0.30f},
    {0.1f, -0.6f, 0.8f, 0.8f, 0.22f},
    {-0.7f, -0.4f, -0.5f, 1.0f, 0.28f},
    {0.4f, 0.3f, -0.85f, 0.6f, 0.20f},
  };
  int nb = sizeof(bumps) / sizeof(bumps[0]);
  for (int i = 0; i < n; ++i) {
    float z = uu(rng), phi = up(rng), r2 = std::sqrt(std::max(0.f, 1.f - z * z));
    float dx = r2 * std::cos(phi), dy = r2 * std::sin(phi), dz = z;
    float R = 2.0f + 0.35f * std::sin(3.f * phi) * (1.f - z * z) +
      0.30f * dz * dx + 0.20f * std::cos(2.f * phi);
    for (int b = 0; b < nb; ++b) {
      float d = dx * bumps[b][0] + dy * bumps[b][1] + dz * bumps[b][2];
      float ang = 1.f - d;
      R += bumps[b][3] * std::exp(-ang * ang / (2.f * bumps[b][4] * bumps[b][4]));
    }
    pts[i * 3 + 0] = R * dx;
    pts[i * 3 + 1] = R * dy;
    pts[i * 3 + 2] = R * dz;
  }
  return pts;
}

static void render_gif(
  const std::vector<float> & X, const std::vector<float> & Y,
  const std::vector<Pose> & traj)
{
  const int W = 1280, H = 720;
  const int CX = 380, CY = 360;
  const float SCALE = 78.f;
  auto sub = [](const std::vector<float> & P, int stride) {
      std::vector<float> q;
      for (size_t i = 0; i < P.size() / 3; i += stride) {
        q.push_back(P[i * 3]);
        q.push_back(P[i * 3 + 1]);
        q.push_back(P[i * 3 + 2]);
      }
      return q;
    };
  std::vector<float> Xs = sub(X, 4), Ys = sub(Y, 4);

  if (system("mkdir -p tmp") != 0) {
    std::fprintf(stderr, "warning: mkdir tmp failed\n");
  }
  cv::VideoWriter video(
    "tmp/gpu_filterreg.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 20,
    cv::Size(W, H));
  const float elev = 0.42f;
  int ntraj = static_cast<int>(traj.size());
  const int HOLD = 26;
  int nframes = ntraj + HOLD;
  struct Splat {
    float sx, sy, depth;
    cv::Scalar col;
  };
  for (int f = 0; f < nframes; ++f) {
    int k = std::min(f, ntraj - 1);
    float az = 0.6f + f * 0.018f;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(26, 26, 32));
    const Pose & T = traj[static_cast<size_t>(k)];
    float ca = std::cos(az), sa = std::sin(az), ce = std::cos(elev), se = std::sin(elev);
    auto project = [&](float x, float y, float z, float & sx, float & sy, float & depth) {
        float x1 = x * ca - y * sa, y1 = x * sa + y * ca, z1 = z;
        sx = CX + SCALE * x1;
        sy = CY - SCALE * (z1 * ce - y1 * se);
        depth = y1 * ce + z1 * se;
      };
    std::vector<Splat> sp;
    sp.reserve(Xs.size() / 3 + Ys.size() / 3);
    for (size_t i = 0; i < Xs.size() / 3; ++i) {
      Splat s;
      project(Xs[i * 3], Xs[i * 3 + 1], Xs[i * 3 + 2], s.sx, s.sy, s.depth);
      s.col = cv::Scalar(210, 180, 60);
      sp.push_back(s);
    }
    for (size_t i = 0; i < Ys.size() / 3; ++i) {
      float y0[3] = {Ys[i * 3], Ys[i * 3 + 1], Ys[i * 3 + 2]}, p[3];
      pose_apply(T, y0, p);
      Splat s;
      project(p[0], p[1], p[2], s.sx, s.sy, s.depth);
      s.col = cv::Scalar(40, 130, 240);
      sp.push_back(s);
    }
    std::sort(sp.begin(), sp.end(), [](const Splat & a, const Splat & b) {
        return a.depth < b.depth;
      });
    float dmin = 1e9f, dmax = -1e9f;
    for (auto & s : sp) {
      dmin = std::min(dmin, s.depth);
      dmax = std::max(dmax, s.depth);
    }
    for (auto & s : sp) {
      float t = (s.depth - dmin) / (dmax - dmin + 1e-6f);
      float b = 0.45f + 0.55f * t;
      cv::circle(img, cv::Point(static_cast<int>(s.sx), static_cast<int>(s.sy)), 2, s.col * b, -1, cv::LINE_AA);
    }
    int px = 800, py = 70;
    auto put = [&](const std::string & s, int yy, double sc, cv::Scalar c, int th) {
        cv::putText(img, s, cv::Point(px, yy), cv::FONT_HERSHEY_SIMPLEX, sc, c, th, cv::LINE_AA);
      };
    put("GPU FilterReg", py, 1.0, cv::Scalar(235, 235, 245), 2);
    py += 38;
    put("probabilistic registration", py, 0.62, cv::Scalar(180, 180, 200), 1);
    py += 52;
    cv::circle(img, cv::Point(px + 8, py - 6), 6, cv::Scalar(210, 180, 60), -1);
    cv::putText(
      img, "fixed cloud", cv::Point(px + 26, py), cv::FONT_HERSHEY_SIMPLEX, 0.6,
      cv::Scalar(200, 200, 210), 1, cv::LINE_AA);
    py += 30;
    cv::circle(img, cv::Point(px + 8, py - 6), 6, cv::Scalar(40, 130, 240), -1);
    cv::putText(
      img, "source (aligning)", cv::Point(px + 26, py), cv::FONT_HERSHEY_SIMPLEX, 0.6,
      cv::Scalar(200, 200, 210), 1, cv::LINE_AA);
    py += 52;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "iteration %d / %d", k, ntraj - 1);
    put(buf, py, 0.62, cv::Scalar(210, 210, 225), 1);
    py += 40;
    put("E-step: Gaussian filter (O(N+M))", py, 0.5, cv::Scalar(150, 200, 150), 1);
    py += 26;
    put("M-step: SE(3) twist Gauss-Newton", py, 0.5, cv::Scalar(150, 200, 150), 1);
    py += 26;
    put("coarse-to-fine sigma annealing", py, 0.5, cv::Scalar(150, 200, 150), 1);
    py += 44;
    if (f >= nframes - HOLD) {
      put("ALIGNED", py, 0.8, cv::Scalar(120, 230, 250), 2);
    }
    video.write(img);
  }
  video.release();
  avi_to_gif("tmp/gpu_filterreg.avi", "gif/gpu_filterreg.gif", 20, 900);
  std::printf("wrote gif/gpu_filterreg.gif\n");
}

}  // namespace
}  // namespace cudabot

int main()
{
  using namespace cudabot;
  std::printf("=== GPU FilterReg: probabilistic rigid registration (verification) ===\n");

  const int N = 12000;
  std::vector<float> X = make_lumpy(N, 1);

  std::mt19937 rng(7);
  const char * envp = std::getenv("FR_EASY");
  bool easy = envp && envp[0] == '1';
  float gt_w[3] = {0.25f, -0.35f, 0.20f};
  float gt_t[3] = {0.7f, -0.5f, 0.4f};
  if (easy) {
    gt_w[0] = 0.05f;
    gt_w[1] = -0.04f;
    gt_w[2] = 0.03f;
    gt_t[0] = 0.15f;
    gt_t[1] = -0.1f;
    gt_t[2] = 0.08f;
  }
  Mat3 Rgt = so3_exp(gt_w);
  Pose Tgt;
  Tgt.R = Rgt;
  for (int k = 0; k < 3; ++k) {
    Tgt.t[k] = gt_t[k];
  }
  bool zero = std::getenv("FR_ZERO") && std::getenv("FR_ZERO")[0] == '1';
  if (zero) {
    for (int k = 0; k < 3; ++k) {
      gt_w[k] = 0;
      gt_t[k] = 0;
    }
    Rgt = so3_exp(gt_w);
    Tgt.R = Rgt;
    for (int k = 0; k < 3; ++k) {
      Tgt.t[k] = 0;
    }
  }
  std::normal_distribution<float> noise(0.f, zero ? 0.f : 0.02f);
  std::uniform_real_distribution<float> keep(0.f, 1.f);
  std::vector<float> Y;
  for (int i = 0; i < N; ++i) {
    if (!zero && keep(rng) > 0.85f) {
      continue;
    }
    float y[3] = {X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]}, p[3];
    pose_apply(Tgt, y, p);
    Y.push_back(p[0] + noise(rng));
    Y.push_back(p[1] + noise(rng));
    Y.push_back(p[2] + noise(rng));
  }
  int M = static_cast<int>(Y.size() / 3);
  std::printf("fixed N=%d  source M=%d  (15%% dropped, sigma noise 0.02)\n", N, M);
  std::printf(
    "ground-truth  rot=(% .3f % .3f % .3f) rad   trans=(% .3f % .3f % .3f)\n",
    gt_w[0], gt_w[1], gt_w[2], gt_t[0], gt_t[1], gt_t[2]);

  cudarobotics::FilterRegGpu registrar;
  auto t0 = std::chrono::high_resolution_clock::now();
  cudarobotics::FilterRegResult res = registrar.registerClouds(
    X.data(), N, Y.data(), M, nullptr, nullptr);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  Mat3 RgtT;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      RgtT.m[i * 3 + j] = Rgt.m[j * 3 + i];
    }
  }
  float texp[3];
  mat3_vec(RgtT, gt_t, texp);
  for (int k = 0; k < 3; ++k) {
    texp[k] = -texp[k];
  }
  Mat3 Rres;
  for (int i = 0; i < 9; ++i) {
    Rres.m[i] = res.rotation[i];
  }
  Mat3 Rerr = mat3_mul(Rgt, Rres);
  float tr = Rerr.m[0] + Rerr.m[4] + Rerr.m[8];
  float ang = std::acos(std::min(1.f, std::max(-1.f, (tr - 1.f) * 0.5f)));
  float terr = 0;
  for (int k = 0; k < 3; ++k) {
    float e = res.translation[k] - texp[k];
    terr += e * e;
  }
  terr = std::sqrt(terr);

  std::printf(
    "recovered     rot-matrix vs Tgt^{-1}: angle err = %.4f rad (%.3f deg)\n", ang,
    ang * 57.2958f);
  std::printf(
    "              trans = (% .3f % .3f % .3f)  expected (% .3f % .3f % .3f)  err=%.4f\n",
    res.translation[0], res.translation[1], res.translation[2], texp[0], texp[1], texp[2], terr);
  std::printf(
    "iters=%d  final weighted RMSE=%.4f  wall=%.1f ms\n", res.iterations, res.final_rmse, ms);
  if (ang < 0.02f && terr < 0.05f) {
    std::printf("RESULT: PASS -- FilterReg recovered the known transform.\n");
  } else {
    std::printf("RESULT: CHECK -- transform not recovered within tolerance.\n");
  }

  if (!zero && !easy) {
    std::vector<Pose> traj = {Pose{}, toPose(res)};
    render_gif(X, Y, traj);
  }
  return 0;
}
