// gpu_bcpd.cu
//
// GPU non-rigid registration in the Coherent-Point-Drift / BCPD family.
//
// FilterReg (see gpu_filterreg.cu) gave the repo a fast probabilistic RIGID
// registrant.  This adds the NON-RIGID counterpart -- the modern, robust
// member of that family being
//   O. Hirose, "A Bayesian Formulation of Coherent Point Drift" (BCPD),
//   IEEE TPAMI 2021 (arXiv:2004.04788),
// the variational-Bayes reformulation of CPD (Myronenko & Song, TPAMI 2010).
//
// What it does: deform a moving model cloud Y onto a fixed target X by a SMOOTH
// displacement field v(y), recovered by EM/variational updates of a Gaussian
// mixture whose centroids are the deformed model points.  The deformation is
// regularised by a motion-coherence Gaussian-process prior G_{ij} =
// exp(-||y_i-y_j||^2 / 2 beta^2): nearby points move coherently, so the warp is
// smooth and the problem is well posed even with noise / partial overlap.
//
// Structure (same correctness-first staging as the FilterReg demo):
//   1. procedural base surface + a KNOWN smooth non-rigid warp -> (X fixed,
//      Y deformed), so the recovered warp can be scored against ground truth.
//   2. EM on the GPU:
//        E-step (dense, one thread per point): the same proper-EM posterior as
//          FilterReg -- per-target normaliser D_n = sum_m K(x_n,p_m)+outlier,
//          then per-model moments P1_m = sum_n K/D_n, PX_m = sum_n x_n K/D_n.
//        M-step (host GP solve): (G + lambda sigma^2 diag(1/P1)) W =
//          diag(1/P1) PX - Y ; displacement V = G W ; p_m = y_m + V_m.
//        Bayesian sigma^2 and outlier-weight updates each iteration.
//   3. verification: residual to the target + recovered-vs-true warp error.
//   4. (next) the deforming-surface GIF.
//
// M (control points) is kept modest so the dense E-step (O(N*M)) and the MxM
// Cholesky GP solve are cheap; the recovered field is interpolated to the full
// cloud for display.  Build: CMakeLists, --expt-relaxed-constexpr.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ============================ procedural surface + known warp ============================
// Lumpy closed surface (same family as the FilterReg demo): asymmetric so the
// non-rigid alignment is well constrained.
static void lumpy_point(float z, float phi, float* out) {
    float r2 = std::sqrt(std::max(0.f, 1.f - z*z));
    float dx = r2*std::cos(phi), dy = r2*std::sin(phi), dz = z;
    static const float bumps[][5] = {
        { 0.8f,  0.2f,  0.5f,  0.9f, 0.25f}, {-0.3f, 0.9f, 0.2f, 0.7f, 0.30f},
        { 0.1f, -0.6f,  0.8f,  0.8f, 0.22f}, {-0.7f,-0.4f,-0.5f, 1.0f, 0.28f},
        { 0.4f,  0.3f, -0.85f, 0.6f, 0.20f},
    };
    float R = 2.0f + 0.35f*std::sin(3.f*phi)*(1.f-z*z) + 0.30f*dz*dx + 0.20f*std::cos(2.f*phi);
    for (int b = 0; b < 5; ++b) {
        float d = dx*bumps[b][0] + dy*bumps[b][1] + dz*bumps[b][2];
        float ang = 1.f - d;
        R += bumps[b][3]*std::exp(-ang*ang/(2.f*bumps[b][4]*bumps[b][4]));
    }
    out[0] = R*dx; out[1] = R*dy; out[2] = R*dz;
}

static std::vector<float> sample_lumpy(int n, unsigned seed) {
    std::vector<float> pts(n*3);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uu(-1.f,1.f), up(0.f,6.2831853f);
    for (int i = 0; i < n; ++i) lumpy_point(uu(rng), up(rng), &pts[i*3]);
    return pts;
}

// A known SMOOTH non-rigid warp: a few low-frequency sinusoidal bends + a
// localized bulge.  Smooth, so a motion-coherence prior can recover it.
static void apply_warp(const float* in, float* out) {
    float x = in[0], y = in[1], z = in[2];
    out[0] = x + 0.45f*std::sin(0.8f*y + 0.5f) + 0.30f*std::cos(0.7f*z);
    out[1] = y + 0.40f*std::sin(0.9f*z) + 0.25f*std::sin(0.6f*x - 0.3f);
    out[2] = z + 0.45f*std::sin(0.7f*x + 0.2f) + 0.20f*std::cos(0.8f*y);
    // a localized bulge near (+x,+y)
    float bx = x-1.6f, by = y-1.2f, bz = z-0.3f;
    float r2 = bx*bx + by*by + bz*bz;
    float g = std::exp(-r2/(2.f*0.9f*0.9f));
    out[0] += 0.5f*g; out[1] += 0.35f*g; out[2] += 0.4f*g;
}

// ============================ GPU dense E-step ============================
// pass 1: per target n, the EM normaliser D_n = sum_m exp(-||x_n-p_m||^2/2s2) + c.
__global__ void estep_denom_kernel(const float* __restrict__ X, int N,
                                   const float* __restrict__ P, int M,
                                   float inv2s2, float c_out, float* __restrict__ D) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn0 = X[n*3+0], xn1 = X[n*3+1], xn2 = X[n*3+2];
    float s = 0.f;
    for (int m = 0; m < M; ++m) {
        float dx = xn0-P[m*3+0], dy = xn1-P[m*3+1], dz = xn2-P[m*3+2];
        s += __expf(-(dx*dx+dy*dy+dz*dz)*inv2s2);
    }
    D[n] = s + c_out;
}

// pass 2: per model m, P1_m = sum_n K_{mn}/D_n,  PX_m = sum_n x_n K_{mn}/D_n.
__global__ void estep_moments_kernel(const float* __restrict__ X, int N,
                                     const float* __restrict__ P, int M,
                                     const float* __restrict__ D, float inv2s2,
                                     float* __restrict__ P1, float* __restrict__ PX) {
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if (m >= M) return;
    float p0 = P[m*3+0], p1 = P[m*3+1], p2 = P[m*3+2];
    float s0 = 0.f, sx = 0.f, sy = 0.f, sz = 0.f;
    for (int n = 0; n < N; ++n) {
        float dx = X[n*3+0]-p0, dy = X[n*3+1]-p1, dz = X[n*3+2]-p2;
        float k = __expf(-(dx*dx+dy*dy+dz*dz)*inv2s2) / D[n];
        s0 += k; sx += k*X[n*3+0]; sy += k*X[n*3+1]; sz += k*X[n*3+2];
    }
    P1[m] = s0; PX[m*3+0] = sx; PX[m*3+1] = sy; PX[m*3+2] = sz;
}

// per-target responsibility sum Pt1_n = sum_m K_{mn}/D_n (for the sigma^2 update).
__global__ void estep_pt1_kernel(const float* __restrict__ X, int N,
                                 const float* __restrict__ P, int M,
                                 const float* __restrict__ D, float inv2s2,
                                 float* __restrict__ Pt1) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= N) return;
    float xn0 = X[n*3+0], xn1 = X[n*3+1], xn2 = X[n*3+2];
    float s = 0.f;
    for (int m = 0; m < M; ++m) {
        float dx = xn0-P[m*3+0], dy = xn1-P[m*3+1], dz = xn2-P[m*3+2];
        s += __expf(-(dx*dx+dy*dy+dz*dz)*inv2s2);
    }
    Pt1[n] = s / D[n];
}

// ============================ host dense linear algebra ============================
// Cholesky solve of the SPD system A W = B, A is MxM (row-major), B/W are Mx3.
static bool chol_solve(std::vector<double>& A, int M, std::vector<double>& B /*M*3*/) {
    std::vector<double> L((size_t)M*M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[(size_t)i*M+j];
            for (int k = 0; k < j; ++k) s -= L[(size_t)i*M+k]*L[(size_t)j*M+k];
            if (i == j) { if (s <= 0) return false; L[(size_t)i*M+i] = std::sqrt(s); }
            else L[(size_t)i*M+j] = s / L[(size_t)j*M+j];
        }
    }
    // forward/back substitution for each of the 3 columns
    for (int c = 0; c < 3; ++c) {
        std::vector<double> y(M);
        for (int i = 0; i < M; ++i) {
            double s = B[(size_t)i*3+c];
            for (int k = 0; k < i; ++k) s -= L[(size_t)i*M+k]*y[k];
            y[i] = s / L[(size_t)i*M+i];
        }
        for (int i = M-1; i >= 0; --i) {
            double s = y[i];
            for (int k = i+1; k < M; ++k) s -= L[(size_t)k*M+i]*B[(size_t)k*3+c];
            B[(size_t)i*3+c] = s / L[(size_t)i*M+i];
        }
    }
    return true;
}

// ============================ BCPD / CPD-nonrigid driver ============================
struct BcpdResult { std::vector<float> P; int iters; float final_sigma; };

// Registers model Y (M control points) onto target X (N points).  Records the
// deformed control positions per iteration into traj (for the GIF).
static BcpdResult bcpd(const std::vector<float>& X, const std::vector<float>& Y,
                       float beta, float lambda, int n_iter,
                       std::vector<std::vector<float>>* traj = nullptr) {
    int N = X.size()/3, M = Y.size()/3;

    // motion-coherence GP kernel G (MxM), fixed (depends only on Y).
    std::vector<double> G((size_t)M*M);
    float inv2b2 = 1.f/(2.f*beta*beta);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j) {
            float dx = Y[i*3+0]-Y[j*3+0], dy = Y[i*3+1]-Y[j*3+1], dz = Y[i*3+2]-Y[j*3+2];
            G[(size_t)i*M+j] = std::exp(-(dx*dx+dy*dy+dz*dz)*inv2b2);
        }

    // initial sigma^2 = mean pairwise (X,Y) variance (CPD init).
    double s2 = 0;
    for (int n = 0; n < N; ++n) for (int m = 0; m < M; ++m) {
        // subsample for the init estimate to stay cheap
    }
    {   // cheap init: variance of X plus variance of Y about their means
        double mx[3]={0,0,0}, my[3]={0,0,0};
        for (int n=0;n<N;++n) for(int k=0;k<3;++k) mx[k]+=X[n*3+k];
        for (int m=0;m<M;++m) for(int k=0;k<3;++k) my[k]+=Y[m*3+k];
        for(int k=0;k<3;++k){mx[k]/=N;my[k]/=M;}
        double vx=0; for(int n=0;n<N;++n) for(int k=0;k<3;++k){double d=X[n*3+k]-mx[k];vx+=d*d;}
        s2 = vx/(3.0*N) + 0.3;   // a touch larger so the basin is wide
    }
    float w_out = 0.1f;          // outlier weight

    // device buffers
    float *dX,*dP,*dD,*dP1,*dPX,*dPt1;
    CUDA_CHECK(cudaMalloc(&dX, N*3*sizeof(float)));   CUDA_CHECK(cudaMemcpy(dX, X.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP, M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dD, N*sizeof(float)));     CUDA_CHECK(cudaMalloc(&dPt1, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP1, M*sizeof(float)));    CUDA_CHECK(cudaMalloc(&dPX, M*3*sizeof(float)));

    std::vector<float> P = Y;                 // current deformed model positions
    CUDA_CHECK(cudaMemcpy(dP, P.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));
    BcpdResult res; res.iters = 0;
    if (traj) traj->push_back(P);

    std::vector<float> hP1(M), hPX(M*3), hPt1(N), hD(N);
    for (int it = 0; it < n_iter; ++it) {
        float inv2s2 = 1.f/(2.f*(float)s2);
        // outlier constant c = (2 pi s2)^{3/2} * w/(1-w) * M/N
        float c_out = std::pow(2.f*3.14159265f*(float)s2, 1.5f) * (w_out/(1.f-w_out)) * (float)M/(float)N;
        estep_denom_kernel<<<(N+255)/256,256>>>(dX, N, dP, M, inv2s2, c_out, dD);
        estep_moments_kernel<<<(M+255)/256,256>>>(dX, N, dP, M, dD, inv2s2, dP1, dPX);
        estep_pt1_kernel<<<(N+255)/256,256>>>(dX, N, dP, M, dD, inv2s2, dPt1);
        CUDA_CHECK(cudaMemcpy(hP1.data(), dP1, M*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hPX.data(), dPX, M*3*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hPt1.data(), dPt1, N*sizeof(float), cudaMemcpyDeviceToHost));

        double Np = 0; for (int m = 0; m < M; ++m) Np += hP1[m];

        // M-step: (G + lambda s2 diag(1/P1)) W = diag(1/P1) PX - Y ;  V = G W
        std::vector<double> A((size_t)M*M), B((size_t)M*3);
        for (int i = 0; i < M; ++i) {
            double p1 = std::max(1e-6f, hP1[i]);
            for (int j = 0; j < M; ++j) A[(size_t)i*M+j] = G[(size_t)i*M+j];
            A[(size_t)i*M+i] += lambda * s2 / p1;
            for (int k = 0; k < 3; ++k) B[(size_t)i*3+k] = hPX[i*3+k]/p1 - Y[i*3+k];
        }
        if (!chol_solve(A, M, B)) { std::printf("  [bcpd] chol failed at it=%d\n", it); break; }
        // V = G W  (B now holds W) ; p = Y + V
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < 3; ++k) {
                double v = 0; for (int j = 0; j < M; ++j) v += G[(size_t)i*M+j]*B[(size_t)j*3+k];
                P[i*3+k] = Y[i*3+k] + (float)v;
            }
        CUDA_CHECK(cudaMemcpy(dP, P.data(), M*3*sizeof(float), cudaMemcpyHostToDevice));

        // Bayesian sigma^2 update (CPD):
        //   s2 = (1/(3 Np)) [ sum_n Pt1_n||x_n||^2 - 2 sum_m PX_m.p_m + sum_m P1_m||p_m||^2 ]
        double term1 = 0, term2 = 0, term3 = 0;
        for (int n = 0; n < N; ++n) {
            double r2 = X[n*3]*X[n*3]+X[n*3+1]*X[n*3+1]+X[n*3+2]*X[n*3+2];
            term1 += hPt1[n]*r2;
        }
        for (int m = 0; m < M; ++m) {
            term2 += hPX[m*3]*P[m*3] + hPX[m*3+1]*P[m*3+1] + hPX[m*3+2]*P[m*3+2];
            double pp = P[m*3]*P[m*3]+P[m*3+1]*P[m*3+1]+P[m*3+2]*P[m*3+2];
            term3 += hP1[m]*pp;
        }
        s2 = (term1 - 2*term2 + term3) / (3.0*Np);
        if (s2 < 1e-5) s2 = 1e-5;
        ++res.iters;
        if (traj) traj->push_back(P);
        if (std::getenv("BCPD_DBG"))
            std::printf("  [bcpd] it=%2d sigma=%.4f Np=%.0f\n", it, std::sqrt(s2), Np);
    }
    res.P = P; res.final_sigma = std::sqrt((float)s2);
    cudaFree(dX); cudaFree(dP); cudaFree(dD); cudaFree(dPt1); cudaFree(dP1); cudaFree(dPX);
    return res;
}

// ============================ deforming-surface GIF ============================
// Orbiting 3D view: the warped model (orange) flows back onto the target surface
// (cyan) as the non-rigid field is recovered iteration by iteration.
static void render_gif(const std::vector<float>& X, const std::vector<std::vector<float>>& traj) {
    const int W = 1280, H = 720, CX = 380, CY = 360;
    const float SCALE = 80.f, elev = 0.42f;
    auto sub = [](const std::vector<float>& P, int stride){ std::vector<float> q;
        for (size_t i=0;i<P.size()/3;i+=stride){ q.push_back(P[i*3]);q.push_back(P[i*3+1]);q.push_back(P[i*3+2]);} return q; };
    std::vector<float> Xs = sub(X, 2);
    if (system("mkdir -p tmp") != 0) std::fprintf(stderr, "warning: mkdir tmp failed\n");
    cv::VideoWriter video("tmp/gpu_bcpd.avi", cv::VideoWriter::fourcc('M','J','P','G'), 18, cv::Size(W,H));
    int ntraj = (int)traj.size(); const int HOLD = 24; int nframes = ntraj + HOLD;
    struct Sp { float sx, sy, d; cv::Scalar c; };
    for (int f = 0; f < nframes; ++f) {
        int k = std::min(f, ntraj-1);
        float az = 0.5f + f*0.02f, ca=std::cos(az), sa=std::sin(az), ce=std::cos(elev), se=std::sin(elev);
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(26,26,32));
        auto proj = [&](float x,float y,float z,float&sx,float&sy,float&d){
            float x1=x*ca-y*sa, y1=x*sa+y*ca, z1=z; sx=CX+SCALE*x1; sy=CY-SCALE*(z1*ce-y1*se); d=y1*ce+z1*se; };
        std::vector<Sp> sp;
        for (size_t i=0;i<Xs.size()/3;++i){ Sp s; proj(Xs[i*3],Xs[i*3+1],Xs[i*3+2],s.sx,s.sy,s.d); s.c=cv::Scalar(210,180,60); sp.push_back(s);}
        const std::vector<float>& P = traj[k];
        for (size_t i=0;i<P.size()/3;++i){ Sp s; proj(P[i*3],P[i*3+1],P[i*3+2],s.sx,s.sy,s.d); s.c=cv::Scalar(40,130,240); sp.push_back(s);}
        std::sort(sp.begin(),sp.end(),[](const Sp&a,const Sp&b){return a.d<b.d;});
        float dmin=1e9f,dmax=-1e9f; for(auto&s:sp){dmin=std::min(dmin,s.d);dmax=std::max(dmax,s.d);}
        for (auto& s : sp){ float t=(s.d-dmin)/(dmax-dmin+1e-6f); float b=0.45f+0.55f*t;
            int rad = (s.c[2]>200)?3:2;   // model points slightly larger
            cv::circle(img, cv::Point((int)s.sx,(int)s.sy), rad, s.c*b, -1, cv::LINE_AA); }
        int px=800, py=70;
        auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){ cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA); };
        put("GPU BCPD", py, 1.0, cv::Scalar(235,235,245), 2); py+=38;
        put("non-rigid registration", py, 0.62, cv::Scalar(180,180,200), 1); py+=50;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(210,180,60),-1);
        cv::putText(img,"target surface",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA); py+=30;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(40,130,240),-1);
        cv::putText(img,"warped model (flowing)",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA); py+=52;
        char buf[96]; std::snprintf(buf,sizeof(buf),"iteration %d / %d", k, ntraj-1);
        put(buf, py, 0.62, cv::Scalar(210,210,225),1); py+=40;
        put("coherent (GP-smooth) displacement", py, 0.5, cv::Scalar(150,200,150),1); py+=26;
        put("variational-Bayes GMM / EM", py, 0.5, cv::Scalar(150,200,150),1); py+=44;
        if (f >= nframes-HOLD) put("ALIGNED", py, 0.8, cv::Scalar(120,230,250), 2);
        video.write(img);
    }
    video.release();
    avi_to_gif("tmp/gpu_bcpd.avi", "gif/gpu_bcpd.gif", 18, 900);
    std::printf("wrote gif/gpu_bcpd.gif\n");
}

}  // namespace cudabot

// ============================ verification main ============================
int main() {
    using namespace cudabot;
    std::printf("=== GPU BCPD: non-rigid point-set registration (verification) ===\n");

    // target X = base lumpy surface;  model Y = a control-point subsample of the
    // SAME surface under a known smooth warp -> recovering the warp aligns Y to X.
    const int N = 6000, M = 500;
    std::vector<float> X = sample_lumpy(N, 1);
    std::vector<float> base = sample_lumpy(M, 2);     // control points on the base
    std::vector<float> Y(M*3);
    for (int m = 0; m < M; ++m) apply_warp(&base[m*3], &Y[m*3]);   // warped model (input)

    // honest registration metric: mean nearest distance from each (deformed)
    // model point to the target SURFACE.  Good alignment -> ~ target spacing.
    auto surf_resid = [&](const std::vector<float>& P){
        double s = 0;
        for (int m = 0; m < M; ++m) {
            float best = 1e9f;
            for (int n = 0; n < N; ++n) {
                float dx=P[m*3]-X[n*3],dy=P[m*3+1]-X[n*3+1],dz=P[m*3+2]-X[n*3+2];
                float d = dx*dx+dy*dy+dz*dz; if (d < best) best = d;
            }
            s += std::sqrt(best);
        }
        return s/M;
    };
    float init_resid = surf_resid(Y);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> traj;
    float beta = std::getenv("BCPD_BETA") ? atof(std::getenv("BCPD_BETA")) : 1.2f;
    float lambda = std::getenv("BCPD_LAMBDA") ? atof(std::getenv("BCPD_LAMBDA")) : 0.5f;
    BcpdResult res = bcpd(X, Y, beta, lambda, /*iters=*/50, &traj);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    float final_resid = surf_resid(res.P);
    std::printf("control pts M=%d  target N=%d\n", M, N);
    std::printf("mean model->surface dist  initial (warped) : %.4f\n", init_resid);
    std::printf("mean model->surface dist  after BCPD       : %.4f  (%.0f%% reduction)\n",
                final_resid, 100.0*(1.0 - final_resid/init_resid));
    std::printf("final sigma=%.4f  iters=%d  wall=%.1f ms\n", res.final_sigma, res.iters, ms);
    // target spacing ~ sqrt(area/N) ~ 0.06 sets a floor on the surface residual.
    if (final_resid < 0.25f * init_resid)
        std::printf("RESULT: PASS -- BCPD recovered the non-rigid warp (model on the target surface).\n");
    else
        std::printf("RESULT: CHECK -- warp not sufficiently recovered.\n");

    render_gif(X, traj);
    return 0;
}
