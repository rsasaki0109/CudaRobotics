// gpu_sinkhorn_reg.cu
//
// GPU optimal-transport point-cloud registration (Sinkhorn / unbalanced OT).
//
// Third pillar of the probabilistic-registration line, and a DIFFERENT paradigm
// from the EM/GMM members (FilterReg, BCPD): instead of a Gaussian-mixture
// likelihood it computes a soft correspondence as an entropic optimal-transport
// plan, in the spirit of
//   Feydy et al., "Interpolating between Optimal Transport and MMD" / GeomLoss,
//   and the robust-OT registration line (e.g. Shen et al., NeurIPS 2021).
//
// Why OT here: the transport plan is a doubly-(near-)stochastic coupling, so
// mass is conserved by construction, and the UNBALANCED variant relaxes the
// marginals with a KL penalty -- which makes it robust to outliers and partial
// overlap (unmatched mass is simply dropped) without any explicit outlier model.
// The core is log-domain Sinkhorn scaling: alternating soft-min reductions over
// the M x N cost matrix.  That is a dense matrix-vector pattern -- exactly what a
// GPU eats for breakfast (one thread per row / per column, on-the-fly costs).
//
// Pipeline (coarse-to-fine in the entropy epsilon):
//   1. cost C_{mn} = ||T y_m - x_n||^2  (formed on the fly, never stored).
//   2. log-domain (unbalanced) Sinkhorn -> potentials f_m, g_n -> plan P.
//   3. barycentric map  mu_m = (sum_n P_{mn} x_n) / (sum_n P_{mn})  + weight a_m.
//   4. weighted rigid fit p_m -> mu_m by a few se(3) twist Gauss-Newton steps.
//   5. update T, anneal epsilon, repeat.
//
// Verification recovers a known SE(3); then the convergence GIF.  Build:
// CMakeLists, --expt-relaxed-constexpr.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ============================ SE(3) helpers (host) ============================
struct Mat3 { float m[9]; };
struct Pose { Mat3 R; float t[3]; };
static inline void mat3_vec(const Mat3& R, const float* v, float* o){
    o[0]=R.m[0]*v[0]+R.m[1]*v[1]+R.m[2]*v[2]; o[1]=R.m[3]*v[0]+R.m[4]*v[1]+R.m[5]*v[2]; o[2]=R.m[6]*v[0]+R.m[7]*v[1]+R.m[8]*v[2]; }
static inline void pose_apply(const Pose& T, const float* y, float* p){ mat3_vec(T.R,y,p); p[0]+=T.t[0];p[1]+=T.t[1];p[2]+=T.t[2]; }
static inline Mat3 mat3_mul(const Mat3&A,const Mat3&B){ Mat3 C; for(int i=0;i<3;++i)for(int j=0;j<3;++j){float s=0;for(int k=0;k<3;++k)s+=A.m[i*3+k]*B.m[k*3+j];C.m[i*3+j]=s;} return C; }
static inline Mat3 so3_exp(const float* w){
    float th=std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]); Mat3 R;
    if(th<1e-9f){R={1,0,0,0,1,0,0,0,1};return R;}
    float a=w[0]/th,b=w[1]/th,c=w[2]/th,s=std::sin(th),co=std::cos(th),v=1-co;
    R.m[0]=a*a*v+co; R.m[1]=a*b*v-c*s; R.m[2]=a*c*v+b*s;
    R.m[3]=a*b*v+c*s; R.m[4]=b*b*v+co; R.m[5]=b*c*v-a*s;
    R.m[6]=a*c*v-b*s; R.m[7]=b*c*v+a*s; R.m[8]=c*c*v+co; return R; }
static inline Pose se3_exp(const float* xi){
    const float*v=xi,*w=xi+3; Pose T; T.R=so3_exp(w);
    float th=std::sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]); float Vm[9];
    if(th<1e-6f){for(int i=0;i<9;++i)Vm[i]=(i%4==0)?1.f:0.f;}
    else{ float A=(1-std::cos(th))/(th*th),B=(th-std::sin(th))/(th*th*th);
        float wx[9]={0,-w[2],w[1],w[2],0,-w[0],-w[1],w[0],0},wx2[9];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){float s=0;for(int k=0;k<3;++k)s+=wx[i*3+k]*wx[k*3+j];wx2[i*3+j]=s;}
        for(int i=0;i<9;++i)Vm[i]=((i%4==0)?1.f:0.f)+A*wx[i]+B*wx2[i]; }
    T.t[0]=Vm[0]*v[0]+Vm[1]*v[1]+Vm[2]*v[2]; T.t[1]=Vm[3]*v[0]+Vm[4]*v[1]+Vm[5]*v[2]; T.t[2]=Vm[6]*v[0]+Vm[7]*v[1]+Vm[8]*v[2];
    return T; }
static inline Pose pose_mul(const Pose&A,const Pose&B){ Pose C; C.R=mat3_mul(A.R,B.R); float Rt[3]; mat3_vec(A.R,B.t,Rt); for(int k=0;k<3;++k)C.t[k]=Rt[k]+A.t[k]; return C; }

// ============================ procedural cloud ============================
static std::vector<float> make_lumpy(int n, unsigned seed){
    std::vector<float> pts(n*3); std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uu(-1,1),up(0,6.2831853f);
    const float bumps[][5]={{0.8f,0.2f,0.5f,0.9f,0.25f},{-0.3f,0.9f,0.2f,0.7f,0.30f},{0.1f,-0.6f,0.8f,0.8f,0.22f},{-0.7f,-0.4f,-0.5f,1.0f,0.28f},{0.4f,0.3f,-0.85f,0.6f,0.20f}};
    for(int i=0;i<n;++i){ float z=uu(rng),phi=up(rng),r2=std::sqrt(std::max(0.f,1-z*z));
        float dx=r2*std::cos(phi),dy=r2*std::sin(phi),dz=z;
        float R=2.0f+0.35f*std::sin(3*phi)*(1-z*z)+0.30f*dz*dx+0.20f*std::cos(2*phi);
        for(int b=0;b<5;++b){float d=dx*bumps[b][0]+dy*bumps[b][1]+dz*bumps[b][2];float a=1-d;R+=bumps[b][3]*std::exp(-a*a/(2*bumps[b][4]*bumps[b][4]));}
        pts[i*3]=R*dx;pts[i*3+1]=R*dy;pts[i*3+2]=R*dz; }
    return pts; }

// ============================ log-domain (unbalanced) Sinkhorn ============================
// f update: f_m = (rho/(rho+eps)) * ( eps*log a_m - eps * LSE_n[(g_n - C_mn)/eps] ).
// rho = marginal-relaxation strength (rho -> inf recovers balanced OT).
__global__ void sinkhorn_f_kernel(const float* __restrict__ P, int M,
                                  const float* __restrict__ X, int N,
                                  const float* __restrict__ g, float eps, float logaM,
                                  float scale, float* __restrict__ f) {
    int m = blockIdx.x*blockDim.x + threadIdx.x; if (m >= M) return;
    float p0=P[m*3],p1=P[m*3+1],p2=P[m*3+2];
    float mx = -1e30f;
    for (int n = 0; n < N; ++n) {
        float dx=p0-X[n*3],dy=p1-X[n*3+1],dz=p2-X[n*3+2];
        float v = (g[n] - (dx*dx+dy*dy+dz*dz)) / eps; if (v > mx) mx = v;
    }
    float s = 0.f;
    for (int n = 0; n < N; ++n) {
        float dx=p0-X[n*3],dy=p1-X[n*3+1],dz=p2-X[n*3+2];
        s += __expf((g[n] - (dx*dx+dy*dy+dz*dz))/eps - mx);
    }
    float lse = mx + __logf(s + 1e-30f);
    f[m] = scale * (eps*logaM - eps*lse);
}
__global__ void sinkhorn_g_kernel(const float* __restrict__ P, int M,
                                  const float* __restrict__ X, int N,
                                  const float* __restrict__ f, float eps, float logbN,
                                  float scale, float* __restrict__ g) {
    int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= N) return;
    float x0=X[n*3],x1=X[n*3+1],x2=X[n*3+2];
    float mx = -1e30f;
    for (int m = 0; m < M; ++m) {
        float dx=P[m*3]-x0,dy=P[m*3+1]-x1,dz=P[m*3+2]-x2;
        float v = (f[m] - (dx*dx+dy*dy+dz*dz)) / eps; if (v > mx) mx = v;
    }
    float s = 0.f;
    for (int m = 0; m < M; ++m) {
        float dx=P[m*3]-x0,dy=P[m*3+1]-x1,dz=P[m*3+2]-x2;
        s += __expf((f[m] - (dx*dx+dy*dy+dz*dz))/eps - mx);
    }
    float lse = mx + __logf(s + 1e-30f);
    g[n] = scale * (eps*logbN - eps*lse);
}
// barycentric map: w_m = sum_n P_mn,  mu_m = sum_n P_mn x_n / w_m,
// with P_mn = exp((f_m + g_n - C_mn)/eps).
__global__ void barycentric_kernel(const float* __restrict__ P, int M,
                                   const float* __restrict__ X, int N,
                                   const float* __restrict__ f, const float* __restrict__ g,
                                   float eps, float* __restrict__ mu, float* __restrict__ w) {
    int m = blockIdx.x*blockDim.x + threadIdx.x; if (m >= M) return;
    float p0=P[m*3],p1=P[m*3+1],p2=P[m*3+2], fm=f[m];
    float s0=0,sx=0,sy=0,sz=0;
    for (int n = 0; n < N; ++n) {
        float dx=p0-X[n*3],dy=p1-X[n*3+1],dz=p2-X[n*3+2];
        float pmn = __expf((fm + g[n] - (dx*dx+dy*dy+dz*dz))/eps);
        s0 += pmn; sx += pmn*X[n*3]; sy += pmn*X[n*3+1]; sz += pmn*X[n*3+2];
    }
    w[m] = s0;
    float inv = 1.f/(s0 + 1e-20f);
    mu[m*3]=sx*inv; mu[m*3+1]=sy*inv; mu[m*3+2]=sz*inv;
}

// transform model: P = R Y + t
__global__ void transform_kernel(const float* __restrict__ Y, int M, const float* __restrict__ R,
                                 const float* __restrict__ t, float* __restrict__ P){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return;
    float y0=Y[j*3],y1=Y[j*3+1],y2=Y[j*3+2];
    P[j*3]=R[0]*y0+R[1]*y1+R[2]*y2+t[0]; P[j*3+1]=R[3]*y0+R[4]*y1+R[5]*y2+t[1]; P[j*3+2]=R[6]*y0+R[7]*y1+R[8]*y2+t[2]; }

// weighted twist Gauss-Newton accumulation (p_m -> mu_m, weight w_m), as FilterReg.
__global__ void mstep_kernel(const float* __restrict__ P, const float* __restrict__ MU,
                             const float* __restrict__ W, int M, float* __restrict__ Hg /*21+6+2*/){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return;
    float w=W[j]; if(w<1e-12f)return;
    float px=P[j*3],py=P[j*3+1],pz=P[j*3+2];
    float rx=px-MU[j*3],ry=py-MU[j*3+1],rz=pz-MU[j*3+2];
    float J[18]={1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float Hl[21]; int c=0;
    for(int a=0;a<6;++a)for(int b=a;b<6;++b){ float s=J[0*6+a]*J[0*6+b]+J[1*6+a]*J[1*6+b]+J[2*6+a]*J[2*6+b]; Hl[c++]=w*s; }
    float gl[6]; for(int a=0;a<6;++a) gl[a]=w*(J[0*6+a]*rx+J[1*6+a]*ry+J[2*6+a]*rz);
    for(int k=0;k<21;++k) atomicAdd(&Hg[k],Hl[k]);
    for(int k=0;k<6;++k) atomicAdd(&Hg[21+k],gl[k]);
    atomicAdd(&Hg[27], w*(rx*rx+ry*ry+rz*rz)); atomicAdd(&Hg[28], w); }

static bool solve6(const float* Hut, const float* gg, float* d){
    float H[36]; int c=0; for(int a=0;a<6;++a)for(int b=a;b<6;++b){H[a*6+b]=H[b*6+a]=Hut[c++];}
    for(int i=0;i<6;++i)H[i*6+i]+=1e-6f;
    float L[36]={0};
    for(int i=0;i<6;++i)for(int j=0;j<=i;++j){ float s=H[i*6+j]; for(int k=0;k<j;++k)s-=L[i*6+k]*L[j*6+k];
        if(i==j){if(s<=0)return false;L[i*6+i]=std::sqrt(s);} else L[i*6+j]=s/L[j*6+j]; }
    float y[6]; for(int i=0;i<6;++i){float s=-gg[i];for(int k=0;k<i;++k)s-=L[i*6+k]*y[k];y[i]=s/L[i*6+i];}
    for(int i=5;i>=0;--i){float s=y[i];for(int k=i+1;k<6;++k)s-=L[k*6+i]*d[k];d[i]=s/L[i*6+i];}
    return true; }

// ============================ driver ============================
struct SinkResult { Pose T; int iters; };
static SinkResult sinkhorn_register(const std::vector<float>& X, const std::vector<float>& Y,
                                    Pose T0, float rho, std::vector<Pose>* traj = nullptr) {
    int N=X.size()/3, M=Y.size()/3;
    float *dX,*dY,*dP,*dR,*dt,*df,*dg,*dmu,*dw,*dHg;
    CUDA_CHECK(cudaMalloc(&dX,N*3*sizeof(float))); CUDA_CHECK(cudaMemcpy(dX,X.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dY,M*3*sizeof(float))); CUDA_CHECK(cudaMemcpy(dY,Y.data(),M*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP,M*3*sizeof(float))); CUDA_CHECK(cudaMalloc(&dR,9*sizeof(float))); CUDA_CHECK(cudaMalloc(&dt,3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&df,M*sizeof(float))); CUDA_CHECK(cudaMalloc(&dg,N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dmu,M*3*sizeof(float))); CUDA_CHECK(cudaMalloc(&dw,M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dHg,29*sizeof(float)));
    float logaM=-std::log((float)M), logbN=-std::log((float)N);

    Pose T=T0; SinkResult res; res.iters=0; if(traj)traj->push_back(T);
    // coarse-to-fine entropy: large eps = soft/global, small eps = sharp.
    const float epsilons[]={1.2f,0.7f,0.4f,0.25f,0.15f,0.10f};
    for (float eps : epsilons) {
        float scale = rho/(rho+eps);             // unbalanced relaxation
        for (int outer = 0; outer < 8; ++outer) {
            CUDA_CHECK(cudaMemcpy(dR,T.R.m,9*sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt,T.t,3*sizeof(float),cudaMemcpyHostToDevice));
            transform_kernel<<<(M+255)/256,256>>>(dY,M,dR,dt,dP);
            CUDA_CHECK(cudaMemset(dg,0,N*sizeof(float)));
            // Sinkhorn iterations (warm-started across outer steps would need
            // persistent f,g; we reset per outer for simplicity/robustness)
            CUDA_CHECK(cudaMemset(df,0,M*sizeof(float)));
            for (int s = 0; s < 60; ++s) {
                sinkhorn_f_kernel<<<(M+255)/256,256>>>(dP,M,dX,N,dg,eps,logaM,scale,df);
                sinkhorn_g_kernel<<<(N+255)/256,256>>>(dP,M,dX,N,df,eps,logbN,scale,dg);
            }
            barycentric_kernel<<<(M+255)/256,256>>>(dP,M,dX,N,df,dg,eps,dmu,dw);
            // a few weighted twist GN steps to fit the rigid transform to mu
            for (int gn = 0; gn < 3; ++gn) {
                if (gn > 0) {
                    CUDA_CHECK(cudaMemcpy(dR,T.R.m,9*sizeof(float),cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dt,T.t,3*sizeof(float),cudaMemcpyHostToDevice));
                    transform_kernel<<<(M+255)/256,256>>>(dY,M,dR,dt,dP);
                }
                CUDA_CHECK(cudaMemset(dHg,0,29*sizeof(float)));
                mstep_kernel<<<(M+255)/256,256>>>(dP,dmu,dw,M,dHg);
                float Hg[29]; CUDA_CHECK(cudaMemcpy(Hg,dHg,29*sizeof(float),cudaMemcpyDeviceToHost));
                float d[6]; if(!solve6(Hg,Hg+21,d))break;
                T = pose_mul(se3_exp(d), T);
            }
            ++res.iters; if(traj)traj->push_back(T);
        }
    }
    res.T=T;
    cudaFree(dX);cudaFree(dY);cudaFree(dP);cudaFree(dR);cudaFree(dt);cudaFree(df);cudaFree(dg);cudaFree(dmu);cudaFree(dw);cudaFree(dHg);
    return res;
}

// ============================ convergence GIF ============================
static void render_gif(const std::vector<float>& X, const std::vector<float>& Y,
                       const std::vector<Pose>& traj){
    const int W=1280,H=720,CX=380,CY=360; const float SCALE=80.f,elev=0.42f;
    auto sub=[](const std::vector<float>&P,int st){std::vector<float>q;for(size_t i=0;i<P.size()/3;i+=st){q.push_back(P[i*3]);q.push_back(P[i*3+1]);q.push_back(P[i*3+2]);}return q;};
    std::vector<float> Xs=sub(X,2),Ys=sub(Y,2);
    if(system("mkdir -p tmp")!=0)std::fprintf(stderr,"warn mkdir\n");
    cv::VideoWriter video("tmp/gpu_sinkhorn_reg.avi",cv::VideoWriter::fourcc('M','J','P','G'),18,cv::Size(W,H));
    int nt=(int)traj.size(); const int HOLD=24; int nf=nt+HOLD;
    struct Sp{float sx,sy,d;cv::Scalar c;};
    for(int f=0;f<nf;++f){ int k=std::min(f,nt-1); float az=0.5f+f*0.02f,ca=std::cos(az),sa=std::sin(az),ce=std::cos(elev),se=std::sin(elev);
        cv::Mat img(H,W,CV_8UC3,cv::Scalar(26,26,32)); const Pose&T=traj[k];
        auto proj=[&](float x,float y,float z,float&sx,float&sy,float&d){float x1=x*ca-y*sa,y1=x*sa+y*ca,z1=z;sx=CX+SCALE*x1;sy=CY-SCALE*(z1*ce-y1*se);d=y1*ce+z1*se;};
        std::vector<Sp> sp;
        for(size_t i=0;i<Xs.size()/3;++i){Sp s;proj(Xs[i*3],Xs[i*3+1],Xs[i*3+2],s.sx,s.sy,s.d);s.c=cv::Scalar(210,180,60);sp.push_back(s);}
        for(size_t i=0;i<Ys.size()/3;++i){float y0[3]={Ys[i*3],Ys[i*3+1],Ys[i*3+2]},p[3];pose_apply(T,y0,p);Sp s;proj(p[0],p[1],p[2],s.sx,s.sy,s.d);s.c=cv::Scalar(40,130,240);sp.push_back(s);}
        std::sort(sp.begin(),sp.end(),[](const Sp&a,const Sp&b){return a.d<b.d;});
        float dmin=1e9f,dmax=-1e9f;for(auto&s:sp){dmin=std::min(dmin,s.d);dmax=std::max(dmax,s.d);}
        for(auto&s:sp){float t=(s.d-dmin)/(dmax-dmin+1e-6f);float b=0.45f+0.55f*t;cv::circle(img,cv::Point((int)s.sx,(int)s.sy),2,s.c*b,-1,cv::LINE_AA);}
        int px=800,py=70; auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA);};
        put("GPU Sinkhorn-OT",py,0.95,cv::Scalar(235,235,245),2);py+=38;
        put("optimal-transport registration",py,0.58,cv::Scalar(180,180,200),1);py+=50;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(210,180,60),-1);cv::putText(img,"target",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=30;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(40,130,240),-1);cv::putText(img,"source (transporting)",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=52;
        char buf[96];std::snprintf(buf,sizeof(buf),"outer step %d / %d",k,nt-1);put(buf,py,0.62,cv::Scalar(210,210,225),1);py+=40;
        put("log-domain Sinkhorn coupling",py,0.5,cv::Scalar(150,200,150),1);py+=26;
        put("unbalanced OT (outlier-robust)",py,0.5,cv::Scalar(150,200,150),1);py+=44;
        if(f>=nf-HOLD)put("ALIGNED",py,0.8,cv::Scalar(120,230,250),2);
        video.write(img); }
    video.release(); avi_to_gif("tmp/gpu_sinkhorn_reg.avi","gif/gpu_sinkhorn_reg.gif",18,900);
    std::printf("wrote gif/gpu_sinkhorn_reg.gif\n");
}

}  // namespace cudabot

int main() {
    using namespace cudabot;
    std::printf("=== GPU Sinkhorn-OT registration (unbalanced optimal transport) ===\n");
    const int N=2600;
    std::vector<float> X = make_lumpy(N, 1);
    std::mt19937 rng(7);
    float gt_w[3]={0.22f,-0.30f,0.18f}, gt_t[3]={0.6f,-0.45f,0.35f};
    Mat3 Rgt=so3_exp(gt_w); Pose Tgt; Tgt.R=Rgt; for(int k=0;k<3;++k)Tgt.t[k]=gt_t[k];
    std::normal_distribution<float> noise(0.f,0.02f); std::uniform_real_distribution<float> keep(0,1);
    std::vector<float> Y;
    for(int i=0;i<N;++i){ if(keep(rng)>0.85f)continue; float y[3]={X[i*3],X[i*3+1],X[i*3+2]},p[3]; pose_apply(Tgt,y,p);
        Y.push_back(p[0]+noise(rng));Y.push_back(p[1]+noise(rng));Y.push_back(p[2]+noise(rng)); }
    int Mr=Y.size()/3;
    std::printf("target N=%d  source M=%d  (15%% dropped, noise 0.02)\n", N, Mr);

    Pose T0; T0.R={1,0,0,0,1,0,0,0,1}; T0.t[0]=T0.t[1]=T0.t[2]=0;
    std::vector<Pose> traj;
    auto t0=std::chrono::high_resolution_clock::now();
    SinkResult res = sinkhorn_register(X, Y, T0, /*rho=*/3.0f, &traj);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    Mat3 RgtT; for(int i=0;i<3;++i)for(int j=0;j<3;++j)RgtT.m[i*3+j]=Rgt.m[j*3+i];
    float texp[3]; mat3_vec(RgtT,gt_t,texp); for(int k=0;k<3;++k)texp[k]=-texp[k];
    Mat3 Rerr=mat3_mul(Rgt,res.T.R); float tr=Rerr.m[0]+Rerr.m[4]+Rerr.m[8];
    float ang=std::acos(std::min(1.f,std::max(-1.f,(tr-1.f)*0.5f)));
    float terr=0; for(int k=0;k<3;++k){float e=res.T.t[k]-texp[k];terr+=e*e;} terr=std::sqrt(terr);
    std::printf("recovered rot angle err = %.4f rad (%.3f deg)\n", ang, ang*57.2958f);
    std::printf("recovered trans=(% .3f % .3f % .3f) expected (% .3f % .3f % .3f) err=%.4f\n",
                res.T.t[0],res.T.t[1],res.T.t[2],texp[0],texp[1],texp[2],terr);
    std::printf("outer iters=%d  wall=%.1f ms\n", res.iters, ms);
    if(ang<0.03f && terr<0.06f) std::printf("RESULT: PASS -- Sinkhorn-OT recovered the known transform.\n");
    else std::printf("RESULT: CHECK -- transform not recovered within tolerance.\n");

    render_gif(X, Y, traj);
    return 0;
}
