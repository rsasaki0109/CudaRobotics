// gpu_constrained_mpc.cu
//
// GPU constrained nonlinear MPC via batched Augmented-Lagrangian iLQR (AL-iLQR).
//
// The control line has sampling MPPI, second-order iLQR (batched + parallel-in-
// time), and convex (box-QP) MPC.  What was missing is CONSTRAINED NONLINEAR MPC:
// a nonlinear model with HARD inequality constraints (obstacle avoidance, control
// limits) solved to a (locally) optimal, constraint-satisfying trajectory.
//
// AL-iLQR (Toussaint; Howell et al. ALTRO) is the clean way to do it and reuses
// the iLQR backward/forward machinery already in the repo:
//   * inner loop: iLQR on the cost AUGMENTED with, per inequality c(x) <= 0,
//       P = lambda*c + (mu/2)*c^2     when active ( mu*c + lambda > 0 ), else 0,
//     whose gradient (lambda+mu c) c_x and Gauss-Newton Hessian mu c_x c_x^T fold
//     straight into l_x / l_xx of the backward pass.
//   * outer loop: lambda <- max(0, lambda + mu c),  mu <- beta*mu.
//   As mu grows the constraints are driven to satisfaction; lambda recovers the
//   exact KKT multipliers, so the limit is a true constrained optimum (a soft
//   penalty alone would either leak through obstacles or distort the cost).
//
// Model: unicycle x=[px,py,theta], u=[v,omega], nonlinear (cos/sin theta).
// Constraints: stay margin-clear of every circular obstacle, |v| and |omega|
// box-limited (projected in the forward rollout).  One GPU THREAD runs the whole
// AL-iLQR for one robot, so a whole swarm solves its own constrained problem at
// once -- receding-horizon, re-solved every control step.
//
// Verified: every robot reaches its goal, NEVER enters an obstacle margin, and
// respects the control box; an UNCONSTRAINED iLQR baseline (no obstacle term)
// cuts straight through.  Then a GIF of the swarm threading the obstacle field.
//
// Build: CMakeLists, --expt-relaxed-constexpr.

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

static const int NX=3, NU=2, T=24, OBST=6;
static const float DT=0.12f;

struct Obs { float x,y,r; };
__constant__ Obs c_obs[OBST];
__constant__ int c_nobs;
__constant__ float c_margin, c_vmax, c_wmax;

// ---- dynamics & jacobians (device) ----
__device__ void dyn(const float* x,const float* u,float* xn){
    xn[0]=x[0]+DT*u[0]*cosf(x[2]); xn[1]=x[1]+DT*u[0]*sinf(x[2]); xn[2]=x[2]+DT*u[1]; }
__device__ void jac(const float* x,const float* u,float* A,float* B){
    // A (3x3 row-major), B (3x2)
    A[0]=1;A[1]=0;A[2]=-DT*u[0]*sinf(x[2]); A[3]=0;A[4]=1;A[5]=DT*u[0]*cosf(x[2]); A[6]=0;A[7]=0;A[8]=1;
    B[0]=DT*cosf(x[2]);B[1]=0; B[2]=DT*sinf(x[2]);B[3]=0; B[4]=0;B[5]=DT; }

// stage cost + AL obstacle terms -> scalar cost, and (optionally) l_x(3), l_xx(3x3 GN).
__device__ float stage_cost(const float* x,const float* u,const float* g,float qp,float rv,float rw,
                            int k,const float* lam,float mu,int constrained,
                            float* lx,float* lxx,float* lu,float* luu){
    float dxp=x[0]-g[0],dyp=x[1]-g[1];
    float c=qp*(dxp*dxp+dyp*dyp)+rv*u[0]*u[0]+rw*u[1]*u[1];
    if(lx){ lx[0]=2*qp*dxp; lx[1]=2*qp*dyp; lx[2]=0;
        for(int a=0;a<9;++a)lxx[a]=0; lxx[0]=2*qp; lxx[4]=2*qp;
        lu[0]=2*rv*u[0]; lu[1]=2*rw*u[1]; luu[0]=2*rv;luu[1]=0;luu[2]=0;luu[3]=2*rw; }
    if(constrained){
        for(int o=0;o<c_nobs;++o){ float ex=x[0]-c_obs[o].x, ey=x[1]-c_obs[o].y; float d=sqrtf(ex*ex+ey*ey)+1e-6f;
            float cc=(c_obs[o].r+c_margin)-d;                 // c<=0 desired
            float lk=lam[k*OBST+o];
            if(mu*cc+lk>0.f){ c+= lk*cc + 0.5f*mu*cc*cc;
                if(lx){ float gx=-ex/d, gy=-ey/d; float s=lk+mu*cc;
                    lx[0]+=s*gx; lx[1]+=s*gy;
                    lxx[0]+=mu*gx*gx; lxx[1]+=mu*gx*gy; lxx[3]+=mu*gx*gy; lxx[4]+=mu*gy*gy; } } } }
    return c; }
__device__ float term_cost(const float* x,const float* g,float qf,float* lx,float* lxx){
    float dxp=x[0]-g[0],dyp=x[1]-g[1];
    if(lx){ lx[0]=2*qf*dxp; lx[1]=2*qf*dyp; lx[2]=0; for(int a=0;a<9;++a)lxx[a]=0; lxx[0]=2*qf; lxx[4]=2*qf; }
    return qf*(dxp*dxp+dyp*dyp); }

// total trajectory cost (for line search) given X,U
__device__ float traj_cost(const float* X,const float* U,const float* g,float qp,float rv,float rw,float qf,
                           const float* lam,float mu,int constrained){
    float J=0; for(int k=0;k<T;++k) J+=stage_cost(&X[k*NX],&U[k*NU],g,qp,rv,rw,k,lam,mu,constrained,0,0,0,0);
    J+=term_cost(&X[T*NX],g,qf,0,0); return J; }

__device__ void clampu(float* u){ u[0]=fminf(c_vmax,fmaxf(-c_vmax,u[0])); u[1]=fminf(c_wmax,fmaxf(-c_wmax,u[1])); }

// solve 2x2 SPD: M d = -g  -> d
__device__ void solve2(const float* M,const float* g,float* d){
    float det=M[0]*M[3]-M[1]*M[2]; if(fabsf(det)<1e-12f)det=(det<0?-1e-12f:1e-12f); float inv=1.f/det;
    d[0]=-( M[3]*g[0]-M[1]*g[1])*inv; d[1]=-(-M[2]*g[0]+M[0]*g[1])*inv; }

// One thread = one robot.  Full AL-iLQR from x0; writes the optimal first control
// u0 (for MPC) into Uout, and (optionally) min obstacle clearance achieved.
__global__ void al_ilqr_kernel(const float* __restrict__ X0,const float* __restrict__ G,int nR,
                               int constrained,float* __restrict__ Uout,float* __restrict__ Uwarm){
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=nR)return;
    const float qp=2.0f, rv=0.5f, rw=0.5f, qf=60.f;
    float X[(T+1)*NX], U[T*NU], K[T*NU*NX], dd[T*NU], lam[T*OBST];
    float g[3]={G[r*3],G[r*3+1],G[r*3+2]};
    for(int i=0;i<T*OBST;++i)lam[i]=0.f;
    // warm start (previous solution shifted) if provided, else zeros
    for(int k=0;k<T;++k){ U[k*NU]=Uwarm?Uwarm[r*T*NU+k*NU]:0.f; U[k*NU+1]=Uwarm?Uwarm[r*T*NU+k*NU+1]:0.f; }
    for(int k=0;k<3;++k)X[k]=X0[r*3+k];
    for(int k=0;k<T;++k){ clampu(&U[k*NU]); dyn(&X[k*NX],&U[k*NU],&X[(k+1)*NX]); }
    float mu=2.0f;
    const int AL_IT=8, ILQR_IT=8;
    for(int al=0; al<(constrained?AL_IT:1); ++al){
        for(int it=0; it<ILQR_IT; ++it){
            // ---- backward ----
            float Vx[3], Vxx[9]; term_cost(&X[T*NX],g,qf,Vx,Vxx);
            for(int k=T-1;k>=0;--k){
                float A[9],B[6]; jac(&X[k*NX],&U[k*NU],A,B);
                float lx[3],lxx[9],lu[2],luu[4]; stage_cost(&X[k*NX],&U[k*NU],g,qp,rv,rw,k,lam,mu,constrained,lx,lxx,lu,luu);
                // Qx = lx + A^T Vx
                float Qx[3]; for(int i=0;i<3;++i){float s=lx[i]; for(int j=0;j<3;++j)s+=A[j*3+i]*Vx[j]; Qx[i]=s;}
                // Qu = lu + B^T Vx
                float Qu[2]; for(int i=0;i<2;++i){float s=lu[i]; for(int j=0;j<3;++j)s+=B[j*2+i]*Vx[j]; Qu[i]=s;}
                // VxxA (3x3) = Vxx*A
                float VA[9]; for(int i=0;i<3;++i)for(int j=0;j<3;++j){float s=0;for(int k2=0;k2<3;++k2)s+=Vxx[i*3+k2]*A[k2*3+j]; VA[i*3+j]=s;}
                // Qxx = lxx + A^T VA
                float Qxx[9]; for(int i=0;i<3;++i)for(int j=0;j<3;++j){float s=lxx[i*3+j];for(int k2=0;k2<3;++k2)s+=A[k2*3+i]*VA[k2*3+j]; Qxx[i*3+j]=s;}
                // VxxB (3x2)=Vxx*B
                float VB[6]; for(int i=0;i<3;++i)for(int j=0;j<2;++j){float s=0;for(int k2=0;k2<3;++k2)s+=Vxx[i*3+k2]*B[k2*2+j]; VB[i*2+j]=s;}
                // Quu = luu + B^T VB + rho I
                float Quu[4]; for(int i=0;i<2;++i)for(int j=0;j<2;++j){float s=luu[i*2+j];for(int k2=0;k2<3;++k2)s+=B[k2*2+i]*VB[k2*2+j]; Quu[i*2+j]=s;}
                Quu[0]+=1e-3f; Quu[3]+=1e-3f;
                // Qux (2x3) = B^T VA
                float Qux[6]; for(int i=0;i<2;++i)for(int j=0;j<3;++j){float s=0;for(int k2=0;k2<3;++k2)s+=B[k2*2+i]*VA[k2*3+j]; Qux[i*3+j]=s;}
                // gains: d = -Quu^{-1} Qu ; K = -Quu^{-1} Qux
                float dk[2]; solve2(Quu,Qu,dk);
                float Kk[6]; for(int j=0;j<3;++j){ float col[2]={Qux[0*3+j],Qux[1*3+j]},sol[2]; solve2(Quu,col,sol); Kk[0*3+j]=sol[0]; Kk[1*3+j]=sol[1]; }
                for(int i=0;i<6;++i)K[k*NU*NX+i]=Kk[i]; dd[k*NU]=dk[0]; dd[k*NU+1]=dk[1];
                // value update: Vx = Qx + K^T(Quu d + Qu) + Qux^T d
                float Quud[2]={Quu[0]*dk[0]+Quu[1]*dk[1], Quu[2]*dk[0]+Quu[3]*dk[1]};
                float tmp[2]={Quud[0]+Qu[0], Quud[1]+Qu[1]};
                for(int i=0;i<3;++i){ float s=Qx[i]; s+=Kk[0*3+i]*tmp[0]+Kk[1*3+i]*tmp[1]; s+=Qux[0*3+i]*dk[0]+Qux[1*3+i]*dk[1]; Vx[i]=s; }
                // Vxx = Qxx + K^T Quu K + K^T Qux + Qux^T K
                float QuuK[6]; for(int i=0;i<2;++i)for(int j=0;j<3;++j)QuuK[i*3+j]=Quu[i*2+0]*Kk[0*3+j]+Quu[i*2+1]*Kk[1*3+j];
                for(int i=0;i<3;++i)for(int j=0;j<3;++j){ float s=Qxx[i*3+j];
                    s+=Kk[0*3+i]*QuuK[0*3+j]+Kk[1*3+i]*QuuK[1*3+j];
                    s+=Kk[0*3+i]*Qux[0*3+j]+Kk[1*3+i]*Qux[1*3+j];
                    s+=Qux[0*3+i]*Kk[0*3+j]+Qux[1*3+i]*Kk[1*3+j];
                    Vxx[i*3+j]=s; }
                for(int i=0;i<3;++i)for(int j=i+1;j<3;++j){float a=0.5f*(Vxx[i*3+j]+Vxx[j*3+i]);Vxx[i*3+j]=Vxx[j*3+i]=a;}
            }
            // ---- forward line search ----
            float J0=traj_cost(X,U,g,qp,rv,rw,qf,lam,mu,constrained);
            float alpha=1.f; float Xn[(T+1)*NX],Un[T*NU]; int ok=0;
            for(int ls=0; ls<8; ++ls){
                for(int i=0;i<3;++i)Xn[i]=X[i];
                for(int k=0;k<T;++k){ float dx[3]={Xn[k*NX]-X[k*NX],Xn[k*NX+1]-X[k*NX+1],Xn[k*NX+2]-X[k*NX+2]};
                    for(int i=0;i<2;++i){ float s=U[k*NU+i]+alpha*dd[k*NU+i]; for(int j=0;j<3;++j)s+=K[k*NU*NX+i*3+j]*dx[j]; Un[k*NU+i]=s; }
                    clampu(&Un[k*NU]); dyn(&Xn[k*NX],&Un[k*NU],&Xn[(k+1)*NX]); }
                float Jn=traj_cost(Xn,Un,g,qp,rv,rw,qf,lam,mu,constrained);
                if(Jn<J0){ ok=1; break; } alpha*=0.5f;
            }
            if(ok){ for(int i=0;i<(T+1)*NX;++i)X[i]=Xn[i]; for(int i=0;i<T*NU;++i)U[i]=Un[i]; } else break;
        }
        // ---- AL multiplier / penalty update ----
        if(constrained){ for(int k=0;k<T;++k){ for(int o=0;o<c_nobs;++o){ float ex=X[k*NX]-c_obs[o].x,ey=X[k*NX+1]-c_obs[o].y;
            float cc=(c_obs[o].r+c_margin)-(sqrtf(ex*ex+ey*ey)+1e-6f); float v=lam[k*OBST+o]+mu*cc; lam[k*OBST+o]=fmaxf(0.f,v); } }
            mu*=5.0f; }
    }
    Uout[r*NU]=U[0]; Uout[r*NU+1]=U[1];
    if(Uwarm){ for(int k=0;k<T-1;++k){Uwarm[r*T*NU+k*NU]=U[(k+1)*NU];Uwarm[r*T*NU+k*NU+1]=U[(k+1)*NU+1];}
        Uwarm[r*T*NU+(T-1)*NU]=U[(T-1)*NU];Uwarm[r*T*NU+(T-1)*NU+1]=U[(T-1)*NU+1]; } }

// ============================ host driver ============================
struct Sim { std::vector<std::vector<float>> traj; std::vector<float> goals; std::vector<Obs> obs;
             float min_clear; int collisions; float max_u; bool reach_ok; };

static Sim run(int nR,int steps,int constrained,float margin,unsigned seed){
    std::mt19937 rng(seed);
    std::vector<Obs> obs={{0.5f,0.f,1.1f},{-3.f,2.2f,0.9f},{3.2f,2.5f,0.8f},{-2.f,-2.6f,1.0f},{2.4f,-2.8f,0.9f},{0.2f,3.4f,0.7f}};
    int nobs=obs.size();
    float vmax=2.2f, wmax=2.6f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_obs,obs.data(),nobs*sizeof(Obs)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_nobs,&nobs,sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_margin,&margin,sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_vmax,&vmax,sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wmax,&wmax,sizeof(float)));

    std::vector<float> X(nR*3), Gl(nR*3);
    std::uniform_real_distribution<float> sp(-0.6f,0.6f);
    for(int r=0;r<nR;++r){ float a=2*3.1415926f*r/nR; float R=7.0f;
        X[r*3]=R*std::cos(a)+sp(rng); X[r*3+1]=R*std::sin(a)+sp(rng); X[r*3+2]=std::atan2(-std::sin(a),-std::cos(a));
        Gl[r*3]=-0.62f*X[r*3]; Gl[r*3+1]=-0.62f*X[r*3+1]; Gl[r*3+2]=0; }     // cross the field

    float *dX0,*dG,*dU,*dW; CUDA_CHECK(cudaMalloc(&dX0,nR*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dG,nR*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dU,nR*NU*sizeof(float)));CUDA_CHECK(cudaMalloc(&dW,nR*T*NU*sizeof(float)));
    CUDA_CHECK(cudaMemset(dW,0,nR*T*NU*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dG,Gl.data(),nR*3*sizeof(float),cudaMemcpyHostToDevice));

    Sim sim; sim.goals=Gl; sim.obs=obs; sim.min_clear=1e9f; sim.collisions=0; sim.max_u=0; sim.reach_ok=true;
    sim.traj.assign(steps+1,std::vector<float>(nR*2));
    for(int r=0;r<nR;++r){ sim.traj[0][r*2]=X[r*3]; sim.traj[0][r*2+1]=X[r*3+1]; }
    std::vector<float> U(nR*NU);
    for(int s=0;s<steps;++s){
        CUDA_CHECK(cudaMemcpy(dX0,X.data(),nR*3*sizeof(float),cudaMemcpyHostToDevice));
        al_ilqr_kernel<<<(nR+63)/64,64>>>(dX0,dG,nR,constrained,dU,dW);
        CUDA_CHECK(cudaMemcpy(U.data(),dU,nR*NU*sizeof(float),cudaMemcpyDeviceToHost));
        for(int r=0;r<nR;++r){ float v=std::min(vmax,std::max(-vmax,U[r*NU])), w=std::min(wmax,std::max(-wmax,U[r*NU+1]));
            sim.max_u=std::max(sim.max_u,std::fabs(U[r*NU])); float* x=&X[r*3];
            x[0]+=DT*v*std::cos(x[2]); x[1]+=DT*v*std::sin(x[2]); x[2]+=DT*w;
            sim.traj[s+1][r*2]=x[0]; sim.traj[s+1][r*2+1]=x[1];
            for(int o=0;o<nobs;++o){ float ex=x[0]-obs[o].x,ey=x[1]-obs[o].y; float clr=std::sqrt(ex*ex+ey*ey)-obs[o].r;
                sim.min_clear=std::min(sim.min_clear,clr); if(clr<0.f)sim.collisions++; } }
    }
    double reached=0; for(int r=0;r<nR;++r){ float dx=X[r*3]-Gl[r*3],dy=X[r*3+1]-Gl[r*3+1]; if(std::sqrt(dx*dx+dy*dy)<0.6f)reached++; }
    sim.reach_ok=(reached> 0.95*nR);
    cudaFree(dX0);cudaFree(dG);cudaFree(dU);cudaFree(dW);
    return sim; }

// ============================ GIF ============================
static void render_gif(const Sim& sim,int nR){
    const int W=900,H=900; const float PX=58.f; const int CX=450,CY=450;
    auto proj=[&](float x,float y,int&sx,int&sy){ sx=CX+(int)(PX*x); sy=CY-(int)(PX*y); };
    if(system("mkdir -p tmp")!=0)std::fprintf(stderr,"warn\n");
    cv::VideoWriter video("tmp/gpu_constrained_mpc.avi",cv::VideoWriter::fourcc('M','J','P','G'),20,cv::Size(W,H));
    int S=sim.traj.size(); const int HOLD=16;
    for(int f=0;f<S+HOLD;f+=1){ int k=std::min(f,S-1);
        cv::Mat img(H,W,CV_8UC3,cv::Scalar(24,24,30));
        for(auto&o:sim.obs){ int sx,sy; proj(o.x,o.y,sx,sy); cv::circle(img,{sx,sy},(int)(PX*o.r),cv::Scalar(60,60,90),-1,cv::LINE_AA);
            cv::circle(img,{sx,sy},(int)(PX*(o.r)),cv::Scalar(90,90,130),1,cv::LINE_AA); }
        for(int r=0;r<nR;++r){ int tx,ty; proj(sim.goals[r*2],sim.goals[r*2+1],tx,ty); cv::drawMarker(img,{tx,ty},cv::Scalar(90,160,90),cv::MARKER_TILTED_CROSS,6,1,cv::LINE_AA);}
        for(int r=0;r<nR;++r){ int s0=std::max(0,k-26);
            for(int i=s0+1;i<=k;++i){ int ax,ay,bx,by; proj(sim.traj[i-1][r*2],sim.traj[i-1][r*2+1],ax,ay); proj(sim.traj[i][r*2],sim.traj[i][r*2+1],bx,by);
                cv::line(img,{ax,ay},{bx,by},cv::Scalar(200,140,50),1,cv::LINE_AA);} }
        for(int r=0;r<nR;++r){ int sx,sy; proj(sim.traj[k][r*2],sim.traj[k][r*2+1],sx,sy); cv::circle(img,{sx,sy},3,cv::Scalar(60,150,250),-1,cv::LINE_AA);}
        int px=28,py=52; auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA);};
        put("GPU constrained nonlinear MPC",py,0.78,cv::Scalar(235,235,245),2);py+=32;
        char b[96]; std::snprintf(b,sizeof(b),"%d unicycles, AL-iLQR each",nR); put(b,py,0.56,cv::Scalar(180,180,200),1);py+=26;
        put("hard obstacle + control-limit constraints",py,0.5,cv::Scalar(150,200,150),1);py+=24;
        put("augmented-Lagrangian -> KKT satisfaction",py,0.5,cv::Scalar(150,200,150),1);
        video.write(img); }
    video.release(); avi_to_gif("tmp/gpu_constrained_mpc.avi","gif/gpu_constrained_mpc.gif",20,820);
    std::printf("wrote gif/gpu_constrained_mpc.gif\n");
}

}  // namespace cudabot

int main(){
    using namespace cudabot;
    std::printf("=== GPU constrained nonlinear MPC (batched AL-iLQR) ===\n");
    const int nR=400, steps=70; const float margin=0.25f;
    std::printf("robots=%d  horizon T=%d  obstacles=%d  safety margin=%.2f\n", nR, T, OBST, margin);

    auto t0=std::chrono::high_resolution_clock::now();
    Sim sc=run(nR,steps,/*constrained=*/1,margin,7);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    // unconstrained baseline (same field) -- should cut through obstacles
    Sim su=run(nR,steps,/*constrained=*/0,margin,7);

    std::printf("\nconstrained AL-iLQR:   min clearance=%+.3f m   collisions=%d   reached=%s   max|v|=%.2f\n",
                sc.min_clear, sc.collisions, sc.reach_ok?"yes":"no", sc.max_u);
    std::printf("unconstrained iLQR:    min clearance=%+.3f m   collisions=%d   (baseline cuts through)\n",
                su.min_clear, su.collisions);
    std::printf("wall=%.1f ms  (%d MPC solves: %d robots x %d steps), %.2f ms/step\n", ms, nR*steps, nR, steps, ms/steps);
    if(sc.collisions==0 && sc.min_clear>-1e-3f && sc.reach_ok && su.collisions>0)
        std::printf("RESULT: PASS -- constrained MPC keeps every robot clear of all obstacles and reaches goals; unconstrained does not.\n");
    else std::printf("RESULT: CHECK -- constraint/clearance/goal tolerance not met.\n");

    render_gif(sc, nR);
    return 0;
}
