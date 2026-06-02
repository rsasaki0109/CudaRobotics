// gpu_robust_p2plane_reg.cu
//
// Flagship probabilistic registrant: robust (Student's-t) point-to-PLANE EM.
//
// The registration line built two orthogonal robustness tools and measured each
// in isolation:
//   * gpu_robust_treg      -- a Student's-t mixture (heavy tail) whose per-match
//                             weight u = (nu+D)/(nu + d^2/s^2) survives OUTLIERS
//                             where a Gaussian mixture is dragged off.
//   * gpu_filterreg_p2plane -- a point-to-PLANE residual n.(p-mu) that removes the
//                             O(s^2 * curvature) soft-mean bias which corrupts a
//                             point-to-point fit at any non-tiny kernel width.
//
// A real scan is BOTH outlier-ridden AND curved, so neither tool alone suffices.
// This demo fuses them into one EM: a Student's-t correspondence (heavy tail) AND
// a point-to-plane M-step (target normals, filtered through the same responsibility
// weights).  The result is a single registrant that is robust on both axes at once.
//
// The point is made with a 2x2 ABLATION on a deliberately combined-hard scene
// (curved surface, fixed MODERATE kernel width so curvature bias bites, 25% gross
// outliers, partial overlap).  Vary only two switches:
//
//       kernel:  Gaussian   vs  Student's-t        (outlier axis)
//       metric:  point-to-point vs point-to-plane  (curvature axis)
//
//   Gaussian + point   -> hurt by BOTH outliers and curvature  (worst)
//   t        + point   -> outliers handled, curvature bias remains
//   Gaussian + plane   -> curvature handled, outliers drag it
//   t        + plane   -> both handled                          (flagship, best)
//
// Each correspondence weight is P_mn * u_mn (responsibility * t-scale); the plane
// M-step projects the residual on the filtered target normal.  Normals on the
// target are estimated once by GPU kNN-PCA.  Verified: the flagship recovers the
// known SE(3) under the combined stress and beats every ablation; then a GIF.
//
// Build: CMakeLists, --expt-relaxed-constexpr.

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
static inline void pose_apply(const Pose& T,const float* y,float* p){ mat3_vec(T.R,y,p); p[0]+=T.t[0];p[1]+=T.t[1];p[2]+=T.t[2]; }
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

// ============================ kNN-PCA normals (target) ============================
__device__ static void sym3_smallest_evec(const float C[6], float n[3]){
    float c00=C[0],c01=C[1],c02=C[2],c11=C[3],c12=C[4],c22=C[5];
    float p1=c01*c01+c02*c02+c12*c12;
    if(p1<1e-20f){ n[0]=0;n[1]=0;n[2]=1; return; }
    float q=(c00+c11+c22)/3.f; float b00=c00-q,b11=c11-q,b22=c22-q;
    float p2=b00*b00+b11*b11+b22*b22+2.f*p1; float p=sqrtf(p2/6.f); float i_p=1.f/p;
    float d00=b00*i_p,d01=c01*i_p,d02=c02*i_p,d11=b11*i_p,d12=c12*i_p,d22=b22*i_p;
    float detB=d00*(d11*d22-d12*d12)-d01*(d01*d22-d12*d02)+d02*(d01*d12-d11*d02);
    float r=detB*0.5f; r=fminf(1.f,fmaxf(-1.f,r)); float phi=acosf(r)/3.f;
    float e0=q+2.f*p*cosf(phi+2.0943951f);
    float r0[3]={c00-e0,c01,c02},r1[3]={c01,c11-e0,c12},r2[3]={c02,c12,c22-e0};
    float x0[3]={r0[1]*r1[2]-r0[2]*r1[1], r0[2]*r1[0]-r0[0]*r1[2], r0[0]*r1[1]-r0[1]*r1[0]};
    float x1[3]={r0[1]*r2[2]-r0[2]*r2[1], r0[2]*r2[0]-r0[0]*r2[2], r0[0]*r2[1]-r0[1]*r2[0]};
    float x2[3]={r1[1]*r2[2]-r1[2]*r2[1], r1[2]*r2[0]-r1[0]*r2[2], r1[0]*r2[1]-r1[1]*r2[0]};
    float n0=x0[0]*x0[0]+x0[1]*x0[1]+x0[2]*x0[2],n1=x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2],n2=x2[0]*x2[0]+x2[1]*x2[1]+x2[2]*x2[2];
    const float* best=x0; float bn=n0; if(n1>bn){best=x1;bn=n1;} if(n2>bn){best=x2;bn=n2;}
    float inv=rsqrtf(bn+1e-20f); n[0]=best[0]*inv;n[1]=best[1]*inv;n[2]=best[2]*inv; }
__global__ void knn_normal_kernel(const float* __restrict__ X,int N,int K,float cx,float cy,float cz,float* __restrict__ NX){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
    float xi=X[i*3],yi=X[i*3+1],zi=X[i*3+2];
    const int KMAX=20; float dk[KMAX]; int ik[KMAX]; int kk=K<KMAX?K:KMAX;
    for(int a=0;a<kk;++a){dk[a]=1e30f;ik[a]=-1;}
    for(int j=0;j<N;++j){ if(j==i)continue; float dx=X[j*3]-xi,dy=X[j*3+1]-yi,dz=X[j*3+2]-zi; float d=dx*dx+dy*dy+dz*dz;
        if(d<dk[kk-1]){int p=kk-1; while(p>0&&dk[p-1]>d){dk[p]=dk[p-1];ik[p]=ik[p-1];--p;} dk[p]=d;ik[p]=j;} }
    float mx=xi,my=yi,mz=zi; int cnt=1; for(int a=0;a<kk;++a){int j=ik[a];if(j<0)continue;mx+=X[j*3];my+=X[j*3+1];mz+=X[j*3+2];++cnt;}
    float inv=1.f/cnt; mx*=inv;my*=inv;mz*=inv; float C[6]={0,0,0,0,0,0};
    auto acc=[&](float px,float py,float pz){float ex=px-mx,ey=py-my,ez=pz-mz;C[0]+=ex*ex;C[1]+=ex*ey;C[2]+=ex*ez;C[3]+=ey*ey;C[4]+=ey*ez;C[5]+=ez*ez;};
    acc(xi,yi,zi); for(int a=0;a<kk;++a){int j=ik[a];if(j<0)continue;acc(X[j*3],X[j*3+1],X[j*3+2]);}
    float nrm[3]; sym3_smallest_evec(C,nrm);
    float ox=xi-cx,oy=yi-cy,oz=zi-cz; if(nrm[0]*ox+nrm[1]*oy+nrm[2]*oz<0){nrm[0]=-nrm[0];nrm[1]=-nrm[1];nrm[2]=-nrm[2];}
    NX[i*3]=nrm[0];NX[i*3+1]=nrm[1];NX[i*3+2]=nrm[2]; }

// ============================ TMM E-step (with normal accumulation) ============================
__device__ __forceinline__ float comp_K(float d2,float s2,float nu,int gaussian){
    if(gaussian) return __expf(-d2/(2.f*s2));
    return __powf(1.f+d2/(nu*s2), -0.5f*(nu+3.f)); }
__global__ void estep_denom_kernel(const float* __restrict__ P,int M,const float* __restrict__ X,int N,
                                   float s2,float nu,float c_out,int gaussian,float* __restrict__ Dn){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N)return;
    float x0=X[n*3],x1=X[n*3+1],x2=X[n*3+2],s=0;
    for(int m=0;m<M;++m){ float dx=P[m*3]-x0,dy=P[m*3+1]-x1,dz=P[m*3+2]-x2; s+=comp_K(dx*dx+dy*dy+dz*dz,s2,nu,gaussian); }
    Dn[n]=s+c_out; }
// per model m: mu (t-weighted target mean) AND filtered target normal.
__global__ void estep_moments_kernel(const float* __restrict__ P,int M,const float* __restrict__ X,
                                     const float* __restrict__ NX,int N,const float* __restrict__ Dn,
                                     float s2,float nu,int gaussian,
                                     float* __restrict__ MU,float* __restrict__ NRM,float* __restrict__ Wm){
    int m=blockIdx.x*blockDim.x+threadIdx.x; if(m>=M)return;
    float p0=P[m*3],p1=P[m*3+1],p2=P[m*3+2];
    float sw=0,sx=0,sy=0,sz=0,nx=0,ny=0,nz=0;
    for(int n=0;n<N;++n){ float dx=p0-X[n*3],dy=p1-X[n*3+1],dz=p2-X[n*3+2]; float d2=dx*dx+dy*dy+dz*dz;
        float pmn=comp_K(d2,s2,nu,gaussian)/Dn[n]; float u=gaussian?1.f:(nu+3.f)/(nu+d2/s2); float w=pmn*u;
        sw+=w; sx+=w*X[n*3]; sy+=w*X[n*3+1]; sz+=w*X[n*3+2]; nx+=w*NX[n*3]; ny+=w*NX[n*3+1]; nz+=w*NX[n*3+2]; }
    Wm[m]=sw; float inv=1.f/(sw+1e-20f);
    MU[m*3]=sx*inv;MU[m*3+1]=sy*inv;MU[m*3+2]=sz*inv;
    float nn=rsqrtf(nx*nx+ny*ny+nz*nz+1e-20f); NRM[m*3]=nx*nn;NRM[m*3+1]=ny*nn;NRM[m*3+2]=nz*nn; }

__global__ void transform_kernel(const float* __restrict__ Y,int M,const float* __restrict__ R,const float* __restrict__ t,float* __restrict__ P){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return; float y0=Y[j*3],y1=Y[j*3+1],y2=Y[j*3+2];
    P[j*3]=R[0]*y0+R[1]*y1+R[2]*y2+t[0]; P[j*3+1]=R[3]*y0+R[4]*y1+R[5]*y2+t[1]; P[j*3+2]=R[6]*y0+R[7]*y1+R[8]*y2+t[2]; }

// M-step: point-to-point (plane==0) or point-to-plane (plane==1), weight Wm.
__global__ void mstep_kernel(const float* __restrict__ P,const float* __restrict__ MU,const float* __restrict__ NRM,
                             const float* __restrict__ W,int M,int plane,float* __restrict__ Hg){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return; float w=W[j]; if(w<1e-12f)return;
    float px=P[j*3],py=P[j*3+1],pz=P[j*3+2];
    float ex=px-MU[j*3],ey=py-MU[j*3+1],ez=pz-MU[j*3+2];
    float J[18]={1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    if(plane){
        float nx=NRM[j*3],ny=NRM[j*3+1],nz=NRM[j*3+2]; float rs=nx*ex+ny*ey+nz*ez;
        float jp[6]; for(int a=0;a<6;++a)jp[a]=nx*J[a]+ny*J[6+a]+nz*J[12+a];
        float Hl[21]; int c=0; for(int a=0;a<6;++a)for(int b=a;b<6;++b)Hl[c++]=w*jp[a]*jp[b];
        for(int k=0;k<21;++k)atomicAdd(&Hg[k],Hl[k]);
        for(int a=0;a<6;++a)atomicAdd(&Hg[21+a],w*jp[a]*rs);
        atomicAdd(&Hg[27],w*rs*rs); atomicAdd(&Hg[28],w);
    } else {
        float Hl[21]; int c=0; for(int a=0;a<6;++a)for(int b=a;b<6;++b){float s=J[a]*J[b]+J[6+a]*J[6+b]+J[12+a]*J[12+b];Hl[c++]=w*s;}
        float gl[6]; for(int a=0;a<6;++a)gl[a]=w*(J[a]*ex+J[6+a]*ey+J[12+a]*ez);
        for(int k=0;k<21;++k)atomicAdd(&Hg[k],Hl[k]);
        for(int k=0;k<6;++k)atomicAdd(&Hg[21+k],gl[k]);
        atomicAdd(&Hg[27],w*(ex*ex+ey*ey+ez*ez)); atomicAdd(&Hg[28],w);
    } }

static bool solve6(const float* Hut,const float* g,float* d){
    float H[36]; int c=0; for(int a=0;a<6;++a)for(int b=a;b<6;++b){H[a*6+b]=H[b*6+a]=Hut[c++];}
    for(int i=0;i<6;++i)H[i*6+i]+=1e-6f; float L[36]={0};
    for(int i=0;i<6;++i)for(int j=0;j<=i;++j){float s=H[i*6+j];for(int k=0;k<j;++k)s-=L[i*6+k]*L[j*6+k];
        if(i==j){if(s<=0)return false;L[i*6+i]=std::sqrt(s);}else L[i*6+j]=s/L[j*6+j];}
    float y[6]; for(int i=0;i<6;++i){float s=-g[i];for(int k=0;k<i;++k)s-=L[i*6+k]*y[k];y[i]=s/L[i*6+i];}
    for(int i=5;i>=0;--i){float s=y[i];for(int k=i+1;k<6;++k)s-=L[k*6+i]*d[k];d[i]=s/L[i*6+i];}
    return true; }

// ============================ registrant (gaussian/plane switches) ============================
struct RResult { Pose T; int iters; };
// fixed MODERATE sigma (no fine annealing) so the curvature bias is exposed; the
// kernel (gaussian) and metric (plane) switches are the only differences.
static RResult reg(const std::vector<float>& X,const std::vector<float>& NX,const std::vector<float>& Y,
                   Pose T0,float nu,int gaussian,int plane,std::vector<Pose>* traj=nullptr){
    int N=X.size()/3, M=Y.size()/3;
    float *dX,*dNX,*dY,*dP,*dR,*dt,*dDn,*dMU,*dNRM,*dWm,*dHg;
    CUDA_CHECK(cudaMalloc(&dX,N*3*sizeof(float)));CUDA_CHECK(cudaMemcpy(dX,X.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dNX,N*3*sizeof(float)));CUDA_CHECK(cudaMemcpy(dNX,NX.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dY,M*3*sizeof(float)));CUDA_CHECK(cudaMemcpy(dY,Y.data(),M*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP,M*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dR,9*sizeof(float)));CUDA_CHECK(cudaMalloc(&dt,3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDn,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dMU,M*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dNRM,M*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dWm,M*sizeof(float)));CUDA_CHECK(cudaMalloc(&dHg,29*sizeof(float)));
    Pose T=T0; RResult res; res.iters=0; if(traj)traj->push_back(T);
    // short anneal lead-in then HOLD a COARSE floor: at a coarse kernel width the
    // soft-mean curvature bias is large (point-to-point carries several degrees of
    // it), so the point-to-plane metric has room to matter -- one of the two axes
    // this ablation needs to be live.
    const float fl=0.45f;                            // coarse floor: curvature bias is large here
    std::vector<float> sigmas={0.8f,0.6f,0.5f}; for(int i=0;i<6;++i)sigmas.push_back(fl);
    for(float sig:sigmas){ float s2=sig*sig; float c_out=0.05f;
        for(int outer=0;outer<6;++outer){
            CUDA_CHECK(cudaMemcpy(dR,T.R.m,9*sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt,T.t,3*sizeof(float),cudaMemcpyHostToDevice));
            transform_kernel<<<(M+255)/256,256>>>(dY,M,dR,dt,dP);
            estep_denom_kernel<<<(N+255)/256,256>>>(dP,M,dX,N,s2,nu,c_out,gaussian,dDn);
            estep_moments_kernel<<<(M+255)/256,256>>>(dP,M,dX,dNX,N,dDn,s2,nu,gaussian,dMU,dNRM,dWm);
            for(int gn=0;gn<3;++gn){
                if(gn>0){ CUDA_CHECK(cudaMemcpy(dR,T.R.m,9*sizeof(float),cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(dt,T.t,3*sizeof(float),cudaMemcpyHostToDevice));
                    transform_kernel<<<(M+255)/256,256>>>(dY,M,dR,dt,dP); }
                CUDA_CHECK(cudaMemset(dHg,0,29*sizeof(float)));
                mstep_kernel<<<(M+255)/256,256>>>(dP,dMU,dNRM,dWm,M,plane,dHg);
                float Hg[29]; CUDA_CHECK(cudaMemcpy(Hg,dHg,29*sizeof(float),cudaMemcpyDeviceToHost));
                float d[6]; if(!solve6(Hg,Hg+21,d))break; T=pose_mul(se3_exp(d),T);
            }
            ++res.iters; if(traj)traj->push_back(T);
        } }
    res.T=T; cudaFree(dX);cudaFree(dNX);cudaFree(dY);cudaFree(dP);cudaFree(dR);cudaFree(dt);cudaFree(dDn);cudaFree(dMU);cudaFree(dNRM);cudaFree(dWm);cudaFree(dHg);
    return res; }

static void errs(const Pose& res,const Mat3& Rgt,const float* gt_t,float& ang,float& terr){
    Mat3 RgtT; for(int i=0;i<3;++i)for(int j=0;j<3;++j)RgtT.m[i*3+j]=Rgt.m[j*3+i];
    float texp[3]; mat3_vec(RgtT,gt_t,texp); for(int k=0;k<3;++k)texp[k]=-texp[k];
    Mat3 Rerr=mat3_mul(Rgt,res.R); float tr=Rerr.m[0]+Rerr.m[4]+Rerr.m[8];
    ang=std::acos(std::min(1.f,std::max(-1.f,(tr-1.f)*0.5f)));
    terr=0; for(int k=0;k<3;++k){float e=res.t[k]-texp[k];terr+=e*e;} terr=std::sqrt(terr); }

// ============================ GIF (flagship convergence) ============================
static void render_gif(const std::vector<float>& X,const std::vector<char>& is_out,const std::vector<float>& Y,const std::vector<Pose>& traj){
    const int W=1280,H=720,CX=380,CY=360; const float SCALE=80.f,elev=0.42f;
    if(system("mkdir -p tmp")!=0)std::fprintf(stderr,"warn\n");
    cv::VideoWriter video("tmp/gpu_robust_p2plane_reg.avi",cv::VideoWriter::fourcc('M','J','P','G'),18,cv::Size(W,H));
    int nt=traj.size(),Nx=X.size()/3,My=Y.size()/3; const int HOLD=24;
    struct Sp{float sx,sy,d;cv::Scalar c;};
    for(int f=0;f<nt+HOLD;++f){ int k=std::min(f,nt-1); float az=0.5f+f*0.02f,ca=std::cos(az),sa=std::sin(az),ce=std::cos(elev),se=std::sin(elev);
        cv::Mat img(H,W,CV_8UC3,cv::Scalar(26,26,32)); const Pose&T=traj[k];
        auto proj=[&](float x,float y,float z,float&sx,float&sy,float&d){float x1=x*ca-y*sa,y1=x*sa+y*ca,z1=z;sx=CX+SCALE*x1;sy=CY-SCALE*(z1*ce-y1*se);d=y1*ce+z1*se;};
        std::vector<Sp> sp;
        for(int i=0;i<Nx;i+=2){Sp s;proj(X[i*3],X[i*3+1],X[i*3+2],s.sx,s.sy,s.d); s.c=is_out[i]?cv::Scalar(110,110,120):cv::Scalar(210,180,60); sp.push_back(s);}
        for(int i=0;i<My;i+=2){float y0[3]={Y[i*3],Y[i*3+1],Y[i*3+2]},p[3];pose_apply(T,y0,p);Sp s;proj(p[0],p[1],p[2],s.sx,s.sy,s.d);s.c=cv::Scalar(40,130,240);sp.push_back(s);}
        std::sort(sp.begin(),sp.end(),[](const Sp&a,const Sp&b){return a.d<b.d;});
        float dmin=1e9f,dmax=-1e9f;for(auto&s:sp){dmin=std::min(dmin,s.d);dmax=std::max(dmax,s.d);}
        for(auto&s:sp){float t=(s.d-dmin)/(dmax-dmin+1e-6f);float b=0.45f+0.55f*t;cv::circle(img,cv::Point((int)s.sx,(int)s.sy),2,s.c*b,-1,cv::LINE_AA);}
        int px=800,py=70; auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA);};
        put("GPU robust point-to-plane",py,0.85,cv::Scalar(235,235,245),2);py+=36;
        put("Student's-t  x  point-to-plane",py,0.58,cv::Scalar(180,180,200),1);py+=48;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(210,180,60),-1);cv::putText(img,"target surface",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=30;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(110,110,120),-1);cv::putText(img,"25% outliers",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=30;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(40,130,240),-1);cv::putText(img,"source (aligning)",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=50;
        put("heavy tail rejects outliers",py,0.5,cv::Scalar(150,200,150),1);py+=24;
        put("plane residual kills curvature bias",py,0.5,cv::Scalar(150,200,150),1);py+=42;
        if(f>=nt+HOLD-HOLD && f>=nt-1)put("ALIGNED",py,0.8,cv::Scalar(120,230,250),2);
        video.write(img); }
    video.release(); avi_to_gif("tmp/gpu_robust_p2plane_reg.avi","gif/gpu_robust_p2plane_reg.gif",18,900);
    std::printf("wrote gif/gpu_robust_p2plane_reg.gif\n");
}

}  // namespace cudabot

int main(){
    using namespace cudabot;
    std::printf("=== GPU flagship registrant: robust (Student's-t) point-to-plane EM ===\n");
    const int Nc=2600;
    std::vector<float> Xc=make_lumpy(Nc,1);
    float gt_w[3]={0.22f,-0.30f,0.18f}, gt_t[3]={0.6f,-0.45f,0.35f};
    Mat3 Rgt=so3_exp(gt_w); Pose Tgt; Tgt.R=Rgt; for(int k=0;k<3;++k)Tgt.t[k]=gt_t[k];
    Pose T0; T0.R={1,0,0,0,1,0,0,0,1}; T0.t[0]=T0.t[1]=T0.t[2]=0;

    // target X = clean surface + 25% uniform gross outliers
    std::mt19937 rng(7);
    std::vector<float> X; std::vector<char> is_out;
    for(int i=0;i<Nc;++i){ X.push_back(Xc[i*3]);X.push_back(Xc[i*3+1]);X.push_back(Xc[i*3+2]); is_out.push_back(0); }
    int n_out=(int)std::lround(0.25f*Nc/0.75f);
    std::uniform_real_distribution<float> box(-3.2f,3.2f);
    for(int i=0;i<n_out;++i){ X.push_back(box(rng));X.push_back(box(rng));X.push_back(box(rng)); is_out.push_back(1); }
    int N=X.size()/3;
    // source Y = transformed clean subset (70% overlap), noised
    std::normal_distribution<float> noise(0.f,0.02f); std::uniform_real_distribution<float> keep(0,1);
    std::vector<float> Y;
    for(int i=0;i<Nc;++i){ if(keep(rng)>0.70f)continue; float y[3]={Xc[i*3],Xc[i*3+1],Xc[i*3+2]},p[3]; pose_apply(Tgt,y,p);
        Y.push_back(p[0]+noise(rng));Y.push_back(p[1]+noise(rng));Y.push_back(p[2]+noise(rng)); }
    int M=Y.size()/3;
    std::printf("target N=%d (clean %d + %d outliers=25%%)   source M=%d (70%% overlap)\n", N, Nc, n_out, M);

    // target normals (kNN-PCA), once
    std::vector<float> NX(N*3);
    { float c[3]={0,0,0}; for(int i=0;i<N;++i)for(int k=0;k<3;++k)c[k]+=X[i*3+k]; for(int k=0;k<3;++k)c[k]/=N;
      float*dX,*dNX; CUDA_CHECK(cudaMalloc(&dX,N*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dNX,N*3*sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dX,X.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
      knn_normal_kernel<<<(N+127)/128,128>>>(dX,N,14,c[0],c[1],c[2],dNX);
      CUDA_CHECK(cudaMemcpy(NX.data(),dNX,N*3*sizeof(float),cudaMemcpyDeviceToHost)); cudaFree(dX);cudaFree(dNX); }

    // 2x2 ablation: kernel (Gaussian/Student-t) x metric (point/plane)
    std::printf("\n2x2 ablation on the combined-hard scene (25%% outliers x curvature x partial):\n");
    std::printf("  %-22s | rot err (deg) | trans err\n","variant");
    std::printf("  -----------------------+---------------+----------\n");
    struct V{const char*name;int gaussian,plane;}; V vs[]={
        {"Gaussian + point",1,0},{"Student-t + point",0,0},{"Gaussian + plane",1,1},{"Student-t + plane*",0,1}};
    float flag_ang=9,flag_terr=9; float best_other_ang=9,best_other_terr=9;
    std::vector<Pose> traj;
    for(int v=0;v<4;++v){ bool flagship=(v==3);
        RResult r=reg(X,NX,Y,T0,3.0f,vs[v].gaussian,vs[v].plane, flagship?&traj:nullptr);
        float a,t; errs(r.T,Rgt,gt_t,a,t);
        std::printf("  %-22s | %9.2f     | %.4f\n", vs[v].name, a*57.2958f, t);
        if(flagship){flag_ang=a;flag_terr=t;} else {best_other_ang=std::min(best_other_ang,a);best_other_terr=std::min(best_other_terr,t);}
    }
    std::printf("  (* = flagship: Student's-t heavy tail + point-to-plane residual)\n");

    bool ok=(flag_ang<0.03f && flag_terr<0.06f);
    bool wins=(flag_ang<best_other_ang && flag_terr<best_other_terr && best_other_ang>1.25f*flag_ang);
    if(ok&&wins) std::printf("RESULT: PASS -- flagship is best on BOTH axes and recovers under combined stress (outliers x curvature x partial).\n");
    else if(ok)  std::printf("RESULT: PARTIAL -- flagship recovered, but margin over ablations is small.\n");
    else         std::printf("RESULT: CHECK -- flagship not within tolerance.\n");

    render_gif(X, is_out, Y, traj);
    return 0;
}
