// gpu_filterreg_p2plane.cu
//
// GPU FilterReg, point-to-PLANE: the declared "future work" of gpu_filterreg.cu.
//
// The point-to-point FilterReg (Gao & Tedrake, CVPR 2019) replaces the dense
// EM responsibility matrix with a Gaussian FILTER, and fits the rigid motion to
// the filtered correspondence  mu_j = M1(p_j) / M0(p_j)  by a twist Gauss-Newton
// step on the point-to-point residual  r_j = p_j - mu_j.  That filtered mean has
// a known, systematic flaw, noted verbatim in gpu_filterreg.cu:
//
//     "point-to-plane would also remove the residual O(sigma^2 * curvature)
//      soft-mean bias."
//
// The Gaussian-blurred correspondence mu_j is the local DENSITY mean of the
// target surface seen through a sigma-wide kernel.  On a curved surface that mean
// sits slightly INSIDE the surface, toward the centre of curvature, by O(sigma^2
// * kappa).  The point-to-point residual trusts that biased mean in full, so the
// pose carries a curvature- and sigma-dependent error that only vanishes as
// sigma -> 0 (i.e. only with aggressive, slow annealing).
//
// Point-to-plane fixes it.  If n_j is the surface NORMAL at the correspondence,
// the soft-mean bias is almost entirely TANGENTIAL (it slides the mean along the
// surface / toward curvature centre); projecting the residual onto the normal,
//     r_j = n_j . (p_j - mu_j)   (a scalar),
// discards the tangential component and keeps only the geometrically meaningful
// surface-to-surface distance.  So point-to-plane reaches the SAME accuracy at a
// much COARSER sigma -- fewer/cheaper iterations, more robust.
//
// We get the normal "for free" from the same voxel filter: alongside the a_n and
// a_n*x_n channels we splat a_n*nx_n (the observation normals, kNN-PCA estimated
// once on the fixed cloud), blur, and SLICE at the model points -> a filtered
// normal field N1_j.  The plane M-step is then a rank-1 normal-equation update.
//
// Headline experiment: a sigma-FLOOR sweep.  Both metrics run the identical
// filter pipeline; we stop annealing at a floor sigma and read the final pose
// error.  Point-to-plane holds low error at coarse floors where point-to-point's
// curvature bias is still large -- the bias the original file flagged, measured.
//
// Build: CMakeLists, --expt-relaxed-constexpr.

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ============================ SE(3)/SO(3) helpers (host) ============================
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

// ============================ procedural cloud (lumpy closed surface) ============================
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

// ============================ GPU kNN-PCA normal estimation ============================
// smallest-eigenvector of a symmetric 3x3 covariance via Cardano eigenvalues +
// null-space of (C - lambda_min I) from a robust cross-product of its rows.
__device__ static void sym3_smallest_evec(const float C[6], float n[3]){
    // C = [c00 c01 c02 c11 c12 c22]
    float c00=C[0],c01=C[1],c02=C[2],c11=C[3],c12=C[4],c22=C[5];
    float p1=c01*c01+c02*c02+c12*c12;
    if(p1<1e-20f){ n[0]=0;n[1]=0;n[2]=1; return; }       // already diagonal
    float q=(c00+c11+c22)/3.f;
    float b00=c00-q,b11=c11-q,b22=c22-q;
    float p2=b00*b00+b11*b11+b22*b22+2.f*p1; float p=sqrtf(p2/6.f);
    // det(B/p)/2
    float i_p=1.f/p;
    float d00=b00*i_p,d01=c01*i_p,d02=c02*i_p,d11=b11*i_p,d12=c12*i_p,d22=b22*i_p;
    float detB=d00*(d11*d22-d12*d12)-d01*(d01*d22-d12*d02)+d02*(d01*d12-d11*d02);
    float r=detB*0.5f; r=fminf(1.f,fmaxf(-1.f,r));
    float phi=acosf(r)/3.f;
    float e0=q+2.f*p*cosf(phi+2.0943951f);               // smallest eigenvalue
    // null space of (C - e0 I): cross product of two independent rows
    float a00=c00-e0,a11=c11-e0,a22=c22-e0;
    float r0[3]={a00,c01,c02},r1[3]={c01,a11,c12},r2[3]={c02,c12,a22};
    float x0[3]={r0[1]*r1[2]-r0[2]*r1[1], r0[2]*r1[0]-r0[0]*r1[2], r0[0]*r1[1]-r0[1]*r1[0]};
    float x1[3]={r0[1]*r2[2]-r0[2]*r2[1], r0[2]*r2[0]-r0[0]*r2[2], r0[0]*r2[1]-r0[1]*r2[0]};
    float x2[3]={r1[1]*r2[2]-r1[2]*r2[1], r1[2]*r2[0]-r1[0]*r2[2], r1[0]*r2[1]-r1[1]*r2[0]};
    float n0=x0[0]*x0[0]+x0[1]*x0[1]+x0[2]*x0[2];
    float n1=x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2];
    float n2=x2[0]*x2[0]+x2[1]*x2[1]+x2[2]*x2[2];
    const float* best=x0; float bn=n0;
    if(n1>bn){best=x1;bn=n1;} if(n2>bn){best=x2;bn=n2;}
    float inv=rsqrtf(bn+1e-20f); n[0]=best[0]*inv;n[1]=best[1]*inv;n[2]=best[2]*inv;
}
// one thread per point: find K nearest (brute force), accumulate covariance, normal.
// orient outward (away from cloud centroid c).
__global__ void knn_normal_kernel(const float* __restrict__ X, int N, int K,
                                  float cx, float cy, float cz, float* __restrict__ NX){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    float xi=X[i*3],yi=X[i*3+1],zi=X[i*3+2];
    const int KMAX=24; float dk[KMAX]; int ik[KMAX]; int kk=K<KMAX?K:KMAX;
    for(int a=0;a<kk;++a){dk[a]=1e30f;ik[a]=-1;}
    for(int j=0;j<N;++j){ if(j==i)continue;
        float dx=X[j*3]-xi,dy=X[j*3+1]-yi,dz=X[j*3+2]-zi; float d=dx*dx+dy*dy+dz*dz;
        if(d<dk[kk-1]){ int p=kk-1; while(p>0&&dk[p-1]>d){dk[p]=dk[p-1];ik[p]=ik[p-1];--p;} dk[p]=d;ik[p]=j; } }
    float mx=xi,my=yi,mz=zi; int cnt=1;
    for(int a=0;a<kk;++a){ int j=ik[a]; if(j<0)continue; mx+=X[j*3];my+=X[j*3+1];mz+=X[j*3+2];++cnt; }
    float inv=1.f/cnt; mx*=inv;my*=inv;mz*=inv;
    float C[6]={0,0,0,0,0,0};
    auto acc=[&](float px,float py,float pz){float ex=px-mx,ey=py-my,ez=pz-mz;
        C[0]+=ex*ex;C[1]+=ex*ey;C[2]+=ex*ez;C[3]+=ey*ey;C[4]+=ey*ez;C[5]+=ez*ez;};
    acc(xi,yi,zi); for(int a=0;a<kk;++a){int j=ik[a]; if(j<0)continue; acc(X[j*3],X[j*3+1],X[j*3+2]);}
    float n[3]; sym3_smallest_evec(C,n);
    // orient outward from the cloud centroid
    float ox=xi-cx,oy=yi-cy,oz=zi-cz;
    if(n[0]*ox+n[1]*oy+n[2]*oz<0){n[0]=-n[0];n[1]=-n[1];n[2]=-n[2];}
    NX[i*3]=n[0];NX[i*3+1]=n[1];NX[i*3+2]=n[2];
}

// ============================ voxel Gaussian filter ============================
struct Grid { float ox, oy, oz; float inv_h; int nx, ny, nz; };
__host__ __device__ static inline int grid_idx(const Grid& g,int ix,int iy,int iz){ return (iz*g.ny+iy)*g.nx+ix; }

__global__ void transform_kernel(const float* __restrict__ Y,int M,const float* __restrict__ R,const float* __restrict__ t,float* __restrict__ P){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return;
    float y0=Y[j*3],y1=Y[j*3+1],y2=Y[j*3+2];
    P[j*3]=R[0]*y0+R[1]*y1+R[2]*y2+t[0]; P[j*3+1]=R[3]*y0+R[4]*y1+R[5]*y2+t[1]; P[j*3+2]=R[6]*y0+R[7]*y1+R[8]*y2+t[2]; }

// weighted splat with an OPTIONAL extra 3-vector channel E (e.g. a_n * normal).
__global__ void splat_kernel(const float* __restrict__ P,const float* __restrict__ A,
                             const float* __restrict__ E,int N,Grid g,
                             float* __restrict__ m0,float* __restrict__ m1,float* __restrict__ me){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
    float x=P[i*3],y=P[i*3+1],z=P[i*3+2];
    int ix=(int)floorf((x-g.ox)*g.inv_h),iy=(int)floorf((y-g.oy)*g.inv_h),iz=(int)floorf((z-g.oz)*g.inv_h);
    if(ix<0||iy<0||iz<0||ix>=g.nx||iy>=g.ny||iz>=g.nz)return;
    int idx=grid_idx(g,ix,iy,iz); float a=A?A[i]:1.f;
    atomicAdd(&m0[idx],a);
    if(m1){atomicAdd(&m1[idx*3],a*x);atomicAdd(&m1[idx*3+1],a*y);atomicAdd(&m1[idx*3+2],a*z);}
    if(me&&E){atomicAdd(&me[idx*3],a*E[i*3]);atomicAdd(&me[idx*3+1],a*E[i*3+1]);atomicAdd(&me[idx*3+2],a*E[i*3+2]);}
}
__global__ void slice_scalar_kernel(const float* __restrict__ Q,int M,Grid g,const float* __restrict__ bm0,float* __restrict__ out){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return;
    float px=Q[j*3],py=Q[j*3+1],pz=Q[j*3+2];
    float fx=(px-g.ox)*g.inv_h-0.5f,fy=(py-g.oy)*g.inv_h-0.5f,fz=(pz-g.oz)*g.inv_h-0.5f;
    int ix=(int)floorf(fx),iy=(int)floorf(fy),iz=(int)floorf(fz);
    float tx=fx-ix,ty=fy-iy,tz=fz-iz,a=0;
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int jx=ix+dx,jy=iy+dy,jz=iz+dz;
        if(jx<0||jy<0||jz<0||jx>=g.nx||jy>=g.ny||jz>=g.nz)continue;
        a+=(dx?tx:1-tx)*(dy?ty:1-ty)*(dz?tz:1-tz)*bm0[grid_idx(g,jx,jy,jz)];}
    out[j]=a; }
__global__ void compute_a_kernel(const float* __restrict__ Z,int N,float c_out,float* __restrict__ A){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N)return; A[n]=1.f/(Z[n]+c_out); }
__global__ void blur_axis_kernel(const float* __restrict__ in,float* __restrict__ out,Grid g,int axis,int R,const float* __restrict__ w,int comp){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; int total=g.nx*g.ny*g.nz; if(idx>=total)return;
    int ix=idx%g.nx,iy=(idx/g.nx)%g.ny,iz=idx/(g.nx*g.ny);
    for(int c=0;c<comp;++c){ float acc=0;
        for(int d=-R;d<=R;++d){int jx=ix,jy=iy,jz=iz; if(axis==0)jx+=d;else if(axis==1)jy+=d;else jz+=d;
            if(jx<0||jy<0||jz<0||jx>=g.nx||jy>=g.ny||jz>=g.nz)continue; acc+=w[d+R]*in[grid_idx(g,jx,jy,jz)*comp+c];}
        out[idx*comp+c]=acc; } }

// SLICE moments AND the filtered normal at the transformed model points.
__global__ void slice_kernel(const float* __restrict__ Y,int M,Grid g,const float* __restrict__ R,const float* __restrict__ t,
                             const float* __restrict__ bm0,const float* __restrict__ bm1,const float* __restrict__ bmn,
                             float* __restrict__ outM0,float* __restrict__ outM1,float* __restrict__ outN,float* __restrict__ outP){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return;
    float y0=Y[j*3],y1=Y[j*3+1],y2=Y[j*3+2];
    float px=R[0]*y0+R[1]*y1+R[2]*y2+t[0],py=R[3]*y0+R[4]*y1+R[5]*y2+t[1],pz=R[6]*y0+R[7]*y1+R[8]*y2+t[2];
    outP[j*3]=px;outP[j*3+1]=py;outP[j*3+2]=pz;
    float fx=(px-g.ox)*g.inv_h-0.5f,fy=(py-g.oy)*g.inv_h-0.5f,fz=(pz-g.oz)*g.inv_h-0.5f;
    int ix=(int)floorf(fx),iy=(int)floorf(fy),iz=(int)floorf(fz);
    float tx=fx-ix,ty=fy-iy,tz=fz-iz;
    float a0=0,a1x=0,a1y=0,a1z=0,anx=0,any=0,anz=0;
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int jx=ix+dx,jy=iy+dy,jz=iz+dz;
        if(jx<0||jy<0||jz<0||jx>=g.nx||jy>=g.ny||jz>=g.nz)continue;
        float wgt=(dx?tx:1-tx)*(dy?ty:1-ty)*(dz?tz:1-tz); int idx=grid_idx(g,jx,jy,jz);
        a0+=wgt*bm0[idx]; a1x+=wgt*bm1[idx*3];a1y+=wgt*bm1[idx*3+1];a1z+=wgt*bm1[idx*3+2];
        if(bmn){anx+=wgt*bmn[idx*3];any+=wgt*bmn[idx*3+1];anz+=wgt*bmn[idx*3+2];} }
    outM0[j]=a0; outM1[j*3]=a1x;outM1[j*3+1]=a1y;outM1[j*3+2]=a1z;
    if(outN){ float nn=rsqrtf(anx*anx+any*any+anz*anz+1e-20f); outN[j*3]=anx*nn;outN[j*3+1]=any*nn;outN[j*3+2]=anz*nn; }
}

// point-to-POINT twist GN accumulation (mu=M1/M0, weight=M0).
__global__ void mstep_point_kernel(const float* __restrict__ P,const float* __restrict__ M0,const float* __restrict__ M1,
                                   int M,float m0_floor,float* __restrict__ Hg){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return; float w=M0[j]; if(w<m0_floor)return;
    float px=P[j*3],py=P[j*3+1],pz=P[j*3+2];
    float rx=px-M1[j*3]/w,ry=py-M1[j*3+1]/w,rz=pz-M1[j*3+2]/w;
    float J[18]={1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float Hl[21]; int c=0;
    for(int a=0;a<6;++a)for(int b=a;b<6;++b){float s=J[a]*J[b]+J[6+a]*J[6+b]+J[12+a]*J[12+b];Hl[c++]=w*s;}
    float gl[6]; for(int a=0;a<6;++a)gl[a]=w*(J[a]*rx+J[6+a]*ry+J[12+a]*rz);
    for(int k=0;k<21;++k)atomicAdd(&Hg[k],Hl[k]);
    for(int k=0;k<6;++k)atomicAdd(&Hg[21+k],gl[k]);
    atomicAdd(&Hg[27],w*(rx*rx+ry*ry+rz*rz)); atomicAdd(&Hg[28],w); }

// point-to-PLANE twist GN accumulation: scalar residual rs = n.(p-mu),
//   jacobian row jp = n^T J  (1x6),  rank-1 update  H += w jp^T jp, g += w jp rs.
__global__ void mstep_plane_kernel(const float* __restrict__ P,const float* __restrict__ M0,const float* __restrict__ M1,
                                   const float* __restrict__ NN,int M,float m0_floor,float* __restrict__ Hg){
    int j=blockIdx.x*blockDim.x+threadIdx.x; if(j>=M)return; float w=M0[j]; if(w<m0_floor)return;
    float px=P[j*3],py=P[j*3+1],pz=P[j*3+2];
    float rx=px-M1[j*3]/w,ry=py-M1[j*3+1]/w,rz=pz-M1[j*3+2]/w;
    float nx=NN[j*3],ny=NN[j*3+1],nz=NN[j*3+2];
    float rs=nx*rx+ny*ry+nz*rz;                          // signed plane distance
    float J[18]={1,0,0,0,pz,-py, 0,1,0,-pz,0,px, 0,0,1,py,-px,0};
    float jp[6]; for(int a=0;a<6;++a)jp[a]=nx*J[a]+ny*J[6+a]+nz*J[12+a];
    float Hl[21]; int c=0; for(int a=0;a<6;++a)for(int b=a;b<6;++b)Hl[c++]=w*jp[a]*jp[b];
    for(int k=0;k<21;++k)atomicAdd(&Hg[k],Hl[k]);
    for(int a=0;a<6;++a)atomicAdd(&Hg[21+a],w*jp[a]*rs);
    atomicAdd(&Hg[27],w*rs*rs); atomicAdd(&Hg[28],w); }

static bool solve6(const float* Hut,const float* g,float* d){
    float H[36]; int c=0; for(int a=0;a<6;++a)for(int b=a;b<6;++b){H[a*6+b]=H[b*6+a]=Hut[c++];}
    for(int i=0;i<6;++i)H[i*6+i]+=1e-6f; float L[36]={0};
    for(int i=0;i<6;++i)for(int j=0;j<=i;++j){float s=H[i*6+j];for(int k=0;k<j;++k)s-=L[i*6+k]*L[j*6+k];
        if(i==j){if(s<=0)return false;L[i*6+i]=sqrtf(s);}else L[i*6+j]=s/L[j*6+j];}
    float y[6]; for(int i=0;i<6;++i){float s=-g[i];for(int k=0;k<i;++k)s-=L[i*6+k]*y[k];y[i]=s/L[i*6+i];}
    for(int i=5;i>=0;--i){float s=y[i];for(int k=i+1;k<6;++k)s-=L[k*6+i]*d[k];d[i]=s/L[i*6+i];}
    return true; }

// ============================ FilterReg driver (point or plane) ============================
struct FRResult { Pose T; int iters; };
// plane!=0 -> point-to-plane M-step.  sig_floor truncates the anneal at a floor.
static FRResult filterreg(const std::vector<float>& X, const std::vector<float>& dNXh,
                          const std::vector<float>& Y, Pose T0, int plane, float sig_floor,
                          std::vector<Pose>* traj=nullptr){
    int N=X.size()/3, M=Y.size()/3;
    float lo[3]={1e9f,1e9f,1e9f},hi[3]={-1e9f,-1e9f,-1e9f};
    for(int i=0;i<N;++i)for(int k=0;k<3;++k){lo[k]=std::min(lo[k],X[i*3+k]);hi[k]=std::max(hi[k],X[i*3+k]);}
    for(int k=0;k<3;++k){lo[k]-=2.0f;hi[k]+=2.0f;}
    float *dX,*dY,*dNX; CUDA_CHECK(cudaMalloc(&dX,N*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dY,M*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dNX,N*3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dX,X.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY,Y.data(),M*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNX,dNXh.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
    float *dR,*dt; CUDA_CHECK(cudaMalloc(&dR,9*sizeof(float)));CUDA_CHECK(cudaMalloc(&dt,3*sizeof(float)));
    float *dP,*dM0,*dM1,*dNm; CUDA_CHECK(cudaMalloc(&dP,M*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dM0,M*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dM1,M*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dNm,M*3*sizeof(float)));
    float *dZ,*dA,*dHg; CUDA_CHECK(cudaMalloc(&dZ,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dA,N*sizeof(float)));CUDA_CHECK(cudaMalloc(&dHg,29*sizeof(float)));
    float Iden[9]={1,0,0,0,1,0,0,0,1},Zero[3]={0,0,0};
    float *dIden,*dZero; CUDA_CHECK(cudaMalloc(&dIden,9*sizeof(float)));CUDA_CHECK(cudaMalloc(&dZero,3*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dIden,Iden,9*sizeof(float),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dZero,Zero,3*sizeof(float),cudaMemcpyHostToDevice));

    float h0=0.07f; Grid g; g.ox=lo[0];g.oy=lo[1];g.oz=lo[2];g.inv_h=1.f/h0;
    g.nx=(int)std::ceil((hi[0]-lo[0])/h0)+1; g.ny=(int)std::ceil((hi[1]-lo[1])/h0)+1; g.nz=(int)std::ceil((hi[2]-lo[2])/h0)+1;
    int total=g.nx*g.ny*g.nz, gb=(total+255)/256;
    float *m0a,*m0b,*m1a,*m1b,*mna,*mnb;
    CUDA_CHECK(cudaMalloc(&m0a,total*sizeof(float)));CUDA_CHECK(cudaMalloc(&m0b,total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m1a,total*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&m1b,total*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mna,total*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&mnb,total*3*sizeof(float)));

    Pose T=T0; FRResult res; res.iters=0; if(traj)traj->push_back(T);
    const float sigmas_all[]={0.7f,0.5f,0.35f,0.25f,0.17f,0.11f,0.07f,0.05f};
    for(float sigma : sigmas_all){
        if(sigma < sig_floor-1e-6f) break;               // stop annealing at the floor
        float sv=sigma/h0; int Rk=std::max(1,(int)std::ceil(3.f*sv));
        std::vector<float> wk(2*Rk+1); float wsum=0;
        for(int d=-Rk;d<=Rk;++d){float wv=std::exp(-0.5f*d*d/(sv*sv));wk[d+Rk]=wv;wsum+=wv;} for(auto&v:wk)v/=wsum;
        float* dW; CUDA_CHECK(cudaMalloc(&dW,wk.size()*sizeof(float)));CUDA_CHECK(cudaMemcpy(dW,wk.data(),wk.size()*sizeof(float),cudaMemcpyHostToDevice));
        const int iters_per_level=8;
        for(int it=0;it<iters_per_level;++it){
            CUDA_CHECK(cudaMemcpy(dR,T.R.m,9*sizeof(float),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dt,T.t,3*sizeof(float),cudaMemcpyHostToDevice));
            transform_kernel<<<(M+255)/256,256>>>(dY,M,dR,dt,dP);
            // E-step filter A: model density Z_n
            CUDA_CHECK(cudaMemset(m0a,0,total*sizeof(float)));
            splat_kernel<<<(M+255)/256,256>>>(dP,nullptr,nullptr,M,g,m0a,nullptr,nullptr);
            blur_axis_kernel<<<gb,256>>>(m0a,m0b,g,0,Rk,dW,1);blur_axis_kernel<<<gb,256>>>(m0b,m0a,g,1,Rk,dW,1);blur_axis_kernel<<<gb,256>>>(m0a,m0b,g,2,Rk,dW,1);
            slice_scalar_kernel<<<(N+255)/256,256>>>(dX,N,g,m0b,dZ);
            float meanZ; { std::vector<float> hZ(N); CUDA_CHECK(cudaMemcpy(hZ.data(),dZ,N*sizeof(float),cudaMemcpyDeviceToHost)); double s=0; for(float z:hZ)s+=z; meanZ=(float)(s/N); }
            compute_a_kernel<<<(N+255)/256,256>>>(dZ,N,0.1f*meanZ+1e-9f,dA);
            // E-step filter B: correspondence moments + normal channel
            CUDA_CHECK(cudaMemset(m0a,0,total*sizeof(float)));CUDA_CHECK(cudaMemset(m1a,0,total*3*sizeof(float)));CUDA_CHECK(cudaMemset(mna,0,total*3*sizeof(float)));
            splat_kernel<<<(N+255)/256,256>>>(dX,dA,dNX,N,g,m0a,m1a,plane?mna:nullptr);
            blur_axis_kernel<<<gb,256>>>(m0a,m0b,g,0,Rk,dW,1);blur_axis_kernel<<<gb,256>>>(m0b,m0a,g,1,Rk,dW,1);blur_axis_kernel<<<gb,256>>>(m0a,m0b,g,2,Rk,dW,1);
            blur_axis_kernel<<<gb,256>>>(m1a,m1b,g,0,Rk,dW,3);blur_axis_kernel<<<gb,256>>>(m1b,m1a,g,1,Rk,dW,3);blur_axis_kernel<<<gb,256>>>(m1a,m1b,g,2,Rk,dW,3);
            if(plane){blur_axis_kernel<<<gb,256>>>(mna,mnb,g,0,Rk,dW,3);blur_axis_kernel<<<gb,256>>>(mnb,mna,g,1,Rk,dW,3);blur_axis_kernel<<<gb,256>>>(mna,mnb,g,2,Rk,dW,3);}
            slice_kernel<<<(M+255)/256,256>>>(dP,M,g,dIden,dZero,m0b,m1b,plane?mnb:nullptr,dM0,dM1,plane?dNm:nullptr,dP);
            // M-step
            CUDA_CHECK(cudaMemset(dHg,0,29*sizeof(float)));
            if(plane) mstep_plane_kernel<<<(M+255)/256,256>>>(dP,dM0,dM1,dNm,M,1e-12f,dHg);
            else      mstep_point_kernel<<<(M+255)/256,256>>>(dP,dM0,dM1,M,1e-12f,dHg);
            float Hg[29]; CUDA_CHECK(cudaMemcpy(Hg,dHg,29*sizeof(float),cudaMemcpyDeviceToHost));
            float d[6]; if(!solve6(Hg,Hg+21,d))break;
            T=pose_mul(se3_exp(d),T); ++res.iters; if(traj)traj->push_back(T);
            float step=0; for(int k=0;k<6;++k)step+=d[k]*d[k]; if(std::sqrt(step)<1e-5f)break;
        }
        cudaFree(dW);
    }
    cudaFree(m0a);cudaFree(m0b);cudaFree(m1a);cudaFree(m1b);cudaFree(mna);cudaFree(mnb);
    cudaFree(dZ);cudaFree(dA);cudaFree(dIden);cudaFree(dZero);
    res.T=T; cudaFree(dX);cudaFree(dY);cudaFree(dNX);cudaFree(dR);cudaFree(dt);cudaFree(dP);cudaFree(dM0);cudaFree(dM1);cudaFree(dNm);cudaFree(dHg);
    return res;
}

static void errs(const Pose& res,const Mat3& Rgt,const float* gt_t,float& ang,float& terr){
    Mat3 RgtT; for(int i=0;i<3;++i)for(int j=0;j<3;++j)RgtT.m[i*3+j]=Rgt.m[j*3+i];
    float texp[3]; mat3_vec(RgtT,gt_t,texp); for(int k=0;k<3;++k)texp[k]=-texp[k];
    Mat3 Rerr=mat3_mul(Rgt,res.R); float tr=Rerr.m[0]+Rerr.m[4]+Rerr.m[8];
    ang=std::acos(std::min(1.f,std::max(-1.f,(tr-1.f)*0.5f)));
    terr=0; for(int k=0;k<3;++k){float e=res.t[k]-texp[k];terr+=e*e;} terr=std::sqrt(terr);
}

// ============================ convergence GIF (point vs plane, side metric) ============================
static void render_gif(const std::vector<float>& X,const std::vector<float>& Y,const std::vector<Pose>& traj){
    const int W=1280,H=720,CX=380,CY=360; const float SCALE=78.f,elev=0.42f;
    auto sub=[](const std::vector<float>&P,int st){std::vector<float>q;for(size_t i=0;i<P.size()/3;i+=st){q.push_back(P[i*3]);q.push_back(P[i*3+1]);q.push_back(P[i*3+2]);}return q;};
    std::vector<float> Xs=sub(X,4),Ys=sub(Y,4);
    if(system("mkdir -p tmp")!=0)std::fprintf(stderr,"warn mkdir\n");
    cv::VideoWriter video("tmp/gpu_filterreg_p2plane.avi",cv::VideoWriter::fourcc('M','J','P','G'),20,cv::Size(W,H));
    int nt=(int)traj.size(); const int HOLD=26; int nf=nt+HOLD;
    struct Sp{float sx,sy,d;cv::Scalar c;};
    for(int f=0;f<nf;++f){int k=std::min(f,nt-1); float az=0.6f+f*0.018f,ca=std::cos(az),sa=std::sin(az),ce=std::cos(elev),se=std::sin(elev);
        cv::Mat img(H,W,CV_8UC3,cv::Scalar(26,26,32)); const Pose&T=traj[k];
        auto proj=[&](float x,float y,float z,float&sx,float&sy,float&d){float x1=x*ca-y*sa,y1=x*sa+y*ca,z1=z;sx=CX+SCALE*x1;sy=CY-SCALE*(z1*ce-y1*se);d=y1*ce+z1*se;};
        std::vector<Sp> sp;
        for(size_t i=0;i<Xs.size()/3;++i){Sp s;proj(Xs[i*3],Xs[i*3+1],Xs[i*3+2],s.sx,s.sy,s.d);s.c=cv::Scalar(210,180,60);sp.push_back(s);}
        for(size_t i=0;i<Ys.size()/3;++i){float y0[3]={Ys[i*3],Ys[i*3+1],Ys[i*3+2]},p[3];pose_apply(T,y0,p);Sp s;proj(p[0],p[1],p[2],s.sx,s.sy,s.d);s.c=cv::Scalar(40,130,240);sp.push_back(s);}
        std::sort(sp.begin(),sp.end(),[](const Sp&a,const Sp&b){return a.d<b.d;});
        float dmin=1e9f,dmax=-1e9f;for(auto&s:sp){dmin=std::min(dmin,s.d);dmax=std::max(dmax,s.d);}
        for(auto&s:sp){float t=(s.d-dmin)/(dmax-dmin+1e-6f);float b=0.45f+0.55f*t;cv::circle(img,cv::Point((int)s.sx,(int)s.sy),2,s.c*b,-1,cv::LINE_AA);}
        int px=800,py=70; auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA);};
        put("GPU FilterReg",py,1.0,cv::Scalar(235,235,245),2);py+=38;
        put("point-to-plane M-step",py,0.62,cv::Scalar(180,180,200),1);py+=50;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(210,180,60),-1);cv::putText(img,"fixed cloud",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=30;
        cv::circle(img,cv::Point(px+8,py-6),6,cv::Scalar(40,130,240),-1);cv::putText(img,"source (aligning)",cv::Point(px+26,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=52;
        char buf[96];std::snprintf(buf,sizeof(buf),"iteration %d / %d",k,nt-1);put(buf,py,0.62,cv::Scalar(210,210,225),1);py+=40;
        put("residual projected on surface normal",py,0.5,cv::Scalar(150,200,150),1);py+=26;
        put("removes soft-mean curvature bias",py,0.5,cv::Scalar(150,200,150),1);py+=44;
        if(f>=nf-HOLD)put("ALIGNED",py,0.8,cv::Scalar(120,230,250),2);
        video.write(img);}
    video.release(); avi_to_gif("tmp/gpu_filterreg_p2plane.avi","gif/gpu_filterreg_p2plane.gif",20,900);
    std::printf("wrote gif/gpu_filterreg_p2plane.gif\n");
}

}  // namespace cudabot

int main(){
    using namespace cudabot;
    std::printf("=== GPU FilterReg point-to-plane vs point-to-point (soft-mean bias) ===\n");
    const int N=9000;
    std::vector<float> X=make_lumpy(N,1);
    // centroid for outward normal orientation
    float c[3]={0,0,0}; for(int i=0;i<N;++i)for(int k=0;k<3;++k)c[k]+=X[i*3+k]; for(int k=0;k<3;++k)c[k]/=N;
    // GPU kNN-PCA normals on the fixed cloud (once)
    std::vector<float> NX(N*3);
    { float*dX,*dNX; CUDA_CHECK(cudaMalloc(&dX,N*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dNX,N*3*sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dX,X.data(),N*3*sizeof(float),cudaMemcpyHostToDevice));
      knn_normal_kernel<<<(N+127)/128,128>>>(dX,N,16,c[0],c[1],c[2],dNX);
      CUDA_CHECK(cudaMemcpy(NX.data(),dNX,N*3*sizeof(float),cudaMemcpyDeviceToHost)); cudaFree(dX);cudaFree(dNX); }

    std::mt19937 rng(7);
    float gt_w[3]={0.25f,-0.35f,0.20f}, gt_t[3]={0.7f,-0.5f,0.4f};
    Mat3 Rgt=so3_exp(gt_w); Pose Tgt; Tgt.R=Rgt; for(int k=0;k<3;++k)Tgt.t[k]=gt_t[k];
    std::normal_distribution<float> noise(0.f,0.02f); std::uniform_real_distribution<float> keep(0,1);
    std::vector<float> Y;
    for(int i=0;i<N;++i){ if(keep(rng)>0.85f)continue; float y[3]={X[i*3],X[i*3+1],X[i*3+2]},p[3]; pose_apply(Tgt,y,p);
        Y.push_back(p[0]+noise(rng));Y.push_back(p[1]+noise(rng));Y.push_back(p[2]+noise(rng)); }
    int M=Y.size()/3;
    Pose T0; T0.R={1,0,0,0,1,0,0,0,1}; T0.t[0]=T0.t[1]=T0.t[2]=0;
    std::printf("fixed N=%d  source M=%d  normals via kNN-PCA (k=16)\n", N, M);

    // ---------------- sigma-floor sweep: where does each metric land? ----------------
    // The anneal is stopped at a floor sigma; coarser floor = stronger soft-mean
    // curvature bias.  Point-to-plane projects it out, so it stays accurate at
    // coarse floors where point-to-point is still biased.
    std::printf("\nsigma-floor sweep (final rot err, deg) -- coarser floor = more soft-mean bias:\n");
    std::printf("  %-10s | %-14s | %-14s\n","floor sig","point-to-plane","point-to-point");
    std::printf("  -----------+----------------+---------------\n");
    const float floors[]={0.30f,0.25f,0.17f,0.11f,0.07f,0.05f};
    for(float fl : floors){
        FRResult rp=filterreg(X,NX,Y,T0,1,fl,nullptr);
        FRResult rq=filterreg(X,NX,Y,T0,0,fl,nullptr);
        float ap,tp,aq,tq; errs(rp.T,Rgt,gt_t,ap,tp); errs(rq.T,Rgt,gt_t,aq,tq);
        std::printf("  %8.2f   | %7.3f        | %7.3f        %s\n",
                    fl, ap*57.2958f, aq*57.2958f, (aq>1.8f*ap+0.02f*0.0175f)?"<- plane wins":"");
    }

    // ---------------- representative operating point (coarse floor) for the GIF ----------------
    const float REP=0.17f;
    std::vector<Pose> traj;
    auto t0=std::chrono::high_resolution_clock::now();
    FRResult rp=filterreg(X,NX,Y,T0,1,REP,&traj);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    FRResult rq=filterreg(X,NX,Y,T0,0,REP,nullptr);
    float ap,tp,aq,tq; errs(rp.T,Rgt,gt_t,ap,tp); errs(rq.T,Rgt,gt_t,aq,tq);
    std::printf("\nhead-to-head at coarse floor sigma=%.2f (no fine annealing):\n",REP);
    std::printf("  [point-to-plane] rot err = %.3f deg  trans err = %.4f\n",ap*57.2958f,tp);
    std::printf("  [point-to-point] rot err = %.3f deg  trans err = %.4f\n",aq*57.2958f,tq);
    std::printf("point-to-plane wall=%.1f ms (%d iters)\n",ms,rp.iters);

    bool plane_ok=(ap<0.02f && tp<0.05f);
    bool plane_better=(aq>1.5f*ap+3e-3f)||(tq>1.5f*tp+0.01f);
    if(plane_ok && plane_better) std::printf("RESULT: PASS -- point-to-plane stays accurate at a coarse sigma where point-to-point carries soft-mean bias.\n");
    else if(plane_ok)            std::printf("RESULT: PARTIAL -- point-to-plane accurate, but the gap to point-to-point is small here.\n");
    else                         std::printf("RESULT: CHECK -- point-to-plane not within tolerance at the coarse floor.\n");

    render_gif(X,Y,traj);
    return 0;
}
