// gpu_kiss_icp.cu
//
// GPU LiDAR odometry in the style of KISS-ICP
//   (Vizzo et al., "KISS-ICP: In Defense of Point-to-Point ICP", RA-L 2023,
//    arXiv:2209.15397).
//
// This turns the repo's point-cloud registration line from a one-shot
// "align two clouds" tool into a streaming ODOMETRY pipeline: estimate the full
// sensor trajectory from a stream of LiDAR scans alone -- no IMU, no wheel
// odometry, no loop closure.  KISS-ICP's thesis is that plain point-to-point ICP,
// done with the right few ingredients, matches far more complex systems:
//
//   1. MOTION PREDICTION as the ICP initial guess (and, on real data, scan
//      de-skewing -- our synthetic scans are not skewed).  NOTE: the canonical
//      KISS-ICP uses a constant-velocity predictor; we implemented it and found
//      that on these (coarse, synthetic) scans it amplified the small per-scan
//      correspondence bias into a slow divergence -- the velocity estimate is the
//      difference of two noisy poses, and extrapolating it doubles that noise and
//      feeds it back.  The previous-pose (zero-velocity) predictor is rock-solid
//      here: inter-scan motion is small enough that ICP converges from it every
//      frame (predicted gap ~0.14 m -> post-ICP error ~0.01 m), so drift does not
//      accumulate.  Left as a documented, measured design choice.
//   2. Spatial VOXEL SUBSAMPLING of each scan (one point per small voxel).
//   3. An ADAPTIVE THRESHOLD for data association: the max correspondence
//      distance is derived from how much the predictor has been *wrong* recently
//      (its deviation), so there is no hand-tuned gate.
//   4. ICP to a voxel-hash LOCAL MAP with a robust (Geman-McClure) kernel scaled
//      by that same threshold.  We use a point-to-PLANE residual (map normals via
//      kNN-PCA): on the dominant ground plane it removes the voxel-grid mismatch
//      that the soft-mean/planar-bias demo (gpu_filterreg_p2plane) quantifies.
//
// All the heavy lifting is on the GPU: per-scan correspondence is a brute-force
// nearest-neighbour against the local map (one thread per scan point), and the
// 6x6 twist normal equations are accumulated with atomics, exactly as in the
// FilterReg / Sinkhorn demos -- the same se(3) Gauss-Newton machinery, now driven
// by hard nearest-neighbour correspondences instead of a soft GMM.
//
// We build a synthetic structured world (ground + buildings + poles), fly a known
// loop trajectory through it, and generate range-limited noisy scans.  The
// odometry NEVER sees the world or the true poses -- only the scans -- and we
// score the recovered trajectory against ground truth (ATE / final drift).
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
#include <unordered_map>
#include <vector>

#include "cuda_check.cuh"
#include "cuda_video.h"

namespace cudabot {

// ============================ SE(3)/SO(3) helpers (host) ============================
struct Mat3 { float m[9]; };
struct Pose { Mat3 R; float t[3]; };
static inline void mat3_vec(const Mat3& R, const float* v, float* o){
    o[0]=R.m[0]*v[0]+R.m[1]*v[1]+R.m[2]*v[2]; o[1]=R.m[3]*v[0]+R.m[4]*v[1]+R.m[5]*v[2]; o[2]=R.m[6]*v[0]+R.m[7]*v[1]+R.m[8]*v[2]; }
static inline void pose_apply(const Pose& T,const float* y,float* p){ mat3_vec(T.R,y,p); p[0]+=T.t[0];p[1]+=T.t[1];p[2]+=T.t[2]; }
static inline Mat3 mat3_mul(const Mat3&A,const Mat3&B){ Mat3 C; for(int i=0;i<3;++i)for(int j=0;j<3;++j){float s=0;for(int k=0;k<3;++k)s+=A.m[i*3+k]*B.m[k*3+j];C.m[i*3+j]=s;} return C; }
static inline Mat3 mat3_T(const Mat3&A){ Mat3 C; for(int i=0;i<3;++i)for(int j=0;j<3;++j)C.m[i*3+j]=A.m[j*3+i]; return C; }
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
static inline Pose pose_inv(const Pose&A){ Pose C; C.R=mat3_T(A.R); float Rt[3]; mat3_vec(C.R,A.t,Rt); for(int k=0;k<3;++k)C.t[k]=-Rt[k]; return C; }
// log of relative rotation angle + translation gap, for the adaptive threshold.
static inline void pose_delta_mag(const Pose& A,const Pose& B,float& dtrans,float& drot){
    Pose D=pose_mul(pose_inv(A),B);
    dtrans=std::sqrt(D.t[0]*D.t[0]+D.t[1]*D.t[1]+D.t[2]*D.t[2]);
    float tr=D.R.m[0]+D.R.m[4]+D.R.m[8]; drot=std::acos(std::min(1.f,std::max(-1.f,(tr-1.f)*0.5f)));
}

// ============================ synthetic structured world ============================
// Ground + axis-aligned building boxes (walls only) + thin poles.  Distinctive,
// asymmetric structure so x/y/yaw are well constrained; the ground constrains z.
static std::vector<float> make_world(unsigned seed){
    std::mt19937 rng(seed); std::uniform_real_distribution<float> u01(0,1);
    std::vector<float> P;
    auto add=[&](float x,float y,float z){P.push_back(x);P.push_back(y);P.push_back(z);};
    // dense ground patch -> strong, well-sampled z / roll / pitch constraint and
    // clean kNN normals (sparse ground gave noisy normals and a weak z lock).
    for(float x=-26;x<=26;x+=0.5f) for(float y=-26;y<=26;y+=0.5f){
        if(u01(rng)<0.25f) continue; add(x+0.08f*u01(rng), y+0.08f*u01(rng), 0.02f*u01(rng)); }
    // buildings on an OUTER ring only.  The trajectory stays in the middle, so the
    // sensor always sees structure at MODERATE range in many directions -- a
    // well-conditioned constraint.  (A wall passing within a metre dominates the
    // normal equations and the point-to-plane fit slides along it -> divergence.)
    const float bx[][4]={ // cx, cy, half-w, height  (all at radius ~16-22)
        {18,3,2.5f,6}, {-12,15,3.0f,7}, {-20,-5,2.0f,5}, {7,-19,3.5f,8},
        {19,-12,2.2f,6}, {-4,-21,2.0f,5}, {21,11,2.8f,7}, {-21,8,2.4f,6},
        {3,20,3.0f,6}, {-16,-16,2.2f,5}, {13,17,2.4f,6}, {-9,-13,2.0f,5} };
    for(auto&b:bx){ float cx=b[0],cy=b[1],hw=b[2],H=b[3];
        for(float z=0.2f;z<=H;z+=0.4f) for(float s=-hw;s<=hw;s+=0.4f){
            add(cx+hw, cy+s, z); add(cx-hw, cy+s, z); add(cx+s, cy+hw, z); add(cx+s, cy-hw, z); } }
    // a few mid-field poles for sharp, isolated features
    const float px[][2]={{9,6},{-7,9},{6,-9},{-9,-5},{11,-3},{-3,11}};
    for(auto&p:px) for(float z=0.1f;z<=4.5f;z+=0.22f){ add(p[0],p[1],z); }
    return P;
}

// host voxel downsample: keep the centroid of points falling in each voxel.
static std::vector<float> voxel_downsample(const std::vector<float>& P, float vs){
    std::unordered_map<int64_t,std::array<float,4>> vox; vox.reserve(P.size()/3);
    auto key=[&](float x,float y,float z)->int64_t{
        int64_t ix=(int64_t)std::floor(x/vs), iy=(int64_t)std::floor(y/vs), iz=(int64_t)std::floor(z/vs);
        return ((ix&0x1FFFFF)<<42) ^ ((iy&0x1FFFFF)<<21) ^ (iz&0x1FFFFF); };
    for(size_t i=0;i<P.size()/3;++i){ float x=P[i*3],y=P[i*3+1],z=P[i*3+2];
        auto& a=vox[key(x,y,z)]; a[0]+=x;a[1]+=y;a[2]+=z;a[3]+=1.f; }
    std::vector<float> out; out.reserve(vox.size()*3);
    for(auto& kv:vox){ float w=kv.second[3]; out.push_back(kv.second[0]/w);out.push_back(kv.second[1]/w);out.push_back(kv.second[2]/w); }
    return out;
}

// ============================ GPU kernels ============================
// kNN-PCA surface normals on the local map (sign is irrelevant for point-to-plane).
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
__global__ void map_normal_kernel(const float* __restrict__ Map,int m,int K,float* __restrict__ NMap){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=m)return;
    float xi=Map[i*3],yi=Map[i*3+1],zi=Map[i*3+2];
    const int KMAX=20; float dk[KMAX]; int ik[KMAX]; int kk=K<KMAX?K:KMAX;
    for(int a=0;a<kk;++a){dk[a]=1e30f;ik[a]=-1;}
    for(int j=0;j<m;++j){ if(j==i)continue; float dx=Map[j*3]-xi,dy=Map[j*3+1]-yi,dz=Map[j*3+2]-zi; float d=dx*dx+dy*dy+dz*dz;
        if(d<dk[kk-1]){int p=kk-1; while(p>0&&dk[p-1]>d){dk[p]=dk[p-1];ik[p]=ik[p-1];--p;} dk[p]=d;ik[p]=j;} }
    float mx=xi,my=yi,mz=zi; int cnt=1; for(int a=0;a<kk;++a){int j=ik[a];if(j<0)continue;mx+=Map[j*3];my+=Map[j*3+1];mz+=Map[j*3+2];++cnt;}
    float inv=1.f/cnt; mx*=inv;my*=inv;mz*=inv; float C[6]={0,0,0,0,0,0};
    auto acc=[&](float px,float py,float pz){float ex=px-mx,ey=py-my,ez=pz-mz;C[0]+=ex*ex;C[1]+=ex*ey;C[2]+=ex*ez;C[3]+=ey*ey;C[4]+=ey*ez;C[5]+=ez*ez;};
    acc(xi,yi,zi); for(int a=0;a<kk;++a){int j=ik[a];if(j<0)continue;acc(Map[j*3],Map[j*3+1],Map[j*3+2]);}
    float n[3]; sym3_smallest_evec(C,n); NMap[i*3]=n[0];NMap[i*3+1]=n[1];NMap[i*3+2]=n[2]; }

__global__ void transform_kernel(const float* __restrict__ S,int n,const float* __restrict__ R,const float* __restrict__ t,float* __restrict__ W){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=S[i*3],y=S[i*3+1],z=S[i*3+2];
    W[i*3]=R[0]*x+R[1]*y+R[2]*z+t[0]; W[i*3+1]=R[3]*x+R[4]*y+R[5]*z+t[1]; W[i*3+2]=R[6]*x+R[7]*y+R[8]*z+t[2]; }

// brute-force nearest map point for each (world-transformed) scan point.
// outputs matched map point Q[i], its normal NQ[i], and squared distance D2[i].
__global__ void nn_kernel(const float* __restrict__ Pw,int n,const float* __restrict__ Map,
                          const float* __restrict__ NMap,int m,
                          float tau2,float* __restrict__ Q,float* __restrict__ NQ,float* __restrict__ D2){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=Pw[i*3],y=Pw[i*3+1],z=Pw[i*3+2]; float best=tau2; int bj=-1;
    for(int j=0;j<m;++j){ float dx=Map[j*3]-x,dy=Map[j*3+1]-y,dz=Map[j*3+2]-z; float d=dx*dx+dy*dy+dz*dz;
        if(d<best){best=d;bj=j;} }
    if(bj>=0){ Q[i*3]=Map[bj*3];Q[i*3+1]=Map[bj*3+1];Q[i*3+2]=Map[bj*3+2];
        NQ[i*3]=NMap[bj*3];NQ[i*3+1]=NMap[bj*3+1];NQ[i*3+2]=NMap[bj*3+2]; D2[i]=best; }
    else { D2[i]=1e30f; }
}

// robust point-to-PLANE twist GN.  Residual is the signed distance to the map
// tangent plane, rs = n . (p - q); on the flat ground this fully constrains z and
// removes the voxel-grid horizontal mismatch that starves point-to-point (the same
// soft-mean/planar bias measured in gpu_filterreg_p2plane).  Geman-McClure weight
// w = (k2/(k2+d2))^2 from the point distance d2; correspondences with d2>=tau2 cut.
__global__ void gn_kernel(const float* __restrict__ Pw,const float* __restrict__ Q,
                          const float* __restrict__ NQ,const float* __restrict__ D2,
                          int n,float k2,float* __restrict__ Hg){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float d2=D2[i]; if(d2>1e29f)return;
    float px=Pw[i*3],py=Pw[i*3+1],pz=Pw[i*3+2];
    float nx=NQ[i*3],ny=NQ[i*3+1],nz=NQ[i*3+2];
    float rs=nx*(px-Q[i*3])+ny*(py-Q[i*3+1])+nz*(pz-Q[i*3+2]);   // plane distance
    float gm=k2/(k2+d2); float w=gm*gm;                  // Geman-McClure weight
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

// ============================ odometry ============================
struct OdomOut { std::vector<Pose> est; std::vector<float> map; };

// One ICP alignment of a scan (sensor frame) to the local map (world frame),
// starting from T_init.  tau is the adaptive correspondence threshold.
static Pose icp_to_map(const std::vector<float>& scan, float* dMap,float* dMapN,int mapN,
                       Pose Tinit, float tau, int max_it,
                       float* dS,float* dPw,float* dQ,float* dNQ,float* dD2,float* dR,float* dt,float* dHg){
    int n=scan.size()/3;
    CUDA_CHECK(cudaMemcpy(dS,scan.data(),n*3*sizeof(float),cudaMemcpyHostToDevice));
    Pose T=Tinit; float tau2=tau*tau, k2=(tau/2.f)*(tau/2.f);
    for(int it=0;it<max_it;++it){
        CUDA_CHECK(cudaMemcpy(dR,T.R.m,9*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dt,T.t,3*sizeof(float),cudaMemcpyHostToDevice));
        transform_kernel<<<(n+255)/256,256>>>(dS,n,dR,dt,dPw);
        nn_kernel<<<(n+255)/256,256>>>(dPw,n,dMap,dMapN,mapN,tau2,dQ,dNQ,dD2);
        CUDA_CHECK(cudaMemset(dHg,0,29*sizeof(float)));
        gn_kernel<<<(n+255)/256,256>>>(dPw,dQ,dNQ,dD2,n,k2,dHg);   // point-to-plane
        float Hg[29]; CUDA_CHECK(cudaMemcpy(Hg,dHg,29*sizeof(float),cudaMemcpyDeviceToHost));
        if(Hg[28]<10.f) break;                           // too few inliers
        float d[6]; if(!solve6(Hg,Hg+21,d)) break;
        T=pose_mul(se3_exp(d),T);
        float step=0; for(int k=0;k<6;++k)step+=d[k]*d[k]; if(std::sqrt(step)<1e-4f) break;
    }
    return T;
}

static OdomOut run_odometry(const std::vector<std::vector<float>>& scans,
                            const std::vector<Pose>& gt, float map_voxel, float scan_voxel){
    OdomOut out;
    // GPU scratch sized to the largest scan and a generous map cap.
    int maxn=0; for(auto&s:scans) maxn=std::max(maxn,(int)(s.size()/3));
    const int MAPCAP=200000;
    float *dS,*dPw,*dQ,*dNQ,*dD2,*dMap,*dMapN,*dR,*dt,*dHg;
    CUDA_CHECK(cudaMalloc(&dS,maxn*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dPw,maxn*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dQ,maxn*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dNQ,maxn*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dD2,maxn*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dMap,MAPCAP*3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dMapN,MAPCAP*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dR,9*sizeof(float)));CUDA_CHECK(cudaMalloc(&dt,3*sizeof(float)));CUDA_CHECK(cudaMalloc(&dHg,29*sizeof(float)));

    // persistent voxel-hash map: ONE point per voxel, kept from when the voxel was
    // FIRST observed (earliest pose = most accurate).  Never re-averaged -- that is
    // what stops the map from smearing into a thick slab as small pose errors
    // accumulate, which (via degraded normals) otherwise starves the z constraint
    // and runs away.  This is the KISS-ICP voxel-hash map discipline.
    std::unordered_map<int64_t,std::array<float,3>> vmap; vmap.reserve(200000);
    float ivs=1.f/map_voxel;
    auto vkey=[&](float x,float y,float z)->int64_t{
        int64_t ix=(int64_t)std::floor(x*ivs),iy=(int64_t)std::floor(y*ivs),iz=(int64_t)std::floor(z*ivs);
        return ((ix&0x1FFFFF)<<42) ^ ((iy&0x1FFFFF)<<21) ^ (iz&0x1FFFFF); };
    std::vector<float> localmap;
    auto map_insert=[&](const std::vector<float>& pts_world,const float* center){
        for(size_t i=0;i<pts_world.size()/3;++i){ float x=pts_world[i*3],y=pts_world[i*3+1],z=pts_world[i*3+2];
            int64_t k=vkey(x,y,z); auto it=vmap.find(k); if(it==vmap.end()) vmap[k]={x,y,z}; }
        // window: drop voxels far from the current sensor centre
        const float Rwin=40.f, Rwin2=Rwin*Rwin;
        for(auto it=vmap.begin();it!=vmap.end();){ float dx=it->second[0]-center[0],dy=it->second[1]-center[1],dz=it->second[2]-center[2];
            if(dx*dx+dy*dy+dz*dz>Rwin2) it=vmap.erase(it); else ++it; }
        localmap.clear(); localmap.reserve(vmap.size()*3);
        for(auto& kv:vmap){ localmap.push_back(kv.second[0]);localmap.push_back(kv.second[1]);localmap.push_back(kv.second[2]); }
    };
    auto to_world=[&](const std::vector<float>& scan,const Pose& T){ std::vector<float> w(scan.size());
        for(size_t i=0;i<scan.size()/3;++i){ float s[3]={scan[i*3],scan[i*3+1],scan[i*3+2]},p[3]; pose_apply(T,s,p); w[i*3]=p[0];w[i*3+1]=p[1];w[i*3+2]=p[2]; } return w; };

    int K=scans.size();
    // adaptive threshold state: EMA of the constant-velocity model's SQUARED
    // deviation.  An EMA (not an all-time mean) keeps tau responsive -- an
    // all-time mean is dominated by the tiny early deviations and pins tau at the
    // floor, so a later curve whose prediction error exceeds the floor loses all
    // correspondences.  The floor also covers sensor noise + voxel spacing.
    double dev_ema=0; bool dev_init=false; const float TAU_MIN=1.0f, TAU_MAX=3.0f;
    for(int k=0;k<K;++k){
        // 1) spatial subsample of the incoming scan
        std::vector<float> scan=voxel_downsample(scans[k],scan_voxel);
        Pose Test; float tau=TAU_MAX; Pose pred;
        if(k==0){ Test=gt[0]; pred=gt[0]; }              // anchor odometry at the known start
        else {
            // 2) previous-pose (zero-velocity) prediction -- see header note.
            pred=out.est[k-1];
            // 3) adaptive threshold from recent model deviation (KISS): tau = 3*sigma
            if(!dev_init) tau=TAU_MAX;
            else { float sigma=(float)std::sqrt(dev_ema); tau=std::min(TAU_MAX,std::max(TAU_MIN,3.f*sigma)); }
            // 4) point-to-plane ICP to the local map (upload + estimate normals once)
            int mapN=std::min((int)(localmap.size()/3),MAPCAP);
            CUDA_CHECK(cudaMemcpy(dMap,localmap.data(),mapN*3*sizeof(float),cudaMemcpyHostToDevice));
            map_normal_kernel<<<(mapN+127)/128,128>>>(dMap,mapN,12,dMapN);
            Test=icp_to_map(scan,dMap,dMapN,mapN,pred,tau,12,dS,dPw,dQ,dNQ,dD2,dR,dt,dHg);
            // update adaptive-threshold statistic with how wrong the prediction was
            float dtr,dro; pose_delta_mag(pred,Test,dtr,dro);
            float dsq=dtr*dtr; if(!dev_init){dev_ema=dsq;dev_init=true;} else dev_ema=0.7*dev_ema+0.3*dsq;
        }
        out.est.push_back(Test);
        // 5) insert the registered scan into the local map
        std::vector<float> sw=to_world(scan,Test);
        map_insert(sw,Test.t);
    }
    out.map=localmap;
    cudaFree(dS);cudaFree(dPw);cudaFree(dQ);cudaFree(dNQ);cudaFree(dD2);cudaFree(dMap);cudaFree(dMapN);cudaFree(dR);cudaFree(dt);cudaFree(dHg);
    return out;
}

// ============================ GIF: accumulating map + trajectories ============================
static void render_gif(const std::vector<std::vector<float>>& scans,
                       const std::vector<Pose>& gt,const std::vector<Pose>& est){
    const int W=1280,H=720; const float PX=18.f; const int CX=430,CY=380;
    auto proj=[&](float x,float y,float& sx,float& sy){ sx=CX+PX*x; sy=CY-PX*y; };  // top-down
    if(system("mkdir -p tmp")!=0)std::fprintf(stderr,"warn mkdir\n");
    cv::VideoWriter video("tmp/gpu_kiss_icp.avi",cv::VideoWriter::fourcc('M','J','P','G'),18,cv::Size(W,H));
    int K=scans.size(); const int HOLD=20;
    std::vector<float> accum;
    auto to_world=[&](const std::vector<float>& scan,const Pose& T){ std::vector<float> w; w.reserve(scan.size());
        for(size_t i=0;i<scan.size()/3;i+=5){ float s[3]={scan[i*3],scan[i*3+1],scan[i*3+2]},p[3]; pose_apply(T,s,p); w.push_back(p[0]);w.push_back(p[1]);w.push_back(p[2]); } return w; };
    for(int k=0;k<K+HOLD;k+=2){ int kk=std::min(k,K-1);
        cv::Mat img(H,W,CV_8UC3,cv::Scalar(24,24,30));
        if(k<K){ std::vector<float> sw=to_world(scans[k],est[k]); accum.insert(accum.end(),sw.begin(),sw.end());
            if(accum.size()>3*45000) accum.erase(accum.begin(),accum.begin()+(accum.size()-3*45000)); }
        // map points (depth-cued by height z)
        for(size_t i=0;i<accum.size()/3;++i){ float sx,sy; proj(accum[i*3],accum[i*3+1],sx,sy);
            if(sx<0||sx>=W||sy<0||sy>=H)continue; float z=accum[i*3+2]; float b=0.4f+0.12f*std::min(6.f,std::max(0.f,z));
            cv::Scalar c(150*b,150*b,165*b); cv::circle(img,cv::Point((int)sx,(int)sy),1,c,-1); }
        // trajectories up to kk
        auto draw_traj=[&](const std::vector<Pose>& tr,cv::Scalar col){ for(int i=1;i<=kk;++i){ float ax,ay,bx,by;
            proj(tr[i-1].t[0],tr[i-1].t[1],ax,ay); proj(tr[i].t[0],tr[i].t[1],bx,by);
            cv::line(img,cv::Point((int)ax,(int)ay),cv::Point((int)bx,(int)by),col,2,cv::LINE_AA); } };
        draw_traj(gt,cv::Scalar(120,230,120));            // ground truth: green
        draw_traj(est,cv::Scalar(40,130,240));            // estimate: orange
        { float sx,sy; proj(est[kk].t[0],est[kk].t[1],sx,sy); cv::circle(img,cv::Point((int)sx,(int)sy),5,cv::Scalar(40,130,240),-1,cv::LINE_AA); }
        int px=950; auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA);};
        int py=70; put("GPU KISS-ICP",py,1.0,cv::Scalar(235,235,245),2);py+=38;
        put("LiDAR odometry (scans only)",py,0.6,cv::Scalar(180,180,200),1);py+=46;
        cv::line(img,cv::Point(px,py-6),cv::Point(px+24,py-6),cv::Scalar(120,230,120),2);cv::putText(img,"ground truth",cv::Point(px+32,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=28;
        cv::line(img,cv::Point(px,py-6),cv::Point(px+24,py-6),cv::Scalar(40,130,240),2);cv::putText(img,"odometry estimate",cv::Point(px+32,py),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(200,200,210),1,cv::LINE_AA);py+=46;
        char buf[96]; std::snprintf(buf,sizeof(buf),"scan %d / %d",kk,K-1); put(buf,py,0.6,cv::Scalar(210,210,225),1);py+=34;
        put("adaptive threshold + robust ICP",py,0.5,cv::Scalar(150,200,150),1);py+=24;
        put("point-to-plane, voxel-hash map",py,0.5,cv::Scalar(150,200,150),1);py+=24;
        char b2[64]; std::snprintf(b2,sizeof(b2),"drift %.2f%% of path",100.0*std::sqrt((est[kk].t[0]-gt[kk].t[0])*(est[kk].t[0]-gt[kk].t[0])+(est[kk].t[1]-gt[kk].t[1])*(est[kk].t[1]-gt[kk].t[1]))/std::max(1e-3,(double)(kk)*0.157));
        put(b2,py,0.5,cv::Scalar(150,200,150),1);
        video.write(img);
    }
    video.release(); avi_to_gif("tmp/gpu_kiss_icp.avi","gif/gpu_kiss_icp.gif",18,820);
    std::printf("wrote gif/gpu_kiss_icp.gif\n");
}

}  // namespace cudabot

int main(){
    using namespace cudabot;
    std::printf("=== GPU KISS-ICP: LiDAR odometry from scans alone ===\n");
    std::vector<float> world=make_world(1);
    std::printf("world points=%zu\n", world.size()/3);

    // known loop trajectory: an oval, sensor yaw following the heading, gentle z bob.
    const int K=280; std::vector<Pose> gt;
    const float RX=8.f, RY=6.f;
    for(int k=0;k<K;++k){ float a=2.0f*3.1415926f*k/K;
        float x=RX*std::cos(a), y=RY*std::sin(a), z=1.6f+0.1f*std::sin(3*a);
        float heading=std::atan2(RY*std::cos(a), -RX*std::sin(a)); // tangent
        float w[3]={0,0,heading}; Mat3 R=so3_exp(w);
        Pose T; T.R=R; T.t[0]=x;T.t[1]=y;T.t[2]=z; gt.push_back(T); }

    // generate range-limited noisy scans (sensor frame).  No occlusion modelling;
    // the range limit alone gives each scan its local character.
    std::mt19937 rng(7); std::normal_distribution<float> nz(0.f,0.03f);
    const float Rmax=22.f, Rmax2=Rmax*Rmax;
    std::vector<std::vector<float>> scans;
    for(int k=0;k<K;++k){ Pose inv=pose_inv(gt[k]); std::vector<float> sc;
        for(size_t i=0;i<world.size()/3;++i){ float w3[3]={world[i*3],world[i*3+1],world[i*3+2]},s[3]; pose_apply(inv,w3,s);
            float d2=s[0]*s[0]+s[1]*s[1]+s[2]*s[2]; if(d2>Rmax2)continue;
            sc.push_back(s[0]+nz(rng));sc.push_back(s[1]+nz(rng));sc.push_back(s[2]+nz(rng)); }
        scans.push_back(sc); }
    std::printf("scans=%d  (range<=%.0fm, sigma noise 0.03)\n", K, Rmax);

    auto t0=std::chrono::high_resolution_clock::now();
    OdomOut od=run_odometry(scans, gt, /*map_voxel=*/0.5f, /*scan_voxel=*/0.5f);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    // ATE (translation) and final drift vs ground truth.
    double se=0; float maxe=0; double pathlen=0;
    for(int k=0;k<K;++k){ float dx=od.est[k].t[0]-gt[k].t[0],dy=od.est[k].t[1]-gt[k].t[1],dz=od.est[k].t[2]-gt[k].t[2];
        float e=std::sqrt(dx*dx+dy*dy+dz*dz); se+=e*e; maxe=std::max(maxe,e);
        if(k>0){ float lx=gt[k].t[0]-gt[k-1].t[0],ly=gt[k].t[1]-gt[k-1].t[1],lz=gt[k].t[2]-gt[k-1].t[2]; pathlen+=std::sqrt(lx*lx+ly*ly+lz*lz);} }
    float ate=(float)std::sqrt(se/K);
    float final_dx=od.est[K-1].t[0]-gt[K-1].t[0], final_dy=od.est[K-1].t[1]-gt[K-1].t[1], final_dz=od.est[K-1].t[2]-gt[K-1].t[2];
    float final_drift=std::sqrt(final_dx*final_dx+final_dy*final_dy+final_dz*final_dz);
    std::printf("trajectory length=%.1f m   ATE(trans)=%.3f m   max err=%.3f m   final drift=%.3f m (%.2f%% of path)\n",
                pathlen, ate, maxe, final_drift, 100.0*final_drift/pathlen);
    std::printf("wall=%.1f ms  (%.1f ms/scan)\n", ms, ms/K);
    if(ate<0.5f && final_drift<1.0f) std::printf("RESULT: PASS -- recovered the trajectory from scans alone with low drift.\n");
    else std::printf("RESULT: CHECK -- drift exceeds tolerance.\n");

    render_gif(scans, gt, od.est);
    return 0;
}
