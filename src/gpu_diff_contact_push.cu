// gpu_diff_contact_push.cu
//
// GPU DIFFERENTIABLE contact simulation: autodiff-through-contact planar pushing.
//
// Pushing is the canonical contact-rich manipulation task, and the reason it is
// hard is exactly the reason it is interesting here: the object's ORIENTATION is
// controlled only through the CONTACT TORQUE -- you cannot set it without
// reasoning through the contact.  This demo builds a smooth, fully differentiable
// pusher/disk simulator and back-propagates the final object pose through the
// whole rollout (forward-mode dual numbers, the repo's autodiff idiom), then uses
// the analytic gradient to optimise a push that drives the object to a target
// SE(2) pose -- position AND heading.
//
// Model (smooth so it is differentiable everywhere -- verified vs finite diffs):
//   * a point pusher travels in a straight line with a chosen IMPACT PARAMETER b
//     (perpendicular offset of the push line from the disk centre), heading phi,
//     and length L  -- three control parameters.
//   * contact is a soft spring: normal force F = k * softplus(penetration) (softplus,
//     not relu, so the force and its gradient are C-infinity).  A pure normal force
//     points at the COM and gives ZERO torque; the rotation comes from TANGENTIAL
//     CONTACT FRICTION (smooth Coulomb, mu*N*tanh(slip)) resisting the pusher
//     sliding across the off-centre surface -- Mason's pushing mechanics.  Over-
//     damped (quasi-static) plane friction brings the disk to rest after the push.
//   * loss = w_p ||p - p*||^2 + w_th (1 - cos(theta - theta*)) ; both smooth.
//
// Autodiff: a Dual<3> number carries d/d(b,phi,L) through the entire rollout, so
// ONE simulation yields the loss AND its exact 3-gradient.  Gradient descent then
// finds the push.  One GPU THREAD optimises one (object, target) problem, so a
// whole batch is solved at once.
//
// Verified head-to-head: with the contact torque available (b free) the optimiser
// hits position AND heading; a CENTRE-PUSH baseline (b forced to 0, no torque)
// hits position but cannot set heading.  Then a GIF of objects being pushed and
// reoriented onto their targets.  Build: CMakeLists, --expt-relaxed-constexpr.

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

static const int NP=3;              // control params: b, phi, L
static const int TSTEP=40;
static const float DT=0.05f;

// ---- forward-mode dual number with a 3-gradient ----
struct D { float v; float g[NP]; };
__device__ __host__ D dc(float v){ D r; r.v=v; for(int i=0;i<NP;++i)r.g[i]=0; return r; }
__device__ __host__ D dvar(float v,int idx){ D r; r.v=v; for(int i=0;i<NP;++i)r.g[i]=(i==idx)?1.f:0.f; return r; }
__device__ __host__ D add(D a,D b){ D r; r.v=a.v+b.v; for(int i=0;i<NP;++i)r.g[i]=a.g[i]+b.g[i]; return r; }
__device__ __host__ D sub(D a,D b){ D r; r.v=a.v-b.v; for(int i=0;i<NP;++i)r.g[i]=a.g[i]-b.g[i]; return r; }
__device__ __host__ D mul(D a,D b){ D r; r.v=a.v*b.v; for(int i=0;i<NP;++i)r.g[i]=a.g[i]*b.v+a.v*b.g[i]; return r; }
__device__ __host__ D muls(D a,float s){ D r; r.v=a.v*s; for(int i=0;i<NP;++i)r.g[i]=a.g[i]*s; return r; }
__device__ __host__ D dvd(D a,D b){ D r; float inv=1.f/b.v; r.v=a.v*inv; for(int i=0;i<NP;++i)r.g[i]=(a.g[i]*b.v-a.v*b.g[i])*inv*inv; return r; }
__device__ __host__ D dsqrt(D a){ D r; float s=sqrtf(a.v); r.v=s; float d=(s>1e-12f)?0.5f/s:0.f; for(int i=0;i<NP;++i)r.g[i]=a.g[i]*d; return r; }
__device__ __host__ D dsin(D a){ D r; r.v=sinf(a.v); float c=cosf(a.v); for(int i=0;i<NP;++i)r.g[i]=a.g[i]*c; return r; }
__device__ __host__ D dcos(D a){ D r; r.v=cosf(a.v); float s=-sinf(a.v); for(int i=0;i<NP;++i)r.g[i]=a.g[i]*s; return r; }
__device__ __host__ D dtanh(D a){ D r; float t=tanhf(a.v); r.v=t; float d=1.f-t*t; for(int i=0;i<NP;++i)r.g[i]=a.g[i]*d; return r; }
// smooth (C-infinity) penetration: softplus.  Replaces the relu kink so the
// contact force -- and the autodiff gradient through it -- is exact everywhere.
__device__ __host__ D dsoftplus(D a){ const float be=0.03f; D r; float e=expf(a.v/be);
    r.v=be*logf(1.f+e); float s=e/(1.f+e); for(int i=0;i<NP;++i)r.g[i]=a.g[i]*s; return r; }

// physical constants
__device__ __host__ struct Phys { float R,rp,k,m,I,clin,cang; };
__device__ __host__ Phys phys(){ Phys p; p.R=0.5f;p.rp=0.12f;p.k=120.f;p.m=1.f;p.I=0.5f*1.f*0.5f*0.5f;p.clin=6.f;p.cang=6.f; return p; }

// Differentiable rollout: push params (b,phi,L) -> final disk pose (dx,dy,dth) as Duals.
// Disk starts at origin pose 0.  Returns via out[3].
__device__ __host__ void rollout(const D* P, D* out){
    Phys ph=phys();
    D b=P[0], phi=P[1], L=P[2];
    D cphi=dcos(phi), sphi=dsin(phi);
    D px=dc(0),py=dc(0);                          // perpendicular unit = (-sphi, cphi)
    // pusher start: behind the disk along -d, offset by b along perp
    D startx=add(muls(cphi,-(ph.R+ph.rp+0.4f)), mul(b,muls(sphi,-1.f)));
    D starty=add(muls(sphi,-(ph.R+ph.rp+0.4f)), mul(b,cphi));
    D dx=dc(0),dy=dc(0),dth=dc(0), vx=dc(0),vy=dc(0),om=dc(0);
    for(int s=0;s<TSTEP;++s){
        float frac=(float)(s+1)/TSTEP;
        D pp_x=add(startx, mul(cphi, muls(L,frac)));     // pusher position
        D pp_y=add(starty, mul(sphi, muls(L,frac)));
        D relx=sub(pp_x,dx), rely=sub(pp_y,dy);
        D dist=dsqrt(add(mul(relx,relx),mul(rely,rely)));
        D pen=dsoftplus(sub(dc(ph.R+ph.rp),dist));
        D fmag=muls(pen,ph.k);                            // normal magnitude k*pen
        D nx=muls(dvd(relx,dist),-1.f), ny=muls(dvd(rely,dist),-1.f);   // normal: pusher->disk-centre
        D ax=muls(dvd(relx,dist),ph.R), ay=muls(dvd(rely,dist),ph.R);   // contact arm = (rel/dist)*R
        // a pure normal force points at the COM -> ZERO torque.  The torque that
        // rotates the object comes from TANGENTIAL CONTACT FRICTION resisting the
        // pusher sliding across the surface (Mason's pushing mechanics).
        D vpush=muls(L, 1.f/(TSTEP*DT));                  // pusher speed along d
        D vpx=mul(cphi,vpush), vpy=mul(sphi,vpush);
        D vcx=sub(vx,mul(om,ay)), vcy=add(vy,mul(om,ax)); // disk surface velocity at contact
        D vrx=sub(vpx,vcx), vry=sub(vpy,vcy);             // relative slip velocity
        D tx=muls(ny,-1.f), ty=nx;                        // tangent = perp(normal)
        D vtan=add(mul(vrx,tx),mul(vry,ty));
        D ffric=mul(muls(fmag,0.55f), dtanh(muls(vtan,7.f)));  // mu*N*tanh(vtan/eps), smooth Coulomb
        D Fx=add(mul(fmag,nx), mul(ffric,tx));
        D Fy=add(mul(fmag,ny), mul(ffric,ty));
        D torque=sub(mul(ax,Fy),mul(ay,Fx));             // friction (tangential) part gives the moment
        // integrate (semi-implicit) with viscous plane friction
        vx=muls(add(vx, muls(Fx, DT/ph.m)), 1.f-DT*ph.clin);
        vy=muls(add(vy, muls(Fy, DT/ph.m)), 1.f-DT*ph.clin);
        om=muls(add(om, muls(torque, DT/ph.I)), 1.f-DT*ph.cang);
        dx=add(dx,muls(vx,DT)); dy=add(dy,muls(vy,DT)); dth=add(dth,muls(om,DT));
        (void)px;(void)py;
    }
    out[0]=dx; out[1]=dy; out[2]=dth; }

__device__ __host__ D loss_fn(const D* P,float tx,float ty,float tth,float wp,float wth){
    D out[3]; rollout(P,out);
    D ex=sub(out[0],dc(tx)), ey=sub(out[1],dc(ty));
    D lp=muls(add(mul(ex,ex),mul(ey,ey)),wp);
    D ct=dcos(sub(out[2],dc(tth)));                  // 1-cos heading error
    D lth=muls(sub(dc(1.f),ct),wth);
    return add(lp,lth); }

// optimise one (target) problem by gradient descent.  centre!=0 -> force b=0 (baseline).
__global__ void optimize_kernel(const float* __restrict__ TGT,int nP,int centre,
                                float* __restrict__ outParams,float* __restrict__ outErr){
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=nP)return;
    float tx=TGT[r*3],ty=TGT[r*3+1],tth=TGT[r*3+2];
    float dist=sqrtf(tx*tx+ty*ty)+1e-3f, head=atan2f(ty,tx);
    const float wp=3.f, wth=2.5f, lr=0.03f;
    // The off-centre push couples position & heading non-convexly (|rotation| is
    // non-monotone in b), and grad wrt b is huge near the contact kink while grad
    // wrt phi/L is small -- so use (a) a few diverse restarts, keeping the best, and
    // (b) per-dimension RMSprop so each parameter gets an appropriately scaled step.
    const int NR=6; const float binit[NR]={0.f,0.05f,-0.05f,0.10f,-0.10f,0.03f};
    const float dh[NR]={0.f,0.f,0.f,0.25f,-0.25f,-0.15f};
    float best[NP]; float bestL=1e30f;
    for(int rs=0; rs<(centre?1:NR); ++rs){
        float p[NP]={ centre?0.f:binit[rs], head+dh[rs], dist*1.6f+0.6f };
        float ms[NP]={0,0,0};
        for(int it=0;it<200;++it){
            D P[NP]; for(int i=0;i<NP;++i)P[i]=dvar(p[i],i);
            D L=loss_fn(P,tx,ty,tth,wp,wth);
            for(int i=0;i<NP;++i){ if(centre&&i==0)continue; float gi=L.g[i];
                ms[i]=0.9f*ms[i]+0.1f*gi*gi; p[i]-=lr*gi/(sqrtf(ms[i])+1e-5f); }
            p[2]=fmaxf(0.1f,p[2]);
        }
        D P[NP]; for(int i=0;i<NP;++i)P[i]=dc(p[i]); float Lf=loss_fn(P,tx,ty,tth,wp,wth).v;
        if(Lf<bestL){ bestL=Lf; for(int i=0;i<NP;++i)best[i]=p[i]; }
    }
    for(int i=0;i<NP;++i)outParams[r*NP+i]=best[i];
    D P[NP]; for(int i=0;i<NP;++i)P[i]=dc(best[i]); D out[3]; rollout(P,out);
    float ep=sqrtf((out[0].v-tx)*(out[0].v-tx)+(out[1].v-ty)*(out[1].v-ty));
    float eth=fabsf(atan2f(sinf(out[2].v-tth),cosf(out[2].v-tth)));
    outErr[r*2]=ep; outErr[r*2+1]=eth; }

// host replay (plain float) of the optimised push -> trajectories for the GIF.
static void replay(const float* p,float tx,float ty,float tth,
                   std::vector<std::array<float,2>>& pusher,std::vector<std::array<float,3>>& disk){
    Phys ph=phys(); float b=p[0],phi=p[1],L=p[2]; float cphi=cosf(phi),sphi=sinf(phi);
    float sx=cphi*-(ph.R+ph.rp+0.4f) + (-sphi)*b, sy=sphi*-(ph.R+ph.rp+0.4f)+cphi*b;
    float dx=0,dy=0,dth=0,vx=0,vy=0,om=0;
    pusher.clear(); disk.clear(); pusher.push_back({sx,sy}); disk.push_back({0,0,0});
    for(int s=0;s<TSTEP;++s){ float frac=(float)(s+1)/TSTEP; float ppx=sx+cphi*L*frac, ppy=sy+sphi*L*frac;
        float rx=ppx-dx, ry=ppy-dy; float dist=sqrtf(rx*rx+ry*ry)+1e-9f; float pen0=(ph.R+ph.rp)-dist; float be=0.03f; float pen=be*logf(1.f+expf(pen0/be));
        float fmag=ph.k*pen; float nx=-rx/dist, ny=-ry/dist;
        float ax=rx/dist*ph.R, ay=ry/dist*ph.R;
        float vpush=L/(TSTEP*DT); float vpx=cphi*vpush, vpy=sphi*vpush;
        float vcx=vx-om*ay, vcy=vy+om*ax; float vrx=vpx-vcx, vry=vpy-vcy;
        float tx=-ny, ty=nx; float vtan=vrx*tx+vry*ty;
        float ffric=fmag*0.55f*tanhf(vtan*7.f);
        float Fx=fmag*nx+ffric*tx, Fy=fmag*ny+ffric*ty; float tq=ax*Fy-ay*Fx;
        vx=(vx+Fx*DT/ph.m)*(1-DT*ph.clin); vy=(vy+Fy*DT/ph.m)*(1-DT*ph.clin); om=(om+tq*DT/ph.I)*(1-DT*ph.cang);
        dx+=vx*DT; dy+=vy*DT; dth+=om*DT; pusher.push_back({ppx,ppy}); disk.push_back({dx,dy,dth}); }
    (void)tx;(void)ty;(void)tth; }

// ============================ GIF ============================
static void render_gif(const std::vector<std::vector<std::array<float,2>>>& pushers,
                       const std::vector<std::vector<std::array<float,3>>>& disks,
                       const std::vector<std::array<float,3>>& tgt,const std::vector<std::array<float,2>>& org,int nShow){
    const int W=1000,H=1000; const float PX=70.f; Phys ph=phys();
    auto proj=[&](float x,float y,int&sx,int&sy){ sx=(int)(PX*x); sy=(int)(-PX*y); };
    if(system("mkdir -p tmp")!=0)std::fprintf(stderr,"warn\n");
    cv::VideoWriter video("tmp/gpu_diff_contact_push.avi",cv::VideoWriter::fourcc('M','J','P','G'),20,cv::Size(W,H));
    int F=TSTEP+1; const int HOLD=16;
    for(int f=0;f<F+HOLD;++f){ int k=std::min(f,F-1);
        cv::Mat img(H,W,CV_8UC3,cv::Scalar(24,24,30));
        for(int r=0;r<nShow;++r){ int ox,oy; proj(org[r][0],org[r][1],ox,oy); ox+=W/2; oy+=H/2;
            // target pose (ghost)
            float tgx=org[r][0]+tgt[r][0], tgy=org[r][1]+tgt[r][1]; int tx,ty; proj(tgx,tgy,tx,ty); tx+=W/2;ty+=H/2;
            cv::circle(img,{tx,ty},(int)(PX*ph.R),cv::Scalar(70,120,70),1,cv::LINE_AA);
            cv::line(img,{tx,ty},{tx+(int)(PX*ph.R*cosf(tgt[r][2])),ty-(int)(PX*ph.R*sinf(tgt[r][2]))},cv::Scalar(90,170,90),1,cv::LINE_AA);
            // disk
            float ddx=org[r][0]+disks[r][k][0], ddy=org[r][1]+disks[r][k][1], dth=disks[r][k][2];
            int dx,dy; proj(ddx,ddy,dx,dy); dx+=W/2;dy+=H/2;
            cv::circle(img,{dx,dy},(int)(PX*ph.R),cv::Scalar(200,150,70),-1,cv::LINE_AA);
            cv::circle(img,{dx,dy},(int)(PX*ph.R),cv::Scalar(230,190,120),1,cv::LINE_AA);
            cv::line(img,{dx,dy},{dx+(int)(PX*ph.R*cosf(dth)),dy-(int)(PX*ph.R*sinf(dth))},cv::Scalar(40,40,60),2,cv::LINE_AA);  // heading marker
            // pusher
            float ppx=org[r][0]+pushers[r][k][0], ppy=org[r][1]+pushers[r][k][1]; int qx,qy; proj(ppx,ppy,qx,qy); qx+=W/2;qy+=H/2;
            cv::circle(img,{qx,qy},(int)(PX*ph.rp),cv::Scalar(60,150,250),-1,cv::LINE_AA); }
        int px=24,py=46; auto put=[&](const std::string&s,int yy,double sc,cv::Scalar c,int th){cv::putText(img,s,cv::Point(px,yy),cv::FONT_HERSHEY_SIMPLEX,sc,c,th,cv::LINE_AA);};
        put("GPU differentiable contact: pushing",py,0.72,cv::Scalar(235,235,245),2);py+=30;
        put("autodiff-through-contact (dual numbers)",py,0.5,cv::Scalar(150,200,150),1);py+=22;
        put("gradient-optimised push -> target pose",py,0.5,cv::Scalar(150,200,150),1);py+=22;
        put("contact torque sets the heading",py,0.5,cv::Scalar(150,200,150),1);
        video.write(img); }
    video.release(); avi_to_gif("tmp/gpu_diff_contact_push.avi","gif/gpu_diff_contact_push.gif",20,820);
    std::printf("wrote gif/gpu_diff_contact_push.gif\n");
}

}  // namespace cudabot

int main(){
    using namespace cudabot;
    std::printf("=== GPU differentiable contact: autodiff-through-contact pushing ===\n");
    // --- autodiff-through-contact gradient check vs central differences ---
    { float p[NP]={0.08f,0.6f,2.6f}; float tx=1.2f,ty=0.7f,tth=0.3f, wp=3.f,wth=2.5f;
      D P[NP]; for(int i=0;i<NP;++i)P[i]=dvar(p[i],i); D L=loss_fn(P,tx,ty,tth,wp,wth);
      float maxrel=0; for(int i=0;i<NP;++i){ float e=1e-3f; float pp[NP],pm[NP];
        for(int j=0;j<NP;++j){pp[j]=p[j];pm[j]=p[j];} pp[i]+=e;pm[i]-=e;
        D Pp[NP],Pm[NP]; for(int j=0;j<NP;++j){Pp[j]=dc(pp[j]);Pm[j]=dc(pm[j]);}
        float fd=(loss_fn(Pp,tx,ty,tth,wp,wth).v-loss_fn(Pm,tx,ty,tth,wp,wth).v)/(2*e);
        float rel=std::fabs(L.g[i]-fd)/(std::fabs(fd)+1e-4f); maxrel=std::max(maxrel,rel);
        std::printf("  d loss/d p[%d]: autodiff=% .4f  finite-diff=% .4f\n",i,L.g[i],fd); }
      std::printf("  autodiff-vs-FD max relative error = %.2e  %s\n", maxrel, maxrel<1e-2f?"(gradients agree)":"(MISMATCH)"); }

    const int nP=512;
    std::mt19937 rng(7);
    // Reachable SE(2) targets: a single straight push spans only a coupled 3-DOF
    // manifold, so an arbitrary (position, heading) pair is usually NOT reachable.
    // We sample each target by forward-simulating a random push -- every target is
    // then achievable, and the question is whether the optimiser RECOVERS a push
    // that hits both position and heading (it must reason through the contact torque).
    std::uniform_real_distribution<float> br(-0.13f,0.13f), pr(0,6.2831853f), lr2(2.0f,3.4f);
    std::vector<float> TGT(nP*3);
    for(int r=0;r<nP;++r){ float p[3]={br(rng),pr(rng),lr2(rng)};
        std::vector<std::array<float,2>> pu; std::vector<std::array<float,3>> dk; replay(p,0,0,0,pu,dk);
        TGT[r*3]=dk.back()[0]; TGT[r*3+1]=dk.back()[1]; TGT[r*3+2]=dk.back()[2]; }

    float *dT,*dP,*dE,*dPc,*dEc;
    CUDA_CHECK(cudaMalloc(&dT,nP*3*sizeof(float))); CUDA_CHECK(cudaMemcpy(dT,TGT.data(),nP*3*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dP,nP*NP*sizeof(float)));CUDA_CHECK(cudaMalloc(&dE,nP*2*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPc,nP*NP*sizeof(float)));CUDA_CHECK(cudaMalloc(&dEc,nP*2*sizeof(float)));

    auto t0=std::chrono::high_resolution_clock::now();
    optimize_kernel<<<(nP+127)/128,128>>>(dT,nP,0,dP,dE);          // full (b free)
    optimize_kernel<<<(nP+127)/128,128>>>(dT,nP,1,dPc,dEc);        // centre-push baseline (b=0)
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    std::vector<float> E(nP*2),Ec(nP*2),P(nP*NP);
    CUDA_CHECK(cudaMemcpy(E.data(),dE,nP*2*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Ec.data(),dEc,nP*2*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P.data(),dP,nP*NP*sizeof(float),cudaMemcpyDeviceToHost));
    double mp=0,mt=0,mpc=0,mtc=0; for(int r=0;r<nP;++r){ mp+=E[r*2];mt+=E[r*2+1];mpc+=Ec[r*2];mtc+=Ec[r*2+1]; }
    mp/=nP;mt/=nP;mpc/=nP;mtc/=nP;
    std::printf("targets=%d  optimise 200 GD steps x6 restarts, autodiff-through-contact\n", nP);
    std::printf("\n                       mean pos err   mean heading err (deg)\n");
    std::printf("  contact torque (b free): %8.4f        %8.2f\n", mp, mt*57.2958f);
    std::printf("  centre-push (b=0):       %8.4f        %8.2f   <- cannot set heading\n", mpc, mtc*57.2958f);
    std::printf("wall=%.1f ms  (%d optimisations x2)\n", ms, nP);
    if(mp<0.15f && mt<0.26f && mtc>1.8f*mt)
        std::printf("RESULT: PASS -- autodiff-through-contact reaches the target pose; without contact torque the heading is unreachable.\n");
    else std::printf("RESULT: CHECK -- tolerance not met.\n");

    // GIF: replay a handful of optimised pushes laid out on a grid
    const int nShow=12; std::vector<std::vector<std::array<float,2>>> pushers(nShow);
    std::vector<std::vector<std::array<float,3>>> disks(nShow); std::vector<std::array<float,3>> tg(nShow); std::vector<std::array<float,2>> org(nShow);
    for(int r=0;r<nShow;++r){ replay(&P[r*NP],TGT[r*3],TGT[r*3+1],TGT[r*3+2],pushers[r],disks[r]);
        tg[r]={TGT[r*3],TGT[r*3+1],TGT[r*3+2]}; int gx=r%4, gy=r/4; org[r]={(gx-1.5f)*4.0f,(gy-1.0f)*4.0f}; }
    render_gif(pushers,disks,tg,org,nShow);

    cudaFree(dT);cudaFree(dP);cudaFree(dE);cudaFree(dPc);cudaFree(dEc);
    return 0;
}
