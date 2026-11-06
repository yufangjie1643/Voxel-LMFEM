#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
__device__ __forceinline__ int packed(int i,int j){if(i>j){int t=i;i=j;j=t;}return 6*i-i*(i-1)/2+j-i;}
__device__ __forceinline__ int wrap(int i,int n){return i==n?0:i;}
__device__ __forceinline__ void structural_stress(const double* e,const double* frame,const double* p,bool yarn,double* s){
  const double t=e[0]+e[1]+e[2];
  if(!yarn){
    #pragma unroll
    for(int i=0;i<3;++i)s[i]=p[6]*t+2*p[7]*e[i];
    #pragma unroll
    for(int i=3;i<6;++i)s[i]=p[7]*e[i];
    return;
  }
  const double nx=frame[0],ny=frame[1],nz=frame[2];
  const double rx=e[0]*nx+.5*(e[5]*ny+e[4]*nz);
  const double ry=e[1]*ny+.5*(e[5]*nx+e[3]*nz);
  const double rz=e[2]*nz+.5*(e[4]*nx+e[3]*ny);
  const double q=nx*rx+ny*ry+nz*rz;
  const double a=p[0]*t+p[2]*q,b=p[2]*t+p[3]*q;
  double transverse=0.;
  #pragma unroll
  for(int i=0;i<6;++i)transverse+=frame[3+i]*e[i];
  transverse*=p[5];
  s[0]=a+2*p[1]*e[0]+b*nx*nx+4*p[4]*nx*rx+transverse*frame[3];
  s[1]=a+2*p[1]*e[1]+b*ny*ny+4*p[4]*ny*ry+transverse*frame[4];
  s[2]=a+2*p[1]*e[2]+b*nz*nz+4*p[4]*nz*rz+transverse*frame[5];
  s[3]=p[1]*e[3]+b*ny*nz+2*p[4]*(ny*rz+nz*ry)+transverse*frame[6];
  s[4]=p[1]*e[4]+b*nx*nz+2*p[4]*(nx*rz+nz*rx)+transverse*frame[7];
  s[5]=p[1]*e[5]+b*nx*ny+2*p[4]*(nx*ry+ny*rx)+transverse*frame[8];
}

#include "structural_modal.cuh"

template<int mode,bool energy>
__global__ void apply_kernel(const double* u,const double* c,const double* scales,
 const int32_t* active_index,const double* frames,const double* params,double* out,double* partial,
 int nz,int ny,int nx,double hx,double hy,double hz){
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,n=int64_t(nx)*ny*nz;
  double value=0.;
  if(e<n){
    const int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
    int64_t ids[8];double local[24],q[12];
    #pragma unroll
    for(int a=0;a<8;++a){
      ids[a]=(int64_t(wrap(z+((a>>2)&1),nz))*ny+wrap(y+((a>>1)&1),ny))*nx+wrap(x+(a&1),nx);
      #pragma unroll
      for(int d=0;d<3;++d)local[3*a+d]=u[3*ids[a]+d];
    }
    if constexpr(mode==2){
      const int index=active_index[e];double f[24];
      const double g[4]={1/(4*hx),1/(4*hy),1/(4*hz),hx*hy*hz};
      value=structural_modal_eval<energy>(local,index>=0?frames+9*index:frames,params,index>=0,g,f);
      #pragma unroll
      for(int a=0;a<8;++a){
        #pragma unroll
        for(int d=0;d<3;++d)atomicAdd(out+3*ids[a]+d,f[3*a+d]);
      }
    }else{
    #pragma unroll
    for(int d=0;d<3;++d){
      const double s0=local[d]+local[3+d],s1=local[6+d]+local[9+d];
      const double s2=local[12+d]+local[15+d],s3=local[18+d]+local[21+d];
      const double dx0=local[3+d]-local[d],dx1=local[9+d]-local[6+d];
      const double dx2=local[15+d]-local[12+d],dx3=local[21+d]-local[18+d];
      q[d]=(s0+s1)+(s2+s3);
      q[3+d]=(dx0+dx1)+(dx2+dx3);
      q[6+d]=(s1-s0)+(s3-s2);
      q[9+d]=(s2+s3)-(s0+s1);
    }
    const double gx=1/(4*hx),gy=1/(4*hy),gz=1/(4*hz),volume=hx*hy*hz;
    const double strain[6]={gx*q[3],gy*q[7],gz*q[11],gz*q[10]+gy*q[8],gz*q[9]+gx*q[5],gy*q[6]+gx*q[4]};
    double stress[6],scale;
    if constexpr(mode==1){
      const int index=active_index[e];
      structural_stress(strain,index>=0?frames+9*index:frames,params,index>=0,stress);
      scale=params[index>=0?9:8];
    }else{
      #pragma unroll
      for(int i=0;i<6;++i){stress[i]=0.;
        #pragma unroll
        for(int j=0;j<6;++j)stress[i]+=c[21*e+packed(i,j)]*strain[j];
      }scale=scales[e];
    }
    const double factor[3]={gx,gy,gz};const int which[9]={0,5,4,5,1,3,4,3,2};
    #pragma unroll
    for(int d=0;d<3;++d){
      const double center=-scale*q[d]/8;
      const double ax=volume*factor[0]*stress[which[d]]-scale*q[3+d]/8;
      const double ay=volume*factor[1]*stress[which[3+d]]-scale*q[6+d]/8;
      const double az=volume*factor[2]*stress[which[6+d]]-scale*q[9+d]/8;
      const double left=center-ax,right=center+ax;
      const double xy[4]={left-ay,right-ay,left+ay,right+ay};
      #pragma unroll
      for(int a=0;a<8;++a){
        const double f=(xy[a&3]+((a&4)?az:-az))+scale*local[3*a+d];
        atomicAdd(out+3*ids[a]+d,f);
        if constexpr(energy)value+=local[3*a+d]*f;
      }
    }
    }
  }
  if constexpr(energy){
    for(int d=16;d>0;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
    __shared__ double shared[4];
    if((threadIdx.x&31)==0)shared[threadIdx.x/32]=value;
    __syncthreads();
    if(threadIdx.x==0){double total=0;for(int i=0;i<4;++i)total+=shared[i];partial[blockIdx.x]=total;}
  }
}
}

std::vector<torch::Tensor> apply(torch::Tensor u,torch::Tensor c,torch::Tensor scales,torch::Tensor index,
 torch::Tensor frames,torch::Tensor params,int nz,int ny,int nx,double hx,double hy,double hz,int mode,bool energy){
  TORCH_CHECK(u.is_cuda()&&u.is_contiguous()&&u.scalar_type()==torch::kFloat64,"Contiguous CUDA FP64 u required");
  const int64_t n=int64_t(nx)*ny*nz;TORCH_CHECK(u.numel()==3*n,"B=1 periodic shape required");
  auto out=torch::zeros_like(u);auto partial=torch::empty({energy?(n+127)/128:0},u.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  #define LAUNCH(S,E) apply_kernel<S,E><<<(n+127)/128,128,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),scales.data_ptr<double>(),index.data_ptr<int32_t>(),frames.data_ptr<double>(),params.data_ptr<double>(),out.data_ptr<double>(),partial.data_ptr<double>(),nz,ny,nx,hx,hy,hz)
  if(mode==2){if(energy){LAUNCH(2,true);}else{LAUNCH(2,false);}}
  else if(mode==1){if(energy){LAUNCH(1,true);}else{LAUNCH(1,false);}}
  else{if(energy){LAUNCH(0,true);}else{LAUNCH(0,false);}}
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,partial};
}

namespace {
__device__ __forceinline__ void element_nodes(int64_t e,int nz,int ny,int nx,int64_t* ids){
  const int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
  #pragma unroll
  for(int a=0;a<8;++a)ids[a]=(int64_t(wrap(z+((a>>2)&1),nz))*ny+wrap(y+((a>>1)&1),ny))*nx+wrap(x+(a&1),nx);
}
__global__ void rhs_compact_kernel(const int32_t* active,const double* frames,const double* params,const double* macro,
 double* out,int64_t count,int nz,int ny,int nx,double hx,double hy,double hz){
  int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(j>=count)return;
  double s[6],base[6];structural_stress(macro,frames+9*j,params,true,s);structural_stress(macro,frames,params,false,base);
  #pragma unroll
  for(int d=0;d<6;++d)s[d]-=base[d];
  int64_t ids[8];element_nodes(active[j],nz,ny,nx,ids);double volume=hx*hy*hz;
  #pragma unroll
  for(int a=0;a<8;++a){
    double x=((a&1)?1.:-1.)/(4*hx),y=((a&2)?1.:-1.)/(4*hy),z=((a&4)?1.:-1.)/(4*hz);
    atomicAdd(out+3*ids[a],-volume*(x*s[0]+y*s[5]+z*s[4]));
    atomicAdd(out+3*ids[a]+1,-volume*(x*s[5]+y*s[1]+z*s[3]));
    atomicAdd(out+3*ids[a]+2,-volume*(x*s[4]+y*s[3]+z*s[2]));
  }
}
__global__ void stress_compact_kernel(const int32_t* active,const double* frames,const double* params,const double* macro,
 const double* u,double* out,int64_t count,int nz,int ny,int nx,double hx,double hy,double hz){
  int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;double s[6]={};
  if(j<count){
    int64_t ids[8];element_nodes(active[j],nz,ny,nx,ids);double strain[6];
    #pragma unroll
    for(int i=0;i<6;++i)strain[i]=macro[i];
    #pragma unroll
    for(int a=0;a<8;++a){
      double x=((a&1)?1.:-1.)/(4*hx),y=((a&2)?1.:-1.)/(4*hy),z=((a&4)?1.:-1.)/(4*hz);
      double ux=u[3*ids[a]],uy=u[3*ids[a]+1],uz=u[3*ids[a]+2];
      strain[0]+=x*ux;strain[1]+=y*uy;strain[2]+=z*uz;
      strain[3]+=z*uy+y*uz;strain[4]+=z*ux+x*uz;strain[5]+=y*ux+x*uy;
    }
    double base[6];structural_stress(strain,frames+9*j,params,true,s);structural_stress(strain,frames,params,false,base);
    #pragma unroll
    for(int i=0;i<6;++i)s[i]-=base[i];
  }
  __shared__ double sums[6*128];
  #pragma unroll
  for(int i=0;i<6;++i)sums[128*i+threadIdx.x]=s[i];
  __syncthreads();
  for(int d=64;d>0;d/=2){if(threadIdx.x<d){
    #pragma unroll
    for(int i=0;i<6;++i)sums[128*i+threadIdx.x]+=sums[128*i+threadIdx.x+d];
  }__syncthreads();}
  if(threadIdx.x==0){
    #pragma unroll
    for(int i=0;i<6;++i)out[i*gridDim.x+blockIdx.x]=sums[128*i];
  }
}
}

torch::Tensor rhs_compact(torch::Tensor active,torch::Tensor frames,torch::Tensor params,torch::Tensor macro,
 int nz,int ny,int nx,double hx,double hy,double hz){
  auto out=torch::zeros({1,int64_t(nx)*ny*nz*3},frames.options());int64_t count=active.numel();
  if(count)rhs_compact_kernel<<<(count+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(active.data_ptr<int32_t>(),frames.data_ptr<double>(),params.data_ptr<double>(),macro.data_ptr<double>(),out.data_ptr<double>(),count,nz,ny,nx,hx,hy,hz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor stress_compact(torch::Tensor active,torch::Tensor frames,torch::Tensor params,torch::Tensor macro,torch::Tensor u,
 int nz,int ny,int nx,double hx,double hy,double hz){
  int64_t count=active.numel();auto out=torch::empty({6,(count+127)/128},frames.options());
  if(count)stress_compact_kernel<<<(count+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(active.data_ptr<int32_t>(),frames.data_ptr<double>(),params.data_ptr<double>(),macro.data_ptr<double>(),u.data_ptr<double>(),out.data_ptr<double>(),count,nz,ny,nx,hx,hy,hz);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("rhs",&rhs_compact);m.def("stress",&stress_compact);}
