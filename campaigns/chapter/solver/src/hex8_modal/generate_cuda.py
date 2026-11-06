"""Translate the CPU-verified scalar modal arithmetic into native CUDA.

Usage: python generate_cuda.py
The generated device functions have identical algebra and lexicographic corners.
"""
import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent
CPU = HERE.parent/"modal_reference/generated_kernels.py"


def scalar_device(function):
    name = function.name.removesuffix("_local")
    lines = [f"__device__ __forceinline__ void {name}(const double* u, const double* c, double scale, const double* g, double* f) {{"]
    for node in function.body:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            lines.append(f"  const double {node.targets[0].id} = {ast.unparse(node.value)};")
        elif isinstance(node, ast.Return):
            assert isinstance(node.value, ast.Tuple) and len(node.value.elts) == 24
            lines.extend(f"  f[{i}] = {ast.unparse(value)};" for i, value in enumerate(node.value.elts))
        else:
            raise TypeError(ast.dump(node))
    lines.append("}")
    return "\n".join(lines)


HEADER = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
'''

FOOTER = r'''
template<bool fi>
__device__ __forceinline__ void evaluate(const double* u, const double* c, double scale,
                                        const double* g, double* f) {
  if constexpr(fi) fi_modal(u,c,scale,g,f);
  else hg_modal(u,c,scale,g,f);
}

__device__ __forceinline__ int wrap1(int value, int n) {
  return value < 0 ? value+n : (value >= n ? value-n : value);
}

template<bool fi>
__device__ __forceinline__ void element(const double* u,const double* c,const double* scales,
                                       int x,int y,int z,int nx,int ny,int nz,
                                       const double* g,double* f) {
  double local[24];
  #pragma unroll
  for(int a=0;a<8;++a) {
    const int xx=wrap1(x+(a&1),nx), yy=wrap1(y+((a>>1)&1),ny), zz=wrap1(z+((a>>2)&1),nz);
    const int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
    #pragma unroll
    for(int d=0;d<3;++d) local[3*a+d]=u[3*node+d];
  }
  const int64_t e=(int64_t(z)*ny+y)*nx+x;
  double scale=0.;
  if constexpr(!fi) scale=scales[e];
  evaluate<fi>(local,c+e*21,scale,g,f);
}

template<bool fi,bool atomic>
__global__ void element_kernel(const double* u,const double* c,const double* scales,
    double* output,int nx,int ny,int nz,double gx,double gy,double gz,double volume) {
  const int64_t e=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const int64_t n=int64_t(nx)*ny*nz;
  if(e>=n)return;
  const int x=e%nx,y=(e/nx)%ny,z=e/(int64_t(nx)*ny);
  const double g[4]={gx,gy,gz,volume};
  double f[24];
  element<fi>(u,c,scales,x,y,z,nx,ny,nz,g,f);
  #pragma unroll
  for(int a=0;a<8;++a) {
    if constexpr(atomic) {
      const int xx=wrap1(x+(a&1),nx),yy=wrap1(y+((a>>1)&1),ny),zz=wrap1(z+((a>>2)&1),nz);
      const int64_t node=(int64_t(zz)*ny+yy)*nx+xx;
      #pragma unroll
      for(int d=0;d<3;++d) atomicAdd(output+3*node+d,f[3*a+d]);
    } else {
      #pragma unroll
      for(int d=0;d<3;++d) output[24*e+3*a+d]=f[3*a+d];
    }
  }
}

__global__ void gather_kernel(const double* forces,double* output,int nx,int ny,int nz) {
  const int64_t node=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(node>=int64_t(nx)*ny*nz)return;
  const int x=node%nx,y=(node/nx)%ny,z=node/(int64_t(nx)*ny);
  double force[3]={0.,0.,0.};
  #pragma unroll
  for(int a=0;a<8;++a) {
    const int ex=wrap1(x-(a&1),nx),ey=wrap1(y-((a>>1)&1),ny),ez=wrap1(z-((a>>2)&1),nz);
    const int64_t e=(int64_t(ez)*ny+ey)*nx+ex;
    #pragma unroll
    for(int d=0;d<3;++d)force[d]+=forces[24*e+3*a+d];
  }
  #pragma unroll
  for(int d=0;d<3;++d)output[3*node+d]=force[d];
}

// Stream z planes through two shared force planes. Unlike a full 3D tile,
// shared storage is O(tx*ty), independent of tz. Every output node has one owner.
template<bool fi>
__global__ void stream_kernel(const double* u,const double* c,const double* scales,
    double* output,int nx,int ny,int nz,int tx,int ty,int tz,
    double gx,double gy,double gz,double volume) {
  extern __shared__ double buffer[];
  const int bx=blockIdx.x%((nx+tx-1)/tx);
  const int by=(blockIdx.x/((nx+tx-1)/tx))%((ny+ty-1)/ty);
  const int bz=blockIdx.x/(((nx+tx-1)/tx)*((ny+ty-1)/ty));
  const int sx=bx*tx,sy=by*ty,sz=bz*tz;
  const int dx=min(tx,nx-sx),dy=min(ty,ny-sy),dz=min(tz,nz-sz);
  const int plane=(dx+1)*(dy+1);
  const double g[4]={gx,gy,gz,volume};
  for(int layer=0;layer<=dz;++layer) {
    double* current=buffer+(layer&1)*plane*24;
    for(int e2=threadIdx.x;e2<plane;e2+=blockDim.x) {
      const int ex=e2%(dx+1),ey=e2/(dx+1);
      const int x=wrap1(sx+ex-1,nx),y=wrap1(sy+ey-1,ny),z=wrap1(sz+layer-1,nz);
      double f[24];
      element<fi>(u,c,scales,x,y,z,nx,ny,nz,g,f);
      #pragma unroll
      for(int j=0;j<24;++j)current[e2*24+j]=f[j];
    }
    __syncthreads();
    if(layer>0) {
      for(int i2=threadIdx.x;i2<dx*dy;i2+=blockDim.x) {
        const int ix=i2%dx,iy=i2/dx;
        double force[3]={0.,0.,0.};
        #pragma unroll
        for(int a=0;a<8;++a) {
          const int ex=ix+1-(a&1),ey=iy+1-((a>>1)&1),ez=layer-((a>>2)&1);
          const double* local=buffer+((ez&1)*plane+ey*(dx+1)+ex)*24+3*a;
          #pragma unroll
          for(int d=0;d<3;++d)force[d]+=local[d];
        }
        const int64_t node=(int64_t(sz+layer-1)*ny+sy+iy)*nx+sx+ix;
        #pragma unroll
        for(int d=0;d<3;++d)output[3*node+d]=force[d];
      }
    }
    __syncthreads();
  }
}

void check(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda()&&t.scalar_type()==torch::kFloat64&&t.is_contiguous(),"contiguous CUDA FP64 required");
}

template<bool fi>
void launch(const torch::Tensor& c,const torch::Tensor& scales,const torch::Tensor& u,
    torch::Tensor& output,torch::Tensor& forces,int nx,int ny,int nz,
    double hx,double hy,double hz,int variant,int tx,int ty,int tz,int block) {
  const int64_t n=int64_t(nx)*ny*nz;
  const int blocks=(n+block-1)/block;
  const double gx=1./(4*hx),gy=1./(4*hy),gz=1./(4*hz),volume=hx*hy*hz;
  const auto stream=at::cuda::getCurrentCUDAStream();
  if(variant==0) {
    TORCH_CHECK(forces.numel()==24*n,"two-stage force buffer shape");
    element_kernel<fi,false><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),
      scales.data_ptr<double>(),forces.data_ptr<double>(),nx,ny,nz,gx,gy,gz,volume);
    gather_kernel<<<blocks,block,0,stream>>>(forces.data_ptr<double>(),output.data_ptr<double>(),nx,ny,nz);
  } else if(variant==1) {
    output.zero_();
    element_kernel<fi,true><<<blocks,block,0,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),
      scales.data_ptr<double>(),output.data_ptr<double>(),nx,ny,nz,gx,gy,gz,volume);
  } else if(variant==2) {
    const int tiles=((nx+tx-1)/tx)*((ny+ty-1)/ty)*((nz+tz-1)/tz);
    const int shared=2*(tx+1)*(ty+1)*24*sizeof(double);
    if(shared>49152)C10_CUDA_CHECK(cudaFuncSetAttribute(stream_kernel<fi>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    stream_kernel<fi><<<tiles,block,shared,stream>>>(u.data_ptr<double>(),c.data_ptr<double>(),
      scales.data_ptr<double>(),output.data_ptr<double>(),nx,ny,nz,tx,ty,tz,gx,gy,gz,volume);
  } else TORCH_CHECK(false,"unknown variant");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

torch::Tensor apply(torch::Tensor c,torch::Tensor scales,torch::Tensor u,torch::Tensor forces,
    int nz,int ny,int nx,double hx,double hy,double hz,bool fi,int variant,
    int tx,int ty,int tz,int block) {
  check(c);check(scales);check(u);check(forces);
  c10::cuda::CUDAGuard guard(u.device());
  const int64_t n=int64_t(nx)*ny*nz;
  TORCH_CHECK(nx>=2&&ny>=2&&nz>=2&&c.numel()==21*n&&u.numel()==3*n,"B=1 shape required");
  TORCH_CHECK(fi||scales.numel()==n,"HG scale shape");
  TORCH_CHECK(hx>0&&hy>0&&hz>0&&tx>0&&ty>0&&tz>0,"positive geometry/tile required");
  TORCH_CHECK(block==64||block==128||block==256,"block must be 64, 128 or 256");
  auto output=torch::empty_like(u);
  if(fi)launch<true>(c,scales,u,output,forces,nx,ny,nz,hx,hy,hz,variant,tx,ty,tz,block);
  else launch<false>(c,scales,u,output,forces,nx,ny,nz,hx,hy,hz,variant,tx,ty,tz,block);
  return output;
}

pybind11::dict attributes(bool fi,int variant) {
  cudaFuncAttributes a;
  if(fi) {
    if(variant==2)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,stream_kernel<true>));
    else if(variant==1)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<true,true>));
    else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<true,false>));
  } else {
    if(variant==2)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,stream_kernel<false>));
    else if(variant==1)C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<false,true>));
    else C10_CUDA_CHECK(cudaFuncGetAttributes(&a,element_kernel<false,false>));
  }
  pybind11::dict r;r["registers_per_thread"]=a.numRegs;r["local_bytes_per_thread"]=a.localSizeBytes;
  r["static_shared_bytes"]=a.sharedSizeBytes;r["max_threads_per_block"]=a.maxThreadsPerBlock;
  return r;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("apply",&apply);m.def("attributes",&attributes);}
'''


def main():
    tree = ast.parse(CPU.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("hg_modal_local", "fi_modal_local")]
    assert len(functions) == 2
    output = HERE/"modal_kernels.cu"
    output.write_text(HEADER+"\n\n".join(scalar_device(f) for f in functions)+FOOTER, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
