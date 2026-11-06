#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Each thread owns one node and one tap/component. No atomics, no dense Ke.
__global__ void assemble_kernel(const double* C, const double* scale,
    const double* T, const double* H, const int* valid, const int* offsets,
    double* S, int nz, int ny, int nx, int taps, int64_t N, int64_t BN) {
  const int64_t linear=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(linear>=BN) return;
  const int64_t batch=linear/N, node=linear%N;
  const int i=node%nx, j=(node/nx)%ny, k=node/(nx*ny);
  const int coeff=blockIdx.y, tap=coeff/9, pq=coeff%9;
  double sum=0;
  #pragma unroll
  for(int a=0;a<8;++a) {
    if(!valid[tap*8+a]) continue;
    int x=i-offsets[a*3], y=j-offsets[a*3+1], z=k-offsets[a*3+2];
    if(x<0) x+=nx; if(y<0) y+=ny; if(z<0) z+=nz;
    const int64_t elem=batch*N+(int64_t(z)*ny+y)*nx+x;
    const double* weights=T+((tap*8+a)*9+pq)*21;
    double value=H[(tap*8+a)*9+pq]*scale[elem];
    #pragma unroll
    for(int c=0;c<21;++c) value+=weights[c]*C[elem*21+c];
    sum+=value;
  }
  S[(batch*(taps*9)+coeff)*N+node]=sum;
}

torch::Tensor assemble_hg_cuda(torch::Tensor C, torch::Tensor scale,
    torch::Tensor T, torch::Tensor H, torch::Tensor valid,
    torch::Tensor offsets, int64_t nz, int64_t ny, int64_t nx) {
  TORCH_CHECK(C.is_cuda() && C.scalar_type()==torch::kFloat64 && C.is_contiguous());
  TORCH_CHECK(C.dim()==3 && C.size(2)==21 && C.size(1)==nz*ny*nx);
  const int taps=T.size(0);
  TORCH_CHECK(taps==14 || taps==27);
  const c10::cuda::CUDAGuard guard(C.device());
  auto S=torch::empty({C.size(0),int64_t(taps),int64_t(9),nz,ny,nx},C.options());
  const int64_t N=nz*ny*nx, BN=C.size(0)*N;
  assemble_kernel<<<dim3((BN+127)/128,taps*9),128,0,at::cuda::getCurrentCUDAStream()>>>(
    C.data_ptr<double>(),scale.data_ptr<double>(),T.data_ptr<double>(),H.data_ptr<double>(),
    valid.data_ptr<int>(),offsets.data_ptr<int>(),S.data_ptr<double>(),nz,ny,nx,taps,N,BN);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return S;
}
