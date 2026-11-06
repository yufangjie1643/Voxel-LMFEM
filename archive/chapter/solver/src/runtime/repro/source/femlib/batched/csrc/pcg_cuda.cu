#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

template <typename scalar_t, typename index_t>
__global__ void pcg_dot_partial_kernel(
    const scalar_t* __restrict__ p,
    const scalar_t* __restrict__ Ap,
    scalar_t* __restrict__ partial,
    const index_t n,
    const int n_blocks) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  const index_t batch = static_cast<index_t>(blockIdx.y);
  const index_t index =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  scalar_t value = scalar_t(0);
  if (index < n) {
    const index_t offset = batch * n + index;
    value = p[offset] * Ap[offset];
  }
  shared[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial[batch * static_cast<index_t>(n_blocks) + blockIdx.x] = shared[0];
  }
}

template <typename scalar_t, typename index_t>
__global__ void pcg_finalize_alpha_kernel(
    const scalar_t* __restrict__ partial,
    const scalar_t* __restrict__ rz,
    scalar_t* __restrict__ state,
    const int n_blocks,
    const index_t batch_count) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  const index_t batch = static_cast<index_t>(blockIdx.x);
  scalar_t sum = scalar_t(0);
  for (int block = threadIdx.x; block < n_blocks; block += blockDim.x) {
    sum += partial[batch * static_cast<index_t>(n_blocks) + block];
  }
  shared[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const scalar_t pAp = shared[0];
    const scalar_t eps = scalar_t(1e-30);
    state[batch] =
        isfinite(pAp) && pAp > scalar_t(0)
        ? rz[batch] / (pAp > eps ? pAp : eps)
        : scalar_t(0);
  }
}

template <typename scalar_t, typename index_t>
__global__ void pcg_update_ur_partial_kernel(
    scalar_t* __restrict__ u,
    scalar_t* __restrict__ r,
    const scalar_t* __restrict__ p,
    const scalar_t* __restrict__ Ap,
    const scalar_t* __restrict__ M_inv,
    const scalar_t* __restrict__ state,
    scalar_t* __restrict__ partial_rr,
    scalar_t* __restrict__ partial_rz,
    const index_t n,
    const int n_blocks) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  scalar_t* shared_rr = shared;
  scalar_t* shared_rz = shared + blockDim.x;
  const index_t batch = static_cast<index_t>(blockIdx.y);
  const index_t index =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  scalar_t rr_value = scalar_t(0);
  scalar_t rz_value = scalar_t(0);
  if (index < n) {
    const index_t offset = batch * n + index;
    const scalar_t alpha = state[batch];
    const scalar_t updated_u = u[offset] + alpha * p[offset];
    const scalar_t updated_r = r[offset] - alpha * Ap[offset];
    u[offset] = updated_u;
    r[offset] = updated_r;
    rr_value = updated_r * updated_r;
    rz_value = updated_r * (updated_r * M_inv[offset]);
  }
  shared_rr[threadIdx.x] = rr_value;
  shared_rz[threadIdx.x] = rz_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_rr[threadIdx.x] += shared_rr[threadIdx.x + stride];
      shared_rz[threadIdx.x] += shared_rz[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const index_t partial_index =
        batch * static_cast<index_t>(n_blocks) + blockIdx.x;
    partial_rr[partial_index] = shared_rr[0];
    partial_rz[partial_index] = shared_rz[0];
  }
}

template <typename scalar_t, typename index_t>
__global__ void pcg_finalize_state_kernel(
    const scalar_t* __restrict__ partial_rr,
    const scalar_t* __restrict__ partial_rz,
    scalar_t* __restrict__ rz,
    const scalar_t* __restrict__ rhs_norm,
    scalar_t* __restrict__ best_res,
    scalar_t* __restrict__ res,
    scalar_t* __restrict__ state,
    const int n_blocks,
    const index_t batch_count) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  scalar_t* shared_rr = shared;
  scalar_t* shared_rz = shared + blockDim.x;
  const index_t batch = static_cast<index_t>(blockIdx.x);
  scalar_t rr_sum = scalar_t(0);
  scalar_t rz_sum = scalar_t(0);
  for (int block = threadIdx.x; block < n_blocks; block += blockDim.x) {
    const index_t index =
        batch * static_cast<index_t>(n_blocks) + block;
    rr_sum += partial_rr[index];
    rz_sum += partial_rz[index];
  }
  shared_rr[threadIdx.x] = rr_sum;
  shared_rz[threadIdx.x] = rz_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_rr[threadIdx.x] += shared_rr[threadIdx.x + stride];
      shared_rz[threadIdx.x] += shared_rz[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const scalar_t eps = scalar_t(1e-30);
    const scalar_t old_rz = rz[batch];
    const scalar_t new_rz = shared_rz[0];
    const scalar_t residual =
        sqrt(shared_rr[0] > scalar_t(0) ? shared_rr[0] : scalar_t(0))
        / rhs_norm[batch];
    const bool improved = residual < best_res[batch];
    if (improved) {
      best_res[batch] = residual;
    }
    res[batch] = residual;
    rz[batch] = new_rz;
    state[batch_count + batch] =
        new_rz / (old_rz > eps ? old_rz : eps);
    state[2 * batch_count + batch] =
        improved ? scalar_t(1) : scalar_t(0);
  }
}

template <typename scalar_t, typename index_t>
__global__ void pcg_update_p_best_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ r,
    scalar_t* __restrict__ p,
    const scalar_t* __restrict__ M_inv,
    scalar_t* __restrict__ best_u,
    const scalar_t* __restrict__ state,
    const index_t n,
    const index_t batch_count) {
  const index_t index =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }
  const index_t batch = static_cast<index_t>(blockIdx.y);
  const index_t offset = batch * n + index;
  p[offset] =
      r[offset] * M_inv[offset]
      + state[batch_count + batch] * p[offset];
  if (state[2 * batch_count + batch] != scalar_t(0)) {
    best_u[offset] = u[offset];
  }
}

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    const c10::Device& device,
    const c10::ScalarType scalar_type) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(
      tensor.scalar_type() == scalar_type,
      name,
      " must use the solver dtype");
}

template <typename scalar_t, typename index_t>
void launch_pcg_step(
    torch::Tensor& u,
    torch::Tensor& r,
    torch::Tensor& p,
    const torch::Tensor& Ap,
    const torch::Tensor& M_inv,
    torch::Tensor& rz,
    const torch::Tensor& rhs_norm,
    torch::Tensor& best_u,
    torch::Tensor& best_res,
    torch::Tensor& res,
    torch::Tensor& workspace,
    torch::Tensor& state,
    const int64_t n,
    const int batch,
    const int n_blocks,
    const int threads,
    cudaStream_t stream) {
  const dim3 vector_grid(
      static_cast<unsigned int>(n_blocks),
      static_cast<unsigned int>(batch));
  const size_t scalar_shared =
      static_cast<size_t>(threads) * sizeof(scalar_t);
  const size_t pair_shared = 2 * scalar_shared;
  scalar_t* workspace_ptr = workspace.mutable_data_ptr<scalar_t>();
  const int64_t partial_stride = static_cast<int64_t>(batch) * n_blocks;
  scalar_t* partial_pAp = workspace_ptr;
  scalar_t* partial_rr = workspace_ptr + partial_stride;
  scalar_t* partial_rz = workspace_ptr + 2 * partial_stride;

  pcg_dot_partial_kernel<scalar_t, index_t>
      <<<vector_grid, threads, scalar_shared, stream>>>(
          p.const_data_ptr<scalar_t>(),
          Ap.const_data_ptr<scalar_t>(),
          partial_pAp,
          static_cast<index_t>(n),
          n_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pcg_finalize_alpha_kernel<scalar_t, index_t>
      <<<batch, threads, scalar_shared, stream>>>(
          partial_pAp,
          rz.const_data_ptr<scalar_t>(),
          state.mutable_data_ptr<scalar_t>(),
          n_blocks,
          static_cast<index_t>(batch));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pcg_update_ur_partial_kernel<scalar_t, index_t>
      <<<vector_grid, threads, pair_shared, stream>>>(
          u.mutable_data_ptr<scalar_t>(),
          r.mutable_data_ptr<scalar_t>(),
          p.const_data_ptr<scalar_t>(),
          Ap.const_data_ptr<scalar_t>(),
          M_inv.const_data_ptr<scalar_t>(),
          state.const_data_ptr<scalar_t>(),
          partial_rr,
          partial_rz,
          static_cast<index_t>(n),
          n_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pcg_finalize_state_kernel<scalar_t, index_t>
      <<<batch, threads, pair_shared, stream>>>(
          partial_rr,
          partial_rz,
          rz.mutable_data_ptr<scalar_t>(),
          rhs_norm.const_data_ptr<scalar_t>(),
          best_res.mutable_data_ptr<scalar_t>(),
          res.mutable_data_ptr<scalar_t>(),
          state.mutable_data_ptr<scalar_t>(),
          n_blocks,
          static_cast<index_t>(batch));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  pcg_update_p_best_kernel<scalar_t, index_t>
      <<<vector_grid, threads, 0, stream>>>(
          u.const_data_ptr<scalar_t>(),
          r.const_data_ptr<scalar_t>(),
          p.mutable_data_ptr<scalar_t>(),
          M_inv.const_data_ptr<scalar_t>(),
          best_u.mutable_data_ptr<scalar_t>(),
          state.const_data_ptr<scalar_t>(),
          static_cast<index_t>(n),
          static_cast<index_t>(batch));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor pcg_jacobi_step_cuda(
    torch::Tensor u,
    torch::Tensor r,
    torch::Tensor p,
    torch::Tensor Ap,
    torch::Tensor M_inv,
    torch::Tensor rz,
    torch::Tensor rhs_norm,
    torch::Tensor best_u,
    torch::Tensor best_res,
    torch::Tensor res,
    torch::Tensor workspace,
    torch::Tensor state,
    int64_t block_size) {
  TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
  TORCH_CHECK(u.is_contiguous(), "u must be contiguous");
  TORCH_CHECK(u.dim() == 2, "u must have shape (B, N)");
  TORCH_CHECK(
      u.scalar_type() == torch::kFloat32 || u.scalar_type() == torch::kFloat64,
      "u must use float32 or float64");
  TORCH_CHECK(
      block_size == 64 || block_size == 128 || block_size == 256
          || block_size == 512,
      "block_size must be one of 64, 128, 256, or 512");

  const int64_t batch = u.size(0);
  const int64_t n = u.size(1);
  TORCH_CHECK(batch > 0 && batch <= 65535, "batch dimension is outside CUDA grid limits");
  TORCH_CHECK(n > 0, "solver vector length must be positive");
  const int64_t n_blocks = (n + block_size - 1) / block_size;
  TORCH_CHECK(
      n_blocks <= std::numeric_limits<int>::max(),
      "solver vector is too large for the CUDA grid");

  const auto device = u.device();
  const auto scalar_type = u.scalar_type();
  check_tensor(r, "r", device, scalar_type);
  check_tensor(p, "p", device, scalar_type);
  check_tensor(Ap, "Ap", device, scalar_type);
  check_tensor(M_inv, "M_inv", device, scalar_type);
  check_tensor(rz, "rz", device, scalar_type);
  check_tensor(rhs_norm, "rhs_norm", device, scalar_type);
  check_tensor(best_u, "best_u", device, scalar_type);
  check_tensor(best_res, "best_res", device, scalar_type);
  check_tensor(res, "res", device, scalar_type);
  check_tensor(workspace, "workspace", device, scalar_type);
  check_tensor(state, "state", device, scalar_type);
  TORCH_CHECK(r.sizes() == u.sizes(), "r must match u");
  TORCH_CHECK(p.sizes() == u.sizes(), "p must match u");
  TORCH_CHECK(Ap.sizes() == u.sizes(), "Ap must match u");
  TORCH_CHECK(M_inv.sizes() == u.sizes(), "M_inv must match u");
  TORCH_CHECK(best_u.sizes() == u.sizes(), "best_u must match u");
  TORCH_CHECK(rz.numel() == batch, "rz must have shape (B,)");
  TORCH_CHECK(rhs_norm.numel() == batch, "rhs_norm must have shape (B,)");
  TORCH_CHECK(best_res.numel() == batch, "best_res must have shape (B,)");
  TORCH_CHECK(res.numel() == batch, "res must have shape (B,)");
  TORCH_CHECK(
      workspace.sizes() == torch::IntArrayRef({3, batch, n_blocks}),
      "workspace must have shape (3, B, ceil(N / block_size))");
  TORCH_CHECK(
      state.sizes() == torch::IntArrayRef({3, batch}),
      "state must have shape (3, B)");

  const c10::cuda::CUDAGuard device_guard(device);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(u.get_device()).stream();
  const bool needs_int64 = batch * n > std::numeric_limits<int32_t>::max();
  AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "pcg_jacobi_step_cuda", [&] {
        if (needs_int64) {
          launch_pcg_step<scalar_t, int64_t>(
              u,
              r,
              p,
              Ap,
              M_inv,
              rz,
              rhs_norm,
              best_u,
              best_res,
              res,
              workspace,
              state,
              n,
              static_cast<int>(batch),
              static_cast<int>(n_blocks),
              static_cast<int>(block_size),
              stream);
        } else {
          launch_pcg_step<scalar_t, int32_t>(
              u,
              r,
              p,
              Ap,
              M_inv,
              rz,
              rhs_norm,
              best_u,
              best_res,
              res,
              workspace,
              state,
              n,
              static_cast<int>(batch),
              static_cast<int>(n_blocks),
              static_cast<int>(block_size),
              stream);
        }
      });
  return res;
}
