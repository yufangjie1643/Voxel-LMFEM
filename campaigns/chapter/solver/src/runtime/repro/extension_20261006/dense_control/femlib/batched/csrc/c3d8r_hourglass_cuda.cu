#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

template <typename complex_t, typename index_t, bool packed_hermitian>
__global__ void fft_spectral_multiply_3x3_kernel(
    const complex_t* __restrict__ symbol_inverse,
    const complex_t* __restrict__ rhs_spectrum,
    complex_t* __restrict__ output,
    const index_t frequencies_per_sample,
    const index_t total_frequencies,
    const int symbol_batch) {
  const index_t linear =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_frequencies) {
    return;
  }

  const index_t batch = linear / frequencies_per_sample;
  const index_t frequency = linear - batch * frequencies_per_sample;
  const index_t rhs_base = batch * 3 * frequencies_per_sample + frequency;
  const complex_t rhs0 = rhs_spectrum[rhs_base];
  const complex_t rhs1 = rhs_spectrum[rhs_base + frequencies_per_sample];
  const complex_t rhs2 = rhs_spectrum[rhs_base + 2 * frequencies_per_sample];
  const index_t symbol_sample = symbol_batch == 1 ? 0 : batch;
  const index_t symbol_base =
      symbol_sample * (packed_hermitian ? 6 : 9) * frequencies_per_sample
      + frequency;

  if (packed_hermitian) {
    // Upper-triangular order: 00, 01, 02, 11, 12, 22.  The lower triangle
    // is reconstructed in registers; this is exact for the real stencil's
    // Hermitian Fourier symbol and avoids retaining three redundant entries.
    const complex_t c00 = symbol_inverse[symbol_base];
    const complex_t c01 = symbol_inverse[symbol_base + frequencies_per_sample];
    const complex_t c02 = symbol_inverse[symbol_base + 2 * frequencies_per_sample];
    const complex_t c11 = symbol_inverse[symbol_base + 3 * frequencies_per_sample];
    const complex_t c12 = symbol_inverse[symbol_base + 4 * frequencies_per_sample];
    const complex_t c22 = symbol_inverse[symbol_base + 5 * frequencies_per_sample];
    const complex_t c10 = complex_t(c01.real(), -c01.imag());
    const complex_t c20 = complex_t(c02.real(), -c02.imag());
    const complex_t c21 = complex_t(c12.real(), -c12.imag());
    output[rhs_base] = c00 * rhs0 + c01 * rhs1 + c02 * rhs2;
    output[rhs_base + frequencies_per_sample] =
        c10 * rhs0 + c11 * rhs1 + c12 * rhs2;
    output[rhs_base + 2 * frequencies_per_sample] =
        c20 * rhs0 + c21 * rhs1 + c22 * rhs2;
    return;
  }

#pragma unroll
  for (int row = 0; row < 3; ++row) {
    const index_t row_base =
        symbol_base + static_cast<index_t>(row * 3) * frequencies_per_sample;
    const complex_t value =
        symbol_inverse[row_base] * rhs0
        + symbol_inverse[row_base + frequencies_per_sample] * rhs1
        + symbol_inverse[row_base + 2 * frequencies_per_sample] * rhs2;
    output[rhs_base + static_cast<index_t>(row) * frequencies_per_sample] =
        value;
  }
}

template <typename scalar_t>
__device__ __forceinline__ void corner_data(
    const int corner,
    int& ox,
    int& oy,
    int& oz,
    scalar_t& sx,
    scalar_t& sy,
    scalar_t& sz) {
  ox = 0;
  oy = 0;
  oz = 0;
  sx = scalar_t(-1);
  sy = scalar_t(-1);
  sz = scalar_t(-1);
  switch (corner) {
    case 1:
      ox = 1;
      sx = scalar_t(1);
      break;
    case 2:
      ox = 1;
      oy = 1;
      sx = scalar_t(1);
      sy = scalar_t(1);
      break;
    case 3:
      oy = 1;
      sy = scalar_t(1);
      break;
    case 4:
      oz = 1;
      sz = scalar_t(1);
      break;
    case 5:
      ox = 1;
      oz = 1;
      sx = scalar_t(1);
      sz = scalar_t(1);
      break;
    case 6:
      ox = 1;
      oy = 1;
      oz = 1;
      sx = scalar_t(1);
      sy = scalar_t(1);
      sz = scalar_t(1);
      break;
    case 7:
      oy = 1;
      oz = 1;
      sy = scalar_t(1);
      sz = scalar_t(1);
      break;
    default:
      break;
  }
}

template <typename scalar_t>
__device__ __forceinline__ void packed_constitutive_matvec(
    const scalar_t* __restrict__ packed,
    const scalar_t* __restrict__ vector,
    scalar_t* __restrict__ result) {
  // Matched dense36 consumer: same scalar operation order per row.
  result[0] = packed[0] * vector[0] + packed[1] * vector[1] + packed[2] * vector[2] + packed[3] * vector[3] + packed[4] * vector[4] + packed[5] * vector[5];
  result[1] = packed[6] * vector[0] + packed[7] * vector[1] + packed[8] * vector[2] + packed[9] * vector[3] + packed[10] * vector[4] + packed[11] * vector[5];
  result[2] = packed[12] * vector[0] + packed[13] * vector[1] + packed[14] * vector[2] + packed[15] * vector[3] + packed[16] * vector[4] + packed[17] * vector[5];
  result[3] = packed[18] * vector[0] + packed[19] * vector[1] + packed[20] * vector[2] + packed[21] * vector[3] + packed[22] * vector[4] + packed[23] * vector[5];
  result[4] = packed[24] * vector[0] + packed[25] * vector[1] + packed[26] * vector[2] + packed[27] * vector[3] + packed[28] * vector[4] + packed[29] * vector[5];
  result[5] = packed[30] * vector[0] + packed[31] * vector[1] + packed[32] * vector[2] + packed[33] * vector[3] + packed[34] * vector[4] + packed[35] * vector[5];
}

template <typename scalar_t, typename index_t, bool atomic_output>
__global__ void c3d8r_element_force_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ stiffness,
    const scalar_t* __restrict__ hourglass_scale,
    const scalar_t* __restrict__ geometry,
    scalar_t* __restrict__ element_force,
    const int nz,
    const int ny,
    const int nx,
    const index_t n_elem) {
  const index_t elem =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (elem >= n_elem) {
    return;
  }
  const index_t batch = static_cast<index_t>(blockIdx.y);

  const int i = static_cast<int>(elem % nx);
  const index_t yz = elem / nx;
  const int j = static_cast<int>(yz % ny);
  const int k = static_cast<int>(yz / ny);

  const scalar_t inv4dx = geometry[0];
  const scalar_t inv4dy = geometry[1];
  const scalar_t inv4dz = geometry[2];
  const scalar_t volume = geometry[3];
  const scalar_t hnorm = geometry[4];

  scalar_t strain[6] = {
      scalar_t(0), scalar_t(0), scalar_t(0),
      scalar_t(0), scalar_t(0), scalar_t(0)};
  scalar_t q[3][4] = {
      {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)},
      {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)},
      {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)}};

#pragma unroll
  for (int corner = 0; corner < 8; ++corner) {
    int ox;
    int oy;
    int oz;
    scalar_t sx;
    scalar_t sy;
    scalar_t sz;
    corner_data(corner, ox, oy, oz, sx, sy, sz);

    int ni = i + ox;
    int nj = j + oy;
    int nk = k + oz;
    if (ni == nx) {
      ni = 0;
    }
    if (nj == ny) {
      nj = 0;
    }
    if (nk == nz) {
      nk = 0;
    }
    const index_t node =
        (static_cast<index_t>(nk) * ny + nj) * nx + ni;
    const index_t u_base = (batch * n_elem + node) * 3;
    const scalar_t ux = u[u_base];
    const scalar_t uy = u[u_base + 1];
    const scalar_t uz = u[u_base + 2];

    const scalar_t dx = sx * inv4dx;
    const scalar_t dy = sy * inv4dy;
    const scalar_t dz = sz * inv4dz;
    strain[0] += dx * ux;
    strain[1] += dy * uy;
    strain[2] += dz * uz;
    strain[3] += dz * uy + dy * uz;
    strain[4] += dz * ux + dx * uz;
    strain[5] += dy * ux + dx * uy;

    const scalar_t h0 = sx * sy * hnorm;
    const scalar_t h1 = sy * sz * hnorm;
    const scalar_t h2 = sz * sx * hnorm;
    const scalar_t h3 = sx * sy * sz * hnorm;
    q[0][0] += h0 * ux;
    q[0][1] += h1 * ux;
    q[0][2] += h2 * ux;
    q[0][3] += h3 * ux;
    q[1][0] += h0 * uy;
    q[1][1] += h1 * uy;
    q[1][2] += h2 * uy;
    q[1][3] += h3 * uy;
    q[2][0] += h0 * uz;
    q[2][1] += h1 * uz;
    q[2][2] += h2 * uz;
    q[2][3] += h3 * uz;
  }

  const index_t stiffness_base = (batch * n_elem + elem) * 36;
  scalar_t stress[6] = {
      scalar_t(0), scalar_t(0), scalar_t(0),
      scalar_t(0), scalar_t(0), scalar_t(0)};
  packed_constitutive_matvec(
      stiffness + stiffness_base, strain, stress);

  const scalar_t scale = hourglass_scale[batch * n_elem + elem];
    const index_t force_base = (batch * n_elem + elem) * 24;
#pragma unroll
  for (int corner = 0; corner < 8; ++corner) {
    int ox;
    int oy;
    int oz;
    scalar_t sx;
    scalar_t sy;
    scalar_t sz;
    corner_data(corner, ox, oy, oz, sx, sy, sz);
    const scalar_t dx = sx * inv4dx;
    const scalar_t dy = sy * inv4dy;
    const scalar_t dz = sz * inv4dz;
    const scalar_t h0 = sx * sy * hnorm;
    const scalar_t h1 = sy * sz * hnorm;
    const scalar_t h2 = sz * sx * hnorm;
    const scalar_t h3 = sx * sy * sz * hnorm;

    const scalar_t fx =
        volume * (dx * stress[0] + dz * stress[4] + dy * stress[5])
        + scale * (
            h0 * q[0][0] + h1 * q[0][1] + h2 * q[0][2] + h3 * q[0][3]);
    const scalar_t fy =
        volume * (dy * stress[1] + dz * stress[3] + dx * stress[5])
        + scale * (
            h0 * q[1][0] + h1 * q[1][1] + h2 * q[1][2] + h3 * q[1][3]);
    const scalar_t fz =
        volume * (dz * stress[2] + dy * stress[3] + dx * stress[4])
        + scale * (
            h0 * q[2][0] + h1 * q[2][1] + h2 * q[2][2] + h3 * q[2][3]);

    if constexpr (atomic_output) {
      int ni = i + ox;
      int nj = j + oy;
      int nk = k + oz;
      if (ni == nx) {
        ni = 0;
      }
      if (nj == ny) {
        nj = 0;
      }
      if (nk == nz) {
        nk = 0;
      }
      const index_t node =
          (static_cast<index_t>(nk) * ny + nj) * nx + ni;
      const index_t output = (batch * n_elem + node) * 3;
      atomicAdd(element_force + output, fx);
      atomicAdd(element_force + output + 1, fy);
      atomicAdd(element_force + output + 2, fz);
    } else {
      const index_t output = force_base + corner * 3;
      element_force[output] = fx;
      element_force[output + 1] = fy;
      element_force[output + 2] = fz;
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void c3d8r_node_gather_kernel(
    const scalar_t* __restrict__ element_force,
    scalar_t* __restrict__ output,
    const int nz,
    const int ny,
    const int nx,
    const index_t n_elem) {
  const index_t node =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= n_elem) {
    return;
  }
  const index_t batch = static_cast<index_t>(blockIdx.y);

  const int i = static_cast<int>(node % nx);
  const index_t yz = node / nx;
  const int j = static_cast<int>(yz % ny);
  const int k = static_cast<int>(yz / ny);

  scalar_t force_x = scalar_t(0);
  scalar_t force_y = scalar_t(0);
  scalar_t force_z = scalar_t(0);
#pragma unroll
  for (int corner = 0; corner < 8; ++corner) {
    int ox;
    int oy;
    int oz;
    scalar_t sx;
    scalar_t sy;
    scalar_t sz;
    corner_data(corner, ox, oy, oz, sx, sy, sz);

    int ex = i - ox;
    int ey = j - oy;
    int ez = k - oz;
    if (ex < 0) {
      ex += nx;
    }
    if (ey < 0) {
      ey += ny;
    }
    if (ez < 0) {
      ez += nz;
    }
    const index_t elem =
        (static_cast<index_t>(ez) * ny + ey) * nx + ex;
    const index_t input = (batch * n_elem + elem) * 24 + corner * 3;
    force_x += element_force[input];
    force_y += element_force[input + 1];
    force_z += element_force[input + 2];
  }

  const index_t output_base = (batch * n_elem + node) * 3;
  output[output_base] = force_x;
  output[output_base + 1] = force_y;
  output[output_base + 2] = force_z;
}

template <typename scalar_t, typename index_t, bool atomic_output>
__global__ void c3d8r_rhs_kernel(
    const scalar_t* __restrict__ stiffness,
    const scalar_t* __restrict__ geometry,
    const scalar_t* __restrict__ macro_strain,
    scalar_t* __restrict__ element_force,
    const int nz,
    const int ny,
    const int nx,
    const index_t n_elem) {
  const index_t elem =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (elem >= n_elem) {
    return;
  }
  const index_t batch = static_cast<index_t>(blockIdx.y);

  const int i = static_cast<int>(elem % nx);
  const index_t yz = elem / nx;
  const int j = static_cast<int>(yz % ny);
  const int k = static_cast<int>(yz / ny);
  const scalar_t inv4dx = geometry[0];
  const scalar_t inv4dy = geometry[1];
  const scalar_t inv4dz = geometry[2];
  const scalar_t volume = geometry[3];

  const index_t stiffness_base = (batch * n_elem + elem) * 36;
  scalar_t stress[6] = {
      scalar_t(0), scalar_t(0), scalar_t(0),
      scalar_t(0), scalar_t(0), scalar_t(0)};
  packed_constitutive_matvec(
      stiffness + stiffness_base, macro_strain, stress);

  const index_t force_base = (batch * n_elem + elem) * 24;
#pragma unroll
  for (int corner = 0; corner < 8; ++corner) {
    int ox;
    int oy;
    int oz;
    scalar_t sx;
    scalar_t sy;
    scalar_t sz;
    corner_data(corner, ox, oy, oz, sx, sy, sz);
    const scalar_t dx = sx * inv4dx;
    const scalar_t dy = sy * inv4dy;
    const scalar_t dz = sz * inv4dz;
    const scalar_t fx =
        -volume * (dx * stress[0] + dz * stress[4] + dy * stress[5]);
    const scalar_t fy =
        -volume * (dy * stress[1] + dz * stress[3] + dx * stress[5]);
    const scalar_t fz =
        -volume * (dz * stress[2] + dy * stress[3] + dx * stress[4]);

    if constexpr (atomic_output) {
      int ni = i + ox;
      int nj = j + oy;
      int nk = k + oz;
      if (ni == nx) {
        ni = 0;
      }
      if (nj == ny) {
        nj = 0;
      }
      if (nk == nz) {
        nk = 0;
      }
      const index_t node =
          (static_cast<index_t>(nk) * ny + nj) * nx + ni;
      const index_t output = (batch * n_elem + node) * 3;
      atomicAdd(element_force + output, fx);
      atomicAdd(element_force + output + 1, fy);
      atomicAdd(element_force + output + 2, fz);
    } else {
      const index_t output = force_base + corner * 3;
      element_force[output] = fx;
      element_force[output + 1] = fy;
      element_force[output + 2] = fz;
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void c3d8r_avg_stress_partial_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ stiffness,
    const scalar_t* __restrict__ geometry,
    const scalar_t* __restrict__ macro_strain,
    scalar_t* __restrict__ partial,
    const int nz,
    const int ny,
    const int nx,
    const index_t n_elem,
    const int n_blocks) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  const index_t elem =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const index_t batch = static_cast<index_t>(blockIdx.y);
  scalar_t strain[6] = {
      macro_strain[0], macro_strain[1], macro_strain[2],
      macro_strain[3], macro_strain[4], macro_strain[5]};

  if (elem < n_elem) {
    const int i = static_cast<int>(elem % nx);
    const index_t yz = elem / nx;
    const int j = static_cast<int>(yz % ny);
    const int k = static_cast<int>(yz / ny);
    const scalar_t inv4dx = geometry[0];
    const scalar_t inv4dy = geometry[1];
    const scalar_t inv4dz = geometry[2];

#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
      int ox;
      int oy;
      int oz;
      scalar_t sx;
      scalar_t sy;
      scalar_t sz;
      corner_data(corner, ox, oy, oz, sx, sy, sz);
      int ni = i + ox;
      int nj = j + oy;
      int nk = k + oz;
      if (ni == nx) {
        ni = 0;
      }
      if (nj == ny) {
        nj = 0;
      }
      if (nk == nz) {
        nk = 0;
      }
      const index_t node =
          (static_cast<index_t>(nk) * ny + nj) * nx + ni;
      const index_t input = (batch * n_elem + node) * 3;
      const scalar_t ux = u[input];
      const scalar_t uy = u[input + 1];
      const scalar_t uz = u[input + 2];
      const scalar_t dx = sx * inv4dx;
      const scalar_t dy = sy * inv4dy;
      const scalar_t dz = sz * inv4dz;
      strain[0] += dx * ux;
      strain[1] += dy * uy;
      strain[2] += dz * uz;
      strain[3] += dz * uy + dy * uz;
      strain[4] += dz * ux + dx * uz;
      strain[5] += dy * ux + dx * uy;
    }
  }

  scalar_t stress[6] = {
      scalar_t(0), scalar_t(0), scalar_t(0),
      scalar_t(0), scalar_t(0), scalar_t(0)};
  if (elem < n_elem) {
    const index_t stiffness_base = (batch * n_elem + elem) * 36;
    packed_constitutive_matvec(
        stiffness + stiffness_base, strain, stress);
  }

#pragma unroll
  for (int component = 0; component < 6; ++component) {
    shared[component * blockDim.x + threadIdx.x] = stress[component];
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
#pragma unroll
      for (int component = 0; component < 6; ++component) {
        shared[component * blockDim.x + threadIdx.x] +=
            shared[component * blockDim.x + threadIdx.x + offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const index_t output =
        (batch * static_cast<index_t>(n_blocks) + blockIdx.x) * 6;
#pragma unroll
    for (int component = 0; component < 6; ++component) {
      partial[output + component] = shared[component * blockDim.x];
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void c3d8r_avg_stress_finalize_kernel(
    const scalar_t* __restrict__ partial,
    scalar_t* __restrict__ output,
    const int n_blocks,
    const index_t n_elem) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  const index_t batch = static_cast<index_t>(blockIdx.x);

#pragma unroll
  for (int component = 0; component < 6; ++component) {
    scalar_t sum = scalar_t(0);
    for (int block = threadIdx.x; block < n_blocks; block += blockDim.x) {
      sum += partial[
          (batch * static_cast<index_t>(n_blocks) + block) * 6 + component];
    }
    shared[component * blockDim.x + threadIdx.x] = sum;
  }
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
#pragma unroll
      for (int component = 0; component < 6; ++component) {
        shared[component * blockDim.x + threadIdx.x] +=
            shared[component * blockDim.x + threadIdx.x + offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int component = 0; component < 6; ++component) {
      output[batch * 6 + component] =
          shared[component * blockDim.x] / static_cast<scalar_t>(n_elem);
    }
  }
}

template <typename scalar_t, typename index_t>
void launch_c3d8r(
    const torch::Tensor& u_flat,
    const torch::Tensor& stiffness,
    const torch::Tensor& hourglass_scale,
    const torch::Tensor& geometry,
    torch::Tensor& element_force,
    torch::Tensor& output,
    const int nz,
    const int ny,
    const int nx,
    const int64_t n_elem,
    const int batch,
    const int threads,
    const int variant,
    cudaStream_t stream) {
  const dim3 gather_grid(
      static_cast<unsigned int>((n_elem + threads - 1) / threads),
      static_cast<unsigned int>(batch));
  if (variant == 0) {
    c3d8r_element_force_kernel<scalar_t, index_t, false>
        <<<gather_grid, threads, 0, stream>>>(
            u_flat.const_data_ptr<scalar_t>(),
            stiffness.const_data_ptr<scalar_t>(),
            hourglass_scale.const_data_ptr<scalar_t>(),
            geometry.const_data_ptr<scalar_t>(),
            element_force.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(n_elem));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    c3d8r_node_gather_kernel<scalar_t, index_t>
        <<<gather_grid, threads, 0, stream>>>(
            element_force.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(n_elem));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.mutable_data_ptr<scalar_t>(),
        0,
        output.numel() * sizeof(scalar_t),
        stream));
    c3d8r_element_force_kernel<scalar_t, index_t, true>
        <<<gather_grid, threads, 0, stream>>>(
            u_flat.const_data_ptr<scalar_t>(),
            stiffness.const_data_ptr<scalar_t>(),
            hourglass_scale.const_data_ptr<scalar_t>(),
            geometry.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(n_elem));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t, typename index_t>
void launch_c3d8r_rhs(
    const torch::Tensor& stiffness,
    const torch::Tensor& geometry,
    const torch::Tensor& macro_strain,
    torch::Tensor& element_force,
    torch::Tensor& output,
    const int nz,
    const int ny,
    const int nx,
    const int64_t n_elem,
    const int batch,
    const int threads,
    const int variant,
    cudaStream_t stream) {
  const dim3 grid(
      static_cast<unsigned int>((n_elem + threads - 1) / threads),
      static_cast<unsigned int>(batch));
  if (variant == 0) {
    c3d8r_rhs_kernel<scalar_t, index_t, false>
        <<<grid, threads, 0, stream>>>(
            stiffness.const_data_ptr<scalar_t>(),
            geometry.const_data_ptr<scalar_t>(),
            macro_strain.const_data_ptr<scalar_t>(),
            element_force.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(n_elem));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    c3d8r_node_gather_kernel<scalar_t, index_t>
        <<<grid, threads, 0, stream>>>(
            element_force.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(n_elem));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.mutable_data_ptr<scalar_t>(),
        0,
        output.numel() * sizeof(scalar_t),
        stream));
    c3d8r_rhs_kernel<scalar_t, index_t, true>
        <<<grid, threads, 0, stream>>>(
            stiffness.const_data_ptr<scalar_t>(),
            geometry.const_data_ptr<scalar_t>(),
            macro_strain.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(n_elem));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t, typename index_t>
void launch_c3d8r_avg_stress(
    const torch::Tensor& u_flat,
    const torch::Tensor& stiffness,
    const torch::Tensor& geometry,
    const torch::Tensor& macro_strain,
    torch::Tensor& partial,
    torch::Tensor& output,
    const int nz,
    const int ny,
    const int nx,
    const int64_t n_elem,
    const int batch,
    const int threads,
    const int n_blocks,
    cudaStream_t stream) {
  const dim3 partial_grid(
      static_cast<unsigned int>(n_blocks),
      static_cast<unsigned int>(batch));
  const size_t shared_bytes =
      static_cast<size_t>(threads) * 6 * sizeof(scalar_t);
  c3d8r_avg_stress_partial_kernel<scalar_t, index_t>
      <<<partial_grid, threads, shared_bytes, stream>>>(
          u_flat.const_data_ptr<scalar_t>(),
          stiffness.const_data_ptr<scalar_t>(),
          geometry.const_data_ptr<scalar_t>(),
          macro_strain.const_data_ptr<scalar_t>(),
          partial.mutable_data_ptr<scalar_t>(),
          nz,
          ny,
          nx,
          static_cast<index_t>(n_elem),
          n_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  c3d8r_avg_stress_finalize_kernel<scalar_t, index_t>
      <<<batch, threads, shared_bytes, stream>>>(
          partial.const_data_ptr<scalar_t>(),
          output.mutable_data_ptr<scalar_t>(),
          n_blocks,
          static_cast<index_t>(n_elem));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    const c10::Device& expected_device,
    const c10::ScalarType expected_type) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.device() == expected_device, name, " must be on ", expected_device);
  TORCH_CHECK(tensor.scalar_type() == expected_type, name, " must have the same dtype as u_flat");
}

}  // namespace

torch::Tensor c3d8r_fft_spectral_multiply_cuda(
    torch::Tensor symbol_inverse,
    torch::Tensor rhs_spectrum,
    int64_t block_size) {
  TORCH_CHECK(symbol_inverse.is_cuda(), "symbol_inverse must be a CUDA tensor");
  TORCH_CHECK(rhs_spectrum.is_cuda(), "rhs_spectrum must be a CUDA tensor");
  TORCH_CHECK(symbol_inverse.is_contiguous(), "symbol_inverse must be contiguous");
  TORCH_CHECK(rhs_spectrum.is_contiguous(), "rhs_spectrum must be contiguous");
  TORCH_CHECK(
      symbol_inverse.scalar_type() == torch::kComplexFloat
          || symbol_inverse.scalar_type() == torch::kComplexDouble,
      "symbol_inverse must use complex64 or complex128");
  TORCH_CHECK(
      rhs_spectrum.scalar_type() == symbol_inverse.scalar_type(),
      "rhs_spectrum must have the same dtype as symbol_inverse");
  TORCH_CHECK(
      rhs_spectrum.device() == symbol_inverse.device(),
      "rhs_spectrum must be on ", symbol_inverse.device());
  TORCH_CHECK(
      rhs_spectrum.dim() == 5,
      "rhs_spectrum must have shape (B, 3, Nz, Ny, Nx_spec)");
  TORCH_CHECK(
      rhs_spectrum.size(1) == 3,
      "rhs_spectrum must have three vector components");
  const bool packed_hermitian = symbol_inverse.dim() == 5;
  TORCH_CHECK(
      packed_hermitian || symbol_inverse.dim() == 6,
      "symbol_inverse must have shape (S,6,Nz,Ny,Nx_spec) or "
      "(S,3,3,Nz,Ny,Nx_spec)");
  if (packed_hermitian) {
    TORCH_CHECK(
        symbol_inverse.size(1) == 6,
        "packed symbol_inverse must have six upper-triangular entries");
    TORCH_CHECK(
        symbol_inverse.size(2) == rhs_spectrum.size(2)
            && symbol_inverse.size(3) == rhs_spectrum.size(3)
            && symbol_inverse.size(4) == rhs_spectrum.size(4),
        "packed symbol_inverse and rhs_spectrum frequency dimensions must match");
  } else {
    TORCH_CHECK(
        symbol_inverse.size(1) == 3 && symbol_inverse.size(2) == 3,
        "symbol_inverse must have 3x3 matrix dimensions");
    TORCH_CHECK(
        symbol_inverse.size(3) == rhs_spectrum.size(2)
            && symbol_inverse.size(4) == rhs_spectrum.size(3)
            && symbol_inverse.size(5) == rhs_spectrum.size(4),
        "symbol_inverse and rhs_spectrum frequency dimensions must match");
  }
  const int64_t batch = rhs_spectrum.size(0);
  const int64_t symbol_batch = symbol_inverse.size(0);
  TORCH_CHECK(batch > 0, "rhs_spectrum batch dimension must be positive");
  TORCH_CHECK(
      symbol_batch == 1 || symbol_batch == batch,
      "symbol_inverse batch dimension must be one or match rhs_spectrum");
  TORCH_CHECK(
      block_size == 64 || block_size == 128 || block_size == 256
          || block_size == 512,
      "block_size must be one of 64, 128, 256, or 512");

  const int64_t frequencies_per_sample =
      rhs_spectrum.size(2) * rhs_spectrum.size(3) * rhs_spectrum.size(4);
  const int64_t total_frequencies = batch * frequencies_per_sample;
  TORCH_CHECK(
      frequencies_per_sample > 0,
      "rhs_spectrum frequency dimensions must be positive");

  const c10::cuda::CUDAGuard device_guard(rhs_spectrum.device());
  auto output = torch::empty_like(rhs_spectrum);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(rhs_spectrum.get_device()).stream();
  const int64_t blocks = (total_frequencies + block_size - 1) / block_size;
  const bool needs_int64 =
      symbol_inverse.numel() > std::numeric_limits<int32_t>::max()
      || rhs_spectrum.numel() > std::numeric_limits<int32_t>::max();

  AT_DISPATCH_COMPLEX_TYPES(
      rhs_spectrum.scalar_type(), "c3d8r_fft_spectral_multiply_cuda", [&] {
        if (packed_hermitian) {
          if (needs_int64) {
            fft_spectral_multiply_3x3_kernel<scalar_t, int64_t, true>
                <<<blocks, block_size, 0, stream>>>(
                    symbol_inverse.const_data_ptr<scalar_t>(),
                    rhs_spectrum.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    frequencies_per_sample,
                    total_frequencies,
                    static_cast<int>(symbol_batch));
          } else {
            fft_spectral_multiply_3x3_kernel<scalar_t, int32_t, true>
                <<<blocks, block_size, 0, stream>>>(
                    symbol_inverse.const_data_ptr<scalar_t>(),
                    rhs_spectrum.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    static_cast<int32_t>(frequencies_per_sample),
                    static_cast<int32_t>(total_frequencies),
                    static_cast<int>(symbol_batch));
          }
        } else {
          if (needs_int64) {
            fft_spectral_multiply_3x3_kernel<scalar_t, int64_t, false>
                <<<blocks, block_size, 0, stream>>>(
                    symbol_inverse.const_data_ptr<scalar_t>(),
                    rhs_spectrum.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    frequencies_per_sample,
                    total_frequencies,
                    static_cast<int>(symbol_batch));
          } else {
            fft_spectral_multiply_3x3_kernel<scalar_t, int32_t, false>
                <<<blocks, block_size, 0, stream>>>(
                    symbol_inverse.const_data_ptr<scalar_t>(),
                    rhs_spectrum.const_data_ptr<scalar_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    static_cast<int32_t>(frequencies_per_sample),
                    static_cast<int32_t>(total_frequencies),
                    static_cast<int>(symbol_batch));
          }
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

torch::Tensor c3d8r_two_stage_cuda(
    torch::Tensor u_flat,
    torch::Tensor stiffness,
    torch::Tensor hourglass_scale,
    torch::Tensor geometry,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    int64_t block_size,
    int64_t variant) {
  TORCH_CHECK(u_flat.is_cuda(), "u_flat must be a CUDA tensor");
  TORCH_CHECK(u_flat.is_contiguous(), "u_flat must be contiguous");
  TORCH_CHECK(u_flat.dim() == 2, "u_flat must have shape (B, 3*N)");
  TORCH_CHECK(
      u_flat.scalar_type() == torch::kFloat32
          || u_flat.scalar_type() == torch::kFloat64,
      "u_flat must use float32 or float64");
  TORCH_CHECK(nz > 0 && ny > 0 && nx > 0, "grid dimensions must be positive");
  TORCH_CHECK(
      block_size == 64 || block_size == 128 || block_size == 256
          || block_size == 512,
      "block_size must be one of 64, 128, 256, or 512");
  TORCH_CHECK(
      variant == 0 || variant == 1,
      "variant must be 0 (thread) or 1 (atomic)");

  const int64_t n_elem = nz * ny * nx;
  const int64_t batch = u_flat.size(0);
  TORCH_CHECK(batch > 0 && batch <= 65535, "batch dimension is outside CUDA grid limits");
  TORCH_CHECK(u_flat.size(1) == 3 * n_elem, "u_flat has the wrong number of dofs");
  TORCH_CHECK(
      stiffness.sizes() == torch::IntArrayRef({batch, n_elem, 36}),
      "stiffness must have shape (B, N, 36)");
  TORCH_CHECK(
      hourglass_scale.sizes() == torch::IntArrayRef({batch, n_elem}),
      "hourglass_scale must have shape (B, N)");
  TORCH_CHECK(
      geometry.numel() == 5 && geometry.dim() == 1,
      "geometry must have shape (5,)");

  const auto device = u_flat.device();
  const auto scalar_type = u_flat.scalar_type();
  check_cuda_tensor(stiffness, "stiffness", device, scalar_type);
  check_cuda_tensor(hourglass_scale, "hourglass_scale", device, scalar_type);
  check_cuda_tensor(geometry, "geometry", device, scalar_type);

  const c10::cuda::CUDAGuard device_guard(device);
  auto element_force = variant == 1
      ? torch::Tensor()
      : torch::empty({batch, n_elem, 8, 3}, u_flat.options());
  auto output = torch::empty_like(u_flat);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(u_flat.get_device()).stream();
  const bool needs_int64 =
      batch * n_elem * 36 > std::numeric_limits<int32_t>::max();

  AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "c3d8r_two_stage_cuda", [&] {
        if (needs_int64) {
          launch_c3d8r<scalar_t, int64_t>(
              u_flat,
              stiffness,
              hourglass_scale,
              geometry,
              element_force,
              output,
              static_cast<int>(nz),
              static_cast<int>(ny),
              static_cast<int>(nx),
              n_elem,
              static_cast<int>(batch),
              static_cast<int>(block_size),
              static_cast<int>(variant),
              stream);
        } else {
          launch_c3d8r<scalar_t, int32_t>(
              u_flat,
              stiffness,
              hourglass_scale,
              geometry,
              element_force,
              output,
              static_cast<int>(nz),
              static_cast<int>(ny),
              static_cast<int>(nx),
              n_elem,
              static_cast<int>(batch),
              static_cast<int>(block_size),
              static_cast<int>(variant),
              stream);
        }
      });
  return output;
}

torch::Tensor c3d8r_rhs_cuda(
    torch::Tensor stiffness,
    torch::Tensor geometry,
    torch::Tensor macro_strain,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    int64_t block_size,
    int64_t variant) {
  TORCH_CHECK(stiffness.is_cuda(), "stiffness must be a CUDA tensor");
  TORCH_CHECK(stiffness.is_contiguous(), "stiffness must be contiguous");
  TORCH_CHECK(stiffness.dim() == 3, "stiffness must have shape (B, N, 36)");
  TORCH_CHECK(
      stiffness.scalar_type() == torch::kFloat32
          || stiffness.scalar_type() == torch::kFloat64,
      "stiffness must use float32 or float64");
  TORCH_CHECK(nz > 0 && ny > 0 && nx > 0, "grid dimensions must be positive");
  TORCH_CHECK(
      block_size == 64 || block_size == 128 || block_size == 256
          || block_size == 512,
      "block_size must be one of 64, 128, 256, or 512");
  TORCH_CHECK(
      variant == 0 || variant == 1,
      "variant must be 0 (thread) or 1 (atomic)");

  const int64_t n_elem = nz * ny * nx;
  const int64_t batch = stiffness.size(0);
  TORCH_CHECK(batch > 0 && batch <= 65535, "batch dimension is outside CUDA grid limits");
  TORCH_CHECK(
      stiffness.sizes() == torch::IntArrayRef({batch, n_elem, 36}),
      "stiffness must have shape (B, N, 36)");

  const auto device = stiffness.device();
  const auto scalar_type = stiffness.scalar_type();
  check_cuda_tensor(geometry, "geometry", device, scalar_type);
  check_cuda_tensor(macro_strain, "macro_strain", device, scalar_type);
  TORCH_CHECK(
      geometry.numel() == 5 && geometry.dim() == 1,
      "geometry must have shape (5,)");
  TORCH_CHECK(
      macro_strain.numel() == 6 && macro_strain.dim() == 1,
      "macro_strain must have shape (6,)");

  const c10::cuda::CUDAGuard device_guard(device);
  auto element_force = variant == 1
      ? torch::Tensor()
      : torch::empty({batch, n_elem, 8, 3}, stiffness.options());
  auto output = torch::empty({batch, 3 * n_elem}, stiffness.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(stiffness.get_device()).stream();
  const bool needs_int64 =
      batch * n_elem * 36 > std::numeric_limits<int32_t>::max();

  AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "c3d8r_rhs_cuda", [&] {
        if (needs_int64) {
          launch_c3d8r_rhs<scalar_t, int64_t>(
              stiffness,
              geometry,
              macro_strain,
              element_force,
              output,
              static_cast<int>(nz),
              static_cast<int>(ny),
              static_cast<int>(nx),
              n_elem,
              static_cast<int>(batch),
              static_cast<int>(block_size),
              static_cast<int>(variant),
              stream);
        } else {
          launch_c3d8r_rhs<scalar_t, int32_t>(
              stiffness,
              geometry,
              macro_strain,
              element_force,
              output,
              static_cast<int>(nz),
              static_cast<int>(ny),
              static_cast<int>(nx),
              n_elem,
              static_cast<int>(batch),
              static_cast<int>(block_size),
              static_cast<int>(variant),
              stream);
        }
      });
  return output;
}

torch::Tensor c3d8r_avg_stress_cuda(
    torch::Tensor u_flat,
    torch::Tensor stiffness,
    torch::Tensor geometry,
    torch::Tensor macro_strain,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    int64_t block_size) {
  TORCH_CHECK(u_flat.is_cuda(), "u_flat must be a CUDA tensor");
  TORCH_CHECK(u_flat.is_contiguous(), "u_flat must be contiguous");
  TORCH_CHECK(u_flat.dim() == 2, "u_flat must have shape (B, 3*N)");
  TORCH_CHECK(
      u_flat.scalar_type() == torch::kFloat32
          || u_flat.scalar_type() == torch::kFloat64,
      "u_flat must use float32 or float64");
  TORCH_CHECK(nz > 0 && ny > 0 && nx > 0, "grid dimensions must be positive");
  TORCH_CHECK(
      block_size == 64 || block_size == 128 || block_size == 256
          || block_size == 512,
      "block_size must be one of 64, 128, 256, or 512");

  const int64_t n_elem = nz * ny * nx;
  const int64_t batch = u_flat.size(0);
  TORCH_CHECK(batch > 0 && batch <= 65535, "batch dimension is outside CUDA grid limits");
  TORCH_CHECK(u_flat.size(1) == 3 * n_elem, "u_flat has the wrong number of dofs");
  TORCH_CHECK(
      stiffness.sizes() == torch::IntArrayRef({batch, n_elem, 36}),
      "stiffness must have shape (B, N, 36)");

  const auto device = u_flat.device();
  const auto scalar_type = u_flat.scalar_type();
  check_cuda_tensor(stiffness, "stiffness", device, scalar_type);
  check_cuda_tensor(geometry, "geometry", device, scalar_type);
  check_cuda_tensor(macro_strain, "macro_strain", device, scalar_type);
  TORCH_CHECK(
      geometry.numel() == 5 && geometry.dim() == 1,
      "geometry must have shape (5,)");
  TORCH_CHECK(
      macro_strain.numel() == 6 && macro_strain.dim() == 1,
      "macro_strain must have shape (6,)");

  const c10::cuda::CUDAGuard device_guard(device);
  const int n_blocks = static_cast<int>((n_elem + block_size - 1) / block_size);
  auto partial = torch::empty({batch, n_blocks, 6}, u_flat.options());
  auto output = torch::empty({batch, 6}, u_flat.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(u_flat.get_device()).stream();
  const bool needs_int64 =
      batch * n_elem * 36 > std::numeric_limits<int32_t>::max();

  AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "c3d8r_avg_stress_cuda", [&] {
        if (needs_int64) {
          launch_c3d8r_avg_stress<scalar_t, int64_t>(
              u_flat,
              stiffness,
              geometry,
              macro_strain,
              partial,
              output,
              static_cast<int>(nz),
              static_cast<int>(ny),
              static_cast<int>(nx),
              n_elem,
              static_cast<int>(batch),
              static_cast<int>(block_size),
              n_blocks,
              stream);
        } else {
          launch_c3d8r_avg_stress<scalar_t, int32_t>(
              u_flat,
              stiffness,
              geometry,
              macro_strain,
              partial,
              output,
              static_cast<int>(nz),
              static_cast<int>(ny),
              static_cast<int>(nx),
              n_elem,
              static_cast<int>(batch),
              static_cast<int>(block_size),
              n_blocks,
              stream);
        }
      });
  return output;
}
