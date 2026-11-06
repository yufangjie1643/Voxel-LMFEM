#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int64_t kMaxGridX = 2147483647LL;

void check_block_size(const int64_t block_size) {
  TORCH_CHECK(
      block_size == 64 || block_size == 128 || block_size == 256
          || block_size == 512,
      "block_size must be one of 64, 128, 256, or 512");
}

int64_t checked_node_count(
    const int64_t nz,
    const int64_t ny,
    const int64_t nx) {
  TORCH_CHECK(nz > 0 && ny > 0 && nx > 0, "grid dimensions must be positive");
  TORCH_CHECK(
      nz <= std::numeric_limits<int>::max()
          && ny <= std::numeric_limits<int>::max()
          && nx <= std::numeric_limits<int>::max(),
      "grid dimensions exceed the CUDA kernel integer range");
  TORCH_CHECK(
      nz <= std::numeric_limits<int64_t>::max() / ny
          && nz * ny <= std::numeric_limits<int64_t>::max() / nx,
      "grid node count overflows int64");
  return nz * ny * nx;
}

int64_t checked_blocks(const int64_t work, const int threads) {
  TORCH_CHECK(work > 0, "CUDA launch work size must be positive");
  const int64_t blocks = (work + threads - 1) / threads;
  TORCH_CHECK(blocks <= kMaxGridX, "CUDA launch exceeds the grid-x limit");
  return blocks;
}

void check_cuda_tensor(
    const torch::Tensor& tensor,
    const char* name,
    const c10::Device& expected_device,
    const c10::ScalarType expected_type) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.device() == expected_device,
      name,
      " must be on ",
      expected_device);
  TORCH_CHECK(
      tensor.scalar_type() == expected_type,
      name,
      " must have dtype ",
      expected_type);
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

__device__ __forceinline__ int packed_voigt_index(int row, int column) {
  if (row > column) {
    const int swap = row;
    row = column;
    column = swap;
  }
  constexpr int offsets[6] = {0, 6, 11, 15, 18, 20};
  return offsets[row] + column - row;
}

template <typename scalar_t, typename index_t, bool compressed>
__global__ void c3d8_stencil_apply_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ stencil,
    scalar_t* __restrict__ output,
    const int nz,
    const int ny,
    const int nx,
    const index_t nodes_per_sample,
    const index_t total_nodes) {
  const index_t linear =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_nodes) {
    return;
  }

  const index_t batch = linear / nodes_per_sample;
  const index_t node = linear - batch * nodes_per_sample;
  const int i = static_cast<int>(node % nx);
  const index_t yz = node / nx;
  const int j = static_cast<int>(yz % ny);
  const int k = static_cast<int>(yz / ny);

  scalar_t force_x = scalar_t(0);
  scalar_t force_y = scalar_t(0);
  scalar_t force_z = scalar_t(0);

  const int stencil_taps = compressed ? 14 : 27;
#pragma unroll
  for (int tap = 0; tap < stencil_taps; ++tap) {
    const int ix = tap % 3;
    const int iy = (tap / 3) % 3;
    const int iz = tap / 9;
    int ni = i + ix - 1;
    int nj = j + iy - 1;
    int nk = k + iz - 1;
    if (ni < 0) {
      ni += nx;
    } else if (ni >= nx) {
      ni -= nx;
    }
    if (nj < 0) {
      nj += ny;
    } else if (nj >= ny) {
      nj -= ny;
    }
    if (nk < 0) {
      nk += nz;
    } else if (nk >= nz) {
      nk -= nz;
    }

    const index_t neighbor =
        (static_cast<index_t>(nk) * ny + nj) * nx + ni;
    const index_t u_base =
        (batch * nodes_per_sample + neighbor) * index_t(3);
    const scalar_t ux = u[u_base];
    const scalar_t uy = u[u_base + 1];
    const scalar_t uz = u[u_base + 2];

    const index_t stencil_base =
        (batch * index_t(compressed ? 126 : 243)
             + static_cast<index_t>(tap * 9))
            * nodes_per_sample
        + node;
    const scalar_t s00 = stencil[stencil_base];
    const scalar_t s01 = stencil[stencil_base + nodes_per_sample];
    const scalar_t s02 = stencil[stencil_base + index_t(2) * nodes_per_sample];
    const scalar_t s10 = stencil[stencil_base + index_t(3) * nodes_per_sample];
    const scalar_t s11 = stencil[stencil_base + index_t(4) * nodes_per_sample];
    const scalar_t s12 = stencil[stencil_base + index_t(5) * nodes_per_sample];
    const scalar_t s20 = stencil[stencil_base + index_t(6) * nodes_per_sample];
    const scalar_t s21 = stencil[stencil_base + index_t(7) * nodes_per_sample];
    const scalar_t s22 = stencil[stencil_base + index_t(8) * nodes_per_sample];

    force_x += s00 * ux + s01 * uy + s02 * uz;
    force_y += s10 * ux + s11 * uy + s12 * uz;
    force_z += s20 * ux + s21 * uy + s22 * uz;

    if (compressed && tap != 13) {
      int mi = i - (ix - 1);
      int mj = j - (iy - 1);
      int mk = k - (iz - 1);
      if (mi < 0) {
        mi += nx;
      } else if (mi >= nx) {
        mi -= nx;
      }
      if (mj < 0) {
        mj += ny;
      } else if (mj >= ny) {
        mj -= ny;
      }
      if (mk < 0) {
        mk += nz;
      } else if (mk >= nz) {
        mk -= nz;
      }
      const index_t negative_neighbor =
          (static_cast<index_t>(mk) * ny + mj) * nx + mi;
      const index_t negative_u_base =
          (batch * nodes_per_sample + negative_neighbor) * index_t(3);
      const scalar_t nxv = u[negative_u_base];
      const scalar_t nyv = u[negative_u_base + 1];
      const scalar_t nzv = u[negative_u_base + 2];
      const index_t negative_stencil_base =
          (batch * index_t(126) + static_cast<index_t>(tap * 9))
              * nodes_per_sample
          + negative_neighbor;
      const scalar_t n00 = stencil[negative_stencil_base];
      const scalar_t n01 = stencil[negative_stencil_base + nodes_per_sample];
      const scalar_t n02 = stencil[negative_stencil_base + index_t(2) * nodes_per_sample];
      const scalar_t n10 = stencil[negative_stencil_base + index_t(3) * nodes_per_sample];
      const scalar_t n11 = stencil[negative_stencil_base + index_t(4) * nodes_per_sample];
      const scalar_t n12 = stencil[negative_stencil_base + index_t(5) * nodes_per_sample];
      const scalar_t n20 = stencil[negative_stencil_base + index_t(6) * nodes_per_sample];
      const scalar_t n21 = stencil[negative_stencil_base + index_t(7) * nodes_per_sample];
      const scalar_t n22 = stencil[negative_stencil_base + index_t(8) * nodes_per_sample];
      force_x += n00 * nxv + n10 * nyv + n20 * nzv;
      force_y += n01 * nxv + n11 * nyv + n21 * nzv;
      force_z += n02 * nxv + n12 * nyv + n22 * nzv;
    }
  }

  const index_t output_base =
      (batch * nodes_per_sample + node) * index_t(3);
  output[output_base] = force_x;
  output[output_base + 1] = force_y;
  output[output_base + 2] = force_z;
}

template <typename scalar_t, typename index_t, bool atomic_output, bool packed>
__global__ void c3d8_rhs_element_kernel(
    const scalar_t* __restrict__ stiffness,
    const scalar_t* __restrict__ geometry,
    const scalar_t* __restrict__ macro_strain,
    scalar_t* __restrict__ element_force,
    const int nz,
    const int ny,
    const int nx,
    const index_t nodes_per_sample,
    const index_t total_elements) {
  const index_t linear =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_elements) {
    return;
  }
  const index_t batch = linear / nodes_per_sample;
  const index_t elem = linear - batch * nodes_per_sample;
  const int i = static_cast<int>(elem % nx);
  const index_t yz = elem / nx;
  const int j = static_cast<int>(yz % ny);
  const int k = static_cast<int>(yz / ny);

  const scalar_t inv4dx = geometry[0];
  const scalar_t inv4dy = geometry[1];
  const scalar_t inv4dz = geometry[2];
  const scalar_t volume = geometry[3];

  const index_t stiffness_base =
      (batch * nodes_per_sample + elem) * index_t(packed ? 21 : 36);
  scalar_t stress[6] = {
      scalar_t(0), scalar_t(0), scalar_t(0),
      scalar_t(0), scalar_t(0), scalar_t(0)};
#pragma unroll
  for (int row = 0; row < 6; ++row) {
#pragma unroll
    for (int column = 0; column < 6; ++column) {
      const int index = packed ? packed_voigt_index(row, column)
                               : row * 6 + column;
      stress[row] += stiffness[stiffness_base + index]
          * macro_strain[column];
    }
  }

  const index_t force_base =
      (batch * nodes_per_sample + elem) * index_t(24);
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
      const index_t output =
          (batch * nodes_per_sample + node) * index_t(3);
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
__global__ void c3d8_node_gather_kernel(
    const scalar_t* __restrict__ element_force,
    scalar_t* __restrict__ output,
    const int nz,
    const int ny,
    const int nx,
    const index_t nodes_per_sample,
    const index_t total_nodes) {
  const index_t linear =
      static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_nodes) {
    return;
  }
  const index_t batch = linear / nodes_per_sample;
  const index_t node = linear - batch * nodes_per_sample;
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
    const index_t input =
        (batch * nodes_per_sample + elem) * index_t(24) + corner * 3;
    force_x += element_force[input];
    force_y += element_force[input + 1];
    force_z += element_force[input + 2];
  }

  const index_t output_base =
      (batch * nodes_per_sample + node) * index_t(3);
  output[output_base] = force_x;
  output[output_base + 1] = force_y;
  output[output_base + 2] = force_z;
}

template <typename scalar_t, typename index_t, bool packed>
__global__ void c3d8_avg_stress_partial_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ stiffness,
    const scalar_t* __restrict__ geometry,
    const scalar_t* __restrict__ macro_strain,
    scalar_t* __restrict__ partial,
    const int nz,
    const int ny,
    const int nx,
    const index_t nodes_per_sample,
    const index_t blocks_per_sample,
    const index_t total_partial_blocks) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  const index_t linear_block = static_cast<index_t>(blockIdx.x);
  if (linear_block >= total_partial_blocks) {
    return;
  }
  const index_t batch = linear_block / blocks_per_sample;
  const index_t sample_block =
      linear_block - batch * blocks_per_sample;
  const index_t elem =
      sample_block * static_cast<index_t>(blockDim.x) + threadIdx.x;

  scalar_t strain[6] = {
      macro_strain[0], macro_strain[1], macro_strain[2],
      macro_strain[3], macro_strain[4], macro_strain[5]};
  if (elem < nodes_per_sample) {
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
      const index_t input =
          (batch * nodes_per_sample + node) * index_t(3);
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
  if (elem < nodes_per_sample) {
    const index_t stiffness_base =
        (batch * nodes_per_sample + elem) * index_t(packed ? 21 : 36);
#pragma unroll
    for (int row = 0; row < 6; ++row) {
#pragma unroll
      for (int column = 0; column < 6; ++column) {
        const int index = packed ? packed_voigt_index(row, column)
                                 : row * 6 + column;
        stress[row] += stiffness[stiffness_base + index]
            * strain[column];
      }
    }
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
        (batch * blocks_per_sample + sample_block) * index_t(6);
#pragma unroll
    for (int component = 0; component < 6; ++component) {
      partial[output + component] =
          shared[component * blockDim.x];
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void c3d8_avg_stress_finalize_kernel(
    const scalar_t* __restrict__ partial,
    scalar_t* __restrict__ output,
    const index_t blocks_per_sample,
    const index_t nodes_per_sample,
    const index_t batch_size) {
  extern __shared__ unsigned char shared_bytes[];
  scalar_t* shared = reinterpret_cast<scalar_t*>(shared_bytes);
  const index_t batch = static_cast<index_t>(blockIdx.x);
  if (batch >= batch_size) {
    return;
  }

#pragma unroll
  for (int component = 0; component < 6; ++component) {
    scalar_t sum = scalar_t(0);
    for (index_t block = threadIdx.x;
         block < blocks_per_sample;
         block += blockDim.x) {
      sum += partial[
          (batch * blocks_per_sample + block) * index_t(6) + component];
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
      output[batch * index_t(6) + component] =
          shared[component * blockDim.x]
          / static_cast<scalar_t>(nodes_per_sample);
    }
  }
}

template <typename complex_t, typename index_t>
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
  const index_t rhs_base =
      batch * index_t(3) * frequencies_per_sample + frequency;
  const complex_t rhs0 = rhs_spectrum[rhs_base];
  const complex_t rhs1 = rhs_spectrum[rhs_base + frequencies_per_sample];
  const complex_t rhs2 =
      rhs_spectrum[rhs_base + index_t(2) * frequencies_per_sample];
  const index_t symbol_sample = symbol_batch == 1 ? 0 : batch;
  const index_t symbol_base =
      symbol_sample * index_t(9) * frequencies_per_sample + frequency;

#pragma unroll
  for (int row = 0; row < 3; ++row) {
    const index_t row_base =
        symbol_base + static_cast<index_t>(row * 3) * frequencies_per_sample;
    output[rhs_base + static_cast<index_t>(row) * frequencies_per_sample] =
        symbol_inverse[row_base] * rhs0
        + symbol_inverse[row_base + frequencies_per_sample] * rhs1
        + symbol_inverse[row_base + index_t(2) * frequencies_per_sample] * rhs2;
  }
}

template <typename scalar_t, typename index_t, bool compressed>
void launch_stencil_apply(
    const torch::Tensor& u_flat,
    const torch::Tensor& stencil,
    torch::Tensor& output,
    const int nz,
    const int ny,
    const int nx,
    const int64_t nodes_per_sample,
    const int64_t total_nodes,
    const int threads,
    cudaStream_t stream) {
  const int64_t blocks = checked_blocks(total_nodes, threads);
  c3d8_stencil_apply_kernel<scalar_t, index_t, compressed>
      <<<blocks, threads, 0, stream>>>(
          u_flat.const_data_ptr<scalar_t>(),
          stencil.const_data_ptr<scalar_t>(),
          output.mutable_data_ptr<scalar_t>(),
          nz,
          ny,
          nx,
          static_cast<index_t>(nodes_per_sample),
          static_cast<index_t>(total_nodes));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t, typename index_t, bool packed>
void launch_rhs(
    const torch::Tensor& stiffness,
    const torch::Tensor& geometry,
    const torch::Tensor& macro_strain,
    torch::Tensor& element_force,
    torch::Tensor& output,
    const int nz,
    const int ny,
    const int nx,
    const int64_t nodes_per_sample,
    const int64_t total_elements,
    const int threads,
    const int variant,
    cudaStream_t stream) {
  const int64_t blocks = checked_blocks(total_elements, threads);
  if (variant == 0) {
    c3d8_rhs_element_kernel<scalar_t, index_t, false, packed>
        <<<blocks, threads, 0, stream>>>(
            stiffness.const_data_ptr<scalar_t>(),
            geometry.const_data_ptr<scalar_t>(),
            macro_strain.const_data_ptr<scalar_t>(),
            element_force.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(nodes_per_sample),
            static_cast<index_t>(total_elements));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    c3d8_node_gather_kernel<scalar_t, index_t>
        <<<blocks, threads, 0, stream>>>(
            element_force.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(nodes_per_sample),
            static_cast<index_t>(total_elements));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    C10_CUDA_CHECK(cudaMemsetAsync(
        output.mutable_data_ptr<scalar_t>(),
        0,
        output.numel() * sizeof(scalar_t),
        stream));
    c3d8_rhs_element_kernel<scalar_t, index_t, true, packed>
        <<<blocks, threads, 0, stream>>>(
            stiffness.const_data_ptr<scalar_t>(),
            geometry.const_data_ptr<scalar_t>(),
            macro_strain.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            nz,
            ny,
            nx,
            static_cast<index_t>(nodes_per_sample),
            static_cast<index_t>(total_elements));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t, typename index_t, bool packed>
void launch_avg_stress(
    const torch::Tensor& u_flat,
    const torch::Tensor& stiffness,
    const torch::Tensor& geometry,
    const torch::Tensor& macro_strain,
    torch::Tensor& partial,
    torch::Tensor& output,
    const int nz,
    const int ny,
    const int nx,
    const int64_t nodes_per_sample,
    const int64_t blocks_per_sample,
    const int64_t total_partial_blocks,
    const int64_t batch,
    const int threads,
    cudaStream_t stream) {
  const size_t shared_bytes =
      static_cast<size_t>(threads) * 6 * sizeof(scalar_t);
  c3d8_avg_stress_partial_kernel<scalar_t, index_t, packed>
      <<<total_partial_blocks, threads, shared_bytes, stream>>>(
          u_flat.const_data_ptr<scalar_t>(),
          stiffness.const_data_ptr<scalar_t>(),
          geometry.const_data_ptr<scalar_t>(),
          macro_strain.const_data_ptr<scalar_t>(),
          partial.mutable_data_ptr<scalar_t>(),
          nz,
          ny,
          nx,
          static_cast<index_t>(nodes_per_sample),
          static_cast<index_t>(blocks_per_sample),
          static_cast<index_t>(total_partial_blocks));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  c3d8_avg_stress_finalize_kernel<scalar_t, index_t>
      <<<batch, threads, shared_bytes, stream>>>(
          partial.const_data_ptr<scalar_t>(),
          output.mutable_data_ptr<scalar_t>(),
          static_cast<index_t>(blocks_per_sample),
          static_cast<index_t>(nodes_per_sample),
          static_cast<index_t>(batch));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor c3d8_stencil_apply_cuda(
    torch::Tensor u_flat,
    torch::Tensor stencil,
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
  check_block_size(block_size);
  const int64_t nodes_per_sample = checked_node_count(nz, ny, nx);
  const int64_t batch = u_flat.size(0);
  TORCH_CHECK(batch > 0, "batch dimension must be positive");
  TORCH_CHECK(
      batch <= std::numeric_limits<int64_t>::max() / nodes_per_sample,
      "total node count overflows int64");
  const int64_t total_nodes = batch * nodes_per_sample;
  TORCH_CHECK(
      u_flat.size(1) == 3 * nodes_per_sample,
      "u_flat has the wrong number of dofs");
  TORCH_CHECK(
      stencil.dim() == 6
          && stencil.size(0) == batch
          && (stencil.size(1) == 27 || stencil.size(1) == 14)
          && stencil.size(2) == 9
          && stencil.size(3) == nz
          && stencil.size(4) == ny
          && stencil.size(5) == nx,
      "stencil must have shape (B, 27|14, 9, Nz, Ny, Nx)");

  const auto device = u_flat.device();
  const auto scalar_type = u_flat.scalar_type();
  check_cuda_tensor(stencil, "stencil", device, scalar_type);
  const c10::cuda::CUDAGuard device_guard(device);
  auto output = torch::empty_like(u_flat);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(u_flat.get_device()).stream();
  const bool needs_int64 =
      stencil.numel() > std::numeric_limits<int32_t>::max()
      || u_flat.numel() > std::numeric_limits<int32_t>::max();
  const bool compressed = stencil.size(1) == 14;

  AT_DISPATCH_FLOATING_TYPES(
      scalar_type, "c3d8_stencil_apply_cuda", [&] {
        if (compressed) {
          if (needs_int64) {
            launch_stencil_apply<scalar_t, int64_t, true>(
                u_flat, stencil, output, static_cast<int>(nz),
                static_cast<int>(ny), static_cast<int>(nx), nodes_per_sample,
                total_nodes, static_cast<int>(block_size), stream);
          } else {
            launch_stencil_apply<scalar_t, int32_t, true>(
                u_flat, stencil, output, static_cast<int>(nz),
                static_cast<int>(ny), static_cast<int>(nx), nodes_per_sample,
                total_nodes, static_cast<int>(block_size), stream);
          }
        } else if (needs_int64) {
          launch_stencil_apply<scalar_t, int64_t, false>(
              u_flat, stencil, output, static_cast<int>(nz),
              static_cast<int>(ny), static_cast<int>(nx), nodes_per_sample,
              total_nodes, static_cast<int>(block_size), stream);
        } else {
          launch_stencil_apply<scalar_t, int32_t, false>(
              u_flat, stencil, output, static_cast<int>(nz),
              static_cast<int>(ny), static_cast<int>(nx), nodes_per_sample,
              total_nodes, static_cast<int>(block_size), stream);
        }
      });
  return output;
}

torch::Tensor c3d8_rhs_cuda(
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
  TORCH_CHECK(
      (stiffness.dim() == 4 && stiffness.size(2) == 6 && stiffness.size(3) == 6)
          || (stiffness.dim() == 3 && stiffness.size(2) == 21),
      "stiffness must have shape (B, N, 6, 6) or packed (B, N, 21)");
  TORCH_CHECK(
      stiffness.scalar_type() == torch::kFloat32
          || stiffness.scalar_type() == torch::kFloat64,
      "stiffness must use float32 or float64");
  check_block_size(block_size);
  TORCH_CHECK(
      variant == 0 || variant == 1,
      "variant must be 0 (thread) or 1 (atomic)");
  const int64_t nodes_per_sample = checked_node_count(nz, ny, nx);
  const int64_t batch = stiffness.size(0);
  TORCH_CHECK(batch > 0, "batch dimension must be positive");
  const bool packed = stiffness.dim() == 3;
  const bool full = stiffness.dim() == 4;
  TORCH_CHECK(
      stiffness.size(1) == nodes_per_sample && (packed || full),
      "stiffness must have shape (B, N, 6, 6) or packed (B, N, 21)");
  TORCH_CHECK(
      batch <= std::numeric_limits<int64_t>::max() / nodes_per_sample,
      "total element count overflows int64");
  const int64_t total_elements = batch * nodes_per_sample;

  const auto device = stiffness.device();
  const auto scalar_type = stiffness.scalar_type();
  check_cuda_tensor(geometry, "geometry", device, scalar_type);
  check_cuda_tensor(macro_strain, "macro_strain", device, scalar_type);
  TORCH_CHECK(
      geometry.dim() == 1 && geometry.numel() == 4,
      "geometry must have shape (4,)");
  TORCH_CHECK(
      macro_strain.dim() == 1 && macro_strain.numel() == 6,
      "macro_strain must have shape (6,)");

  const c10::cuda::CUDAGuard device_guard(device);
  auto element_force = variant == 0
      ? torch::empty({batch, nodes_per_sample, 8, 3}, stiffness.options())
      : torch::Tensor();
  auto output =
      torch::empty({batch, 3 * nodes_per_sample}, stiffness.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(stiffness.get_device()).stream();
  const bool needs_int64 =
      stiffness.numel() > std::numeric_limits<int32_t>::max()
      || total_elements * 24 > std::numeric_limits<int32_t>::max();

  if (packed) {
    AT_DISPATCH_FLOATING_TYPES(
        scalar_type, "c3d8_rhs_cuda_packed", [&] {
          if (needs_int64) {
            launch_rhs<scalar_t, int64_t, true>(
                stiffness, geometry, macro_strain, element_force, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, total_elements, static_cast<int>(block_size),
                static_cast<int>(variant), stream);
          } else {
            launch_rhs<scalar_t, int32_t, true>(
                stiffness, geometry, macro_strain, element_force, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, total_elements, static_cast<int>(block_size),
                static_cast<int>(variant), stream);
          }
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        scalar_type, "c3d8_rhs_cuda", [&] {
          if (needs_int64) {
            launch_rhs<scalar_t, int64_t, false>(
                stiffness, geometry, macro_strain, element_force, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, total_elements, static_cast<int>(block_size),
                static_cast<int>(variant), stream);
          } else {
            launch_rhs<scalar_t, int32_t, false>(
                stiffness, geometry, macro_strain, element_force, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, total_elements, static_cast<int>(block_size),
                static_cast<int>(variant), stream);
          }
        });
  }
  return output;
}

torch::Tensor c3d8_avg_stress_cuda(
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
  check_block_size(block_size);
  const int64_t nodes_per_sample = checked_node_count(nz, ny, nx);
  const int64_t batch = u_flat.size(0);
  TORCH_CHECK(batch > 0, "batch dimension must be positive");
  TORCH_CHECK(
      u_flat.size(1) == 3 * nodes_per_sample,
      "u_flat has the wrong number of dofs");
  const bool packed = stiffness.dim() == 3 && stiffness.size(2) == 21;
  const bool full = stiffness.dim() == 4
      && stiffness.size(2) == 6 && stiffness.size(3) == 6;
  TORCH_CHECK(
      (packed || full) && stiffness.size(0) == batch
          && stiffness.size(1) == nodes_per_sample,
      "stiffness must have shape (B, N, 6, 6) or packed (B, N, 21)");

  const auto device = u_flat.device();
  const auto scalar_type = u_flat.scalar_type();
  check_cuda_tensor(stiffness, "stiffness", device, scalar_type);
  check_cuda_tensor(geometry, "geometry", device, scalar_type);
  check_cuda_tensor(macro_strain, "macro_strain", device, scalar_type);
  TORCH_CHECK(
      geometry.dim() == 1 && geometry.numel() == 4,
      "geometry must have shape (4,)");
  TORCH_CHECK(
      macro_strain.dim() == 1 && macro_strain.numel() == 6,
      "macro_strain must have shape (6,)");

  const int64_t blocks_per_sample =
      (nodes_per_sample + block_size - 1) / block_size;
  TORCH_CHECK(
      batch <= kMaxGridX,
      "batch dimension exceeds the CUDA grid-x limit");
  TORCH_CHECK(
      batch <= kMaxGridX / blocks_per_sample,
      "average-stress partial reduction exceeds the CUDA grid-x limit");
  const int64_t total_partial_blocks = batch * blocks_per_sample;
  const c10::cuda::CUDAGuard device_guard(device);
  auto partial =
      torch::empty({batch, blocks_per_sample, 6}, u_flat.options());
  auto output = torch::empty({batch, 6}, u_flat.options());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(u_flat.get_device()).stream();
  const bool needs_int64 =
      stiffness.numel() > std::numeric_limits<int32_t>::max()
      || u_flat.numel() > std::numeric_limits<int32_t>::max()
      || partial.numel() > std::numeric_limits<int32_t>::max();

  if (packed) {
    AT_DISPATCH_FLOATING_TYPES(
        scalar_type, "c3d8_avg_stress_cuda_packed", [&] {
          if (needs_int64) {
            launch_avg_stress<scalar_t, int64_t, true>(
                u_flat, stiffness, geometry, macro_strain, partial, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, blocks_per_sample, total_partial_blocks, batch,
                static_cast<int>(block_size), stream);
          } else {
            launch_avg_stress<scalar_t, int32_t, true>(
                u_flat, stiffness, geometry, macro_strain, partial, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, blocks_per_sample, total_partial_blocks, batch,
                static_cast<int>(block_size), stream);
          }
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        scalar_type, "c3d8_avg_stress_cuda", [&] {
          if (needs_int64) {
            launch_avg_stress<scalar_t, int64_t, false>(
                u_flat, stiffness, geometry, macro_strain, partial, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, blocks_per_sample, total_partial_blocks, batch,
                static_cast<int>(block_size), stream);
          } else {
            launch_avg_stress<scalar_t, int32_t, false>(
                u_flat, stiffness, geometry, macro_strain, partial, output,
                static_cast<int>(nz), static_cast<int>(ny), static_cast<int>(nx),
                nodes_per_sample, blocks_per_sample, total_partial_blocks, batch,
                static_cast<int>(block_size), stream);
          }
        });
  }
  return output;
}

torch::Tensor c3d8_fft_spectral_multiply_cuda(
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
      "rhs_spectrum must be on ",
      symbol_inverse.device());
  TORCH_CHECK(
      symbol_inverse.dim() == 6,
      "symbol_inverse must have shape (S, 3, 3, Nz, Ny, Nx_spec)");
  TORCH_CHECK(
      rhs_spectrum.dim() == 5,
      "rhs_spectrum must have shape (B, 3, Nz, Ny, Nx_spec)");
  TORCH_CHECK(
      symbol_inverse.size(1) == 3 && symbol_inverse.size(2) == 3,
      "symbol_inverse must have 3x3 matrix dimensions");
  TORCH_CHECK(
      rhs_spectrum.size(1) == 3,
      "rhs_spectrum must have three vector components");
  TORCH_CHECK(
      symbol_inverse.size(3) == rhs_spectrum.size(2)
          && symbol_inverse.size(4) == rhs_spectrum.size(3)
          && symbol_inverse.size(5) == rhs_spectrum.size(4),
      "symbol_inverse and rhs_spectrum frequency dimensions must match");
  check_block_size(block_size);

  const int64_t batch = rhs_spectrum.size(0);
  const int64_t symbol_batch = symbol_inverse.size(0);
  TORCH_CHECK(batch > 0, "rhs_spectrum batch dimension must be positive");
  TORCH_CHECK(
      symbol_batch == 1 || symbol_batch == batch,
      "symbol_inverse batch dimension must be one or match rhs_spectrum");
  const int64_t frequencies_per_sample =
      rhs_spectrum.size(2) * rhs_spectrum.size(3) * rhs_spectrum.size(4);
  TORCH_CHECK(
      frequencies_per_sample > 0,
      "rhs_spectrum frequency dimensions must be positive");
  TORCH_CHECK(
      batch <= std::numeric_limits<int64_t>::max() / frequencies_per_sample,
      "total frequency count overflows int64");
  const int64_t total_frequencies = batch * frequencies_per_sample;
  const int64_t blocks =
      checked_blocks(total_frequencies, static_cast<int>(block_size));

  const c10::cuda::CUDAGuard device_guard(rhs_spectrum.device());
  auto output = torch::empty_like(rhs_spectrum);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(rhs_spectrum.get_device()).stream();
  const bool needs_int64 =
      symbol_inverse.numel() > std::numeric_limits<int32_t>::max()
      || rhs_spectrum.numel() > std::numeric_limits<int32_t>::max();

  AT_DISPATCH_COMPLEX_TYPES(
      rhs_spectrum.scalar_type(), "c3d8_fft_spectral_multiply_cuda", [&] {
        if (needs_int64) {
          fft_spectral_multiply_3x3_kernel<scalar_t, int64_t>
              <<<blocks, block_size, 0, stream>>>(
                  symbol_inverse.const_data_ptr<scalar_t>(),
                  rhs_spectrum.const_data_ptr<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>(),
                  frequencies_per_sample,
                  total_frequencies,
                  static_cast<int>(symbol_batch));
        } else {
          fft_spectral_multiply_3x3_kernel<scalar_t, int32_t>
              <<<blocks, block_size, 0, stream>>>(
                  symbol_inverse.const_data_ptr<scalar_t>(),
                  rhs_spectrum.const_data_ptr<scalar_t>(),
                  output.mutable_data_ptr<scalar_t>(),
                  static_cast<int32_t>(frequencies_per_sample),
                  static_cast<int32_t>(total_frequencies),
                  static_cast<int>(symbol_batch));
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}
