#include <torch/extension.h>

torch::Tensor c3d8_stencil_apply_cuda(
    torch::Tensor u_flat,
    torch::Tensor stencil,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    int64_t block_size);

torch::Tensor c3d8_rhs_cuda(
    torch::Tensor stiffness,
    torch::Tensor geometry,
    torch::Tensor macro_strain,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    int64_t block_size,
    int64_t variant);

torch::Tensor c3d8_avg_stress_cuda(
    torch::Tensor u_flat,
    torch::Tensor stiffness,
    torch::Tensor geometry,
    torch::Tensor macro_strain,
    int64_t nz,
    int64_t ny,
    int64_t nx,
    int64_t block_size);

torch::Tensor c3d8_fft_spectral_multiply_cuda(
    torch::Tensor symbol_inverse,
    torch::Tensor rhs_spectrum,
    int64_t block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "apply",
      &c3d8_stencil_apply_cuda,
      "Batched periodic C3D8 stencil apply (CUDA)",
      pybind11::arg("u_flat"),
      pybind11::arg("stencil"),
      pybind11::arg("nz"),
      pybind11::arg("ny"),
      pybind11::arg("nx"),
      pybind11::arg("block_size") = 256);
  module.def(
      "rhs",
      &c3d8_rhs_cuda,
      "C3D8 fused right-hand side assembly (CUDA)",
      pybind11::arg("stiffness"),
      pybind11::arg("geometry"),
      pybind11::arg("macro_strain"),
      pybind11::arg("nz"),
      pybind11::arg("ny"),
      pybind11::arg("nx"),
      pybind11::arg("block_size") = 128,
      pybind11::arg("variant") = 0);
  module.def(
      "avg_stress",
      &c3d8_avg_stress_cuda,
      "C3D8 average stress recovery (CUDA)",
      pybind11::arg("u_flat"),
      pybind11::arg("stiffness"),
      pybind11::arg("geometry"),
      pybind11::arg("macro_strain"),
      pybind11::arg("nz"),
      pybind11::arg("ny"),
      pybind11::arg("nx"),
      pybind11::arg("block_size") = 128);
  module.def(
      "fft_spectral_multiply",
      &c3d8_fft_spectral_multiply_cuda,
      "Batched complex 3x3 FFT-symbol multiply (CUDA)",
      pybind11::arg("symbol_inverse"),
      pybind11::arg("rhs_spectrum"),
      pybind11::arg("block_size") = 256);
}

