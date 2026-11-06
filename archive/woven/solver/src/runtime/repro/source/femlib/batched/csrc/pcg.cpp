#include <torch/extension.h>

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
    int64_t block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "jacobi_step",
      &pcg_jacobi_step_cuda,
      "Fused Jacobi-PCG vector update and reductions (CUDA)",
      pybind11::arg("u"),
      pybind11::arg("r"),
      pybind11::arg("p"),
      pybind11::arg("Ap"),
      pybind11::arg("M_inv"),
      pybind11::arg("rz"),
      pybind11::arg("rhs_norm"),
      pybind11::arg("best_u"),
      pybind11::arg("best_res"),
      pybind11::arg("res"),
      pybind11::arg("workspace"),
      pybind11::arg("state"),
      pybind11::arg("block_size") = 256);
}
