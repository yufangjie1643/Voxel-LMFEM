#include <torch/extension.h>
torch::Tensor assemble_hg_cuda(torch::Tensor C, torch::Tensor scale,
    torch::Tensor T, torch::Tensor H, torch::Tensor valid,
    torch::Tensor offsets, int64_t nz, int64_t ny, int64_t nx);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("assemble", &assemble_hg_cuda);}
