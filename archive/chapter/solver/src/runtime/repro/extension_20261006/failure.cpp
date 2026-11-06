#include <torch/extension.h>
torch::Tensor evaluate_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("evaluate",&evaluate_cuda);}
