#include <torch/extension.h>

#include <vector>

namespace extension_cpp
{

  at::Tensor sdmm_cpu(at::Tensor &indices, at::Tensor &values, int64_t m, int64_t n, at::Tensor &dense_tensor)
  {
    TORCH_CHECK(n == dense_tensor.sizes()[0]);
    TORCH_CHECK(values.dtype() == at::kFloat);
    std::cout << "HERE" << indices.dtype() << std::endl;
    TORCH_CHECK(indices.dtype() == at::kLong);
    TORCH_CHECK(dense_tensor.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(dense_tensor.device().type() == at::DeviceType::CPU);
    auto sparse_tensor = torch::sparse_coo_tensor(
        indices, // Indices
        values,  // Values
        {m, n}   // Shape of the sparse tensor
    );
    at::Tensor result = at::matmul(sparse_tensor, dense_tensor);
    return result;
  }

  // Registers _C as a Python extension module.
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

  // Defines the operators
  TORCH_LIBRARY(extension_cpp, m)
  {
    m.def("sdmm(Tensor indices, Tensor values, int m, int n, Tensor dense_tensor) -> Tensor");
  }

  // Registers CUDA implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(extension_cpp, CPU, m)
  {
    m.impl("sdmm", &sdmm_cpu);
  }

}
