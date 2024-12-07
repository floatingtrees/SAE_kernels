#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp
{

  __global__ void sdmm_kernel(int numel, const float *a, const float *b, float c, float *result)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
      result[idx] = a[idx] * b[idx] + c;
  }

  at::Tensor sdmm_cuda(at::Tensor &indices, at::Tensor &values, int64_t m, int64_t n, at::Tensor &dense_tensor)
  {
    TORCH_CHECK(values.dtype() == at::kFloat);
    TORCH_CHECK(dense_tensor.dtype() == at::kFloat);
    TORCH_CHECK(indices.dtype() == at::kLong);
    TORCH_CHECK(!dense_tensor.is_sparse());
    TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(dense_tensor.device().type() == at::DeviceType::CUDA);
    auto dense_sizes = dense_tensor.sizes();
    int p = dense_sizes[1];
    TORCH_CHECK(n == dense_sizes[0]);

    at::Tensor indices_contig = indices.contiguous();
    at::Tensor values_contig = values.contiguous();
    at::Tensor dense_contig = dense_tensor.contiguous();

    at::Tensor result_tensor = torch::empty({m, p}, dense_contig.options());
    const float *values_ptr = values_contig.data_ptr<float>();
    const float *dense_ptr = dense_contig.data_ptr<float>();
    const int *index_ptr = indices_contig.data_ptr<int>();
    float *result_ptr = result_tensor.data_ptr<float>();
    int numel = result_tensor.numel();

    return result_tensor;
  }

  // Registers CUDA implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m)
  {
    m.impl("sdmm", &sdmm_cuda);
  }

}
