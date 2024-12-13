#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp
{

  __global__ void sdmm_kernel(const int64_t *index_ptr, const float *values_ptr, int64_t m, int64_t n, int64_t p, const float *dense_ptr, float *result_ptr, int64_t num_sparse_indices)
  {
    int index_number = blockIdx.x * blockDim.y + threadIdx.y; // all with the same x thread index operate on the same element
    if (index_number >= num_sparse_indices)
    {
      return;
    }
    int column_index = index_ptr[index_number + num_sparse_indices];
    int row_index = index_ptr[index_number];
    float sparse_value = values_ptr[index_number];

    int result_row_base = row_index * p;   // result has shape (m, p)
    int dense_row_base = column_index * p; // dense has shape (n, p)

    for (int i = 0; i < p; i += blockDim.x)
    {
      atomicAdd(&result_ptr[result_row_base + i + threadIdx.x], sparse_value * dense_ptr[dense_row_base + i + threadIdx.x]);
      /*if (threadIdx.x == 1)
      {
        printf("Row: %d, Column: %d, Value: %f, Write index:, %d, Dense read location: %d, Dense read value %f\n", row_index, column_index, sparse_value, result_row_base + i + threadIdx.x, dense_row_base + i + threadIdx.x, dense_ptr[dense_row_base + i + threadIdx.x]);
      }*/
    }

    // choose sparse element to matmul with
  }

  at::Tensor sdmm_cuda(const at::Tensor &indices, const at::Tensor &values, int64_t m, int64_t n, const at::Tensor &dense_tensor)
  {

    TORCH_CHECK(values.dtype() == at::kFloat);
    TORCH_CHECK(dense_tensor.dtype() == at::kFloat);
    TORCH_CHECK(indices.dtype() == at::kLong);
    int64_t num_sparse_indices = values.sizes()[0];
    TORCH_CHECK(num_sparse_indices % 4 == 0, "unable to assign efficient block sizes");
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

    double sparsity_percentage = (double)num_sparse_indices / (double)dense_tensor.numel();
    // assign

    at::Tensor result_tensor = torch::zeros({m, p}, dense_contig.options());
    const float *values_ptr = values_contig.data_ptr<float>();
    const float *dense_ptr = dense_contig.data_ptr<float>();
    const int64_t *index_ptr = indices_contig.data_ptr<int64_t>();
    float *result_ptr = result_tensor.data_ptr<float>();
    int threads_per_element = 4;
    dim3 block_size;
    int num_blocks;
    if (num_sparse_indices * threads_per_element <= 1024)
    {

      block_size = dim3(threads_per_element, num_sparse_indices);
      num_blocks = 1;
    }
    else
    {
      int elements_per_block = 1024 / threads_per_element;
      block_size = dim3(threads_per_element, elements_per_block);
      num_blocks = (num_sparse_indices + 1023) / elements_per_block;
    }

    sdmm_kernel<<<num_blocks, block_size>>>(index_ptr, values_ptr, m, n, p, dense_ptr, result_ptr, num_sparse_indices);
    cudaDeviceSynchronize();
    return result_tensor;
  }

  // Registers CUDA implementations for mymuladd, mymul, myadd_out
  TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m)
  {
    m.impl("sdmm", &sdmm_cuda);
  }

}
