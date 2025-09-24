import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import extension_cpp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def make_sparse_tensor_transposed(size, k_sparsity):
    random_tensor = torch.randn(*size, device = "cuda", requires_grad = False)
    _, topk_indices = torch.topk(random_tensor, k_sparsity, dim=-1)

    mask = torch.zeros_like(random_tensor, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    masked_latent = torch.where(mask, random_tensor, torch.tensor(0))
    sparse_latent = masked_latent.to_sparse()
    sparse_latent = sparse_latent.T.coalesce()
    return sparse_latent.indices(), sparse_latent.values(), sparse_latent.shape[0], sparse_latent.shape[1]

def reference_sdmm(a, b):
    sparse_tensor = torch.sparse_coo_tensor(a[0], a[1], (a[2], a[3]))
    return torch.matmul(sparse_tensor, b)

def sample_inputs(device, *, requires_grad=False):
    def make_sparse_tensor(size, k_sparsity):
        random_tensor = torch.randn(*size, device = device, requires_grad = requires_grad)
        _, topk_indices = torch.topk(random_tensor, k_sparsity, dim=-1)
    
        mask = torch.zeros_like(random_tensor, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)
        masked_latent = torch.where(mask, random_tensor, torch.tensor(0))
        sparse_latent = masked_latent.to_sparse()
        return sparse_latent.indices(), sparse_latent.values(), sparse_latent.shape[0], sparse_latent.shape[1]

    def make_dense_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)


    return [
        [make_sparse_tensor((8, 32), 4), make_dense_tensor(32, 4)],
        [make_sparse_tensor((16, 32), 4), make_dense_tensor(32, 4)],
        [make_sparse_tensor((1, 1024), 32), make_dense_tensor(1024, 4)],
        [make_sparse_tensor((20, 64), 0), make_dense_tensor(64, 80)],
    ]

def test_correctness(device):#
    samples = sample_inputs(device)
    for args in samples:
        result = extension_cpp.ops.sdmm(*args)
        expected = reference_sdmm(*args)
        a = args[0]
        torch.testing.assert_close(result, expected)
        
        assert torch.allclose(result, expected)

def speed_test(): # 10 times faster naively
    from time import perf_counter
    sparse = make_sparse_tensor_transposed((64, 2 ** 12), 4) # runs transposed
    dense = torch.randn((64, 3072), device = "cuda")
    result = extension_cpp.ops.sdmm(sparse, dense)
    expected = reference_sdmm(sparse, dense)
    torch.testing.assert_close(result, expected, atol = 1e-4, rtol = 1e-4)
    print("Correctness tasks passed")
    custom_start = perf_counter()
    for i in range(100):
        extension_cpp.ops.sdmm(sparse, dense)
        torch.cuda.synchronize()
    custom_time = perf_counter() - custom_start
    print("Custom kernel time: ", custom_time)

    reference_start = perf_counter()
    for j in range(100):
        reference_sdmm(sparse, dense)
        torch.cuda.synchronize()
    default_time = perf_counter() - reference_start
    print("Default torch sparse matmul time: ", default_time)
    print(f"Custom kernel is {round(default_time / custom_time, 3)} times faster")

outputs = test_correctness("cuda")
speed_test()

