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
        [make_sparse_tensor((64, 32), 32), make_dense_tensor(32, 4)],
        [make_sparse_tensor((32, 1024), 32), make_dense_tensor(1024, 20)],
        [make_sparse_tensor((20, 64), 0), make_dense_tensor(64, 80)],
    ]

def test_correctness(device):
        samples = sample_inputs(device)
        for args in samples:
            result = extension_cpp.ops.sdmm(*args)
            expected = reference_sdmm(*args)
            '''print(result)
            print(expected)
            print(args[0])
            a = args[0]
            print(torch.sparse_coo_tensor(a[0], a[1], (a[2], a[3])).to_dense())
            print(args[1])'''
            torch.testing.assert_close(result, expected)
            
            print(torch.allclose(result, expected))



outputs = test_correctness("cuda")
