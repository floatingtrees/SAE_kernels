import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import extension_cpp
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_sparse_tensor(size, k_sparsity):
            random_tensor = torch.randn(*size, device = device, requires_grad = requires_grad)
            _, topk_indices = torch.topk(random_tensor, k_sparsity, dim=-1)
        
            mask = torch.zeros_like(random_tensor, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            masked_latent = torch.where(mask, random_tensor, torch.tensor(0))
            sparse_latent = masked_latent.to_sparse()
            return sparse_latent

        def make_dense_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)


        return [
            [make_sparse_tensor((3, 10), 5), make_dense_tensor(10, 3)],
            [make_sparse_tensor((20, 32), 32), make_dense_tensor(32, 5)],
            [make_sparse_tensor((20, 1024), 32), make_dense_tensor(1024, 20)],
            [make_sparse_tensor((8, 2), 0), make_dense_tensor(2, 3)],
        ]
x = TestMyMulAdd()
print(x.sample_inputs("cuda"))