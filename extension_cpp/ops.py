import torch
from torch import Tensor
from typing import Tuple


__all__ = ["sdmm"]


def sdmm(sparse_tensor: Tuple[Tensor, Tensor, int, int], dense_tensor: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    indices, values, m, n = sparse_tensor
    #if dense_tensor.shape[1] % 32 != 0:
    #    raise ValueError("Unable to efficiently implement matmul on CUDA")
    return torch.ops.extension_cpp.sdmm(indices, values, m, n, dense_tensor)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.


@torch.library.register_fake("extension_cpp::sdmm")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)
