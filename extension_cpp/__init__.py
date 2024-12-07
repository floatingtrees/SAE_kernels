import torch
from . import _C
def _lazy_import_ops():
    from . import ops
    return ops

ops = _lazy_import_ops()