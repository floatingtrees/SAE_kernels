# Sparse Dense Multiplication Kernel for Sparse Autoencoders

Code skeleton from https://pytorch.org/tutorials/advanced/cpp_custom_ops.html

Sparse Autoencoders have very few nonzero elements in each dimension, and we know precisely how many nonzero elements there will be along each axis. Thus, they can benefit from kernels beyond the default kernels provided by torch sparse. 



To build:
```
pip install torch==2.5.0 numpy==2.0.2
python -m pip install -U pip setuptools wheel
pip install expecttest
pip install --no-build-isolation -e .
```


To benchmark Pytorch sparse default vs CUDA:
```
python test/sdmm_tests.py
```
On an RTX 4090, you should see: 

```
Custom kernel time:  0.021685993298888206
Default torch sparse matmul time:  0.061105010099709034
Custom kernel is 2.818 times faster
```

Forked from pytorch CUDA implementation reference
