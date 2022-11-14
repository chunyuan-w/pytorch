from distutils.core import Extension

import Cython
from Cython.Build.Inline import _get_build_extension
from Cython.Build.Dependencies import cythonize
import os

srcfile = "/home/chunyuan/torch-inductor/pytorch/my_kernel_as_strided.cpp"
build_dir = "/home/chunyuan/torch-inductor/pytorch"
# you can add include_dirs= and extra_compile_args= here
extension = Extension(name='_module', language='c++', sources=[srcfile], include_dirs=[
    "/home/chunyuan/torch-inductor/pytorch/",
    "/home/chunyuan/torch-inductor/pytorch/torch/include",
    "/home/chunyuan/torch-inductor/pytorch/torch/include/torch/csrc/api/include",
    "/home/chunyuan/torch-inductor/pytorch/torch/include/TH",
    "/home/chunyuan/torch-inductor/pytorch/torch/include/THC",
    ], 
    library_dirs=["/home/chunyuan/torch-inductor/pytorch/torch/lib", "/home/chunyuan/miniconda3/envs/torch-inductor/lib"],
    libraries=["c10", "torch", "torch_cpu", "torch_python", "gomp"])

build_extension = _get_build_extension()
build_extension.extensions = cythonize([extension],
                                       include_path=[],
                                       quiet=False)
build_extension.build_temp = os.path.dirname(srcfile)
build_extension.build_lib = build_dir  # where you want the output

build_extension.run()

import importlib
import sys

import torch
from torch._dynamo.testing import rand_strided
sys.path.insert(0, "/home/chunyuan/torch-inductor/pytorch")
m = importlib.import_module("_module")


arg0_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)

print(m.call_0([arg0_1]))