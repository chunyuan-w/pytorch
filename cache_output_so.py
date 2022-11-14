
from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel0 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/os/cosa6ygqfwn3e7all36epaxgvxcu2r2idsylr7657mlkau6gex5d.h"
#include <chrono>
#include <iostream>
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2,
                       float* __restrict__ out_ptr3,
                       float* __restrict__ out_ptr4,
                       double* __restrict__ out_ptr5,
                       double* __restrict__ out_ptr6)
{
    #pragma omp parallel num_threads(56)
    {
        #pragma omp for 
        for(long i0=0; i0<8; ++i0)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<16; ++i1)
            {
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (16*i0)];
                        auto tmp1 = static_cast<float>(2);
                        auto tmp2 = tmp0 + tmp1;
                        out_ptr0[i1 + (36*i0)] = tmp0;
                        out_ptr1[i1 + (36*i0)] = tmp2;
                    }
                }
            }
        }
        #pragma omp single
        {
            #pragma GCC ivdep
            for(long i0=0; i0<8; ++i0)
            {
                #pragma GCC ivdep
                for(long i1=0; i1<4; ++i1)
                {
                    {
                        {
                            auto tmp0 = in_ptr0[i1 + (16*i0)];
                            auto tmp1 = static_cast<float>(1);
                            auto tmp2 = tmp0 + tmp1;
                            out_ptr2[i1 + (36*i0)] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp for 
        for(long i0=0; i0<128; ++i0)
        {
            {
                {
                    auto tmp0 = in_ptr0[i0];
                    auto tmp1 = static_cast<float>(2);
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp3 = static_cast<double>(tmp2);
                    out_ptr3[i0] = tmp2;
                    out_ptr4[i0] = tmp2;
                    out_ptr5[i0] = tmp3;
                    out_ptr6[i0] = tmp3;
                }
            }
        }
    }
}
''')

async_compile.wait(globals())
del async_compile
from torch.utils.cpp_extension import load_inline
wrapper = (
'''
#include <dlfcn.h>
#include <assert.h>
#include <chrono>

class LoadKernel_call0{
  public:
    LoadKernel_call0() {
    auto kernel0_lib = dlopen("/tmp/torchinductor_chunyuan/li/clij5z7pzkgi5gv5nf5ennrv3mszna67yf6t2xix2nq7y25wgvru.so", RTLD_NOW);
    assert(kernel0_lib != nullptr);
    *(void **) (&kernel0) = dlsym(kernel0_lib, "kernel");

}
void (*kernel0)(const float*,float*,float*,float*,float*,float*,double*,double*);

};
    std::vector<at::Tensor> call_0(std::tuple<at::Tensor&> args) {
    auto& [arg0_1] = args;
    static LoadKernel_call0 load_kernel_;
    auto buf3 = at::empty_strided({8, 36}, {36, 1}, at::ScalarType::Float); 
    auto buf0 = at::as_strided(buf3, {8, 16}, {36, 1});  // alias
    auto buf2 = at::as_strided(buf3, {8, 16}, {36, 1}, 20);  // alias
    auto buf1 = at::as_strided(buf3, {8, 4}, {36, 1}, 16);  // alias
    auto buf6 = at::empty_strided({16, 16}, {16, 1}, at::ScalarType::Float); 
    auto buf4 = at::as_strided(buf6, {8, 16}, {16, 1});  // alias
    auto buf5 = at::as_strided(buf6, {8, 16}, {16, 1}, 128);  // alias
    auto buf9 = at::empty_strided({16, 16}, {16, 1}, at::ScalarType::Double); 
    auto buf7 = at::as_strided(buf9, {8, 16}, {16, 1});  // alias
    auto buf8 = at::as_strided(buf9, {8, 16}, {16, 1}, 128);  // alias
    load_kernel_.kernel0((float*)(arg0_1.data_ptr()), (float*)(buf0.data_ptr()), (float*)(buf2.data_ptr()), (float*)(buf1.data_ptr()), (float*)(buf4.data_ptr()), (float*)(buf5.data_ptr()), (double*)(buf7.data_ptr()), (double*)(buf8.data_ptr()));
    arg0_1.reset();
    return std::vector<at::Tensor>({buf3, buf6, buf9}); }''' )

name = 'inline_extension_cmwqndennbxflz35syenlzd4wg766ajamm6zsobd5wwxmmm4vrht'
build_dir = "/tmp/torchinductor_chunyuan/torch_extensions"
import os, importlib
filepath = os.path.join(build_dir, f'{name}.so')
#  mkdir /tmp/torchinductor_chunyuan/torch_extensions
if not os.path.exists(build_dir):
    os.mkdir(build_dir)

if os.path.exists(filepath):
    print("found lib")
    spec = importlib.util.spec_from_file_location(name, filepath)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, importlib.abc.Loader)
    spec.loader.exec_module(module)
else:
    print("build lib from source")
    module = load_inline(
        name=name,
        build_directory=build_dir,
        cpp_sources=[wrapper],
        functions=['call_0'],
        extra_cflags=['-fPIC -Wall -std=c++14 -Wno-unused-variable -ftemplate-depth-5000 -fconstexpr-depth=1024 -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp'],
        extra_ldflags=['-shared -L/home/chunyuan/torch-inductor/pytorch/torch/lib -L/home/chunyuan/miniconda3/envs/torch-inductor/lib -lc10 -ltorch -ltorch_cpu -ltorch_python -lgomp'],
        extra_include_paths=['-I/home/chunyuan/torch-inductor/pytorch/torch/include -I/home/chunyuan/torch-inductor/pytorch/torch/include/torch/csrc/api/include -I/home/chunyuan/torch-inductor/pytorch/torch/include/TH -I/home/chunyuan/torch-inductor/pytorch/torch/include/THC -I/home/chunyuan/miniconda3/envs/torch-inductor/include/python3.8'])

def _wrap_func(f):
    def g(args):
        return f(args)
    return g
call = _wrap_func(module.call_0)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 16), (16, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))
