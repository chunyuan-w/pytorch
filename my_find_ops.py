from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_chunyuan/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(56)
    {
        #pragma omp for  collapse(2)
        for(long i0=0; i0<3; i0+=1)
        {
            for(long i1=0; i1<12544; i1+=1)
            {
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (12544*i0)];
                        out_ptr0[i0 + (3*i1)] = tmp0;
                    }
                }
            }
        }
    }
}
''')

async_compile.wait(globals())
del async_compile
from torch._inductor.codecache import CppWrapperCodeCache
wrapper = (
'''
#include <dlfcn.h>
#include <assert.h>

class LoadKernel_call0{
  public:
    LoadKernel_call0() {
    auto kernel_cpp_0_lib = dlopen("/tmp/torchinductor_chunyuan/sq/csqik24syo2l2xkora6spqbpdodpnomqvg3deef6igwh74blsz3o.so", RTLD_NOW);
    assert(kernel_cpp_0_lib != nullptr);
    *(void **) (&kernel_cpp_0) = dlsym(kernel_cpp_0_lib, "kernel");

}
void (*kernel_cpp_0)(const float*,float*);

};
    at::Tensor call_0(std::vector<at::Tensor> args) {
    at::Tensor arg0_1, arg1_1, arg2_1;
    arg0_1 = args[0];
    arg1_1 = args[1];
    arg2_1 = args[2];
    static LoadKernel_call0 load_kernel_;
    auto buf0 = at::empty_strided({1, 3, 112, 112}, {37632, 1, 336, 3}, at::ScalarType::Float); 
    load_kernel_.kernel_cpp_0((float*)(arg2_1.data_ptr()), (float*)(buf0.data_ptr()));
    arg2_1.reset();

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "mkldnn::_convolution_pointwise",
                "")
            .typed<at::Tensor(
                const at::Tensor& input_t,
                const at::Tensor& weight_t,
                const c10::optional<at::Tensor>& bias_opt,
                at::IntArrayRef padding,
                at::IntArrayRef stride,
                at::IntArrayRef dilation,
                int64_t groups,
                c10::string_view attr,
                torch::List<c10::optional<at::Scalar>> scalars,
                c10::optional<c10::string_view> algorithm)>();


    auto buf1 = op.call(buf0, arg0_1, arg1_1, {0, 0}, {1, 1}, {1, 1}, 1, "relu", {}, "");
    //assert_size_stride(buf1, {1, 32, 112, 112}, {401408, 1, 3584, 32})
    arg0_1.reset();
    arg1_1.reset();
    return buf1; }''' )

module = CppWrapperCodeCache.load(wrapper, 'call_0')

def _wrap_func(f):
    def g(args):
        return f(args)
    return g
call = _wrap_func(module.call_0)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 1, 1), (1, 0, 0, 0), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 112, 112), (37632, 12544, 112, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))