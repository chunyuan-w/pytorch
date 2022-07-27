#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

namespace at { namespace native {

// result = beta * result + alpha * gemm(mat1, mat2)
TORCH_API void mkldnn_matmul(
        const Tensor &mat1,
        const Tensor &mat2,
        const Tensor &result,
        float beta=1,
        float alpha=1);
bool use_mkldnn_bf16_matmul(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& result_opt);

}


}

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace linear {

Tensor mkldnn_matmul_binary_run(
    const Tensor &mat1,
    const Tensor &other,
    const Tensor &mat2,
    std::string post_op);


} // namespace linear
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
