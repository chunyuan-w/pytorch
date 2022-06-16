#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/OpContext.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace convolution {

#define DECLARE_CREATE_CONVOLUTION_PREPACK_OP(FUNC_NAME, ...) \
  c10::intrusive_ptr<mkldnn::ConvOpContext> FUNC_NAME(        \
      Tensor weight,                                          \
      c10::optional<Tensor> bias,                             \
      std::vector<int64_t> stride,                            \
      std::vector<int64_t> padding,                           \
      std::vector<int64_t> dilation,                          \
      int64_t groups,                                         \
      std::vector<int64_t> input_size,                        \
      __VA_ARGS__);

DECLARE_CREATE_CONVOLUTION_PREPACK_OP(createConvPrePackOpContext, std::string);
DECLARE_CREATE_CONVOLUTION_PREPACK_OP(
    createConvPrePackOpContextWithScalar,
    std::string,
    at::Scalar,
    at::Scalar,
    at::Scalar,
    std::string);
DECLARE_CREATE_CONVOLUTION_PREPACK_OP(
    createConvPrePackOpContextWithOptional,
    std::string,
    c10::optional<at::Scalar>,
    c10::optional<at::Scalar>);

Tensor conv_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::ConvOpContext>& op_context);

ContextConv create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size,
    const ideep::attr_t& attr);

Tensor run(ContextConv& context, const Tensor& input);

void run(ContextConv& context, const Tensor& input, void* output);

} // namespace convolution
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
