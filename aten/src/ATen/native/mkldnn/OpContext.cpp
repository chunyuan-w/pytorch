#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/LinearPrepack.h>
#include <ATen/native/mkldnn/OpContext.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {

c10::intrusive_ptr<ConvOpContext> MkldnnConvOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    std::vector<int64_t>&& input_size,
    const ideep::attr_t& attr) {
  auto op_context = mkldnn::internal::convolution::create(
      weight, bias, padding, stride, dilation, groups, input_size, attr);

  auto conv_op_context = c10::make_intrusive<MkldnnConvOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      std::move(input_size),
      std::move(op_context));

  return conv_op_context;
}

Tensor MkldnnConvOpContext::run(const Tensor& input) {
  return mkldnn::internal::convolution::run(op_context_, input);
}

void MkldnnConvOpContext::run(const Tensor& input, void* output) {
  return mkldnn::internal::convolution::run(op_context_, input, output);
}

c10::intrusive_ptr<LinearOpContext> MkldnnLinearOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& input_size,
    const ideep::attr_t& attr) {
  auto op_context =
      mkldnn::internal::linear::create(weight, bias, input_size, attr);

  auto linear_op_context = c10::make_intrusive<MkldnnLinearOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(input_size),
      std::move(op_context));

  return linear_op_context;
}

Tensor MkldnnLinearOpContext::run(const Tensor& input) {
  return mkldnn::internal::linear::run(op_context_, input);
}

Tensor MkldnnLinearOpContext::run(const Tensor& input, const Tensor& other) {
  return mkldnn::internal::linear::run(op_context_, input, other);
}

void MkldnnLinearOpContext::run(const Tensor& input, void* output) {
  return mkldnn::internal::linear::run(op_context_, input, output);
}

void MkldnnLinearOpContext::run(const Tensor& input, const Tensor& other, void* output) {
  return mkldnn::internal::linear::run(op_context_, input, other, output);
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
