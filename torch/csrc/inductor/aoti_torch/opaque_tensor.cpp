#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/opaque_tensor.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#endif

namespace torch {
namespace aot_inductor {

#if AT_MKLDNN_ENABLED()

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device) {
  printf("from mkldnn_tensor_from_data_ptr\n");
  ideep::tensor::data_type ideep_dtype = at::native::get_mkldnn_dtype(dtype);
  // TODO: why s8 here? Seems dtype does not matter. Only the data_ptr is important here.
  auto a = ideep::tensor({dims.vec(), ideep::tensor::data_type::s8}, data_ptr);
  // auto a = ideep::tensor({dims.vec(), ideep_dtype}, data_ptr);
  return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);
}

#else

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

#endif

} // namespace aot_inductor
} // namespace torch
