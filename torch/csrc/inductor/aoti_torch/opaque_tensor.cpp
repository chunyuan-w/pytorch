#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/opaque_tensor.h>

#if AT_MKLDNN_ENABLED()
#include <ideep.hpp>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#endif

namespace torch {
namespace aot_inductor {

#if AT_MKLDNN_ENABLED()

void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor) {
  void* data_ptr = at::native::data_ptr_from_mkldnn_aot(mkldnn_tensor);
  return data_ptr;
}

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* serialized_md,
    int64_t serialized_md_size) {
  // TODO: add deserialize here        
  auto a = ideep::tensor({dims.vec(), ideep::tensor::data_type::s8}, data_ptr);
  return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);
}

#else

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* serialized_md,
    int64_t serialized_md_size) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

#endif

} // namespace aot_inductor
} // namespace torch