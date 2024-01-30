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
    at::Device device,
    const uint8_t* serialized_md,
    int64_t serialized_md_size) {

  std::vector<uint8_t> vector_serialized_md {serialized_md, serialized_md + serialized_md_size};
  dnnl_memory_desc_t deserialized_wei_desc;
  dnnl_memory_desc_create_with_blob(
          &deserialized_wei_desc, vector_serialized_md.data());

  auto a = ideep::tensor(deserialized_wei_desc, data_ptr);
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