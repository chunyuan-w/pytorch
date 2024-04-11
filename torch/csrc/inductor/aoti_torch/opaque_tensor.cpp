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
    int64_t serialized_md_size,
    int groups) {
  // TODO: add deserialize here


  std::vector<uint8_t> vector_serialized_md{
      serialized_md, serialized_md + serialized_md_size};

//   dnnl_memory_desc_t deserialized_wei_desc;
//   dnnl_memory_desc_create_with_blob(
    //   &deserialized_wei_desc, vector_serialized_md.data());
  // TODO: deconv
  
  // TODO: test ideep versioning
#if IDEEP_PREREQ(3, 4, 1, 2)
  // groups is needed for grouped conv
  ideep::tensor::desc deserialized_ideep_desc(vector_serialized_md);
#else
      TORCH_CHECK(false, "Unexpected IDeep version to do weight deserialization.");
#endif
  
  auto a = ideep::tensor(deserialized_ideep_desc, data_ptr);
  
  // TODO: workaround to let ideep allocate buffer to make it have good alignment
  // Should fix the alignment when creating data ptr
  ideep::tensor aligned_a;
  aligned_a.init(deserialized_ideep_desc);
  aligned_a.feed_from(a);
  return at::native::new_with_itensor_mkldnn(std::move(aligned_a), dtype, device);  
  
  
  // return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);



//   auto a = ideep::tensor({dims.vec(), ideep::tensor::data_type::s8}, data_ptr);
//   return at::native::new_with_itensor_mkldnn(std::move(a), dtype, device);
}

#else

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* serialized_md,
    int64_t serialized_md_size,
    int groups) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

#endif

} // namespace aot_inductor
} // namespace torch
