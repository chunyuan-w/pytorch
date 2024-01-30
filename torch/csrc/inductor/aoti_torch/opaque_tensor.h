#pragma once

#include <ATen/Tensor.h>

namespace torch {
namespace aot_inductor {

at::Tensor mkldnn_tensor_from_data_ptr(
        void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* serialized_md,
    int64_t serialized_md_size);
}
} // namespace torch