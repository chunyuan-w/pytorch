#pragma once

#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {
namespace tensorexpr {
Tensor prepareVectorizationForReduceOps(
    Tensor t,
    size_t softmax_dim,
    size_t rank);

Tensor computeSoftmax(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    bool log_softmax);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
