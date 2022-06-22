#pragma once

#include <ATen/Config.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#if AT_MKLDNN_ENABLED()

#include <ideep/tensor.hpp>

#endif // AT_MKLDNN_ENABLED()

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()

namespace mkldnn {

using AttrFunction = std::function<ideep::attr_t(
    std::vector<c10::optional<at::Scalar>>,
    c10::optional<std::string>)>;

struct PostOp {
  AttrFunction attr_function;
  std::vector<std::string> scalar_input;
  std::string algorithm_input = "";
  std::vector<torch::jit::MatchFilter> filters = {};
};

const std::map<std::string, PostOp>& fusion_attr_map();

} // namespace mkldnn

#endif // AT_MKLDNN_ENABLED()

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
