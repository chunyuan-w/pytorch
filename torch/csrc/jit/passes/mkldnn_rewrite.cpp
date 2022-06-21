#include <ATen/Config.h>
#include <ATen/code_template.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/mkldnn_rewrite.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

namespace torch {
namespace jit {

#if AT_MKLDNN_ENABLED()

namespace mkldnn {

static constexpr float kMin = -std::numeric_limits<float>::infinity();
static constexpr float kMax = std::numeric_limits<float>::infinity();

const std::string no_scalar_op_list_construct = R"(
%scalars : Scalar?[] = prim::ListConstruct()
%algorithm : str? = prim::Constant()
)";

const std::string leaky_relu_op_list_construct = R"(
%scalars : Scalar?[] = prim::ListConstruct(%alpha)
%algorithm : str? = prim::Constant()
)";

const std::string hardtanh_op_list_construct = R"(
%scalars : Scalar?[] = prim::ListConstruct(%alpha, %beta)
%algorithm : str? = prim::Constant()
)";

const std::string gelu_op_list_construct = R"(
%scalars : Scalar?[] = prim::ListConstruct()
)";

const std::string clamp_op_list_construct = R"(
%scalars : Scalar?[] = prim::ListConstruct(%alpha, %beta)
%algorithm : str? = prim::Constant()
)";

bool aten_gelu_approximate_is_supported(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  auto approximate_value =
      graph_rewrite_helper::getIValue("algorithm", match_vmap, vmap).value();
  return approximate_value == "none" || approximate_value == "tanh";
}

AttrFunction none_attr_func = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  return ideep::attr_t();
};

AttrFunction relu_attr_func = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  return ideep::attr_t::fuse_relu();
};

AttrFunction sigmoid_attr_func =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      return ideep::attr_t::fuse_sigmoid();
    };

AttrFunction tanh_attr_func = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  return ideep::attr_t::fuse_tanh();
};

AttrFunction hardswish_attr_func =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      return ideep::attr_t::fuse_hardswish();
    };

AttrFunction leaky_relu_attr_func =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      auto alpha_value = scalars[0].value().to<float>();
      return ideep::attr_t::fuse_relu(1.0, alpha_value);
    };

AttrFunction hardtanh_attr_func =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      auto lower_bound_value = scalars[0].value().to<float>();
      auto upper_bound_value = scalars[1].value().to<float>();
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

AttrFunction gelu_attr_func = [](std::vector<c10::optional<at::Scalar>> scalars,
                                 c10::optional<std::string> algorithm) {
  dnnl::algorithm gelu_type;
  if (algorithm.value() == "none") {
    gelu_type = dnnl::algorithm::eltwise_gelu_erf;
  } else if (algorithm.value() == "tanh") {
    gelu_type = dnnl::algorithm::eltwise_gelu_tanh;
  } else {
    TORCH_CHECK(
        false, "ipex::linear_gelu_run only support tanh approximate now");
  }

  return ideep::attr_t::fuse_gelu(1.0, 0.f, 0.f, gelu_type);
};

AttrFunction clamp_attr_func =
    [](std::vector<c10::optional<at::Scalar>> scalars,
       c10::optional<std::string> algorithm) {
      float lower_bound_value =
          scalars[0] ? scalars[0].value().to<float>() : kMin;
      float upper_bound_value =
          scalars[1] ? scalars[1].value().to<float>() : kMax;

      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    };

const std::map<std::string, PostOp>& fusion_attr_map() {
  static const std::map<std::string, PostOp> fusion_attr_map{
      {"none",
       {none_attr_func,
        std::vector<std::string>({}),
        no_scalar_op_list_construct}},

      {"relu",
       {relu_attr_func,
        std::vector<std::string>({}),
        no_scalar_op_list_construct}},

      {"sigmoid",
       {sigmoid_attr_func,
        std::vector<std::string>({}),
        no_scalar_op_list_construct}},

      {"tanh",
       {tanh_attr_func,
        std::vector<std::string>({}),
        no_scalar_op_list_construct}},

      {"hardswish",
       {hardswish_attr_func,
        std::vector<std::string>({}),
        no_scalar_op_list_construct}},

      // TODO: support elu in NNC firstly
      {"leaky_relu",
       {leaky_relu_attr_func,
        std::vector<std::string>({"%alpha"}),
        leaky_relu_op_list_construct}},

      {"hardtanh",
       {hardtanh_attr_func,
        std::vector<std::string>({"%alpha", "%beta"}),
        hardtanh_op_list_construct}},

      {"gelu",
       {gelu_attr_func,
        std::vector<std::string>({"%algorithm"}),
        gelu_op_list_construct,
        {aten_gelu_approximate_is_supported}}},

      {"clamp",
       {clamp_attr_func,
        std::vector<std::string>({"%alpha", "%beta"}),
        clamp_op_list_construct}},

  };
  return fusion_attr_map;
}

} // namespace mkldnn

c10::VaryingShape<int64_t> getSizesOf(Node* n, size_t idx) {
  auto tt = n->input(idx)->type()->cast<TensorType>();
  return tt->sizes();
}

void insertPrePackedConvOpForNode(Node* n) {
  constexpr int POS_INPUT = 0;
  constexpr int POS_WEIGHT = 1;
  if (!tensorexpr::isContiguous(
          n->input(POS_INPUT), at::MemoryFormat::ChannelsLast)) {
    GRAPH_DEBUG(
        "insertPrePackedConvOpForNode: input is not ChannelsLast contiguous");
    return;
  }

  if (!tensorexpr::isContiguous(
          n->input(POS_WEIGHT), at::MemoryFormat::ChannelsLast)) {
    GRAPH_DEBUG(
        "insertPrePackedConvOpForNode: weight is not ChannelsLast contiguous");
    return;
  }

  // Leave depthwise conv2d to NNC
  if (tensorexpr::conv2dIsSupportedJit(n)) {
    GRAPH_DEBUG("insertPrePackedConvOpForNode: leave depthwise conv2d to NNC");
    return;
  }

  WithInsertPoint guard(n);
  auto graph = n->owningGraph();

  auto input_sizes = getSizesOf(n, POS_INPUT);
  IValue input_size_value(*input_sizes.concrete_sizes());
  auto input_size = graph->insertConstant(input_size_value);

  auto prepack_node = graph->create(
      Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"), 1);

  // skip input value
  for (auto i = 1; i < n->inputs().size(); i++) {
    Value* v = n->input(i);
    prepack_node->addInput(v);
  }
  prepack_node->addInput(input_size);
  auto attr = graph->insertConstant(IValue("none"));
  prepack_node->addInput(attr);

  std::vector<c10::optional<at::Scalar>> empty_scalars;
  auto scalars = graph->insertConstant(IValue(empty_scalars));
  prepack_node->addInput(scalars);

  c10::optional<std::string> empty_algorithm;
  auto algorithm = graph->insertConstant(IValue(empty_algorithm));
  prepack_node->addInput(algorithm);

  prepack_node->output()->setType(
      getCustomClass("__torch__.torch.classes.mkldnn.ConvOpContext"));
  graph->insertNode(prepack_node);

  auto prepack_conv = graph->insertNode(
      graph->create(Symbol::fromQualString("mkldnn_prepacked::conv2d_run"), 1));
  prepack_conv->addInput(n->input(0));
  prepack_conv->addInput(prepack_node->output());
  prepack_conv->output()->setType(n->output()->type()->cast<TensorType>());

  n->output()->replaceAllUsesWith(prepack_conv->output());
}

bool isTensorTypeCPU(Node* node) {
  for (const auto& input : node->inputs()) {
    auto type = input->type()->cast<TensorType>();
    if (!type) {
      continue;
    }
    auto device = type->device();
    if (!device) {
      return false;
    }
    if (!device->is_cpu()) {
      return false;
    }
  }
  return true;
}

void insertPrePackedConvOp(Block* b) {
  for (Node* n : b->nodes()) {
    for (Block* b : n->blocks()) {
      insertPrePackedConvOp(b);
    }

    if (n->kind() == aten::conv2d) {
      if (isTensorTypeCPU(n)) {
        insertPrePackedConvOpForNode(n);
      }
    }
  }
  EliminateDeadCode(b);
}

void insertMkldnnPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // Replace _convolution with conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  insertPrePackedConvOp(graph->block());
}

void insertMkldnnPrePackedOps(std::shared_ptr<Graph>& graph) {
  insertMkldnnPrePackedConv2dOp(graph);
}

void insertMkldnnPrePackedOps(script::Module& module) {
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    insertMkldnnPrePackedOps(graph);
  }
  for (script::Module m : module.children()) {
    insertMkldnnPrePackedOps(m);
  }
}

template <typename T>
void RewriteEltwiseGraph(
    std::shared_ptr<Graph>& graph,
    const std::map<std::string, T>& fusion_attr_map) {
  auto conv_op_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %dummy_attr:str, %dummy_scalars: Scalar?[], %dummy_algorithm: str?${op_input_str}):
        %packed_weight_bias = mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %dummy_attr, %dummy_scalars, %dummy_algorithm)
        %conv2d_res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        %res = aten::${op}(%conv2d_res${op_input_str})
        return (%res))");

  auto conv_op_fused_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %input_size:int[], %dummy_attr:str, %dummy_scalars: Scalar?[], %dummy_algorithm: str?${op_input_str}):
        %attr: str = prim::Constant[value="${op_attr}"]()
        ${op_list_construct}
        %packed_weight_bias : __torch__.torch.classes.mkldnn.ConvOpContext =  mkldnn_prepacked::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %input_size, %attr, %scalars, %algorithm)
        %res = mkldnn_prepacked::conv2d_run(%input, %packed_weight_bias)
        return (%res))");

  for (auto const& it : fusion_attr_map) {
    std::string op = it.first;
    if (op == std::string("none")) {
      continue;
    }
    std::vector<std::string> op_input_list = it.second.op_input_list;
    std::string op_input_str = c10::Join(", ", op_input_list);

    if (op_input_list.size() > 0) {
      op_input_str = ", " + op_input_str;
    }

    at::jit::TemplateEnv env;
    env.s("op", op);
    env.s("op_input_str", op_input_str);

    at::jit::TemplateEnv env_fused;
    env_fused.s("op_attr", op);
    env_fused.s("op_input_str", op_input_str);
    env_fused.s("op_list_construct", it.second.op_list_construct);

    SubgraphRewriter rewriter;
    rewriter.RegisterRewritePattern(
        conv_op_rstring.format(env), conv_op_fused_rstring.format(env_fused));

    auto filters = it.second.filters;
    rewriter.runOnGraph(graph, filters);
  }
}

void FuseEltwiseWithPackedOps(std::shared_ptr<Graph>& graph) {
  RewriteEltwiseGraph<mkldnn::PostOp>(graph, mkldnn::fusion_attr_map());
}

void PrePackingOpsFolder(Block* b) {
  auto is_foldable_op = [](const Node* n) -> bool {
    return (
        n->kind() ==
        Symbol::fromQualString("mkldnn_prepacked::conv2d_prepack"));
  };

  std::unordered_set<Node*> nodes_to_delete;
  for (Node* n : b->nodes()) {
    for (Block* block : n->blocks()) {
      PrePackingOpsFolder(block);
    }
    if (is_foldable_op(n)) {
      auto optional_outputs = torch::jit::runNodeIfInputsAreConstant(n);
      if (optional_outputs) {
        auto outputs = optional_outputs.value();
        TORCH_CHECK(outputs.size() == 1, "Prepack ops have single output");
        Value* prepack_op_value = n->output(0);
        auto graph = n->owningGraph();
        WithInsertPoint ins(prepack_op_value->node());
        auto weak_class_obj =
            outputs[0].toObject()->copy_to_weak_compilation_ref();
        Value* packed_weight = graph->insertConstant(weak_class_obj)
                                   ->setType(n->output(0)->type());
        prepack_op_value->replaceAllUsesWith(packed_weight);
        nodes_to_delete.insert(n);
      }
    }
  }
  for (auto n : nodes_to_delete) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete) {
    n->destroy();
  }
}

void FoldPrePackingOps(std::shared_ptr<Graph>& graph) {
  PrePackingOpsFolder(graph->block());
}

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG(
      "Before insertMkldnnPrePackedOps. Beginning of FuseConvWithEltwise\n",
      *graph);
  insertMkldnnPrePackedOps(graph);
  GRAPH_DEBUG(
      "After insertMkldnnPrePackedOps, before FuseEltwiseWithPackedOps\n",
      *graph);
  FuseEltwiseWithPackedOps(graph);
  GRAPH_DEBUG(
      "After FuseEltwiseWithPackedOps, before ConstantPropagation\n", *graph);
  ConstantPropagation(graph);
  GRAPH_DEBUG("After ConstantPropagation, before FoldPrePackingOps\n", *graph);
  FoldPrePackingOps(graph);
  GRAPH_DEBUG("After FoldPrePackingOps. End of FuseConvWithEltwise\n", *graph);
}

#else

void FuseConvWithEltwise(std::shared_ptr<Graph>& graph) {
  GRAPH_DEBUG("MKLDNN Not enabled");
}

#endif // AT_MKLDNN_ENABLED()

} // namespace jit
} // namespace torch
