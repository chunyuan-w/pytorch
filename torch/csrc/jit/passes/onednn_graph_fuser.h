#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

#include <ATen/Config.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

TORCH_API void fuseGraph(std::shared_ptr<Graph>& g);

} // namespace onednn
} // namespace fuser

struct C10_EXPORT RegisterLlgaFuseGraph
    : public PassManager<RegisterLlgaFuseGraph> {
  static bool setEnabled(bool enabled) {
    TORCH_CHECK(
        AT_MKLDNN_ENABLED(),
        "Running oneDNN Graph fuser is only supported with MKLDNN builds.");
    bool oldState = PassManager::isRegistered();
    if (enabled) {
      PassManager::registerPass(fuser::onednn::fuseGraph);
    } else {
      PassManager::clearPass();
    }
    return oldState;
  }

  static bool isEnabled() {
    return PassManager::isRegistered();
  }
};

} // namespace jit
} // namespace torch
