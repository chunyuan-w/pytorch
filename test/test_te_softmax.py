# Owner(s): ["module: mkldnn"]
import itertools
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase

from test_tensorexpr import warmup_and_run_forward

FUSION_GROUP = 'prim::TensorExprGroup'


@unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnnFusion(JitTestCase):
    def assertFused(self, graph, fused_patterns):
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def _check_model(self, m, x):
        old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        torch._C._jit_override_can_fuse_on_cpu(True)

        old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)
        
        old_reduction = torch._C._jit_texpr_reductions_enabled()
        torch._C._jit_set_texpr_reductions_enabled(True)

        m.eval()
        with torch.no_grad():
            script = torch.jit.trace(m, x)
        script = torch.jit.freeze(script)

        with torch.no_grad():
            y = warmup_and_run_forward(script, x)
            y = script(x)
            y_ref = m(x)

            graph = script.graph_for(*x)
            self.assertEqual(y, y_ref)

        torch._C._debug_set_fusion_group_inlining(old_fusion_inlining)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(old_te_must_use_llvm_cpu)
        torch._C._jit_set_texpr_reductions_enabled(old_reduction)
        return graph

    def test_single_sum(self):
        class M(torch.nn.Module):
            def __init__(self, eltwise, **kargs):
                super(M, self).__init__()
                self.eltwise = eltwise
                self.kargs = kargs

            def forward(self, x):
                return self.eltwise(x, **self.kargs)
        kwargs = {"dim": -1}
        m = M(torch.sum, **kwargs)
        # x = torch.randn(1, 256, 384, 384)
        # for size in [128, 9]:
        for size in [128]:
            print("size: ", size)
            x = torch.randn(size)
            # x = torch.randn(9)
            graph = self._check_model(m, x)
            self.assertFused(graph, ['aten::add'])
            self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)


    def test_single_softmax(self):
        class M(torch.nn.Module):
            def __init__(self, eltwise, **kargs):
                super(M, self).__init__()
                self.eltwise = eltwise
                self.kargs = kargs

            def forward(self, x):
                return self.eltwise(x, **self.kargs)
        kwargs = {"dim": -1}
        m = M(torch.softmax, **kwargs)
        # x = torch.randn(1, 256, 384, 384)
        # for size in [128, 9]:
        for size in [128]:
            print("size: ", size)
            x = torch.randn(size)
            # x = torch.randn(9)
            graph = self._check_model(m, x)
            self.assertFused(graph, ['aten::softmax'])
            self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)

if __name__ == "__main__":
    run_tests()