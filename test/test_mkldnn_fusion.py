# Owner(s): ["module: mkldnn"]

import torch
import torch.nn.functional as F
from torch import nn
import unittest

from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.jit_utils import JitTestCase

from test_tensorexpr import warmup_and_run_forward


def get_eltwise_fn(name):
    if hasattr(torch, name):
        return getattr(torch, name)
    elif hasattr(F, name):
        return getattr(F, name)
    else:
        raise NameError("Eltwise function %s not found" % name)


@unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnnFusion(JitTestCase):
    def _check_model(self, m, x):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        m.eval()
        with torch.no_grad():
            script = torch.jit.script(m)
        script = torch.jit.freeze(script)

        with torch.no_grad():
            y = warmup_and_run_forward(script, x)
            y = script(x)
            y_ref = m(x)

            graph = script.graph_for(*x)
            self.assertEqual(y, y_ref)
        torch._C._debug_set_fusion_group_inlining(old)
        return graph

    def test_conv(self):
        class M(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

            def forward(self, x):
                res = self.conv(x)
                return res

        for memory_format, enabled in [
            [torch.contiguous_format, True],
            [torch.channels_last, False],  # TODO: enable support on channels_last
        ]:
            m = M(3, 10, kernel_size=(3, 3)).to(memory_format=memory_format)
            x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)
            graph = self._check_model(m, x)
            if enabled:
                self.assertAllFused(graph)
            else:
                self.assertGraphContains(graph, kind='aten::conv2d')

    def test_conv_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, **kwargs):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.conv(x)
                x = self.eltwise(x)
                return x

        for eltwise in ["relu"]:
            for inplace in [False, True]:
                eltwise_fn_name = eltwise + "_" if inplace else eltwise
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                m = M(eltwise_fn, 3, 10, kernel_size=(3, 3))
                x = torch.randn(1, 3, 224, 224)

                graph = self._check_model(m, x)
                self.assertAllFused(graph)


if __name__ == "__main__":
    run_tests()
