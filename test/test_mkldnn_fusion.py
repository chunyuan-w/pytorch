# Owner(s): ["module: mkldnn"]
import itertools
import unittest

import torch
from torch import nn

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

        torch._C._debug_set_fusion_group_inlining(old_fusion_inlining)
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuser_state)
        torch._C._jit_set_te_must_use_llvm_cpu(old_te_must_use_llvm_cpu)
        return graph

    def _eltwise_list(self):
        eltwise_list = [
            [torch.relu, 'aten::relu'],
            [torch.sigmoid, 'aten::sigmoid'],
            [torch.tanh, 'aten::tanh'],
            [torch.nn.Hardswish(inplace=False), 'aten::hardswish'],
            [nn.LeakyReLU(0.1, inplace=False), 'aten::leaky_relu'],
            [nn.Hardtanh(inplace=False), 'aten::hardtanh'],
            [nn.GELU(approximate="none"), 'aten::gelu'],
            [nn.GELU(approximate="tanh"), 'aten::gelu'],
        ]
        return eltwise_list

    def _clamp_modules(self):
        class MNoOpt(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MNoOpt, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=-0.5, max=0.9)
                return x

        class MInf(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MInf, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=0, max=float('inf'))
                return x

        class MNegInf(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MNegInf, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=float('-inf'), max=0)
                return x

        class MOptMin(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MOptMin, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, max=2)
                return x

        class MOptMax(nn.Module):
            def __init__(self, m, in_channels, out_channels, bias, **kwargs):
                super(MOptMax, self).__init__()
                self.conv = m(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                x = self.conv(x)
                x = torch.clamp(x, min=0)
                return x

        return [MNoOpt, MInf, MNegInf, MOptMin, MOptMax]

    def test_single_conv(self):
        class M(nn.Module):
            def __init__(self, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                res = self.conv(x)
                return res

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            input_size = 224
            batch_size = 1
            kernel_size = 3
            options = itertools.product([True, False], [1, 2], [1, 4])
            for bias, dilation, groups in options:
                iC = 3 * groups
                oC = 10 * groups
                m = M(iC,
                      oC,
                      bias,
                      kernel_size=(kernel_size, kernel_size),
                      stride=2,
                      padding=1,
                      dilation=dilation,
                      groups=groups).to(memory_format=memory_format)
                x = torch.randn(batch_size, iC, input_size, input_size).to(memory_format=memory_format)
                graph = self._check_model(m, x)
                if enabled:
                    self.assertFused(graph, ['aten::conv2d'])
                    self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                else:
                    self.assertGraphContains(graph, kind='aten::conv2d')

    def test_conv_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.conv(x)
                x = self.eltwise(x)
                return x

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for eltwise_fn, op_name in self._eltwise_list():
                for bias in [True, False]:
                    for oC in [1, 10]:
                        m = M(eltwise_fn, 3, oC, bias, kernel_size=(3, 3)).to(memory_format=memory_format)
                        x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)

                        graph = self._check_model(m, x)
                        if enabled:
                            self.assertFused(graph, ['aten::conv2d', 'aten::' + op_name])
                            self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                        else:
                            self.assertGraphContains(graph, kind='aten::conv2d')

    def test_conv_clamp(self):
        modules = self._clamp_modules()
        op_name = 'aten::clamp'

        for memory_format, enabled in [
            [torch.contiguous_format, False],
            [torch.channels_last, True],
        ]:
            for M in modules:
                for bias in [True, False]:
                    m = M(nn.Conv2d, 3, 10, bias, kernel_size=(3, 3)).to(memory_format=memory_format)
                    x = torch.randn(1, 3, 224, 224).to(memory_format=memory_format)

                    graph = self._check_model(m, x)
                    if enabled:
                        self.assertFused(graph, ['aten::conv2d', op_name])
                        self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
                    else:
                        self.assertGraphContains(graph, kind='aten::conv2d')

    def test_single_linear(self):
        class M(nn.Module):
            def __init__(self, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)

            def forward(self, x):
                res = self.linear(x)
                return res
        iC = 2
        oC = 3
        for bias in [True, False]:
            # TODO: refactor x_sghape generation
            for x_shape in [
                [1, iC],
                [2, iC],
                [3, 2, iC]
            ]:
                m = M(iC, oC, bias)
                x = torch.randn(x_shape)
                graph = self._check_model(m, x)
                self.assertFused(graph, ['aten::linear'])
                self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)

    def test_linear_eltwise(self):
        class M(nn.Module):
            def __init__(self, eltwise_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
                self.eltwise = eltwise_fn

            def forward(self, x):
                x = self.linear(x)
                x = self.eltwise(x)
                return x
        iC = 2
        oC = 3
        for eltwise_fn, op_name in self._eltwise_list():
            for bias in [True, False]:
                for x_shape in [
                    [1, iC],
                    [2, iC],
                    [3, 2, iC]
                ]:
                    m = M(eltwise_fn, iC, oC, bias)
                    x = torch.randn(x_shape)

                    graph = self._check_model(m, x)
                    self.assertFused(graph, ['aten::linear', 'aten::' + op_name])
                    self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)
    
    def test_linear_binary(self):
        class M(nn.Module):
            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                super(M, self).__init__()
                self.linear1 = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
                self.linear2 = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
                self.binary = binary_fn

            def forward(self, x):
                out1 = self.linear1(x)
                out2 = self.linear2(x)
                y = self.binary(out1, out2)
                return y
        iC = 2
        oC = 3
        for binary_fn, op_name in [[torch.add, "aten::add"]]:
            for bias in [True, False]:
                for x_shape in [
                    [1, iC],
                    [2, iC],
                    [3, 2, iC]
                ]:
                    m = M(binary_fn, iC, oC, bias)
                    x = torch.randn(x_shape)

                    graph = self._check_model(m, x)
                    print(graph)
                    self.assertFused(graph, ['aten::linear', 'aten::' + op_name])
                    self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)

    def test_matmul_binary(self):
        class M(nn.Module):
            def __init__(self, binary_fn, **kwargs):
                super(M, self).__init__()
                self.binary = binary_fn

            def forward(self, x):
                out1 = torch.matmul(x, x + 1)
                y = self.binary(out1, x.clone())
                return y
        iC = 2
        oC = 3
        for binary_fn, op_name in [[torch.add, "aten::add"]]:
            for bias in [True, False]:
                for x_shape in [
                    #[1, iC],
                    [33, 33],
                    #[3, 2, iC]
                ]:
                    m = M(binary_fn)
                    x = torch.randn(x_shape)

                    graph = self._check_model(m, x)
                    print(graph)
                    self.assertFused(graph, ['aten::matmul', 'aten::' + op_name])
                    self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)

    def test_linear_clamp(self):
        modules = self._clamp_modules()
        op_name = 'aten::clamp'
        iC = 2
        oC = 3
        for M in modules:
            for bias in [True, False]:
                for x_shape in [
                    [1, iC],
                    [2, iC],
                    [3, 2, iC]
                ]:
                    m = M(nn.Linear, iC, oC, bias)
                    x = torch.randn(x_shape)
                    graph = self._check_model(m, x)
                    self.assertFused(graph, ['aten::linear', 'aten::' + op_name])
                    self.assertGraphContainsExactly(graph, FUSION_GROUP, 1)

if __name__ == "__main__":
    run_tests()
