import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.jit_utils import JitTestCase
import unittest
import torch.nn.functional as F

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

class IPEXConvConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvConvRelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        return F.relu(res, inplace=True)

class IPEXConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvRelu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x, y, z, a):
        res = self.conv(x)
        res = res / y
        res = res * z
        res = res + a
        res = res.to(torch.float32)
        return res

class IPEXConvAddRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(IPEXConvAddRelu, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.conv2 = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x, y):
        a = F.relu(self.conv1(x))
        b = self.conv2(x)
        return F.relu(torch.add(a, b, alpha=y.item()), inplace=True)


class IPEXBottleneck_v1(nn.Module):
    def __init__(self):
        super(IPEXBottleneck_v1, self).__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        x = self.conv(x)
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += x
        return y3.relu_()


class IPEXBottleneck_v2(nn.Module):
    def __init__(self):
        super(IPEXBottleneck_v2, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.downsample = nn.Conv2d(2, 3, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        y1 = self.conv1(x).relu_()
        y2 = self.conv2(y1).relu_()
        y3 = self.conv3(y2)
        y3 += self.downsample(x)
        return y3.relu_()

class IPEXMatmulDiv(nn.Module):
    def __init__(self):
        super(IPEXMatmulDiv, self).__init__()
        seed = 2018
        torch.manual_seed(seed)

    def forward(self, x1, x2, x3):
        return torch.matmul(x1, x2) / x3 + x3

class TestTEAdd(JitTestCase):
    def test_ipex_matmul_div(self):
        te_matmul_div = IPEXMatmulDiv()
        x1 = torch.randn(5, 5)
        x2 = torch.randn(5, 5)
        x3 = torch.randn(5, 5)
        te_matmul_div_traced = torch.jit.script(te_matmul_div).eval()
        te_matmul_div_traced = torch.jit.freeze(te_matmul_div_traced)
        te_matmul_div_traced(x1, x2, x3)
        self.assertAllFused(te_matmul_div_traced.graph_for(x1, x2, x3))
        res_jit = te_matmul_div_traced(x1, x2, x3)
        res_imperative = te_matmul_div(x1, x2, x3)
        self.assertEqual(res_jit, res_imperative)

    def test_ipex_conv_add_relu(self):
        te_model = IPEXConvAddRelu(3, 2, kernel_size=(3, 3)).eval()
        # input (1, 3, 10, 10) -> won't use mkldnn conv
        # input (1, 3, 224, 224) -> will use mkldnn conv
        x1 = torch.randn(1, 3, 10, 10).to(memory_format=torch.channels_last)
        
        # alpha = torch.Tensor([3.4])
        alpha = torch.Tensor([1])
        # alpha = torch.IntTensor([1])
        
        te_model_traced = torch.jit.trace(te_model, (x1, alpha))
        te_model_traced = torch.jit.freeze(te_model_traced)
        te_model_traced(x1, alpha)

        print(te_model_traced.graph_for(x1, alpha))

        self.assertAllFused(te_model_traced.graph_for(x1, alpha))

        res_jit = te_model_traced(x1, alpha)
        res_imperative = te_model(x1, alpha)
        self.assertEqual(res_jit, res_imperative)

        x1 = torch.randn(3, 3, 20, 20)
        res_jit = te_model_traced(x1, alpha)
        res_imperative = te_model(x1, alpha)
        self.assertEqual(res_jit, res_imperative)

    def test_ipex_conv_conv_relu(self):
        te_model = IPEXConvConvRelu(3, 10, kernel_size=(3, 3)).eval()
        x1 = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
        te_model_traced = torch.jit.script(te_model)
        te_model_traced = torch.jit.freeze(te_model_traced)
        te_model_traced(x1)

        self.assertAllFused(te_model_traced.graph_for(x1))

        res_jit = te_model_traced(x1)
        res_imperative = te_model(x1)
        self.assertEqual(res_jit, res_imperative)

        x1 = torch.randn(3, 3, 500, 500)
        res_jit = te_model_traced(x1)
        res_imperative = te_model(x1)
        self.assertEqual(res_jit, res_imperative)

    def test_ipex_conv_relu(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        te_add = IPEXConvRelu(3, 10, kernel_size=(3, 3)).eval()
        x1 = torch.randn(1, 3, 224, 224)
        y1 = torch.randn(1, 10, 222, 222)
        z1 = torch.randn(1, 10, 222, 222)
        a1 = torch.randn(1, 10, 222, 222)
        
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            te_add_traced = torch.jit.trace(te_add, (x1, y1, z1, a1))
        te_add_traced = torch.jit.freeze(te_add_traced)
        
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
        
            te_add_traced(x1, y1, z1, a1)

            graph = te_add_traced.graph_for(x1, y1, z1, a1)
            print(graph)
            self.assertAllFused(graph)
            print("before TE")
            res_jit = te_add_traced(x1, y1, z1, a1)
            print("after TE")

            res_imperative = te_add(x1, y1, z1, a1)
            self.assertEqual(res_jit, res_imperative)

            x1 = torch.randn(3, 3, 500, 500)
            res_jit = te_add_traced(x1, y1, z1, a1)
        res_imperative = te_add(x1, y1, z1, a1)
        self.assertEqual(res_jit, res_imperative)

        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_bottleneck_v2(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        input = torch.randn(1, 2, 5, 5).to(memory_format=torch.channels_last)
        te_model = IPEXBottleneck_v2().to(memory_format=torch.channels_last).eval()
        te_model = ipex.optimize(te_model, dtype=torch.float32, level='O1')

        te_model_traced = torch.jit.trace(te_model, input)
        te_model_traced = torch.jit.freeze(te_model_traced)
        te_model_traced(input)

        te_model_traced(input)

        print("before TE")
        res_jit = te_model_traced(input)
        print("after TE")


        res_imperative = te_model(input)
        self.assertEqual(res_jit, res_imperative)

        print(te_model_traced.graph_for(input))
        self.assertAllFused(te_model_traced.graph_for(input))

        te_model_traced(torch.randn(45, 2, 5, 5).to(memory_format=torch.channels_last))
        torch._C._debug_set_fusion_group_inlining(old)

    def test_ipex_conv_bottleneck_v1(self):
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)

        input = torch.randn(1, 64, 56, 56).to(memory_format=torch.channels_last)
        te_model = IPEXBottleneck_v1().to(memory_format=torch.channels_last).eval()
        te_model = ipex.optimize(te_model, dtype=torch.float32, level='O1')
        te_model_traced = torch.jit.trace(te_model, input)
        te_model_traced = torch.jit.freeze(te_model_traced)
        te_model_traced(input)

        # res_jit = te_model_traced(input)
        # res_imperative = te_model(input)
        # self.assertEqual(res_jit, res_imperative)
        self.assertAllFused(te_model_traced.graph_for(input))
        te_model_traced(torch.randn(5, 64, 56, 56).to(memory_format=torch.channels_last))
        torch._C._debug_set_fusion_group_inlining(old)

class TestModel(JitTestCase):
    @skipIfNoTorchVision
    def _test_vision(self, model_name):
        seed = 2018
        torch.manual_seed(seed)
        old = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        
        # for memory_format in [torch.contiguous_format, torch.channels_last]:
        for memory_format in [torch.channels_last]:
            # TODO: if only x.to(channels_last) but model is nchw, the result is incorrect
            te_model = getattr(torchvision.models, model_name)().eval().to(memory_format=memory_format)
            x = (torch.rand(1, 3, 224, 224) / 10).to(memory_format=memory_format)

            te_model = ipex.optimize(te_model, dtype=torch.float32, level='O1')
            te_model_traced = torch.jit.trace(te_model, x)
            te_model_traced = torch.jit.freeze(te_model_traced)

            te_model_traced(x)
            print(te_model_traced.graph_for(x))

            res_imperative = te_model(x)
            print("before TE")
            res_jit = te_model_traced(x)
            print("after TE")
            self.assertEqual(res_jit, res_imperative)

            te_model_traced((torch.rand(2, 3, 224, 224) / 10).to(memory_format=memory_format))

        torch._C._debug_set_fusion_group_inlining(old)


for model_name, enabled in [
    ['resnet50', True],
]:
    def wrapper(mname):
        @unittest.skipIf(not enabled, 'Disabled')
        def test(self):
            return self._test_vision(mname)
        return test

    setattr(TestModel, 'test_vision_%s' % model_name, wrapper(model_name))

if __name__ == '__main__':
    test = unittest.main()

# PYTORCH_JIT_LOG_LEVEL=":>>kernel:>>fusion_pass:>>tensorexpr_fuser:>>" python -u my_test_te.py -k test_ipex_conv_relu  2>&1 | tee debug_te_bf16.log
