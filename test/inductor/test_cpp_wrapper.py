# Owner(s): ["module: inductor"]

import math

import torch

import torch._dynamo
import torch.nn as nn
from torch._inductor import config
from torch.testing._internal.common_utils import run_tests, TestCase

config.cpp.simdlen = 8
# config.dynamic_shapes = True


class TestCppWrapper(TestCase):
    def _common(self, m, inputs):
        with torch.no_grad():
            optimized_mod = torch._dynamo.optimize()(m)
            dynamo_result = optimized_mod(*inputs)
            ref_result = m(*inputs)
            self.assertEqual(dynamo_result, ref_result)

    # TODO: this duplicates with UT in test_torchinductor.py
    def test_adaptive_avg_pool2d1(self):
        def fn(x):
            return torch.ops.aten._adaptive_avg_pool2d(x, (6, 6))        

        self._common(
            fn,
            [torch.randn(2, 4, 4, 4),],
        )

        self._common(
            fn,
            [torch.randn(2, 4, 3, 3),],
        )        

    def test_single_kernel(self):
        def func_softmax(a):
            return torch.nn.functional.softmax(a)

        shape = (2, 512)
        a = torch.rand(shape)

        self._common(func_softmax, [a])
        self._common(func_softmax, [torch.rand(1, 215)])

    def test_two_kernels(self):
        class MHAScoresCalculation(nn.Module):
            def __init__(self, dim_per_head, softmax_dim=-1):
                super(MHAScoresCalculation, self).__init__()
                self.softmax = nn.Softmax(dim=softmax_dim)
                self.dim_per_head = dim_per_head

            def forward(self, mat1, mat2, bias):
                mat1 = mat1 / math.sqrt(self.dim_per_head)
                qk = torch.matmul(mat1, mat2.transpose(2, 3))
                scores = qk + bias
                return self.softmax(scores), scores

        seq_length = 384
        mat1 = torch.randn(56, 16, seq_length, 64)
        mat2 = torch.randn(56, 16, seq_length, 64)
        bias = torch.randn(56, 16, seq_length, seq_length)

        mha = MHAScoresCalculation(64)

        self._common(mha, [mat1, mat2, bias])


if __name__ == "__main__":
    run_tests()
