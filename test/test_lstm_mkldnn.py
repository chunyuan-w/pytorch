    
import copy
import itertools
import functools
import unittest

import torch.nn.functional as F
import torch.nn as nn
import torch.jit
import torch.backends.mkldnn
from torch.utils import mkldnn as mkldnn_utils
from torch.testing._internal.common_utils import TestCase, \
    run_tests, TemporaryFileName, gradcheck, gradgradcheck, IS_WINDOWS  

torch.manual_seed(2021)
    
class TestMkldnn(TestCase):
    def test_rnn(self):
        I = torch.randint(1, 1000, (1,)).item()
        H = torch.randint(1, 1000, (1,)).item()
        T = torch.randint(10, 100, (1,)).item()
        N = torch.randint(1, 100, (1,)).item()
        x = torch.randn(T, N, I, dtype=torch.float32)
        x_bf16 = copy.deepcopy(x).to(torch.bfloat16)
        for mod in [torch.nn.LSTM]:
            for bidirectional in [True, False]:
            # for bidirectional in [False]:
                D = 2 if bidirectional else 1
                for L in [1, 4]:
                # for L in [4]:
                    if mod is torch.nn.LSTM:
                        hx0 = torch.randn(L * D, N, H, dtype=torch.float32)
                        cx0 = torch.randn(L * D, N, H, dtype=torch.float32)
                        h0 = (hx0, cx0)

                        hx0_bf16 = copy.deepcopy(hx0).to(torch.bfloat16)
                        cx0_bf16 = copy.deepcopy(cx0).to(torch.bfloat16)
                        h0_bf16 = (hx0_bf16, cx0_bf16)

                    rnn = mod(I, H, L, bidirectional=bidirectional).float().eval()
                    rnn_bf16 = copy.deepcopy(rnn).to(torch.bfloat16)
                    y0, hn0 = rnn(x, h0)
                    y0_bf16, hn0_bf16 = rnn_bf16(x_bf16, h0_bf16)

                    self.assertEqual(y0.bfloat16(), y0_bf16, atol=1e-2, rtol=0)
                    if mod is torch.nn.LSTM:
                        hy0, cy0 = hn0
                        hy0_bf16, cy0_bf16 = hn0_bf16
                        self.assertEqual(hy0.bfloat16(), hy0_bf16, atol=1e-2, rtol=0)
                        self.assertEqual(cy0, cy0_bf16, atol=1e-2, rtol=0)
                    else:
                        self.assertEqual(hn0, hn1.to_dense())


    # def test_lstm(self):
    #     rnn = nn.LSTM(10, 20, 1)
    #     input = torch.randn(5, 3, 10)
    #     h0 = torch.randn(1, 3, 20)
    #     c0 = torch.randn(1, 3, 20)
    #     # output, (hn, cn) = rnn(input, (h0, c0))
    #     output, (hn, cn) = rnn(input)

    #     print(output)
    #     print(hn)
    #     print(cn)

    #     input_bf16 = input.to(torch.bfloat16)
    #     h0_bf16 = h0.to(torch.bfloat16)
    #     # h0_bf16 = h0
        
    #     c0_bf16 = c0.to(torch.bfloat16)
    #     # c0_bf16 = c0
        
    #     rnn = rnn.to(torch.bfloat16)
    #     # output, (hn, cn) = rnn(input_bf16, (h0_bf16, c0_bf16))
    #     output, (hn, cn) = rnn(input_bf16)

    #     print(output.float())
    #     print(hn.float())
    #     print(cn)


if __name__ == '__main__':
    run_tests()
