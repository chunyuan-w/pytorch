    
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
    # def test_rnn(self):
    #     I = torch.randint(1, 1000, (1,)).item()
    #     H = torch.randint(1, 1000, (1,)).item()
    #     T = torch.randint(10, 100, (1,)).item()
    #     N = torch.randint(1, 100, (1,)).item()
    #     x = torch.randn(T, N, I, dtype=torch.float32)
    #     for mod in [torch.nn.LSTM]:
    #         # for bidirectional in [True, False]:
    #         for bidirectional in [False]:
    #             D = 2 if bidirectional else 1
    #             # for L in [1, 4]:
    #             for L in [1]:
    #                 if mod is torch.nn.LSTM:
    #                     hx0 = torch.randn(L * D, N, H, dtype=torch.float32)
    #                     cx0 = torch.randn(L * D, N, H, dtype=torch.float32)
    #                     h0 = (hx0, cx0)
    #                     h1 = (hx0.to_mkldnn(), cx0.to_mkldnn())
    #                 else:
    #                     h0 = torch.randn(L * D, N, H, dtype=torch.float32)
    #                     h1 = h0.to_mkldnn()

    #                 rnn = mod(I, H, L, bidirectional=bidirectional).float().eval()
    #                 mkldnn_rnn = mkldnn_utils.to_mkldnn(copy.deepcopy(rnn))
    #                 y0, hn0 = rnn(x, h0)
    #                 y1, hn1 = mkldnn_rnn(x.to_mkldnn(), h1)

    #                 self.assertEqual(y0, y1.to_dense())
    #                 if mod is torch.nn.LSTM:
    #                     hy0, cy0 = hn0
    #                     hy1, cy1 = hn1
    #                     self.assertEqual(hy0, hy1.to_dense())
    #                     self.assertEqual(cy0, cy1.to_dense())
    #                 else:
    #                     self.assertEqual(hn0, hn1.to_dense())


    def test_lstm(self):
        rnn = nn.LSTM(10, 20, 1)
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(1, 3, 20)
        c0 = torch.randn(1, 3, 20)
        output, (hn, cn) = rnn(input, (h0, c0))

        print(output)
        print(hn)
        print(cn)

        input_bf16 = input.to(torch.bfloat16)
        h0_bf16 = h0.to(torch.bfloat16)
        
        # c0_bf16 = c0.to(torch.bfloat16)
        c0_bf16 = c0
        
        rnn = rnn.to(torch.bfloat16)

        rnn.bias_ih_l0 = torch.nn.Parameter(rnn.bias_ih_l0.to(torch.float))
        rnn.bias_hh_l0 = torch.nn.Parameter(rnn.bias_hh_l0.to(torch.float))



        output, (hn, cn) = rnn(input_bf16, (h0_bf16, c0_bf16))

        print(output.float())
        print(hn.float())
        print(cn)

        # print(input)
        # print(input_bf16)



if __name__ == '__main__':
    run_tests()
