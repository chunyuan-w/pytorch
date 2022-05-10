import itertools

from . import benchmark

import torch
import torch.nn.functional as F


def get_eltwise_fn(name):
    if hasattr(torch, name):
        return getattr(torch, name)
    elif hasattr(F, name):
        return getattr(F, name)
    else:
        raise NameError("Eltwise function %s not found" % name)


class M(torch.nn.Module):
    def __init__(self, eltwise_fn, in_channels, out_channels, kernel_size, groups, **kwargs):
        super(M, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, groups=groups, bias=True, **kwargs)
        self.eltwise = eltwise_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.eltwise(x)
        return x

class ConvEltwise(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, kernel_size, N, iC, H, W, oC, groups):
        super().__init__(mode, device, dtype)
        self.kernel_size = kernel_size
        self.N = N
        self.iC = iC
        self.H = H
        self.W = W
        self.oC = oC
        self.groups = groups
        self.inputs = [torch.randn(N, iC, H, W, device=device, requires_grad=self.requires_grad)]

        self.module_layer = M(get_eltwise_fn("relu"), iC, oC, kernel_size, groups)

    def config(self):
        return [self.kernel_size, self.N, self.iC, self.H, self.W, self.oC, self.groups]

    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = {"i": 1, "o": 1, "k": 1}
            algorithmic_count = {"i": 1, "o": 1, "k": 1}
        else:
            sol_count = {"i": 1 + 1, "o": 1 + 1, "k": 1 + 1}
            algorithmic_count = {"i": 1 + (1 + 1), "o": 1 + (1 + 1), "k": 1 + (1 + 1)}

        buffer_size = {
            "i": self.N * self.iC * self.H * self.W,
            "o": self.N * self.oC * self.H * self.W,
            "k": self.oC
            * (self.iC / self.groups)
            * self.kernel_size
            * self.kernel_size,
        }
        sol_size = 0
        algorithmic_size = 0
        for key in sol_count:
            sol_size += buffer_size[key] * sol_count[key]
            algorithmic_size += buffer_size[key] * algorithmic_count[key]
        return {"sol": sol_size, "algorithmic": algorithmic_size}

    @staticmethod
    def module():
        return "conv_eltwise"

    def compute_workload(self):
        if self.mode == "fwd":
            count = 1
        elif self.mode == "both":
            count = 1 + (1 + 1)
        else:
            raise ValueError("invalid mode: %s" % (self.mode))

        op_count = (
            self.N
            * self.iC
            / self.groups
            * self.oC
            * self.kernel_size
            * self.kernel_size
            * self.H
            * self.W
        )
        op_count *= 2

        return op_count * count

    @staticmethod
    def default_configs():
        def _conv_params_list():
            return [
                [7, 1, 3, 224, 224, 64, 1],
                [1, 1, 64, 56, 56, 256, 1], # thnn
                [3, 1, 256, 56, 56, 256, 32],
                [3, 1, 2048, 7, 7, 2048, 32],
                [1, 1, 64, 56, 56, 64, 1], # thnn
                [1, 1, 512, 28, 28, 128, 1], # thnn
                [3, 1, 64, 56, 56, 64, 1],
            ]
            # params_dict = {
            #     "kernel_size": [3],
            #     "N": [1],
            #     "iC": [128, 256],
            #     "H": [28],
            #     "W": [28],
            #     "oC": [128, 256],
            #     "groups": [1, 2],
            # }

            # params_list = []
            # for key, value in params_dict.items():
            #     params_list.append(value)
            # return itertools.product(*params_list)

        return _conv_params_list()


benchmark.register_benchmark_class(ConvEltwise)

# TODO: ref path
# conv_eltwise_fwd_cpu_3_1_128_32_32_256_1
# onednn_verbose,exec,cpu,convolution,ref:any,forward_training,src_f32::blocked:abcd:f0 wei_f32::blocked:ABcd16b16a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:abcd:f0,attr-scratchpad:user attr-post-ops:eltwise_relu ,alg:convolution_direct,mb1_ic128oc256_ih32oh30kh3sh1dh0ph0_iw32ow30kw3sw1dw0pw0,266.258
