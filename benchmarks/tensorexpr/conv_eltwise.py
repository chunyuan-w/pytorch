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
    def __init__(self, eltwise_fn, in_channels, out_channels, kernel_size, **kwargs):
        super(M, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=True, **kwargs)
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

        self.module_layer = M(get_eltwise_fn("relu"), iC, oC, kernel_size)

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
            params_dict = {
                "kernel_size": [2, 3],
                "N": [1, 128],
                "iC": [3, 128, 256],
                "H": [16, 32],
                "W": [16, 32],
                "oC": [10, 128, 256],
                "groups": [1, 2],
            }

            params_list = []
            for key, value in params_dict.items():
                params_list.append(value)
            return itertools.product(*params_list)

        return _conv_params_list()


benchmark.register_benchmark_class(ConvEltwise)
