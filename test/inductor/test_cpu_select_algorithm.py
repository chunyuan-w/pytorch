# Owner(s): ["oncall: cpu inductor"]
import contextlib
import functools
import sys
import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch._dynamo.config
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as select_algorithm
from torch._dynamo.utils import counters
from torch._inductor.cpu_vec_isa import VecAMX
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_quantization import _generate_qdq_quantized_model
from torch.testing._internal.common_quantized import (
    _calculate_dynamic_per_channel_qparams,
)
from torch.testing._internal.common_utils import IS_MACOS, parametrize, TEST_MKL


try:
    try:
        from . import test_cpu_repro, test_torchinductor
    except ImportError:
        import test_cpu_repro
        import test_torchinductor
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

check_model = test_torchinductor.check_model
set_num_threads = test_cpu_repro.set_num_threads

aten = torch.ops.aten


def patches(fn):
    def skip_cache(self, choices, name, key, benchmark):
        if benchmark is None:
            return {}
        timings = benchmark(choices)
        for choice, timing in timings.items():
            if isinstance(choice, select_algorithm.ExternKernelCaller):
                # we intentionally make ATEN kernel slower to cover the cases
                # where template kernels are always chosen with fusions applied
                # and correctness checks at runtime.
                timings[choice] = timing * 1000
        return timings

    for patcher in [
        dynamo_config.patch(verbose=True),
        # Fails due to https://github.com/pytorch/pytorch/issues/131929
        dynamo_config.patch(inline_inbuilt_nn_modules=False),
        inductor_config.patch(
            debug=True,
            max_autotune=True,
            epilogue_fusion=True,
            max_autotune_gemm_backends="CPP,ATEN",
        ),
        patch.object(select_algorithm, "VERIFY", dict(atol=1e-4, rtol=1e-4)),
        patch.object(select_algorithm.AlgorithmSelectorCache, "lookup", skip_cache),
    ]:
        fn = patcher(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        counters.clear()
        torch.manual_seed(12345)
        return fn(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def verify(dtype):
    # For bfloat16 and half, we have to relax the tolerance
    # due to the difference associave orders in different
    # kernel implementations
    atol, rtol = 1e-4, 1e-4
    if dtype == torch.half or dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    with patch.object(select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)):
        yield atol, rtol


def _get_epilogue(epilogue: str, other: Optional[torch.Tensor] = None):
    if epilogue == "none":
        return lambda x: x
    elif epilogue == "relu":
        return torch.nn.ReLU()
    elif epilogue == "gelu":
        return torch.nn.GELU()
    elif epilogue == "silu":
        return torch.nn.SiLU()
    elif epilogue == "sigmoid":
        return torch.nn.Sigmoid()
    elif epilogue == "tanh":
        return torch.nn.Tanh()
    elif epilogue == "hardswish":
        return torch.nn.Hardswish()
    elif epilogue == "hardsigmoid":
        return torch.nn.Hardsigmoid()
    elif epilogue == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif epilogue == "hardtanh":
        return torch.nn.Hardtanh()
    elif epilogue == "add":
        return lambda x: x + other
    elif epilogue == "sub":
        return lambda x: x - other
    elif epilogue == "mul":
        return lambda x: x * other
    elif epilogue == "div":
        return lambda x: x / other


class BaseTestSelectAlgorithm(TestCase):
    def _check_amx_counter(self, vec_amx):
        if vec_amx:
            self.assertTrue(counters["inductor"]["cpp_micro_gemm_amx_counter"] > 0)
        else:
            self.assertEqual(counters["inductor"]["cpp_micro_gemm_amx_counter"], 0)


class TestSelectAlgorithm(BaseTestSelectAlgorithm):
    common = check_model

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (1, 2, 1000))
    @parametrize("in_features", (1, 1000))
    @parametrize("out_features", (1, 1024))
    @parametrize("bias", (True, False))
    @parametrize("input_3d", (True, False))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_static_shapes(
        self, batch_size, in_features, out_features, bias, input_3d, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        mod = M(bias=bias).to(dtype=dtype).eval()
        B = (2, batch_size) if input_3d else (batch_size,)
        v = torch.randn(*B, in_features).to(dtype=dtype)
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        if (
            counters["inductor"]["decompose_mm"] > 0
            or counters["inductor"]["decompose_addmm"] > 0
        ):
            # This is a special case where we go directly with vectorized codegen
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)
        else:
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("bias", (True, False))
    @dtypes(torch.float)
    def test_linear_input_transpose(self, bias, dtype):
        batch_size = 384
        in_features = 196
        out_features = 384

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            @torch.compile
            def forward(self, x):
                return self.linear(x)

        counters.clear()
        mod = M(bias=bias).to(dtype=dtype).eval()
        v = torch.randn(in_features, batch_size).to(dtype=dtype)
        self.common(mod, (v.transpose(0, 1),))
        # TODO(jgong5): support transposed input
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 0)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384, 385))
    @parametrize("bias", (True, False))
    @parametrize(
        "epilogue",
        (
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "tanh",
            "hardswish",
            "hardsigmoid",
            "leaky_relu",
            "hardtanh",
            "add",
            "sub",
            "mul",
            "div",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_pointwise(
        self, batch_size, in_features, out_features, bias, epilogue, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue, other)

            def forward(self, x):
                return self.epilogue(self.linear(x))

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, epilogue=epilogue, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        if (
            (
                dtype == torch.bfloat16
                or (
                    dtype == torch.float16
                    and torch.ops.mkldnn._is_mkldnn_fp16_supported()
                )
            )
            and epilogue != "mul"
            and epilogue != "div"
            or (dtype == torch.half and epilogue == "add" and not bias)
        ):
            # Several scenarios where epilogue fusion is not counted in:
            # 1. For bfloat16, the epilogue fusion is part of the template,
            #    not fused via scheduler. This will also be true for float16 when
            #    hardware has the float16 instruction. The exception is mul or
            #    div fusion which is not supported for oneDNN linear.
            # 2. For float16, since oneDNN linear is not applied, linear w/o bias
            #    plus epilogue add is treated as linear w/ bias.
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 0)
        else:
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (128, 129))
    @parametrize("bias", (True, False))
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
            "add",
            "sub",
            "mul",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_transpose(
        self, batch_size, in_features, out_features, bias, epilogue, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                self.epilogue = _get_epilogue(epilogue, other)
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x, y):
                return self.epilogue(self.linear(x)).transpose(0, 1) + y

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(out_features, batch_size).to(dtype=dtype)
        other = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, epilogue=epilogue, other=other).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v, u), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (128,))
    @parametrize("image_size", (56,))
    @parametrize("out_features", (512,))
    @parametrize(
        "bias",
        (
            False,
            # True,
        ),
    )
    @parametrize(
        "has_non_epilogue_users",
        (
            True,
            # False,
        ),
    )
    @dtypes(torch.float32)
    def test_linear_with_permute(
        self,
        batch_size,
        in_features,
        image_size,
        out_features,
        bias,
        has_non_epilogue_users,
        dtype,
    ):
        # Reproducer from the convnext model in timm
        class M(torch.nn.Module):
            def __init__(self, bias, has_non_epilogue_users):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self._frozen_param398 = torch.randn(batch_size, out_features, 1, 1)
                self.conv = torch.nn.Conv2d(
                    out_features,
                    out_features,
                    kernel_size=7,
                    padding=3,
                    groups=out_features,
                )
                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self._frozen_param400 = torch.randn(batch_size, out_features, 1, 1)
                self.has_non_epilogue_users = has_non_epilogue_users

            def forward(self, mul_272, _convolution_pointwise_default_31):
                out1 = torch.ops.prims.convert_element_type.default(
                    mul_272, torch.float32
                )
                mul_272 = None

                _linear_pointwise_default_131 = self.linear(out1)
                permute_188 = torch.ops.aten.permute.default(
                    _linear_pointwise_default_131, [0, 3, 1, 2]
                )

                mul_273 = torch.ops.aten.mul.Tensor(permute_188, self._frozen_param398)
                add_187 = torch.ops.aten.add.Tensor(
                    mul_273, _convolution_pointwise_default_31
                )
                convert_element_type_847 = torch.ops.prims.convert_element_type.default(
                    add_187, torch.float32
                )
                _convolution_pointwise_default_29 = self.conv(convert_element_type_847)
                permute_189 = torch.ops.aten.permute.default(
                    _convolution_pointwise_default_29, [0, 2, 3, 1]
                )
                permute_189 = self.linear2(permute_189)
                permute_189 = torch.ops.aten.permute.default(permute_189, [0, 3, 1, 2])
                permute_189 = torch.ops.aten.mul.Tensor(
                    permute_189, self._frozen_param400
                )
                # If template_buffer will be used by nodes other than the epilogue nodes,
                # we can't alias the template_buffer with the Y buffer.
                if self.has_non_epilogue_users:
                    add_191 = torch.ops.aten.add.Tensor(permute_189, add_187)
                    return add_191
                return permute_189

        view_12 = torch.randn(batch_size, image_size, image_size, in_features)
        _convolution_pointwise_default_31 = torch.randn(
            batch_size, out_features, image_size, image_size
        ).to(memory_format=torch.channels_last)

        mod = M(bias=bias, has_non_epilogue_users=has_non_epilogue_users).eval()
        with verify(dtype) as (atol, rtol), torch.cpu.amp.autocast(enabled = dtype == torch.bfloat16):
            self.common(
                mod,
                (
                    view_12,
                    _convolution_pointwise_default_31,
                ),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 2)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 2)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (3,))
    @parametrize("in_features2", (192,))
    @parametrize("image_size", (224,))
    @parametrize("out_features", (64,))
    @parametrize(
        "bias",
        (
            # False,
            True,
        ),
    )
    @parametrize(
        "has_non_epilogue_users",
        (
            True,
            # False,
        ),
    )
    @dtypes(torch.float32)
    def test_linear_with_user(
        self,
        batch_size,
        in_features,
        in_features2,
        image_size,
        out_features,
        bias,
        has_non_epilogue_users,
        dtype,
    ):
        # Reproducer from the coat_lite_mini model in timm
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self._frozen_param398 = torch.randn(batch_size, out_features, 1, 1)
                self.conv = torch.nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    padding=0,
                    stride=4,
                    dilation=1,
                    groups=1,
                )
                self.conv2 = torch.nn.Conv2d(
                    out_features,
                    out_features,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    dilation=1,
                    groups=out_features,
                )

                self.conv3 = torch.nn.Conv2d(
                    16,
                    16,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    dilation=1,
                    groups=16,
                )

                self.conv4 = torch.nn.Conv2d(
                    24,
                    24,
                    kernel_size=5,
                    padding=2,
                    stride=1,
                    dilation=1,
                    groups=24,
                )

                self.conv5 = torch.nn.Conv2d(
                    24,
                    24,
                    kernel_size=7,
                    padding=3,
                    stride=1,
                    dilation=1,
                    groups=24,
                )

                self.linear = torch.nn.Linear(out_features, in_features2, bias)

                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self._frozen_param2 = torch.randn(out_features)
                self._frozen_param3 = torch.randn(out_features)
                self._frozen_param7 = torch.randn(out_features)
                self._frozen_param8 = torch.randn(out_features)
                self._frozen_param153 = torch.randn(batch_size, 1, out_features)
                self.has_non_epilogue_users = has_non_epilogue_users

            def forward(self, arg152_1):
                _convolution_pointwise_default_35 = self.conv(arg152_1)
                arg152_1 = None

                view_168 = torch.ops.aten.reshape.default(
                    _convolution_pointwise_default_35, [8, 64, 3136]
                )
                _convolution_pointwise_default_35 = None
                permute_97 = torch.ops.aten.permute.default(view_168, [0, 2, 1])
                view_168 = None
                clone_65 = torch.ops.aten.clone.default(
                    permute_97, memory_format=torch.contiguous_format
                )
                permute_97 = None
                var_mean_21 = torch.ops.aten.var_mean.correction(
                    clone_65, [2], correction=0, keepdim=True
                )
                getitem_90 = var_mean_21[0]
                getitem_91 = var_mean_21[1]
                var_mean_21 = None
                add_82 = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
                getitem_90 = None
                rsqrt_21 = torch.ops.aten.rsqrt.default(add_82)
                add_82 = None
                sub_29 = torch.ops.aten.sub.Tensor(clone_65, getitem_91)
                clone_65 = getitem_91 = None
                mul_82 = torch.ops.aten.mul.Tensor(sub_29, rsqrt_21)
                sub_29 = rsqrt_21 = None
                mul_83 = torch.ops.aten.mul.Tensor(mul_82, self._frozen_param2)
                mul_82 = None
                add_83 = torch.ops.aten.add.Tensor(mul_83, self._frozen_param3)
                mul_83 = None
                _frozen_param153 = self._frozen_param153
                cat_20 = torch.ops.aten.cat.default([_frozen_param153, add_83], 1)
                _frozen_param153 = add_83 = None
                slice_111 = torch.ops.aten.slice.Tensor(cat_20, 1, 0, 1)
                slice_113 = torch.ops.aten.slice.Tensor(
                    cat_20, 1, 1, 9223372036854775807
                )
                cat_20 = None
                permute_98 = torch.ops.aten.permute.default(slice_113, [0, 2, 1])
                slice_113 = None
                view_169 = torch.ops.aten.reshape.default(permute_98, [8, 64, 56, 56])
                permute_98 = None
                _convolution_pointwise_default_34 = self.conv2(view_169)

                add_84 = torch.ops.aten.add.Tensor(
                    _convolution_pointwise_default_34, view_169
                )
                _convolution_pointwise_default_34 = view_169 = None
                view_170 = torch.ops.aten.reshape.default(add_84, [8, 64, 3136])
                add_84 = None
                permute_99 = torch.ops.aten.permute.default(view_170, [0, 2, 1])
                view_170 = None
                cat_21 = torch.ops.aten.cat.default([slice_111, permute_99], 1)
                slice_111 = permute_99 = None
                var_mean_22 = torch.ops.aten.var_mean.correction(
                    cat_21, [2], correction=0, keepdim=True
                )
                getitem_92 = var_mean_22[0]
                getitem_93 = var_mean_22[1]
                var_mean_22 = None
                add_85 = torch.ops.aten.add.Tensor(getitem_92, 1e-06)
                getitem_92 = None
                rsqrt_22 = torch.ops.aten.rsqrt.default(add_85)
                add_85 = None
                sub_30 = torch.ops.aten.sub.Tensor(cat_21, getitem_93)
                getitem_93 = None
                mul_84 = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22)
                sub_30 = rsqrt_22 = None
                mul_85 = torch.ops.aten.mul.Tensor(mul_84, self._frozen_param7)
                mul_84 = None
                add_86 = torch.ops.aten.add.Tensor(mul_85, self._frozen_param8)
                mul_85 = None
                view_171 = torch.ops.aten.reshape.default(add_86, [25096, 64])
                add_86 = None

                _mkl_linear_32 = self.linear(view_171)
                view_171 = None

                view_172 = torch.ops.aten.reshape.default(
                    _mkl_linear_32, [8, 3137, 192]
                )
                _mkl_linear_32 = None
                view_173 = torch.ops.aten.reshape.default(view_172, [8, 3137, 3, 8, 8])
                view_172 = None
                permute_101 = torch.ops.aten.permute.default(view_173, [2, 0, 3, 1, 4])
                view_173 = None
                unbind_8 = torch.ops.aten.unbind.int(permute_101)
                permute_101 = None
                getitem_94 = unbind_8[0]
                getitem_95 = unbind_8[1]
                getitem_96 = unbind_8[2]
                unbind_8 = None
                clone_66 = torch.ops.aten.clone.default(
                    getitem_95, memory_format=torch.contiguous_format
                )
                getitem_95 = None
                amax_8 = torch.ops.aten.amax.default(clone_66, [2], True)
                sub_31 = torch.ops.aten.sub.Tensor(clone_66, amax_8)
                clone_66 = amax_8 = None
                exp_8 = torch.ops.aten.exp.default(sub_31)
                sub_31 = None
                sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [2], True)
                div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9)
                exp_8 = sum_9 = None
                permute_102 = torch.ops.aten.permute.default(div_8, [0, 1, 3, 2])
                div_8 = None
                expand_37 = torch.ops.aten.expand.default(permute_102, [8, 8, 8, 3137])
                permute_102 = None
                view_174 = torch.ops.aten.reshape.default(expand_37, [64, 8, 3137])
                expand_37 = None
                expand_38 = torch.ops.aten.expand.default(getitem_96, [8, 8, 3137, 8])
                clone_67 = torch.ops.aten.clone.default(
                    expand_38, memory_format=torch.contiguous_format
                )
                expand_38 = None
                view_175 = torch.ops.aten.reshape.default(clone_67, [64, 3137, 8])
                clone_67 = None
                bmm_16 = torch.ops.aten.bmm.default(view_174, view_175)
                view_174 = view_175 = None
                view_176 = torch.ops.aten.reshape.default(bmm_16, [8, 8, 8, 8])
                bmm_16 = None
                expand_39 = torch.ops.aten.expand.default(getitem_94, [8, 8, 3137, 8])
                clone_68 = torch.ops.aten.clone.default(
                    expand_39, memory_format=torch.contiguous_format
                )
                expand_39 = None
                view_177 = torch.ops.aten.reshape.default(clone_68, [64, 3137, 8])
                clone_68 = None
                expand_40 = torch.ops.aten.expand.default(view_176, [8, 8, 8, 8])
                view_176 = None
                view_178 = torch.ops.aten.reshape.default(expand_40, [64, 8, 8])
                expand_40 = None
                bmm_17 = torch.ops.aten.bmm.default(view_177, view_178)
                view_177 = view_178 = None
                view_179 = torch.ops.aten.reshape.default(bmm_17, [8, 8, 3137, 8])
                bmm_17 = None
                slice_116 = torch.ops.aten.slice.Tensor(
                    getitem_94, 2, 1, 9223372036854775807
                )
                getitem_94 = None
                slice_120 = torch.ops.aten.slice.Tensor(
                    getitem_96, 2, 1, 9223372036854775807
                )
                getitem_96 = None
                permute_103 = torch.ops.aten.permute.default(slice_120, [0, 1, 3, 2])
                slice_120 = None
                view_180 = torch.ops.aten.reshape.default(permute_103, [8, 64, 56, 56])
                permute_103 = None
                split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(
                    view_180, [16, 24, 24], 1
                )
                view_180 = None
                getitem_97 = split_with_sizes_8[0]
                getitem_98 = split_with_sizes_8[1]
                getitem_99 = split_with_sizes_8[2]
                split_with_sizes_8 = None

                _convolution_pointwise_default_33 = self.conv3(getitem_97)
                _convolution_pointwise_default_32 = self.conv4(getitem_98)
                _convolution_pointwise_default_31 = self.conv5(getitem_99)

                cat_22 = torch.ops.aten.cat.default(
                    [
                        _convolution_pointwise_default_33,
                        _convolution_pointwise_default_32,
                        _convolution_pointwise_default_31,
                    ],
                    1,
                )
                _convolution_pointwise_default_33 = (
                    _convolution_pointwise_default_32
                ) = _convolution_pointwise_default_31 = None
                view_181 = torch.ops.aten.reshape.default(cat_22, [8, 8, 8, 3136])
                cat_22 = None
                permute_104 = torch.ops.aten.permute.default(view_181, [0, 1, 3, 2])
                view_181 = None

                mul_86 = torch.ops.aten.mul.Tensor(slice_116, permute_104)
                slice_116 = permute_104 = None
                constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(
                    mul_86, [0, 0, 1, 0, 0, 0], 0.0
                )
                mul_86 = None
                mul_87 = torch.ops.aten.mul.Tensor(view_179, 0.3535533905932738)
                view_179 = None
                add_87 = torch.ops.aten.add.Tensor(mul_87, constant_pad_nd_8)
                mul_87 = constant_pad_nd_8 = None
                return add_87

        view_12 = torch.randn(batch_size, in_features, image_size, image_size)

        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol), torch.cpu.amp.autocast(
            enabled=dtype == torch.bfloat16
        ):
            self.common(
                mod,
                (view_12,),
                atol=atol,
                rtol=rtol,
            )
        # self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 2)
        # self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 2)

    # TODO: this UT can pass
    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (8,))
    @parametrize("in_features", (512,))
    @parametrize("image_size", (56,))
    @parametrize("out_features", (128,))
    @parametrize(
        "bias",
        (
            False,
            # True,
        ),
    )
    @parametrize(
        "has_non_epilogue_users",
        (
            True,
            # False,
        ),
    )
    @dtypes(torch.float32)
    def test_linear_reindexer(
        self,
        batch_size,
        in_features,
        image_size,
        out_features,
        bias,
        has_non_epilogue_users,
        dtype,
    ):
        # Reproducer from the convnext model in timm
        class M(torch.nn.Module):
            def __init__(self, bias, has_non_epilogue_users):
                super().__init__()
                self.linear2 = torch.nn.Linear(128, 128, bias=True)
                
                self._frozen_param2 = torch.randn(1, 16, 196, 128)
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self._frozen_param398 = torch.randn(batch_size, out_features, 1, 1)

                self._frozen_param414 = torch.randn(384, 128)
                self._frozen_param415 = torch.randn(1982689, 1)

                self._frozen_param15 = torch.randn(128)
                self._frozen_param16 = torch.randn(128)

                self.has_non_epilogue_users = has_non_epilogue_users

            def forward(self, mul_230, view_408, _convolution_pointwise_default_2):

                # File: /home/chunyuan/miniforge3/envs/inductor/lib/python3.10/site-packages/timm/models/nest.py:233 in forward, code: x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
                permute_187: "f32[8, 56, 56, 128][401408, 7168, 128, 1]cpu" = torch.ops.aten.permute.default(_convolution_pointwise_default_2, [0, 2, 3, 1])

                # File: /home/chunyuan/miniforge3/envs/inductor/lib/python3.10/site-packages/timm/models/nest.py:159 in blockify, code: x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
                view_397: "f32[8, 4, 14, 4, 14, 128][401408, 100352, 7168, 1792, 128, 1]cpu" = torch.ops.aten.reshape.default(permute_187, [8, 4, 14, 4, 14, 128])

                # File: /home/chunyuan/miniforge3/envs/inductor/lib/python3.10/site-packages/timm/models/nest.py:160 in blockify, code: x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
                permute_188: "f32[8, 4, 4, 14, 14, 128][401408, 100352, 1792, 7168, 128, 1]cpu" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2, 4, 5])
                clone_173: "f32[8, 4, 4, 14, 14, 128][401408, 100352, 25088, 1792, 128, 1]cpu" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format)
                view_398: "f32[8, 16, 196, 128][401408, 25088, 128, 1]cpu" = torch.ops.aten.reshape.default(clone_173, [8, 16, 196, 128])
                
                
                add_177: "f32[8, 16, 196, 128][401408, 25088, 128, 1]cpu" = torch.ops.aten.add.Tensor(view_398, self._frozen_param2)


                # File: /home/chunyuan/miniforge3/envs/inductor/lib/python3.10/site-packages/timm/models/nest.py:79 in forward, code: x = self.proj(x)
                view_409: "f32[25088, 128][128, 1]cpu" = torch.ops.aten.reshape.default(view_408, [25088, 128])
                _mkl_linear_95: "f32[25088, 128][128, 1]cpu" = self.linear2(view_409)
                view_410: "f32[8, 16, 196, 128][401408, 25088, 128, 1]cpu" = torch.ops.aten.reshape.default(_mkl_linear_95, [8, 16, 196, 128])


                add_180: "f32[8, 16, 196, 128][401408, 25088, 128, 1]cpu" = torch.ops.aten.add.Tensor(add_177, view_410)
                
                
                view_413 = torch.ops.aten.reshape.default(mul_230, [25088, 512])
                _mkl_linear_93 = self.linear(view_413)
                view_414 = torch.ops.aten.reshape.default(_mkl_linear_93, [8, 16, 196, 128])
                add_184 = torch.ops.aten.add.Tensor(add_180, view_414)


                # # File: /home/chunyuan/miniforge3/envs/inductor/lib/python3.10/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                # var_mean_53 = torch.ops.aten.var_mean.correction(add_184, [3], correction = 0, keepdim = True)
                # getitem_185: "f32[8, 16, 196, 1][3136, 196, 1, 1]cpu" = var_mean_53[0]
                # getitem_186: "f32[8, 16, 196, 1][3136, 196, 1, 1]cpu" = var_mean_53[1];  var_mean_53 = None

                # # No stacktrace found for following nodes
                # _frozen_param414: "f32[384, 128][128, 1]cpu" = self._frozen_param414
                # _frozen_param415: "f32[1982689, 1][1, 0]cpu" = self._frozen_param415

                # # File: /home/chunyuan/miniforge3/envs/inductor/lib/python3.10/site-packages/timm/layers/norm.py:57 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                # sub_78 = torch.ops.aten.sub.Tensor(add_184, getitem_186);  getitem_186 = None
                # add_185 = torch.ops.aten.add.Tensor(getitem_185, 1e-06);  getitem_185 = None
                # rsqrt_53 = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
                # mul_231 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_53);  sub_78 = rsqrt_53 = None
                # mul_232 = torch.ops.aten.mul.Tensor(mul_231, self._frozen_param15);  mul_231 = _frozen_param15 = None
                # add_186 = torch.ops.aten.add.Tensor(mul_232, self._frozen_param16);  mul_232 = _frozen_param16 = None
                # return add_186
                return add_184

        view_12 = torch.randn(batch_size, 16, 196, 512)
        view_408 = torch.randn(batch_size, 16, 196, 128)
        _convolution_pointwise_default_2 = torch.randn(batch_size, 128, 56, 56).to(memory_format=torch.channels_last)

        mod = M(bias=bias, has_non_epilogue_users=has_non_epilogue_users).eval()
        with verify(dtype) as (atol, rtol), torch.cpu.amp.autocast(enabled = dtype == torch.bfloat16):
            self.common(
                mod,
                (
                    view_12,
                    view_408,
                    _convolution_pointwise_default_2,
                ),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 2)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384, 385))
    @parametrize("bias", (True, False))
    @parametrize(
        "unary",
        ("relu",),
    )
    @parametrize(
        "binary",
        (
            "add",
            "sub",
            "mul",
            "div",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_with_unary_binary(
        self, batch_size, in_features, out_features, bias, unary, binary, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, unary, binary, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.unary = _get_epilogue(unary)
                self.binary = _get_epilogue(binary, other)

            def forward(self, x):
                return self.binary(self.unary(self.linear(x)))

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, unary=unary, binary=binary, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024, 1025))
    @parametrize("bias", (True, False))
    @dtypes(torch.bfloat16)
    def test_linear_amx(self, batch_size, in_features, out_features, bias, dtype):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        vec_amx = VecAMX()
        self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (384,))
    @parametrize("in_features", (196,))
    @parametrize("out_features", (384,))
    @parametrize("bias", (True, False))
    @dtypes(torch.bfloat16)
    def test_linear_with_embedding(
        self, batch_size, in_features, out_features, bias, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias).to(
                    dtype=dtype
                )
                self.emb = torch.nn.Embedding(64, out_features)

            def forward(self, idx, x):
                return self.emb(idx) + self.linear(x)

        idx = torch.randint(0, 64, (batch_size,))
        x = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (idx, x), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (2,))
    @parametrize("in_features", (16,))
    @parametrize("seq_lens", (128,))
    @parametrize("out_features", (32,))
    @parametrize("bias", (True,))
    @dtypes(torch.bfloat16)
    def test_linear_with_indirect_indexing(
        self, batch_size, in_features, seq_lens, out_features, bias, dtype
    ):
        # Reproducer from the GPT2ForSequenceClassification model in HuggingFace
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.wte = torch.nn.Embedding(128, seq_lens)
                self.wpe = torch.nn.Embedding(in_features, seq_lens)
                self.linear = torch.nn.Linear(out_features, seq_lens, bias)

            def forward(self, view_12, input_ids, view_9):
                inputs_embeds = self.wte(input_ids)

                position_ids = torch.arange(0, in_features, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0)
                position_embeds = self.wpe(position_ids)

                add = inputs_embeds + position_embeds
                add_4 = view_9 + add

                _linear_pointwise_default_45 = self.linear(view_12)

                view_13 = torch.ops.aten.reshape.default(
                    _linear_pointwise_default_45, [batch_size, in_features, seq_lens]
                )
                out = torch.ops.aten.add.Tensor(add_4, view_13)

                return out

        view_12 = torch.randn(batch_size * in_features, out_features)
        input_ids = torch.randint(0, 128, (batch_size, in_features))
        view_9 = torch.randn(batch_size, in_features, seq_lens)
        mod = M(bias=bias).eval()
        with verify(dtype) as (atol, rtol), torch.cpu.amp.autocast():
            self.common(
                mod,
                (
                    view_12,
                    input_ids,
                    view_9,
                ),
                atol=atol,
                rtol=rtol,
            )
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 1)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (32,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64, 65))
    @parametrize("bias", (False, True))
    @parametrize("input_3d", (False, True))
    @dtypes(torch.float32, torch.bfloat16)
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
            "gelu",
        ),
    )
    def test_quantized_linear_with_pointwise(
        self, batch_size, in_features, out_features, bias, input_3d, dtype, epilogue
    ):
        B = (2, batch_size) if input_3d else (batch_size,)
        input = torch.randn(*B, in_features).to(dtype=torch.float32)

        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue)
                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self.epilogue2 = _get_epilogue(epilogue)

            def forward(self, x):
                res = self.epilogue(self.linear(x))
                res = self.epilogue2(self.linear2(res))
                return res

        counters.clear()
        ref_quantized_mod = _generate_qdq_quantized_model(
            M(bias=bias).eval(),
            (input,),
        )

        atol, rtol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            atol, rtol = 5e-2, 5e-2

        with patch.object(
            select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)
        ), torch.no_grad(), torch.autocast(
            "cpu", enabled=(dtype == torch.bfloat16), dtype=dtype
        ):
            ref_res = ref_quantized_mod(input)
            cfn = torch.compile(ref_quantized_mod)
            res = cfn(input)
            self.assertEqual(
                res,
                ref_res,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=True,
            )
            self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 2)
            self.assertEqual(counters["inductor"]["cpp_epilogue_fusion_counter"], 0)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @dtypes(torch.bfloat16)
    @parametrize("batch_size", (32,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64, 65))
    def test_int8_woq_mm(self, dtype, batch_size, in_features, out_features):
        # x will be reshaped from 3d to 2d
        second_dim_size = 8

        def _convert_weight_to_int8pack(w):
            scale, zp = _calculate_dynamic_per_channel_qparams(
                w.to(torch.float), torch.int8
            )
            scale = torch.from_numpy(scale)
            zp = torch.from_numpy(zp)
            w_int8 = torch.ao.quantization.fx._decomposed.quantize_per_channel(
                input=w,
                scales=scale,
                zero_points=zp,
                axis=0,
                quant_min=-128,
                quant_max=127,
                dtype=torch.int8,
            )
            return w_int8, scale.to(torch.bfloat16)

        class M(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.linear_weight = torch.nn.Parameter(w, requires_grad=False)

            def forward(self, x, scale):
                return (
                    torch.nn.functional.linear(x, self.linear_weight.to(x.dtype))
                    * scale
                )

        counters.clear()
        # Currently, the corresponding torch.fx pattern only supports 3D x
        # Add 2D X case once the corresponding pattern-matcher pattern is added
        x = torch.rand((batch_size, second_dim_size, in_features), dtype=dtype)
        w = torch.rand((out_features, in_features), dtype=dtype)
        w_int8pack, w_scales = _convert_weight_to_int8pack(w)
        mod = M(w_int8pack).eval()
        self.common(mod, (x, w_scales))
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        vec_amx = VecAMX()
        self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (32,))
    @parametrize("in_features", (128,))
    @parametrize("out_features", (64, 65))
    @parametrize("bias", (False, True))
    @parametrize("input_3d", (False, True))
    @parametrize("int8_mixed_bf16", (False, True))
    @dtypes(torch.float32, torch.bfloat16)
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
        ),
    )
    def test_quantized_linear_with_pointwise_binary(
        self,
        batch_size,
        in_features,
        out_features,
        bias,
        input_3d,
        int8_mixed_bf16,
        dtype,
        epilogue,
    ):
        if not int8_mixed_bf16 and dtype == torch.bfloat16:
            return
        B = (2, batch_size) if input_3d else (batch_size,)
        input = torch.randn(*B, in_features).to(dtype=torch.float32)

        other = torch.randn(*B, out_features).to(dtype=dtype)
        # Avoid hiting qlinear inplace sum fusion
        if input_3d:
            other2 = torch.randn(B[0] * B[1], out_features).to(dtype=dtype)
        else:
            other2 = torch.randn(1, *B, out_features).to(dtype=dtype)

        class M(torch.nn.Module):
            def __init__(self, bias, input_3d):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue)
                self.linear2 = torch.nn.Linear(out_features, out_features, bias)
                self.epilogue2 = _get_epilogue(epilogue)
                self.input_3d = input_3d

            def forward(self, x, other, other2):
                res = self.epilogue(self.linear(x) + other)
                # Avoid hiting qlinear inplace sum fusion
                if self.input_3d:
                    other2 = other2.view(2, other2.size(0) // 2, other2.size(1))
                else:
                    other2 = other2.view(other2.size(1), other2.size(2))
                res = self.epilogue2(self.linear2(res) + other2)
                return res

        counters.clear()
        ref_quantized_mod = _generate_qdq_quantized_model(
            M(bias=bias, input_3d=input_3d).eval(),
            (input, other, other2),
        )
        atol, rtol = 5e-2, 5e-2
        with patch.object(
            select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)
        ), torch.no_grad(), torch.autocast(
            "cpu", enabled=int8_mixed_bf16, dtype=torch.bfloat16
        ):
            ref_res = ref_quantized_mod(input, other, other2)
            cfn = torch.compile(ref_quantized_mod)
            res = cfn(input, other, other2)
            self.assertEqual(
                res,
                ref_res,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
                exact_dtype=True,
            )
            self.assertEqual(
                counters["inductor"]["select_algorithm_autotune"],
                2,
            )
            self.assertEqual(
                counters["inductor"]["cpp_epilogue_fusion_counter"],
                0,
            )

    @inductor_config.patch({"freezing": True})
    @patches
    @torch.no_grad
    @parametrize("batch_size", (3, 16, 32, 49))
    @parametrize("in_features", (4, 68, 128))  # k should be a multiple of 4
    @parametrize("out_features", (64, 65))
    @parametrize("bias", (True, False))
    def test_quantized_linear_amx(self, batch_size, in_features, out_features, bias):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=torch.float32)
        ref_quantized_mod = _generate_qdq_quantized_model(
            M(bias=bias).eval(),
            (v,),
        )
        atol, rtol = 1e-2, 1e-2
        with patch.object(select_algorithm, "VERIFY", dict(atol=atol, rtol=rtol)):
            self.common(ref_quantized_mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)
        vec_amx = VecAMX()
        self._check_amx_counter(vec_amx)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.gemm_max_k_slices": 0})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @parametrize("batch_size", (2,))
    @parametrize("in_features", (1000,))
    @parametrize("out_features", (2,))
    @parametrize("bias", (True, False))
    @parametrize(
        "epilogue",
        (
            "none",
            "relu",
        ),
    )
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_k_slicing(
        self, batch_size, in_features, out_features, bias, epilogue, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias, epilogue, other):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)
                self.epilogue = _get_epilogue(epilogue, other)

            def forward(self, x):
                return self.epilogue(self.linear(x))

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        u = torch.randn(batch_size, out_features).to(dtype=dtype)
        mod = M(bias=bias, epilogue=epilogue, other=u).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.gemm_cache_blocking": "2,2,2"})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @set_num_threads(1)
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024,))
    @parametrize("bias", (True, False))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_cache_blocking(
        self, batch_size, in_features, out_features, bias, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)

    @inductor_config.patch({"freezing": True})
    @inductor_config.patch({"cpp.gemm_thread_factors": "4,2,7"})
    @patches
    @torch.no_grad
    @unittest.skipIf(not TEST_MKL, "Test requires MKL")
    @set_num_threads(56)
    @parametrize("batch_size", (1024,))
    @parametrize("in_features", (1024,))
    @parametrize("out_features", (1024,))
    @parametrize("bias", (True, False))
    @dtypes(torch.float, torch.bfloat16, torch.half)
    def test_linear_thread_factors(
        self, batch_size, in_features, out_features, bias, dtype
    ):
        class M(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias)

            def forward(self, x):
                return self.linear(x)

        counters.clear()
        v = torch.randn(batch_size, in_features).to(dtype=dtype)
        mod = M(bias=bias).to(dtype=dtype).eval()
        with verify(dtype) as (atol, rtol):
            self.common(mod, (v,), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["select_algorithm_autotune"], 1)


@dynamo_config.patch({"dynamic_shapes": True, "assume_static_by_default": False})
class _DynamicShapesTestBase(BaseTestSelectAlgorithm):
    pass


class TestSelectAlgorithmDynamicShapes(_DynamicShapesTestBase):
    common = check_model
    test_linear_dynamic_shapes = TestSelectAlgorithm.test_linear_static_shapes
    test_linear_with_pointwise_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_pointwise
    )
    test_linear_with_transpose_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_transpose
    )
    test_linear_with_unary_binary_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_unary_binary
    )
    test_linear_amx_dynamic_shapes = TestSelectAlgorithm.test_linear_amx
    test_linear_with_embedding_dynamic_shapes = (
        TestSelectAlgorithm.test_linear_with_embedding
    )
    test_quantized_linear_with_pointwise_dynamic_shapes = (
        TestSelectAlgorithm.test_quantized_linear_with_pointwise
    )
    test_quantized_linear_with_pointwise_binary_dynamic_shapes = (
        TestSelectAlgorithm.test_quantized_linear_with_pointwise_binary
    )
    test_quantized_linear_amx_dynamic_shapes = (
        TestSelectAlgorithm.test_quantized_linear_amx
    )


instantiate_device_type_tests(TestSelectAlgorithm, globals(), only_for="cpu")
instantiate_device_type_tests(
    TestSelectAlgorithmDynamicShapes, globals(), only_for="cpu"
)


if __name__ == "__main__":
    from torch.testing._internal.inductor_utils import HAS_CPU

    if HAS_CPU and not IS_MACOS:
        run_tests()
