# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# Wrap this in a try-block with better error message and
# instructions for building the c++ operations.
try:
    import megablocks.backend.triton_ops as ops_triton  # type: ignore
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'megablocks_ops'.") from e


# Autograd wrapper for histogram kernel.
# NOTE: Does not support gradients.
class HistogramTritonOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, max_val: float):
        return ops_triton.histogram(x, max_val)


histogram_triton = HistogramTritonOp.apply
