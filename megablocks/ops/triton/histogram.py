# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch
import triton
import triton.lanuage as tl

# @triton.jit
# def histogram_kernel(
#     x_ptr, hist_ptr, N, num_bins: tl.constexpr, max_val: tl.constexpr
# ):
#     pid = tl.program_id(0)
#     num_programs = tl.num_programs(0)
    
#     idx = pid + tl.arange(0, 1024) * num_programs
#     mask = idx < N 

#     x = tl.load(x_ptr + idx, mask=mask, other=0.0)

#     bin_width = (max_val) / num_bins
#     bin_idx = tl.minimum(tl.maximum(((x) / bin_width).to(tl.int32), 0), num_bins - 1)

#     tl.histogram(bin_idx, hist_ptr, num_bins, mask=mask)

def histogram_fn(X, N):
    pass
#     # input:
#     #   X: tensor [bs * seq_len * top_k]
#     #   N: integer representing number of bins
#     # return:
#     #   bins: [N]
    
#     no_batch = X.ndimension() == 1
#     if no_batch:
#         X = X.view(1, X.numel())
    
#     hist = torch.zeros(num_bins, dtype=torch.int32, device=x.device)

#     grid = (triton.cdiv(N, 1024),)
#     histogram_kernel[grid](
#         x_ptr = X,
#         hist_ptr = hist,
#         N = N,
#         num_bins = num_bins,
#         max_val = max_val
#     )

#     return hist

    
# Autograd wrapper for histogram kernel.
# NOTE: Does not support gradients.
class HistogramTritonOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, max_val: float):
        return histogram_fn(x, max_val)

histogram_triton = HistogramTritonOp.apply
