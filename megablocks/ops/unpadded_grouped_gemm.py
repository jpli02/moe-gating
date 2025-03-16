import torch
from megablocks.backend.kernels import grouped_gemm
import pdb


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor, wc: torch.Tensor, wd: torch.Tensor, sizes: list[tuple[int, int, int]]):
        ctx.x = x
        w = [wa, wb, wc, wd]
        ctx.w = w
        ctx.sizes = sizes
        return grouped_gemm(x, w, sizes)


    @staticmethod
    def backward(ctx, grads):
        ## We only need to compute the gradients of x and w. ##

        ## Gradient of x = dOw^T.
        ## Gradient of w = x^Td0.
        dx = grouped_gemm(
            grads, 
            [torch.transpose(local_w, 0, 1) for local_w in ctx.w],
            [(m, k, n) for m, n, k in ctx.sizes]
            )

        with torch.no_grad():
            slice_idxs = [i[0] for i in ctx.sizes]

        sliced_xs = torch.split(ctx.x, slice_idxs)
        sliced_grads = torch.split(grads, slice_idxs)
        dw = [
            torch.matmul(torch.transpose(a, 0, 1), b) for a, b in zip(sliced_xs, sliced_grads)
        ]


        return dx, dw[0], dw[1], dw[2], dw[3], None


GroupedGemm = GroupedGemm.apply
