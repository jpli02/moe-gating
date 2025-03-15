import torch
from megablocks.backend.kernels import grouped_gemm


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: list[torch.Tensor], sizes: list[tuple[int, int, int]]):
        ctx.x = x
        ctx.w = w

        return grouped_gemm(x, w, sizes)


    @staticmethod
    def backward(ctx, grads):
        ## TODO(ahangupta): fill out later. ##
        return None, None, None


GroupedGemm = GroupedGemm.apply
