import torch
from .arguments import Arguments
from . import dmlp_registry
import time
from functools import partial

def test_grouped_gemm(
        num_tokens: int, hidden_dim: int,
        num_experts: int, topk: int, ffn_hidden_size: int,
        dtype: torch.dtype, args: Arguments
        ):
    args.hidden_size = hidden_dim
    args.moe_num_packed_experts = num_experts
    args.moe_num_experts = num_experts
    args.moe_top_k = topk
    args.mlp_impl = "OptGrouped"
    args.ffn_hidden_size = ffn_hidden_size

    inp = torch.randn((num_tokens, hidden_dim), dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True)

    mlp = dmlp_registry.get(args)
    grads = torch.randn_like(inp)
    for _ in range(10):
        a = mlp(inp)
        a.backward(grads, retain_graph=True)

    torch.cuda.synchronize()
    st = time.time()

    for _ in range(10):
        b = mlp(inp)
        b.backward(grads, retain_graph=True)

    torch.cuda.synchronize()
    ed = time.time()

    print(f'grouped-gemm tokens: {num_tokens}, hidden_dim: {hidden_dim}, ffn_dim: {ffn_hidden_size}, time: {(ed-st)/10}')


def test_sequential_gemm(
        num_tokens: int, hidden_dim: int,
        num_experts: int, topk: int, ffn_hidden_size: int,
        dtype: torch.dtype, args: Arguments
        ):

    assert num_tokens % num_experts == 0, 'Incorrect token count.'
    ## Loops over to compute the Gemm Sequentially.
    def internal_gemm(tokens: list[torch.Tensor], experts_l_one: list[torch.Tensor], experts_l_two: list[torch.Tensor], activation_func: torch.nn.Module):

        inter_result = []
        for t, e in zip(tokens, experts_l_one):
            inter_result.append(activation_func(torch.matmul(t, e)))

        final_result = []
        for t, e in zip(inter_result, experts_l_two):
            final_result.append(torch.matmul(t, e))

        return torch.stack(final_result)

    ## First create the requsite tensor.
    grads = torch.randn((num_tokens, hidden_dim), dtype=dtype, device="gpu" if torch.cuda.is_available() else "cpu")
    token_inps = [torch.randn((num_tokens//num_experts, hidden_dim), dtype=dtype, device="gpu" if torch.cuda.is_available() else "cpu", requires_grad=True) for _ in range(num_experts)]
    l_one_experts = [torch.randn((hidden_dim, ffn_hidden_size), dtype=dtype, device="gpu" if torch.cuda.is_available() else "cpu", requires_grad=True) for _ in range(num_experts)]
    l_two_experts = [torch.randn((ffn_hidden_size, hidden_dim), dtype=dtype, device="gpu" if torch.cuda.is_available() else "cpu", requires_grad=True) for _ in range(num_experts)]
    activ_func = torch.nn.GELU(approximate="tanh")

    for _ in range(5):
        a = internal_gemm(token_inps, l_one_experts, l_two_experts, activ_func)
        a.backward(grads, retain_grad=True)

    torch.cuda.synchronize()
    st=time.time()

    for _ in range(10):
        b = internal_gemm(token_inps, l_one_experts, l_two_experts, activ_func)
        b.backward(grads, retain_grad=True)

    torch.cuda.synchronize()
    ed=time.time()

    print(f'sequential-gemm tokens: {num_tokens}, hidden_dim: {hidden_dim}, ffn_dim: {ffn_hidden_size}, time: {(ed-st)/10}')


if __name__ == '__main__':
    pass