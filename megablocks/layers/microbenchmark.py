import torch
from .arguments import Arguments
from . import dmlp_registry
import time
from functools import partial, reduce

def test_grouped_gemm(
        num_tokens: int, hidden_dim: int,
        num_experts: int, topk: int, ffn_hidden_size: int,
        dtype: torch.dtype, args: Arguments, token_dist : list[int]  ## Token_dist gives a list of ints representing the token count per expert.
        ):
    assert len(token_dist) == num_experts, 'incorrect token_dist length'
    assert reduce(lambda a, b: a+b, token_dist) == num_tokens, 'incorrect token_dist composition'
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
        a = mlp(inp, [(cnt, ffn_hidden_size, hidden_dim) for cnt in token_dist])
        a.backward(grads, retain_graph=True)

    torch.cuda.synchronize()
    st = time.time()

    for _ in range(10):
        b = mlp(inp, [(cnt, ffn_hidden_size, hidden_dim) for cnt in token_dist])
        b.backward(grads, retain_graph=True)

    torch.cuda.synchronize()
    ed = time.time()

    print(f'grouped-gemm tokens: {num_tokens}, hidden_dim: {hidden_dim}, ffn_dim: {ffn_hidden_size}, experts: {num_experts}, token_dist: {token_dist} time: {(ed-st)/10}')


def test_sequential_gemm(
        num_tokens: int, hidden_dim: int,
        num_experts: int, topk: int, ffn_hidden_size: int,
        dtype: torch.dtype, args: Arguments, token_dist: list[int]  ## Token distribution
        ):
    assert len(token_dist) == num_experts, 'incorrect token_dist size.'
    assert reduce(lambda a,b: a+b, token_dist) == num_tokens, 'incorrect token_dist composition.'
    assert num_tokens % num_experts == 0, 'Incorrect token count.'
    ## Loops over to compute the Gemm Sequentially.
    def internal_gemm(tokens: list[torch.Tensor], experts_l_one: list[torch.Tensor], experts_l_two: list[torch.Tensor], activation_func: torch.nn.Module):

        inter_result = []
        for t, e in zip(tokens, experts_l_one):
            inter_result.append(activation_func(torch.matmul(t, e)))

        final_result = []
        for t, e in zip(inter_result, experts_l_two):
            final_result.append(torch.matmul(t, e))

        return torch.cat(final_result, dim=0)

    ## First create the requsite tensor.
    grads = torch.randn((num_tokens, hidden_dim), dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu")
    token_inps = [torch.randn((cnt, hidden_dim), dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True) for cnt in token_dist]
    l_one_experts = [torch.randn((hidden_dim, ffn_hidden_size), dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True) for _ in range(num_experts)]
    l_two_experts = [torch.randn((ffn_hidden_size, hidden_dim), dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True) for _ in range(num_experts)]
    activ_func = torch.nn.GELU(approximate="tanh")

    for _ in range(10):
        a = internal_gemm(token_inps, l_one_experts, l_two_experts, activ_func)
        a.backward(grads, retain_graph=True)

    torch.cuda.synchronize()
    st=time.time()

    for _ in range(10):
        b = internal_gemm(token_inps, l_one_experts, l_two_experts, activ_func)
        b.backward(grads, retain_graph=True)

    torch.cuda.synchronize()
    ed=time.time()

    print(f'sequential-gemm tokens: {num_tokens}, hidden_dim: {hidden_dim}, ffn_dim: {ffn_hidden_size}, experts: {num_experts} token_dist: {token_dist} time: {(ed-st)/10}')

def two_two_split(ratio : float, num_tokens : int, num_experts : int):
    #assert num_experts == 4, 'Incorrect expert count'
    half_experts = num_experts // 2
    first = [round((ratio/(half_experts*ratio+half_experts))*num_tokens) for _ in range(num_experts // 2)]
    second = [round((1/(half_experts*ratio+half_experts))*num_tokens) for _ in range(num_experts // 2)]

    assert reduce(lambda a,b:a+b, first+second) == num_tokens, 'incorrect token count'
    return first + second

def even_split(num_tokens: int, num_experts: int):
    return [num_tokens//num_experts for _ in range(num_experts)]

def irregular_split(num_tokens: int, num_experts: int):
    import random
    ## Seed for reproducibility. ##
    random.seed(0)
    assert num_tokens % num_experts == 0, 'incorrect token and/or expert count.'
    ## we generate random numbers between [0, num_tokens//num_experts].

    token_dist = [round(random.random() * (num_tokens//num_experts)) for _ in range(num_experts - 1)]

    total_token_cnt = reduce(lambda a,b: a+b, token_dist)
    assert total_token_cnt > 0, 'incorrect distribution generated.'
    return token_dist + [num_tokens - total_token_cnt]

if __name__ == '__main__':
    token_cnt = [1024]
    #token_cnt = [1024, 2048, 4096, 16384, 32768, 16384]
    #inner_dimensions = [(2048, 1408), (5120, 1536), (7168, 2048)]
    #inner_dimensions = [(7168, 2048)]
    inner_dimensions = [(5120, 1536)]
    #inner_dimensions = [(2048, 1408)]
    args = Arguments()
    num_experts = 32
    ratio = 64
    for tc in token_cnt:
        for hid_dim, ffn_dim in inner_dimensions:
            #test_grouped_gemm(tc, hid_dim, num_experts, 8, ffn_dim, torch.float16, args, two_two_split(ratio, tc, num_experts))
            test_grouped_gemm(tc, hid_dim, num_experts, 8, ffn_dim, torch.float16, args, irregular_split(tc, num_experts))

    for tc in token_cnt:
        for hid_dim, ffn_dim in inner_dimensions:
            #test_sequential_gemm(tc, hid_dim, num_experts, 8, ffn_dim, torch.float16, args, two_two_split(ratio, tc, num_experts))
            test_sequential_gemm(tc, hid_dim, num_experts, 8, ffn_dim, torch.float16, args, irregular_split(tc, num_experts))
