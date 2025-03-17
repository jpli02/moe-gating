# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import os
import sys

#sys.path.append(os.path.join(os.path.dirname(__file__), "..", "../ops", "."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops"))
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, '/home/exouser/Desktop/moe-gating/megablocks/ops')
sys.path.insert(0, '/home/exouser/Desktop/moe-gating/megablocks/')
sys.path.insert(0, '/home/exouser/Desktop/moe-gating/megablocks/layers')
#sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ops"))
#sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))


from typing import Tuple
import numpy as np
import torch
from functools import partial
import time
import pdb

from ..ops import gather, scatter 
from . import dmlp_registry, moe, mpu, dmoe
from .arguments import Arguments


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x


#class OPTParallelDroplessMLP(moe.ParallelMLP):
class OPTParallelDroplessMLP(torch.nn.Module):

    def __init__(self, args: Arguments):
        assert args.mlp_impl == 'OptGrouped', 'Must be called with OptGrouped impl.'
        assert args.mlp_type == 'mlp', 'Must be an MLP layer'
        #super(OPTParallelDroplessMLP, self).__init__(args)
        super(OPTParallelDroplessMLP, self).__init__()
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)
        self.blocking = 128
        self.mlp = dmlp_registry.get(args)
        self.num_experts = args.moe_num_experts
        self.top_k = args.moe_top_k
        self.args = args

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = ((self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))),
            1,
        )

        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)


    def indices_and_bins(self, top_expert: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        output = torch.sort(top_expert)
        assert output is not None
        bin_ids, indices = output

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = torch.histc(top_expert, self.num_experts, 0, self.num_experts - 1)

        # Calculate the bin bounds for the sorted tokens.
        bins = torch.cumsum(tokens_per_expert, 0)
        assert bins is not None
        bins = bins.view(1) if not len(bins.size()) else bins

        return indices, bin_ids, bins.int(), tokens_per_expert

    def generate_sizes(self, tokens_per_expert):
        """Takes histogram and produces sizes list that's comprehensible by the no-padded MLP layer.
        """
        with torch.no_grad():
            m = tokens_per_expert.tolist()
            n = [self.ffn_hidden_size for _ in range(self.args.moe_num_packed_experts)]
            k = [self.hidden_size for _ in range(self.args.moe_num_packed_experts)]

            mlp_sizes = [(a, b, c) for a, b, c in zip(m, n, k)]

            return mlp_sizes

    def grouped_forward_once(self, x, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (self.indices_and_bins(top_experts))

        out = self.grouped_permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            -1,  # unused
            self.args.moe_top_k,
        )
        return out, tokens_per_expert

    def grouped_permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capactiy,  # unused, for now.
        top_k,
    ):

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = gather(x, indices, bin_ids, bins, top_k)

        # Perform the expert computation.
        x = self.mlp(x, self.generate_sizes(tokens_per_expert))

        # Un-route the data for the MoE output.
        return scatter(x, indices, bin_ids, expert_weights, bins, top_k)

    def forward_once(self, x, expert_weights, top_experts):
        assert self.args.mlp_impl == 'OptGrouped'
        return self.grouped_forward_once(x, expert_weights, top_experts)

    def permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capactiy,
        top_k,
    ):
        assert self.args.mlp_impl == 'OptGrouped', 'Should only call with mlp_impl == OptGrouped'
        return self.grouped_permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capactiy,
            top_k,
        )


class OPTdMoE(moe.MoE):

    def _init_experts_mlp(self, args: Arguments):
        return OPTParallelDroplessMLP(args)

if __name__ == '__main__':
    ## Here we test our dropless no-padding moe. ##
    ## In the single GPU case we pack with 4 experts to a GPU. ##
    from .dmoe import dMoE
    args_unpadded = Arguments()
    args_padded = Arguments()

    ## Construct fake input and test. for correctness ONLY. ##
    def test_case(batch_size: int, num_tokens: int, hidden_dim: int, 
                  num_experts: int, topk: int, dtype: torch.dtype, 
                  args_unpadded: Arguments,
                  args_padded: Arguments):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        x = torch.randn((batch_size, num_tokens, hidden_dim), 
                        dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True)
        x_torch = x.clone().detach().requires_grad_(True)
        args_unpadded.hidden_size = hidden_dim 
        args_unpadded.moe_num_packed_experts = num_experts
        args_padded.hidden_size = hidden_dim 
        args_padded.moe_num_packed_experts = num_experts  ## Extra thing to set.
        args_padded.moe_num_experts = num_experts
        args_unpadded.moe_num_experts = num_experts
        args_unpadded.moe_top_k = topk 
        args_padded.moe_top_k = topk 
        args_unpadded.mlp_impl = "OptGrouped"  ## Extra thing to set.

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        optMoE = OPTParallelDroplessMLP(args_unpadded)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        paddedMoE = dmoe.ParallelDroplessMLP(args_padded)
        sm = torch.nn.Softmax(dim=-1)
        topk_weights, topk_args = torch.topk(sm(torch.randn((batch_size*num_tokens, num_experts), device="cuda" if torch.cuda.is_available() else "cpu")), k=topk)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1):
            opt_res = optMoE.forward_once(x, topk_weights, topk_args)
        torch.cuda.synchronize()
        end = time.time()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1):
            ground_truth = paddedMoE.forward_once(x_torch, topk_weights, topk_args)
        torch.cuda.synchronize()
        end = time.time()

        incoming_grads = torch.randn_like(opt_res[0])

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1):
            opt_res[0].backward(incoming_grads, retain_graph=True)
        torch.cuda.synchronize()
        end = time.time()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1):
            ground_truth[0].backward(incoming_grads, retain_graph=True)
        torch.cuda.synchronize()
        end = time.time()

        print(f'max diff, fwd: {torch.abs(opt_res[0] - ground_truth[0]).max().item()}')
        print(f'max diff inps, bwd: {torch.abs(x.grad - x_torch.grad).max().item()}')

        ## Correctness checks for weight grads. ##
        opt_w1_grads = optMoE.mlp.w1.grad.double().sum()
        opt_w2_grads = optMoE.mlp.w2.grad.double().sum()
        ## Temp unocmment, remove once debugging finished TODO(ahangupta).
        # for w1 in optMoE.mlp.w1:
        #     opt_w1_grads += w1.grad.double().sum()

        # for w2 in optMoE.mlp.w2:
        #     opt_w2_grads += w2.grad.double().sum()

        nonopt_w1_grads = paddedMoE.mlp.w1.grad.double().sum()

        nonopt_w2_grads = paddedMoE.mlp.w2.grad.double().sum()

        print(f'avg diff w1 grads: {torch.abs(opt_w1_grads - nonopt_w1_grads)/(args_padded.ffn_hidden_size * args_padded.hidden_size)}')
        print(f'avg diff w2 grads: {torch.abs(opt_w2_grads - nonopt_w2_grads)/(args_padded.ffn_hidden_size * args_padded.hidden_size)}')

    def mem_consump(batch_size: int, num_tokens: int, hidden_dim: int, 
                    num_experts: int, topk: int, ffn_hidden_size: int,
                    dtype: torch.dtype, args_unpadded: Arguments,
                    args_padded: Arguments, custom: bool):
        args_unpadded.hidden_size = hidden_dim 
        args_unpadded.moe_num_packed_experts = num_experts
        args_padded.hidden_size = hidden_dim 
        args_padded.moe_num_packed_experts = num_experts  ## Extra thing to set.
        args_padded.moe_num_experts = num_experts
        args_unpadded.moe_num_experts = num_experts
        args_unpadded.moe_top_k = topk 
        args_padded.moe_top_k = topk 
        args_unpadded.mlp_impl = "OptGrouped"  ## Extra thing to set.
        args_padded.ffn_hidden_size = ffn_hidden_size
        args_unpadded.ffn_hidden_size = ffn_hidden_size
        if custom:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            x = torch.randn((batch_size, num_tokens, hidden_dim), 
                            dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            optMoE = OPTParallelDroplessMLP(args_unpadded)
            sm = torch.nn.Softmax(dim=-1)
            topk_weights, topk_args = torch.topk(sm(torch.randn((batch_size*num_tokens, num_experts), device="cuda" if torch.cuda.is_available() else "cpu")), k=topk)

            comp_fun = torch.compile(optMoE.forward_once)
            
            for _ in range(3):
                opt_res = optMoE.forward_once(x, topk_weights, topk_args)

            incoming_grads = torch.randn_like(opt_res[0])

            for _ in range(3):
                opt_res[0].backward(incoming_grads, retain_graph=True)

            ## Now we can test memory consumption. ##
            torch.cuda.synchronize()
            start = time.time()
            torch.cuda.memory._record_memory_history(max_entries=100000, stacks='all')
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            for _ in range(1):
                core_out = optMoE.forward_once(x, topk_weights, topk_args)
                core_out[0].backward(incoming_grads, retain_graph=True)

            torch.cuda.synchronize()
            torch.cuda.memory._dump_snapshot("perf_history")
            torch.cuda.memory._record_memory_history(enabled=None)
            end = time.time()
            peak_memory = torch.cuda.max_memory_allocated()
            peak_memory_used = peak_memory - start_memory

            print(f'opt fwd+bwd, num_tokens: {num_tokens}, topk: {topk}, num_experts: {num_experts}, peak memory: {peak_memory_used / 1024 ** 2:.2f} MB, time: {(end-start)/10:.2f}')
        else:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            x = torch.randn((batch_size, num_tokens, hidden_dim), 
                            dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu", requires_grad=True)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            paddedMoE = dmoe.ParallelDroplessMLP(args_padded)
            sm = torch.nn.Softmax(dim=-1)
            topk_weights, topk_args = torch.topk(sm(torch.randn((batch_size*num_tokens, num_experts), device="cuda" if torch.cuda.is_available() else "cpu")), k=topk)
            
            for _ in range(3):
                padded_res = paddedMoE.forward_once(x, topk_weights, topk_args)

            incoming_grads = torch.randn_like(padded_res[0])

            for _ in range(3):
                padded_res[0].backward(incoming_grads, retain_graph=True)

            ## Now we can test memory consumption. ##
            torch.cuda.synchronize()
            start = time.time()
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

            for _ in range(10):
                core_out = paddedMoE.forward_once(x, topk_weights, topk_args)
                core_out[0].backward(incoming_grads, retain_graph=True)

            torch.cuda.synchronize()
            end = time.time()
            peak_memory = torch.cuda.max_memory_allocated()
            peak_memory_used = peak_memory - start_memory

            print(f'padded fwd+bwd, num_tokens: {num_tokens}, topk: {topk}, num_experts: {num_experts}, peak memory: {peak_memory_used / 1024 ** 2:.2f} MB, time: {(end-start)/10:.2f}')

    """
    Certain constraints on megablocks:
        - m and n and k need to be multiples of 128.
        - only works on float16 (though this may be due to improper calling -> need to investigate).
    """

    test_correctness: bool = False 
    ## Try a sample test case on 16-bit precision, easy for debugging. ##
    ## Just for simple correctness fill everything with the same value. Otherwise randomness is hard to check. ##
    if test_correctness:
        args_padded.init_method = partial(torch.nn.init.constant_, val=0.1)
        args_padded.output_layer_init_method = partial(torch.nn.init.constant_, val=0.2)
        args_unpadded.init_method = partial(torch.nn.init.constant_, val=0.1)
        args_unpadded.output_layer_init_method = partial(torch.nn.init.constant_, val=0.2)
        test_case(1, 128, 128, 4, 4, torch.float16, args_unpadded, args_padded)

        test_case(6, 128, 128, 4, 4, torch.float16, args_unpadded, args_padded)

        test_case(1, 2048, 4096, 4, 4, torch.float16, args_unpadded, args_padded)
        test_case(1, 2048, 4096, 8, 4, torch.float16, args_unpadded, args_padded)
        test_case(1, 2048, 4096, 8, 6, torch.float16, args_unpadded, args_padded)

    ## More aggressive test cases with random init from normal distribution. ##
    ## However cannot get megablocks and custom mlp weights to sync up... Investigate when there's more time TODO(ahangupta). ##
    # args_padded.init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    # args_padded.output_layer_init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    # args_unpadded.init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    # args_unpadded.output_layer_init_method = partial(torch.nn.init.normal_, mean=0.0, std=0.02)
    # test_case(1, 128, 128, 4, torch.float16, args_unpadded, args_padded)


    ## True benchmark here. ##
    mem_consump(1, 4096, 7168, 8, 1, 2048, torch.float16, args_unpadded, args_padded, custom=True)
    # mem_consump(1, 4096, 7168, 8, 1, 2048, torch.float16, args_unpadded, args_padded, custom=False)
    # mem_consump(1, 4096, 7168, 8, 2, 2048, torch.float16, args_unpadded, args_padded, custom=True)
    # mem_consump(1, 4096, 7168, 8, 2, 2048, torch.float16, args_unpadded, args_padded, custom=False)
    # mem_consump(1, 4096, 7168, 8, 4, 2048, torch.float16, args_unpadded, args_padded, custom=True)
    # mem_consump(1, 4096, 7168, 8, 4, 2048, torch.float16, args_unpadded, args_padded, custom=False)
    # mem_consump(1, 4096, 7168, 8, 6, 2048, torch.float16, args_unpadded, args_padded, custom=True)
    # mem_consump(1, 4096, 7168, 8, 6, 2048, torch.float16, args_unpadded, args_padded, custom=False)
    # mem_consump(1, 4096, 7168, 8, 8, 2048, torch.float16, args_unpadded, args_padded, custom=True)
    # mem_consump(1, 4096, 7168, 8, 8, 2048, torch.float16, args_unpadded, args_padded, custom=False)


    
