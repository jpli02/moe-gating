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


import numpy as np
import stk.ops
import torch
import pdb
from stk import Matrix
from functools import partial

from ..ops import histogram, sort, inclusive_cumsum, topology, padded_scatter, padded_gather, gather, scatter, round_up
#from ..ops import histogram
#import megablocks.ops as ops
from . import common, dmlp_registry, moe, mpu, dmoe
from .arguments import Arguments


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x


class OPTParallelDroplessMLP(moe.ParallelMLP):

    def __init__(self, args: Arguments):
        assert args.mlp_impl == 'OptGrouped', 'Must be called with OptGrouped impl.'
        assert args.mlp_type == 'mlp', 'Must be an MLP layer'
        super(OPTParallelDroplessMLP, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)
        self.blocking = 128
        self.mlp = dmlp_registry.get(args)
        self.args = args

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = ((self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))),
            1,
        )

    def indices_and_padded_bins(self, top_experts):
        top_experts = top_experts.int()
        bin_ids, indices = torch.sort(top_experts)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = torch.histc(top_experts, self.num_experts, 0, self.num_experts - 1)
        
        # Calculate the bin bounds for the sorted tokens.
        bins = torch.cumsum(tokens_per_expert, 0)
        return indices, bin_ids, bins, tokens_per_expert

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

    ## Construct fake input and test. ##
    def test_case(batch_size: int, num_tokens: int, hidden_dim: int, 
                  num_experts: int, dtype: torch.dtype, 
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
        args_unpadded.moe_top_k = num_experts
        args_padded.moe_top_k = num_experts
        args_unpadded.mlp_impl = "OptGrouped"  ## Extra thing to set.

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        optMoE = OPTParallelDroplessMLP(args_unpadded)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        paddedMoE = dmoe.ParallelDroplessMLP(args_padded)
        sm = torch.nn.Softmax(dim=-1)
        topk_weights, topk_args = torch.topk(sm(torch.randn((batch_size*num_tokens, num_experts), device="cuda" if torch.cuda.is_available() else "cpu")), k=num_experts)
        opt_res = optMoE.forward_once(x, topk_weights, topk_args)
        ground_truth = paddedMoE.forward_once(x_torch, topk_weights, topk_args)

        incoming_grads = torch.randn_like(opt_res[0])

        opt_res[0].backward(incoming_grads)
        ground_truth[0].backward(incoming_grads)
        
        # print(f'opt result: {opt_res[0]}')
        # print(f'non-opt result: {ground_truth[0]}')
        print(f'opt grad: {x.grad}')
        print(f'non-opt grad: {x_torch.grad}')
        print(f'max diff, fwd: {torch.abs(opt_res[0] - ground_truth[0]).max().item()}')
        print(f'max diff inps, bwd: {torch.abs(x.grad - x_torch.grad).max().item()}')
    """
    Certain constraints on megablocks:
        - m and n and k need to be multiples of 128.
        - only works on float16 (though this may be due to improper calling -> need to investigate).
    """

    ## Try a sample test case on 32-bit precision, easy for debugging. ##
    ## Just for simple correctness fill everything with the same value. Otherwise randomness is hard to check. ##
    args_padded.init_method = partial(torch.nn.init.constant_, val=0.1)
    args_padded.output_layer_init_method = partial(torch.nn.init.constant_, val=0.2)
    args_unpadded.init_method = partial(torch.nn.init.constant_, val=0.1)
    args_unpadded.output_layer_init_method = partial(torch.nn.init.constant_, val=0.2)
    test_case(1, 128, 128, 4, torch.float16, args_unpadded, args_padded)

    test_case(6, 128, 128, 4, torch.float16, args_unpadded, args_padded)

    test_case(6, 11024, 4096, 4, torch.float16, args_unpadded, args_padded)

    
