# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import stk.ops
import torch
from stk import Matrix

import megablocks.ops as ops
from megablocks.layers import common, dmlp_registry, moe, mpu
from megablocks.layers.arguments import Arguments


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

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = ((self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))),
            1,
        )

    def indices_and_padded_bins(self, top_experts):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_experts = top_experts.int()
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, tokens_per_expert

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
        x = ops.gather(x, indices, bin_ids, bins, top_k)

        # Perform the expert computation.
        x = self.mlp(x, tokens_per_expert)

        # Un-route the data for the MoE output.
        return ops.scatter(x, indices, bin_ids, expert_weights, bins, top_k)

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