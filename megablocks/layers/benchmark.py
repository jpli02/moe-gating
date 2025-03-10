# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
# Jianping Li: benchmarking gating kernel mem & speed

import numpy as np
import stk.ops
import torch
from stk import Matrix
import argparse
import time

import megablocks.ops as ops
from megablocks.layers import common, dmlp_registry, moe, mpu
from megablocks.layers.arguments import Arguments

torch.manual_seed(0)


def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x


class ParallelDroplessMLP(moe.ParallelMLP):

    def __init__(self, args: Arguments):
        super(ParallelDroplessMLP, self).__init__(args)
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = mpu.features_per_rank(args)
        self.blocking = 128
        # self.mlp = dmlp_registry.get(args)

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = ((self.ffn_hidden_size * self.num_experts) // self.blocking)
        self.transpose_sort_end_bit = max(
            int(np.ceil(np.log2(max_column_index))),
            1,
        )

    def sparse_transpose(self, size, row_indices, column_indices, offsets):
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input values.
        # To avoid overflow when we have large activation matrices we cast to
        # 32-bit before sorting.
        _, gather_indices = ops.sort(
            column_indices.int(),
            self.transpose_sort_end_bit,
        )

        # There are a constant number of blocks in every row of the sparse matrix.
        # A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can divide
        # by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        if nnz_per_column.dim() == 0:
            # This addresses an edge case when ffn_hidden_size is equal to self.blocking.
            nnz_per_column = nnz_per_column.unsqueeze(0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x, padded_bins):
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        if self.ffn_hidden_size % self.blocking != 0:
            raise ValueError(
                f'The ffn_hidden_size {self.ffn_hidden_size} must be divisible by ' +
                f'the block size {self.blocking}. Please update your configuration.',
            )

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_hidden_size // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device,
        )

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(
            padded_bins,
            self.blocking,
            block_rows,
            blocks_per_row,
        )

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=common.dtype(self.args),
            device='meta',
        )
        shape = (
            padded_tokens,
            self.ffn_hidden_size * mpu.experts_per_rank(self.args),
        )
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(
            shape,
            row_indices,
            column_indices,
            offsets,
        )
        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

    def indices_and_padded_bins(self, top_experts):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        top_experts = top_experts.int()
        bin_ids, indices = ops.sort(top_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(top_experts, self.num_experts)
        # print(f"tokens_per_expert(histogram output) shape {tokens_per_expert.shape}")
        

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(
            tokens_per_expert,
            self.blocking,
        )
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def sparse_forward_once(self, x, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        # expert_weights = expert_weights.flatten()
        # top_experts = top_experts.flatten()
        # with torch.no_grad():
        #     indices, bin_ids, bins, padded_bins, tokens_per_expert = (self.indices_and_padded_bins(top_experts))

        # # Route the tokens for MoE computation.
        # x = x.view(-1, x.shape[-1])
        # x = ops.padded_gather(
        #     x,
        #     indices,
        #     bin_ids,
        #     bins,
        #     padded_bins,
        #     self.top_k,
        # )

        # # Create the sparse matrix topology.
        # with torch.no_grad():
        #     topo = self.topology(x, padded_bins)

        # Perform the expert computation.
        x = self.mlp(x, topo)

        # Un-route the data for the MoE output.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.top_k,
        )
        return x, tokens_per_expert

    # For use in the base-class parallel_forward_once.
    def sparse_permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        expert_capactiy,  # unused
        top_k,
    ):

        # Round the token counts up to the block size used in the matrix
        # multiplication. Calculate the starting position of each bin.
        padded_tokens_per_expert = ops.round_up(
            tokens_per_expert,
            self.blocking,
        )
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, top_k)

        # Create the sparse matrix topology.
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # Perform the expert computation.
        x = self.mlp(x, topo)

        # Un-route the data for the MoE output.
        return ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            top_k,
        )

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
        expert_capactiy,  # unused
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
        if self.args.mlp_impl == 'sparse':
            return self.sparse_forward_once(x, expert_weights, top_experts)
        else:
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
        if self.args.mlp_impl == 'sparse':
            return self.sparse_permute_and_compute(
                x,
                tokens_per_expert,
                indices,
                bin_ids,
                expert_weights,
                bins,
                expert_capactiy,
                top_k,
            )
        else:
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

# benchmark kernel sperately for megablocks
def run_megablocks_seperate(top_k, expert_num, bs, seq_len, hid_dim):
    # x: [sl, bs, hs]
    # expert_weights: [sl * bs, top-k]
    # top_experts: [sl * bs, top-k]
    x = torch.rand((seq_len, bs, hid_dim), dtype=torch.float16, device='cuda')
    logits = torch.rand((seq_len * bs, expert_num), dtype=torch.float16, device='cuda')
    scores = logits.softmax(dim=-1)
    expert_weights, top_experts = torch.topk(scores, top_k, dim=-1)
    expert_weights = expert_weights.flatten()
    top_experts = top_experts.flatten()
    
    args = Arguments(
        hidden_size=hid_dim,
        ffn_hidden_size=2048,  
        moe_num_experts=expert_num,
        moe_top_k=top_k,
        mlp_impl='sparse' 
    )
    model = ParallelDroplessMLP(args).cuda()
    
    # Warm-up 
    for _ in range(10):
        _ = model.indices_and_padded_bins(top_experts)
    
    #################################################################
    # test indices_and_padded_bins
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        indices, bin_ids, bins, padded_bins, tokens_per_expert = model.indices_and_padded_bins(top_experts)
    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print("---------- benchmarking the routing kernel ----------")
    # print(f"indices shape {indices.shape}")
    # print(f"bin_ids shape {bin_ids.shape}")
    # print(f"bins shape {bins.shape}")
    # print(f"padded_bins shape {padded_bins.shape}")
    # print(f"tokens_per_expert shape {tokens_per_expert.shape}")
    
    # mem summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    
    print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    #################################################################
    
    #################################################################
    # test padded gather

    x = x.view(-1, x.shape[-1])

    for _ in range(10):
        # Route the tokens for MoE computation.
        tmp = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            model.top_k,
        )
        print(f'gather shape: {tmp.shape}, dtype: {tmp.dtype}')

    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    start_time = time.time()
    for _ in range(10):
        # Route the tokens for MoE computation.
        tmp = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            padded_bins,
            model.top_k,
        )
        
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    end_time = time.time()
    x = tmp
    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print("---------- benchmarking the dispatching kernel ----------")
    # mem summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    
    print(f"Execution Time fwd+bwd: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used fwd+bwd: {peak_memory_used / 1024 ** 2:.2f} MB")
    #################################################################
    
    #################################################################
    # test topo matirx

    with torch.no_grad():
        for _ in range(10):
            topo = model.topology(x, padded_bins)

    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    start_time = time.time()
    # Create the sparse matrix topology.
    with torch.no_grad():
        for _ in range(10):
            topo = model.topology(x, padded_bins)
    torch.cuda.synchronize()
    end_time = time.time()
    
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print("---------- benchmarking the topo matrix kernel ----------")
    # mem summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    
    print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    #################################################################
    
    
    #################################################################
    # # test mlp 
    # for _ in range(10):
    #     tmp = model.mlp(x, topo)

    # torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    # torch.cuda.reset_peak_memory_stats()
    # start_memory = torch.cuda.memory_allocated()
    
    # torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    # start_time = time.time()
    # # Perform the expert computation.
    # for _ in range(10):
    #     tmp = model.mlp(x, topo)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # x = tmp
    
    # torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    # end_memory = torch.cuda.memory_allocated()
    # peak_memory = torch.cuda.max_memory_allocated()
    
    # print("---------- benchmarking the mlp kernel ----------")
    # # mem summary
    # memory_used = end_memory - start_memory
    # peak_memory_used = peak_memory - start_memory
    
    # print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    # print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    #################################################################
    
    #################################################################
    # test unroute

    for _ in range(10):
        # Un-route the data for the MoE output.
        tmp = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            top_k,
        )
        
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    start_time = time.time()
    # Perform the expert computation.
    for _ in range(10):
        # Un-route the data for the MoE output.
        tmp = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            top_k,
        )
    torch.cuda.synchronize()
    end_time = time.time()
    x = tmp
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print("---------- benchmarking the unroute kernel ----------")
    # mem summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    
    print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    #################################################################
    

def routing_cuda(top_experts, sort_end_bit, expert_num):
    # cuda kernel
    top_experts = top_experts.int()
    bin_ids, indices = ops.sort(top_experts, sort_end_bit)

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    tokens_per_expert = ops.histogram(top_experts, expert_num)

    # Calculate the bin bounds for the sorted tokens.
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)
    bins = promote_scalar(bins)
    return indices, bin_ids, bins, tokens_per_expert

@torch.compile
def routing_torch(top_experts, sort_end_bit, expert_num):
    # torch kernel
    top_experts = top_experts.int()
    bin_ids, indices = torch.sort(top_experts, sort_end_bit)

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    tokens_per_expert = torch.histogram(top_experts, expert_num)

    # Calculate the bin bounds for the sorted tokens.
    bins = torch.cumsum(tokens_per_expert, 0)
    bins = promote_scalar(bins)
    return indices, bin_ids, bins, tokens_per_expert

# benchmark kernel sperately for megablocks
def bench_routing_kernels(top_k, expert_num, bs, seq_len, hid_dim):
    # x: [sl, bs, hs]
    # expert_weights: [sl * bs, top-k]
    # top_experts: [sl * bs, top-k]
    x = torch.rand((seq_len, bs, hid_dim), dtype=torch.float16, device='cuda')
    logits = torch.rand((seq_len * bs, expert_num), dtype=torch.float16, device='cuda')
    scores = logits.softmax(dim=-1)
    expert_weights, top_experts = torch.topk(scores, top_k, dim=-1)
    expert_weights = expert_weights.flatten()
    top_experts = top_experts.flatten()
    sort_end_bit = max(int(np.ceil(np.log2(expert_num))), 1)
    
    args = Arguments(
        hidden_size=hid_dim,
        ffn_hidden_size=2048,  
        moe_num_experts=expert_num,
        moe_top_k=top_k,
        mlp_impl='sparse' 
    )
    model = ParallelDroplessMLP(args).cuda()
    
    # Warm-up 
    for _ in range(10):
        routing_cuda(top_experts, sort_end_bit, expert_num)
    
    for _ in range(10):
        routing_torch(top_experts, sort_end_bit, expert_num)
    
    # test routing kernel CUDA version
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        indices, bin_ids, bins, tokens_per_expert = routing_cuda(top_experts, sort_end_bit, expert_num)

    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print("---------- benchmarking the routing kernel: CUDA version ----------")
    # print(f"indices shape {indices.shape}")
    # print(f"bin_ids shape {bin_ids.shape}")
    # print(f"bins shape {bins.shape}")
    # print(f"padded_bins shape {padded_bins.shape}")
    # print(f"tokens_per_expert shape {tokens_per_expert.shape}")
    
    # mem summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    
    print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")

    # test routing kernel torch compile version
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        indices, bin_ids, bins, tokens_per_expert = routing_torch(top_experts, sort_end_bit, expert_num)

    torch.cuda.synchronize()
    end_time = time.time()
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    print("---------- benchmarking the routing kernel: pytorch version ----------")
    # print(f"indices shape {indices.shape}")
    # print(f"bin_ids shape {bin_ids.shape}")
    # print(f"bins shape {bins.shape}")
    # print(f"padded_bins shape {padded_bins.shape}")
    # print(f"tokens_per_expert shape {tokens_per_expert.shape}")
    
    # mem summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    
    print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    # print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    #################################################################
    



# bechmark megablocks gating's fwd and bwd
def run_megablocks_all(logits):
    pass
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark dMoE's indices_and_padded_bins.")
    parser.add_argument("--top_k", type=int, required=True, help="Top-k experts")
    parser.add_argument("--e", type=int, required=True, help="Number of experts")
    parser.add_argument("--bs", type=int, required=True, help="Batch size")
    parser.add_argument("--s", type=int, required=True, help="Sequence length")
    parser.add_argument("--hid_dim", type=int, required=True, help="Hidden dimension")
    args = parser.parse_args()

    print(f"Arguments: {args}")
    # run_megablocks_seperate(args.top_k, args.e, args.bs, args.s, args.hid_dim)
    bench_routing_kernels(args.top_k, args.e, args.bs, args.s, args.hid_dim)
    print("\n \n")



    
