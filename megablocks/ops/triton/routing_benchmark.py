# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import torch
from absl.testing import parameterized

# from megablocks import ops

_CUDA_TESTS = (
    (16384, torch.int32, 1),
    (16384, torch.int32, 2),
    (16384, torch.int32, 4),
    (16384, torch.int32, 128),
    (16384, torch.int32, 256),
)

_BASELINE_TORCH_TESTS = (
    (16384, torch.int32, 1),
    (16384, torch.int32, 2),
    (16384, torch.int32, 4),
    (16384, torch.int32, 128),
    (16384, torch.int32, 256),
)


def numpy_dtype(dtype):
    types = {
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
    }
    return types[dtype]


def benchmark_function(fn, iterations=10):
    # Run once to get rid of startup overhead.
    for _ in range(iterations):
        fn()
        
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times = np.array(times)
    return times.mean(), times.std(), times.max(), times.min()


def log_benchmark_torch(arguments, mean_t, std_t):
    print('=' * 60)
    print('Benchmark pytorch ops with Parameters:')
    for (key, value) in arguments.items():
        print(f'{key} = {value}')
    print('Results:')
    print('mean / std = {:.2f}ms / {:.2f}ms'.format(mean_t, std_t))
    print('=' * 60)
    
def log_benchmark_CUDA(arguments, mean_t, std_t):
    print('=' * 60)
    print('Benchmark CUDA ops with Parameters:')
    for (key, value) in arguments.items():
        print(f'{key} = {value}')
    print('Results:')
    print('mean / std = {:.2f}ms / {:.2f}ms'.format(mean_t, std_t))
    print('=' * 60)


# @torch.compile
def routing_torch(top_experts, end_bit, expert_num):
    # torch kernel
    top_experts = top_experts.int()
    bin_ids, indices = torch.sort(top_experts)

    # Histogram the expert ids to identify the number of
    # tokens routed to each expert.
    tokens_per_expert = torch.histc(top_experts, expert_num, 0, expert_num - 1)
    # tokens_per_expert = ops.histogram_triton(top_experts, expert_num)
    
    # Calculate the bin bounds for the sorted tokens.
    bins = torch.cumsum(tokens_per_expert, 0)
    return indices, bin_ids, bins, tokens_per_expert

# def routing_CDUA(top_experts, end_bit, expert_num):
#     # Sort the expert ids to produce the scatter/gather
#     # indices for the permutation.
#     top_experts = top_experts.int()
#     bin_ids, indices = ops.sort(top_experts, end_bit)

#     # Histogram the expert ids to identify the number of
#     # tokens routed to each expert.
#     tokens_per_expert = ops.histogram(top_experts, expert_num)

#     # Calculate the bin bounds for the sorted tokens.
#     bins = ops.inclusive_cumsum(tokens_per_expert, 0)
#     return indices, bin_ids, bins, tokens_per_expert

class RouteBenchmark(parameterized.TestCase):

    # @parameterized.parameters(*_CUDA_TESTS)
    # def testRouteCUDA(self, n, dtype, expert_num):
    #     if expert_num is None:
    #         expert_num = np.iinfo(numpy_dtype(dtype)).max
    #     end_bit = int(np.ceil(np.log2(expert_num)))
    #     x = torch.randint(0, expert_num, (n,)).cuda().to(dtype)

    #     mean_t, std_t, max_t, min_t = benchmark_function(lambda: routing_CDUA(x, end_bit, expert_num),)
    #     arguments = {
    #         'n': n,
    #         'dtype': dtype,
    #         'expert_num': expert_num
    #     }
    #     log_benchmark_CUDA(arguments, mean_t, std_t)

    @parameterized.parameters(*_BASELINE_TORCH_TESTS)
    def testRouteTorch(self, n, dtype, expert_num):
        if expert_num is None:
            expert_num = np.iinfo(numpy_dtype(dtype)).max
        end_bit = int(np.ceil(np.log2(expert_num)))
        x = torch.randint(0, expert_num, (n,)).cuda().to(dtype)
        
        # Correctness checks
        indices_cuda, bin_ids_cuda, bins_cuda, tokens_per_expert_cuda = routing_CDUA(x, end_bit, expert_num)
        indices_torch, bin_ids_torch, bins_torch, tokens_per_expert_torch = routing_torch(x, end_bit, expert_num)

        assert torch.equal(indices_cuda, indices_torch), "Mismatch in indices!"
        assert torch.equal(bin_ids_cuda, bin_ids_torch), "Mismatch in bin_ids!"
        assert torch.equal(bins_cuda, bins_torch), "Mismatch in bins!"
        assert torch.equal(tokens_per_expert_cuda, tokens_per_expert_torch), "Mismatch in tokens_per_expert!"


        mean_t, std_t, max_t, min_t = benchmark_function(lambda: routing_torch(x, end_bit, expert_num),)
        arguments = {
            'n': n,
            'dtype': dtype,
            'expert_num': expert_num
        }
        log_benchmark_torch(arguments, mean_t, std_t)



if __name__ == '__main__':
    unittest.main()
