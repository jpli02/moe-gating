# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union, Callable
from functools import partial
import pdb

import stk
import stk.backend.triton_kernels
import stk.ops
import torch
from packaging import version
import pdb
from functools import reduce

from megablocks import grouped_gemm_util as gg
from megablocks.ops.unpadded_grouped_gemm import GroupedGemm
from megablocks.layers import common, gelu, mpu
from megablocks.layers.activation_fn import act_fn
from megablocks.layers.arguments import DEFAULT_ACTIVATION_FN, Arguments, InitFn


class ScaleGradient(torch.autograd.Function):

    @staticmethod
    @torch.amp.autocast_mode.custom_fwd(device_type='cuda')
    def forward(ctx: Any, x: torch.Tensor, scale: float):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.amp.autocast_mode.custom_bwd(device_type='cuda')
    def backward(ctx: torch.Tensor, grad: torch.Tensor):
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply


def resolve_dtensor(weight: torch.Tensor):
    if version.parse(torch.__version__) >= version.parse('2.0.0'):
        from torch.distributed._tensor import DTensor
        if isinstance(weight, DTensor):
            return weight.to_local()
    return weight


def create_moe_expert_weights(
    args: Arguments,
    num_experts: int,
    ffn_hidden_size: int,
    hidden_size: int,
    init_method: InitFn,
):
    # Create the entire weight matrix such that the sampled weights will
    # not vary between data parallelism and expert model parallelism for
    # the same random seed.
    master_weights = torch.empty(
        num_experts,
        ffn_hidden_size,
        hidden_size,
        device=args.device,
        dtype=common.dtype(args),
    )
    init_method(master_weights)

    if not args.moe_expert_model_parallelism:
        return master_weights

    # Calculate the amount of sharding in each dimension.
    expert_sharding_degree = mpu.expert_sharding_degree(args)
    hidden_sharding_degree = mpu.hidden_sharding_degree(args)

    # Calculate the experts per rank.
    #
    # NOTE: We assign ranks to be expert parallel before going
    # tensor parallel.
    rank = mpu.get_expert_parallel_rank(args)
    expert_rank = rank % expert_sharding_degree
    num_experts_per_rank = num_experts // expert_sharding_degree
    start_expert = expert_rank * num_experts_per_rank
    end_expert = (expert_rank + 1) * num_experts_per_rank

    # Calculate the rows per rank.
    row_rank = rank // expert_sharding_degree
    num_rows_per_rank = ffn_hidden_size // hidden_sharding_degree
    start_row = row_rank * num_rows_per_rank
    end_row = (row_rank + 1) * num_rows_per_rank

    # Slice the weight matrix to get the chunk for this rank.
    with torch.no_grad():
        weights = master_weights[start_expert:end_expert, start_row:end_row]
    return weights


class MLP(torch.nn.Module):

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        # expert_parallel_world_size = mpu.get_expert_parallel_world_size(args)
        experts_per_rank = mpu.experts_per_rank(args)

        self.w1 = torch.nn.Parameter(
            torch.empty(
                experts_per_rank,
                args.hidden_size,
                mpu.features_per_rank(args),
                device=args.device,
                dtype=common.dtype(args),
            ),
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                experts_per_rank,
                mpu.features_per_rank(args),
                args.hidden_size,
                device=args.device,
                dtype=common.dtype(args),
            ),
        )
        mpu.set_expert_model_parallel_attributes(
            self.w1,
            args.moe_expert_model_parallelism,
        )
        mpu.set_expert_model_parallel_attributes(
            self.w2,
            args.moe_expert_model_parallelism,
        )

        # Initialize the parameters for the MLP.
        #
        # NOTE: It is important that we create the weight tensors prior
        # to creating the master weights and slicing our the piece for
        # this rank. If the master weights are created first the PyTorch
        # caching allocator appears to use the same memory block for these
        # and the slice which causes large increases in our peak memory
        # usage.
        with torch.no_grad():
            w1 = create_moe_expert_weights(
                args,
                args.moe_num_experts,
                args.ffn_hidden_size,
                args.hidden_size,
                args.init_method,
            )
            self.w1.copy_(w1.transpose(1, 2).contiguous())
            self.w2.copy_(
                create_moe_expert_weights(
                    args,
                    args.moe_num_experts,
                    args.ffn_hidden_size,
                    args.hidden_size,
                    args.output_layer_init_method,
                ),
            )

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args,)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x):
        w1, w2 = self.scale_grad(self.w1), self.scale_grad(self.w2)
        w1, w2 = resolve_dtensor(w1), resolve_dtensor(w2)
        x = torch.bmm(x, w1)
        x = self.args.activation_fn(x)
        return torch.bmm(x, w2)


def create_dmoe_expert_weights(
    args: Arguments,
    num_experts: int,
    rows: int,
    columns: int,
    init_method: InitFn,
):
    weights = create_moe_expert_weights(
        args,
        num_experts,
        rows,
        columns,
        init_method,
    )
    return weights.view([-1, columns])


class MemoryOptimizedMLP(torch.autograd.Function):
    """Sparse MLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.amp.autocast_mode.custom_fwd(device_type='cuda')
    def forward(ctx, x, w1, w2, topo, activation_fn):
        # Cast inputs using ctx dtype from AMP
        if ctx._fwd_used_autocast:
            x = x.to(ctx._dtype)
            w1 = w1.to(ctx._dtype)
            w2 = w2.to(ctx._dtype)
        # x: [m, k], w1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        topo_tensors = (
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )

        # Layer 0: x @ w1.t().
        sdd_out = stk.ops.sdd(x, w1.t(), topo)

        # GeLU.
        activation_fn_out = act_fn(sdd_out, activation_fn)

        # Layer 1: x @ w2.
        dsd_out = stk.ops.dsd(activation_fn_out, w2)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.shape = topo.shape
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.data.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w2, *topo_tensors, x, sdd_out.data)
        return dsd_out

    @staticmethod
    @torch.amp.autocast_mode.custom_bwd(device_type='cuda')
    def backward(ctx, ddsd_out):
        if (not ctx.needs_input_grad[0] or not ctx.needs_input_grad[1] or not ctx.needs_input_grad[2]):
            raise ValueError('Expected all MLP inputs to need grad.')

        # unpack saved tensors
        # dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w2 = saved_tensors[:2]
        topo_tensors = saved_tensors[2:8]
        x = saved_tensors[8]
        sdd_out_data = saved_tensors[9]

        # rematerialize activation function output
        activation_fn = ctx.activation_fn
        sdd_out = stk.Matrix(ctx.shape, sdd_out_data, *topo_tensors)
        activation_fn_out, activation_grad_fn = act_fn(
            sdd_out,
            activation_fn,
            return_grad_fn=True,
        )

        # Compute dw2 with recomputed activation_fn output.
        dw2 = stk.ops.dsd(activation_fn_out.t(), ddsd_out)

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        stk.backend.triton_kernels.sdd(
            ddsd_out,
            w2.t(),
            dactivation_fn_out.shape,
            dactivation_fn_out.data,
            dactivation_fn_out.offsets,
            dactivation_fn_out.row_indices,
            dactivation_fn_out.column_indices,
        )

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        if activation_fn is DEFAULT_ACTIVATION_FN:
            dsdd_out = gelu.gelu_backward_(dactivation_fn_out, sdd_out)
        else:
            assert activation_grad_fn is not None
            activation_grad_fn(dactivation_fn_out.data)
            dsdd_out = stk.Matrix(ctx.shape, sdd_out.data.grad, *topo_tensors)

        # Compute dw1.
        dw1 = stk.ops.dsd(dsdd_out.t(), x)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        stk.backend.triton_kernels.dsd(
            dsdd_out.shape,
            dsdd_out.data,
            dsdd_out.offsets,
            dsdd_out.row_indices,
            dsdd_out.column_indices,
            dsdd_out.offsets_t,
            dsdd_out.column_indices_t,
            dsdd_out.block_offsets_t,
            False,
            w1,
            ddsd_out,
        )
        dx = ddsd_out
        return dx, dw1, dw2, None, None


memory_optimized_mlp = MemoryOptimizedMLP.apply

## Custom function to preapre the weights of a function. ##
def prepare_weights(init_method: Union[Callable[[torch.Tensor], None], partial[torch.Tensor]], 
                    args: Arguments, trp: bool = False):
    weights = torch.nn.Parameter(init_method(
    torch.empty(
        (args.moe_num_packed_experts, args.hidden_size, args.ffn_hidden_size),
        device=args.device,
        dtype=common.dtype(args))))

    if trp:
        weights = torch.transpose(weights, 1, 2)

    return list(torch.unbind(weights))


## Potentially change to a torch.nn.Module? ##
class UnPaddedMLP(torch.nn.Module):

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self._num_rows_per_rank = mpu.experts_per_rank(args) * mpu.features_per_rank(args)

        ## We have to restructure the experts. ##
        ## Store them as a list at init time to avoid 
        ## extra per iteration overhead. ##
        # self.w1 = [torch.nn.Parameter(
        #     self.args.init_method(torch.empty((
        #         args.hidden_size,
        #         args.ffn_hidden_size
        #     ),
        #         device=args.device,
        #         dtype=common.dtype(args),
        #     )),
        # ) for _ in range(args.moe_num_packed_experts)]

        self.w1 = torch.nn.Parameter(
            self.args.init_method(torch.empty((
                args.moe_num_packed_experts,
                args.hidden_size,
                args.ffn_hidden_size
            ),
            device=args.device,
            dtype=common.dtype(args),
            ))
        )

        #self.w1 = torch.stack(self.w1)
        #self.w1.retain_grad()

        #self.w2 = [torch.nn.Parameter(
        #    self.args.output_layer_init_method(torch.empty((
        #        args.ffn_hidden_size,
        #        args.hidden_size
        #    ),
        #        device=args.device,
        #        dtype=common.dtype(args),
        #    )),
        #) for _ in range(args.moe_num_packed_experts)]

        self.w2 = torch.nn.Parameter(
            self.args.output_layer_init_method(torch.empty((
                args.moe_num_packed_experts,
                args.ffn_hidden_size,
                args.hidden_size
            ),
            device=args.device,
            dtype=common.dtype(args),
            ))
        )

        #self.w2 = torch.stack(self.w2)
        #self.w2.retain_grad()

        ## Slightly buggy for now. TODO(ahangupta): debug. ##
        # self.w1 = prepare_weights(self.args.init_method, args, True)
        # self.w2 = prepare_weights(self.args.output_layer_init_method, args, False)

        # print(f'w1 no-padded mlp: {self.w1}')
        # print(f'w2 no-padded mlp: {self.w2}')

        # print(f'w1 no-padded mlp sum: {reduce(lambda x, y: x+y.sum(), self.w1, 0)}')
        # print(f'w2 no-padded mlp sum: {reduce(lambda x, y: x+y.sum(), self.w2, 0)}')

        #assert len(self.w1) == len(self.w2) and len(self.w1) == 4, 'Needs 4-way expert packing.'

        ## Create individual pointers, required for backprop only. ##
        # self.w1_a, self.w1_b, self.w1_c, self.w1_d = (*self.w1,)
        # self.w2_a, self.w2_b, self.w2_c, self.w2_d = (*self.w2,)

        self.activation_fn = args.activation_fn


    def forward(self, x: torch.Tensor, 
                sizes: list[tuple[int, int, int]]):
        ## First, we call the forward pass simply. ##
        activ = GroupedGemm(x, self.w1, sizes)

        inter_sizes = [(i[0], self.args.hidden_size, self.args.ffn_hidden_size) for i in sizes]
        second_activs = GroupedGemm(self.activation_fn(activ), self.w2, inter_sizes)
        return second_activs


class SparseMLP(torch.nn.Module):

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self._num_rows_per_rank = mpu.experts_per_rank(args) * mpu.features_per_rank(args)

        t = torch.nn.Parameter(
            self.args.init_method(torch.empty((
                args.moe_num_packed_experts,
                args.hidden_size,
                args.ffn_hidden_size
            ),
            device=args.device,
            dtype=common.dtype(args),
            ))
        )

        self.w1 = torch.transpose(t, 0, 1).reshape([args.hidden_size, -1])

        self.w1.retain_grad()

        self.w2 = torch.nn.Parameter(
            self.args.output_layer_init_method(torch.empty((
                args.moe_num_packed_experts,
                args.ffn_hidden_size,
                args.hidden_size
            ),
            device=args.device,
            dtype=common.dtype(args),
            ))
        ).view([-1, args.hidden_size])

        self.w2.retain_grad()

        # self.w1 = torch.nn.Parameter(
        #     torch.empty(
        #         self._num_rows_per_rank,
        #         args.hidden_size,
        #         device=args.device,
        #         dtype=common.dtype(args),
        #     ),
        # )
        # self.w2 = torch.nn.Parameter(
        #     torch.empty(
        #         self._num_rows_per_rank,
        #         args.hidden_size,
        #         device=args.device,
        #         dtype=common.dtype(args),
        #     ),
        # )

        # # Initialize the parameters for the MLP.
        # #
        # # NOTE: It is important that we create the weight tensors prior
        # # to creating the master weights and slicing our the piece for
        # # this rank. If the master weights are created first the PyTorch
        # # caching allocator appears to use the same memory block for these
        # # and the slice which causes large increases in our peak memory
        # # usage.
        # with torch.no_grad():
        #     self.w1.copy_(
        #         create_dmoe_expert_weights(
        #             args,
        #             args.moe_num_experts,
        #             args.ffn_hidden_size,
        #             args.hidden_size,
        #             args.init_method,
        #         ),
        #     )
        #     self.w2.copy_(
        #         create_dmoe_expert_weights(
        #             args,
        #             args.moe_num_experts,
        #             args.ffn_hidden_size,
        #             args.hidden_size,
        #             args.output_layer_init_method,
        #         ),
        #     )

        # print(f'w1 weights sparse-mlp: {self.w1}')

        # print(f'w2 weights sparse-mlp: {self.w2}')

        # print(f'w1 sparse-mlp sum: {self.w1.sum()}')

        # print(f'w2 sparse-mlp sum: {self.w2.sum()}')

        # self._should_set_parallelism_attribute = args.moe_expert_model_parallelism
        # mpu.set_expert_model_parallel_attributes(
        #     self.w1,
        #     self._should_set_parallelism_attribute,
        # )
        # mpu.set_expert_model_parallel_attributes(
        #     self.w2,
        #     self._should_set_parallelism_attribute,
        # )

        self.gradient_scale = None
        # if self.args.moe_expert_model_parallelism:
        #     self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args,)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x, topo):
        w1, w2 = self.scale_grad(self.w1), self.scale_grad(self.w2)
        w1, w2 = resolve_dtensor(w1), resolve_dtensor(w2)
        if self.args.memory_optimized_mlp:
            return memory_optimized_mlp(
                x,
                w1,
                w2,
                topo,
                self.args.activation_fn,
            )

        # Compute the MLP.
        x = stk.ops.sdd(x, w1, topo)
        activation_fn_out = act_fn(x, self.args.activation_fn)
        return stk.ops.dsd(activation_fn_out, w2)


class MemoryOptimizedGroupedMLP(torch.autograd.Function):
    """GroupedMLP with manually scheduled memory reuse."""

    @staticmethod
    @torch.amp.autocast_mode.custom_fwd(device_type='cuda')
    def forward(ctx, x, w1, w2, batch_sizes, activation_fn):
        # Cast inputs using ctx dtype from AMP
        if ctx._fwd_used_autocast:
            x = x.to(ctx._dtype)
            w1 = w1.to(ctx._dtype)
            w2 = w2.to(ctx._dtype)
        # x: [m, k], w1: [n, k], w2: [n, k]
        if (not x.is_contiguous() or not w1.is_contiguous() or not w2.is_contiguous()):
            raise ValueError("Expected contiguous 'x', 'w1' and 'w2'.")

        # Layer 0: x @ w1.t().
        assert gg.backend is not None
        sdd_out = gg.backend.gmm(x, w1, batch_sizes, trans_b=True)

        # activation_fn
        activation_fn_out = activation_fn(sdd_out)

        # Layer 1: x @ w2.
        dsd_out = gg.backend.gmm(activation_fn_out, w2, batch_sizes)

        # NOTE: Save the input to the layer and the activation_fn input for
        # gradient computation. We'll re-compute the activation_fn forward
        # pass in the backward pass to avoid materializing another
        # intermediate.
        ctx.x_shape = x.shape
        ctx.sdd_out_shape = sdd_out.shape
        ctx.dtype = x.dtype
        ctx.activation_fn = activation_fn
        ctx.save_for_backward(w1, w2, batch_sizes, x, sdd_out)
        return dsd_out

    @staticmethod
    @torch.amp.autocast_mode.custom_bwd(device_type='cuda')
    def backward(ctx: Any, ddsd_out: torch.Tensor):
        if (not ctx.needs_input_grad[0] or not ctx.needs_input_grad[1] or not ctx.needs_input_grad[2]):
            raise ValueError('Expected all MLP inputs to need grad.')

        # Unpack saved tensors
        # dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        w1, w2 = saved_tensors[:2]
        batch_sizes = saved_tensors[2]
        x = saved_tensors[3]
        sdd_out = saved_tensors[4]

        # Rematerialize activation_fn output.
        activation_fn = ctx.activation_fn
        with torch.set_grad_enabled(True):
            sdd_out.requires_grad = True
            activation_fn_out = activation_fn(sdd_out)
            activation_grad_fn = activation_fn_out.backward

        # Compute dw2 with recomputed activation_fn output.
        assert gg.backend is not None
        dw2 = gg.backend.gmm(
            activation_fn_out,
            ddsd_out,
            batch_sizes,
            trans_a=True,
        )

        # Compute dactivation_fn_out.
        #
        # NOTE: We reuse the activation_fn_out allocation.
        dactivation_fn_out = activation_fn_out
        gg.backend.gmm(
            ddsd_out,
            w2,
            batch_sizes,
            trans_b=True,
            c=dactivation_fn_out,
        )

        # Compute dsdd_out.
        #
        # NOTE: This reuses the dactivation_fn_out allocation.
        if activation_fn is DEFAULT_ACTIVATION_FN:
            dsdd_out = gelu.gelu_backward_(dactivation_fn_out, sdd_out)
        else:
            assert activation_grad_fn is not None
            activation_grad_fn(dactivation_fn_out)
            dsdd_out = sdd_out.grad

        # Compute dw1.
        dw1 = gg.backend.gmm(dsdd_out, x, batch_sizes, trans_a=True)

        # Compute dx.
        #
        # NOTE: This reuses the ddsd_out allocation.
        gg.backend.gmm(dsdd_out, w1, batch_sizes, c=ddsd_out)
        dx = ddsd_out
        return dx, dw1, dw2, None, None


memory_optimized_grouped_mlp = MemoryOptimizedGroupedMLP.apply


class GroupedMLP(SparseMLP):

    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = resolve_dtensor(w1).view(ne, -1, self.args.hidden_size)
        w2 = resolve_dtensor(w2).view(ne, -1, self.args.hidden_size)

        if self.args.memory_optimized_mlp:
            return memory_optimized_grouped_mlp(
                x,
                w1,
                w2,
                batch_sizes,
                self.args.activation_fn,
            )

        # Compute the MLP.
        assert gg.ops is not None
        x = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = self.args.activation_fn(x)
        return gg.ops.gmm(x, w2, batch_sizes)


class SharedMLP(torch.nn.Module):
    """MLP for shared expert.

    Note: this is a copy -> pasta -> modify of the LLM-Foundry MPTMLP class
    """

    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args
        self.fc_kwargs: dict[str, Any] = {
            'bias': args.bias,
            'device': args.device,
        }
        self.fc_kwargs.update(args.fc_kwargs)

        self.up_proj = args.fc_cls(
            args.hidden_size,
            args.shared_expert_hidden_size,
            **self.fc_kwargs,
        )
        self.act = args.activation_fn
        self.down_proj = args.fc_cls(
            args.shared_expert_hidden_size,
            args.hidden_size,
            **self.fc_kwargs,
        )
        self.down_proj._is_residual = True  # a flag for llm-foundry init

    def add_experts_sharedexpert(
        self,
        shared_expert_out: torch.Tensor,
        expert_out: torch.Tensor,
    ) -> torch.Tensor:
        # Helper function to add expert output to shared expert output
        # with optional weighted sum.
        if self.args.shared_expert_weighted_sum:
            # enable using weighted sum for shared expert output
            # wieghted by number of experts used
            t_experts = self.args.moe_top_k + 1
            sh_mlp_out = shared_expert_out / t_experts
            return sh_mlp_out.add(
                expert_out,
                alpha=(self.args.moe_top_k / t_experts),
            )

        return shared_expert_out + expert_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))
