import torch
import triton
import triton.lanuage as tl

@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    bias = tl.full([M, N], 1, dtype=tl.int32)
    # check that histogram produces object compatible with broadcasting
    biased = z + bias
    tl.store(z_ptr + offset2, z)


def histogram_fn(X, N):
    # input:
    #   X: tensor [bs * seq_len * top_k]
    #   N: expert number [exp_num]
    # return:
    #   bins: [N]
    no_batch = X.ndimension() == 1;
    if no_batch: 
        X = X.view({1, X.numel()});    
    
    out = histogram_kernel[grid](X,  num_bins);
    if no_batch:
        return out.flatten()
    else:
        return out