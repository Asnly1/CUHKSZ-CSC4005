import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def native_softmax(x, mask=None, scale=1.0, dropout_p=0.0):

    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    x = x * scale
    if mask is not None:
        x = torch.where(mask, x, torch.tensor(float('-inf'), device=x.device))
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    if dropout_p > 0.0:
        ret = torch.nn.functional.dropout(ret, p=dropout_p, training=True)
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

@triton.autotune(
    configs = [
        triton.Config({'num_stages': 2, 'num_warps':4}, num_stages=2, num_warps=4),
        triton.Config({'num_stages': 2, 'num_warps':8}, num_stages=2, num_warps=8),
        triton.Config({'num_stages': 4, 'num_warps':4}, num_stages=4, num_warps=4),
        triton.Config({'num_stages': 4, 'num_warps':8}, num_stages=4, num_warps=8),
    ],
    key=['n_rows', 'n_cols'],
)
@triton.jit
def softmax_kernel(output_ptr, 
                   input_ptr,
                   mask_ptr,
                   input_row_stride, 
                   output_row_stride, 
                   n_rows, 
                   n_cols,
                   scale,            
                   dropout_p,        
                   seed,               
                   HAS_DROPOUT: tl.constexpr,       
                   HAS_MASK: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    if row_start >= n_rows:
        return
    row_step = tl.num_programs(0)
    
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        
        boundary_mask = col_offsets < n_cols
        if HAS_MASK:
            mask_start_ptr = mask_ptr + row_idx * input_row_stride
            mask_ptrs = mask_start_ptr + col_offsets
            loaded_mask = tl.load(mask_ptrs, mask=boundary_mask, other=0.0)
            all_mask = (loaded_mask != 0)
        else:
            all_mask = boundary_mask
        row = tl.load(input_ptrs, mask=all_mask, other=float('-inf'))
        
        row = row * scale
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        if HAS_DROPOUT:
            random_offset = row_idx * n_cols + col_offsets
            random_value = tl.rand(seed, random_offset)
            dropout_mask = random_value < (1-dropout_p)
            softmax_output = tl.where(dropout_mask, softmax_output * (1.0 / (1-dropout_p)), 0.0)
        
        output_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=boundary_mask)
        

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x, mask=None, scale=1.0, dropout_p=0.0):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    seed = torch.randint(0, 2**31, (1,), device=x.device).item()
    HAS_DROPOUT = True if dropout_p != 0.0 else False
    HAS_MASK = True if mask is not None else False
    
    y = torch.empty_like(x)
    grid = (n_rows, 1, 1)
    softmax_kernel[grid](y, x, mask, x.stride(0), y.stride(0), n_rows, n_cols, scale=scale, dropout_p=dropout_p, seed=seed,
                                 HAS_DROPOUT=HAS_DROPOUT, HAS_MASK=HAS_MASK, BLOCK_SIZE=BLOCK_SIZE)
    return y

def torch_softmax(x, mask=None, scale=1.0, dropout_p=0.0):
    x = x * scale
    if mask is not None:
        x = torch.where(mask, x, torch.tensor(float('-inf'), device=x.device))
    y = torch.softmax(x, dim=-1)
    if dropout_p > 0.0:
        y = torch.nn.functional.dropout(y, p=dropout_p, training=True)
        
    return y

def unit_test(n_rows, n_cols):
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, device=DEVICE)
    mask = torch.rand_like(x) < 0.7
    mask[torch.arange(n_rows), torch.randint(0, n_cols, (n_rows,))] = True
    
    scale = 1.0 / (n_cols ** 0.5) # 典型的 scaling
    
    # Test 1: Correctness without Dropout
    y_triton = softmax(x, mask, scale=scale, dropout_p=0.0)
    y_torch = torch_softmax(x, mask, scale=scale, dropout_p=0.0)
    
    assert torch.allclose(y_triton, y_torch, atol=1e-5), "Mismatch with dropout=0"
    print(f"{n_rows} * {n_cols} [Scaling]: Correct!")

    # Test 2: Run with Dropout
    dropout_p = 0.5
    y_triton_drop = softmax(x, mask, scale=scale, dropout_p=dropout_p)
    
    zeros_fraction = (y_triton_drop == 0).float().mean().item()
    print(f"{n_rows} * {n_cols} [Dropout]: Ran successfully. Zeros fraction: {zeros_fraction:.2f}")
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(6, 14)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    mask = torch.rand_like(x) < 0.7
    mask[torch.arange(M), torch.randint(0, N, (M,))] = True
    
    scale = 0.5
    dropout_p = 0.1
    
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_softmax(x, mask, scale, dropout_p))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x, mask, scale, dropout_p))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: native_softmax(x, mask, scale, dropout_p))
    
    return ms

if __name__ == '__main__':
    for i in range(8, 14):
        unit_test(2 ** i, 2 ** (i // 2))
    print("pass all correctness test")
    print("-----------------------------------------")    
    
    benchmark.run(show_plots=False, print_data=True)    