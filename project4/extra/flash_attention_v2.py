import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': 32, 'BLOCK_N':32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N':64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':128}, num_stages=2, num_warps=4),
    ],
    key=['seq_len', 'd_model'],
)
@triton.jit
def flash_attention_v2(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm, stride_om,
    BLOCK_M: tl.constexpr, # total line of Q that current program controls, Bc in paper
    BLOCK_N: tl.constexpr, # loop columns in K and V, Br in paper
):  
    row_id = tl.program_id(0)
    
    # offset_r: the idx of lines of Q that current program controls
    # offset_d: total columns
    offset_r = row_id * BLOCK_M+ tl.arange(0, BLOCK_M) # (BLOCK_M, )
    offset_d = tl.arange(0, d_model) # (d_model, )
    
    # It should be (BLOCK_M, d_model)
    q_inputs = q_ptr + offset_r[:, None] * stride_qm + offset_d[None, :]
    
    mask_i = offset_r < seq_len # (BLOCK_M, )
    # Cannot be padding with float('-inf')
    # Reason: -inf * -inf = inf or 0 * -inf = Nan will happen when multiply with T
    q_i = tl.load(q_inputs, mask=mask_i[:, None], other=0.0)
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    output = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
    
    scale = 1.0 / (d_model ** 0.5)
    
    for col_idx in tl.range(0, seq_len, BLOCK_N):
        # offset_c: the idx of lines of K and V that current loop controls
        offset_c = col_idx + tl.arange(0, BLOCK_N) # (BLOCK_N, )
        
        # It should be (BLOCK_N, d_model)
        k_inputs = k_ptr + offset_c[:, None] * stride_km + offset_d[None, :]
        v_inputs = v_ptr + offset_c[:, None] * stride_vm + offset_d[None, :]
        mask_j = offset_c < seq_len # (BLOCK_N, )
        k_j = tl.load(k_inputs, mask=mask_j[:, None], other=0.0)
        v_j = tl.load(v_inputs, mask=mask_j[:, None], other=0.0)
        
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale # (BLOCK_M, BLOCK_N)
        s_ij = tl.where(offset_c[None, :] < seq_len, s_ij, float('-inf'))
        
        m_ij = tl.max(s_ij, axis=1) # (BLOCK_M, )
        p_ij = tl.exp(s_ij-m_ij[:, None]) # (BLOCK_M, BLOCK_N)
        l_ij = tl.sum(p_ij, axis=1) # (BLOCK_M, )
        
        m_i_new = tl.maximum(m_i, m_ij) # (BLOCK_M, )
        alpha = tl.exp(m_i-m_i_new) # (BLOCK_M, )
        beta = tl.exp(m_ij-m_i_new) # (BLOCK_M, )
        l_i_new = alpha * l_i + beta * l_ij # (BLOCK_M, )
        
        output = alpha[:, None] * output + beta[:, None] * tl.dot(p_ij, v_j) 
        # (BLOCK_M, 1) * (BLOCK_M, d_model) + (BLOCK_M, 1) * (BLOCK_M, BLOCK_N) * (BLOCK_N, d_model) = (BLOCK_M, d_model)
        
        l_i = l_i_new
        m_i = m_i_new
        
    output = output / l_i[:, None]
    o_outputs = o_ptr + offset_r[:, None] * stride_om + offset_d[None, :]
    tl.store(o_outputs, output, mask=mask_i[:, None])
    
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def call_flash_attention_v2(q, k, v):
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    assert q.dim() == 2, "Only support 2D input: (seq_len, d_model)"
    seq_len, d_model = q.shape
    o = torch.empty_like(q)

    # Paper's Implementation
    # BLOCK_M = triton.cdiv(SIZE_SMEM, (4*d_model))
    # BLOCK_N = triton.cdiv(BLOCK_M, d_model)
    grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_M']), 1, 1)
    
    flash_attention_v2[grid](q, k, v, o, seq_len, d_model,
                             q.stride(0), k.stride(0), v.stride(0), o.stride(0))
    return o

def pytorch_attention(q, k, v):
        d_model = q.shape[-1]
        scale_factor = 1.0 / (d_model**0.5)
        scores = q @ k.T * scale_factor
        softmax = torch.softmax(scores, dim=1)
        attention = softmax @ v
        return attention

def benchmark_internal(seq_len, d_model, provider):
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    if provider == 'flash_attention':
        fn = lambda: call_flash_attention_v2(q, k, v)
    elif provider == 'pytorch':
        fn = lambda: pytorch_attention(q, k, v)

    return triton.testing.do_bench(fn)

def benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_len'],
            x_vals=[2 ** i for i in range(2, 13)],
            line_arg='d_model',
            line_vals=[64, 128, 256],
            line_names=['d=64', 'd=128', 'd=256'],
            styles=[('blue', '-'), ('green', '--'), ('red', '-.')],
            ylabel="Latency (ms)",
            plot_name="flash-attention-performance",
            args={'provider': 'flash_attention'},
        )
    )
    def bench_flash(seq_len, d_model, provider):
        return benchmark_internal(seq_len, d_model, provider)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_len'],
            x_vals=[2 ** i for i in range(2, 13)],
            line_arg='d_model',
            line_vals=[64, 128, 256],
            line_names=['d=64', 'd=128', 'd=256'],
            styles=[('orange', '-'), ('purple', '--'), ('brown', '-.')],
            ylabel="Latency (ms)",
            plot_name="pytorch-attention-performance",
            args={'provider': 'pytorch'},
        )
    )
    def bench_pytorch(seq_len, d_model, provider):
        return benchmark_internal(seq_len, d_model, provider)

    bench_flash.run(show_plots=False, print_data=True)
    bench_pytorch.run(show_plots=False, print_data=True)
    
    
def unit_test(seq_len, d_model):
    torch.manual_seed(0)
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    o_triton = call_flash_attention_v2(q, k, v)
    o_torch = pytorch_attention(q, k, v)

    assert torch.allclose(o_triton, o_torch, atol=1e-3, rtol=1e-3), (o_triton, o_torch)
    print(f"Attention output correct for seq_len={seq_len}, d_model={d_model}!")

if __name__ == "__main__":
    for i in range(8, 12):
        for d_model in [32, 64, 128]:
            unit_test(2 ** i, d_model)
    print("pass all unit test")
    print("-----------------------------------------")   
    benchmark()