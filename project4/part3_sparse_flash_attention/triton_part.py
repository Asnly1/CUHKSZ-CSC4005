import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def flash_attention_v1(
    q_ptr, k_ptr, v_ptr, o_ptr, sparse_mask_ptr,
    seq_len, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
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
    
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    
    for col_idx in tl.range(0, seq_len, BLOCK_N):
        block_n_id = col_idx // BLOCK_N
        
        offset_mask = row_id * num_blocks_n + block_n_id

        is_active = tl.load(sparse_mask_ptr + offset_mask)
        
        if is_active != 0:            
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
            
            # Do division in a small dimension: (BLOCK_M, ) instead of (BLOCK_M, d_model)
            output =((l_i[:, None] * alpha[:, None])  / l_i_new[:, None]) * output + beta[:, None]  / l_i_new[:, None] * tl.dot(p_ij, v_j)
            # (BLOCK_M, ) * (BLOCK_M, d_model) + (BLOCK_M, ) * (BLOCK_M, d_model) = (BLOCK_M, d_model)
            
            l_i = l_i_new
            m_i = m_i_new
        
    o_outputs = o_ptr + offset_r[:, None] * stride_om + offset_d[None, :]
    tl.store(o_outputs, output, mask=mask_i[:, None])
    
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def get_block_MN(d_model):
    return 32, 16

def call_flash_attention_v1_sparse(q, k, v, mask_ptr):
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    assert q.dim() == 2, "Only support 2D input: (seq_len, d_model)"
    seq_len, d_model = q.shape
    o = torch.empty_like(q)
    
    BLOCK_M, BLOCK_N = get_block_MN(d_model)
    grid = (triton.cdiv(seq_len, BLOCK_M), 1, 1)
    
    flash_attention_v1[grid](q, k, v, o, mask_ptr, seq_len, d_model,
                             q.stride(0), k.stride(0), v.stride(0), o.stride(0),
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        
    return o

def pytorch_attention(q, k, v):
        d_model = q.shape[-1]
        scale_factor = 1.0 / (d_model**0.5)
        scores = q @ k.T * scale_factor
        softmax = torch.softmax(scores, dim=1)
        attention = softmax @ v
        return attention
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[ 2 ** i for i in range(7, 13)],
        line_arg='d_model',
        line_vals=[64, 128, 256],
        line_names=['d=64', 'd=128', 'd=256'],
        styles=[('blue', '-'), ('green', '--'), ('red', '-.')],
        ylabel="Latency (ms)",
        plot_name="flash-attention-performance",
        args={'provider': 'flash_attention'},
    )
)
def benchmark(seq_len, d_model, provider):
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    BLOCK_M, BLOCK_N = get_block_MN(d_model)
    mask = generate_block_sparse_mask(seq_len, BLOCK_M, BLOCK_N)
    
    if provider == 'flash_attention':
        ms = triton.testing.do_bench(lambda: call_flash_attention_v1_sparse(q, k, v, mask))
    return ms

def unit_test(seq_len, d_model):
    torch.manual_seed(0)
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    BLOCK_M, BLOCK_N = get_block_MN(d_model)
    mask = generate_block_sparse_mask(seq_len, BLOCK_M, BLOCK_N)
    
    o_triton = call_flash_attention_v1_sparse(q, k, v, mask)
    o_native = native_block_sparse_attention(q, k, v, mask, BLOCK_M, BLOCK_N)

    assert torch.allclose(o_triton, o_native, atol=2e-3, rtol=2e-3), (o_triton, o_native)
    print(f"Attention output correct for seq_len={seq_len}, d_model={d_model}!")

def generate_block_sparse_mask(seq_len, BLOCK_M, BLOCK_N):
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    mask = torch.zeros((num_blocks_m, num_blocks_n), dtype=torch.int8, device='cuda')
    
    for i in range(num_blocks_m):
        if i < num_blocks_n:
            mask[i, i] = 1
        for k in range(int(torch.log2(torch.tensor(num_blocks_n))) + 1):
            j = i ^ (1 << k)  
            if j < num_blocks_n:
                mask[i, j] = 1
                
    return mask.view(-1) 

def native_block_sparse_attention(q, k, v, sparse_mask_flat, BLOCK_M, BLOCK_N):
    seq_len, d_model = q.shape
    scale = 1.0 / (d_model ** 0.5)

    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    assert sparse_mask_flat.numel() == num_blocks_m * num_blocks_n, f"sparse_mask_flat legnth wrong!"

    block_mask = sparse_mask_flat.view(num_blocks_m, num_blocks_n)
    
    m_idx = torch.arange(seq_len, device=q.device) // BLOCK_M  
    n_idx = torch.arange(seq_len, device=q.device) // BLOCK_N  
    m_idx_expand = m_idx.unsqueeze(1).expand(-1, seq_len)
    n_idx_expand = n_idx.unsqueeze(0).expand(seq_len, -1)
    expanded_mask = block_mask[m_idx_expand, n_idx_expand]
    
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale 
    scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
    mask_sum = expanded_mask.sum(dim=-1, keepdim=True)
    attn_weights = torch.where(
        mask_sum > 0,
        torch.softmax(scores, dim=-1),
        torch.zeros_like(scores)
    )
    
    output = torch.matmul(attn_weights, v)
    return output

if __name__ == "__main__":
    for i in range(6, 12):
        for d_model in [32, 64, 128]:
            unit_test(2 ** i, d_model)
    print("pass all unit test")
    print("-----------------------------------------")    
    
    benchmark.run(show_plots=False, print_data=True)    