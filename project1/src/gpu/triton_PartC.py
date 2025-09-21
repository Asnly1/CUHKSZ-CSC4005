import torch
import torch.nn.functional as F
import triton.testing as tt
import triton
import triton.language as tl
import numpy as np
import cv2
import math

#并不是pixel，而是某一个pixel的r/g/b
@triton.jit
def bilateral_filter_kernel(
    img_pad_ptr,  # *Pointer* to first input vector.
    pad_size,
    output_ptr,
    stride_h_pad,
    stride_w_pad,
    stride_h_out,
    stride_w_out,
    sigma_R,
    sigma_D,
    ACTIVATION: tl.constexpr,
):

    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    
    offset = (
        (pid_h + pad_size) * stride_h_pad + (pid_w + pad_size) * stride_w_pad + pid_c
    )

    pixel_value_11 = tl.load(img_pad_ptr + offset - 1 * stride_h_pad - 1 * stride_w_pad)
    pixel_value_12  = tl.load(img_pad_ptr + offset - 1 * stride_h_pad + 0 * stride_w_pad)
    pixel_value_13  = tl.load(img_pad_ptr + offset - 1 * stride_h_pad + 1 * stride_w_pad)
    pixel_value_21  = tl.load(img_pad_ptr + offset + 0 * stride_h_pad - 1 * stride_w_pad)
    pixel_value_22 = tl.load(img_pad_ptr + offset) # 中心像素值
    pixel_value_23   = tl.load(img_pad_ptr + offset + 0 * stride_h_pad + 1 * stride_w_pad)
    pixel_value_31  = tl.load(img_pad_ptr + offset + 1 * stride_h_pad - 1 * stride_w_pad)
    pixel_value_32   = tl.load(img_pad_ptr + offset + 1 * stride_h_pad + 0 * stride_w_pad)
    pixel_value_33   = tl.load(img_pad_ptr + offset + 1 * stride_h_pad + 1 * stride_w_pad)
            
    w_border = tl.exp(-0.5 / (sigma_D * sigma_D))
    w_corner = tl.exp(-1.0 / (sigma_D * sigma_D))
    sigma_r_sq_inv = -0.5 / (sigma_R * sigma_R)
    
    center_value = pixel_value_22
    sum_weight = 0.0
    filtered_value = 0.0


    diff = center_value - pixel_value_11
    w = w_corner * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_11

    diff = center_value - pixel_value_12
    w = w_border * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_12

    diff = center_value - pixel_value_13
    w = w_corner * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_13

    diff = center_value - pixel_value_21
    w = w_border * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_21

    diff = 0.0
    w = 1.0
    sum_weight += w
    filtered_value += w * pixel_value_22

    diff = center_value - pixel_value_23
    w = w_border * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_23

    diff = center_value - pixel_value_31
    w = w_corner * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_31

    diff = center_value - pixel_value_32
    w = w_border * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_32

    diff = center_value - pixel_value_33
    w = w_corner * tl.exp(diff * diff * sigma_r_sq_inv)
    sum_weight += w
    filtered_value += w * pixel_value_33
   
    
    filtered_value = filtered_value / sum_weight
    
    output_offset = pid_h * stride_h_out + pid_w * stride_w_out + pid_c
    tl.store(output_ptr + output_offset, filtered_value)
    
def bilateral_filter_kernel_helper(img_pad, k_size, activation=""):
    assert img_pad.is_contiguous(), "Matrix A must be contiguous"
    H, W, C = img_pad.shape
    sigma_D = 1.7
    sigma_R = 50.0
    pad = (k_size - 1) // 2
    H_orig, W_orig = H - 2 * pad, W - 2 * pad  # ignore bounary pixels

    output = torch.empty(
        (H_orig, W_orig, C), device=img_pad.device, dtype=torch.float32
    )
    grid = lambda META: (
        triton.cdiv(H_orig, 1), # ceil(H_orig/1)
        triton.cdiv(W_orig, 1),
        triton.cdiv(C, 1),
    )

    elapsed_time = tt.do_bench(
        lambda: bilateral_filter_kernel[grid](
            img_pad,
            pad,
            output,
            img_pad.stride(0), # 沿着height方向的步长
            img_pad.stride(1), # 沿着width方向的步长
            output.stride(0),
            output.stride(1),
            sigma_D,
            sigma_R,
            ACTIVATION=activation,
        ),
        warmup=25,
        rep=100,
    )
    print(f"Execution Time: {elapsed_time:.2f} ms")
    return output

def main(input_image_path, output_image_path):
    print(f"Input file to: {input_image_path}")

    ksize = 3  # ksize=7

    # read image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32)

    # add padding to the image
    pad = (ksize - 1) // 2

    pad_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    pad_img = torch.tensor(pad_img, device="cuda", dtype=torch.float32)

    # apply the filter
    output_triton = bilateral_filter_kernel_helper(pad_img, ksize)
    output_img = output_triton.cpu().numpy()

    # save the output
    print(f"Output file to: {output_image_path}")
    # Transform pixel data from float32 to uint8 as image output
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_image_path, output_img)

    del output_triton
    del pad_img
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg"
        )
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
