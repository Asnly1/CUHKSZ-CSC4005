//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

// Gloabl Memory Coalesing
// Shared Memory
// unroll
// 增加每个thread的工作量

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils.hpp"

/**
 * Demo kernel device function to clamp pixel value
 * 
 * You may mimic this to implement your own kernel device functions
 */

__constant__ float d_w_border;
__constant__ float d_w_corner;
__constant__ float d_sigma_r_sq_inv;
__constant__ float d_w_spatial[9];

__device__ unsigned char d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

__device__ ColorValue d_bilateral_filter(ColorValue* __restrict__ values, float* __restrict__ d_expf_look_up,
                                         int local_row, int local_col, int width)
{
    int center_row = local_row + 1;
    int center_col = local_col + 1;

    ColorValue neighbor_values[9];
    int index = 0;
    #pragma unroll
    for (int i = -1; i <= 1; i++)
    {
        #pragma unroll
        for (int j = -1; j <=1; j++)
        {
            neighbor_values[index++] = values[(center_row+i) * width + (center_col+j)];
        }
    }

    int center_value = (int)neighbor_values[4];
    float weights[9];
    float sum_weights = 0.0f;
    float filtered_value = 0.0f;

    #pragma unroll
    for (int i = 0; i < 9; i++){
        int  difference = (int)center_value - (int)neighbor_values[i];
        weights[i] = d_w_spatial[i] * d_expf_look_up[abs(difference)];
        sum_weights += weights[i];
        filtered_value += weights[i] * (float)neighbor_values[i];
    }

    filtered_value = filtered_value / sum_weights;

    return d_clamp_pixel_value(filtered_value);
}

template <const uint BLOCKSIZE_X, const uint BLOCKSIZE_Y, const uint PIXELS_PER_THREAD>
__global__ void apply_filter_kernel(ColorValue* __restrict__ input_r_values,
                                    ColorValue* __restrict__ input_g_values,
                                    ColorValue* __restrict__ input_b_values,
                                    ColorValue* __restrict__ output_r,
                                    ColorValue* __restrict__ output_g,
                                    ColorValue* __restrict__ output_b,
                                    float* __restrict__ d_expf_look_up,
                                    int width, int height)
{   
    int local_col = threadIdx.x;
    int local_row = threadIdx.y;
    int global_col = (blockIdx.x * BLOCKSIZE_X + threadIdx.x) * PIXELS_PER_THREAD;
    int global_row = blockIdx.y * BLOCKSIZE_Y + threadIdx.y;

    __shared__ ColorValue shared_input_r_values[(BLOCKSIZE_X * PIXELS_PER_THREAD + 2) * (BLOCKSIZE_Y + 2)];
    __shared__ ColorValue shared_input_g_values[(BLOCKSIZE_X * PIXELS_PER_THREAD + 2) * (BLOCKSIZE_Y + 2)];
    __shared__ ColorValue shared_input_b_values[(BLOCKSIZE_X * PIXELS_PER_THREAD + 2) * (BLOCKSIZE_Y + 2)];

    int global_col_start = blockIdx.x * BLOCKSIZE_X * PIXELS_PER_THREAD;
    int global_row_start = blockIdx.y * BLOCKSIZE_Y;

    int tid = threadIdx.y * BLOCKSIZE_X + threadIdx.x;

    for (int i = tid; i < (BLOCKSIZE_X * PIXELS_PER_THREAD + 2) * (BLOCKSIZE_Y + 2); i += BLOCKSIZE_X * BLOCKSIZE_Y)
    {
        int local_copy_col = i % (BLOCKSIZE_X * PIXELS_PER_THREAD + 2);
        int local_copy_row = i / (BLOCKSIZE_X * PIXELS_PER_THREAD + 2);

        int global_copy_col = global_col_start + local_copy_col - 1;
        int global_copy_row = global_row_start + local_copy_row - 1;

        if (global_copy_row >= 0 && global_copy_row < height && global_copy_col >= 0 && global_copy_col < width)
        {
            shared_input_r_values[i] = input_r_values[global_copy_row * width + global_copy_col];
            shared_input_g_values[i] = input_g_values[global_copy_row * width + global_copy_col];
            shared_input_b_values[i] = input_b_values[global_copy_row * width + global_copy_col];
        }
    }
    __syncthreads();

    if (global_row >= 1 && global_row < height - 1)
    {
        #pragma unroll 
        for (int i = 0; i < PIXELS_PER_THREAD; i++)
        {
            int current_global_col = global_col + i;
            if (current_global_col >= 1 && current_global_col < width -1)
            {
                int current_local_col = local_col * PIXELS_PER_THREAD + i;
                
                ColorValue red   = d_bilateral_filter(shared_input_r_values, d_expf_look_up, local_row, current_local_col, BLOCKSIZE_X * PIXELS_PER_THREAD + 2);
                ColorValue green = d_bilateral_filter(shared_input_g_values, d_expf_look_up, local_row, current_local_col, BLOCKSIZE_X * PIXELS_PER_THREAD + 2);
                ColorValue blue  = d_bilateral_filter(shared_input_b_values, d_expf_look_up, local_row, current_local_col, BLOCKSIZE_X * PIXELS_PER_THREAD + 2);
                
                output_r[global_row * width + current_global_col] = red;
                output_g[global_row * width + current_global_col] = green;
                output_b[global_row * width + current_global_col] = blue;
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image in structure-of-array form
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    const int width = input_jpeg.width;
    const int height = input_jpeg.height;
    const int num_channels = input_jpeg.num_channels;
    auto output_r_values = new ColorValue[width * height];
    auto output_g_values = new ColorValue[width * height];
    auto output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    ColorValue* __restrict__ input_r_values = input_jpeg.get_channel(0);
    ColorValue* __restrict__ input_g_values = input_jpeg.get_channel(1);
    ColorValue* __restrict__ input_b_values = input_jpeg.get_channel(2);
    ColorValue* output_r = output_jpeg.r_values;
    ColorValue* output_g = output_jpeg.g_values;
    ColorValue* output_b = output_jpeg.b_values;

    ColorValue* d_input_r_values;
    ColorValue* d_input_g_values;
    ColorValue* d_input_b_values;
    ColorValue* d_output_r;
    ColorValue* d_output_g;
    ColorValue* d_output_b;
    size_t buffer_size = width * height;
    cudaMalloc((void**)&d_input_r_values, buffer_size);
    cudaMalloc((void**)&d_input_g_values, buffer_size);
    cudaMalloc((void**)&d_input_b_values, buffer_size);
    cudaMalloc((void**)&d_output_r, buffer_size);
    cudaMalloc((void**)&d_output_g, buffer_size);
    cudaMalloc((void**)&d_output_b, buffer_size);

    cudaMemcpy(d_input_r_values, input_r_values, buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_g_values, input_g_values, buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b_values, input_b_values, buffer_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_r, d_input_r_values, buffer_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_output_g, d_input_g_values, buffer_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_output_b, d_input_b_values, buffer_size, cudaMemcpyDeviceToDevice);

    const float h_w_border = expf(-0.5f / (SIGMA_D * SIGMA_D));
    const float h_w_corner = expf(-1.0f / (SIGMA_D * SIGMA_D));
    const float h_sigma_r_sq_inv = -0.5f / (SIGMA_R * SIGMA_R);
    const float w_spatial[9] = {
        h_w_corner, h_w_border, h_w_corner,
        h_w_border, 1.0f,       h_w_border,
        h_w_corner, h_w_border, h_w_corner
    };
    const int max_difference = 255;
    float h_expf_look_up [max_difference+1];

    for (int i = 0; i <= max_difference; i++)
    {
        float diff_sq = static_cast<float>(i * i);
        h_expf_look_up[i] = expf(diff_sq * h_sigma_r_sq_inv);
    }

    cudaMemcpyToSymbol(d_w_border, &h_w_border, sizeof(float));
    cudaMemcpyToSymbol(d_w_corner, &h_w_corner, sizeof(float));
    cudaMemcpyToSymbol(d_sigma_r_sq_inv, &h_sigma_r_sq_inv, sizeof(float));
    cudaMemcpyToSymbol(d_w_spatial, w_spatial, sizeof(w_spatial));

    float* d_expf_look_up;
    cudaMalloc((void**)&d_expf_look_up, 256 * sizeof(float));
    cudaMemcpy(d_expf_look_up, h_expf_look_up, 256 * sizeof(float), cudaMemcpyHostToDevice);

    const unsigned int PIXELS_PER_THREAD = 2;
    const unsigned int BLOCKSIZE_X = 32;
    const unsigned int BLOCKSIZE_Y = 8;

    dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 gridDim((width + (BLOCKSIZE_X * PIXELS_PER_THREAD) - 1) / (BLOCKSIZE_X * PIXELS_PER_THREAD),
                 (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    // Launch CUDA kernel
    apply_filter_kernel<BLOCKSIZE_X, BLOCKSIZE_Y, PIXELS_PER_THREAD><<<gridDim, blockDim>>>(
        d_input_r_values,
        d_input_g_values,
        d_input_b_values,
        d_output_r,
        d_output_g,
        d_output_b,
        d_expf_look_up,
        width,
        height);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from GPU
    cudaMemcpy(output_r, d_output_r, buffer_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(output_g, d_output_g, buffer_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(output_b, d_output_b, buffer_size,
               cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    // Release GPU memory
    cudaFree(d_input_r_values);
    cudaFree(d_input_g_values);
    cudaFree(d_input_b_values);
    cudaFree(d_output_r);
    cudaFree(d_output_g);
    cudaFree(d_output_b);
    cudaFree(d_expf_look_up);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
