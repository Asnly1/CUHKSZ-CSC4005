//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

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
__device__ ColorValue d_bilateral_filter(ColorValue* values,
                                    int row, int col, int width)
{
    static const float w_border = expf(-0.5f / (SIGMA_D * SIGMA_D));
    static const float w_corner = expf(-1.0f / (SIGMA_D * SIGMA_D));
    static const float sigma_r_sq_inv = -0.5f / (SIGMA_R * SIGMA_R);
    static const float w_spatial[9] = {
        w_corner, w_border, w_corner,
        w_border, 1.0f, w_border,
        w_corner, w_border, w_corner
    };

    const int indices[9] = {(row-1)*width + (col-1),
                            (row-1)*width + col,
                            (row-1)*width + (col+1),
                            row*width + (col-1),
                            row*width + col,
                            row*width + (col+1),
                            (row+1)*width + (col-1),
                            (row+1)*width + col,
                            (row+1)*width + (col+1)};
    ColorValue neighbor_values[9];

    for (int i = 0; i < 9; i++)
    {
        neighbor_values[i] = values[indices[i]];
    }
 
    float center_value = (float)neighbor_values[4];
    float weights[9];
    float sum_weights = 0.0f;
    float filtered_value = 0.0f;

    for (int i = 0; i < 9; i++){
        float difference = center_value - (float)neighbor_values[i];
        weights[i] = w_spatial[i] * expf(difference * difference * sigma_r_sq_inv);
        sum_weights += weights[i];
        filtered_value += weights[i] * (float)neighbor_values[i];
    }

    filtered_value = filtered_value / sum_weights;

    return d_clamp_pixel_value(filtered_value);
}

__device__ unsigned char d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

__global__ void apply_filter_kernel(ColorValue* input_r_values,
                                    ColorValue* input_g_values,
                                    ColorValue* input_b_values,
                                    ColorValue* output_r,
                                    ColorValue* output_g,
                                    ColorValue* output_b,
                                    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int index = y * width + x;
        ColorValue red = d_bilateral_filter(input_r_values, y, x, width);
        ColorValue green = d_bilateral_filter(input_g_values, y, x, width);
        ColorValue blue = d_bilateral_filter(input_b_values, y, x, width);
        output_r[index] = red;
        output_g[index] = green;
        output_b[index] = blue;
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

    cudaMemset(d_output_r, 0, buffer_size);
    cudaMemset(d_output_g, 0, buffer_size);
    cudaMemset(d_output_b, 0, buffer_size);

    cudaMemcpy(d_input_r_values, input_r_values, buffer_size,
            cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_g_values, input_g_values, buffer_size,
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_b_values, input_b_values, buffer_size,
        cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    // Launch CUDA kernel
    apply_filter_kernel<<<gridDim, blockDim>>>(
        d_input_r_values,
        d_input_g_values,
        d_input_b_values,
        d_output_r,
        d_output_g,
        d_output_b,
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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
