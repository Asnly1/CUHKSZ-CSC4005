//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

#pragma acc routine seq
ColorValue acc_bilateral_filter(const ColorValue* values, int row, int col, int width)
{
    const float w_border = expf(-0.5f / (SIGMA_D * SIGMA_D));
    const float w_corner = expf(-1.0f / (SIGMA_D * SIGMA_D));
    const float sigma_r_sq_inv = -0.5f / (SIGMA_R * SIGMA_R);
    const float w_spatial[9] = {
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

    return acc_clamp_pixel_value(filtered_value);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // Apply the filter to the image
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
    size_t buffer_size = width * height;

    #pragma acc enter data copyin(input_r_values[0:buffer_size], \
                                input_g_values[0:buffer_size], \
                                input_b_values[0:buffer_size]) \
                        create(output_r[0:buffer_size], \
                                output_g[0:buffer_size], \
                                output_b[0:buffer_size])

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma acc parallel present(input_r_values[0 : buffer_size], \
                                input_g_values[0 : buffer_size], \
                                input_b_values[0 : buffer_size], \
                                output_r[0 : buffer_size], \
                                output_g[0 : buffer_size], \
                                output_b[0 : buffer_size]) \
                                num_gangs(1024)
    {
    #pragma acc loop independent
        for (int y = 1; y < height - 1; y++)
        {
    #pragma acc loop independent
            for (int x = 1; x < width - 1; x++)
            {
                int index = y * width + x;
                ColorValue red = acc_bilateral_filter(input_r_values, y, x, width);
                ColorValue green = acc_bilateral_filter(input_g_values, y, x, width);
                ColorValue blue = acc_bilateral_filter(input_b_values, y, x, width);
                output_r[index] = red;
                output_g[index] = green;
                output_b[index] = blue;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    #pragma acc exit data copyout(output_r[0:buffer_size], \
                                output_g[0:buffer_size], \
                                output_b[0:buffer_size]) \
                            delete(input_r_values[0:buffer_size], \
                                input_g_values[0:buffer_size], \
                                input_b_values[0:buffer_size])

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
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
