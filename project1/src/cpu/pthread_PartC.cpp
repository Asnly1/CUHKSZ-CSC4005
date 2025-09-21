//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

struct ThreadData
{
    ColorValue* input_r_values;
    ColorValue* input_g_values;
    ColorValue* input_b_values;
    ColorValue* output_r;
    ColorValue* output_g;
    ColorValue* output_b;
    int width;
    int height;
    int num_channels;
    int start_row;
    int end_row;
};

void* filter_thread_function(void* arg)
{
    ThreadData* data = (ThreadData*)arg;

    for (int y = data->start_row; y < data->end_row; y++)
    {
        for (int x = 1; x < data->width - 1; x++)
        {
            int index = y * data->width + x;
            ColorValue red = bilateral_filter(data->input_r_values, y, x, data->width);
            ColorValue green = bilateral_filter(data->input_g_values, y, x, data->width);
            ColorValue blue = bilateral_filter(data->input_b_values, y, x, data->width);
            data->output_r[index] = red;
            data->output_g[index] = green;
            data->output_b[index] = blue;
        }
    }
    return nullptr;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);

    int NUM_THREADS = std::stoi(argv[3]); // Convert the input to integer
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

    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];

    // 总共需要处理的行数
    int total_line_num = height - 2;
    int line_per_thread = total_line_num / NUM_THREADS;
    int left_line_num = total_line_num % NUM_THREADS;

    int current_start_row = 1; // 从第 1 行开始处理

    for (int i = 0; i < NUM_THREADS; i++)
    {
        int rows_of_this_thread = line_per_thread;
        if (i < left_line_num) { // 将余数行分配给前面的线程
            rows_of_this_thread++;
        }
        int end_row = current_start_row + rows_of_this_thread;

        threadData[i] = {input_r_values,
                         input_g_values,
                         input_b_values,
                         output_r,
                         output_g,
                         output_b,
                         width,
                         height,
                         num_channels,
                         current_start_row,
                         end_row};

        current_start_row = end_row;
    }

    auto start_time =
        std::chrono::high_resolution_clock::now(); // Start time recording

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, filter_thread_function,
                       &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // clean up
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    delete[] threads;
    delete[] threadData;

    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
