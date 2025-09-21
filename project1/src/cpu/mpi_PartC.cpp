//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

void set_filtered_image(ColorValue* values, ColorValue* output_values, int width, int start_line,
                        int end_line, int offset);

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;

    // Read input JPEG File
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    const int width = input_jpeg.width;
    const int height = input_jpeg.height;
    const int num_channels = input_jpeg.num_channels;
    ColorValue* __restrict__ input_r_values = input_jpeg.get_channel(0);
    ColorValue* __restrict__ input_g_values = input_jpeg.get_channel(1);
    ColorValue* __restrict__ input_b_values = input_jpeg.get_channel(2);
    auto output_r_values = new ColorValue[width * height];
    auto output_g_values = new ColorValue[width * height];
    auto output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};

    // Divide the task
    // For example, there are 11 lines and 3 tasks,
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_line_num = input_jpeg.height - 2;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 1); // 跳过第一个pixel
    int divided_left_line_num = 0;

    for (int i = 0; i < numtasks; i++)
    {
        if (divided_left_line_num < left_line_num)
        {
            cuts[i + 1] = cuts[i] + line_per_task + 1;
            divided_left_line_num++;
        }
        else
            cuts[i + 1] = cuts[i] + line_per_task;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (taskid == MASTER)
    {
        std::cout << "Input file from: " << input_filepath << "\n";
        ColorValue* output_r = output_jpeg.r_values;
        ColorValue* output_g = output_jpeg.g_values;
        ColorValue* output_b = output_jpeg.b_values;

        // // Filter the first division of the contents
        set_filtered_image(input_r_values, output_r, width, cuts[taskid], cuts[taskid+1], 0);
        set_filtered_image(input_g_values, output_g, width, cuts[taskid], cuts[taskid+1], 0);
        set_filtered_image(input_b_values, output_b, width, cuts[taskid], cuts[taskid+1], 0);

        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++)
        {
            int line_width = input_jpeg.width;
            ColorValue* start_pos_r = output_r+ cuts[i] * line_width;
            ColorValue* start_pos_g = output_g+ cuts[i] * line_width;
            ColorValue* start_pos_b = output_b+ cuts[i] * line_width;
            int length = (cuts[i + 1] - cuts[i]) * line_width;
            MPI_Recv(start_pos_r, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(start_pos_g, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
            MPI_Recv(start_pos_b, length, MPI_CHAR, i, TAG_GATHER, MPI_COMM_WORLD, &status);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

        // Save output JPEG image
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        if (export_jpeg(output_jpeg, output_filepath))
        {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }
        // Cleanup
        delete[] output_r_values;
        delete[] output_g_values;
        delete[] output_b_values;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds\n";
    }
    // The tasks for the slave executor
    // 1. Filter a division of image
    // 2. Send the Filterd contents back to the master executor
    else
    {
        int length = input_jpeg.width * (cuts[taskid + 1] - cuts[taskid]);
        int offset = input_jpeg.width * cuts[taskid];

        auto filteredImage_r = new unsigned char[length];
        auto filteredImage_g = new unsigned char[length];
        auto filteredImage_b = new unsigned char[length];
        set_filtered_image(input_r_values, filteredImage_r, width, cuts[taskid], cuts[taskid+1], offset);
        set_filtered_image(input_g_values, filteredImage_g, width, cuts[taskid], cuts[taskid+1], offset);
        set_filtered_image(input_b_values, filteredImage_b, width, cuts[taskid], cuts[taskid+1], offset);

        // Send the filtered image back to the master
        MPI_Send(filteredImage_r, length, MPI_CHAR, MASTER, TAG_GATHER,
                 MPI_COMM_WORLD);
        MPI_Send(filteredImage_g, length, MPI_CHAR, MASTER, TAG_GATHER,
                 MPI_COMM_WORLD);
        MPI_Send(filteredImage_b, length, MPI_CHAR, MASTER, TAG_GATHER,
                 MPI_COMM_WORLD);
        // Release the memory
        delete[] filteredImage_r;
        delete[] filteredImage_g;
        delete[] filteredImage_b;
    }

    MPI_Finalize();
    return 0;
}

void set_filtered_image(const ColorValue* input_values, ColorValue* output_values,
                        int width, int start_line,
                        int end_line, int offset)
{
    for (int y = start_line; y < end_line; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int index = y * width + x;
            ColorValue filtered_value = bilateral_filter(input_values, y, x, width);
            output_values[index-offset] = clamp_pixel_value(filtered_value);
        }
    }
}