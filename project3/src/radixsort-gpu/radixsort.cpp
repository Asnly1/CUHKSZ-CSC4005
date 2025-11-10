//
// Created by Lyu You on 2024/10/16
// Email: 121090404@link.cuhk.edu.cn
//
// Sequential Radix Sort
//

#include <openacc.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../utils.hpp"

#define BASE 256
#define BASE_BITS 8
#define NUM_GANGS 1024

void radixSort(std::vector<int> &vec) {
    const int n = vec.size();
    int* __restrict__ vec_raw = vec.data();

    int* __restrict__ output = new int[n];
    int count[BASE];
    int (* __restrict__ local_counts)[BASE] = new int[NUM_GANGS][BASE];
    int start_pos[BASE];
    int (* __restrict__ gang_prefix_sum)[BASE] = new int[NUM_GANGS][BASE];
    int (* __restrict__ local_offsets)[BASE] = new int[NUM_GANGS][BASE];

    const int tile_size = 2048;

    #pragma acc data copy(vec_raw[0:n]) create(output[0:n], count[0:BASE], local_counts[0:NUM_GANGS][0:BASE], \
                          start_pos[0:BASE], gang_prefix_sum[0:NUM_GANGS][0:BASE], local_offsets[0:NUM_GANGS][0:BASE])
    {
        const int chunk_size = (n + NUM_GANGS - 1) / NUM_GANGS;
        for (int shift = 0; shift < 32; shift += BASE_BITS)
        {
            #pragma acc parallel loop collapse(2)
            for (int g = 0; g < NUM_GANGS; g++) 
            {
                for (int b = 0; b < BASE; b++) 
                {
                    local_counts[g][b] = 0;
                }
            }

            #pragma acc parallel loop gang num_gangs(NUM_GANGS) vector_length(128)
            for (int gid = 0; gid < NUM_GANGS; gid++)
            {
                const int start = gid * chunk_size;
                const int end = (start + chunk_size > n) ? n : (start + chunk_size);
                
                #pragma acc loop worker vector
                for (int i = start; i < end; i++) 
                {
                    int d = (vec_raw[i] >> shift) & (BASE - 1);

                    #pragma acc atomic update
                    local_counts[gid][d]++;
                }
            }
            
            #pragma acc parallel loop
            for (int b = 0; b < BASE; b++) 
            {
                int sum = 0;
                #pragma acc loop vector reduction(+:sum)
                for (int g = 0; g < NUM_GANGS; g++)
                {
                    sum += local_counts[g][b];
                }
                count[b] = sum;
            }

            #pragma acc serial
            {
                start_pos[0] = 0;
                for (int d = 1; d < BASE; d++)
                {
                    start_pos[d] = start_pos[d - 1] + count[d - 1];
                }
            }

            #pragma acc parallel loop
            for (int b = 0; b < BASE; b++) 
            {
                gang_prefix_sum[0][b] = 0;
                #pragma acc loop seq
                for (int g = 1; g < NUM_GANGS; g++) 
                {
                    gang_prefix_sum[g][b] = gang_prefix_sum[g-1][b] + local_counts[g-1][b];
                }
            }

            #pragma acc parallel loop collapse(2)
            for (int g = 0; g < NUM_GANGS; ++g) {
                for (int b = 0; b < BASE; ++b) {
                    local_offsets[g][b] = start_pos[b] + gang_prefix_sum[g][b];
                }
            }

            #pragma acc parallel loop gang num_gangs(NUM_GANGS) vector_length(128)
            for (int gid = 0; gid < NUM_GANGS; gid++)
            {
                const int start = gid * chunk_size;
                const int end   = (start + chunk_size > n) ? n : (start + chunk_size);

                for (int tile = start; tile < end; tile += tile_size)
                {
                    const int t_end = (tile + tile_size > end) ? end : (tile + tile_size);
                    int tile_count[BASE];

                    #pragma acc loop vector
                    for (int d = 0; d < BASE; d++) 
                    {
                        tile_count[d] = 0;
                    }

                    #pragma acc loop worker vector
                    for (int i = tile; i < t_end; i++) 
                    {
                        int d = (vec_raw[i] >> shift) & (BASE - 1);
                        #pragma acc atomic update
                        tile_count[d]++;
                    }

                    int tile_base[BASE];
                    #pragma acc loop seq
                    for (int d = 0; d < BASE; d++)
                    {
                        int base_offset;
                        #pragma acc atomic capture
                        {   
                            base_offset = local_offsets[gid][d];
                            local_offsets[gid][d] += tile_count[d];
                        }
                        tile_base[d] = base_offset;
                    }

                    #pragma acc loop seq
                    for (int i = tile; i < t_end; i++) 
                    {
                        int d = (vec_raw[i] >> shift) & (BASE - 1);
                        int pos = tile_base[d];
                        tile_base[d]++;
                        output[pos] = vec_raw[i];
                    }
                }
            }

            #pragma acc parallel loop
            for (int i = 0; i < n; i++)
            {
                vec_raw[i] = output[i];
            }
        }
    }
    delete[] output;
    delete[] local_counts;
    delete[] gang_prefix_sum;
    delete[] local_offsets;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }

    const int size = atoi(argv[1]);

    const int seed = 4005;

    std::vector<int> vec = createUniformVec(size, seed);
    std::vector<int> vec_clone = vec;
    auto start_time = std::chrono::high_resolution_clock::now();

    radixSort(vec);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Radix Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    
    checkSortResult(vec_clone, vec);
    return 0;
}