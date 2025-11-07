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
    int n = vec.size();
    int *vec_raw = vec.data();

    int *output = new int[n];
    int count[BASE];
    int (*local_counts)[BASE] = new int[NUM_GANGS][BASE];
    int start_pos[BASE];
    int (*gang_prefix_sum)[BASE] = new int[NUM_GANGS][BASE];
    int (*local_offsets)[BASE] = new int[NUM_GANGS][BASE];

    #pragma acc data copy(vec_raw[0:n]) create(output[0:n], count[0:BASE], local_counts[0:NUM_GANGS][0:BASE], \
                          start_pos[0:BASE], gang_prefix_sum[0:NUM_GANGS][0:BASE], local_offsets[0:NUM_GANGS][0:BASE])
    {
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

            #pragma acc parallel num_gangs(NUM_GANGS)
            {
                int gang_id = __pgi_gangidx();
                int chunk_size = (n + NUM_GANGS - 1) / NUM_GANGS;
                int start = gang_id * chunk_size;
                int end = (start + chunk_size > n) ? n : (start + chunk_size);
                
                #pragma acc loop worker vector
                for (int i = start; i < end; i++) 
                {
                    int digit = (vec_raw[i] >> shift) & (BASE - 1);

                    #pragma acc atomic update
                    local_counts[gang_id][digit]++;
                }
            }

            #pragma acc wait
            
            #pragma acc parallel loop
            for (int b = 0; b < BASE; b++) {
                int sum = 0;
                for (int g = 0; g < NUM_GANGS; g++) 
                {
                    sum += local_counts[g][b];
                }
                
                count[b] = sum;
            }
            
            #pragma acc wait

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
                for (int g = 1; g < NUM_GANGS; g++) 
                {
                    gang_prefix_sum[g][b] = gang_prefix_sum[g-1][b] + local_counts[g-1][b];
                }
            }
            
            #pragma acc wait
            
            #pragma acc parallel loop collapse(2)
            for (int g = 0; g < NUM_GANGS; g++) 
            {
                for (int b = 0; b < BASE; b++) 
                {
                    local_offsets[g][b] = 0;
                }
            }

            #pragma acc parallel num_gangs(NUM_GANGS)
            {
                int gang_id = __pgi_gangidx();
                int chunk_size = (n + NUM_GANGS - 1) / NUM_GANGS;
                int start = gang_id * chunk_size;
                int end = (start + chunk_size > n) ? n : (start + chunk_size);

                #pragma acc loop seq
                for (int i = start; i < end; i++) {
                    int digit = (vec_raw[i] >> shift) & (BASE - 1);

                    int global_start = start_pos[digit];
                    int prior_gang_offset = gang_prefix_sum[gang_id][digit];
                    int within_gang_offset = local_offsets[gang_id][digit];
                    local_offsets[gang_id][digit]++;

                    int pos = global_start + prior_gang_offset + within_gang_offset;
                    output[pos] = vec_raw[i];
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