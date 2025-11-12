//
// Created by Mengkang Li on 2025/10/27.
//
// Parallel Binary Search for Data Array on CPU
//

#include <iostream>
#include <vector>
#include "../utils.hpp"
#include <climits>

// Binary Search - finds the FIRST occurrence of targets from [i, i + BATCH_SIZE - 1] in range [0, size - 1]
#pragma acc routine
int find_partition(const int* __restrict__ vec, const int* __restrict__ search, const int i_min, const int i_max, const int diag, const int search_size) 
{
    int low = i_min;
    int high = i_max;
    int result = i_min;

    while (low <= high)
    {
        int i_mid = low + (high - low) / 2;
        int j_mid = diag - i_mid;

        int val_A = (i_mid == 0) ? INT_MIN : vec[i_mid - 1];
        int val_B = (j_mid == search_size) ? INT_MAX : search[j_mid];

        if (val_A <= val_B)
        {
            result = i_mid;
            low = i_mid + 1;
        }
        else
        {
            high = i_mid - 1;
        }
    }

    return result;
}

std::vector<int> binarySearchArray(const std::vector<int>& vec, 
                                    const std::vector<int>& search_targets) {
    const int n = vec.size();
    const int nbits = 31 - __builtin_clz(n); // log2(n)
    const int search_size = search_targets.size();
    const int parts = 4096;
    std::vector<int> partition_i(parts + 1);
    std::vector<int> partition_j(parts + 1);
    std::vector<int> results(search_size);

    const int* __restrict__ vec_ptr = vec.data();
    const int* __restrict__ target_ptr = search_targets.data();
    int * __restrict__ partition_i_ptr = partition_i.data();
    int * __restrict__ partition_j_ptr = partition_j.data();
    int* __restrict__ results_ptr = results.data();

    #pragma acc data copyin(vec_ptr[0:n], target_ptr[0:search_size]) \
                    create (partition_i_ptr[0:parts+1], partition_j_ptr[0:parts+1]) \ 
                    copyout(results_ptr[0:search_size])
    {
        #pragma acc parallel loop gang vector
        for (int k = 0; k <= parts; k++)
        {
            int diag = k * (n + search_size) / parts;
            int i_min = std::max(0, diag - search_size);
            int i_max = std::min(n, diag);
            int i = find_partition(vec_ptr, target_ptr, i_min, i_max, diag, search_size);
            partition_i_ptr[k] = i;
            partition_j_ptr[k] = diag - i;
        }

        #pragma acc parallel loop gang vector
        for (int k = 0; k < parts; k++)
        {
            int i_begin = partition_i_ptr[k];
            int i_end = partition_i_ptr[k+1];
            int j_begin = partition_j_ptr[k];
            int j_end = partition_j_ptr[k+1];
            while (j_begin < j_end)
            {
                while (i_begin < i_end && vec_ptr[i_begin] < target_ptr[j_begin])
                {
                    i_begin++;
                }

                results_ptr[j_begin] = i_begin;

                j_begin++;
            }
        }
    }

    return results;
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 2) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable vector_size\n"
            );
    }
    const int size = atoi(argv[1]);
    
    std::vector<int> vec = createUniformVec(size);
    std::sort(vec.begin(), vec.end());
    
    const int search_size = size / 10;
    std::vector<int> search_targets(search_size);
    
    std::mt19937 gen(CSC4005_SEED);
    std::uniform_int_distribution<> dis(0, size - 1);
    
    for (int i = 0; i < search_size; i++) {
        int idx = dis(gen);
        search_targets[i] = vec[idx];
    }
    std::sort(search_targets.begin(), search_targets.end());
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<int> results = binarySearchArray(vec, search_targets);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Parallel Array Binary Search Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    
    checkSearchResult(vec, search_targets, results);
    
    return 0;
}

