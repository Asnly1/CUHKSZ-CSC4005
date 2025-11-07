//
// Created by Fang Zihao on 2025/10/31.
// Email: zihaofang1@link.cuhk.edu.cn
//
// Parallel Quick Sort
//

#include <iostream>
#include <vector>
#include <thread>
#include <omp.h> 
#include "../utils.hpp"
#include <cmath>
#include <cstring>

int partition_sequential(std::vector<int> &vec, int low, int high) {
    int mid = low + (high - low) / 2;
    if ((vec[low] <= vec[mid] && vec[mid] <= vec[high]) || (vec[high] <= vec[mid] && vec[mid] <= vec[low])) 
    {
        std::swap(vec[mid], vec[high]);
    } 
    else if ((vec[mid] <= vec[low] && vec[low] <= vec[high]) || (vec[high] <= vec[low] && vec[low] <= vec[mid])) 
    {
        std::swap(vec[low], vec[high]);
    }
    int pivot = vec[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (vec[j] <= pivot) {
            i++;
            std::swap(vec[i], vec[j]);
        }
    }

    std::swap(vec[i + 1], vec[high]);
    return i + 1;
}

std::pair<int, std::vector<int>> prefix_sum_parallel(std::vector<int> &vec, int low, int high) {
    std::vector<int> block_sums;
    std::vector<int> block_offsets;
    int num_threads;
    int n = high - low;
    if (n <= 0) 
    {
        return {0, block_offsets};
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            block_sums.assign(num_threads, 0);
            block_offsets.assign(num_threads, 0);
        }

        int chunk_size = n / num_threads;
        int rem = n % num_threads;
        int length = tid < rem ? (chunk_size + 1) : chunk_size;
        int offset = tid < rem ? (chunk_size + 1) * tid : rem + chunk_size * tid;
        int begin = low + offset;
        int end = begin + length;

        int local_sum = 0;
        #pragma omp simd reduction(+:local_sum)
        for (int i = begin; i < end; ++i) 
        {
            local_sum += vec[i];
        }
        block_sums[tid] = local_sum;

        #pragma omp barrier
        #pragma omp single
        {
            int acc = 0;
            for (int t = 0; t < num_threads; ++t) 
            {
                block_offsets[t] = acc;
                acc += block_sums[t];
            }
        }
    }

        int total = 0;
        for (int v : block_sums) 
        {
            total += v;
        }
        return {total, block_offsets};
}

int partition_parallel(std::vector<int> &vec, std::vector<int> &S, std::vector<int> &temp,
                       int low, int high) {
    int mid = low + (high - low) / 2;
    // vec[mid] is median
    if ((vec[low] <= vec[mid] && vec[mid] <= vec[high]) || (vec[high] <= vec[mid] && vec[mid] <= vec[low])) 
    {
        std::swap(vec[mid], vec[high]);
    } 
    // vec[low] is median
    else if ((vec[mid] <= vec[low] && vec[low] <= vec[high]) || (vec[high] <= vec[low] && vec[low] <= vec[mid])) 
    {
        std::swap(vec[low], vec[high]);
    }
    int pivot = vec[high];
    int n = high - low;
    int num_small;
    std::vector<int>block_offsets;
    
    #pragma omp parallel for schedule(static)
    for (int i = low; i < high; i++)
    {
        S[i] = (vec[i] <= pivot) ? 1 : 0;
    }

    auto prefix_result = prefix_sum_parallel(S, low, high);
    num_small = prefix_result.first;
    block_offsets = prefix_result.second;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int chunk_size = n / num_threads;
        int rem = n % num_threads;
        int length = tid < rem ? (chunk_size + 1) : chunk_size;
        int offset = tid < rem ? (chunk_size + 1) * tid : rem + chunk_size * tid;
        int begin = low + offset;
        int end = begin + length;
        int small_block_offset = block_offsets[tid];
        int prior_elements = begin - low;
        int big_block_offset = prior_elements - small_block_offset;

        int local_prefix = 0;
        int local_large_prefix = 0;

        for (int i = begin; i < end; i++)
        {
            int x = vec[i];
            if (x <= pivot)
            {
                temp[low + small_block_offset + local_prefix] = x;
                local_prefix++;
            }
            else
            {
                temp[low + num_small + big_block_offset + local_large_prefix] = x;
                local_large_prefix++;
            }
        }
    }

    std::memcpy(&vec[low], &temp[low], sizeof(int) * (high - low));

    std::swap(vec[low+num_small], vec[high]);

    return low + num_small;
}

void quickSort(std::vector<int> &vec, std::vector<int> &S, std::vector<int> &temp,
               int low, int high, int depth, int max_depth) {
    if (low < high) {
        int pivotIndex;
        bool use_parallel = (high - low > 4096) && 
                            (depth < max_depth) && 
                            (omp_get_max_threads() > 1);
        if (use_parallel)
        {
            pivotIndex = partition_parallel(vec, S, temp, low, high);
        }
        else
        {
            pivotIndex = partition_sequential(vec, low, high);
        }
        if (use_parallel)
        {
             #pragma omp task default(none) \
                    firstprivate(low, pivotIndex, depth, max_depth) \
                    shared(vec, S, temp)
            quickSort(vec, S, temp, low, pivotIndex - 1, depth+1, max_depth);
             #pragma omp task default(none) \
                    firstprivate(high, pivotIndex, depth, max_depth) \
                    shared(vec, S, temp)
            quickSort(vec, S, temp, pivotIndex + 1, high, depth+1, max_depth);
        }
        else
        {
            quickSort(vec, S, temp, low, pivotIndex - 1, depth+1, max_depth);
            quickSort(vec, S, temp, pivotIndex + 1, high, depth+1, max_depth);  
        }
        if (use_parallel) {
            #pragma omp taskwait
        }
    }
}

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 3) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable threads_num vector_size\n"
            );
    }
    const int thread_num = atoi(argv[1]);
    const int size = atoi(argv[2]);
    std::vector<int> vec = createUniformVec(size); // use default seed
    std::vector<int> vec_clone = vec;

    std::vector<int> S(size);
    std::vector<int> temp(size);

    omp_set_num_threads(thread_num);
    int max_depth = log2(thread_num);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            quickSort(vec, S, temp, 0, size - 1, 0, max_depth);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    // checkSortResult(vec_clone, vec);

    return 0;
}