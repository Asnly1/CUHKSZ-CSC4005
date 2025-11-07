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

int prefix_sum_parallel(std::vector<int> &vec, std::vector<int> &result, int low, int high) {
    int n = high - low;
    if (n <= 0) {
        return 0;
    }

    int total_sum = 0;
    std::vector<int> block_sums;
    std::vector<int> block_offsets;
    int num_threads = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            block_sums.resize(num_threads);
            block_offsets.resize(num_threads);
        }

        int local_sum = 0;
        #pragma omp for schedule(static) nowait
        for (int i = low; i < high; i++) {
            local_sum += vec[i];
        }
        block_sums[tid] = local_sum;

        #pragma omp barrier

        #pragma omp single
        {
            int offset = 0;
            for (int i = 0; i < num_threads; i++) {
                block_offsets[i] = offset;
                offset += block_sums[i];
            }
            total_sum = offset;
        }

        int thread_offset = block_offsets[tid];
        #pragma omp for schedule(static)
        for (int i = low; i < high; i++) {
            result[i] = thread_offset;
            thread_offset += vec[i];
        }
    }

    return total_sum;
}

int partition_parallel(std::vector<int> &vec, std::vector<int> &S, std::vector<int> &L, 
                       std::vector<int> &S_prefix_sum, std::vector<int> &L_prefix_sum, std::vector<int> &temp,
                       int low, int high) {
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
    int num_small;
    int no_use;
    
    #pragma omp parallel for
    for (int i = low; i < high; i++)
    {
        S[i] = (vec[i] <= pivot) ? 1 : 0;
        L[i] = 1 - S[i];
    }

    num_small = prefix_sum_parallel(S, S_prefix_sum, low, high);
    no_use = prefix_sum_parallel(L, L_prefix_sum, low, high);

    #pragma omp parallel for
    for (int i = low; i < high; i++)
    {
        int x = vec[i];
        if (x <= pivot)
        {
            temp[low+S_prefix_sum[i]] = x;
        }
        else
        {
            temp[low+num_small+L_prefix_sum[i]] = x;
        }
    }

    #pragma omp parallel for
    for (int i = low; i < high; i++)
    {
        vec[i] = temp[i];
    }

    std::swap(vec[low+num_small], vec[high]);

    return low + num_small;
}

void quickSort(std::vector<int> &vec, std::vector<int> &S, std::vector<int> &L,
               std::vector<int> &S_prefix_sum, std::vector<int> &L_prefix_sum, std::vector<int> &temp,
               int low, int high, int depth, int max_depth) {
    if (low < high) {
        int pivotIndex;
        if (high - low <= 1024 || depth > max_depth || omp_get_max_threads() < 2)
        {
            pivotIndex = partition_sequential(vec, low, high);
        }
        else
        {
            pivotIndex = partition_parallel(vec, S, L, S_prefix_sum, L_prefix_sum, temp, low, high);
        }
        if (depth < max_depth)
        {
            #pragma omp task
            quickSort(vec, S, L, S_prefix_sum, L_prefix_sum, temp, low, pivotIndex - 1, depth+1, max_depth);
            #pragma omp task
            quickSort(vec, S, L, S_prefix_sum, L_prefix_sum, temp, pivotIndex + 1, high, depth+1, max_depth);
            #pragma omp taskwait
        }
        else
        {
            quickSort(vec, S, L, S_prefix_sum, L_prefix_sum, temp, low, pivotIndex - 1, depth+1, max_depth);
            quickSort(vec, S, L, S_prefix_sum, L_prefix_sum, temp, pivotIndex + 1, high, depth+1, max_depth);  
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
    std::vector<int> L(size);
    std::vector<int> S_prefix_sum(size);
    std::vector<int> L_prefix_sum(size);
    std::vector<int> temp(size);

    omp_set_num_threads(thread_num);
    int max_depth = log2(thread_num);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            quickSort(vec, S, L, S_prefix_sum, L_prefix_sum, temp, 0, size - 1, 0, max_depth);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Quick Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);

    return 0;
}