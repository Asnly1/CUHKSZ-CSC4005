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
    int total_sum = 0;

    #pragma omp for reduction(inscan, +:total_sum)
    for (int i = low; i < high; i++)
    {
        #pragma omp scan exclusive(total_sum)
        total_sum += vec[i];

        result[i] = total_sum;
    }

    return total_sum;
}

int partition_parallel(std::vector<int> &vec, std::vector<int> &S, std::vector<int> &L, 
                       std::vector<int> &S_prefix_sum, std::vector<int> &L_prefix_sum, std::vector<int> &temp,
                       int low, int high) {
    int pivot = vec[high];
    
    #pragma omp for
    for (int i = low; i < high; i++)
    {
        S[i] = (vec[i] <= pivot) ? 1 : 0;
        L[i] = 1 - S[i];
    }

    #pragma omp task
    int num_small = prefix_sum_parallel(S, S_prefix_sum, low, high);
    #pragma omp task
    int no_use = prefix_sum_parallel(L, L_prefix_sum, low, high);
    #pragma omp taskwait

    #pragma omp for
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

    #pragma omp for
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
        if (high - low <= 1000 || depth > max_depth)
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