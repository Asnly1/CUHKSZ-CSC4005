//
// Created by Mengkang Li on 2025/10/27.
//
// Modified by Liu Yuxuan on 2024/10/28
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Task #4: Parallel Binary Search for Data Array on CPU
//

#include <iostream>
#include <vector>
#include <omp.h>
#include "../utils.hpp"

int binarySearch(const std::vector<int>& vec, int target, int left, int right) {
    int result = right;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (vec[mid] >= target) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    
    return result;
}

std::vector<int> binarySearchArray(const std::vector<int>& vec, 
                                    const std::vector<int>& search_targets) {
    std::vector<int> results(search_targets.size());
    int number_targets = search_targets.size();
    int vector_size = vec.size();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int number_threads = omp_get_num_threads();

        int chunk_size = (number_targets + number_threads - 1) / number_threads;
        int start = tid * chunk_size;
        int end = std::min(number_targets, (tid + 1) * chunk_size);

        int left_hint = 0;

        for (int i = start; i < end; i++)
        {
            int target = search_targets[i];

            int right_bound_step = 1;
            int right_bound_idx = left_hint;

            if (left_hint < vector_size && vec[left_hint] < target)
            {
                right_bound_idx = right_bound_idx + right_bound_step;
                while (right_bound_idx < vector_size && vec[right_bound_idx] < target)
                {
                    right_bound_step *= 2;
                    right_bound_idx += right_bound_step;
                }
            }

            int new_left = std::max(left_hint, right_bound_idx - right_bound_step);
            int new_right = std::min(vector_size - 1, right_bound_idx);

            int result = binarySearch(vec, target, new_left, new_right);
            results[i] = result;
            left_hint = result > vector_size ? vector_size : result;
        }
    }
    
    return results;
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
    omp_set_num_threads(thread_num);
    
    // Create and sort the array
    std::vector<int> vec = createUniformVec(size);
    std::sort(vec.begin(), vec.end());
    
    // Generate search targets (10% of array size)
    const int search_size = size / 10;
    std::vector<int> search_targets(search_size);
    
    std::mt19937 gen(CSC4005_SEED);
    std::uniform_int_distribution<> dis(0, size - 1);
    
    // Randomly select search targets from the sorted array
    for (int i = 0; i < search_size; i++) {
        int idx = dis(gen);
        search_targets[i] = vec[idx];
    }
    // Sort search targets to exploit locality (optional optimization)
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

