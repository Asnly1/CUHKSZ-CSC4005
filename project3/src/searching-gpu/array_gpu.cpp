//
// Created by Mengkang Li on 2025/10/27.
//
// Parallel Binary Search for Data Array on CPU
//

#include <iostream>
#include <vector>
#include "../utils.hpp"

// Binary Search - finds the FIRST occurrence of targets from [i, i + BATCH_SIZE - 1] in range [0, size - 1]
#pragma acc routine seq
static inline void binarySearch(const int* __restrict__ vec, const int size, const int bits, const int* __restrict__ targets, const int i, int* __restrict__ results) {
    const register int target = targets[i];
    int register idx = -1;

    for (register int step = 1 << bits; step > 0; step >>= 1)
    {
        register int pos = idx + step;
        register int safe = (pos < size) ? pos : size - 1;

        register int valid = pos < size && vec[safe] < target;
        idx += valid * step;
    }

    results[i] = idx + 1;
}

std::vector<int> binarySearchArray(const std::vector<int>& vec, 
                                    const std::vector<int>& search_targets) {
    const int n = vec.size();
    const int nbits = 31 - __builtin_clz(n); // log2(n)
    const int search_size = search_targets.size();
    std::vector<int> results(search_size);

    const int* __restrict__ vec_ptr = vec.data();
    const int* __restrict__ target_ptr = search_targets.data();
    int* __restrict__ results_ptr = results.data();

    #pragma acc data copyin(vec_ptr[0:n], target_ptr[0:search_size]) copyout(results_ptr[0:search_size])
    {
        #pragma acc parallel loop vector_length(128)
        for (int i = 0; i < search_size; i++)
        {
            binarySearch(vec_ptr, n, nbits, target_ptr, i, results_ptr);
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

