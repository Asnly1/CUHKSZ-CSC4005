//
// Created by Yang Yufan on 2023/10/31.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Parallel Merge Sort
//

#include <iostream>
#include <vector>
#include "../utils.hpp"
#include <utility>
#include <climits>
#include <omp.h> 
#include <cmath>

std::pair<int, int> findSplit(std::vector<int>& vec, int l_start, int l_end, int r_start, int r_end, int k)
{
    int n1 = l_end - l_start + 1;
    int n2 = r_end - r_start + 1;

    if (n1 == 0) 
    {
        return {0, k};
    }
    if (n2 == 0) 
    {
        return {k, 0};
    }

    if (n1 > n2) 
    {
        auto split = findSplit(vec, r_start, r_end, l_start, l_end, k);
        return {split.second, split.first};
    }

    int low = std::max(0, k - n2);
    int high = std::min(k, n1);

    while (low <= high)
    {
        int i = (low + high) / 2;
        int j = k - i;

        int L_left = (i == 0) ? INT_MIN : vec[l_start + i - 1];
        int L_right = (i == n1) ? INT_MAX : vec[l_start + i];
        int R_left = (j == 0) ? INT_MIN : vec[r_start + j - 1];
        int R_right = (j == n2) ? INT_MAX : vec[r_start + j];

        if (L_left <= R_right && R_left <= L_right)
        {
            return {i ,j};
        }
        else if (L_left > R_right)
        {
            high = i - 1;
        }
        else if (R_left > L_right)
        {
            low = i + 1;
        }
    }
    return {-1, -1};
}

void insertionSort(std::vector<int>& vec, int low, int high) {
    for (int i = low+1; i <= high; ++i) {
        int key = vec[i], j = i - 1;
        while (j >= low && vec[j] > key) {
            vec[j + 1] = vec[j];
            j--;
        }
        vec[j + 1] = key;
    }
}

// Merge two subarrays of vector vec[]
// First subarray is vec[l..m]
// Second subarray is vec[m+1..r]
void sequentialMerge(std::vector<int>& dest, std::vector<int>& src, int l, int m, int r) 
{
    sequentialMergeHelper(src, l, m, m + 1, r, dest, l);
}

void sequentialMergeHelper(const std::vector<int>& src, int l_start, int l_end,
                           int r_start, int r_end, std::vector<int>& dest, int des_start)
{
    int i = l_start;
    int j = r_start;
    int k = des_start;

    while (i <= l_end && j <= r_end) 
    {
        if (src[i] <= src[j]) 
        {
            dest[k] = src[i];
            i++;
        } 
        else 
        {
            dest[k] = src[j];
            j++;
        }
        k++;
    }

    while (i <= l_end) 
    { 
        dest[k] = src[i];
        i++;
        k++;
    }

    while (j <= r_end) 
    { 
        dest[k] = src[j];
        j++;
        k++;
    }
}
                           
void parMerge(std::vector<int>& dest, int dest_start, const std::vector<int>& src,
              int l_start, int l_end, 
              int r_start, int r_end,
              int depth, int max_depth)
{
    int n1 = l_end - l_start + 1;
    int n2 = r_end - r_start + 1;
    int total_size = n1 + n2;

    if (total_size < 2048 || depth > max_depth) {
        sequentialMergeHelper(src, l_start, l_end, r_start, r_end, dest, dest_start);
        return;
    }

    int k = total_size / 2;

    auto split = findSplit(src, l_start, l_end, r_start, r_end, k);
    int i = split.first;
    int j = split.second;

    #pragma omp task shared(dest, src)
    {
        parMerge(dest, dest_start, src,
                 l_start, l_start + i - 1,
                 r_start, r_start + j - 1,
                 depth + 1, max_depth);
    }

    #pragma omp task shared(dest, src)
    {
        parMerge(dest, dest_start + k, src,
                 l_start + i, l_end,
                 r_start + j, r_end,
                 depth + 1, max_depth);
    }
    #pragma omp taskwait
}

void parMergeSort(std::vector<int>& dest, std::vector<int>& src, int l, int r, int depth, int max_depth) {
    if (l == r) {
        dest[l] = src[l];
        return;
    }

    int vec_length = r - l + 1;
    if (vec_length <= 64)
    {
        std::copy(src.begin() + l, src.begin() + r + 1, dest.begin() + l);
        insertionSort(dest, l, r);
        return;
    }

    int m = l + (r - l) / 2;

    if (depth < max_depth) {
        #pragma omp taskgroup
        {
            #pragma omp task shared(src, dest)
            parMergeSort(src, dest, l, m, depth + 1, max_depth);

            #pragma omp task shared(src, dest)
            parMergeSort(src, dest, m + 1, r, depth + 1, max_depth);

            #pragma omp taskwait
        }
        
        parMerge(dest, l, src, l, m, m + 1, r, depth + 1, max_depth);
    }
    else
    {
        parMergeSort(src, dest, l, m, depth + 1, max_depth);
        parMergeSort(src, dest, m + 1, r, depth + 1, max_depth);
        sequentialMerge(dest, src, l, m, r);
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
    std::vector<int> buffer = vec;

    omp_set_num_threads(thread_num);
    int max_depth = log2(thread_num);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parMergeSort(buffer, vec, 0, size-1, 0, max_depth);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, buffer);
}
