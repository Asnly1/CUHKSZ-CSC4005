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

std::pair<int, int> findSplit(std::vector<int>& buffer, int l_start, int l_end, int r_start, int r_end, int k)
{
    int n1 = l_end - l_start + 1;
    int n2 = r_end - r_start + 1;

    if (n1 == 0) return {0, k};
    if (n2 == 0) return {k, 0};

    if (n1 > n2) {
        auto split = findSplit(buffer, r_start, r_end, l_start, l_end, k);
        return {split.second, split.first};
    }

    int low = std::max(0, k - n2);
    int high = std::min(k, n1);

    while (low <= high)
    {
        int i = (low + high) / 2;
        int j = k - i;

        int L_left = (i == 0) ? INT_MIN : buffer[l_start + i - 1];
        int L_right = (i == n1) ? INT_MAX : buffer[l_start + i];
        int R_left = (j == 0) ? INT_MIN : buffer[r_start + j - 1];
        int R_right = (j == n2) ? INT_MAX : buffer[r_start + j];

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
void sequentialMerge(std::vector<int>& vec, std::vector<int>& buffer, int l, int m, int r) {
    for (int i = l; i <= r; i++) {
        buffer[i] = vec[i];
    }

    int i = l;
    int j = m + 1;
    int k = l;

    while (i <= m && j <= r) 
    {
        if (buffer[i] <= buffer[j]) 
        {
            vec[k] = buffer[i];
            i++;
        }
        else 
        {
            vec[k] = buffer[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i <= m) {
        vec[k] = buffer[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j <= r) {
        vec[k] = buffer[j];
        j++;
        k++;
    }
}

void sequentialMergeHelper(const std::vector<int>& buffer, int l_start, int l_end,
                           int r_start, int r_end, std::vector<int>& vec, int des_start)
{
    int i = l_start;
    int j = r_start;
    int k = des_start;

    while (i <= l_end && j <= r_end) 
    {
        if (buffer[i] <= buffer[j]) 
        {
            vec[k] = buffer[i];
            i++;
        } 
        else 
        {
            vec[k] = buffer[j];
            j++;
        }
        k++;
    }

    while (i <= l_end) 
    { 
        vec[k++] = buffer[i++];
    }

    while (j <= r_end) 
    { 
        vec[k++] = buffer[j++];
    }
}
                           
void parMerge(std::vector<int>& vec, std::vector<int>& buffer, int l, int m, int r, int depth, int max_depth) {
    if ((r - l + 1) < (1<<15) || depth > max_depth || omp_get_num_threads() < 2)
    {
        sequentialMerge(vec, buffer, l, m, r);
    }
    else
    {
        for (int iter = l; iter <= r; i++) {
            buffer[iter] = vec[iter];
        }

        int n1 = m - l + 1;
        int n2 = r - m;
        int k = (n1+n2) / 2;

        auto split = findSplit(buffer, l, m, m + 1, r, k);
        int i = split.first;
        int j = split.second;

        #pragma omp task shared(buffer, vec) firstprivate(l,m,r,i,j,k)
        sequentialMergeHelper(buffer, l, l + i - 1, m + 1, m + 1 + j - 1, vec, l);
        
        #pragma omp task shared(buffer, vec) firstprivate(l,m,r,i,j,k)
        sequentialMergeHelper(buffer, l + i, m, m + 1 + j, r, vec, l + k);

        #pragma omp taskwait
    }
}

void parMergeSort(std::vector<int>& vec, std::vector<int>& aux, const int &l, const int &r, int depth, int max_depth) {
    if (l < r)
    {
        int vec_length = r - l + 1;
        if (vec_length <= (1<<15))
        {
            insertionSort(vec, l ,r);
            return;
        }
        else
        {
            int m = l + (r - l) / 2;

            if (depth < max_depth)
            #pragma omp taskgroup
            {
                #pragma omp task shared(vec, aux) firstprivate(l, m, depth, max_depth)
                parMergeSort(vec, aux, l, m, depth+1, max_depth);

                #pragma omp task shared(vec, aux) firstprivate(m, r, depth, max_depth)
                parMergeSort(vec, aux, m + 1, r, depth+1, max_depth);

                #pragma omp taskwait
                parMerge(vec, aux, l, m, r, depth + 1, max_depth);
            }
            else
            {
                parMergeSort(vec, aux, l, m, depth+1, max_depth);
                parMergeSort(vec, aux, m + 1, r, depth+1, max_depth);

                sequentialMerge(vec, aux, l, m, r);
            }
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
    std::vector<int> buffer = vec;

    omp_set_num_threads(thread_num);
    int max_depth = log2(thread_num);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parMergeSort(vec, buffer, 0, size-1, 0, max_depth);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    std::cout << "Merge Sort Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    checkSortResult(vec_clone, vec);
}
