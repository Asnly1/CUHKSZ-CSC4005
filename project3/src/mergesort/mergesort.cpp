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

std::pair<int, int> findSplit(std::vector<int>&L, std::vector<int>&R, int k)
{
    int n1 = L.size();
    int n2 = R.size();

    if (n1 > n2)
    {
        auto split = findSplit(R, L, k);
        return {split.second, split.first};
    }

    int low = std::max(0, k-n2);
    int high = std::min(k, n1);

    while (low <= high)
    {
        int i = (low + high) / 2;
        int j = k - i;

        int L_left = (i == 0) ? INT_MIN : L[i-1];
        int L_right = (i == n1) ? INT_MAX : L[i];
        int R_left = (j == 0) ? INT_MIN : R[j-1];
        int R_right = (j == n2) ? INT_MAX : R[j];

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
void sequentialMerge(std::vector<int>& vec, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    // Create temporary vectors
    std::vector<int> L(n1);
    std::vector<int> R(n2);

    // Copy data to temporary vectors L[] and R[]
    for (int i = 0; i < n1; i++) {
        L[i] = vec[l + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = vec[m + 1 + i];
    }

    // Merge the temporary vectors back into v[l..r]
    int i = 0; // Initial index of the first subarray
    int j = 0; // Initial index of the second subarray
    int k = l; // Initial index of the merged subarray

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        } else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i < n1) {
        vec[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j < n2) {
        vec[k] = R[j];
        j++;
        k++;
    }
}

void sequentialMergeHelper(const std::vector<int>& L, int l_start, int l_end,
                           const std::vector<int>& R, int r_start, int r_end,
                           std::vector<int>&vec, int des_start)
{
    int i = l_start;
    int j = r_start;
    int k = des_start;

    while (i <= l_end && j <= r_end) {
        if (L[i] <= R[j]) {
            vec[k] = L[i];
            i++;
        } else {
            vec[k] = R[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of L[], if there are any
    while (i <= l_end) {
        vec[k] = L[i];
        i++;
        k++;
    }

    // Copy the remaining elements of R[], if there are any
    while (j <= r_end) {
        vec[k] = R[j];
        j++;
        k++;
    }
}
                           
void parMerge(std::vector<int>& vec, int l, int m, int r, int depth, int max_depth) {
    if ((r - l + 1) < 100 || depth > max_depth)
    {
        sequentialMerge(vec, l, m, r);
    }
    else
    {
        int n1 = m - l + 1;
        int n2 = r - m;

        std::vector<int> L(n1);
        std::vector<int> R(n2);

        for (int i = 0; i < n1; i++) {
            L[i] = vec[l + i];
        }
        for (int i = 0; i < n2; i++) {
            R[i] = vec[m + 1 + i];
        }

        int k = (n1+n2) / 2;
        auto split = findSplit(L, R, k);
        int i = split.first;
        int j = split.second;

        #pragma omp task
        sequentialMergeHelper(L, 0, i-1, R, 0, j-1, vec, l);
        
        #pragma omp task
        sequentialMergeHelper(L, i, n1-1, R, j, n2-1, vec, l+k);
    }
}

void parMergeSort(std::vector<int>& vec, const int &l, const int &r, int depth, int max_depth) {
    if (l < r)
    {
        int vec_length = r - l + 1;
        if (vec_length <= 100)
        {
            insertionSort(vec, l ,r);
            return;
        }
        else
        {
            int m = l + (r - l) / 2;

            if (depth <= max_depth)
            {
                #pragma omp task
                parMergeSort(vec, l, m, depth+1, max_depth);

                #pragma omp task
                parMergeSort(vec, m + 1, r, depth+1, max_depth);

                #pragma omp taskwait
                parMerge(vec, l, m, r, depth + 1, max_depth);
            }
            else
            {
                parMergeSort(vec, l, m, depth+1, max_depth);
                parMergeSort(vec, m + 1, r, depth+1, max_depth);

                parMerge(vec, l, m, r, depth+1, max_depth);
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

    omp_set_num_threads(thread_num);
    int max_depth = log2(thread_num);
    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            parMergeSort(vec, 0, size-1, 0, max_depth);
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
