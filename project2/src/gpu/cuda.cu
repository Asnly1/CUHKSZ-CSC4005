//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Matrix Multiplication with CUDA, for bonus
//

#include "../matrix.hpp"
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(((BM * BN) / (TM * TN)), 1)
    matrix_multiply(const int M, const int K, const int N, const MAT_DATATYPE *A, const MAT_DATATYPE *B, MAT_DATATYPE *C)
    {
        const int totalResultsPerBlock = BM * BN;
        const int totalResultsPerThread = TM * TN;
        const int totalThreadsPerBlock = totalResultsPerBlock / totalResultsPerThread;

        const int threadCol = threadIdx.x % (BN / TN);
        const int threadRow = threadIdx.x / (BN / TN);

        __shared__ MAT_DATATYPE As[BM * BK];
        __shared__ MAT_DATATYPE Bs[BK * BN];

        const int Row = blockIdx.y;
        const int Col = blockIdx.x;

        // move the pointer to the beginning 
        A += Row * BM * K;
        B += Col * BN;
        C += Row * BM * N + Col * BN;

        const int innerColA = threadIdx.x % BK;
        const int innerRowA = threadIdx.x / BK;
        const int strideA = totalThreadsPerBlock / BK;
        const int innerColB = threadIdx.x % BN;
        const int innerRowB = threadIdx.x / BN;
        const int strideB = totalThreadsPerBlock / BN;

        MAT_DATATYPE threadResults[TM * TN] = {0.0};
        MAT_DATATYPE resultsPerRow[TM] = {0.0};
        MAT_DATATYPE resultsPerCol[TN] = {0.0};

        for (int blockId = 0; blockId < K; blockId += BK)
        {
            for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA)
            {
                As[(innerRowA + loadOffset) * BK + innerColA] = A[(innerRowA + loadOffset) * K + innerColA];
            }
            for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB)
            {
                Bs[(innerRowB + loadOffset) * BN + innerColB] = B[(innerRowB + loadOffset) * N + innerColB];
            }
            __syncthreads();

            A += BK;
            B += BK * N;

            for (int dotIdx = 0; dotIdx < BK; dotIdx++)
            {
                for (int i = 0; i < TM; i++)
                {
                    resultsPerRow[i] = As[(threadRow * TM + i) * BK + dotIdx]; 
                }
                for (int i = 0; i < TN; i++)
                {
                    resultsPerCol[i] = Bs[dotIdx * BN + threadCol * TN + i];
                }
                for (int resIdxM = 0; resIdxM < TM; ++resIdxM) 
                {
                    for (int resIdxN = 0; resIdxN < TN; ++resIdxN) 
                    {
                    threadResults[resIdxM * TN + resIdxN] += resultsPerRow[resIdxM] * resultsPerCol[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        for (int resIdxM = 0; resIdxM < TM; ++resIdxM) 
        {
            for (int resIdxN = 0; resIdxN < TN; ++resIdxN) 
            {
                C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
            }
        }
    }

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        throw std::invalid_argument("Invalid argument, should be: ./executable "
                                    "/path/to/matrix1 /path/to/matrix2\n");
    }

    const std::string matrix1_path = argv[1];
    const std::string matrix2_path = argv[2];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    const int M = matrix1.getRows();
    const int K = matrix1.getCols();
    const int N = matrix2.getCols();

    Matrix result = Matrix(M, N);
    const MAT_DATATYPE* mat1_data = matrix1.getData();
    const MAT_DATATYPE* mat2_data = matrix2.getData();
    MAT_DATATYPE* result_data = result.getData();

    MAT_DATATYPE* d_mat1_data;
    MAT_DATATYPE* d_mat2_data;
    MAT_DATATYPE* d_result_data;

    size_t mat1_data_size = M * K * sizeof(MAT_DATATYPE);
    size_t mat2_data_size = K * N * sizeof(MAT_DATATYPE);
    size_t result_data_size = M * N * sizeof(MAT_DATATYPE);

    cudaMalloc((void**)&d_mat1_data, mat1_data_size);
    cudaMalloc((void**)&d_mat2_data, mat2_data_size);
    cudaMalloc((void**)&d_result_data, result_data_size);

    cudaMemcpy(d_mat1_data, mat1_data, mat1_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2_data, mat2_data, mat2_data_size, cudaMemcpyHostToDevice);

    const int BM = 128;
    const int BK = 16;
    const int BN = 128;
    const int TM = 8;
    const int TN = 8;
    const int threadPerBlock = (BM * BN) / (TM * TN);

    dim3 blockDim(threadPerBlock);
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    // Launch CUDA kernel
    matrix_multiply<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, K, N, d_mat1_data, d_mat2_data, d_result_data);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);

    cudaMemcpy(result_data, d_result_data, result_data_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_mat1_data);
    cudaFree(d_mat2_data);
    cudaFree(d_result_data);

    Matrix ground_truth = Matrix::getResultMatrix(matrix1_path, matrix2_path);
    std::cout << "Verification: "
              << ((Matrix::isIdentical(result, ground_truth)) ? "Passed"
                                                              : "Failed")
              << std::endl;
    std::cout << "Execution Time: " << gpuDuration << " milliseconds"
              << std::endl;
    return 0;
}