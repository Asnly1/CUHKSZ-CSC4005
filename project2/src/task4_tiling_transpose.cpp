//
// Created by Liu Yuxuan on 2025/10/12.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Tiling MatMul with Transposition
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

/**
 * Matmul with Matrix Transposition, Serving as the utility function for blocks
 */
void matrix_multiply_transpose(const Matrix& matrix1, const Matrix& matrix2,
                               MAT_DATATYPE* result_data)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    const MAT_DATATYPE* const mat1_data = matrix1.getDataConst();
    Matrix T = Matrix::getTranspose(matrix2);
    const MAT_DATATYPE* const T_data = T.getDataConst();
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t k = 0; k < K; ++k)
            {
                result_data[i * N + j] += mat1_data[i * K + k] * T_data[j * K + k];
            }
        }
    }
}

/**
 * Tiled Matmul with Tiling
 * @param block_size: 32, 64, 128, etc
 * @param matrix1
 * @param matrix2
 * @return result matrix for verification
 */
Matrix matrix_multiply_tiling(const Matrix& matrix1, const Matrix& matrix2,
                              size_t block_size = 64)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    Matrix result(M, N);
    Matrix block_ik(block_size, block_size);
    Matrix block_kj(block_size, block_size);
    Matrix result_block_ij(block_size, block_size);
    MAT_DATATYPE * __restrict__ result_block_ij_data = result_block_ij.getData();
    MAT_DATATYPE* block_ik_data = block_ik.getData();
    MAT_DATATYPE* block_kj_data = block_kj.getData();
    for (size_t i = 0; i < M; i += block_size)
    {
        for (size_t j = 0; j < N; j += block_size)
        {
            for (size_t k = 0; k < K; k += block_size)
            {
                matrix1.getBlock(block_ik_data, i, k, block_size);
                matrix2.getBlock(block_kj_data, k, j, block_size);
                matrix_multiply_transpose(block_ik, block_kj, result_block_ij_data);
                result.setBlock(result_block_ij_data, i, j, block_size);
            }
        }
    }

    return result;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable block_size"
            "/path/to/matrix1 /path/to/matrix2\n");
    }

    const size_t block_size = static_cast<size_t>(std::atoi(argv[1]));
    const std::string matrix1_path = argv[2];
    const std::string matrix2_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);
    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_tiling(matrix1, matrix2, block_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    Matrix ground_truth = Matrix::getResultMatrix(matrix1_path, matrix2_path);
    std::cout << "Verification: "
              << ((Matrix::isIdentical(result, ground_truth)) ? "Passed"
                                                              : "Failed")
              << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;
    return 0;
}
