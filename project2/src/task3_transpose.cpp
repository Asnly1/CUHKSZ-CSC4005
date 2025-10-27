//
// Created by Liu Yuxuan on 2025/10/12.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Matrix Multiplication by Transposition
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

/**
 * Matmul with Matrix Transposition
 */
Matrix matrix_multiply_transpose(const Matrix& matrix1, const Matrix& matrix2)
{
    if (matrix1.getCols() != matrix2.getRows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    Matrix result(M, N);
    MAT_DATATYPE* __restrict__ result_data = result.getData();
    const MAT_DATATYPE* const mat1_data = matrix1.getDataConst();
    Matrix T = Matrix::getTranspose(matrix2);
    const MAT_DATATYPE* const T_data = T.getDataConst();
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            MAT_DATATYPE local_sum = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                local_sum += mat1_data[i * K + k] * T_data[j * K + k];
            }
            result_data[i * N + j] = local_sum;
        }
    }

    return result;
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

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_transpose(matrix1, matrix2);

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
