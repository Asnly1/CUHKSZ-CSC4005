## CUDA
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2")

enable_language(CUDA)  # 启用CUDA支持

## OpenACC
find_package(OpenACC REQUIRED)
if (OpenACC_CXX_FOUND)
    message("OpenACC FOUND, Version: " ${OpenACC_CXX_VERSION})
endif ()

add_executable(radixsort_sequential
        sequential.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(radixsort_sequential PRIVATE -O2)

set(CMAKE_CXX_COMPILER pgc++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc")

## OpenACC Radixsort
add_executable(radixsort
        radixsort.cpp
        ../utils.cpp ../utils.hpp)
target_compile_options(radixsort PRIVATE -O2)
target_link_libraries(radixsort cudart)