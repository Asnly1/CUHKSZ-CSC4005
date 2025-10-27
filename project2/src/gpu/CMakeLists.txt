set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/compilers/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2")
enable_language(CUDA)  # 启用CUDA支持
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif ()

set(PARENT_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# 仅在 CUDA 目标上单独添加 NVHPC 的特性或 flags，不改全局编译器
add_executable(cuda
    cuda.cu
    ${PARENT_SRC_DIR}/matrix.cpp
)

# matrix.cpp 用系统默认 C++ 编译器，cuda.cu 用 nvcc
set_target_properties(cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
)

target_link_libraries(cuda PRIVATE ${CUDA_CUDART_LIBRARY})

