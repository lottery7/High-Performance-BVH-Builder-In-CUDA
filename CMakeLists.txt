cmake_minimum_required(VERSION 3.20)

project(HighPerformanceBVHBuilderInCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми директивами
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    # Это не ошибка т.к. если нет OpenMP - код будет работать, просто директивы распараллеливания ничего не будут делать
    message(WARNING "OpenMP not found!")
endif ()

add_subdirectory(libs)

set(HEADERS
        src/kernels/kernels.h
        src/kernels/structs/aabb_gpu.h
        src/kernels/structs/bvh_node_gpu.h
        src/kernels/structs/camera_gpu.h
        src/kernels/structs/morton_code_gpu.h
        src/io/camera_reader.h
        src/io/scene_reader.h
        src/utils/cuda_utils.h
        src/utils/exceptions.h
        src/debug/debug_geometry.h
        src/debug/debug_bvh.h
        src/utils/utils.h
        src/utils/gpu_wrappers.h
        src/experiments/cpu_lbvh.h
        src/experiments/gpu_lbvh.h
        src/experiments/common.h
)

set(SOURCES
        src/io/camera_reader.cpp
        src/io/scene_reader.cpp
        src/utils/cuda_utils.cpp
        src/experiments/cpu_lbvh.cpp
        src/experiments/gpu_lbvh.cpp
)

set(LIBRARIES
        libclew
        libimages
        libjson
        libbase
)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED CONFIG)

# Компиляция CUDA происходит NVCC компилятором, все файлы с расширением .cu компилируются два раза:
# 1) один раз обычным компилятором - для HOST-side кода, компилируются только те функции которые не помечены специально что они для видеокарты (или помечены что они и для видеокарты и для HOST-side кода)
# 2) второй раз специальным CUDA-компилятором NVCC - для DEVICE-side кода, например функции помеченные специальным модификатором __global__ (кернелы) или __device__ (функции которые можно вызвать на видеокарте, т.е. из кернелов)
set(CUDA_SOURCES
        src/kernels/utils/camera_helpers.cu
        src/kernels/utils/geometry_helpers.cu
        src/kernels/utils/random_helpers.cu
        src/kernels/ray_tracing/ray_tracing_render_using_bvh.cu
        src/kernels/utils/fill.cu
        src/kernels/bvh_common/fill_indices.cu
        src/kernels/bvh_common/compute_mortons.cu
        src/kernels/lbvh/build_lbvh.cu
        src/kernels/lbvh/build_aabb.cu
        src/kernels/lbvh/build_aabb_leaves.cu
        src/kernels/utils/sort_by_key.cu
)
add_compile_definitions(
        CMAKE_CUDA_ARCHITECTURES=75
        CMAKE_CUDA_FLAGS=-lineinfo
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
)

add_library(${PROJECT_NAME}_core STATIC ${SOURCES} ${CUDA_SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}_core ${LIBRARIES})
target_include_directories(${PROJECT_NAME}_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(run_experiments src/main.cpp)
target_link_libraries(run_experiments ${PROJECT_NAME}_core)
target_include_directories(run_experiments PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
