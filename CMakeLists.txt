cmake_minimum_required(VERSION 3.20)

project(GPGPUTasks)

set(CMAKE_CXX_STANDARD 17)

# GTest позволяет писать удобные unit-test-ы
find_package(GTest REQUIRED)
set(GTEST_LIBRARIES GTest::gtest)
set(GTEST_MAIN_LIBRARIES GTest::gtest_main)
set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

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
        src/kernels/shared_structs/aabb_gpu_shared.h
        src/kernels/shared_structs/bvh_node_gpu_shared.h
        src/kernels/shared_structs/camera_gpu_shared.h
        src/kernels/shared_structs/morton_code_gpu_shared.h
        src/io/camera_reader.h
        src/io/scene_reader.h
        src/utils/cuda_utils.h
        src/utils/exceptions.h
        src/debug/debug_geometry.h
        src/debug/debug_bvh.h
        src/utils/utils.h
)

set(SOURCES
        src/io/camera_reader.cpp
        src/io/scene_reader.cpp
        src/utils/cuda_utils.cpp
)

set(LIBRARIES
        libclew
        libimages
        libjson
        libbase
)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Компиляция CUDA происходит NVCC компилятором, все файлы с расширением .cu компилируются два раза:
# 1) один раз обычным компилятором - для HOST-side кода, компилируются только те функции которые не помечены специально что они для видеокарты (или помечены что они и для видеокарты и для HOST-side кода)
# 2) второй раз специальным CUDA-компилятором NVCC - для DEVICE-side кода, например функции помеченные специальным модификатором __global__ (кернелы) или __device__ (функции которые можно вызвать на видеокарте, т.е. из кернелов)
# Вероятно вам стоит выставить в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
set(CUDA_SOURCES
        src/kernels/cu/aplusb.cu
        src/kernels/cu/camera_helpers.cu
        src/kernels/cu/geometry_helpers.cu
        src/kernels/cu/random_helpers.cu
        src/kernels/cu/ray_tracing_render_brute_force.cu
        src/kernels/cu/ray_tracing_render_using_lbvh.cu
        src/kernels/cu/fill.cu
)
add_compile_definitions(
        CMAKE_CUDA_ARCHITECTURES=75
        CMAKE_CUDA_FLAGS=-lineinfo
)

add_library(${PROJECT_NAME}_core STATIC ${SOURCES} ${CUDA_SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}_core ${LIBRARIES})
target_include_directories(${PROJECT_NAME}_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(main_aplusb src/main_aplusb.cpp)
add_executable(main_linear_bvh src/main_linear_bvh.cpp)

target_link_libraries(main_aplusb ${PROJECT_NAME}_core)
target_link_libraries(main_linear_bvh ${PROJECT_NAME}_core)

target_include_directories(main_aplusb PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(main_linear_bvh PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
