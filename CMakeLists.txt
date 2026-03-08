cmake_minimum_required(VERSION 3.20)

project(HighPerformanceBVHBuilderInCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(libs)

# ---- Sources / Headers ----
set(HEADERS
        src/kernels/kernels.h
        src/kernels/structs/aabb.h
        src/kernels/structs/bvh_node.h
        src/kernels/structs/camera.h
        src/kernels/structs/morton_code.h
        src/io/camera_reader.h
        src/io/scene_reader.h
        src/utils/cuda_utils.h
        src/utils/exceptions.h
        src/debug/debug_geometry.h
        src/debug/debug_bvh.h
        src/utils/utils.h
        src/utils/device_wrappers.h
        src/experiments/my_cpu_lbvh.h
        src/experiments/my_gpu_lbvh.h
        src/experiments/common.h
        src/experiments/kitten_lbvh.h
)

set(SOURCES
        src/io/camera_reader.cpp
        src/io/scene_reader.cpp
        src/utils/cuda_utils.cpp
        src/experiments/my_cpu_lbvh.cpp
        src/experiments/my_gpu_lbvh.cpp
        src/experiments/kitten_lbvh.cpp
)

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
        src/kernels/kitten_utils/compute_faces_aabb.cu
        src/kernels/kitten_utils/convert_to_bvh_nodes.cu
        src/kernels/toru_ninja_utils/build_bvh.cu
        #        src/kernels/toru_ninja_utils/convert_to_bvh_nodes.cu
)

set(LIBRARIES
        libclew
        libimages
        libjson
        libbase
        glm::glm
        KittenGpuLBVH
        KittenEngine
        ToruNinjaLBVH
)

# ---- Packages ----
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP) # optional
# find_package(Thrust REQUIRED CONFIG)
find_package(glm CONFIG REQUIRED)

# ---- Core library ----
add_library(${PROJECT_NAME}_core STATIC
        ${SOURCES}
        ${CUDA_SOURCES}
        ${HEADERS}   # опционально, чисто для IDE
)

# Архитектура GPU (compute capability). Можно списком: 75;86;89 и т.д.
set_property(TARGET ${PROJECT_NAME}_core PROPERTY CUDA_ARCHITECTURES 75)

# CUDA-only опции компилятора
target_compile_options(${PROJECT_NAME}_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --extended-lambda>
)

target_compile_definitions(${PROJECT_NAME}_core PRIVATE
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
)

target_link_libraries(${PROJECT_NAME}_core
        PUBLIC
        ${LIBRARIES}
        CUDA::cudart
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}_core PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(WARNING "OpenMP not found!")
endif ()

# Если нужно, чтобы #include "..." искались от корня src:
# target_include_directories(${PROJECT_NAME}_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# target_include_directories(${PROJECT_NAME}_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ---- Executable ----
add_executable(run_experiments src/main.cpp)
target_link_libraries(run_experiments PRIVATE ${PROJECT_NAME}_core)
