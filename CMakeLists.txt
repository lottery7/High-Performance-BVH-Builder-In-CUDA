cmake_minimum_required(VERSION 3.20)

project(HighPerformanceBVHBuilderInCUDA LANGUAGES CXX CUDA)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(libs)

# ---- Sources / Headers ----
set(SOURCES
        src/io/camera_reader.cpp
        src/io/scene_reader.cpp
)

set(CUDA_SOURCES
        src/experiments/nexus_bvh2.cu
        src/experiments/nexus_bvh8.cu
        src/experiments/lbvh.cu
        src/experiments/hploc_bvh2.cu
        src/experiments/hploc_wide.cu
        src/kernels/helpers/camera_helpers.cuh
        src/kernels/helpers/geometry_helpers.cu
        src/kernels/helpers/random_helpers.cu
        src/kernels/ray_tracing/rt_lbvh.cu
        src/kernels/ray_tracing/rt_hploc.cu
        src/kernels/ray_tracing/rt_hploc_wide.cu
        src/kernels/ray_tracing/rt_hploc_bvh8.cu
        src/kernels/helpers/fill.cu
        src/kernels/helpers/fill_indices.cu
        src/kernels/lbvh/lbvh.cu
        src/kernels/hploc/hploc_bvh2.cu
        src/kernels/hploc/hploc_bvh8.cu
        src/kernels/hploc/hploc_wide.cu
        src/kernels/nexus_bvh/nexus_bvh.cu
        src/kernels/nexus_bvh/nexus_bvh_wide.cu
        src/kernels/helpers/compute_morton_codes.cu
        src/kernels/helpers/compute_scene_aabb.cu
        src/kernels/helpers/sort_pairs.cu
)

set(LIBRARIES
        libclew
        libimages
        libjson
        libbase
        glm::glm
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
        $<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false -lineinfo --extended-lambda>
)

target_compile_definitions(${PROJECT_NAME}_core PRIVATE
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
)

target_link_libraries(${PROJECT_NAME}_core
        PUBLIC
        ${LIBRARIES}
        CUDA::cudart
)

target_include_directories(${PROJECT_NAME}_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

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
