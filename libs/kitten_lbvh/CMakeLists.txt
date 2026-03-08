cmake_minimum_required(VERSION 3.20)

add_library(KittenGpuLBVH STATIC KittenGpuLBVH/lbvh.cu)

target_include_directories(KittenGpuLBVH PUBLIC ${CMAKE_CURRENT_LIST_DIR})

find_package(glm CONFIG REQUIRED)
target_link_libraries(KittenGpuLBVH PUBLIC glm::glm)

target_compile_options(KittenGpuLBVH PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --extended-lambda>
)
