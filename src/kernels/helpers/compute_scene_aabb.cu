#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>

#include "kernels/structs/bvh_node.h"
#include "utils/utils.h"

namespace
{
  struct AABBUnion {
    __host__ __device__ __forceinline__ AABB operator()(const AABB& a, const AABB& b) const { return AABB::union_of(a, b); }
  };
}  // namespace

namespace cuda
{
  void compute_scene_aabb(cudaStream_t stream, const AABB* __restrict__ d_aabbs, int count, AABB* __restrict__ d_scene_aabb)
  {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CUDA_SAFE_CALL(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_aabbs, d_scene_aabb, count, AABBUnion{}, AABB::neutral(), stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    CUDA_SAFE_CALL(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_aabbs, d_scene_aabb, count, AABBUnion{}, AABB::neutral(), stream));
    CUDA_SAFE_CALL(cudaFreeAsync(d_temp_storage, stream));
  }

}  // namespace cuda
