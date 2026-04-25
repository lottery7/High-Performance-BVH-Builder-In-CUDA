#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>

#include "kernels/structs/bvh_node.h"
#include "utils/utils.h"

namespace
{
  cub::CachingDeviceAllocator g_allocator(true);

  struct AABBUnion {
    __host__ __device__ AABB operator()(const AABB& a, const AABB& b) const { return AABB::union_of(a, b); }
  };
}  // namespace

namespace cuda
{
  void compute_scene_aabb(cudaStream_t stream, const AABB* __restrict__ d_aabbs, int count, AABB* __restrict__ d_scene_aabb)
  {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_aabbs, d_scene_aabb, count, AABBUnion{}, AABB::neutral(), stream));

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    CubDebugExit(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_aabbs, d_scene_aabb, count, AABBUnion{}, AABB::neutral(), stream));

    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  }

}  // namespace cuda
