#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include "utils/utils.h"

namespace
{
  cub::CachingDeviceAllocator g_allocator(true);
}

namespace cuda
{
  template <typename KeyT, typename ValueT>
  void sort_pairs(
      cudaStream_t stream,
      const KeyT* d_keys_in,
      KeyT* d_keys_out,
      const ValueT* d_values_in,
      ValueT* d_values_out,
      int count,
      int begin_bit,
      int end_bit)
  {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CubDebugExit(
        cub::DeviceRadixSort::
            SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, count, begin_bit, end_bit, stream));

    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    CubDebugExit(
        cub::DeviceRadixSort::
            SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, count, begin_bit, end_bit, stream));

    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  }

  template void sort_pairs<unsigned int, unsigned int>(
      cudaStream_t stream,
      const unsigned int* d_keys_in,
      unsigned int* d_keys_out,
      const unsigned int* d_values_in,
      unsigned int* d_values_out,
      int count,
      int begin_bit,
      int end_bit);
}  // namespace cuda
