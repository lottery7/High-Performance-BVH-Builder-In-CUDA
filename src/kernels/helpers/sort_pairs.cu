#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>

#include "utils/utils.h"

namespace cuda
{
  template <typename KeyT, typename ValueT>
  void sort_pairs(
      cudaStream_t stream,
      const KeyT* __restrict__ d_keys_in,
      KeyT* __restrict__ d_keys_out,
      const ValueT* __restrict__ d_values_in,
      ValueT* __restrict__ d_values_out,
      int count,
      int begin_bit,
      int end_bit)
  {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    CUDA_SAFE_CALL(
        cub::DeviceRadixSort::
            SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, count, begin_bit, end_bit, stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
    CUDA_SAFE_CALL(
        cub::DeviceRadixSort::
            SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, count, begin_bit, end_bit, stream));
    CUDA_SAFE_CALL(cudaFreeAsync(d_temp_storage, stream));
  }

  template void sort_pairs<unsigned int, unsigned int>(
      cudaStream_t stream,
      const unsigned int* __restrict__ d_keys_in,
      unsigned int* __restrict__ d_keys_out,
      const unsigned int* __restrict__ d_values_in,
      unsigned int* __restrict__ d_values_out,
      int count,
      int begin_bit,
      int end_bit);
}  // namespace cuda
