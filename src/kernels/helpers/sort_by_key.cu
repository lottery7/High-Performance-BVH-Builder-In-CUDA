#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "../../utils/cuda_utils.h"
#include "../../utils/utils.h"

namespace cuda
{
  template <typename K, typename V>
  void sort_by_key(cudaStream_t stream, K* d_keys, V* d_vals, size_t n)
  {
    auto d_ptr_keys = thrust::device_pointer_cast(d_keys);
    auto d_ptr_vals = thrust::device_pointer_cast(d_vals);
    thrust::sort_by_key(thrust::cuda::par_nosync.on(stream), d_ptr_keys, d_ptr_keys + n, d_ptr_vals);
  }

  template void sort_by_key(cudaStream_t stream, unsigned int* d_keys, unsigned int* d_vals, size_t n);
}  // namespace cuda
