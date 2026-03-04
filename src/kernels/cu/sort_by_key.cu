#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "../../utils/cuda_utils.h"
#include "../../utils/utils.h"

namespace cuda
{
  template <typename K, typename V>
  void sort_by_key(const cudaStream_t& stream, K* d_keys, V* d_vals, size_t n)
  {
    thrust::device_ptr<unsigned int> keys(d_keys);
    thrust::device_ptr<unsigned int> vals(d_vals);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + n, vals);
  }

  template void sort_by_key(const cudaStream_t& stream, unsigned int* d_keys, unsigned int* d_vals, size_t n);
}  // namespace cuda
