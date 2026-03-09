#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

#include <thrust/detail/sequence.inl>

#include "../../utils/utils.h"

namespace cuda
{
  void fill_indices(cudaStream_t stream, unsigned int* d_indices, unsigned int n)
  {
    auto d_ptr = thrust::device_pointer_cast(d_indices);
    thrust::sequence(thrust::cuda::par_nosync.on(stream), d_ptr, d_ptr + n);
  }
}  // namespace cuda
