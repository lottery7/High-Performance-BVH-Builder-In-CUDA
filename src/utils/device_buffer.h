#pragma once
#include <cuda_runtime_api.h>

#include <cstddef>

#include "utils.h"

template <class T>
class DeviceBuffer
{
 public:
  DeviceBuffer() = default;

  DeviceBuffer(std::size_t count, cudaStream_t stream) { allocate(count, stream); }

  ~DeviceBuffer() { cudaFreeAsync(ptr_, stream_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_), stream_(other.stream_)
  {
    other.ptr_ = nullptr;
    other.count_ = 0;
    other.stream_ = nullptr;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
  {
    if (this != &other) {
      reset();

      ptr_ = other.ptr_;
      count_ = other.count_;
      stream_ = other.stream_;

      other.ptr_ = nullptr;
      other.count_ = 0;
      other.stream_ = nullptr;
    }
    return *this;
  }

  void allocate(std::size_t count, cudaStream_t stream)
  {
    reset();

    if (count == 0) return;

    stream_ = stream;
    count_ = count;
    CUDA_SAFE_CALL(cudaMallocAsync(&ptr_, sizeof(T) * count_, stream_));
  }

  void reset()
  {
    if (ptr_ != nullptr) {
      CUDA_SAFE_CALL(cudaFreeAsync(ptr_, stream_));
      ptr_ = nullptr;
      count_ = 0;
      stream_ = nullptr;
    }
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }

  std::size_t size() const { return count_; }
  std::size_t size_bytes() const { return sizeof(T) * count_; }

  cudaStream_t stream() const { return stream_; }

  explicit operator bool() const { return ptr_ != nullptr; }

  operator T*() { return ptr_; }
  operator const T*() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
  cudaStream_t stream_ = nullptr;
};
