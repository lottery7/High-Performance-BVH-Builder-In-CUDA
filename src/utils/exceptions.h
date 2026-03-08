#pragma once

#include <stdexcept>
#include <string>

namespace cuda
{
  struct CudaException : std::runtime_error {
    explicit CudaException(const std::string& msg) noexcept : std::runtime_error(msg) {}
  };

  struct CudaBadAlloc : std::runtime_error {
    explicit CudaBadAlloc(const std::string& msg) noexcept : std::runtime_error(msg) {}
  };
}  // namespace cuda
