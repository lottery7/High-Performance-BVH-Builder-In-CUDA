#pragma once

#include <stdexcept>
#include <string>

namespace cuda
{

  class cuda_exception : public std::runtime_error
  {
   public:
    cuda_exception(const std::string& msg) noexcept : std::runtime_error(msg) {}
  };

  class cuda_bad_alloc : public std::runtime_error
  {
   public:
    cuda_bad_alloc(const std::string& msg) noexcept : std::runtime_error(msg) {}
  };

}  // namespace cuda
