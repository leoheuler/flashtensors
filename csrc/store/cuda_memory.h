#pragma once

#include <cuda_runtime.h>

// #include "cuda_memory_pool.h"

class CudaMemory {
public:
  CudaMemory();
  ~CudaMemory();

  // Disable copying and moving
  CudaMemory(const CudaMemory &) = delete;
  CudaMemory &operator=(const CudaMemory &) = delete;
  CudaMemory(CudaMemory &&) = delete;
  CudaMemory &operator=(CudaMemory &&) = delete;

  int Allocate(size_t size, int device);
  void *get() const;
  cudaIpcMemHandle_t getHandle() const;

private:
  void *data_;
  cudaIpcMemHandle_t handle_;
  size_t size_;
  int device_;
};
