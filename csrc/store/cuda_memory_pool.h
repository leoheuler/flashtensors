#pragma once

#include <cuda_runtime.h>

#include <mutex>
#include <vector>

class CudaMemoryPool {
public:
  CudaMemoryPool(int device_count, size_t size_per_device);
  CudaMemoryPool(const CudaMemoryPool &) = delete;
  CudaMemoryPool &operator=(const CudaMemoryPool &) = delete;
  ~CudaMemoryPool();

  int Allocate(size_t size, int device_id, void *&ptr,
               cudaIpcMemHandle_t &handle);
  int Deallocate(int device_id, void *ptr);

private:
  std::mutex mutex_;
  int device_count_;
  size_t size_per_device_;
  std::vector<void *> pool_;
  std::vector<cudaIpcMemHandle_t> handles_;
  std::vector<bool> free_list_;
};
