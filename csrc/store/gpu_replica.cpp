#include "gpu_replica.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

void GpuReplica::Clear() {
  for (auto &[device_id, device_ptr] : device_ptrs_) {
    cudaSetDevice(device_id);
    cudaError_t err = cudaIpcCloseMemHandle(device_ptr);
    if (err != cudaSuccess) {
      LOG(ERROR) << "Failed to close memory handle for device " << device_id
                 << " error: " << cudaGetErrorString(err);
    }
  }
  gpu_loading_queue_.clear();
  tensor_offsets_.clear();
  state_ = MemoryState::INTERRUPTED;
  cv_.notify_all();
}