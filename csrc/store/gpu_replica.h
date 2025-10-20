#include <condition_variable>
#include <future>
#include <unordered_map>

#include "types_and_defs.h"

class GpuReplica {
  std::condition_variable cv_;
  MemoryState state_ = MemoryState::UNINITIALIZED;

  std::unordered_map<int, std::shared_ptr<BatchQueue>> gpu_loading_queue_;
  std::unordered_map<int, void *> device_ptrs_;

  std::unordered_map<std::string, size_t> tensor_offsets_;

  void Clear();
};