#pragma once

#include <memory>
#include <vector>

#include "pinned_memory_pool.h"

class PinnedMemory {
public:
  PinnedMemory() = default;
  ~PinnedMemory();

  // Disable copying and moving
  PinnedMemory(const PinnedMemory &) = delete;
  PinnedMemory &operator=(const PinnedMemory &) = delete;
  PinnedMemory(PinnedMemory &&) = delete;
  PinnedMemory &operator=(PinnedMemory &&) = delete;

  int Allocate(size_t size, std::shared_ptr<PinnedMemoryPool> mempool);
  std::vector<char *> &get();
  size_t num_chunks() const { return buffers_.size(); }
  size_t chunk_size() const { return mempool_->chunk_size(); }

private:
  std::vector<char *> buffers_;
  std::shared_ptr<PinnedMemoryPool> mempool_;
};
