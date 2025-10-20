#pragma once

#include <mutex>
#include <unordered_set>
#include <vector>

class PinnedMemoryPool {
public:
  PinnedMemoryPool(size_t total_size, size_t chunk_size);
  ~PinnedMemoryPool();

  int Allocate(size_t size, std::vector<char *> &buffers);
  int Deallocate(std::vector<char *> &buffers);
  size_t chunk_size() const { return chunk_size_; }

  // Forbid copy and assignment
  PinnedMemoryPool(const PinnedMemoryPool &) = delete;
  PinnedMemoryPool &operator=(const PinnedMemoryPool &) = delete;

private:
  std::mutex mutex_;
  std::unordered_set<char *> free_list_;
  std::unordered_set<char *> pool_;
  size_t chunk_size_;
};