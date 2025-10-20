#include "pinned_memory.h"

#include <glog/logging.h>

PinnedMemory::~PinnedMemory() {
  LOG(INFO) << "Deallocating " << buffers_.size() << " memory chunks";
  int ret = mempool_->Deallocate(buffers_);
  if (ret != 0) {
    LOG(ERROR) << "Error deallocating CPU memory";
  }
}

int PinnedMemory::Allocate(size_t size,
                           std::shared_ptr<PinnedMemoryPool> mempool) {
  if (buffers_.size() > 0) {
    LOG(ERROR) << "Memory already allocated";
    return 1;
  }

  mempool_ = mempool;
  return mempool_->Allocate(size, buffers_);
}

std::vector<char *> &PinnedMemory::get() { return buffers_; }
