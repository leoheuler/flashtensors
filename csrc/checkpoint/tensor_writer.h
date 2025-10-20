#pragma once

#include <memory>
#include <string>
#include <vector>

#include "aligned_buffer.h"

const size_t kPartitionMaxSize = 10L << 30; // 10GB

// A tensor writer that writes the raw tensor data to a file in raw binary.
class TensorWriter final {
public:
  explicit TensorWriter(const std::string &filename);
  ~TensorWriter();

  uint64_t writeRecord(const char *data, size_t size);

private:
  size_t offset_ = 0;
  int partition_idx_ = -1;
  size_t partition_size_ = 0;
  std::string filename_;
  std::unique_ptr<AlignedBuffer> buffer_;
};