#pragma once

#include <string>

const size_t kAlignment = 4096;      // 4k
const size_t kBufferSize = 1L << 30; // 1GB

// A write buffer that writes to a file (4k aligned).
class AlignedBuffer {
public:
  explicit AlignedBuffer(const std::string &filename);
  ~AlignedBuffer();

  size_t writeData(const void *data, size_t size);
  size_t writePadding(size_t padding_size);

private:
  int fd_;
  size_t buf_size_;
  size_t buf_pos_;
  size_t file_offset_;
  void *buffer_;
};