#include "tensor_writer.h"

#include <iostream>

TensorWriter::TensorWriter(const std::string &filename) : filename_(filename) {}

TensorWriter::~TensorWriter() {}

uint64_t TensorWriter::writeRecord(const char *data, size_t size) {
  if (partition_idx_ == -1 || partition_size_ + size > kPartitionMaxSize) {
    // create a new partition
    partition_idx_++;
    partition_size_ = 0;
    std::string partition_filename =
        filename_ + "_" + std::to_string(partition_idx_);
    buffer_ = std::make_unique<AlignedBuffer>(partition_filename);
  }

  uint64_t start_offset = offset_;
  // make sure the data is 64-bit aligned
  size_t padding = (size % 8) ? (8 - size % 8) : 0;
  size_t written = buffer_->writeData(data, size);
  if (padding) {
    written += buffer_->writePadding(padding);
  }
  offset_ += written;
  partition_size_ += written;
  // std::cerr << "writeRecord: " << partition_idx_ << " " << partition_size_ <<
  // " " << kPartitionMaxSize << std::endl;

  return start_offset;
}
