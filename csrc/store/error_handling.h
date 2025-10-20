#pragma once

#include <cuda_runtime.h>
#include <errno.h>
#include <glog/logging.h>
#include <string.h>

#define CUDA_CHECK(x, msg)                                                     \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      LOG(ERROR) << msg << " " << cudaGetErrorString(cudaGetLastError())       \
                 << std::endl;                                                 \
      return -1;                                                               \
    }                                                                          \
  }

#define CHECK_POSIX(x, msg)                                                    \
  {                                                                            \
    if ((x) < 0) {                                                             \
      LOG(ERROR) << msg << " errno: " << errno << "msg: " << strerror(errno);  \
      return -1;                                                               \
    }                                                                          \
  }

#define WAIT_FUTURES(futures, msg)                                             \
  {                                                                            \
    for (auto &future : futures) {                                             \
      int ret = future.get() if (ret != 0) {                                   \
        LOG(ERROR) << msg;                                                     \
        return ret;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define CHECK_RETURN(x, msg)                                                   \
  {                                                                            \
    if ((x) != 0) {                                                            \
      LOG(ERROR) << msg;                                                       \
      return -1;                                                               \
    }                                                                          \
  }
