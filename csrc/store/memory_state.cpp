#include "memory_state.h"

#include <iostream>

std::ostream &operator<<(std::ostream &os, const MemoryState state) {
  return os << [state]() -> const char * {
#define PROCESS_STATE(p)                                                       \
  case (p):                                                                    \
    return #p;
    switch (state) {
      PROCESS_STATE(UNINITIALIZED);
      PROCESS_STATE(UNALLOCATED);
      PROCESS_STATE(ALLOCATED);
      PROCESS_STATE(LOADING);
      PROCESS_STATE(LOADED);
      PROCESS_STATE(CANCELLED);
      PROCESS_STATE(INTERRUPTED);
    default:
      return "UNKNOWN";
    }
#undef PROCESS_STATE
  }();
}