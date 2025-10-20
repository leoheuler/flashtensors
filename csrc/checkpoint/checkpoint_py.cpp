#include <torch/extension.h>

#include "checkpoint.h"

namespace py = pybind11;

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("save_tensors", &SaveTensors, "Save a state dict")
      .def("restore_tensors", &RestoreTensors, "Restore a state dict")
      .def("allocate_cuda_memory", &AllocateCudaMemory, "Allocate cuda memory")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, void *> &memory_ptrs) {
            std::unordered_map<int, std::string> memory_handles =
                GetCudaMemoryHandles(memory_ptrs);

            std::unordered_map<int, py::bytes> py_memory_handles;
            for (const auto &kv : memory_handles) {
              py_memory_handles[kv.first] = py::bytes(kv.second);
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, std::vector<void *>> &memory_ptrs) {
            auto memory_handles = GetCudaMemoryHandles(memory_ptrs);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto &kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto &handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, std::vector<uint64_t>>
                 &memory_ptrs) {
            std::unordered_map<int, std::vector<void *>> memory_ptrs_void;
            for (const auto &kv : memory_ptrs) {
              std::vector<void *> ptrs;
              for (const auto &ptr : kv.second) {
                ptrs.push_back(reinterpret_cast<void *>(ptr));
              }
              memory_ptrs_void[kv.first] = ptrs;
            }
            auto memory_handles = GetCudaMemoryHandles(memory_ptrs_void);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto &kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto &handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def("get_device_uuid_map", &GetDeviceUuidMap, "Get device uuid map");
}