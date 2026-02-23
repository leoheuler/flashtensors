import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

from ._base import BaseLoader
from ._common import (
    _libc,
    POSIX_FADV_SEQUENTIAL,
    _pread_all,
    _resolve_device,
    _numel,
)


def _load_tensor_pread(ref, fd, device_map, cupy=None):
    """Load a single tensor via pread. No mmap, no intermediate bytes objects."""
    device_type, device_id = _resolve_device(ref.name, device_map)
    is_bfloat16 = ref.dtype in ("bfloat16", "torch.bfloat16")
    np_dtype = np.dtype("uint16") if is_bfloat16 else ref.numpy_dtype()
    numel = _numel(ref)

    if device_type == "cuda":
        mem = cupy.cuda.alloc_pinned_memory(ref.size)
        _pread_all(fd, mem.ptr, ref.size, ref.offset)
        arr = np.frombuffer(mem, dtype=np_dtype, count=numel)

        with cupy.cuda.Device(device_id):
            gpu_arr = cupy.asarray(arr).reshape(ref.shape)
            tensor = torch.utils.dlpack.from_dlpack(gpu_arr)

        if is_bfloat16:
            tensor = tensor.view(torch.bfloat16)

    else:
        # Allocate torch tensor then pread directly into its storage
        torch_dtype = torch.uint16 if is_bfloat16 else ref.torch_dtype()
        tensor = torch.empty(numel, dtype=torch_dtype)
        _pread_all(fd, tensor.data_ptr(), ref.size, ref.offset)
        tensor = tensor.reshape(ref.shape)
        if is_bfloat16:
            tensor = tensor.view(torch.bfloat16)

    expected_stride = list(ref.stride)
    if list(tensor.stride()) != expected_stride:
        tensor = torch.as_strided(tensor, ref.shape, expected_stride)

    return ref.name, tensor


class CpuLoader(BaseLoader):
    def load(self, data_path, file_size, layout, device_map, num_workers, chunk_size):
        devices = {_resolve_device(ref.name, device_map) for ref in layout}
        needs_gpu = any(dt == "cuda" for dt, _ in devices)
        cupy = None
        if needs_gpu:
            import cupy as _cupy
            cupy = _cupy

        result = {}
        fd = os.open(data_path, os.O_RDONLY)
        try:
            _libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL)

            if num_workers == 1 or len(layout) == 1:
                for ref in layout:
                    name, tensor = _load_tensor_pread(ref, fd, device_map, cupy)
                    result[name] = tensor
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_load_tensor_pread, ref, fd, device_map, cupy): ref
                        for ref in layout
                    }
                    for future in as_completed(futures):
                        name, tensor = future.result()
                        result[name] = tensor
        finally:
            os.close(fd)

        return result
