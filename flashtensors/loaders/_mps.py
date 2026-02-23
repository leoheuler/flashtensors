import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np

from ._base import BaseLoader
from ._common import (
    _pread_all,
    _resolve_device,
    _numel,
)


def _load_tensor_pread_mps(ref, fd, device_map):
    """Load a single tensor via pread into CPU, then move to MPS."""
    device_type, _ = _resolve_device(ref.name, device_map)
    is_bfloat16 = ref.dtype in ("bfloat16", "torch.bfloat16")
    torch_dtype = torch.uint16 if is_bfloat16 else ref.torch_dtype()
    numel = _numel(ref)

    # Read directly into a CPU tensor
    tensor = torch.empty(numel, dtype=torch_dtype)
    _pread_all(fd, tensor.data_ptr(), ref.size, ref.offset)
    tensor = tensor.reshape(ref.shape)
    if is_bfloat16:
        tensor = tensor.view(torch.bfloat16)

    expected_stride = list(ref.stride)
    if list(tensor.stride()) != expected_stride:
        tensor = torch.as_strided(tensor, ref.shape, expected_stride)

    if device_type == "mps":
        tensor = tensor.to("mps")

    return ref.name, tensor


class MpsLoader(BaseLoader):
    def load(self, data_path, file_size, layout, device_map, num_workers, chunk_size):
        result = {}
        fd = os.open(data_path, os.O_RDONLY)
        try:
            if num_workers == 1 or len(layout) == 1:
                for ref in layout:
                    name, tensor = _load_tensor_pread_mps(ref, fd, device_map)
                    result[name] = tensor
            else:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_load_tensor_pread_mps, ref, fd, device_map): ref
                        for ref in layout
                    }
                    for future in as_completed(futures):
                        name, tensor = future.result()
                        result[name] = tensor
        finally:
            os.close(fd)

        return result
