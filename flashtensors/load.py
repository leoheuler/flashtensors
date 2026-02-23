import os
import json
from typing import Dict, Union

import torch

from .flash_state import FlashState
from .loaders import get_loader
from .loaders._common import _resolve_device


def load_dict(
    model_path: Union[str, os.PathLike],
    device_map: dict,
    num_workers: int = 4,
    chunk_size: int = 256 * 1024 * 1024,   # 256 MB — matches SLLM default
) -> Dict[str, torch.Tensor]:
    """Load tensors from a .flashtensors checkpoint.

    Uses the ServerlessLLM-style multi-reader pipeline when all GPU tensors
    target the same device: disk -> pinned memory -> GPU DMA overlap across chunks.

    Falls back to a parallel per-tensor approach for mixed-device maps or
    CPU-only loads.

    Args:
        model_path:  Directory containing tensor.flashtensors and tensor_index.json.
        device_map:  Maps tensor names (or "" for default) to devices.
                     e.g. {"": 0}      — all tensors -> GPU 0
                          {"": "cpu"}  — all tensors -> CPU
                          {"a": 0, "b": "cpu"} — per-tensor
        num_workers: Worker threads for the parallel fallback path.
        chunk_size:  Pinned-buffer chunk size (bytes) for the pipeline path.

    Returns:
        Dictionary mapping tensor names to PyTorch tensors on the requested devices.
    """
    index_path = os.path.join(model_path, "tensor_index.json")
    data_path = os.path.join(model_path, "tensor.flashtensors")

    with open(index_path, "r") as f:
        flash_state = FlashState(**json.load(f))

    file_size = os.path.getsize(data_path)
    devices = {_resolve_device(ref.name, device_map) for ref in flash_state.layout}

    loader = get_loader(devices)
    return loader.load(data_path, file_size, flash_state.layout, device_map, num_workers, chunk_size)
