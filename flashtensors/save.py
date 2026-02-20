import os
import json
from typing import Dict, Union

import torch
import numpy as np
from tqdm import tqdm

from .tensor_reference import TensorReference
from .flash_state import FlashState


def _tensor_to_numpy(tensor):
    """Convert a tensor to a contiguous numpy array for serialization.

    bfloat16 is viewed as uint16 since numpy has no bfloat16 support.
    Returns: numpy array whose .data memoryview can be written directly.
    """
    t = tensor.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.uint16)
    return t.numpy()


def save_dict(state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]):
    """Save a state_dict to flashtensors format.

    Creates two files at model_path:
      tensor.flashtensors  — contiguous raw tensor bytes
      tensor_index.json    — metadata index (offsets, shapes, strides, dtypes)

    Tensors sharing storage (weight tying) are deduplicated automatically.
    All tensors are made contiguous before writing so the on-disk layout
    matches the recorded shape and stride exactly.
    """
    os.makedirs(model_path, exist_ok=True)

    # Collect data pointers for deduplication (weight tying)
    data_ptrs = {
        name: param.untyped_storage().data_ptr()
        for name, param in state_dict.items()
    }

    # ── Write binary data ─────────────────────────────────────────────────
    tensor_path = os.path.join(model_path, "tensor.flashtensors")
    offsets = {}
    seen = {}  # data_ptr -> first tensor name that wrote it

    with open(tensor_path, "wb") as f:
        offset = 0
        for name in tqdm(state_dict, desc="Saving tensors"):
            ptr = data_ptrs[name]

            # Dedup: skip if this storage was already written (weight tying)
            if ptr in seen:
                offsets[name] = offsets[seen[ptr]]
                continue
            seen[ptr] = name

            arr = _tensor_to_numpy(state_dict[name])
            f.write(arr.data)           # memoryview — no extra copy
            offsets[name] = offset
            offset += arr.nbytes

    # ── Write metadata index ──────────────────────────────────────────────
    layout = []
    for name, param in state_dict.items():
        t = param.detach().contiguous()
        layout.append(
            TensorReference(
                name=name,
                offset=offsets[name],
                size=t.nelement() * t.element_size(),
                shape=list(t.shape),
                stride=list(t.stride()),   # contiguous strides
                dtype=str(param.dtype),
            )
        )

    flash_state = FlashState(layout=layout)
    index_path = os.path.join(model_path, "tensor_index.json")
    with open(index_path, "w") as f:
        json.dump(flash_state.model_dump(), f, indent=2)
