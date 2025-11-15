# Implementation derived from: https://github.com/ServerlessLLM/ServerlessLLM/blob/main/sllm_store/sllm_store/torch.py
# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import json
import os
import time
import uuid
from typing import Dict, Optional, Union
import torch

from ._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
    save_tensors,
)
from .storage_client import StorageClient
from .utils import (
    init_logger,
    calculate_device_memory,
    calculate_tensor_device_offsets,
    _expand_tensor_name,
)

logger = init_logger(__name__)


def _get_uuid():
    return str(uuid.uuid4())


def save_dict(state_dict: Dict[str, torch.Tensor], model_path: Union[str, os.PathLike]):
    tensor_names = list(state_dict.keys())
    tensor_data_index = {}
    for name, param in state_dict.items():
        param_storage = param.untyped_storage()
        data_ptr = param_storage.data_ptr()
        size = param_storage.size()
        tensor_data_index[name] = (data_ptr, size)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    tensor_offsets = save_tensors(tensor_names, tensor_data_index, model_path)

    tensor_index = {}
    for name, param in state_dict.items():
        # name: offset, size
        tensor_index[name] = (
            tensor_offsets[name],
            tensor_data_index[name][1],
            tuple(param.shape),
            tuple(param.stride()),
            str(param.dtype),
        )

    with open(os.path.join(model_path, "tensor_index.json"), "w") as f:
        json.dump(tensor_index, f)


def load_dict(
    model_path: Union[str, os.PathLike],
    device_map: Dict[str, int],
    storage_path: Optional[str] = None,
):
    logger.info(
        f"🔄 Starting load_dict for model_path='{model_path}', device_map={device_map}"
    )
    logger.debug(f"Storage path: {storage_path}")

    # Check for None values early
    if model_path is None:
        logger.error("❌ model_path is None")
        raise ValueError("model_path cannot be None")
    if device_map is None:
        logger.error("❌ device_map is None")
        raise ValueError("device_map cannot be None")

    replica_uuid, state_dict = load_dict_non_blocking(
        model_path, device_map, storage_path
    )

    storage = StorageClient()
    success = storage.confirm_model_loaded(model_path, replica_uuid)
    if not success:
        logger.error(f"❌ Failed to confirm model {model_path} loaded")
        raise ValueError(f"Failed to confirm model {model_path} loaded")

    logger.info(f"✅ load_dict completed successfully with {len(state_dict)} tensors")
    return state_dict


def load_dict_non_blocking(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: Dict[str, int],
    storage_path: Optional[str] = None,
):
    logger.debug(
        f"🔄 Starting load_dict_non_blocking: model_path='{model_path}', device_map={device_map}, storage_path='{storage_path}'"
    )

    # Check for None values early
    if model_path is None:
        logger.error("❌ model_path is None in load_dict_non_blocking")
        raise ValueError("model_path cannot be None")
    if device_map is None:
        logger.error("❌ device_map is None in load_dict_non_blocking")
        raise ValueError("device_map cannot be None")

    storage = StorageClient()
    logger.debug(f"Loading model {model_path} into CPU")
    ret = storage.load_into_cpu(model_path)
    if not ret:
        logger.error(f"❌ Failed to load model {model_path} into CPU")
        raise ValueError(f"Failed to load model {model_path} into CPU")

    # Give CPU loading a moment to start before beginning GPU setup
    # This prevents the race condition where ToGpu starts before ToHost begins
    import time

    logger.debug("Allowing CPU loading to initialize...")
    time.sleep(0.1)

    if not storage_path:
        from .config import get_storage_path

        storage_path = get_storage_path()
        logger.debug(f"Using default storage path: {storage_path}")

    tensor_index_path = os.path.join(storage_path, model_path, "tensor_index.json")
    logger.debug(f"Reading tensor index from: {tensor_index_path}")

    try:
        with open(tensor_index_path, "r") as f:
            tensor_index = json.load(f)
        logger.debug(
            f"Successfully loaded tensor index with {len(tensor_index)} entries"
        )
    except FileNotFoundError:
        logger.error(f"❌ Tensor index file not found: {tensor_index_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error reading tensor index: {e}")
        raise

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    start = time.time()
    logger.debug("Expanding device map for tensor names")
    expanded_device_map = _expand_tensor_name(device_map, list(tensor_index.keys()))

    logger.debug("Calculating device memory requirements")
    device_memory = calculate_device_memory(expanded_device_map, tensor_data_index)
    logger.debug(f"Device memory requirements: {device_memory}")

    logger.debug("Allocating CUDA memory")
    cuda_memory_ptrs = allocate_cuda_memory(device_memory)
    if cuda_memory_ptrs is None:
        logger.error("❌ allocate_cuda_memory returned None")
        raise ValueError("CUDA memory allocation failed - returned None")

    logger.debug("Getting CUDA memory handles")
    cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
    if cuda_memory_handles is None:
        logger.error("❌ get_cuda_memory_handles returned None")
        raise ValueError("CUDA memory handles retrieval failed - returned None")

    logger.debug("Getting device UUID map")
    device_uuid_map = get_device_uuid_map()
    if device_uuid_map is None:
        logger.error("❌ get_device_uuid_map returned None")
        raise ValueError("Device UUID map retrieval failed - returned None")

    logger.debug("Calculating tensor device offsets")
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        expanded_device_map, tensor_data_index
    )
    if tensor_device_offsets is None:
        logger.error(
            "❌ calculate_tensor_device_offsets returned None for tensor_device_offsets"
        )
        raise ValueError("Tensor device offsets calculation failed - returned None")
    if tensor_copy_chunks is None:
        logger.error(
            "❌ calculate_tensor_device_offsets returned None for tensor_copy_chunks"
        )
        raise ValueError("Tensor copy chunks calculation failed - returned None")

    logger.debug(f"Memory allocation and setup took {time.time() - start} seconds")

    replica_uuid = _get_uuid()
    logger.debug(f"Loading model into GPU with replica_uuid: {replica_uuid}")

    ret = storage.load_into_gpu(
        model_path,
        replica_uuid,
        {device_uuid_map[device_id]: v for device_id, v in tensor_copy_chunks.items()},
        {
            device_uuid_map[device_id]: [v]
            for device_id, v in cuda_memory_handles.items()
        },
    )
    if not ret:
        logger.error(f"❌ Failed to load model {model_path} into GPU")
        raise ValueError(f"Failed to load model {model_path} into GPU")

    logger.debug("GPU loading successful, restoring tensors")

    # load model state_dict
    start = time.time()

    # Check that all arguments to restore_tensors are valid
    if tensor_meta_index is None:
        logger.error("❌ tensor_meta_index is None")
        raise ValueError("tensor_meta_index cannot be None")
    if cuda_memory_ptrs is None:
        logger.error("❌ cuda_memory_ptrs is None")
        raise ValueError("cuda_memory_ptrs cannot be None")
    if tensor_device_offsets is None:
        logger.error("❌ tensor_device_offsets is None")
        raise ValueError("tensor_device_offsets cannot be None")

    logger.debug(f"Restoring {len(tensor_meta_index)} tensors")
    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )

    if state_dict is None:
        logger.error("❌ restore_tensors returned None")
        raise ValueError("restore_tensors failed - returned None")

    logger.info(
        f"✅ Restored state_dict with {len(state_dict)} tensors in {time.time() - start} seconds"
    )

    return replica_uuid, state_dict
