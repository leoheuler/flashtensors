import concurrent.futures
import json
import os
import time
import uuid
from typing import Optional, Union
import importlib

import torch
from accelerate import dispatch_model, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig
from transformers.integrations.bitsandbytes import (
    set_module_quantized_tensor_to_device,
    replace_with_bnb_linear,
)
from torch import nn

from flashtensors.storage_client import StorageClient
from flashtensors.config import get_storage_path
from flashtensors._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
)
from flashtensors.torch_storage import load_dict_non_blocking, save_dict
from flashtensors.utils.device_map_utils import (
    DeviceMapType,
    _compute_device_placement_from_map,
    _compute_device_placement_from_map_fast,
    _expand_tensor_name,
    _transform_device_map_to_dict,
)
from flashtensors.utils import (
    init_logger,
    calculate_device_memory,
    calculate_tensor_device_offsets,
    get_no_split_modules,
    get_tied_no_split_modules,
    send_module_buffers_to_device,
)

# Optional PEFT imports
try:
    from peft import PeftModel, get_peft_model_state_dict, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = init_logger(__name__)


def _get_uuid():
    """Generate a unique UUID for model replicas."""
    return str(uuid.uuid4())


def save_model(model: nn.Module, model_path: str):
    """
    Save a transformers model in flashtensors format for fast loading.
    
    Args:
        model: PyTorch model to save
        model_path: Local path to save the converted model
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    model = model.cpu()
    model_state_dict = model.state_dict()

    # Save tensors using flashtensors's optimized storage
    save_dict(model_state_dict, model_path)

    # Save model configuration
    model.config.save_pretrained(model_path)
    if model.can_generate():
        model.generation_config.save_pretrained(model_path)

    # Save module metadata for device placement
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    with open(os.path.join(model_path, "no_split_modules.json"), "w") as f:
        json.dump(no_split_modules, f)

    # Save tied parameters information
    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)
    with open(os.path.join(model_path, "tied_no_split_modules.json"), "w") as f:
        json.dump(tied_no_split_modules, f)

    logger.info(f"Model saved to {model_path} in flashtensors format")


def save_lora(lora: PeftModel, lora_path: str):
    """
    Save a LoRA adapter in flashtensors format.
    
    Args:
        lora: PeftModel with LoRA adapters
        lora_path: Path to save the LoRA adapter
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA support. Install with: pip install peft")
    
    if not os.path.exists(lora_path):
        os.makedirs(lora_path, exist_ok=True)

    model = lora.cpu()
    lora_state_dict = get_peft_model_state_dict(model)

    # Save LoRA tensors
    save_dict(lora_state_dict, lora_path)

    # Save LoRA configuration
    if (
        hasattr(model, "peft_config")
        and model.peft_config is not None
        and isinstance(model.peft_config, PeftConfig)
    ):
        logger.info(f"Saving LoRA config: {model.peft_config}")
        config_save_path = os.path.join(lora_path, "adapter_config.json")
        model.peft_config.save_pretrained(config_save_path)

    logger.info(f"LoRA adapter saved to {lora_path}")


def load_model(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config=None,
    storage_path: Optional[str] = None,
    fully_parallel: bool = False,
    hf_model_class: str = "AutoModelForCausalLM",
):
    """
    Load a transformers model with flashtensors's fast loading.
    
    Args:
        model_path: Path to the model (relative to storage path)
        device_map: Device placement strategy
        torch_dtype: Target dtype for the model
        quantization_config: Quantization configuration (BitsAndBytesConfig)
        storage_path: Base storage path (uses config if None)
        fully_parallel: Whether to use fully parallel loading
        hf_model_class: HuggingFace model class name
        
    Returns:
        Loaded transformers model
    """
    if storage_path is None:
        storage_path = get_storage_path()
    
    logger.info(f"🚀 Loading {hf_model_class} model {model_path} with flashtensors fast loading...")
    
    if fully_parallel:
        return fully_parallel_load(
            model_path=model_path,
            hf_model_class=hf_model_class,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            storage_path=storage_path,
        )
    else:
        return best_effort_load(
            model_path=model_path,
            hf_model_class=hf_model_class,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            storage_path=storage_path,
        )


def fully_parallel_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config=None,
    storage_path: Optional[str] = None,
):
    """Fully parallel model loading with concurrent tensor loading and model initialization."""
    start = time.time()
    device_map = _transform_device_map_to_dict(device_map)
    
    # Load tied module information
    with open(
        os.path.join(storage_path, model_path, "tied_no_split_modules.json"),
        "r",
    ) as f:
        tied_no_split_modules = json.load(f)

    if isinstance(device_map, str):
        with open(
            os.path.join(storage_path, model_path, "no_split_modules.json"),
            "r",
        ) as f:
            no_split_modules = json.load(f)
        device_map = _compute_device_placement_from_map_fast(
            no_split_modules, tied_no_split_modules, device_map
        )

    # Parallel execution: load tensors and initialize model concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            load_dict_non_blocking, model_path, device_map, storage_path
        )
        logger.debug(f"load_dict_non_blocking takes {time.time() - start} seconds")

        # Load model config while tensors are loading
        start = time.time()
        config = AutoConfig.from_pretrained(
            f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
        )
        if torch_dtype is not None:
            config.torch_dtype = torch_dtype
        logger.debug(f"load config takes {time.time() - start} seconds")

        # Initialize empty model while tensors are loading
        start = time.time()
        with init_empty_weights():
            module = importlib.import_module("transformers")
            _class = getattr(module, hf_model_class)
            if hasattr(_class, "from_config"):
                model = _class.from_config(
                    config,
                    trust_remote_code=True,
                ).to(config.torch_dtype)
            elif hasattr(_class, "_from_config"):
                model = _class._from_config(
                    config
                ).to(config.torch_dtype)
            else:
                model = _class(
                    config
                ).to(config.torch_dtype)
        model.tie_weights()
        logger.debug(f"load model takes {time.time() - start} seconds")

        # Wait for tensor loading to complete
        replica_uuid, state_dict = future.result()

    # Set model parameters with loaded tensors
    _set_model_tensors(
        model=model,
        state_dict=state_dict,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    # Confirm model loading with storage client
    client = StorageClient()
    client.confirm_model_loaded(model_path, replica_uuid)
    
    model.eval()
    logger.info(f"✅ Model {model_path} loaded successfully with fully parallel loading")
    return model

def best_effort_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config=None,
    storage_path: Optional[str] = None,
):
    """Best effort model loading with optimized memory management."""
    client = StorageClient()
    
    # Load model into CPU first
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    replica_uuid = _get_uuid()
    device_map = _transform_device_map_to_dict(device_map)

    # Validate device map
    if isinstance(device_map, dict) and (
        torch.device("cpu") in device_map.values()
        or "cpu" in device_map.values()
    ):
        raise ValueError("CPU is not supported in device_map for fast loading.")

    # Load model configuration
    start = time.time()
    config = AutoConfig.from_pretrained(
        f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
    )
    if torch_dtype is not None:
        config.torch_dtype = torch_dtype
    logger.debug(f"load config takes {time.time() - start} seconds")
    
    # Initialize empty model
    start = time.time()
    with init_empty_weights():
        module = importlib.import_module("transformers")
        _class = getattr(module, hf_model_class)
        model = _class.from_config(
                config,
                trust_remote_code=True,
            ).to(config.torch_dtype)
    model.tie_weights()
    logger.debug(f"load model takes {time.time() - start} seconds")

    # Compute device placement
    start = time.time()
    if isinstance(device_map, str):
        device_map = _compute_device_placement_from_map(
            model, device_map, config.torch_dtype
        )
        logger.debug(f"device_map: {device_map}")
    
    # Validate GPU availability
    if "cpu" in device_map.values():
        raise ValueError(
            "GPUs are either unavailable or do not have enough memory. "
            "Please ensure they are available and ready for use."
        )
    logger.debug(f"compute_device_placement takes {time.time() - start} seconds")

    # Load tensor metadata
    with open(
        os.path.join(storage_path, model_path, "tensor_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    # Allocate CUDA memory and setup GPU loading
    start = time.time()
    expanded_device_map = _expand_tensor_name(
        device_map, list(tensor_index.keys())
    )
    device_memory = calculate_device_memory(
        expanded_device_map, tensor_data_index
    )
    cuda_memory_ptrs = allocate_cuda_memory(device_memory)
    cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
    device_uuid_map = get_device_uuid_map()
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        expanded_device_map, tensor_data_index
    )
    logger.debug(f"allocate_cuda_memory takes {time.time() - start} seconds")

    # Load tensors into GPU
    ret = client.load_into_gpu(
        model_path,
        replica_uuid,
        {
            device_uuid_map[device_id]: v
            for device_id, v in tensor_copy_chunks.items()
        },
        {
            device_uuid_map[device_id]: [v]
            for device_id, v in cuda_memory_handles.items()
        },
    )
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into GPU")

    # Restore tensors from CUDA memory
    start = time.time()
    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )
    logger.info(f"restore state_dict takes {time.time() - start} seconds")

    # Set model parameters
    _set_model_tensors(
        model=model,
        state_dict=state_dict,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    # Finalize model loading
    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )

    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    model.hf_device_map = device_map

    logger.info(f"✅ Model {model_path} loaded successfully with best effort loading")
    return model


def _set_model_tensors(
    model: nn.Module,
    state_dict: dict,
    device_map: dict,
    quantization_config=None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Set model tensors from loaded state dict with optional quantization."""
    with torch.no_grad():
        if quantization_config and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "BitsAndBytesConfig not available. Please update transformers: "
                    "pip install transformers>=4.20.0"
                )

            if not isinstance(quantization_config, BitsAndBytesConfig):
                raise ValueError(
                    f"Invalid config type: {type(quantization_config)}"
                )

            logger.debug(
                f"Using quantization: {quantization_config.quantization_method()}"
            )

            if quantization_config.llm_int8_enable_fp32_cpu_offload:
                logger.debug("CPU offloading is not supported, disabling")
                quantization_config.llm_int8_enable_fp32_cpu_offload = False

            has_torch_dtype = torch_dtype is not None
            model = replace_with_bnb_linear(
                model, quantization_config=quantization_config
            )

            for name, param in state_dict.items():
                final_device = param.device
                if not has_torch_dtype:
                    param = param.to(torch.float16)

                set_module_quantized_tensor_to_device(
                    model, name, final_device, param.to("cpu")
                )
        else:
            if quantization_config is not None:
                logger.debug(
                    "Quantization on current device is not supported yet"
                )

            # Standard tensor loading
            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)
        
        # Send buffers to appropriate devices
        send_module_buffers_to_device(model, device_map)


__all__ = [
    "save_model",
    "save_lora", 
    "load_model",
    "fully_parallel_load",
    "best_effort_load",
]
