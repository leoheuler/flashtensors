from functools import reduce
from typing import Dict, Tuple, List, Any

import torch
from torch import nn
from accelerate.utils import find_tied_parameters


def calculate_device_memory(device_map: Dict[str, Any], tensor_index: Dict[str, Tuple[int, int]]) -> Dict[Any, int]:
    """
    Calculate total memory required for each device based on device mapping.
    
    Args:
        device_map: Mapping of tensor names to devices
        tensor_index: Index of tensor offsets and sizes
        
    Returns:
        Dictionary mapping devices to required memory in bytes
    """
    device_memory = {}
    tensor_record = {}
    
    for tensor_name, device in device_map.items():
        if tensor_name in tensor_index:
            if device not in device_memory:
                device_memory[device] = 0
            offset, size = tensor_index[tensor_name]
            if (offset, size) in tensor_record:
                continue  # Skip duplicate tensors
            tensor_record[(offset, size)] = True
            device_memory[device] += tensor_index[tensor_name][1]
        else:
            raise ValueError(f"Tensor {tensor_name} not found in tensor_index.")

    return device_memory


def calculate_tensor_device_offsets(
    device_map: Dict[str, Any], 
    tensor_index: Dict[str, Tuple[int, int]]
) -> Tuple[Dict[Any, Dict[str, int]], Dict[Any, List[Tuple[int, int, int, int]]]]:
    """
    Calculate tensor device offsets and copy chunks for memory allocation.
    
    Args:
        device_map: Mapping of tensor names to devices
        tensor_index: Index of tensor offsets and sizes
        
    Returns:
        Tuple of (tensor_device_offsets, tensor_copy_chunks)
    """
    tensor_device_offsets = {}
    tensor_copy_chunks = {}
    device_offset = {}
    tensor_record = {}
    
    for tensor_name, device in device_map.items():
        if device not in tensor_device_offsets:
            tensor_device_offsets[device] = {}
            tensor_copy_chunks[device] = []
            device_offset[device] = 0
        if tensor_name in tensor_index:
            offset, size = tensor_index[tensor_name]
            if (offset, size) in tensor_record:
                tensor_device_offsets[device][tensor_name] = tensor_record[
                    (offset, size)
                ]
            else:
                tensor_record[(offset, size)] = device_offset[device]
                tensor_device_offsets[device][tensor_name] = device_offset[
                    device
                ]
                tensor_copy_chunks[device].append(
                    (offset, size, device_offset[device], 0)
                )
                device_offset[device] += size
        else:
            raise ValueError(f"Tensor {tensor_name} not found in tensor_index.")

    return tensor_device_offsets, tensor_copy_chunks


def dtype_byte_size(dtype: torch.dtype) -> int:
    """Get the byte size of a PyTorch dtype."""
    return torch.finfo(dtype).bits // 8


def get_total_parameter_size(module: nn.Module) -> int:
    """
    Calculate the total parameter size of a module in bytes.
    
    Args:
        module: PyTorch module
        
    Returns:
        Total parameter size in bytes
    """
    total_param_size = 0
    for param in module.parameters():
        total_param_size += param.numel() * dtype_byte_size(param.dtype)
    return total_param_size


def get_no_split_modules(
    model: nn.Module, 
    no_split_modules_list: List[str], 
    parent_name: str = ""
) -> Dict[str, int]:
    """
    Get modules that should not be split across devices.
    
    Args:
        model: PyTorch model
        no_split_modules_list: List of module class names that shouldn't be split
        parent_name: Parent module name for recursive calls
        
    Returns:
        Dictionary mapping module names to their parameter sizes
    """
    no_split_modules = {}
    for name, submodule in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        module_class_name = submodule.__class__.__name__
        # If the module is a leaf module or in the no_split_modules_list, we don't split it
        if (
            not list(submodule.children())
            or module_class_name in no_split_modules_list
        ):
            no_split_modules[full_name] = get_total_parameter_size(submodule)
            continue
        no_split_modules.update(
            get_no_split_modules(submodule, no_split_modules_list, full_name)
        )

    return no_split_modules


def get_parameter_size(model: nn.Module, param_path: str) -> int:
    """
    Get the size of a specific parameter in a model.
    
    Args:
        model: PyTorch model
        param_path: Dot-separated path to the parameter
        
    Returns:
        Parameter size in bytes
    """
    # Split the parameter path by dots
    attributes = param_path.split(".")

    # Use reduce to traverse the model's attributes
    param = reduce(getattr, attributes, model)

    # Return the size of the parameter
    return param.numel() * dtype_byte_size(param.dtype)


def get_tied_no_split_modules(
    model: nn.Module, 
    no_split_modules: Dict[str, int]
) -> List[Tuple[List[str], int]]:
    """
    Get modules with tied parameters that should not be split.
    
    Args:
        model: PyTorch model
        no_split_modules: Dictionary of modules that shouldn't be split
        
    Returns:
        List of tuples containing (tied_module_group, shared_size)
    """
    tied_parameters = find_tied_parameters(model)
    tied_modules = []
    for tied_param_group in tied_parameters:
        tied_module_group = []
        shared_size = None
        for tied_param in tied_param_group:
            param_size = get_parameter_size(model, tied_param)
            if shared_size is None:
                shared_size = param_size
            else:
                assert (
                    shared_size == param_size
                ), f"Parameter {tied_param} does not have the same size as the other parameters in the group"
            tied_module = None
            while "." in tied_param:
                tied_param = tied_param.rsplit(".", 1)[0]
                if tied_param in no_split_modules:
                    tied_module = tied_param
                    break
            if tied_module is None:
                raise ValueError(
                    f"Parameter {tied_param} is not in the no_split_modules list"
                )
            tied_module_group.append(tied_module)
        tied_modules.append((tied_module_group, shared_size))

    return tied_modules


def set_module_buffer_to_device(
    module: nn.Module,
    target: str,
    device: torch.device,
):
    """
    Set a specific module buffer to a target device.
    
    Args:
        module: PyTorch module
        target: Dot-separated path to the buffer
        device: Target device
    """
    module_path, _, buffer_name = target.rpartition(".")

    mod: torch.nn.Module = module.get_submodule(module_path)

    if not hasattr(mod, buffer_name):
        raise AttributeError(
            mod._get_name() + " has no attribute `" + buffer_name + "`"
        )

    buffer = mod._buffers[buffer_name]
    mod._buffers[buffer_name] = buffer.to(device)


def send_module_buffers_to_device(
    module: nn.Module,
    device_map: Dict[str, Any],
):
    """
    Send module buffers to their assigned devices based on device map.
    
    Args:
        module: PyTorch module
        device_map: Mapping of modules/tensors to devices
    """
    if "" in device_map and len(device_map) != 1:
        raise RuntimeError(
            f"Device map {device_map} is invalid. If you want to specify the default device, use key ''."
        )

    buffer_names = [name for name, _ in module.named_buffers()]
    for tensor_or_module, device_id in device_map.items():
        if tensor_or_module == "":
            for buffer_name in buffer_names:
                set_module_buffer_to_device(module, buffer_name, device_id)
        else:
            for buffer_name in buffer_names:
                if buffer_name.startswith(tensor_or_module):
                    set_module_buffer_to_device(module, buffer_name, device_id)


__all__ = [
    # Memory and device calculations
    "calculate_device_memory",
    "calculate_tensor_device_offsets",
    
    # Module and parameter analysis
    "dtype_byte_size",
    "get_total_parameter_size",
    "get_no_split_modules",
    "get_parameter_size",
    "get_tied_no_split_modules",
    
    # Buffer and device management
    "set_module_buffer_to_device",
    "send_module_buffers_to_device",
]
