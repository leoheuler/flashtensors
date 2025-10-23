"""
FlashEngine Library - Ultra-fast AI model loading with CUDA acceleration
"""

__version__ = "0.1.0"

try:
    from ._C import (
        allocate_cuda_memory,
        get_cuda_memory_handles,
        get_device_uuid_map,
        restore_tensors,
        save_tensors,
    )
    from ._checkpoint_store import CheckpointStore

    _extensions_available = True
except ImportError as e:
    _extensions_available = False
    _import_error = e
    allocate_cuda_memory = None
    get_cuda_memory_handles = None
    get_device_uuid_map = None
    restore_tensors = None
    save_tensors = None
    CheckpointStore = None

from .api import (
    configure,
    connect,
    register_model,
    load_model,
    cleanup_gpu,
    get_model_info,
    list_models,
    shutdown_server,
)

from .torch_storage import save_dict, load_dict, load_dict_non_blocking
from .device_info import cuda_status, device_count, device_name
from .utils.logger import (
    init_logger,
    set_log_level,
    get_log_level,
    disable_debug_logs,
    enable_debug_logs,
    enable_info_logs,
    enable_minimal_logs,
    enable_quiet_mode,
)

from .utils.model_utils import (
    calculate_device_memory,
    calculate_tensor_device_offsets,
    get_no_split_modules,
    get_tied_no_split_modules,
    send_module_buffers_to_device,
)
from .utils.device_map_utils import (
    _expand_tensor_name,
    _compute_device_placement_from_map,
    _compute_device_placement_from_map_fast,
    _transform_device_map_to_dict,
    DeviceMapType,
)

from .integrations.vllm import (
    FlashLLMLoader,
    activate as activate_vllm_integration,
    save_llm_state,
    patch_model_loader,
)

from .config import (
    get_config,
    update_config,
    get_storage_path,
    get_server_address,
)
from .server_manager import (
    ServerManager,
    get_server_manager,
    ensure_server_running,
)

__all__ = [
    # High-level API
    "configure",
    "connect",
    "register_model",
    "load_model",
    "cleanup_gpu",
    "get_model_info",
    "list_models",
    "shutdown_server",
    # Low-level C++ functions
    "allocate_cuda_memory",
    "get_cuda_memory_handles",
    "get_device_uuid_map",
    "restore_tensors",
    "save_tensors",
    "CheckpointStore",
    # Torch storage functions
    "save_dict",
    "load_dict",
    "load_dict_non_blocking",
    # Device info functions
    "cuda_status",
    "device_count",
    "device_name",
    # Utility functions
    "calculate_device_memory",
    "calculate_tensor_device_offsets",
    "_expand_tensor_name",
    "_compute_device_placement_from_map",
    "_compute_device_placement_from_map_fast",
    "_transform_device_map_to_dict",
    "DeviceMapType",
    "get_no_split_modules",
    "get_tied_no_split_modules",
    "send_module_buffers_to_device",
    # Logger functions
    "init_logger",
    "set_log_level",
    "get_log_level",
    "disable_debug_logs",
    "enable_debug_logs",
    "enable_info_logs",
    "enable_minimal_logs",
    "enable_quiet_mode",
    # VLLM integration
    "FlashLLMLoader",
    "activate_vllm_integration",
    "save_llm_state",
    "patch_model_loader",
    # Configuration and server management
    "get_config",
    "update_config",
    "get_storage_path",
    "get_server_address",
    "ServerManager",
    "get_server_manager",
    "ensure_server_running",
]


def info():
    print(f"FlashEngine library v{__version__}")
    print(f"C++ extensions available: {_extensions_available}")
    if not _extensions_available:
        print(f"Import error: {_import_error}")
        print("To build extensions:")
        print("  1. cd flashtensors")
        print("  2. python setup.py build_ext --inplace")
        print("  3. python setup.py bdist_wheel")
        print("  4. pip install dist/*.whl")
    print()
    print("High-level API functions:")
    high_level = [
        "configure",
        "connect",
        "register_model",
        "load_model",
        "cleanup_gpu",
        "get_model_info",
        "list_models",
        "shutdown_server",
    ]
    for func in high_level:
        print(f"  - {func}")
    print()
    print("Low-level C++ functions:")
    low_level = [
        "allocate_cuda_memory",
        "get_cuda_memory_handles",
        "get_device_uuid_map",
        "restore_tensors",
        "save_tensors",
        "CheckpointStore",
    ]
    for func in low_level:
        available = "✓" if globals().get(func) is not None else "✗"
        print(f"  {available} {func}")
    print()
    print("Storage & utility functions:")
    storage_utils = [
        "StorageClient",
        "StorageServicer",
        "save_dict",
        "load_dict",
        "cuda_status",
        "device_count",
    ]
    for func in storage_utils:
        print(f"  - {func}")
