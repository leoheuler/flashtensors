import os
import json
import threading
from typing import Dict, Any

DEFAULT_STORAGE_PATH = "/workspace"
DEFAULT_MEM_POOL_SIZE = 12 * 1024**3  # 12GB
DEFAULT_CHUNK_SIZE = 32 * 1024**2     # 32MB
DEFAULT_NUM_THREADS = 16
DEFAULT_GPU_MEMORY_UTILIZATION = 0.8
DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8073

CONFIG_FILE = os.path.expanduser("~/.flashtensors/config.json")
_config_lock = threading.Lock()

def _get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "storage_path": DEFAULT_STORAGE_PATH,
        "mem_pool_size": DEFAULT_MEM_POOL_SIZE,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "num_threads": DEFAULT_NUM_THREADS,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "server_host": DEFAULT_SERVER_HOST,
        "server_port": DEFAULT_SERVER_PORT,
        "registration_required": False,
    }

def _load_config_from_file() -> Dict[str, Any]:
    """Load configuration from persistent file."""
    with _config_lock:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults to handle missing keys
                    config = _get_default_config()
                    config.update(file_config)
                    return config
            except (json.JSONDecodeError, IOError):
                pass
        return _get_default_config()

def _save_config_to_file(config: Dict[str, Any]) -> None:
    """Save configuration to persistent file."""
    with _config_lock:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        persistent_config = {k: v for k, v in config.items()}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(persistent_config, f, indent=2)

def _set_environment_variables(config: Dict[str, Any]) -> None:
    """Set environment variables based on config values."""
    env_vars = {
        "FLASHENGINE_HOST": str(config["server_host"]),
        "FLASHENGINE_PORT": str(config["server_port"]),
        "FLASHENGINE_STORAGE_PATH": str(config["storage_path"]),
        "FLASHENGINE_NUM_THREADS": str(config["num_threads"]),
        "FLASHENGINE_CHUNK_SIZE": str(config["chunk_size"]),
        "FLASHENGINE_MEM_POOL_SIZE": str(config["mem_pool_size"]),
        "STORAGE_PATH": str(config["storage_path"]),  # Legacy support
    }
    
    for env_var, value in env_vars.items():
        os.environ[env_var] = value

def get_config() -> Dict[str, Any]:
    """
    Get current configuration from persistent file.
    Returns defaults if config file doesn't exist.
    Also ensures environment variables are set.
    """
    config = _load_config_from_file()
    _set_environment_variables(config)
    return config

def is_server_running() -> bool:
    """Check if FlashEngine server is currently running."""
    import grpc
    from .proto import storage_pb2, storage_pb2_grpc
    
    config = get_config()
    server_address = f"{config['server_host']}:{config['server_port']}"
    
    try:
        channel = grpc.insecure_channel(server_address)
        stub = storage_pb2_grpc.StorageStub(channel)
        request = storage_pb2.GetServerConfigRequest()
        _response = stub.GetServerConfig(request, timeout=10.0)
        channel.close()
        return True
    except (grpc.RpcError, Exception):
        return False

def update_config(**kwargs) -> None:
    """
    Update configuration values, persist to file, and set environment variables.
    
    Raises:
        RuntimeError: If server is currently running (config cannot be changed)
        ValueError: If unknown configuration key is provided
    """
    if is_server_running():
        raise RuntimeError("Cannot update configuration while server is running. Stop the server first.")
    
    config = _load_config_from_file()
    
    valid_keys = set(_get_default_config().keys())
    for key in kwargs:
        if key not in valid_keys:
            raise ValueError(f"Unknown configuration key: {key}")
    
    config.update(kwargs)
    _save_config_to_file(config)
    _set_environment_variables(config)

def reset_config() -> None:
    """Reset configuration to defaults and set environment variables."""
    if is_server_running():
        raise RuntimeError("Cannot reset configuration while server is running. Stop the server first.")
    
    default_config = _get_default_config()
    _save_config_to_file(default_config)
    _set_environment_variables(default_config)

def get_storage_path() -> str:
    """Get the configured storage path."""
    config = get_config()
    return config["storage_path"]

def get_server_address() -> str:
    """Get the server address string."""
    config = get_config()
    return f"{config['server_host']}:{config['server_port']}"

def get_server_config() -> Dict[str, Any]:
    """
    Get server configuration in the format expected by server components.
    """
    config = get_config()
    return {
        "host": config["server_host"],
        "port": config["server_port"],
        "storage_path": config["storage_path"],
        "num_thread": config["num_threads"],
        "chunk_size": config["chunk_size"],
        "mem_pool_size": config["mem_pool_size"],
        "registration_required": config["registration_required"],
    }

def print_config() -> None:
    """Print current configuration for debugging."""
    config = get_config()
    print("FlashEngine Configuration:")
    print("=" * 40)
    for key, value in config.items():
        if key in ["mem_pool_size", "chunk_size"]:
            # Display memory sizes in human-readable format
            if value >= 1024**3:
                display_value = f"{value // 1024**3}GB ({value} bytes)"
            elif value >= 1024**2:
                display_value = f"{value // 1024**2}MB ({value} bytes)"
            else:
                display_value = f"{value} bytes"
        else:
            display_value = value
        print(f"{key:25}: {display_value}")
    print("=" * 40)
    print(f"Config file: {CONFIG_FILE}")
    print(f"Server running: {is_server_running()}")
