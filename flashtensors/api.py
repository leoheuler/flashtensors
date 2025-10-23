import os
import time
import asyncio
from typing import Optional, Dict, Any

from .storage_client import StorageClient
from .utils import init_logger
from .config import (
    update_config, get_storage_path, get_server_address,
    DEFAULT_MEM_POOL_SIZE, DEFAULT_CHUNK_SIZE, DEFAULT_NUM_THREADS,
    DEFAULT_GPU_MEMORY_UTILIZATION, DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
    print_config, reset_config
)
from .server_manager import ensure_server_running, get_server_manager
from .backends import get_registry, get_backend, find_model_backend, list_backends

_vllm_activated = False

def _ensure_vllm_activated():
    global _vllm_activated
    if not _vllm_activated:
        from .integrations.vllm import activate
        activate()
        _vllm_activated = True

def configure(
    storage_path: Optional[str] = None,
    mem_pool_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    num_threads: Optional[int] = None,
    gpu_memory_utilization: Optional[float] = None,
    server_host: str = DEFAULT_SERVER_HOST,
    server_port: int = DEFAULT_SERVER_PORT,
    registration_required: bool = False
) -> None:
    """
    Configure FlashTensorsEngine library settings and start the gRPC server.
    
    Args:
        storage_path: Path where models will be stored (defaults to /workspace)
        mem_pool_size: Size of GPU memory pool in bytes (defaults to 12GB)
        chunk_size: Size of memory chunks in bytes (defaults to 32MB)
        num_threads: Number of processing threads (defaults to 16)
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        server_host: gRPC server host address
        server_port: gRPC server port
        registration_required: Whether model registration is required before loading
    """
    _ensure_vllm_activated()
    
    if storage_path is None:
        storage_path = get_storage_path()
    if mem_pool_size is None:
        mem_pool_size = DEFAULT_MEM_POOL_SIZE
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if num_threads is None:
        num_threads = DEFAULT_NUM_THREADS
    if gpu_memory_utilization is None:
        gpu_memory_utilization = DEFAULT_GPU_MEMORY_UTILIZATION
    
    update_config(
        storage_path=storage_path,
        mem_pool_size=mem_pool_size,
        chunk_size=chunk_size,
        num_threads=num_threads,
        gpu_memory_utilization=gpu_memory_utilization,
        server_host=server_host,
        server_port=server_port,
        registration_required=registration_required
    )
    
    logger = init_logger(__name__)
    os.makedirs(storage_path, exist_ok=True)
    
    registry = get_registry()
    registry.set_storage_path(storage_path)

    logger.info(f"FlashTensorsEngine configured: storage_path={storage_path}, mem_pool_size={mem_pool_size//1024**3}GB")
    logger.info(f"Available backends: {', '.join(list_backends())}")

    logger.info(f"Starting gRPC server on {server_host}:{server_port}...")
    if ensure_server_running():
        logger.info(f"✅ gRPC server is running on {get_server_address()}")
    else:
        logger.error("❌ Failed to start gRPC server")
        raise RuntimeError("Failed to start FlashTensorsEngine gRPC server")


def connect(
    server_host: str = DEFAULT_SERVER_HOST,
    server_port: int = DEFAULT_SERVER_PORT
) -> None:
    """
    Connect to an existing FlashTensorsEngine gRPC server without starting a new one.
    
    This method is used by CLI clients to connect to a background server
    that was started with configure().
    
    Args:
        server_host: gRPC server host address
        server_port: gRPC server port
        
    Raises:
        ConnectionError: If no server is running at the specified address
    """
    _ensure_vllm_activated()
    
    logger = init_logger(__name__)
    
    # Update config with connection details
    update_config(
        server_host=server_host,
        server_port=server_port
    )
    
    # Test connection to server
    try:
        storage = StorageClient()
        # Make a simple health check call
        logger.info(f"Connecting to gRPC server at {server_host}:{server_port}...")
        
        # Try to get server status or make a simple call to verify connection
        # If StorageClient has a health check method, use it, otherwise we'll catch exceptions
        storage_path = get_storage_path()
        registry = get_registry()
        registry.set_storage_path(storage_path)
        
        logger.info(f"✅ Successfully connected to FlashEngine server at {get_server_address()}")
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to FlashEngine server at {server_host}:{server_port}")
        raise ConnectionError(f"No FlashEngine server running at {server_host}:{server_port}. Please start the server first with 'flash start'") from e

def register_model(
    model_id: str,
    backend: str = "vllm",
    torch_dtype: str = "float16",
    force: bool = False,
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Download and prepare a model for the specified backend.
    
    This function implements the complete flow:
    1. Download model from HuggingFace
    2. Transform it to backend-specific format (if needed)
    3. Register with storage system (if needed)
    
    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen3-0.6B")
        backend: Backend to use ("vllm", "transformers", etc)
        torch_dtype: PyTorch dtype for the model
        force: Whether to overwrite existing transformed model
        hf_token: HuggingFace token for private models
        
    Returns:
        Dictionary with download/transformation results and metrics
    """
    storage = StorageClient()
    logger = init_logger(__name__)
    
    start_time = time.time()
    logger.info(f"🔄 Preparing model {model_id} for {backend} backend...")
    
    try:
        backend_instance = get_backend(backend)
        
        # Step 1: Download and transform model
        download_result = asyncio.run(backend_instance.download_model(
            model_id=model_id,
            torch_dtype=torch_dtype,
            force=force,
            hf_token=hf_token
        ))
        
        if not download_result.success:
            return {
                "status": "error", 
                "error": download_result.error,
                "path": download_result.model_path
            }
        
        # Step 2: Register with storage system if needed
        register_start = time.time()
        register_success = True
        
        logger.info(f"Registering model with storage system...")
        register_success = backend_instance.register_with_storage(storage, download_result.model_path)
        if not register_success:
            logger.warning(f"Model download succeeded but storage registration failed")
        
        register_time = time.time() - register_start
        total_time = time.time() - start_time
        
        logger.info(f"✅ Model {model_id} prepared for {backend} successfully in {total_time:.2f}s")
        
        return {
            "status": "success",
            "backend": backend,
            "path": download_result.model_path,
            "metrics": {
                "total_time": total_time,
                "download_time": download_result.download_time,
                "transform_time": download_result.transform_time, 
                "register_time": register_time,
                "model_size": download_result.model_size
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to prepare model {model_id} for {backend}: {e}")
        return {
            "status": "error",
            "error": str(e), 
            "path": ""
        }


def load_model(
    model_id: str,
    backend: Optional[str] = None,
    **kwargs
):
    """
    Load a model using the appropriate backend (auto-detected or specified).
    
    Args:
        model_id: Model identifier
        backend: Backend to use. If None, auto-detect from available models.
        **kwargs: Backend-specific loading arguments:
            - For VLLM: dtype, gpu_memory_utilization
            - For transformers: torch_dtype, device_map
        
    Returns:
        Loaded model instance (format depends on backend)
    """
    logger = init_logger(__name__)
    
    # Auto-detect backend if not specified
    if backend is None:
        backend = find_model_backend(model_id)
        if backend is None:
            raise FileNotFoundError(
                f"Model {model_id} not found. Please run flash.register_model('{model_id}', backend='<backend>') first. "
                f"Available backends: {', '.join(list_backends())}"
            )
    
    logger.info(f"⚡ Loading model {model_id} using {backend} backend...")
    
    # Clear GPU memory before loading
    try:
        cleanup_gpu()
    except Exception as e:
        logger.warning(f"⚠️  Failed to clear GPU memory before loading: {e}")
    
    try:
        # Get backend instance and load model
        backend_instance = get_backend(backend)
        return backend_instance.load_model(model_id, **kwargs)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_id} with {backend} backend: {e}")
        raise



def cleanup_gpu():
    """Clean up GPU memory, vLLM resources, and distributed process groups."""
    storage = StorageClient()
    logger = init_logger(__name__)
    logger.info("🧹 Starting comprehensive GPU and vLLM cleanup...")

    try:
        # First, try to clean up vLLM resources if available
        try:
            import torch
            # Clear CUDA cache
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            import torch.distributed as dist

            if dist.is_initialized():
                logger.info("Destroying distributed process group...")
                dist.destroy_process_group()
                logger.info("✅ Distributed process group destroyed")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up distributed process group: {e}")

        # Clear GPU memory through storage service
        try:
            response = storage.clear_gpu_memory()
            if not response.success:
                logger.error(f"❌ Failed to clear GPU memory: {response.error}")
                raise RuntimeError(response.error)
            logger.info("✅ GPU memory cleared via storage service")
        except Exception as e:
            logger.error(f"❌ Error clearing GPU memory: {e}")

        # Final PyTorch and CUDA cleanup
        try:
            import torch
            import gc

            if torch.cuda.is_available():
                logger.info("Performing final PyTorch CUDA cleanup...")
                # Clear any remaining CUDA tensors
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            del obj
                    except:
                        pass
                # Force garbage collection

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("✅ Final PyTorch CUDA cleanup completed")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"⚠️ Could not perform final CUDA cleanup: {e}")

        # Reset CUDA context if possible
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("Resetting CUDA context...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # This is a more aggressive way to reset CUDA context
                torch.cuda.ipc_collect()
                logger.info("✅ CUDA context reset")
        except Exception as e:
            logger.warning(f"⚠️ Could not reset CUDA context: {e}")

        logger.info("✅ GPU memory and vLLM resources cleanup completed")

    except Exception as e:
        logger.error(f"❌ Failed to clean GPU memory: {e}")
        raise


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a transformed model.
    
    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen-14b")
        
    Returns:
        Dictionary with model information or None if not found
    """
    storage_path = get_storage_path()
    
    # Check each available backend
    for backend_name in list_backends():
        try:
            backend = get_backend(backend_name)
            model_info = backend.get_model_info(model_id)
            if model_info is not None:
                return model_info
        except Exception:
            continue  # Skip backends that fail to initialize
    
    return None


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models across all backends.
    
    Returns:
        Dictionary mapping model_id to model information
    """
    models = {}
    storage_path = get_storage_path()
    
    if not os.path.exists(storage_path):
        return models
    
    # Scan each available backend
    for backend_name in list_backends():
        try:
            backend = get_backend(backend_name)
            backend_path = os.path.join(storage_path, backend_name)
            
            if not os.path.exists(backend_path):
                continue
            
            # Recursively scan for models in this backend
            _scan_backend_models(backend_path, backend, models)
            
        except Exception:
            continue  # Skip backends that fail to initialize
    
    return models


def _scan_backend_models(path: str, backend, models: Dict[str, Dict[str, Any]], prefix: str = ""):
    """Recursively scan for models in a backend directory."""
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if not os.path.isdir(item_path):
                continue
            
            # Build model_id from directory structure
            current_model_id = os.path.join(prefix, item) if prefix else item
            
            # Check if this is a model directory
            if backend.is_model_directory(item_path):
                model_info = backend.get_model_info(current_model_id)
                if model_info:
                    models[current_model_id] = model_info
            else:
                # Continue scanning deeper
                _scan_backend_models(item_path, backend, models, current_model_id)
                
    except OSError:
        pass


def _get_directory_size(path: str) -> int:
    """Get the total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except (OSError, IOError):
        pass
    return total


def shutdown_server() -> None:
    """
    Shutdown the FlashEngine gRPC server.
    
    This stops the storage server and supervisord daemon.
    After calling this, you can reconfigure FlashEngine settings
    before starting the server again.
    """
    logger = init_logger(__name__)
    
    try:
        server_manager = get_server_manager()
        logger.info("🛑 Shutting down FlashEngine server...")
        server_manager.stop_server()
        logger.info("✅ FlashEngine server shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during server shutdown: {e}")
        raise


__all__ = [
    "configure",
    "connect",
    "register_model",
    "load_model", 
    "cleanup_gpu",
    "get_model_info",
    "list_models",
    "shutdown_server",
    "print_config",
    "reset_config",
]
