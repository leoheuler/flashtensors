from typing import Dict, Type, Optional, List
from .base import BaseBackend, DownloadResult
from .vllm_backend import VLLMBackend
from .transformers_backend import TransformersBackend
from .whisper_backend import WhisperBackend
from flashtensors.config import get_storage_path

class BackendRegistry:
    def __init__(self):
        self._backends: Dict[str, Type[BaseBackend]] = {}
        self._instances: Dict[str, BaseBackend] = {}
        self._storage_path: Optional[str] = None
        
        self.register_backend("vllm", VLLMBackend)
        self.register_backend("transformers", TransformersBackend)
        self.register_backend("whisper", WhisperBackend)

    def register_backend(self, name: str, backend_class: Type[BaseBackend]):
        if not issubclass(backend_class, BaseBackend):
            raise ValueError(f"Backend {backend_class} must inherit from BaseBackend")
        
        self._backends[name] = backend_class
    
    def set_storage_path(self, storage_path: str):
        self._storage_path = storage_path
    
    def get_backend(self, name: str) -> BaseBackend:
        if name not in self._backends:
            raise ValueError(f"Unknown backend: {name}. Available: {list(self._backends.keys())}")
        
        if name not in self._instances:
            if self._storage_path is None:
                self._storage_path = get_storage_path()
            self._instances[name] = self._backends[name](self._storage_path)
        
        return self._instances[name]
    
    def list_backends(self) -> List[str]:
        return list(self._backends.keys())
    
    def find_model_backend(self, model_id: str) -> Optional[str]:
        if self._storage_path is None:
            return None
        
        for backend_name in self._backends:
            backend = self.get_backend(backend_name)
            if backend.get_model_info(model_id) is not None:
                return backend_name
        
        return None


_registry = BackendRegistry()

def get_registry() -> BackendRegistry:
    return _registry

def get_backend(name: str) -> BaseBackend:
    return _registry.get_backend(name)

def list_backends() -> List[str]:
    return _registry.list_backends()

def find_model_backend(model_id: str) -> Optional[str]:
    return _registry.find_model_backend(model_id)


__all__ = [
    "BaseBackend",
    "DownloadResult", 
    "BackendRegistry",
    "get_registry",
    "get_backend",
    "list_backends",
    "find_model_backend",
    "VLLMBackend",
    "TransformersBackend",
    "WhisperBackend",
    "BarkBackend",
]
