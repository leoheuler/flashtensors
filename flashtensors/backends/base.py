from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from dataclasses import dataclass


@dataclass
class DownloadResult:
    """Result of model download and transformation."""
    success: bool
    model_path: str
    error: Optional[str] = None
    download_time: float = 0.0
    transform_time: float = 0.0
    total_time: float = 0.0
    model_size: int = 0


class BaseBackend(ABC):
    def __init__(self, storage_path: str):
        """
        Initialize the backend.
        
        Args:
            storage_path: Base storage path for models
        """
        self.storage_path = storage_path
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'vllm', 'transformers')."""
        pass
    
    @abstractmethod
    async def download_model(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        torch_dtype: str = "auto",
        force: bool = False,
        **kwargs
    ) -> DownloadResult:
        """
        Download and prepare a model for this backend.
        
        Args:
            model_id: Model identifier
            hf_token: HuggingFace token
            torch_dtype: PyTorch dtype
            force: Whether to overwrite existing model
            **kwargs: Backend-specific arguments
            
        Returns:
            DownloadResult with download information
        """
        pass
    
    @abstractmethod
    def load_model(
        self,
        model_id: str,
        **kwargs
    ) -> Any:
        """
        Load a model using this backend.
        
        Args:
            model_id: Model identifier
            **kwargs: Backend-specific loading arguments
            
        Returns:
            Loaded model instance (format depends on backend)
        """
        pass
    
    @abstractmethod
    def is_model_directory(self, path: str) -> bool:
        """
        Check if a directory contains a model for this backend.
        
        Args:
            path: Directory path to check
            
        Returns:
            True if directory contains a model for this backend
        """
        pass
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model information or None if not found
        """
        import os
        model_path = os.path.join(self.storage_path, self.name, model_id)
        
        if not os.path.exists(model_path):
            return None
        
        if not self.is_model_directory(model_path):
            return None
        
        size = self._get_directory_size(model_path)
        ranks = self.count_ranks(model_path)
        
        return {
            "model_id": model_id,
            "path": os.path.join(self.name, model_id),  # Relative path for storage
            "full_path": model_path,  # Absolute path for filesystem
            "backend": self.name,
            "size": size,
            "ranks": ranks,
        }
    
    def count_ranks(self, model_path: str) -> int:
        """
        Count the number of ranks/shards for this model.
        Default implementation returns 1 (single rank).
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Number of ranks
        """
        return 1
    
    def register_with_storage(self, storage_client, model_path: str) -> bool:
        """
        Register model with storage system.
        
        Args:
            storage_client: Storage client instance
            model_path: Relative model path
            
        Returns:
            True if registration succeeded or not needed
        """        
        response = storage_client.register_model(model_path)
        return response.success
    
    def _get_directory_size(self, path: str) -> int:
        """Get the total size of a directory in bytes."""
        import os
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
