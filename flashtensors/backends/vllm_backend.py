import os
import gc
import torch
import shutil
from tempfile import TemporaryDirectory
from typing import Optional, Any

from .base import BaseBackend, DownloadResult
from flashtensors.utils import init_logger

logger = init_logger(__name__)


def _vllm_transform(input_dir: str, model_path: str, torch_dtype: str, return_dict: dict):
    try:       
        logger.info(f"Starting isolated VLLM transformation in process {os.getpid()}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Target model path: {model_path}")
        logger.info(f"PyTorch dtype: {torch_dtype}")
        
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Created target directory: {model_path}")
        
        import sys
        sys.path.append(os.path.dirname(__file__))  # Ensure imports work in subprocess
        
        from vllm import LLM
        
        logger.info(f"Loading model from {input_dir}")
        llm_writer = LLM(
            model=input_dir,
            download_dir=input_dir,
            dtype=torch_dtype,
            tensor_parallel_size=1,
            num_gpu_blocks_override=1,
            enforce_eager=True,
            max_model_len=1,
        )
        
        model_executor = llm_writer.llm_engine.model_executor
        model_executor.save_llm_state(
            path=model_path,
            pattern=None,
            max_size=None
        )
        logger.info(f"Saved transformed model to {model_path}")
        
        for file in os.listdir(input_dir):
            if os.path.splitext(file)[1] not in (
                ".bin", 
                ".pt", 
                ".safetensors"
            ):
                src_path = os.path.join(input_dir, file)
                dest_path = os.path.join(model_path, file)
                logger.info(f"Copying {src_path} to {dest_path}")
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copy(src_path, dest_path)
          
        del model_executor
        del llm_writer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return_dict['success'] = True
        return_dict['error'] = None
        logger.info("Isolated VLLM transformation completed successfully")
        
    except Exception as e:
        return_dict['success'] = False
        return_dict['error'] = str(e)
        logger.error(f"Isolated VLLM transformation failed: {e}")


class VLLMBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "vllm"
    
    async def download_model(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        torch_dtype: str = "auto",
        force: bool = False,
        **kwargs
    ) -> DownloadResult:
        import time
        start_time = time.time()
        
        logger.info(f"Downloading VLLM model {model_id}")
        model_path = os.path.join(self.name, model_id)  # Relative path for storage
        full_model_path = os.path.join(self.storage_path, self.name, model_id)  # Absolute path for files
        
        if os.path.exists(full_model_path) and not force:
            logger.info(f"Model {model_id} already exists at {full_model_path}")
            return DownloadResult(
                success=True,
                model_path=model_path,
                total_time=0.0,
                model_size=self._get_directory_size(full_model_path)
            )
        
        os.makedirs(full_model_path, exist_ok=True)
        
        cache_dir = TemporaryDirectory()
        try:
            download_start = time.time()
            logger.info(f"Downloading model {model_id} from HuggingFace...")
            
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                return DownloadResult(
                    success=False,
                    model_path=model_path,
                    error="huggingface_hub not available. Please install: pip install huggingface_hub"
                )
            
            input_dir = snapshot_download(
                model_id,
                cache_dir=cache_dir.name,
                token=hf_token,
                allow_patterns=[
                    "*.safetensors",
                    "*.bin", 
                    "*.json",
                    "*.txt",
                ],
            )
            download_time = time.time() - download_start
            logger.info(f"Downloaded model in {download_time:.2f}s")
            
            transform_start = time.time()
            logger.info(f"Transforming model to fast-loading format...")

            return_dict = {}
            _vllm_transform(input_dir, full_model_path, torch_dtype, return_dict)
            
            if not return_dict.get('success', False):
                raise Exception(return_dict.get('error', 'Unknown transformation error'))
            
            transform_time = time.time() - transform_start
            total_time = time.time() - start_time
            model_size = self._get_directory_size(full_model_path)
            
            logger.info(f"Model {model_id} downloaded and transformed successfully in {total_time:.2f}s")
            
            return DownloadResult(
                success=True,
                model_path=model_path,
                download_time=download_time,
                transform_time=transform_time,
                total_time=total_time,
                model_size=model_size
            )
            
        except Exception as e:
            logger.error(f"Failed to download/transform model {model_id}: {e}")
            if os.path.exists(full_model_path):
                shutil.rmtree(full_model_path)
            
            return DownloadResult(
                success=False,
                model_path=model_path,
                error=str(e),
                total_time=time.time() - start_time
            )
        finally:
            cache_dir.cleanup()
    
    def load_model(
        self,
        model_id: str,
        dtype: str = "float16",
        gpu_memory_utilization: Optional[float] = None,
        **kwargs
    ) -> Any:
        from flashtensors.config import get_config
        
        config = get_config()
        if gpu_memory_utilization is None:
            gpu_memory_utilization = config["gpu_memory_utilization"]
        
        try:
            from vllm import LLM
            
            model_path = os.path.join(self.storage_path, self.name, model_id)
            logger.info(f"Looking for VLLM model at: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"VLLM model not found at {model_path}. "
                    f"Please run flash.register_model('{model_id}', backend='vllm') first."
                )
            
            logger.info(f"Using VLLM model from: {model_path}")
            
            llm = LLM(
                model=model_path,
                load_format="flashtensors",
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                enforce_eager=True
            )
            
            logger.info(f"✅ VLLM model {model_id} loaded successfully")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to load VLLM model {model_id}: {e}")
            raise
    
    def is_model_directory(self, path: str) -> bool:
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path) and item.startswith("rank_"):
                    return True
            return False
        except OSError:
            return False
    
    def count_ranks(self, model_path: str) -> int:
        try:
            ranks = 0
            for item in os.listdir(model_path):
                item_path = os.path.join(model_path, item)
                if os.path.isdir(item_path) and item.startswith("rank_"):
                    ranks += 1
            return ranks if ranks > 0 else 1
        except OSError:
            return 1
    
    def register_with_storage(self, storage_client, model_path: str) -> bool:
        logger.info(f"Registering VLLM model with storage system...")
        
        # TODO: Support configurable tensor_parallel_size
        tensor_parallel_size = 1
        
        for rank in range(tensor_parallel_size):
            model_rank_path = os.path.join(model_path, f"rank_{rank}")
            response = storage_client.register_model(model_rank_path)
            if not response.success:
                logger.error(f"Failed to register rank {rank}: {response.error}")
                return False
            logger.info(f"Registered VLLM model rank {rank} with path: {model_rank_path}")
        
        return True
