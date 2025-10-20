import os
import shutil
from typing import Optional, Any, Tuple

from .base import BaseBackend, DownloadResult
from flashtensors.utils import init_logger

logger = init_logger(__name__)

class TransformersBackend(BaseBackend):
    @property
    def name(self) -> str:
        return "transformers"
    #TODO: vllm_backend and this one does not have the same parameters, should we unified them somehow??
    async def download_model(
        self,
        model_id: str,
        hf_token: Optional[str] = None,
        torch_dtype = None,
        force: bool = False,
        **kwargs
    ) -> DownloadResult:
        import time
        from tempfile import TemporaryDirectory
        from transformers import AutoModelForCausalLM
        
        start_time = time.time()
        
        logger.info(f"Downloading and preparing transformers model {model_id} for fast loading")
        model_path = os.path.join(self.name, model_id)  # Relative path
        full_model_path = os.path.join(self.storage_path, self.name, model_id)  # Absolute path
        
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
            
            temp_model_path = snapshot_download(
                model_id,
                cache_dir=cache_dir.name,
                token=hf_token,
            )
            download_time = time.time() - download_start
            logger.info(f"Downloaded model in {download_time:.2f}s")
            
            transform_start = time.time()
            logger.info(f"Transforming model to fast-loading format...")
            
            from flashtensors.integrations.transformers import save_model
            from transformers import AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                temp_model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(temp_model_path)
            tokenizer.save_pretrained(full_model_path)

            save_model(model, full_model_path)
            
            del model
            del tokenizer
            import gc
            gc.collect()
            
            transform_time = time.time() - transform_start
            total_time = time.time() - start_time
            model_size = self._get_directory_size(full_model_path)
            
            logger.info(f"Model {model_id} downloaded and transformed for fast loading in {total_time:.2f}s")
            
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
        torch_dtype = None,
        device_map: str = "auto",
        hf_model_class: str = "AutoModelForCausalLM",
        quantization_config=None,
        **kwargs
    ) -> Tuple[Any, Any]:
        try:
            from transformers import AutoTokenizer
            from flashtensors.integrations.transformers import load_model as fast_load_model
            
            relative_model_path = os.path.join(self.name, model_id)
            full_model_path = os.path.join(self.storage_path, self.name, model_id)
            
            logger.info(f"🚀 Fast loading transformers model: {model_id}")
            
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(
                    f"Transformers model not found at {full_model_path}. "
                    f"Please run flash.register_model('{model_id}', backend='transformers') first."
                )
            
            if not os.path.exists(os.path.join(full_model_path, "tensor_index.json")):
                raise ValueError(
                    f"Model {model_id} is not in flashtensors fast-loading format. "
                    f"Please re-register with flash.register_model('{model_id}', backend='transformers', force=True)"
                )
            
            model = fast_load_model(
                model_path=relative_model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                storage_path=self.storage_path,
                fully_parallel=True,
                hf_model_class=hf_model_class,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(full_model_path)
            
            logger.info(f"✅ Transformers model {model_id} loaded with fast loading")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to fast load transformers model {model_id}: {e}")
            raise
    
    def is_model_directory(self, path: str) -> bool:
        try:
            return (
                os.path.isfile(os.path.join(path, "tensor_index.json")) and 
                os.path.isfile(os.path.join(path, "config.json")) and
                os.path.isfile(os.path.join(path, "tokenizer_config.json"))
            )
        except OSError:
            return False

    def register_with_storage(self, storage_client, model_path: str) -> bool:
        logger.info(f"Registering transformers model with storage system...")
        response = storage_client.register_model(model_path)
        if not response.success:
            logger.error(f"Failed to register model: {response.error}")
            return False
        logger.info(f"Registered transformers model with path: {model_path}")
        return True
