# Derived from https://github.com/ServerlessLLM/ServerlessLLM/blob/main/sllm_store/vllm_patch/patch.sh
# Original patch is command line triggered. Our current patch is automatically
# added into the inference engine code. This enables easier compatibility maintainance.
#
# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #

from abc import abstractmethod
import collections
import gc
import os
from typing import Dict, Optional, Literal, Union

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig

from vllm.model_executor.model_loader.utils import (
    set_default_torch_dtype,
    initialize_model,
)
from vllm.model_executor.model_loader import (
    BaseModelLoader,
    get_model_loader,
    LoadFormats,
)
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.executor.uniproc_executor import UniProcExecutor
from vllm.worker.worker import Worker
from vllm.model_executor import model_loader
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.engine.arg_utils import EngineArgs

from flashtensors.torch_storage import save_dict, load_dict
from flashtensors.config import get_storage_path
from flashtensors.utils.logger import init_logger

logger = init_logger(__name__)


class FlashLLMLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )

        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        logger.debug("Filtering subtensors")
        same_storage_groups = collections.defaultdict(list)
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        logger.debug("Starting tensor filtering process")
        result = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 > k:
                        # Same tensors, keep the one with the longer key.
                        break
                else:
                    result[k] = t
        return result

    def load_model(
        self, *, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        logger.info("🚀 Starting FlashLLMLoader.load_model")
        from vllm.distributed import get_tensor_model_parallel_rank

        assert os.path.isdir(vllm_config.model_config.model)

        rank = get_tensor_model_parallel_rank()
        logger.debug(f"Tensor model parallel rank: {rank}")

        local_model_path = vllm_config.model_config.model
        local_model_path = os.path.join(local_model_path, f"rank_{rank}")
        logger.debug(f"Local model path: {local_model_path}")

        def remove_prefix(path, prefix):
            # Normalize the paths to ensure consistency across different platforms
            path = os.path.normpath(path)
            prefix = os.path.normpath(prefix)

            # Check if the path starts with the prefix
            if path.startswith(prefix):
                # Return the path without the prefix
                return path[len(prefix) :].lstrip(os.sep)

            # Return the original path if the prefix doesn't exist
            return path

        # vLLM needs a local model path to read model config but
        # FlashLLMLoader Store requires a global model path as the model ID
        storage_path = get_storage_path()
        model_path = remove_prefix(local_model_path, storage_path)

        logger.debug(f"Storage path: {storage_path}")
        logger.debug(f"Local model path: {local_model_path}")
        logger.debug(f"Model path after prefix removal: {model_path}")

        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with torch.device("cpu"):
                logger.debug("Initializing model on CPU")
                model = initialize_model(vllm_config=vllm_config)
                model = model.eval()
            # set all parameters to meta device
            state_dict = self._filter_subtensors(model.state_dict())
            logger.info(f"Filtered state dict with {len(state_dict)} tensors")
            key_list = list(state_dict.keys())

            logger.debug("Setting model parameters to empty CUDA tensors")
            for key, param in model.named_parameters(recurse=True):
                if key in key_list:
                    param.data = torch.empty(1, device="cuda")
            gc.collect()

            device_id = torch.cuda.current_device()
            device_map = {"": device_id}
            logger.debug(f"Device map: {device_map}")

            # Check if model_path is None or empty
            if not model_path:
                raise ValueError(
                    f"Model path is empty or None: '{model_path}'. Local path: {local_model_path}, Storage path: {storage_path}"
                )

            logger.info(f"🔄 Loading tensors from model_path='{model_path}'")
            # Note: we need to pass storage_path to load_dict for proper path resolution
            sllm_state_dict = load_dict(model_path, device_map, storage_path)
            logger.info(
                f"✅ Successfully loaded {len(sllm_state_dict)} tensors from storage"
            )

            for key, param in model.named_parameters(recurse=True):
                if key in key_list:
                    tensor = sllm_state_dict[key]
                    param.data = tensor
                    state_dict.pop(key)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

        return model

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        state_dict = FlashLLMLoader._filter_subtensors(model.state_dict())

        # move all tensors to CPU
        for key, tensor in state_dict.items():
            state_dict[key] = tensor.cpu().contiguous()

        save_path = os.path.join(path, f"rank_{rank}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logger.debug("Saving model")

        save_dict(state_dict, save_path)


def patch_model_loader(load_config: LoadConfig):
    if load_config.load_format == "flash":
        return FlashLLMLoader(load_config=load_config)

    return get_model_loader(load_config)


@abstractmethod
def save_llm_state(
    self,
    path: str,
    pattern: Optional[str] = None,
    max_size: Optional[int] = None,
) -> None:
    """Save the model state in FlashLLMLoader format."""
    raise NotImplementedError("Save LLM state not implemented.")


def save_llm_state_impl(
    self,
    path: str,
    pattern: Optional[str] = None,
    max_size: Optional[int] = None,
) -> None:
    self._run_workers(
        "save_llm_state",
        path=path,
        pattern=pattern,
        max_size=max_size,
    )


def save_llm_state_worker(
    self,
    path: str,
    pattern: Optional[str] = None,
    max_size: Optional[int] = None,
) -> None:
    self.model_runner.save_llm_state(
        path=path,
        pattern=pattern,
        max_size=max_size,
    )


def save_llm_state_runner(
    self,
    path: str,
    pattern: Optional[str] = None,
    max_size: Optional[int] = None,
) -> None:
    """Save the model state in FlashLLMLoader format."""
    FlashLLMLoader.save_model(
        self.model,
        path,
        pattern=pattern,
        max_size=max_size,
    )


def save_llm_state_executor_uniproc(
    self,
    path: str,
    pattern: Optional[str] = None,
    max_size: Optional[int] = None,
) -> None:
    self.collective_rpc(
        "save_llm_state",
        args=(path, pattern, max_size),
    )


def activate():
    os.environ["VLLM_USE_V1"] = "0"
    ExtendedLoadFormats = Union[LoadFormats, Literal["flash"]]

    # _original_annotations_load_config = LoadConfig.__annotations__.copy()
    LoadConfig.__annotations__["load_format"] = Union[str, ExtendedLoadFormats]
    EngineArgs.__annotations__["load_format"] = Union[str, ExtendedLoadFormats]

    model_loader.get_model_loader = patch_model_loader
    setattr(ExecutorBase, "save_llm_state", save_llm_state)
    setattr(MultiprocessingDistributedExecutor, "save_llm_state", save_llm_state_impl)
    setattr(UniProcExecutor, "save_llm_state", save_llm_state_executor_uniproc)
    setattr(Worker, "save_llm_state", save_llm_state_worker)
    setattr(GPUModelRunnerBase, "save_llm_state", save_llm_state_runner)
    return
