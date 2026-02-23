"""
Monkey-patch vLLM to load weights from flashtensors checkpoints.

Usage:
    import flashtensors.vllm  # applies patch on import

    # Then use vLLM normally — any model directory containing
    # tensor_index.json + tensor.flashtensors will be loaded
    # via flashtensors instead of safetensors.

The model directory should contain the usual HuggingFace config files
(config.json, tokenizer, etc.) alongside the flashtensors files.
Use an external converter to produce the flashtensors checkpoint.
"""

import os
import time
import logging

logger = logging.getLogger("flashtensors.vllm")

_patched = False


def patch():
    """Monkey-patch DefaultModelLoader to use flashtensors when a
    flashtensors checkpoint is present in the model directory.

    Patches two methods:
    - ``load_weights``: detects flashtensors files and loads tensors
      directly onto the model's device (e.g. cuda:0), so the subsequent
      ``param.data.copy_()`` is a fast device-to-device transfer.
    - ``_get_weights_iterator``: fallback path is untouched.
    """
    global _patched
    if _patched:
        return
    _patched = True

    from vllm.model_executor.model_loader.default_loader import (
        DefaultModelLoader,
    )

    _original_load_weights = DefaultModelLoader.load_weights

    def _patched_load_weights(self, model, model_config):
        model_path = model_config.model
        index_path = os.path.join(model_path, "tensor_index.json")

        if not os.path.isdir(model_path) or not os.path.isfile(index_path):
            return _original_load_weights(self, model, model_config)

        from flashtensors import load_dict

        # Infer target device from the model's existing parameters
        target_device = next(model.parameters()).device
        device_str = str(target_device)

        logger.info(
            "Loading weights from flashtensors checkpoint: %s -> %s",
            model_path,
            device_str,
        )

        self.counter_before_loading_weights = time.perf_counter()

        tensors = load_dict(model_path, device_map={"": device_str})

        def _weight_iterator():
            for name, tensor in tensors.items():
                yield name, tensor

        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(_weight_iterator())

        self.counter_after_loading_weights = time.perf_counter()
        logger.info(
            "Loading weights took %.2f seconds",
            self.counter_after_loading_weights
            - self.counter_before_loading_weights,
        )

        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

    DefaultModelLoader.load_weights = _patched_load_weights
    logger.info("flashtensors: vLLM monkey-patch applied")


# Auto-patch on import
patch()
