"""Tests for the vLLM monkey-patch in flashtensors.vllm.

Tests verify:
  - The patch correctly replaces DefaultModelLoader.load_weights
  - When flashtensors files exist, the patched path is taken
  - When no flashtensors files exist, the original vLLM path is used
  - Tensors are loaded onto the correct device
  - The weight iterator yields all expected (name, tensor) pairs
"""

import json
import os
import tempfile
from unittest import mock

import pytest
import torch
import torch.nn as nn

from flashtensors import save_dict


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def checkpoint(tmpdir):
    """Create a flashtensors checkpoint with a small model's weights."""
    state_dict = {
        "linear.weight": torch.randn(8, 16),
        "linear.bias": torch.randn(8),
    }
    save_dict(state_dict, tmpdir)
    return tmpdir, state_dict


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

class TestPatchApplication:
    def test_patch_replaces_load_weights(self):
        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )
        import flashtensors.vllm

        assert "_patched" in DefaultModelLoader.load_weights.__qualname__

    def test_patch_is_idempotent(self):
        import flashtensors.vllm

        flashtensors.vllm._patched = False
        flashtensors.vllm.patch()
        flashtensors.vllm.patch()  # second call should be a no-op

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )
        assert "_patched" in DefaultModelLoader.load_weights.__qualname__


# ---------------------------------------------------------------------------
# Fallback to original
# ---------------------------------------------------------------------------

class TestFallback:
    def test_falls_back_when_no_flashtensors_files(self, tmpdir):
        """When there are no flashtensors files, the original vLLM
        load_weights should be called."""
        import flashtensors.vllm

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        mock_model_config = mock.MagicMock()
        mock_model_config.model = tmpdir  # dir exists but no flashtensors files
        mock_model_config.quantization = None

        mock_load_config = mock.MagicMock()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock_load_config
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        mock_model = mock.MagicMock()

        # The original load_weights should be called, which will try to
        # prepare weights and fail (no real model). We catch that.
        with pytest.raises(Exception):
            loader.load_weights(mock_model, mock_model_config)

    def test_falls_back_when_path_is_not_dir(self):
        """When model path is not a directory (e.g. a HF model ID),
        the original load_weights should be called, not the flashtensors path."""
        import flashtensors.vllm

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        mock_model_config = mock.MagicMock()
        mock_model_config.model = "meta-llama/Llama-2-7b"  # not a local dir

        mock_load_config = mock.MagicMock()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock_load_config
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        mock_model = mock.MagicMock()

        # Verify that load_dict is NOT called (original path is taken)
        with mock.patch("flashtensors.load_dict") as mock_ld:
            try:
                loader.load_weights(mock_model, mock_model_config)
            except Exception:
                pass  # original path may fail, that's fine
            mock_ld.assert_not_called()


# ---------------------------------------------------------------------------
# Patched loading path
# ---------------------------------------------------------------------------

class TestPatchedLoading:
    def test_loads_flashtensors_checkpoint(self, checkpoint):
        """When flashtensors files exist, load_dict should be called
        and model.load_weights should receive (name, tensor) pairs."""
        tmpdir, orig_state_dict = checkpoint

        import flashtensors.vllm

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        mock_model_config = mock.MagicMock()
        mock_model_config.model = tmpdir
        mock_model_config.quantization = None

        # Track what model.load_weights receives
        received_weights = {}

        def fake_load_weights(weights_iter):
            for name, tensor in weights_iter:
                received_weights[name] = tensor
            return set(received_weights.keys())

        mock_model = mock.MagicMock()
        mock_model.load_weights.side_effect = fake_load_weights
        # Make named_parameters return the expected names
        mock_model.named_parameters.return_value = [
            (name, torch.empty(1)) for name in orig_state_dict
        ]
        # Device for target detection
        param = torch.nn.Parameter(torch.empty(1))  # CPU
        mock_model.parameters.return_value = iter([param])

        mock_load_config = mock.MagicMock()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock_load_config
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        loader.load_weights(mock_model, mock_model_config)

        # Verify all weights were yielded
        assert set(received_weights.keys()) == set(orig_state_dict.keys())

        # Verify values match
        for name, orig in orig_state_dict.items():
            assert torch.allclose(orig, received_weights[name]), \
                f"value mismatch for {name}"

    def test_device_map_uses_model_device(self, checkpoint):
        """Verify that load_dict is called with the model's device."""
        tmpdir, _ = checkpoint

        import flashtensors.vllm

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        mock_model_config = mock.MagicMock()
        mock_model_config.model = tmpdir
        mock_model_config.quantization = None

        mock_model = mock.MagicMock()
        mock_model.load_weights.return_value = set()
        mock_model.named_parameters.return_value = []

        # Simulate model on cuda:0
        param = mock.MagicMock()
        param.device = torch.device("cuda:0")
        mock_model.parameters.return_value = iter([param])

        mock_load_config = mock.MagicMock()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock_load_config
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        # Patch load_dict where the patched closure imports it
        with mock.patch("flashtensors.load_dict", return_value={}) as mock_ld:
            loader.load_weights(mock_model, mock_model_config)
            mock_ld.assert_called_once()
            _, kwargs = mock_ld.call_args
            assert kwargs["device_map"] == {"": "cuda:0"}

    def test_timing_counters_are_set(self, checkpoint):
        """Verify that performance counters are updated."""
        tmpdir, _ = checkpoint

        import flashtensors.vllm

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        mock_model_config = mock.MagicMock()
        mock_model_config.model = tmpdir
        mock_model_config.quantization = None

        mock_model = mock.MagicMock()
        mock_model.load_weights.return_value = set()
        mock_model.named_parameters.return_value = []
        param = torch.nn.Parameter(torch.empty(1))
        mock_model.parameters.return_value = iter([param])

        mock_load_config = mock.MagicMock()
        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock_load_config
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        loader.load_weights(mock_model, mock_model_config)

        assert loader.counter_before_loading_weights > 0
        assert loader.counter_after_loading_weights > 0
        assert loader.counter_after_loading_weights >= loader.counter_before_loading_weights
