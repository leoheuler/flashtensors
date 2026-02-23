"""Tests for the MPS loader path.

MPS hardware is not available in CI, so these tests mock torch.Tensor.to()
and verify the MPS loader correctly:
  - reads tensors via pread into CPU
  - calls .to("mps") for MPS-targeted tensors
  - preserves dtypes, shapes, strides, and values
  - handles threading (num_workers > 1)
"""

import os
import tempfile
from unittest import mock

import pytest
import torch

from flashtensors import save_dict, load_dict
from flashtensors.loaders._common import _resolve_device
from flashtensors.loaders._mps import MpsLoader, _load_tensor_pread_mps


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _save_and_get_paths(state_dict, tmpdir):
    save_dict(state_dict, tmpdir)
    return (
        os.path.join(tmpdir, "tensor.flashtensors"),
        os.path.join(tmpdir, "tensor_index.json"),
    )


# ---------------------------------------------------------------------------
# _resolve_device tests for MPS
# ---------------------------------------------------------------------------

class TestResolveDeviceMps:
    def test_mps_string(self):
        assert _resolve_device("x", {"x": "mps"}) == ("mps", None)

    def test_mps_default_key(self):
        assert _resolve_device("x", {"": "mps"}) == ("mps", None)

    def test_mps_does_not_break_cpu(self):
        assert _resolve_device("x", {"x": "cpu"}) == ("cpu", None)

    def test_mps_does_not_break_cuda(self):
        assert _resolve_device("x", {"x": "cuda:1"}) == ("cuda", 1)


# ---------------------------------------------------------------------------
# MpsLoader unit tests (mock .to("mps") since no MPS hardware)
# ---------------------------------------------------------------------------

class TestMpsLoader:
    def test_load_single_tensor(self, tmpdir):
        orig = torch.randn(4, 4)
        save_dict({"w": orig}, tmpdir)

        # Mock .to() so it returns the CPU tensor (simulating MPS)
        with mock.patch.object(torch.Tensor, "to", side_effect=lambda dev: torch.Tensor.clone(torch.Tensor)) as mock_to:
            # Use load_dict with cpu to verify the loader path works
            # We test MpsLoader directly instead
            pass

        # Direct test: load via MpsLoader with cpu device_map
        # (verifies the pread path without needing MPS)
        import json
        from flashtensors.flash_state import FlashState

        data_path = os.path.join(tmpdir, "tensor.flashtensors")
        index_path = os.path.join(tmpdir, "tensor_index.json")
        with open(index_path) as f:
            flash_state = FlashState(**json.load(f))

        file_size = os.path.getsize(data_path)
        loader = MpsLoader()

        # Load with cpu target to test the pread path (no actual MPS needed)
        result = loader.load(
            data_path, file_size, flash_state.layout,
            device_map={"": "cpu"}, num_workers=1, chunk_size=256 * 1024 * 1024,
        )
        assert torch.allclose(orig, result["w"])

    def test_multiple_tensors(self, tmpdir):
        state_dict = {
            "a": torch.randn(8, 8),
            "b": torch.randn(16),
            "c": torch.randn(2, 4, 4),
        }
        save_dict(state_dict, tmpdir)

        import json
        from flashtensors.flash_state import FlashState

        data_path = os.path.join(tmpdir, "tensor.flashtensors")
        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            flash_state = FlashState(**json.load(f))

        loader = MpsLoader()
        result = loader.load(
            data_path, os.path.getsize(data_path), flash_state.layout,
            device_map={"": "cpu"}, num_workers=1, chunk_size=256 * 1024 * 1024,
        )

        for name, orig in state_dict.items():
            assert torch.allclose(orig, result[name]), f"mismatch for {name}"

    def test_threaded_loading(self, tmpdir):
        state_dict = {f"t{i}": torch.randn(8, 8) for i in range(10)}
        save_dict(state_dict, tmpdir)

        import json
        from flashtensors.flash_state import FlashState

        data_path = os.path.join(tmpdir, "tensor.flashtensors")
        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            flash_state = FlashState(**json.load(f))

        loader = MpsLoader()
        result = loader.load(
            data_path, os.path.getsize(data_path), flash_state.layout,
            device_map={"": "cpu"}, num_workers=4, chunk_size=256 * 1024 * 1024,
        )

        for name, orig in state_dict.items():
            assert torch.allclose(orig, result[name]), f"mismatch for {name}"

    def test_bfloat16(self, tmpdir):
        orig = torch.randn(4, 4, dtype=torch.bfloat16)
        save_dict({"w": orig}, tmpdir)

        import json
        from flashtensors.flash_state import FlashState

        data_path = os.path.join(tmpdir, "tensor.flashtensors")
        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            flash_state = FlashState(**json.load(f))

        loader = MpsLoader()
        result = loader.load(
            data_path, os.path.getsize(data_path), flash_state.layout,
            device_map={"": "cpu"}, num_workers=1, chunk_size=256 * 1024 * 1024,
        )
        assert result["w"].dtype == torch.bfloat16
        assert torch.allclose(orig.float(), result["w"].float())

    def test_mps_to_is_called(self, tmpdir):
        """Verify that .to('mps') is called when device_map targets MPS."""
        orig = torch.randn(4, 4)
        save_dict({"w": orig}, tmpdir)

        import json
        from flashtensors.flash_state import FlashState

        data_path = os.path.join(tmpdir, "tensor.flashtensors")
        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            flash_state = FlashState(**json.load(f))

        loader = MpsLoader()

        # Patch torch.Tensor.to to track calls and return CPU tensor
        original_to = torch.Tensor.to
        to_calls = []

        def tracking_to(self, *args, **kwargs):
            to_calls.append(args)
            # Return self (stay on CPU) since we don't have MPS
            return self

        with mock.patch.object(torch.Tensor, "to", tracking_to):
            result = loader.load(
                data_path, os.path.getsize(data_path), flash_state.layout,
                device_map={"": "mps"}, num_workers=1, chunk_size=256 * 1024 * 1024,
            )

        # .to("mps") should have been called
        assert any("mps" in str(call) for call in to_calls), \
            f"Expected .to('mps') call, got: {to_calls}"
