"""CPU tests for the flashtensors save/load pipeline.

Tests the full round-trip: save_dict → load_dict with device_map {"": "cpu"}.
No GPU required.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from flashtensors import load_dict, save_dict


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round_trip(state_dict, tmpdir):
    save_dict(state_dict, tmpdir)
    return load_dict(tmpdir, {"": "cpu"})


def assert_tensors_equal(orig, loaded):
    assert orig.shape == loaded.shape, f"shape mismatch: {orig.shape} vs {loaded.shape}"
    assert orig.dtype == loaded.dtype, f"dtype mismatch: {orig.dtype} vs {loaded.dtype}"
    assert torch.allclose(orig.float(), loaded.float()), "values do not match"


# ---------------------------------------------------------------------------
# Format / file tests
# ---------------------------------------------------------------------------

class TestFormat:
    def test_files_created(self, tmpdir):
        save_dict({"w": torch.randn(2, 2)}, tmpdir)
        assert "tensor.flashtensors" in os.listdir(tmpdir)
        assert "tensor_index.json" in os.listdir(tmpdir)

    def test_index_schema(self, tmpdir):
        state_dict = {"fc.weight": torch.randn(4, 8), "fc.bias": torch.randn(4)}
        save_dict(state_dict, tmpdir)

        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            index = json.load(f)

        assert "layout" in index
        assert len(index["layout"]) == 2

        ref = index["layout"][0]
        for field in ("name", "offset", "size", "shape", "stride", "dtype"):
            assert field in ref, f"missing field: {field}"

    def test_offsets_are_sequential(self, tmpdir):
        state_dict = {
            "a": torch.randn(4, 4),
            "b": torch.randn(8),
            "c": torch.randn(2, 2),
        }
        save_dict(state_dict, tmpdir)

        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            layout = json.load(f)["layout"]

        for i in range(1, len(layout)):
            prev = layout[i - 1]
            curr = layout[i]
            assert curr["offset"] == prev["offset"] + prev["size"]

    def test_binary_size_matches_index(self, tmpdir):
        state_dict = {"w": torch.randn(16, 16)}
        save_dict(state_dict, tmpdir)

        binary_size = os.path.getsize(os.path.join(tmpdir, "tensor.flashtensors"))

        with open(os.path.join(tmpdir, "tensor_index.json")) as f:
            layout = json.load(f)["layout"]

        last = layout[-1]
        expected = last["offset"] + last["size"]
        assert binary_size == expected


# ---------------------------------------------------------------------------
# Dtype round-trips
# ---------------------------------------------------------------------------

class TestDtypes:
    def test_float32(self, tmpdir):
        t = torch.randn(4, 4, dtype=torch.float32)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_float16(self, tmpdir):
        t = torch.randn(4, 4, dtype=torch.float16)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_bfloat16(self, tmpdir):
        t = torch.randn(4, 4, dtype=torch.bfloat16)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_float64(self, tmpdir):
        t = torch.randn(4, 4, dtype=torch.float64)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_int8(self, tmpdir):
        t = torch.randint(-128, 127, (4, 4), dtype=torch.int8)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_int32(self, tmpdir):
        t = torch.randint(0, 1000, (4, 4), dtype=torch.int32)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_int64(self, tmpdir):
        t = torch.randint(0, 10**9, (4, 4), dtype=torch.int64)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_bool(self, tmpdir):
        t = torch.randint(0, 2, (4, 4)).bool()
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_mixed_dtypes(self, tmpdir):
        state_dict = {
            "fp32": torch.randn(4, 4),
            "fp16": torch.randn(4, 4, dtype=torch.float16),
            "bf16": torch.randn(4, 4, dtype=torch.bfloat16),
            "i32":  torch.randint(0, 100, (4,), dtype=torch.int32),
        }
        loaded = round_trip(state_dict, tmpdir)
        for name, orig in state_dict.items():
            assert_tensors_equal(orig, loaded[name])


# ---------------------------------------------------------------------------
# Shape / stride tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_1d(self, tmpdir):
        t = torch.randn(128)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_2d(self, tmpdir):
        t = torch.randn(32, 64)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_3d(self, tmpdir):
        t = torch.randn(4, 8, 16)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_scalar_tensor(self, tmpdir):
        t = torch.tensor(3.14)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_single_element(self, tmpdir):
        t = torch.tensor([42.0])
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])

    def test_large_tensor(self, tmpdir):
        t = torch.randn(512, 512)
        loaded = round_trip({"w": t}, tmpdir)
        assert_tensors_equal(t, loaded["w"])


# ---------------------------------------------------------------------------
# Device map tests
# ---------------------------------------------------------------------------

class TestDeviceMap:
    def test_default_key(self, tmpdir):
        state_dict = {"a": torch.randn(4), "b": torch.randn(4)}
        loaded = round_trip(state_dict, tmpdir)
        for name in state_dict:
            assert loaded[name].device.type == "cpu"

    def test_explicit_cpu_string(self, tmpdir):
        state_dict = {"w": torch.randn(4, 4)}
        save_dict(state_dict, tmpdir)
        loaded = load_dict(tmpdir, {"": "cpu"})
        assert loaded["w"].device.type == "cpu"

    def test_per_tensor_device_map(self, tmpdir):
        state_dict = {"a": torch.randn(4), "b": torch.randn(4)}
        save_dict(state_dict, tmpdir)
        loaded = load_dict(tmpdir, {"a": "cpu", "b": "cpu"})
        assert_tensors_equal(state_dict["a"], loaded["a"])
        assert_tensors_equal(state_dict["b"], loaded["b"])

    def test_fallback_to_cpu_when_no_key(self, tmpdir):
        state_dict = {"w": torch.randn(4)}
        save_dict(state_dict, tmpdir)
        # Empty device_map — should fall back to cpu
        loaded = load_dict(tmpdir, {})
        assert loaded["w"].device.type == "cpu"


# ---------------------------------------------------------------------------
# Model round-trips
# ---------------------------------------------------------------------------

class TestModels:
    def test_linear(self, tmpdir):
        model = nn.Linear(64, 128)
        sd = model.state_dict()
        loaded = round_trip(sd, tmpdir)
        for name, orig in sd.items():
            assert_tensors_equal(orig, loaded[name])

    def test_mlp(self, tmpdir):
        model = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        sd = model.state_dict()
        loaded = round_trip(sd, tmpdir)
        for name, orig in sd.items():
            assert_tensors_equal(orig, loaded[name])

    def test_embedding(self, tmpdir):
        model = nn.Embedding(1000, 64)
        sd = model.state_dict()
        loaded = round_trip(sd, tmpdir)
        for name, orig in sd.items():
            assert_tensors_equal(orig, loaded[name])

    def test_conv2d(self, tmpdir):
        model = nn.Conv2d(3, 64, kernel_size=3)
        sd = model.state_dict()
        loaded = round_trip(sd, tmpdir)
        for name, orig in sd.items():
            assert_tensors_equal(orig, loaded[name])

    def test_layernorm(self, tmpdir):
        model = nn.LayerNorm(256)
        sd = model.state_dict()
        loaded = round_trip(sd, tmpdir)
        for name, orig in sd.items():
            assert_tensors_equal(orig, loaded[name])

    def test_loaded_tensors_are_standalone(self, tmpdir):
        """Tensors returned by load_dict should not hold references to mmap."""
        sd = {"w": torch.randn(64, 64)}
        save_dict(sd, tmpdir)
        loaded = load_dict(tmpdir, {"": "cpu"})
        # Modifying the loaded tensor should not raise
        loaded["w"].add_(1.0)

    def test_inference_matches(self, tmpdir):
        """End-to-end: model loaded from flashtensors should produce same output."""
        torch.manual_seed(0)
        model = nn.Linear(16, 8)
        x = torch.randn(4, 16)
        expected = model(x)

        sd = model.state_dict()
        save_dict(sd, tmpdir)
        loaded_sd = load_dict(tmpdir, {"": "cpu"})

        model2 = nn.Linear(16, 8)
        model2.load_state_dict(loaded_sd)
        actual = model2(x)

        assert torch.allclose(expected, actual), "inference output mismatch after load"
