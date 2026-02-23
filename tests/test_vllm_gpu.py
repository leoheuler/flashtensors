"""GPU test for the vLLM monkey-patch, run via Modal.

Verifies that the flashtensors vLLM patch correctly loads weights
onto GPU via the patched DefaultModelLoader.load_weights path.

Run with:
    modal run tests/test_vllm_gpu.py
"""

import modal

app = modal.App("flashtensors-vllm-gpu-test")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy", "vllm")
    .add_local_dir("/workspaces/flashtensors/flashtensors", remote_path="/pkg/flashtensors")
)


def _run_tests():
    import sys
    sys.path.insert(0, "/pkg")

    import os
    import tempfile
    from unittest import mock

    import torch
    import torch.nn as nn

    from flashtensors import save_dict

    results = []

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results.append((name, status, detail))
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    print("\n=== flashtensors vLLM GPU patch tests ===\n")

    # --- Test 1: Patched load_weights delivers tensors on GPU ---
    print("-- Patched load_weights with GPU model --")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dict = {
            "weight": torch.randn(32, 64),
            "bias": torch.randn(32),
        }
        save_dict(state_dict, tmpdir)

        import flashtensors.vllm

        from vllm.model_executor.model_loader.default_loader import (
            DefaultModelLoader,
        )

        mock_model_config = mock.MagicMock()
        mock_model_config.model = tmpdir
        mock_model_config.quantization = None

        # Track what tensors model.load_weights receives
        received = {}

        def fake_load_weights(weights_iter):
            for name, tensor in weights_iter:
                received[name] = tensor
            return set(received.keys())

        mock_model = mock.MagicMock()
        mock_model.load_weights.side_effect = fake_load_weights
        mock_model.named_parameters.return_value = [
            (name, torch.empty(1)) for name in state_dict
        ]

        # Create a real parameter on GPU to simulate model device
        gpu_param = torch.nn.Parameter(torch.empty(1, device="cuda:0"))
        mock_model.parameters.return_value = iter([gpu_param])

        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock.MagicMock()
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        loader.load_weights(mock_model, mock_model_config)

        # Check all tensors arrived
        check(
            "all_weights_loaded",
            set(received.keys()) == set(state_dict.keys()),
            f"expected {set(state_dict.keys())}, got {set(received.keys())}",
        )

        # Check tensors are on GPU
        all_on_gpu = all(t.device.type == "cuda" for t in received.values())
        check(
            "tensors_on_gpu",
            all_on_gpu,
            f"devices: {[str(t.device) for t in received.values()]}",
        )

        # Check values match
        all_match = all(
            torch.allclose(state_dict[n].float(), received[n].float().cpu())
            for n in state_dict
        )
        check("values_correct", all_match)

    # --- Test 2: Multiple dtypes on GPU ---
    print("\n-- Multiple dtypes on GPU --")
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dict = {
            "fp32": torch.randn(8, 8, dtype=torch.float32),
            "fp16": torch.randn(8, 8, dtype=torch.float16),
            "bf16": torch.randn(8, 8, dtype=torch.bfloat16),
        }
        save_dict(state_dict, tmpdir)

        mock_model_config = mock.MagicMock()
        mock_model_config.model = tmpdir
        mock_model_config.quantization = None

        received = {}

        def fake_load_weights2(weights_iter):
            for name, tensor in weights_iter:
                received[name] = tensor
            return set(received.keys())

        mock_model = mock.MagicMock()
        mock_model.load_weights.side_effect = fake_load_weights2
        mock_model.named_parameters.return_value = [
            (name, torch.empty(1)) for name in state_dict
        ]
        gpu_param = torch.nn.Parameter(torch.empty(1, device="cuda:0"))
        mock_model.parameters.return_value = iter([gpu_param])

        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        loader.load_config = mock.MagicMock()
        loader.counter_before_loading_weights = 0.0
        loader.counter_after_loading_weights = 0.0

        loader.load_weights(mock_model, mock_model_config)

        for name, orig in state_dict.items():
            loaded = received[name]
            on_gpu = loaded.device.type == "cuda"
            dtype_ok = loaded.dtype == orig.dtype
            values_ok = torch.allclose(orig.float(), loaded.float().cpu())
            check(
                f"dtype/{name}",
                on_gpu and dtype_ok and values_ok,
                f"device={loaded.device} dtype={loaded.dtype} match={values_ok}",
            )

    # --- Test 3: Timing counters ---
    print("\n-- Timing counters --")
    check(
        "timing_before_set",
        loader.counter_before_loading_weights > 0,
        f"before={loader.counter_before_loading_weights}",
    )
    check(
        "timing_after_set",
        loader.counter_after_loading_weights >= loader.counter_before_loading_weights,
        f"after={loader.counter_after_loading_weights}",
    )

    # --- summary ---
    print("\n=== Summary ===")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"{passed} passed, {failed} failed")
    if failed:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  {name}: {detail}")

    return {"passed": passed, "failed": failed, "results": results}


@app.function(gpu="L4", image=image, timeout=300)
def run_vllm_gpu_tests():
    return _run_tests()


@app.local_entrypoint()
def main():
    result = run_vllm_gpu_tests.remote()
    if result["failed"] > 0:
        raise SystemExit(f"{result['failed']} vLLM GPU test(s) failed")
