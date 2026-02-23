"""GPU tests for the flashtensors save/load pipeline, run via Modal.

Tests the CuPy pinned-memory path: save_dict → load_dict with device_map {"": 0}.
Verifies correctness, dtype preservation, and that tensors land on the right device.

Run with:
    modal run tests/test_loader_gpu.py
"""

import modal

app = modal.App("flashtensors-loader-gpu-test")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy")
    .add_local_dir("/workspaces/flashtensors/flashtensors", remote_path="/pkg/flashtensors")
)


def _run_tests():
    import sys
    sys.path.insert(0, "/pkg")

    import json
    import os
    import tempfile

    import torch
    import torch.nn as nn

    from flashtensors import load_dict, save_dict

    results = []

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results.append((name, status, detail))
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    def round_trip(state_dict, tmpdir, device=0):
        save_dict(state_dict, tmpdir)
        return load_dict(tmpdir, {"": device})

    def tensors_equal(orig, loaded):
        return (
            orig.shape == loaded.shape
            and orig.dtype == loaded.dtype
            and torch.allclose(orig.float().cpu(), loaded.float().cpu())
        )

    print("\n=== flashtensors GPU loader tests ===\n")

    # --- dtype round-trips ---
    print("-- Dtypes --")
    dtypes = [
        ("float32", torch.randn(8, 8, dtype=torch.float32)),
        ("float16", torch.randn(8, 8, dtype=torch.float16)),
        ("bfloat16", torch.randn(8, 8, dtype=torch.bfloat16)),
        ("float64", torch.randn(8, 8, dtype=torch.float64)),
        ("int32",   torch.randint(0, 100, (8, 8), dtype=torch.int32)),
        ("int64",   torch.randint(0, 10**6, (8, 8), dtype=torch.int64)),
        ("int8",    torch.randint(-128, 127, (8, 8), dtype=torch.int8)),
    ]
    for dtype_name, t in dtypes:
        with tempfile.TemporaryDirectory() as d:
            loaded = round_trip({"w": t}, d)["w"]
            on_gpu = loaded.device.type == "cuda"
            values_ok = tensors_equal(t, loaded)
            check(f"dtype/{dtype_name}", on_gpu and values_ok,
                  f"device={loaded.device} values_match={values_ok}")

    # --- shapes ---
    print("\n-- Shapes --")
    shapes = [
        ("scalar", torch.tensor(3.14)),
        ("1d",     torch.randn(256)),
        ("2d",     torch.randn(32, 64)),
        ("3d",     torch.randn(4, 8, 16)),
        ("4d",     torch.randn(2, 3, 8, 8)),
    ]
    for shape_name, t in shapes:
        with tempfile.TemporaryDirectory() as d:
            loaded = round_trip({"w": t}, d)["w"]
            check(f"shape/{shape_name}", tensors_equal(t, loaded),
                  f"shape={tuple(loaded.shape)} device={loaded.device}")

    # --- device map variants ---
    print("\n-- Device map --")
    with tempfile.TemporaryDirectory() as d:
        sd = {"a": torch.randn(4), "b": torch.randn(4)}
        save_dict(sd, d)

        # int key
        loaded = load_dict(d, {"": 0})
        check("device_map/int_key", all(v.device.type == "cuda" for v in loaded.values()))

        # cuda string key
        loaded = load_dict(d, {"": "cuda:0"})
        check("device_map/cuda_string", all(v.device.type == "cuda" for v in loaded.values()))

        # per-tensor key takes precedence
        loaded = load_dict(d, {"a": 0, "b": "cpu"})
        check("device_map/per_tensor",
              loaded["a"].device.type == "cuda" and loaded["b"].device.type == "cpu")

    # --- models ---
    print("\n-- Models --")
    models = [
        ("Linear",    nn.Linear(64, 128)),
        ("Embedding", nn.Embedding(1000, 64)),
        ("Conv2d",    nn.Conv2d(3, 16, 3)),
        ("LayerNorm", nn.LayerNorm(128)),
        ("MLP",       nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Linear(512, 128))),
    ]
    for model_name, model in models:
        with tempfile.TemporaryDirectory() as d:
            sd = model.state_dict()
            loaded = round_trip(sd, d)
            ok = all(tensors_equal(sd[k], loaded[k]) for k in sd)
            on_gpu = all(v.device.type == "cuda" for v in loaded.values())
            check(f"model/{model_name}", ok and on_gpu)

    # --- inference correctness ---
    print("\n-- Inference --")
    torch.manual_seed(0)
    model = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    expected = model(x)

    with tempfile.TemporaryDirectory() as d:
        save_dict(model.state_dict(), d)
        loaded_sd = load_dict(d, {"": 0})
        # bring back to cpu for load_state_dict
        cpu_sd = {k: v.cpu() for k, v in loaded_sd.items()}

    model2 = nn.Linear(16, 8)
    model2.load_state_dict(cpu_sd)
    actual = model2(x)
    check("inference/output_matches", torch.allclose(expected, actual))

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
def run_gpu_tests():
    return _run_tests()


@app.local_entrypoint()
def main():
    result = run_gpu_tests.remote()
    if result["failed"] > 0:
        raise SystemExit(f"{result['failed']} GPU test(s) failed")
