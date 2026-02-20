"""Load / offload / reload tests for the flashtensors GPU path, run via Modal.

Tests the hot-swap lifecycle:
    save → fast-load to GPU → inference → offload to CPU → fast-reload to GPU → inference

Run with:
    modal run tests/test_offload_gpu.py
"""

import modal

app = modal.App("flashtensors-offload-test")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy")
    .add_local_dir("/workspaces/flashtensors/flashtensors", remote_path="/pkg/flashtensors")
)


def _run_tests():
    import sys
    sys.path.insert(0, "/pkg")

    import tempfile
    import torch
    import torch.nn as nn

    from flashtensors import load_dict, save_dict

    results = []

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results.append((name, status, detail))
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    def fast_load(path):
        return load_dict(path, {"": 0})

    def offload(state_dict):
        """Move all tensors to CPU and delete the GPU copies."""
        import cupy
        cpu_sd = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        torch.cuda.empty_cache()
        cupy.get_default_memory_pool().free_all_blocks()
        return cpu_sd

    print("\n=== flashtensors load / offload / reload tests ===\n")

    # ------------------------------------------------------------------
    # 1. Basic load → offload → reload cycle
    # ------------------------------------------------------------------
    print("-- Basic cycle --")
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(0)
        model = nn.Linear(64, 64)
        save_dict(model.state_dict(), d)

        # Load to GPU
        gpu_sd = fast_load(d)
        check("cycle/tensors_on_gpu",
              all(v.device.type == "cuda" for v in gpu_sd.values()))

        # Run inference on GPU
        model_gpu = nn.Linear(64, 64).cuda()
        model_gpu.load_state_dict(gpu_sd)
        x = torch.randn(4, 64, device="cuda")
        out_before = model_gpu(x).detach()

        # Capture data pointers before offload to verify reload gives fresh tensors
        ptrs_before = {k: v.data_ptr() for k, v in gpu_sd.items()}

        # Offload to CPU
        cpu_sd = offload(gpu_sd)
        del model_gpu
        torch.cuda.empty_cache()

        check("cycle/tensors_on_cpu",
              all(v.device.type == "cpu" for v in cpu_sd.values()))

        # Reload to GPU
        gpu_sd2 = fast_load(d)
        check("cycle/reload_on_gpu",
              all(v.device.type == "cuda" for v in gpu_sd2.values()))

        # Reload should give fresh GPU allocations (different data pointers)
        ptrs_after = {k: v.data_ptr() for k, v in gpu_sd2.items()}
        check("cycle/fresh_tensors_after_reload",
              ptrs_before != ptrs_after)

        # Inference after reload should match
        model_gpu2 = nn.Linear(64, 64).cuda()
        model_gpu2.load_state_dict(gpu_sd2)
        out_after = model_gpu2(x).detach()
        check("cycle/inference_matches",
              torch.allclose(out_before, out_after),
              f"max_diff={( out_before - out_after).abs().max().item():.2e}")

    # ------------------------------------------------------------------
    # 2. Multiple hot-swap cycles
    # ------------------------------------------------------------------
    print("\n-- Multiple hot-swap cycles --")
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        save_dict(model.state_dict(), d)

        x = torch.randn(8, 128, device="cuda")

        # Reference output
        ref_model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).cuda()
        ref_model.load_state_dict(fast_load(d))
        ref_out = ref_model(x).detach()
        del ref_model
        torch.cuda.empty_cache()

        for i in range(3):
            gpu_sd = fast_load(d)
            m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).cuda()
            m.load_state_dict(gpu_sd)
            out = m(x).detach()
            match = torch.allclose(ref_out, out)
            del m
            cpu_sd = offload(gpu_sd)
            torch.cuda.empty_cache()
            check(f"hotswap/cycle_{i+1}_inference_matches", match)

    # ------------------------------------------------------------------
    # 3. Mixed-device offload (some tensors CPU, some GPU)
    # ------------------------------------------------------------------
    print("\n-- Mixed device offload --")
    with tempfile.TemporaryDirectory() as d:
        model = nn.Sequential(nn.Linear(32, 64), nn.Linear(64, 32))
        sd = model.state_dict()
        save_dict(sd, d)

        keys = list(sd.keys())
        # Load first two tensors to GPU, rest to CPU
        device_map = {keys[0]: 0, keys[1]: "cpu", keys[2]: 0, keys[3]: "cpu"}
        mixed = load_dict(d, device_map)

        gpu_keys = [k for k, v in mixed.items() if v.device.type == "cuda"]
        cpu_keys = [k for k, v in mixed.items() if v.device.type == "cpu"]
        check("mixed/gpu_keys_correct",  set(gpu_keys) == {keys[0], keys[2]})
        check("mixed/cpu_keys_correct",  set(cpu_keys) == {keys[1], keys[3]})
        check("mixed/values_correct",
              all(torch.allclose(sd[k].float(), mixed[k].float().cpu()) for k in keys))

    # ------------------------------------------------------------------
    # 4. bfloat16 model full cycle (common for LLMs)
    # ------------------------------------------------------------------
    print("\n-- bfloat16 model cycle --")
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(7)
        model = nn.Linear(64, 64, dtype=torch.bfloat16)
        save_dict(model.state_dict(), d)

        gpu_sd = fast_load(d)
        check("bf16/dtype_preserved",
              all(v.dtype == torch.bfloat16 for v in gpu_sd.values()))
        check("bf16/on_gpu",
              all(v.device.type == "cuda" for v in gpu_sd.values()))

        m = nn.Linear(64, 64, dtype=torch.bfloat16).cuda()
        m.load_state_dict(gpu_sd)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        out_bf16 = m(x).detach()

        cpu_sd = offload(gpu_sd)
        del m
        torch.cuda.empty_cache()

        gpu_sd2 = fast_load(d)
        m2 = nn.Linear(64, 64, dtype=torch.bfloat16).cuda()
        m2.load_state_dict(gpu_sd2)
        out_bf16_2 = m2(x).detach()

        check("bf16/inference_matches_after_reload",
              torch.allclose(out_bf16, out_bf16_2),
              f"max_diff={( out_bf16 - out_bf16_2).abs().max().item():.2e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
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
def run_offload_tests():
    return _run_tests()


@app.local_entrypoint()
def main():
    result = run_offload_tests.remote()
    if result["failed"] > 0:
        raise SystemExit(f"{result['failed']} offload test(s) failed")
