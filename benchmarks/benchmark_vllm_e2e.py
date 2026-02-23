"""End-to-end benchmark: real HuggingFace model weights loaded via
flashtensors vs safetensors, disk → GPU.

Measures what matters: how fast do model weights go from disk to GPU tensors.
Drops page cache between runs for cold-load measurement.

Run with:
    modal run benchmarks/benchmark_vllm_e2e.py

Models tested (all open-weight, no gating):
  - SmolLM-135M       (~0.27 GB)  — tiny
  - Qwen2.5-0.5B      (~0.99 GB)  — small
  - Qwen2.5-1.5B      (~3.09 GB)  — medium
  - Qwen2.5-3B        (~6.05 GB)  — large
"""

import modal

app = modal.App("flashtensors-real-model-bench")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy",
        "safetensors", "huggingface_hub",
    )
    .add_local_dir(
        "/workspaces/flashtensors/flashtensors",
        remote_path="/pkg/flashtensors",
    )
)

MODELS = [
    "HuggingFaceTB/SmolLM-135M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]

BENCH_ITERS = 3


def _drop_caches():
    """Drop Linux page cache for cold-load measurements."""
    import os
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except Exception:
        return False


def _bench_one(model_id):
    import sys
    sys.path.insert(0, "/pkg")

    import gc
    import os
    import shutil
    import tempfile
    import time

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as sf_load_file

    from flashtensors import save_dict, load_dict

    result = {"model": model_id}
    workdir = tempfile.mkdtemp()
    ft_dir = os.path.join(workdir, "ft")

    try:
        # --- Download ---
        t0 = time.perf_counter()
        hf_dir = snapshot_download(
            model_id,
            cache_dir=os.path.join(workdir, "cache"),
            allow_patterns=["*.safetensors", "*.json"],
        )
        result["download_s"] = time.perf_counter() - t0

        # --- Discover safetensors files ---
        st_files = sorted([
            os.path.join(hf_dir, f)
            for f in os.listdir(hf_dir)
            if f.endswith(".safetensors")
        ])
        result["num_st_files"] = len(st_files)

        # --- Load into CPU to get size and convert ---
        state_dict = {}
        for sf in st_files:
            state_dict.update(sf_load_file(sf, device="cpu"))

        total_bytes = sum(t.numel() * t.element_size() for t in state_dict.values())
        result["size_gb"] = total_bytes / 1024**3
        result["num_tensors"] = len(state_dict)

        # --- Convert to flashtensors ---
        t0 = time.perf_counter()
        os.makedirs(ft_dir, exist_ok=True)
        save_dict(state_dict, ft_dir)
        result["convert_s"] = time.perf_counter() - t0

        # Verify conversion is lossless (spot check a few tensors)
        loaded_ft = load_dict(ft_dir, {"": "cpu"})
        for name in list(state_dict.keys())[:5]:
            assert torch.equal(state_dict[name], loaded_ft[name]), \
                f"Conversion mismatch for {name}"
        del loaded_ft

        # Keep reference for correctness check later
        ref_names = list(state_dict.keys())[:3]
        ref_tensors = {n: state_dict[n].clone() for n in ref_names}
        del state_dict
        gc.collect()

        ft_data_path = os.path.join(ft_dir, "tensor.flashtensors")

        # --- Benchmark: safetensors → GPU ---
        sf_times = []
        for i in range(BENCH_ITERS):
            _drop_caches()
            gc.collect()
            torch.cuda.empty_cache()

            t0 = time.perf_counter()
            sd = {}
            for sf in st_files:
                sd.update(sf_load_file(sf, device="cuda:0"))
            torch.cuda.synchronize()
            sf_times.append(time.perf_counter() - t0)

            # Correctness check on first iter
            if i == 0:
                for name in ref_names:
                    assert torch.allclose(
                        ref_tensors[name].float(),
                        sd[name].float().cpu()
                    ), f"safetensors mismatch for {name}"
            del sd
            torch.cuda.empty_cache()

        result["sf_mean_s"] = sum(sf_times) / len(sf_times)
        result["sf_min_s"] = min(sf_times)

        # --- Benchmark: flashtensors → GPU ---
        ft_times = []
        for i in range(BENCH_ITERS):
            _drop_caches()
            gc.collect()
            torch.cuda.empty_cache()

            t0 = time.perf_counter()
            sd = load_dict(ft_dir, {"": 0})
            torch.cuda.synchronize()
            ft_times.append(time.perf_counter() - t0)

            if i == 0:
                for name in ref_names:
                    assert torch.allclose(
                        ref_tensors[name].float(),
                        sd[name].float().cpu()
                    ), f"flashtensors mismatch for {name}"
            del sd
            torch.cuda.empty_cache()

        result["ft_mean_s"] = sum(ft_times) / len(ft_times)
        result["ft_min_s"] = min(ft_times)

        # --- Benchmark: flashtensors → CPU (no GPU transfer) ---
        ft_cpu_times = []
        for i in range(BENCH_ITERS):
            _drop_caches()
            gc.collect()

            t0 = time.perf_counter()
            sd = load_dict(ft_dir, {"": "cpu"})
            ft_cpu_times.append(time.perf_counter() - t0)
            del sd

        result["ft_cpu_mean_s"] = sum(ft_cpu_times) / len(ft_cpu_times)

        result["success"] = True

    except Exception as e:
        import traceback
        result["success"] = False
        result["error"] = f"{e}\n{traceback.format_exc()}"

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return result


def _run_all():
    import torch

    print(f"\n{'='*78}")
    print(f"  Real model benchmark: flashtensors vs safetensors (disk → GPU)")
    print(f"{'='*78}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  VRAM:   {total/1024**3:.1f} GB (free: {free/1024**3:.1f} GB)")
    print(f"  Iters:  {BENCH_ITERS} (page cache dropped between each)")
    can_drop = _drop_caches()
    print(f"  Cache:  {'can drop (root)' if can_drop else 'CANNOT drop (warm cache only)'}")
    print(f"{'='*78}")

    all_results = []

    for model_id in MODELS:
        print(f"\n{'─'*78}")
        print(f"  {model_id}")
        print(f"{'─'*78}")

        r = _bench_one(model_id)
        all_results.append(r)

        if not r.get("success"):
            print(f"  FAILED: {r.get('error', 'unknown')[:200]}")
            continue

        sf_gbs = r["size_gb"] / r["sf_mean_s"]
        ft_gbs = r["size_gb"] / r["ft_mean_s"]
        ft_cpu_gbs = r["size_gb"] / r["ft_cpu_mean_s"]
        speedup = r["sf_mean_s"] / r["ft_mean_s"]

        print(f"  Size:         {r['size_gb']:.2f} GB  ({r['num_tensors']} tensors, {r['num_st_files']} shard(s))")
        print(f"  Convert:      {r['convert_s']:.2f}s")
        print(f"  safetensors:  {r['sf_mean_s']:.3f}s  ({sf_gbs:.2f} GB/s)  [min {r['sf_min_s']:.3f}s]")
        print(f"  flashtensors: {r['ft_mean_s']:.3f}s  ({ft_gbs:.2f} GB/s)  [min {r['ft_min_s']:.3f}s]")
        print(f"  ft CPU-only:  {r['ft_cpu_mean_s']:.3f}s  ({ft_cpu_gbs:.2f} GB/s)")
        faster = "flashtensors" if speedup > 1 else "safetensors"
        print(f"  Speedup:      {speedup:.2f}x  ({faster})")

    # --- Summary table ---
    print(f"\n{'='*78}")
    print(f"  Summary — disk → GPU load time (mean of {BENCH_ITERS} cold runs)")
    print(f"{'─'*78}")
    print(f"  {'Model':<28} {'Size':>6} {'SF(s)':>8} {'FT(s)':>8}"
          f" {'SF GB/s':>8} {'FT GB/s':>8} {'Speedup':>8}")
    print(f"{'─'*78}")
    for r in all_results:
        if not r.get("success"):
            print(f"  {r['model']:<28} FAILED")
            continue
        sf_gbs = r["size_gb"] / r["sf_mean_s"]
        ft_gbs = r["size_gb"] / r["ft_mean_s"]
        speedup = r["sf_mean_s"] / r["ft_mean_s"]
        print(f"  {r['model']:<28} {r['size_gb']:>5.2f}G"
              f"  {r['sf_mean_s']:>7.3f}  {r['ft_mean_s']:>7.3f}"
              f"  {sf_gbs:>7.2f}  {ft_gbs:>7.2f}"
              f"  {speedup:>7.2f}x")
    print(f"{'='*78}\n")

    return all_results


@app.function(gpu="A100", image=image, timeout=900)
def run_benchmark():
    return _run_all()


@app.local_entrypoint()
def main():
    run_benchmark.remote()
