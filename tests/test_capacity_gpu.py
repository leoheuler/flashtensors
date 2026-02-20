"""Capacity and throughput tests for the flashtensors GPU loader, run via Modal.

Creates synthetic large model state dicts (transformer-shaped, bfloat16) and measures:
  - Save throughput (CPU → disk)
  - Load throughput  (disk → GPU via pinned memory)
  - Offload throughput (GPU → CPU)
  - Reload throughput (disk → GPU)

Run with:
    modal run tests/test_capacity_gpu.py
"""

import modal

app = modal.App("flashtensors-capacity-test")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy")
    .add_local_dir("/workspaces/flashtensors/flashtensors", remote_path="/pkg/flashtensors")
)


def _make_state_dict(size_gb: float, dtype=None):
    """Build a synthetic transformer-shaped state dict of approximately size_gb GB.

    Uses (4096, 4096) weight blocks — the shape of a typical LLM MLP weight.
    Uses torch.empty (no random fill) so creation is fast even for large models.
    """
    import torch
    if dtype is None:
        dtype = torch.bfloat16

    bytes_per_elem = 2  # bfloat16
    block_elems = 4096 * 4096          # ~33 MB per block in bf16
    total_elems = int(size_gb * 1024 ** 3 / bytes_per_elem)
    n_full = total_elems // block_elems
    remainder = total_elems % block_elems

    sd = {}
    for i in range(n_full):
        sd[f"layer_{i}.weight"] = torch.empty(4096, 4096, dtype=dtype)
    if remainder:
        sd[f"layer_{n_full}.bias"] = torch.empty(remainder, dtype=dtype)
    return sd


def _gb(state_dict):
    """Actual size in GB of a state dict."""
    total = sum(t.numel() * t.element_size() for t in state_dict.values())
    return total / 1024 ** 3


def _free_gpu_gb():
    import torch
    free, _ = torch.cuda.mem_get_info(0)
    return free / 1024 ** 3


def _run_tests():
    import sys
    sys.path.insert(0, "/pkg")

    import gc
    import tempfile
    import time

    import cupy
    import torch

    from flashtensors import load_dict, save_dict

    # Sizes to test in GB — stop early if GPU OOM
    target_sizes = [1, 2, 4, 8, 16, 20]

    print("\n=== flashtensors capacity & throughput tests ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.mem_get_info(0)[1] / 1024 ** 3
    print(f"VRAM: {total_vram:.1f} GB\n")

    rows = []

    for target_gb in target_sizes:
        print(f"--- {target_gb} GB model ---")

        # ── Create state dict ──────────────────────────────────────────────
        try:
            sd = _make_state_dict(target_gb)
        except MemoryError:
            print(f"  CPU OOM creating {target_gb} GB model, stopping.\n")
            break

        actual_gb = _gb(sd)
        n_tensors = len(sd)
        print(f"  Tensors: {n_tensors}  Actual size: {actual_gb:.2f} GB")

        row = {"size_gb": actual_gb, "n_tensors": n_tensors}

        with tempfile.TemporaryDirectory() as d:

            # ── Save (CPU → disk) ──────────────────────────────────────────
            t0 = time.perf_counter()
            save_dict(sd, d)
            row["save_s"] = time.perf_counter() - t0
            print(f"  Save:    {row['save_s']:.2f}s  ({actual_gb / row['save_s']:.2f} GB/s)")

            # Sample a few values for correctness check later
            sample_keys = list(sd.keys())[:3]
            sample_cpu = {k: sd[k].clone() for k in sample_keys}
            del sd
            gc.collect()

            # ── Load (disk → GPU via pinned memory) ───────────────────────
            free_before = _free_gpu_gb()
            try:
                t0 = time.perf_counter()
                gpu_sd = load_dict(d, {"": 0})
                torch.cuda.synchronize()
                row["load_s"] = time.perf_counter() - t0
            except torch.cuda.OutOfMemoryError:
                print(f"  GPU OOM loading {actual_gb:.1f} GB — max loadable size is below this.\n")
                break

            free_after_load = _free_gpu_gb()
            gpu_used = free_before - free_after_load
            print(f"  Load:    {row['load_s']:.2f}s  ({actual_gb / row['load_s']:.2f} GB/s)"
                  f"  [GPU used: {gpu_used:.1f} GB, free: {free_after_load:.1f} GB]")

            # Correctness: sampled values should match
            all_match = all(
                torch.equal(sample_cpu[k], gpu_sd[k].cpu()) for k in sample_keys
            )
            print(f"  Values match: {all_match}")
            row["values_match"] = all_match

            # ── Offload (GPU → CPU) ────────────────────────────────────────
            t0 = time.perf_counter()
            cpu_sd = {k: v.cpu() for k, v in gpu_sd.items()}
            torch.cuda.synchronize()
            del gpu_sd
            torch.cuda.empty_cache()
            cupy.get_default_memory_pool().free_all_blocks()
            row["offload_s"] = time.perf_counter() - t0
            print(f"  Offload: {row['offload_s']:.2f}s  ({actual_gb / row['offload_s']:.2f} GB/s)"
                  f"  [GPU free: {_free_gpu_gb():.1f} GB]")

            del cpu_sd
            gc.collect()

            # ── Reload (disk → GPU again) ──────────────────────────────────
            try:
                t0 = time.perf_counter()
                gpu_sd2 = load_dict(d, {"": 0})
                torch.cuda.synchronize()
                row["reload_s"] = time.perf_counter() - t0
            except torch.cuda.OutOfMemoryError:
                print(f"  GPU OOM on reload.\n")
                break

            print(f"  Reload:  {row['reload_s']:.2f}s  ({actual_gb / row['reload_s']:.2f} GB/s)")

            del gpu_sd2
            torch.cuda.empty_cache()
            cupy.get_default_memory_pool().free_all_blocks()
            gc.collect()

        rows.append(row)
        print()

    # ── Summary table ──────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'Size':>6}  {'Save':>9}  {'Load':>9}  {'Offload':>9}  {'Reload':>9}  {'OK':>4}")
    print(f"{'(GB)':>6}  {'(GB/s)':>9}  {'(GB/s)':>9}  {'(GB/s)':>9}  {'(GB/s)':>9}")
    print("-" * 72)
    for r in rows:
        gb = r["size_gb"]
        def tp(key):
            return f"{gb / r[key]:.2f}" if key in r else "  OOM"
        ok = "✓" if r.get("values_match") else "✗"
        print(f"{gb:>6.1f}  {tp('save_s'):>9}  {tp('load_s'):>9}  "
              f"{tp('offload_s'):>9}  {tp('reload_s'):>9}  {ok:>4}")
    print("=" * 72)

    return rows


@app.function(gpu="L4", image=image, timeout=600)
def run_capacity_tests():
    return _run_tests()


@app.local_entrypoint()
def main():
    run_capacity_tests.remote()
