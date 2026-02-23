"""Throughput comparison benchmark — run via Modal.

Compares disk→GPU loading speed across:
  1. torch.save / torch.load      (pickle-based, standard PyTorch)
  2. safetensors save / load_file (HuggingFace format, mmap-based)
  3. flashtensors save / load_dict (our format, pinned-memory DMA)

Also measures two theoretical ceilings:
  - Raw disk read speed:  how fast the storage can be read into RAM
  - Raw PCIe H2D speed:   how fast pinned RAM → GPU (pure CUDA memcpy)

These ceilings show where the bottleneck is and how close each loader
gets to the physical limits.

Run with:
    modal run benchmarks/benchmark_comparison.py
"""

import modal

app = modal.App("flashtensors-comparison-benchmark")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy", "safetensors", "packaging", "kvikio-cu12")
    .add_local_dir("/workspaces/flashtensors/flashtensors", remote_path="/pkg/flashtensors")
)

SIZES_GB = [4, 8]
WARMUP_ITERS = 1
BENCH_ITERS = 3


def _make_state_dict(size_gb, dtype=None):
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    bytes_per_elem = 2
    block_elems = 4096 * 4096
    total_elems = int(size_gb * 1024 ** 3 / bytes_per_elem)
    n_full = total_elems // block_elems
    remainder = total_elems % block_elems
    sd = {}
    for i in range(n_full):
        sd[f"layer_{i}.weight"] = torch.empty(4096, 4096, dtype=dtype)
    if remainder:
        sd[f"layer_{n_full}.bias"] = torch.empty(remainder, dtype=dtype)
    return sd


def _actual_gb(sd):
    return sum(t.numel() * t.element_size() for t in sd.values()) / 1024 ** 3


def _time_fn(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run fn() warmup+iters times, return mean of timed iters in seconds."""
    import time
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _run():
    import sys
    sys.path.insert(0, "/pkg")

    import gc
    import json
    import mmap
    import os
    import tempfile

    import cupy
    import numpy as np
    import torch
    from safetensors.torch import load_file as sf_load_file
    from safetensors.torch import save_file as sf_save_file

    from flashtensors import load_dict, save_dict

    print(f"\n{'='*68}")
    print(f"  flashtensors vs safetensors vs torch.load — throughput benchmark")
    print(f"{'='*68}")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  VRAM: {total/1024**3:.1f} GB  (free: {free/1024**3:.1f} GB)")
    print(f"  Warmup iters: {WARMUP_ITERS}  Bench iters: {BENCH_ITERS}")

    all_results = []

    for size_gb in SIZES_GB:
        print(f"\n{'─'*68}")
        print(f"  Model size: {size_gb} GB  (bfloat16, transformer-shaped)")
        print(f"{'─'*68}")

        sd = _make_state_dict(size_gb)
        actual_gb = _actual_gb(sd)

        results = {"size_gb": actual_gb}

        with tempfile.TemporaryDirectory() as d:
            pt_path  = os.path.join(d, "model.pt")
            sf_path  = os.path.join(d, "model.safetensors")
            ft_path  = os.path.join(d, "ft")

            # ── Save all formats ───────────────────────────────────────────
            print("\n  [Saving all formats]")

            t = _time_fn(lambda: torch.save(sd, pt_path), warmup=0, iters=1)
            print(f"    torch.save:     {t:.2f}s  ({actual_gb/t:.2f} GB/s)")

            t = _time_fn(lambda: sf_save_file(sd, sf_path), warmup=0, iters=1)
            print(f"    safetensors:    {t:.2f}s  ({actual_gb/t:.2f} GB/s)")

            t = _time_fn(lambda: save_dict(sd, ft_path), warmup=0, iters=1)
            print(f"    flashtensors:   {t:.2f}s  ({actual_gb/t:.2f} GB/s)")

            del sd
            gc.collect()

            # ── Ceiling 1: Raw disk read speed ────────────────────────────
            # Read the flashtensors binary file sequentially into RAM.
            # This is the theoretical max for any loader that must read from disk.
            data_file = os.path.join(ft_path, "tensor.flashtensors")
            file_size = os.path.getsize(data_file)

            def _raw_read():
                with open(data_file, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    # Force all pages into RAM by summing — avoids compiler opt
                    buf = np.frombuffer(mm, dtype=np.uint8)
                    _ = int(buf.sum())
                    del buf  # Release numpy view before closing mmap
                    mm.close()

            t = _time_fn(_raw_read)
            raw_disk_gbs = actual_gb / t
            results["raw_disk_gbs"] = raw_disk_gbs
            print(f"\n  [Theoretical ceilings]")
            print(f"    Raw disk read:  {t:.2f}s  ({raw_disk_gbs:.2f} GB/s)  ← storage limit")

            # ── Ceiling 2: Raw PCIe H2D bandwidth ────────────────────────
            # Allocate pinned memory, fill it, then time the GPU transfer.
            # This is the max speed for getting data from RAM → GPU.
            pinned = cupy.cuda.alloc_pinned_memory(file_size)
            pinned_arr = np.frombuffer(pinned, dtype=np.uint8, count=file_size)
            pinned_arr[:] = 0  # Touch all pages so they're resident

            def _raw_h2d():
                with cupy.cuda.Device(0):
                    gpu = cupy.asarray(pinned_arr)
                    cupy.cuda.runtime.deviceSynchronize()
                    del gpu

            t = _time_fn(_raw_h2d)
            raw_h2d_gbs = actual_gb / t
            results["raw_h2d_gbs"] = raw_h2d_gbs
            print(f"    Raw PCIe H2D:  {t:.2f}s  ({raw_h2d_gbs:.2f} GB/s)  ← PCIe limit")
            del pinned, pinned_arr

            # ── Loader benchmarks ──────────────────────────────────────────
            print(f"\n  [Load to GPU — {BENCH_ITERS} iters each, first {WARMUP_ITERS} warm-up]")

            # torch.load
            def _torch_load():
                sd = torch.load(pt_path, map_location="cuda", weights_only=True)
                torch.cuda.synchronize()
                del sd
                torch.cuda.empty_cache()
            t = _time_fn(_torch_load)
            results["torch_gbs"] = actual_gb / t
            print(f"    torch.load:     {t:.2f}s  ({results['torch_gbs']:.2f} GB/s)"
                  f"  ({100*results['torch_gbs']/raw_disk_gbs:.0f}% of disk, "
                  f"{100*results['torch_gbs']/raw_h2d_gbs:.0f}% of PCIe)")

            # safetensors
            def _sf_load():
                sd = sf_load_file(sf_path, device="cuda")
                torch.cuda.synchronize()
                del sd
                torch.cuda.empty_cache()
            t = _time_fn(_sf_load)
            results["sf_gbs"] = actual_gb / t
            print(f"    safetensors:    {t:.2f}s  ({results['sf_gbs']:.2f} GB/s)"
                  f"  ({100*results['sf_gbs']/raw_disk_gbs:.0f}% of disk, "
                  f"{100*results['sf_gbs']/raw_h2d_gbs:.0f}% of PCIe)")

            # flashtensors (SLLM pipeline) — sweep num_workers
            ft_best_gbs = 0
            ft_best_workers = 1
            for nw in [1, 2, 4, 8]:
                def _ft_load(nw=nw):
                    sd = load_dict(ft_path, {"": 0}, num_workers=nw)
                    torch.cuda.synchronize()
                    del sd
                    torch.cuda.empty_cache()
                    cupy.get_default_memory_pool().free_all_blocks()
                t = _time_fn(_ft_load)
                gbs = actual_gb / t
                marker = " ◀ best" if gbs > ft_best_gbs else ""
                print(f"    flashtensors({nw:>2}w): {t:.2f}s  ({gbs:.2f} GB/s)"
                      f"  ({100*gbs/raw_disk_gbs:.0f}% of disk, "
                      f"{100*gbs/raw_h2d_gbs:.0f}% of PCIe){marker}")
                if gbs > ft_best_gbs:
                    ft_best_gbs, ft_best_workers = gbs, nw
            results["ft_gbs"] = ft_best_gbs
            results["ft_best_workers"] = ft_best_workers

            # flashtensors GDS (kvikio) — if available
            gds_available = False
            gds_compat = True
            try:
                import kvikio
                gds_available = True
                gds_compat = kvikio.defaults.compat_mode()
            except Exception:
                pass

            if gds_available:
                mode_label = "compat" if gds_compat else "GDS"
                from flashtensors.loaders._gds import GdsLoader
                from flashtensors.flash_state import FlashState

                index_path = os.path.join(ft_path, "tensor_index.json")
                data_file_path = os.path.join(ft_path, "tensor.flashtensors")
                with open(index_path) as f:
                    flash_state = FlashState(**json.load(f))
                gds_file_size = os.path.getsize(data_file_path)

                def _ft_gds_load():
                    loader = GdsLoader(device_id=0)
                    sd = loader.load(data_file_path, gds_file_size, flash_state.layout,
                                     {"": 0}, 1, 64 * 1024 * 1024)
                    torch.cuda.synchronize()
                    del sd
                    torch.cuda.empty_cache()
                    cupy.get_default_memory_pool().free_all_blocks()

                t = _time_fn(_ft_gds_load)
                gds_gbs = actual_gb / t
                results["gds_gbs"] = gds_gbs
                print(f"    flashtensors GDS ({mode_label}): {t:.2f}s  ({gds_gbs:.2f} GB/s)"
                      f"  ({100*gds_gbs/raw_disk_gbs:.0f}% of disk, "
                      f"{100*gds_gbs/raw_h2d_gbs:.0f}% of PCIe)")
            else:
                results["gds_gbs"] = None
                print(f"    flashtensors GDS: skipped (kvikio not available)")

        all_results.append(results)
        gc.collect()

    # ── Summary table ──────────────────────────────────────────────────────
    has_gds = any(r.get("gds_gbs") is not None for r in all_results)
    gds_col = f"  {'GDS':>10}" if has_gds else ""

    print(f"\n{'='*78}")
    print(f"  Summary — disk→GPU load throughput (GB/s)")
    print(f"{'─'*78}")
    print(f"  {'Size':>5}  {'Disk ceil':>10}  {'PCIe ceil':>10}  "
          f"{'torch':>8}  {'safetensors':>12}  {'flashtensors':>13}{gds_col}")
    print(f"{'─'*78}")
    for r in all_results:
        gds_str = f"  {r['gds_gbs']:>9.2f}x" if r.get("gds_gbs") is not None else (
            f"  {'n/a':>10}" if has_gds else "")
        print(f"  {r['size_gb']:>4.0f}G"
              f"  {r['raw_disk_gbs']:>9.2f}x"
              f"  {r['raw_h2d_gbs']:>9.2f}x"
              f"  {r['torch_gbs']:>7.2f}x"
              f"  {r['sf_gbs']:>11.2f}x"
              f"  {r['ft_gbs']:>12.2f}x{gds_str}")
    print(f"{'─'*78}")
    r = all_results[-1]
    print(f"  Speedup flashtensors vs torch.load:   {r['ft_gbs']/r['torch_gbs']:.1f}x")
    print(f"  Speedup flashtensors vs safetensors:  {r['ft_gbs']/r['sf_gbs']:.1f}x")
    print(f"  flashtensors best num_workers:        {r['ft_best_workers']}")
    print(f"  flashtensors % of disk ceiling:       {100*r['ft_gbs']/r['raw_disk_gbs']:.0f}%")
    print(f"  flashtensors % of PCIe ceiling:       {100*r['ft_gbs']/r['raw_h2d_gbs']:.0f}%")
    if r.get("gds_gbs") is not None:
        print(f"  GDS % of disk ceiling:               {100*r['gds_gbs']/r['raw_disk_gbs']:.0f}%")
        print(f"  GDS % of PCIe ceiling:               {100*r['gds_gbs']/r['raw_h2d_gbs']:.0f}%")
        print(f"  GDS vs SLLM pipeline:                {r['gds_gbs']/r['ft_gbs']:.2f}x")
    print(f"{'='*78}\n")

    return all_results


@app.function(gpu="L4", image=image, timeout=600)
def run_comparison():
    return _run()


@app.local_entrypoint()
def main():
    run_comparison.remote()
