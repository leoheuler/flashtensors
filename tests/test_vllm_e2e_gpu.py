"""End-to-end GPU test: convert a real HuggingFace model to flashtensors,
then load it via the vLLM monkey-patch and run inference.

Uses SmolLM-135M (tiny model, fast download).

Run with:
    modal run tests/test_vllm_e2e_gpu.py
"""

import modal

app = modal.App("flashtensors-vllm-e2e-test")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "cupy-cuda12x", "pydantic", "tqdm", "numpy",
        "vllm", "safetensors", "huggingface_hub",
    )
    .add_local_dir(
        "/workspaces/flashtensors/flashtensors",
        remote_path="/pkg/flashtensors",
    )
)

MODEL_ID = "HuggingFaceTB/SmolLM-135M"


def _run_tests():
    import sys
    sys.path.insert(0, "/pkg")

    import os
    import shutil
    import tempfile
    import time

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from flashtensors import save_dict

    results = []

    def check(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        results.append((name, status, detail))
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))

    print(f"\n=== flashtensors vLLM e2e test ({MODEL_ID}) ===\n")

    workdir = tempfile.mkdtemp()
    hf_dir = os.path.join(workdir, "hf")
    ft_dir = os.path.join(workdir, "ft")

    try:
        # ------------------------------------------------------------------
        # Step 1: Download model from HuggingFace
        # ------------------------------------------------------------------
        print("-- Downloading model --")
        t0 = time.time()
        hf_dir = snapshot_download(
            MODEL_ID,
            cache_dir=os.path.join(workdir, "cache"),
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )
        print(f"  Downloaded in {time.time() - t0:.1f}s to {hf_dir}")

        # ------------------------------------------------------------------
        # Step 2: Convert safetensors -> flashtensors
        # ------------------------------------------------------------------
        print("\n-- Converting to flashtensors --")
        t0 = time.time()

        # Collect all safetensors weights
        st_files = [
            os.path.join(hf_dir, f)
            for f in os.listdir(hf_dir)
            if f.endswith(".safetensors")
        ]
        print(f"  Found {len(st_files)} safetensors file(s)")

        state_dict = {}
        for st_file in st_files:
            state_dict.update(load_file(st_file, device="cpu"))
        print(f"  Loaded {len(state_dict)} tensors")

        os.makedirs(ft_dir, exist_ok=True)
        save_dict(state_dict, ft_dir)

        # Copy config files alongside flashtensors data
        for f in os.listdir(hf_dir):
            if not f.endswith(".safetensors") and not f.endswith(".bin"):
                src = os.path.join(hf_dir, f)
                dst = os.path.join(ft_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        convert_time = time.time() - t0
        ft_size = os.path.getsize(os.path.join(ft_dir, "tensor.flashtensors"))
        check(
            "conversion",
            os.path.exists(os.path.join(ft_dir, "tensor_index.json")),
            f"{len(state_dict)} tensors, {ft_size / 1e6:.1f} MB, {convert_time:.1f}s",
        )

        # ------------------------------------------------------------------
        # Step 3: Load via vLLM with flashtensors monkey-patch
        # ------------------------------------------------------------------
        print("\n-- Loading via vLLM + flashtensors patch --")

        import flashtensors.vllm  # applies patch

        from vllm import LLM, SamplingParams

        t0 = time.time()
        llm = LLM(
            model=ft_dir,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            max_model_len=128,
        )
        load_time = time.time() - t0
        check("vllm_load", llm is not None, f"{load_time:.1f}s")

        # ------------------------------------------------------------------
        # Step 4: Run inference
        # ------------------------------------------------------------------
        print("\n-- Running inference --")
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=20,
        )
        prompts = ["The capital of France is"]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        infer_time = time.time() - t0

        generated = outputs[0].outputs[0].text
        check(
            "inference_runs",
            len(generated) > 0,
            f"'{generated.strip()[:80]}' ({infer_time:.2f}s)",
        )

        # ------------------------------------------------------------------
        # Step 5: Load same model from safetensors (baseline) and compare
        # ------------------------------------------------------------------
        print("\n-- Baseline comparison (safetensors) --")

        # Free the flashtensors model first to avoid OOM
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        t0 = time.time()
        llm_baseline = LLM(
            model=hf_dir,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.5,
            max_model_len=128,
        )
        baseline_time = time.time() - t0

        outputs_baseline = llm_baseline.generate(prompts, sampling_params)
        baseline_text = outputs_baseline[0].outputs[0].text

        match = generated.strip() == baseline_text.strip()
        check(
            "output_matches_baseline",
            match,
            f"ft='{generated.strip()[:60]}' st='{baseline_text.strip()[:60]}'",
        )

        print(f"\n  Load time: flashtensors={load_time:.2f}s  safetensors={baseline_time:.2f}s")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

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


@app.function(gpu="L4", image=image, timeout=600)
def run_e2e_tests():
    return _run_tests()


@app.local_entrypoint()
def main():
    result = run_e2e_tests.remote()
    if result["failed"] > 0:
        raise SystemExit(f"{result['failed']} e2e test(s) failed")
