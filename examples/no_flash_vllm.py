import os
import time
import torch
from vllm import LLM, SamplingParams
import GPUtil


def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
            return gpu.memoryUsed, gpu.memoryTotal
    except:
        pass
    return None, None


def get_model_size_estimate(model_path):
    """Estimate model size from files (rough approximation)"""
    try:
        if os.path.isdir(model_path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
    except:
        pass
    return None


def main():
    print("🔧 Initializing pure vLLM...")
    
    # Model configuration
    model_id = "Qwen/Qwen3-0.6B"
    
    # vLLM configuration parameters (matching TeilEngine config as much as possible)
    gpu_memory_utilization = 0.8
    dtype = "bfloat16"
    
    print(f"📋 Configuration:")
    print(f"   Model: {model_id}")
    print(f"   GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"   Data Type: {dtype}")
    print(f"   Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Get initial GPU memory usage
    gpu_mem_before, gpu_total = get_gpu_memory_usage()
    if gpu_mem_before is not None:
        print(f"   Initial GPU Memory Usage: {gpu_mem_before:.1f} / {gpu_total:.1f} MB")
    
    # Step 1: Load model with vLLM (this includes download if needed)
    print(f"\n⚡ Loading model {model_id} with pure vLLM...")
    load_start_time = time.time()
    
    try:
        llm = LLM(
            model=model_id,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True
            # Additional vLLM parameters you might want to tune:
            # tensor_parallel_size=1,  # For multi-GPU setups
            # trust_remote_code=True,  # If model requires it
            # download_dir="/tmp/vllm_models",  # Cache directory
        )
        
        load_time = time.time() - load_start_time
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Get GPU memory usage after loading
        gpu_mem_after, _ = get_gpu_memory_usage()
        if gpu_mem_after is not None and gpu_mem_before is not None:
            memory_used = gpu_mem_after - gpu_mem_before
            print(f"   GPU Memory Used by Model: {memory_used:.1f} MB ({memory_used/1024:.2f} GB)")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 2: Run inference (same prompts as TeilEngine version)
    print("\n🤖 Running inference...")
    prompts = [
        "Hello, my name is",
        "The capital of France is", 
        "The future of AI is",
    ]
    
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=50)
    
    inference_start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    inference_time = time.time() - inference_start_time
    
    print(f"✅ Inference completed in {inference_time:.2f}s")
    print("📝 Generated outputs:")
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"   Prompt: {prompt!r}")
        print(f"   Generated: {generated_text!r}")
        print()
    
    # Step 3: Show model information
    print("\n📊 Model information:")
    print(f"   Model ID: {model_id}")
    print(f"   Backend: vLLM")
    print(f"   Data Type: {dtype}")
    print(f"   Load Time: {load_time:.2f}s")
    print(f"   Inference Time: {inference_time:.2f}s")
    
    # Try to estimate model size (this is approximate)
    try:
        # vLLM doesn't expose model size directly, but we can check cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_path = None
        for item in os.listdir(cache_dir):
            if model_id.replace("/", "--") in item:
                model_cache_path = os.path.join(cache_dir, item)
                break
        
        if model_cache_path:
            size = get_model_size_estimate(model_cache_path)
            if size:
                print(f"   Estimated Model Size: {size / (1024**3):.2f} GB")
    except Exception as e:
        print(f"   Could not estimate model size: {e}")
    
    # Step 4: Performance summary
    print("\n⏱️  Performance Summary:")
    print(f"   Total Load Time: {load_time:.2f}s")
    print(f"   Inference Time: {inference_time:.2f}s")
    print(f"   Tokens per second (approx): {(len(prompts) * 50) / inference_time:.1f}")
    
    # Step 5: Memory cleanup (Python garbage collection)
    print("\n🧹 Cleaning up memory...")
    del llm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Final GPU memory check
    gpu_mem_final, _ = get_gpu_memory_usage()
    if gpu_mem_final is not None and gpu_mem_before is not None:
        print(f"   Final GPU Memory Usage: {gpu_mem_final:.1f} MB")
        print(f"   Memory Released: {gpu_mem_after - gpu_mem_final:.1f} MB" if gpu_mem_after else "N/A")
    
    print("✅ Cleanup completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure GPU memory is cleared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
