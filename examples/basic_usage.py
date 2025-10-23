import sys
import os
import torch

# Add the flashtensors directory to Python path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    import flashtensors as flash
except ImportError as e:
    print(f"Failed to import flashtensors: {e}")
    print("Make sure you've built and installed the C++ extensions:")
    print("1. cd flashtensors")
    print("2. python setup.py build_ext --inplace")
    print("3. python setup.py bdist_wheel")
    print("4. pip install dist/*.whl")
    sys.exit(1)


def main():
    # Step 1: Configure FlashEngine
    print("🔧 Configuring FlashEngine...")

    flash.configure(
        storage_path="/tmp/models",   # Where models will be stored
        mem_pool_size=1024**3*30,               # 170GB memory pool (GPU Size)
        chunk_size=1024**2*32,                   # 32MB chunks
        num_threads=8,                           # Number of threads
        gpu_memory_utilization=0.8,             # Use 80% of GPU memory
        server_host="0.0.0.0",                  # gRPC server host
        server_port=8073                        # gRPC server port
    )

    print("✅ FlashEngine configured successfully")

    # Step 2: Transform a model to fast-loading format
    model_id = "Qwen/Qwen3-0.6B"  
    
    print(f"\n🔄 Transforming model {model_id}...")
    result = flash.register_model(
        model_id=model_id,
        backend="transformers", # We should have an "auto" backend option
        torch_dtype="float16",
        force=False,  # Don't overwrite if already exists
        hf_token=None  # Add HuggingFace token if needed for private models
    )
    
    print(f"✅ Model transformation completed:")
    print(f"   Status: {result['status']}")
    print(f"   Path: {result.get('path', 'N/A')}")
    
    if "metrics" in result:
        metrics = result["metrics"]
        print(f"   Download time: {metrics['download_time']:.2f}s")
        print(f"   Transform time: {metrics['transform_time']:.2f}s") 
        print(f"   Total time: {metrics['total_time']:.2f}s")
        print(f"   Model size: {metrics['model_size'] / (1024**3):.2f} GB")
    
    # Step 3: Load model with ultra-fast loading
    print(f"\n⚡ Loading model {model_id} with fast loading...")
    import time
    load_start_time = time.time()
    
    model, tokenizer = flash.load_model(
        model_id=model_id,
        backend="transformers",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    load_time = time.time() - load_start_time
    print(f"✅ Model loaded successfully with fast loading in {load_time:.2f}s")
    
    # Step 4: Use the model for inference
    print("\n🤖 Running inference...")
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    print("📝 Generated outputs:")
    for prompt in prompts:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text∏
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        generated_only = generated_text[len(prompt):].strip()
        
        print(f"   Prompt: {prompt!r}")
        print(f"   Generated: {generated_only!r}")
        print()
    
    # Step 5: Clean up GPU memory
    print("🧹 Cleaning up GPU memory...")
    flash.cleanup_gpu()
    print("✅ GPU memory cleaned")

    flash.shutdown_server()
    
    # Step 6: Show model information
    print("\n📊 Model information:")
    info = flash.get_model_info(model_id)
    if info:
        print(f"   Model ID: {info['model_id']}")
        print(f"   Backend: {info['backend']}")
        print(f"   Size: {info['size'] / (1024**3):.2f} GB")
        print(f"   Ranks: {info['ranks']}")
    
    # Step 7: List all models
    print("\n📋 All available models:")
    models = flash.list_models()
    for model_key, model_info in models.items():
        print(f"   {model_key}: {model_info['size'] / (1024**3):.2f} GB")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        flash.shutdown_server()
