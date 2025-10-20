import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


def get_model_size_gb(model):
    """Calculate model size in GB"""
    total_params = sum(p.numel() for p in model.parameters())
    # Estimate size based on parameter count and dtype
    if hasattr(model.config, 'torch_dtype'):
        dtype = model.config.torch_dtype
    else:
        dtype = next(model.parameters()).dtype
    
    if dtype == torch.float16:
        bytes_per_param = 2
    elif dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.int8:
        bytes_per_param = 1
    else:
        bytes_per_param = 4  # Default assumption
    
    total_size_bytes = total_params * bytes_per_param
    return total_size_bytes / (1024**3)


def main():
    model_id = "Qwen/Qwen3-14B"
    
    print("🔧 Configuring standard PyTorch environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        device = "cuda"
    else:
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
    
    # Step 1: Load tokenizer
    print(f"\n📝 Loading tokenizer for {model_id}...")
    tokenizer_start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer_load_time = time.time() - tokenizer_start_time
    print(f"✅ Tokenizer loaded in {tokenizer_load_time:.2f}s")
    
    # Step 2: Load model with standard transformers
    print(f"\n⚡ Loading model {model_id} with standard loading...")
    model_start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Optimize memory usage during loading
    )
    
    model_load_time = time.time() - model_start_time
    model_size_gb = get_model_size_gb(model)
    
    print(f"✅ Model loaded successfully in {model_load_time:.2f}s")
    print(f"   Model size: {model_size_gb:.2f} GB")
    print(f"   Device: {next(model.parameters()).device}")
    print(f"   Data type: {next(model.parameters()).dtype}")
    
    # Step 3: Use the model for inference
    print("\n🤖 Running inference...")
    prompts = [
        "Hello, my name is",
        "The capital of France is", 
        "The future of AI is",
    ]
    
    print("📝 Generated outputs:")
    
    for prompt in prompts:
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            inference_start = time.time()
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            inference_time = time.time() - inference_start
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the output
        generated_only = generated_text[len(prompt):].strip()
        
        print(f"   Prompt: {prompt!r}")
        print(f"   Generated: {generated_only!r}")
        print(f"   Inference time: {inference_time:.2f}s")
        print()
    
    # Step 4: Show model information
    print("\n📊 Model information:")
    print(f"   Model ID: {model_id}")
    print(f"   Architecture: {model.config.model_type}")
    print(f"   Size: {model_size_gb:.2f} GB")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocabulary size: {model.config.vocab_size:,}")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Number of layers: {model.config.num_hidden_layers}")
    print(f"   Number of attention heads: {model.config.num_attention_heads}")
    print(f"   Max position embeddings: {model.config.max_position_embeddings:,}")
    
    # Step 5: Memory usage information
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        print(f"   Max allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    
    # Step 6: Clean up GPU memory
    print("\n🧹 Cleaning up memory...")
    del model
    del tokenizer
    
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"✅ Memory cleaned. Current GPU usage: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    else:
        print("✅ Memory cleaned")
    
    print("\n🎉 Script completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise
