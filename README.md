

<div align="center">
        <img src="https://i.imgur.com/wg5VTWn.png"> 
    <br>
        <br>
         <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%E2%80%933.12-blue"></a>
    <a href="https://discord.gg/5usuKkmT"><img alt="Discord" src="https://img.shields.io/discord/1436829434815578114?logo=discord&label=join&link=https%3A%2F%2Fdiscord.gg%2F5usuKkmT"></a>
        <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
<br>
<h2> Run 100 Large Models on a single GPU with minimal impact to Time to First Token. </h2>
        <br>
<img src="https://i.imgur.com/r0hbTJ6.png">
</div>

> A blazing-fast inference engine that loads models from SSD to GPU VRAM **up to 10x faster** than alternative loaders.  
> Hotswap large models in **< 2 seconds**.
>
>  <b>Acknowledgement: This project includes substantial code originally developed by
<a href="https://github.com/ServerlessLLM/ServerlessLLM"> ServerlessLLM </a>,
used under the Apache License, Version 2.0. </b>

---



## 🚀 Why flashtensors?  

Traditional model loaders slow down your workflow with painful startup times. flashtensors was built from the ground up to eliminate bottlenecks and maximize performance.  

- ⚡ **Up to 10x faster** than standard loaders  
- ⏱ **Coldstarts < 2 seconds**

The result: An inference engine that scales by usage not by model.

- Host hundreds of models in a single device, and hot-swap them on demand with low to none effect on user experience.
- Run Agentic workflows on constrained devices (like robots, wearables, etc)

Use cases: 
- Affordable Personalized AI
- Serverless AI Inference
- On Prem Deployments
- Robotics
- Local Inference
---



## 🔧 Installation  

```bash
pip install git+https://github.com/leoheuler/flashtensors.git
```

---

# Getting Started 

## Using the command line
``` bash
# Start the daemon server
flash start
```

``` bash
# Pull the model of your preference
flash pull Qwen/Qwen3-0.6B
```

``` bash
# Run the model
flash run Qwen/Qwen3-0.6B "Hello world"
```

## Using the SDK
### vllm
``` Python
import flashtensors as ft
from vllm import SamplingParams
import time

ft.shutdown_server()  # Ensure any existing server is shut down

ft.configure(
    storage_path="/tmp/models",   # Where models will be stored
    mem_pool_size=1024**3*30,                # 30GB memory pool (GPU Size)
    chunk_size=1024**2*32,                   # 32MB chunks
    num_threads=4,                           # Number of threads
    gpu_memory_utilization=0.8,             # Use 80% of GPU memory
    server_host="0.0.0.0",                # gRPC server host
    server_port=8073                        # gRPC server port
)

ft.activate_vllm_integration()

# Step 2: Transform a model to fast-loading format
model_id = "Qwen/Qwen3-0.6B"  

result = ft.register_model(
    model_id=model_id,
    backend="vllm", # We should have an "auto" backend option
    torch_dtype="bfloat16",
    force=False,  # Don't overwrite if already exists
    hf_token=None  # Add HuggingFace token if needed for private models
)

# Step 3: Load model with ultra-fast loading
print(f"\n⚡ Loading model {model_id} with fast loading...")

load_start_time = time.time()

llm = ft.load_model(
    model_id=model_id,
    backend="vllm",
    dtype="bfloat16",
    gpu_memory_utilization=0.8
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

sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"   Prompt: {prompt!r}")
    print(f"   Generated: {generated_text!r}")
    print()

# Step 5: Clean up GPU memory
ft.cleanup_gpu()

# Step 6: Show model information
print("\n📊 Model information:")
info = ft.get_model_info(model_id)
if info:
    print(f"   Model ID: {info['model_id']}")
    print(f"   Backend: {info['backend']}")
    print(f"   Size: {info['size'] / (1024**3):.2f} GB")
    print(f"   Ranks: {info['ranks']}")

# Step 7: List all models
print("\n📋 All available models:")
models = ft.list_models()
for model_key, model_info in models.items():
    print(f"   {model_key}: {model_info['size'] / (1024**3):.2f} GB")

```

### Custom models

``` Python
from flashtensors import flash

class SimpleModel(nn.Module):
    def __init__(self, size=(3,3)):
        super(SimpleModel, self).__init__()
        # Create a single parameter tensor of shape (3, 3)
        self.weight = nn.Parameter(torch.randn(*size))
        
    def forward(self, x):
        return x @ self.weight  # Simple matrix multiplication

model = SimpleModel()

state_dict = model.state_dict()

# Save your state dict
flash.save_dict(state_dict, "/your/model/folder")


# Load your state dict blazing fast
device_map =  {"":0}
new_state_dict = flash.load_dict("/your/model/folder", device_map)

```

---


## 📊 Benchmarks  

flashtensors drastically reduces coldstart times compared to alternative loaders like safetensors.  

| Model            | flashtensors ⚡ (s) | safetensors (mmap) (s) | Speedup |
|------------------|------------|----------|---------|
| Qwen/Qwen3-0.6B  | **2.74**   | 11.68    | ~4.3×   |
| Qwen/Qwen3-4B    | **2.26**   | 8.54     | ~3.8×   |
| Qwen/Qwen3-8B    | **2.57**   | 9.08     | ~3.5×   |
| Qwen/Qwen3-14B   | **3.02**   | 12.91    | ~4.3×   |
| Qwen/Qwen3-32B   | **4.08**   | 24.05    | ~5.9×   |

(Results measured on H100 GPUs using NVLink)
⚡ **Average speedup: ~4–6× faster model loads**  
Coldstarts stay consistently under **5 seconds**, even for **32B parameter models**.  

## Roadmap:
- Run benchmarks on a diversity of hardware
- Docker Integration
- Inference Server
- SGLang Integration
- LlamaCPP Integration
- Dynamo Integration
- Ollama Integration

Credits: 
- Inspired and adapted from the great work of [ServerlessLLM](https://github.com/leoheuler/flashtensors/CREDITS.md)
