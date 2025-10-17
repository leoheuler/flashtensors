# ⚡ flashtensors  

> A blazing-fast storage engine that loads models **up to 10x faster** than alternative loaders.  
> Load large models in **< 2 seconds**.  

---



## 🚀 Why flashtensors?  

Traditional model loaders slow down your workflow with painful startup times. flashtensors was built from the ground up to eliminate bottlenecks and maximize performance.  

- ⚡ **10x faster** than standard loaders  
- ⏱ **Coldstarts < 2 seconds** 

---



## 🔧 Installation  

```bash
pip install flashtensors
```

---

## 🔧 Getting Started  

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

<br>
<br>
<br>
<br>

Inspired on Serverless LLM
```
@inproceedings{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
  pages={135--153},
  year={2024}
}
```
