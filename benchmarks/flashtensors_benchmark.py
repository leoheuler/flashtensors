import torch
from torch import nn
import flashtensors

def get_gpu_status():
    import GPUtil

    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id} — {gpu.name}")
        print(f"  Load: {gpu.load*100:.3f}%")
        print(f"  Free Memory: {gpu.memoryFree / 1024:.3f} GB")
        print(f"  Used Memory: {gpu.memoryUsed / 1024:.3f} GB")
        print(f"  Total Memory: {gpu.memoryTotal / 1024:.3f} GB")


def get_cpu_status():
    import psutil

    mem = psutil.virtual_memory()

    print(f"Total RAM:     {mem.total / 1024**3:.3f} GB")
    print(f"Available RAM: {mem.available / 1024**3:.3f} GB")
    print(f"Used RAM:      {mem.used / 1024**3:.3f} GB")
    print(f"Free RAM:      {mem.free / 1024**3:.3f} GB")


def test_create_flashtensor():
    if not torch.cuda.is_available():
        return False

    device = torch.device("cuda")

    # Input Tensor
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    x = x.to(device)

    # Base Module
    module = nn.Linear(3, 1024**2)
    module = module.to(device)

    base_result = module(x)
    get_gpu_status()
    module.to("cpu")

    get_gpu_status()

    flash_tensor = FlashTensor(module)
    # flash_tensor = flash_tensor.to(device)

    # flash_result = flash_tensor(x)

    # assert_equal = torch.allclose(base_result, flash_result)

    # assert assert_equal

    # get_gpu_status()
    # get_cpu_status()

    return True
