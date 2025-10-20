import torch

def cuda_status():
    """Check if CUDA is available"""
    return torch.cuda.is_available()

def device_count():
    """Get number of CUDA devices"""
    return torch.cuda.device_count()

def device_name():
    """Get name of the first CUDA device"""
    if not cuda_status():
        return None
    return torch.cuda.get_device_name(0)
