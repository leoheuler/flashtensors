from ._cpu import CpuLoader
from ._cuda import CudaLoader


def get_loader(devices):
    """Return the appropriate loader for the given set of (device_type, device_id) pairs."""
    gpu_devices = {(dt, did) for dt, did in devices if dt == "cuda"}
    has_cpu = any(dt == "cpu" for dt, _ in devices)

    if len(gpu_devices) == 1 and not has_cpu:
        return CudaLoader(device_id=next(iter(gpu_devices))[1])
    return CpuLoader()
