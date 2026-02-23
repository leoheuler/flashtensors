from ._cpu import CpuLoader
from ._cuda import CudaLoader


def _kvikio_available():
    """Return True only when kvikio is installed and GDS drivers are active."""
    try:
        import kvikio
        return not kvikio.defaults.compat_mode()
    except Exception:
        return False


def get_loader(devices):
    """Return the appropriate loader for the given set of (device_type, device_id) pairs."""
    gpu_devices = {(dt, did) for dt, did in devices if dt == "cuda"}
    mps_devices = {(dt, did) for dt, did in devices if dt == "mps"}
    has_cpu = any(dt == "cpu" for dt, _ in devices)

    if mps_devices and not gpu_devices:
        from ._mps import MpsLoader
        return MpsLoader()

    if len(gpu_devices) == 1 and not has_cpu:
        device_id = next(iter(gpu_devices))[1]
        if _kvikio_available():
            from ._gds import GdsLoader
            return GdsLoader(device_id=device_id)
        return CudaLoader(device_id=device_id)
    return CpuLoader()
