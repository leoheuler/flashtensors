import numpy as np

from ._base import BaseLoader
from ._cuda import _carve_gpu_pool


def _gds_read(data_path, file_size, device_id, chunk_size, cupy):
    """Read file directly into GPU memory via kvikio GDS.

    Uses chunked async reads (pread -> IOFuture) so the GDS driver can
    pipeline DMA transfers.  Returns a CuPy device allocation containing
    the full file contents.
    """
    import kvikio

    with cupy.cuda.Device(device_id):
        gpu_pool = cupy.cuda.alloc(file_size)
        # Wrap as a uint8 ndarray so kvikio gets __cuda_array_interface__
        gpu_arr = cupy.ndarray(file_size, dtype=np.uint8, memptr=cupy.cuda.MemoryPointer(
            cupy.cuda.UnownedMemory(gpu_pool.ptr, file_size, owner=gpu_pool), 0,
        ))

    futures = []
    with kvikio.CuFile(data_path, "r") as f:
        offset = 0
        while offset < file_size:
            length = min(chunk_size, file_size - offset)
            buf_slice = gpu_arr[offset:offset + length]
            futures.append(f.pread(buf_slice, file_offset=offset))
            offset += length

        for fut in futures:
            fut.get()

    return gpu_pool


class GdsLoader(BaseLoader):
    def __init__(self, device_id):
        self._device_id = device_id

    def load(self, data_path, file_size, layout, device_map, num_workers, chunk_size):
        import cupy

        gpu_pool = _gds_read(data_path, file_size, self._device_id, chunk_size, cupy)
        return _carve_gpu_pool(gpu_pool, layout, self._device_id, device_map, cupy)
