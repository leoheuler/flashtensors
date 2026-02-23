import os
import threading
import queue

import torch
import numpy as np

from ._base import BaseLoader
from ._common import (
    _libc,
    _has_posix_fadvise,
    POSIX_FADV_SEQUENTIAL,
    _BLOCK_SIZE,
    _align_up,
    _pread_all,
    _resolve_device,
    _numel,
)


def _sllm_pipeline(fd, file_size, device_id, chunk_size, num_readers, cupy, use_o_direct=False):
    """Multi-reader pipeline: disk -> pinned -> GPU with overlapping I/O and DMA.

    *num_readers* threads use pread() to read directly into pinned buffers
    (zero intermediate copies).  A single GPU worker drains filled buffers
    via async memcpyAsync on a dedicated CUDA stream.

    Returns a CuPy device allocation containing the full file contents.
    """
    chunk_size = min(chunk_size, file_size)
    num_buffers = num_readers + 2  # enough for full overlap

    # Pinned (page-locked) host buffers
    pinned = [cupy.cuda.alloc_pinned_memory(chunk_size) for _ in range(num_buffers)]

    # Contiguous GPU destination for the entire file
    with cupy.cuda.Device(device_id):
        gpu_pool = cupy.cuda.alloc(file_size)
        stream = cupy.cuda.Stream(non_blocking=True)
        events = [cupy.cuda.Event() for _ in range(num_buffers)]

    # Build chunk list: (file_offset, length)
    chunks = []
    off = 0
    while off < file_size:
        cl = min(chunk_size, file_size - off)
        chunks.append((off, cl))
        off += cl

    # ready_q: readers -> GPU worker  (buf_idx, file_offset, chunk_len)
    # free_q:  GPU worker -> readers  (buf_idx released for reuse)
    ready_q = queue.Queue(maxsize=num_buffers)
    free_q = queue.Queue(maxsize=num_buffers)
    for i in range(num_buffers):
        free_q.put(i)

    # Work queue for reader threads
    work_q = queue.Queue()
    for chunk in chunks:
        work_q.put(chunk)

    _SENTINEL = None  # signals "reader is done" on ready_q
    errors = []

    def reader():
        try:
            while True:
                try:
                    file_off, chunk_len = work_q.get_nowait()
                except queue.Empty:
                    return
                buf_idx = free_q.get()  # wait for a free pinned buffer
                read_count = min(_align_up(chunk_len, _BLOCK_SIZE), file_size - file_off) if use_o_direct else chunk_len
                _pread_all(fd, pinned[buf_idx].ptr, read_count, file_off)
                ready_q.put((buf_idx, file_off, chunk_len))
        except Exception as e:
            errors.append(e)
        finally:
            ready_q.put(_SENTINEL)

    def gpu_worker():
        try:
            in_flight = {}   # chunk_index -> buf_idx
            chunk_idx = 0
            done_readers = 0
            with cupy.cuda.Device(device_id):
                while True:
                    # Reclaim oldest buffer if all are in flight
                    if len(in_flight) >= num_buffers:
                        oldest_ci = min(in_flight)
                        oldest_buf = in_flight.pop(oldest_ci)
                        events[oldest_buf].synchronize()
                        free_q.put(oldest_buf)

                    item = ready_q.get()  # block until data or sentinel

                    if item is _SENTINEL:
                        done_readers += 1
                        if done_readers == num_readers:
                            break
                        continue

                    buf_idx, file_off, chunk_len = item

                    cupy.cuda.runtime.memcpyAsync(
                        gpu_pool.ptr + file_off,
                        pinned[buf_idx].ptr,
                        chunk_len,
                        1,  # cudaMemcpyHostToDevice
                        stream.ptr,
                    )
                    events[buf_idx].record(stream)
                    in_flight[chunk_idx] = buf_idx
                    chunk_idx += 1

                # Drain remaining in-flight transfers
                for ci in sorted(in_flight):
                    bi = in_flight[ci]
                    events[bi].synchronize()
                    free_q.put(bi)
        except Exception as e:
            errors.append(e)

    # Launch threads
    reader_threads = [
        threading.Thread(target=reader, daemon=True) for _ in range(num_readers)
    ]
    gpu_t = threading.Thread(target=gpu_worker, daemon=True)

    for t in reader_threads:
        t.start()
    gpu_t.start()

    for t in reader_threads:
        t.join()
    gpu_t.join()

    if errors:
        raise errors[0]

    return gpu_pool


def _carve_gpu_pool(gpu_pool, layout, device_id, device_map, cupy):
    """Slice the contiguous GPU pool into individual named PyTorch tensors."""
    result = {}
    with cupy.cuda.Device(device_id):
        for ref in layout:
            device_type, _ = _resolve_device(ref.name, device_map)
            if device_type != "cuda":
                continue  # handled separately

            is_bfloat16 = ref.dtype in ("bfloat16", "torch.bfloat16")
            np_dtype = np.dtype("uint16") if is_bfloat16 else ref.numpy_dtype()

            # numel from shape (more reliable than ref.size for non-contiguous tensors)
            numel = _numel(ref)
            nbytes = numel * np_dtype.itemsize

            # UnownedMemory: zero-copy view into the pool; owner= keeps pool alive
            mem = cupy.cuda.UnownedMemory(gpu_pool.ptr + ref.offset, nbytes, owner=gpu_pool)
            memptr = cupy.cuda.MemoryPointer(mem, 0)
            gpu_arr = cupy.ndarray(tuple(ref.shape), dtype=np_dtype, memptr=memptr)
            tensor = torch.utils.dlpack.from_dlpack(gpu_arr)

            if is_bfloat16:
                tensor = tensor.view(torch.bfloat16)

            expected_stride = list(ref.stride)
            if list(tensor.stride()) != expected_stride:
                tensor = torch.as_strided(tensor, ref.shape, expected_stride)

            result[ref.name] = tensor

    return result


class CudaLoader(BaseLoader):
    def __init__(self, device_id):
        self._device_id = device_id

    def load(self, data_path, file_size, layout, device_map, num_workers, chunk_size):
        import cupy

        assert chunk_size % _BLOCK_SIZE == 0, f"chunk_size must be a multiple of {_BLOCK_SIZE}"
        try:
            fd = os.open(data_path, os.O_RDONLY | os.O_DIRECT)
            use_o_direct = True
        except OSError:
            fd = os.open(data_path, os.O_RDONLY)
            use_o_direct = False
        try:
            if not use_o_direct and _has_posix_fadvise:
                _libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL)
            gpu_pool = _sllm_pipeline(
                fd, file_size, self._device_id, chunk_size, num_workers, cupy,
                use_o_direct=use_o_direct,
            )
        finally:
            os.close(fd)

        return _carve_gpu_pool(gpu_pool, layout, self._device_id, device_map, cupy)
