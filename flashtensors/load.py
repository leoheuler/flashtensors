import os
import ctypes
import ctypes.util
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Union

import torch
import numpy as np

from .flash_state import FlashState

# ---------------------------------------------------------------------------
# libc helpers: pread + posix_fadvise
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# ssize_t pread(int fd, void *buf, size_t count, off_t offset)
_libc.pread.restype = ctypes.c_ssize_t
_libc.pread.argtypes = [
    ctypes.c_int,       # fd
    ctypes.c_void_p,    # buf
    ctypes.c_size_t,    # count
    ctypes.c_long,      # offset (off_t)
]

POSIX_FADV_SEQUENTIAL = 2

# int posix_fadvise(int fd, off_t offset, off_t len, int advice)
_libc.posix_fadvise.restype = ctypes.c_int
_libc.posix_fadvise.argtypes = [
    ctypes.c_int,
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_int,
]


def _pread_all(fd, buf_ptr, count, offset):
    """Read exactly *count* bytes from *fd* at *offset* into *buf_ptr*.

    Uses libc pread (thread-safe, no shared file position).  Loops to handle
    short reads which are allowed by POSIX.
    """
    total = 0
    while total < count:
        n = _libc.pread(fd, buf_ptr + total, count - total, offset + total)
        if n < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        if n == 0:
            raise EOFError(f"pread: unexpected EOF after {total}/{count} bytes")
        total += n


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(name: str, device_map: dict):
    """Resolve the target device for a tensor given a device_map.

    Per-tensor keys take precedence over the default key "".
    Returns a (device_type, device_id) tuple where device_type is "cpu" or "cuda".
    """
    if name in device_map:
        dev = device_map[name]
    elif "" in device_map:
        dev = device_map[""]
    else:
        dev = "cpu"

    if isinstance(dev, int):
        return ("cuda", dev)
    if isinstance(dev, str):
        if dev == "cpu":
            return ("cpu", None)
        if dev.startswith("cuda"):
            if ":" in dev:
                return ("cuda", int(dev.split(":")[1]))
            return ("cuda", 0)
    return ("cpu", None)


def _numel(ref):
    n = 1
    for s in ref.shape:
        n *= s
    return n


# ---------------------------------------------------------------------------
# SLLM pipeline: multi-reader pread → pinned → GPU DMA
# ---------------------------------------------------------------------------

def _sllm_pipeline(fd, file_size, device_id, chunk_size, num_readers, cupy):
    """Multi-reader pipeline: disk → pinned → GPU with overlapping I/O and DMA.

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

    # ready_q: readers → GPU worker  (buf_idx, file_offset, chunk_len)
    # free_q:  GPU worker → readers  (buf_idx released for reuse)
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
                _pread_all(fd, pinned[buf_idx].ptr, chunk_len, file_off)
                ready_q.put((buf_idx, file_off, chunk_len))
        except Exception as e:
            errors.append(e)
        finally:
            ready_q.put(_SENTINEL)

    def gpu_worker():
        try:
            in_flight = {}   # chunk_index → buf_idx
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


# ---------------------------------------------------------------------------
# CPU fallback: pread directly into tensor storage
# ---------------------------------------------------------------------------

def _load_tensor_pread(ref, fd, device_map, cupy=None):
    """Load a single tensor via pread. No mmap, no intermediate bytes objects."""
    device_type, device_id = _resolve_device(ref.name, device_map)
    is_bfloat16 = ref.dtype in ("bfloat16", "torch.bfloat16")
    np_dtype = np.dtype("uint16") if is_bfloat16 else ref.numpy_dtype()
    numel = _numel(ref)

    if device_type == "cuda":
        mem = cupy.cuda.alloc_pinned_memory(ref.size)
        _pread_all(fd, mem.ptr, ref.size, ref.offset)
        arr = np.frombuffer(mem, dtype=np_dtype, count=numel)

        with cupy.cuda.Device(device_id):
            gpu_arr = cupy.asarray(arr).reshape(ref.shape)
            tensor = torch.utils.dlpack.from_dlpack(gpu_arr)

        if is_bfloat16:
            tensor = tensor.view(torch.bfloat16)

    else:
        # Allocate torch tensor then pread directly into its storage
        torch_dtype = torch.uint16 if is_bfloat16 else ref.torch_dtype()
        tensor = torch.empty(numel, dtype=torch_dtype)
        _pread_all(fd, tensor.data_ptr(), ref.size, ref.offset)
        tensor = tensor.reshape(ref.shape)
        if is_bfloat16:
            tensor = tensor.view(torch.bfloat16)

    expected_stride = list(ref.stride)
    if list(tensor.stride()) != expected_stride:
        tensor = torch.as_strided(tensor, ref.shape, expected_stride)

    return ref.name, tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dict(
    model_path: Union[str, os.PathLike],
    device_map: dict,
    num_workers: int = 4,
    chunk_size: int = 256 * 1024 * 1024,   # 256 MB — matches SLLM default
) -> Dict[str, torch.Tensor]:
    """Load tensors from a .flashtensors checkpoint.

    Uses the ServerlessLLM-style multi-reader pipeline when all GPU tensors
    target the same device: disk → pinned memory → GPU DMA overlap across chunks.

    Falls back to a parallel per-tensor approach for mixed-device maps or
    CPU-only loads.

    Args:
        model_path:  Directory containing tensor.flashtensors and tensor_index.json.
        device_map:  Maps tensor names (or "" for default) to devices.
                     e.g. {"": 0}      — all tensors → GPU 0
                          {"": "cpu"}  — all tensors → CPU
                          {"a": 0, "b": "cpu"} — per-tensor
        num_workers: Worker threads for the parallel fallback path.
        chunk_size:  Pinned-buffer chunk size (bytes) for the pipeline path.

    Returns:
        Dictionary mapping tensor names to PyTorch tensors on the requested devices.
    """
    index_path = os.path.join(model_path, "tensor_index.json")
    data_path = os.path.join(model_path, "tensor.flashtensors")

    with open(index_path, "r") as f:
        flash_state = FlashState(**json.load(f))

    devices = {_resolve_device(ref.name, device_map) for ref in flash_state.layout}
    gpu_devices = {(dt, did) for dt, did in devices if dt == "cuda"}
    has_cpu = any(dt == "cpu" for dt, _ in devices)

    file_size = os.path.getsize(data_path)

    # ── Fast path: all tensors go to the same GPU ─────────────────────────
    # Use the SLLM multi-reader pipeline for maximum disk→GPU throughput.
    if len(gpu_devices) == 1 and not has_cpu:
        import cupy
        _, device_id = next(iter(gpu_devices))

        fd = os.open(data_path, os.O_RDONLY)
        try:
            _libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL)
            gpu_pool = _sllm_pipeline(
                fd, file_size, device_id, chunk_size, num_workers, cupy,
            )
        finally:
            os.close(fd)

        return _carve_gpu_pool(gpu_pool, flash_state.layout, device_id, device_map, cupy)

    # ── Fallback: mixed devices or CPU-only ───────────────────────────────
    # Parallel per-tensor load with a thread pool using pread.
    needs_gpu = any(dt == "cuda" for dt, _ in devices)
    cupy = None
    if needs_gpu:
        import cupy as _cupy
        cupy = _cupy

    result = {}
    fd = os.open(data_path, os.O_RDONLY)
    try:
        _libc.posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL)

        if num_workers == 1 or len(flash_state.layout) == 1:
            for ref in flash_state.layout:
                name, tensor = _load_tensor_pread(ref, fd, device_map, cupy)
                result[name] = tensor
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(_load_tensor_pread, ref, fd, device_map, cupy): ref
                    for ref in flash_state.layout
                }
                for future in as_completed(futures):
                    name, tensor = future.result()
                    result[name] = tensor
    finally:
        os.close(fd)

    return result
