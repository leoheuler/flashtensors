import os
import sys
import ctypes
import ctypes.util

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

# posix_fadvise is not available on macOS/Darwin
_has_posix_fadvise = sys.platform != "darwin" and hasattr(_libc, "posix_fadvise")

if _has_posix_fadvise:
    # int posix_fadvise(int fd, off_t offset, off_t len, int advice)
    _libc.posix_fadvise.restype = ctypes.c_int
    _libc.posix_fadvise.argtypes = [
        ctypes.c_int,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
    ]


_BLOCK_SIZE = 4096  # O_DIRECT alignment (covers both 512-byte and 4Kn sectors)


def _align_up(n, alignment):
    return (n + alignment - 1) & ~(alignment - 1)


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
        if dev == "mps":
            return ("mps", None)
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
