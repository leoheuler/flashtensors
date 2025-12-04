from enum import Enum
from pydantic import BaseModel
import os
import time


_MEMORY_CLOCK_RATE_KEY = "memoryClockRate"
_MEMORY_BUS_WIDTH_KEY = "memoryBusWidth"
_GPU_NAME_KEY = "name"


class GPUSpecs(BaseModel):
    device_id: int
    gpu_name: str
    theoretical_bandwidth: float
    memory_clock_khz: int
    memory_bus_width: int


class DeviceType(Enum):
    SSD = "SSD"
    VOL = "VOL"


class SSDSpecs(BaseModel):
    device_type: DeviceType
    read_bandwidth: float
    write_bandwidth: float


class HardwareSpecs(BaseModel):
    gpu_specs: list[GPUSpecs]
    ssd_specs: list[SSDSpecs]


def hardware_stats() -> HardwareSpecs:
    """Get hardware specs
    Returns:
        HardwareSpecs: Hardware specs
    """
    return HardwareSpecs(
        gpu_specs=gpu_specs(),
        ssd_specs=ssd_specs(),
    )


def ssd_specs() -> list[SSDSpecs]:
    """Get SSD specs
    Returns:
        list[SSDSpecs]: List of SSD specs
    """

    specs = []

    current_directory = os.getcwd()

    # Test SSD
    test_file, write_bandwidth = test_write_bandwidth(path=current_directory)
    read_bandwidth = test_read_bandwidth(test_file)

    specs.append(
        SSDSpecs(
            device_type=DeviceType.SSD,
            read_bandwidth=read_bandwidth,
            write_bandwidth=write_bandwidth,
        )
    )

    # Test Mounted Volume
    test_file, write_bandwidth = test_write_bandwidth(path="/mnt/storage")
    read_bandwidth = test_read_bandwidth(test_file)

    specs.append(
        SSDSpecs(
            device_type=DeviceType.VOL,
            read_bandwidth=read_bandwidth,
            write_bandwidth=write_bandwidth,
        )
    )
    return specs


def gpu_specs() -> list[GPUSpecs]:
    """Get GPU specs
    Returns:
        list[GPUSpecs]: List of GPU specs
    """
    import cupy as cp

    specs = []
    device_count = cp.cuda.runtime.getDeviceCount()

    for device in range(device_count):
        device_properties = cp.cuda.runtime.getDeviceProperties(device)
        mem_clock_khz, mem_bus_width = (
            device_properties[_MEMORY_CLOCK_RATE_KEY],
            device_properties[_MEMORY_BUS_WIDTH_KEY],
        )
        gpu_name = device_properties[_GPU_NAME_KEY].decode()

        specs.append(
            GPUSpecs(
                device_id=device,
                theoretical_bandwidth=get_theoretical_gpu_bandwidth(
                    mem_clock_khz, mem_bus_width
                ),
                gpu_name=gpu_name,
                memory_clock_khz=mem_clock_khz,
                memory_bus_width=mem_bus_width,
            )
        )

    return specs


def get_theoretical_gpu_bandwidth(mem_clock_khz: int, mem_bus_width: int):
    """Get theoretical peak bandwidth from GPU specs
    Args:
        mem_clock_khz: Memory clock rate in kHz
        mem_bus_width: Memory bus width in bits
    Returns:
        Theoretical peak bandwidth in GB/s
    """

    # Bandwidth (GB/s) = (clock * bus_width * 2) / (8 * 1e6)
    return (mem_clock_khz * mem_bus_width * 2) / (8 * 1e6)


def test_write_bandwidth(
    path="/mnt/storage", size_mb=10000, block_size_mb=1
) -> tuple[str, float]:
    """
    Test write bandwidth (equivalent to dd write test)

    Args:
        path: Directory to write test file
        size_mb: Total size to write in MB
        block_size_mb: Block size in MB (default 1MB like dd bs=1M)
    """

    test_file = os.path.join(path, "test.img")
    block_size = block_size_mb * 1024 * 1024
    num_blocks = size_mb // block_size_mb

    # Create a block of zeros
    block = b"\x00" * block_size

    start_time = time.time()

    with open(test_file, "wb") as f:
        for i in range(num_blocks):
            f.write(block)

        f.flush()
        os.fsync(f.fileno())  # Equivalent to conv=fdatasync

    end_time = time.time()
    duration = end_time - start_time

    bandwidth_mbs = size_mb / duration
    bandwidth_gbs = bandwidth_mbs / 1024

    return test_file, bandwidth_gbs


def test_read_bandwidth(test_file, block_size_mb=1) -> float:
    """
    Test read bandwidth (equivalent to dd read test)

    Args:
        test_file: Path to test file
        block_size_mb: Block size in MB
    """
    block_size = block_size_mb * 1024 * 1024

    # Get file size
    file_size = os.path.getsize(test_file)
    size_mb = file_size / (1024 * 1024)

    start_time = time.time()

    bytes_read = 0
    with open(test_file, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            bytes_read += len(block)

    end_time = time.time()
    duration = end_time - start_time

    bandwidth_mbs = size_mb / duration
    bandwidth_gbs = bandwidth_mbs / 1024

    return bandwidth_gbs
