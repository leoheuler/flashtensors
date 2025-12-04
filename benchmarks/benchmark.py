import modal
import logging
from benchmarks.device_benchmark import hardware_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("flashtensors-benchmark")
image = modal.Image.debian_slim().pip_install(
    "cupy-cuda12x", "pydantic", "psutil", "gputil", "torch"
)
volume = modal.Volume.from_name("storage", create_if_missing=True)


@app.function(gpu="a10g", image=image, volumes={"/mnt/storage": volume})
def run_on_a10g(function):
    return function()


@app.function(gpu="L4", image=image, volumes={"/mnt/storage": volume})
def run_on_L4(function):
    return function()


@app.function(gpu="A10", image=image, volumes={"/mnt/storage": volume})
def run_on_A10(function):
    return function()


@app.function(gpu="A100", image=image, volumes={"/mnt/storage": volume})
def run_on_A100(function):
    return function()


@app.function(gpu="A100-40GB", image=image, volumes={"/mnt/storage": volume})
def run_on_A100_40GB(function):
    return function()


@app.function(gpu="L40S", image=image, volumes={"/mnt/storage": volume})
def run_on_L40S(function):
    return function()


@app.function(gpu="H100", image=image, volumes={"/mnt/storage": volume})
def run_on_H100(function):
    return function()


@app.function(gpu="H200", image=image, volumes={"/mnt/storage": volume})
def run_on_H200(function):
    return function()


@app.function(gpu="B200", image=image, volumes={"/mnt/storage": volume})
def run_on_B200(function):
    return function()


GPU_MAP = {
    "a10g": run_on_a10g,
    "L4": run_on_L4,
    "A10": run_on_A10,
    "A100": run_on_A100,
    "A100-40GB": run_on_A100_40GB,
    "L40S": run_on_L40S,
    # I am not rich!
    # "H100": run_on_H100,
    # "H200": run_on_H200,
    # "B200": run_on_B200,
}

FUNCTION_MAP = {
    "hardware_stats": hardware_stats,
}


@app.local_entrypoint()
def main():
    for gpu in GPU_MAP:
        for function in FUNCTION_MAP:
            logger.info(f"Testing {function} on {gpu}")
            result = GPU_MAP[gpu].remote(FUNCTION_MAP[function])
            logger.info(f"{gpu}: {result}")
