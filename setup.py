import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

sys.path.append(Path.cwd().as_posix())

try:
    torch_available = True
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import ROCM_HOME
except Exception:
    torch_available = False
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."
    )

ROOT_DIR = os.path.dirname(__file__)

def check_nvcc_installed(cuda_home: str) -> None:
    """Check if nvcc (NVIDIA CUDA compiler) is installed."""
    try:
        _ = subprocess.check_output(
            [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
        )
    except Exception:
        raise RuntimeError(
            "nvcc is not installed or not found in your PATH. "
            "Please ensure that the CUDA toolkit is installed and nvcc is available in your PATH."
        ) from None

def check_hipcc_installed(rocm_home: str) -> None:
    """Check if hipcc (AMD HIP compiler) is installed."""
    hipcc_paths = [rocm_home + "/bin/hipcc", rocm_home + "/hip/bin/hipcc"]
    for hipcc in hipcc_paths:
        try:
            _ = subprocess.check_output(
                [hipcc, "--version"], universal_newlines=True
            )
            return
        except Exception:
            continue
    raise RuntimeError(
        "hipcc is not installed or not found in your PATH. "
        "Please ensure that the HIP toolkit is installed and hipcc is available in your PATH."
    ) from None

if torch_available:
    if CUDA_HOME is not None:
        check_nvcc_installed(CUDA_HOME)
    elif ROCM_HOME is not None:
        check_hipcc_installed(ROCM_HOME)
    else:
        raise RuntimeError(
            "CUDA_HOME or ROCM_HOME environment variable must be set to compile CUDA or HIP extensions."
        )

def is_ninja_available() -> bool:
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

class CustomInstall(install):
    """Custom installation to ensure extensions are built before installation."""

    def run(self):
        self.run_command("build_ext")
        super().run()

def fetch_requirements(path):
    """Load requirements from file."""
    try:
        with open(path, "r") as fd:
            return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith('#')]
    except FileNotFoundError:
        raise RuntimeError(
            "requirements.txt file not found"
        )

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(p, "r", encoding="utf-8").read()
    else:
        return ""

class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class cmake_build_ext(build_ext):
    did_config: Dict[str, bool] = {}

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Check for debug build environment variable
        debug_build = os.getenv("DEBUG_BUILD", "0") == "1"
        default_cfg = "Debug" if (self.debug or debug_build) else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DSTORE_PYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
        ]

        verbose = bool(int(os.getenv('VERBOSE', '0')))
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(4),
            ]
        else:
            build_tool = []

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)

            #ext_target_name = remove_prefix(ext.name, "flashtensors.")
            ext_target_name = ext.name.split(".")[-1]
            
            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",  # Use all available cores
            ]

            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

# Load requirements
install_requires = fetch_requirements("requirements.txt")

# Setup configuration
setup(
    name="flashtensors",
    version="0.1.0",
    author="Teil Engine Team",
    author_email="team@flashtensors.com",
    description="Ultra-fast AI model loading library with CUDA acceleration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[
        CMakeExtension(name="flashtensors._C"),
        CMakeExtension(name="flashtensors._checkpoint_store"),
    ],
    cmdclass={
        "build_ext": cmake_build_ext,
        "install": CustomInstall,
    },
    install_requires=install_requires,
    python_requires=">=3.8",
    package_data={
        "flashtensors": ["*.so", "*.pyd", "py.typed", "*.conf"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="AI, machine learning, model loading, CUDA, GPU acceleration",
    entry_points={
        'console_scripts': [
            'teil=cli.teil:cli',
        ],
    },
)
