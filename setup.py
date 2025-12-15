import sys
from pathlib import Path

from setuptools import setup, find_packages

sys.path.append(Path.cwd().as_posix())


def fetch_requirements(path):
    """Load requirements from file."""
    try:
        with open(path, "r") as fd:
            return [
                r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")
            ]
    except FileNotFoundError:
        raise RuntimeError("requirements.txt file not found")


# Load requirements
install_requires = fetch_requirements("requirements.txt")

# Setup configuration
setup(
    name="flashtensors",
    version="0.0.1",
    author="Flash Tensors Team",
    author_email="team@flashtensors.com",
    description="Ultra-fast AI model loading library with CUDA acceleration",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.13",
    package_data={
        "flashtensors": ["*.so", "*.pyd", "py.typed", "*.conf"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="AI, machine learning, model loading, CUDA, GPU acceleration",
    entry_points={
        "console_scripts": [
            "flash=cli.flash:cli",
        ],
    },
)
