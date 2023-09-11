import io
import os
import re
import subprocess
from typing import List, Set

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

# Collect the compute capabilities of all available GPUs.
device_count = torch.cuda.device_count()
compute_capabilities: Set[int] = set()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 7:
        raise RuntimeError(
            "GPUs with compute capability less than 7.0 are not supported.")
    compute_capabilities.add(major * 10 + minor)
    
BASE_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(BASE_DIR, "./build/lib")
if not os.path.exists(LIB_DIR) or not os.path.exists(f"{LIB_DIR}/libvllm_rocm.so"):
    raise RuntimeError(
        f"Could not find the ROCm VLLM library libvllm_rocm.so at {LIB_DIR}. "
        "Please build the ROCm library first or put it at the right place."
    )

def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="vllm-rocm",
    version=find_version(get_path("vllm", "__init__.py")),
    author="PKU Supercomputing Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs, adapted from VLLM."),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/leavelet/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
        "Documentation": "https://vllm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("benchmarks", "csrc", "docs",
                                               "examples", "tests")),
    python_requires=">=3.8",
    install_requires=get_requirements()
)
