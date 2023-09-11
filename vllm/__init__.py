"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster
from vllm.entrypoints.llm import LLM
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

__version__ = "0.1.7"

__all__ = [
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_cluster",
]

import os
import torch

BASE_DIR = os.path.dirname(__file__)
LIB_DIR = os.path.join(BASE_DIR, "../build/lib")

if not os.path.exists(LIB_DIR) or not os.path.exists(f"{LIB_DIR}/libvllm_rocm.so"):
    raise RuntimeError(
        f"Could not find the ROCm VLLM library libvllm_rocm.so at {LIB_DIR}. "
        "Please build the ROCm library first or put it at the right place."
    )

torch.ops.load_library(os.path.join(LIB_DIR, "libvllm_rocm.so"))