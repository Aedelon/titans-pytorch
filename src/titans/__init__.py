# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans: Learning to Memorize at Test Time

A PyTorch implementation of the Titans architecture from Google Research.
Titans introduce a neural long-term memory module that learns to memorize
historical context and helps attention attend to current context while
utilizing long past information.

Reference:
    Behrouz, A., Zhong, P., & Mirrokni, V. (2024).
    Titans: Learning to Memorize at Test Time.
    arXiv preprint arXiv:2501.00663
"""

# Core modules (no circular dependencies)
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.persistent import PersistentMemory
from titans.attention import SegmentedAttention, SlidingWindowAttention
from titans.models import (
    TitansLMM,
    TitansMAC,
    TitansMAG,
    TitansMAL,
)

# Optional modules (may have circular dependencies, import lazily)
from titans.flash_attention import (
    FlashSegmentedAttention,
    FlashSlidingWindowAttention,
    is_flash_attention_available,
    is_torch_sdpa_available,
)
from titans.triton_kernels import is_triton_available

# GPU optimization modules
from titans.cuda_optimizations import (
    compile_model,
    compile_function,
    CUDAPrefetcher,
    benchmark_model,
    get_optimal_batch_size,
)
from titans.optimized_training import (
    OptimizedTrainer,
    OptimizedTrainingConfig,
    enable_gradient_checkpointing,
    memory_efficient_attention,
)
from titans.optimized_inference import (
    OptimizedGenerator,
    StaticKVCache,
    KVCacheConfig,
    ContinuousBatcher,
    benchmark_generation,
)

# Hub must be imported after models to avoid circular imports
from titans.hub import load_from_hub, push_to_hub

__version__ = "0.1.0"
__author__ = "Delanoe Pirard / Aedelon"
__license__ = "Apache-2.0"

__all__ = [
    # Config
    "TitansConfig",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    # Attention
    "SlidingWindowAttention",
    "SegmentedAttention",
    # Flash Attention
    "FlashSlidingWindowAttention",
    "FlashSegmentedAttention",
    "is_flash_attention_available",
    "is_torch_sdpa_available",
    # Triton
    "is_triton_available",
    # Hub
    "push_to_hub",
    "load_from_hub",
    # Persistent Memory
    "PersistentMemory",
    # Models
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
    # GPU Optimizations
    "compile_model",
    "compile_function",
    "CUDAPrefetcher",
    "benchmark_model",
    "get_optimal_batch_size",
    # Optimized Training
    "OptimizedTrainer",
    "OptimizedTrainingConfig",
    "enable_gradient_checkpointing",
    "memory_efficient_attention",
    # Optimized Inference
    "OptimizedGenerator",
    "StaticKVCache",
    "KVCacheConfig",
    "ContinuousBatcher",
    "benchmark_generation",
]
