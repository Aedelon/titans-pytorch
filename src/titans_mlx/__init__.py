# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans MLX Implementation - Optimized for Apple Silicon.

This module provides an MLX implementation of the Titans architecture
that produces identical results to the PyTorch implementation while
being optimized for Apple Silicon (M1/M2/M3/M4) GPUs.

MLX is Apple's array framework for machine learning on Apple silicon.
It provides:
- Unified memory architecture (no CPU/GPU transfers)
- Lazy evaluation for optimized computation graphs
- JIT compilation for maximum performance

Usage:
    import mlx.core as mx
    from titans_mlx import TitansConfig, TitansMAC

    config = TitansConfig(dim=512, num_heads=8, num_layers=6)
    model = TitansMAC(config)

    # Forward pass
    x = mx.random.randint(0, config.vocab_size, (2, 512))
    logits, states = model(x)
"""

from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory
from titans_mlx.attention import SegmentedAttention, SlidingWindowAttention
from titans_mlx.persistent import PersistentMemory
from titans_mlx.models import TitansMAC, TitansMAG, TitansMAL, TitansLMM
from titans_mlx.optimizations import (
    benchmark_function,
    chunked_attention,
    compile_function,
    compile_model,
    evaluate_all,
    fused_rmsnorm,
    fused_silu_gate,
    get_causal_mask,
    get_device_info,
    get_sliding_window_mask,
    OptimizedFeedForward,
    OptimizedMemoryMLP,
    rotary_embedding_optimized,
    scaled_dot_product_attention,
)
from titans_mlx.metal_kernels import (
    benchmark_metal_kernel,
    get_metal_kernel_info,
    metal_causal_attention,
    metal_memory_update,
    metal_rope,
    metal_silu_gate,
    MetalFeedForward,
    MetalRMSNorm,
    MetalRotaryEmbedding,
)

__version__ = "0.1.0"
__all__ = [
    # Config
    "TitansConfig",
    # Memory
    "NeuralLongTermMemory",
    "MemoryState",
    # Attention
    "SlidingWindowAttention",
    "SegmentedAttention",
    # Persistent Memory
    "PersistentMemory",
    # Models
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
    # Optimizations
    "benchmark_function",
    "chunked_attention",
    "compile_function",
    "compile_model",
    "evaluate_all",
    "fused_rmsnorm",
    "fused_silu_gate",
    "get_causal_mask",
    "get_device_info",
    "get_sliding_window_mask",
    "OptimizedFeedForward",
    "OptimizedMemoryMLP",
    "rotary_embedding_optimized",
    "scaled_dot_product_attention",
    # Metal Kernels
    "benchmark_metal_kernel",
    "get_metal_kernel_info",
    "metal_causal_attention",
    "metal_memory_update",
    "metal_rope",
    "metal_silu_gate",
    "MetalFeedForward",
    "MetalRMSNorm",
    "MetalRotaryEmbedding",
]
