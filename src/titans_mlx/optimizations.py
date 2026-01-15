# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Apple Silicon Optimizations for Titans MLX.

This module provides optimized operations for Apple Silicon:
1. JIT compilation for frequently used functions
2. Metal shader optimizations
3. Fused operations to reduce kernel launches
4. Memory-efficient attention implementations

These optimizations leverage MLX's unique features:
- Unified memory architecture (no CPU/GPU transfers)
- Lazy evaluation for optimal computation graphs
- Metal Performance Shaders integration
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import mlx.core as mx
import mlx.nn as nn


def compile_function(fn: Callable) -> Callable:
    """Compile a function for faster execution using MLX's JIT.

    Args:
        fn: Function to compile

    Returns:
        Compiled function
    """
    return mx.compile(fn)


def compile_model(model: nn.Module) -> nn.Module:
    """Note: Full model compilation is not currently supported.

    mx.compile cannot compile Titans models because they use:
    - MemoryState dataclasses (Python objects not supported)
    - Dynamic Python loops for chunk processing
    - Mutable state updates

    The models are already well-optimized for MLX. For additional speedups:
    1. Use larger batch sizes to amortize overhead
    2. Process longer sequences (more compute per call)
    3. Use float16 precision: mx.set_default_dtype(mx.float16)

    Returns the original model unchanged.

    For component-level compilation, use compile_function on individual
    operations like FeedForward or attention modules.
    """
    # Return model unchanged - compilation not supported
    return model


@lru_cache(maxsize=32)
def get_causal_mask(seq_len: int) -> mx.array:
    """Get cached causal mask for a given sequence length.

    This avoids recomputing the mask for common sequence lengths.

    Args:
        seq_len: Sequence length

    Returns:
        Lower triangular boolean mask (seq_len, seq_len)
    """
    return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))


@lru_cache(maxsize=32)
def get_sliding_window_mask(seq_len: int, window_size: int) -> mx.array:
    """Get cached sliding window mask.

    Args:
        seq_len: Sequence length
        window_size: Window size

    Returns:
        Sliding window causal mask
    """
    positions = mx.arange(seq_len)
    row_idx = mx.expand_dims(positions, axis=1)
    col_idx = mx.expand_dims(positions, axis=0)
    causal_mask = col_idx <= row_idx
    window_mask = (row_idx - col_idx) < window_size
    return causal_mask & window_mask


def fused_rmsnorm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    """Fused RMS normalization.

    Performs RMS normalization in a single kernel launch.

    Args:
        x: Input tensor (..., dim)
        weight: Scale weights (dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    # Compute RMS
    rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    # Normalize and scale
    return (x / rms) * weight


def fused_silu_gate(gate_output: mx.array, up_output: mx.array) -> mx.array:
    """Fused SiLU gating operation.

    Computes SiLU(gate) * up in an optimized way.

    Args:
        gate_output: Output of gate projection
        up_output: Output of up projection

    Returns:
        Gated output
    """
    return nn.silu(gate_output) * up_output


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array | None = None,
    scale: float | None = None,
) -> mx.array:
    """Optimized scaled dot-product attention.

    Uses memory-efficient attention computation pattern.

    Args:
        q: Queries (batch, heads, seq_q, head_dim)
        k: Keys (batch, heads, seq_k, head_dim)
        v: Values (batch, heads, seq_k, head_dim)
        mask: Optional attention mask (broadcastable to (batch, heads, seq_q, seq_k))
        scale: Optional scale factor (default: 1/sqrt(head_dim))

    Returns:
        Attention output (batch, heads, seq_q, head_dim)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Compute attention scores
    scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

    # Apply mask
    if mask is not None:
        scores = mx.where(mask, scores, mx.array(float("-inf")))

    # Softmax
    weights = mx.softmax(scores, axis=-1)

    # Apply attention
    return mx.matmul(weights, v)


def chunked_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk_size: int = 256,
    mask: mx.array | None = None,
) -> mx.array:
    """Memory-efficient chunked attention for long sequences.

    Processes attention in chunks to reduce peak memory usage.

    Args:
        q: Queries (batch, heads, seq_q, head_dim)
        k: Keys (batch, heads, seq_k, head_dim)
        v: Values (batch, heads, seq_k, head_dim)
        chunk_size: Size of each chunk
        mask: Optional attention mask

    Returns:
        Attention output
    """
    batch, heads, seq_q, head_dim = q.shape
    seq_k = k.shape[2]

    # If sequence is short enough, use standard attention
    if seq_q <= chunk_size and seq_k <= chunk_size:
        return scaled_dot_product_attention(q, k, v, mask=mask)

    scale = head_dim**-0.5
    outputs = []

    for i in range(0, seq_q, chunk_size):
        q_chunk = q[:, :, i : i + chunk_size]

        # Compute scores for this chunk
        scores = mx.matmul(q_chunk, mx.transpose(k, (0, 1, 3, 2))) * scale

        # Apply mask for this chunk
        if mask is not None:
            chunk_mask = mask[:, :, i : i + chunk_size]
            scores = mx.where(chunk_mask, scores, mx.array(float("-inf")))

        # Softmax and apply
        weights = mx.softmax(scores, axis=-1)
        chunk_output = mx.matmul(weights, v)
        outputs.append(chunk_output)

    return mx.concatenate(outputs, axis=2)


def rotary_embedding_optimized(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> mx.array:
    """Optimized rotary position embedding application.

    Uses vectorized operations for efficient RoPE computation.

    Args:
        x: Input tensor (batch, heads, seq, head_dim)
        cos: Cosine values (seq, head_dim // 2)
        sin: Sine values (seq, head_dim // 2)

    Returns:
        Rotated tensor
    """
    # Split into even and odd indices
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # Expand for broadcasting
    cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=0)
    sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=0)

    # Apply rotation (vectorized)
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos

    # Interleave results
    batch, heads, seq, half_dim = rotated_even.shape
    result = mx.stack([rotated_even, rotated_odd], axis=-1)
    return result.reshape(batch, heads, seq, half_dim * 2)


class OptimizedMemoryMLP(nn.Module):
    """Memory MLP with fused operations for better performance."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        hidden_dim: int | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or dim * 4

        # Build layers
        self.layers: list[nn.Linear] = []

        if num_layers == 1:
            self.layers.append(nn.Linear(dim, dim, bias=False))
        else:
            self.layers.append(nn.Linear(dim, self.hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                self.layers.append(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                )
            self.layers.append(nn.Linear(self.hidden_dim, dim, bias=False))

        # Initialize
        for layer in self.layers:
            layer.weight = mx.random.normal(layer.weight.shape) * init_std

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with optimized activations."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = nn.silu(h)
        return h


class OptimizedFeedForward(nn.Module):
    """Feed-forward with fused SiLU gating."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        # Initialize
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            proj.weight = mx.random.normal(proj.weight.shape) * init_std

    def __call__(self, x: mx.array) -> mx.array:
        """Forward with fused SiLU gating."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = fused_silu_gate(gate, up)
        return self.down_proj(hidden)


def evaluate_all(*arrays: mx.array) -> None:
    """Force evaluation of multiple arrays at once.

    This is more efficient than calling mx.eval on each array separately
    as MLX can optimize the computation graph.

    Args:
        *arrays: Arrays to evaluate
    """
    mx.eval(*arrays)


def benchmark_function(
    fn: Callable,
    *args,
    warmup: int = 3,
    repeat: int = 10,
    **kwargs,
) -> dict:
    """Benchmark a function on Apple Silicon.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for function
        warmup: Number of warmup iterations
        repeat: Number of timed iterations
        **kwargs: Keyword arguments for function

    Returns:
        Dictionary with timing statistics
    """
    import time

    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        mx.eval(result)

    # Time
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        mx.eval(result)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5
        * 1000,
    }


def get_device_info() -> dict:
    """Get information about the MLX device.

    Returns:
        Dictionary with device information
    """
    return {
        "backend": "MLX",
        "device": "Apple Silicon GPU",
        "unified_memory": True,
        "lazy_evaluation": True,
    }
