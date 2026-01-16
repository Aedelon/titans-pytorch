# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
CUDA optimizations for maximizing GPU utilization in Titans.

This module provides:
1. CUDA stream management for overlapping compute and memory operations
2. Fused operations to reduce kernel launch overhead
3. Memory-efficient gradient computation
4. Optimized batch processing utilities
5. torch.compile integration for JIT compilation
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# Check for torch.compile availability (PyTorch 2.0+)
HAS_TORCH_COMPILE = hasattr(torch, "compile")


# =============================================================================
# CUDA Stream Management
# =============================================================================


class CUDAStreamManager:
    """Manages CUDA streams for overlapping operations.

    Uses multiple streams to overlap:
    - Memory transfers (H2D, D2H)
    - Compute operations
    - Memory gradient computation
    """

    def __init__(self, num_streams: int = 3) -> None:
        self.num_streams = num_streams
        self.streams: list[torch.cuda.Stream] = []
        self.default_stream: torch.cuda.Stream | None = None

        if CUDA_AVAILABLE:
            self.default_stream = torch.cuda.current_stream()
            self.streams = [torch.cuda.Stream() for _ in range(num_streams)]

    @contextmanager
    def stream(self, stream_id: int = 0):
        """Context manager for running operations on a specific stream."""
        if not CUDA_AVAILABLE or stream_id >= len(self.streams):
            yield
            return

        stream = self.streams[stream_id]
        with torch.cuda.stream(stream):
            yield

    def synchronize(self) -> None:
        """Synchronize all streams."""
        if CUDA_AVAILABLE:
            for stream in self.streams:
                stream.synchronize()

    def record_event(self, stream_id: int = 0) -> torch.cuda.Event | None:
        """Record an event on a stream."""
        if not CUDA_AVAILABLE or stream_id >= len(self.streams):
            return None

        event = torch.cuda.Event()
        event.record(self.streams[stream_id])
        return event

    def wait_event(self, event: torch.cuda.Event | None, stream_id: int = 0) -> None:
        """Make a stream wait for an event."""
        if event is None or not CUDA_AVAILABLE or stream_id >= len(self.streams):
            return

        self.streams[stream_id].wait_event(event)


# Global stream manager
_stream_manager: CUDAStreamManager | None = None


def get_stream_manager() -> CUDAStreamManager:
    """Get or create the global stream manager."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = CUDAStreamManager()
    return _stream_manager


# =============================================================================
# Fused Operations
# =============================================================================


def fused_add_rms_norm(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Fused residual addition and RMS normalization.

    Combines:
        hidden = x + residual
        normalized = rms_norm(hidden, weight)

    Returns both hidden states for residual connection.
    """
    hidden = x + residual
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden_normalized = hidden * torch.rsqrt(variance + eps)
    return hidden, hidden_normalized * weight


def fused_silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """Fused SiLU activation and multiplication.

    Computes: silu(gate) * up
    More efficient than separate operations.
    """
    return F.silu(gate) * up


def fused_linear_silu_linear(
    x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Fused gated FFN forward pass.

    Computes: down(silu(gate(x)) * up(x))
    """
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = fused_silu_mul(gate, up)
    return F.linear(hidden, down_weight)


# =============================================================================
# Optimized Memory Module Operations
# =============================================================================


class OptimizedMemoryGradients(torch.autograd.Function):
    """Custom autograd function for efficient memory gradient computation.

    Instead of creating a new computation graph for each gradient computation,
    this uses analytical gradients where possible.
    """

    @staticmethod
    def forward(
        ctx: Any,
        keys: Tensor,
        values: Tensor,
        weight: Tensor,
        activation: str = "silu",
    ) -> Tensor:
        """Forward pass computing M(k) for gradient calculation."""
        ctx.save_for_backward(keys, values, weight)
        ctx.activation = activation

        # Compute prediction: M(k)
        if activation == "silu":
            h = F.silu(keys)
        elif activation == "gelu":
            h = F.gelu(keys)
        else:
            h = F.relu(keys)

        prediction = F.linear(h, weight)
        return prediction

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor, None]:
        """Backward pass computing gradients analytically."""
        keys, values, weight = ctx.saved_tensors
        activation = ctx.activation

        # Compute activation and its derivative
        if activation == "silu":
            h = F.silu(keys)
            # SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            sigmoid_keys = torch.sigmoid(keys)
            h_grad = sigmoid_keys * (1 + keys * (1 - sigmoid_keys))
        elif activation == "gelu":
            h = F.gelu(keys)
            # GELU derivative approximation
            h_grad = (
                0.5
                * (1 + torch.tanh(0.7978845608 * (keys + 0.044715 * keys.pow(3))))
                + 0.5
                * keys
                * (1 - torch.tanh(0.7978845608 * (keys + 0.044715 * keys.pow(3))).pow(2))
                * 0.7978845608
                * (1 + 3 * 0.044715 * keys.pow(2))
            )
        else:
            h = F.relu(keys)
            h_grad = (keys > 0).float()

        # Gradient w.r.t. weight: d_loss/d_W = h^T @ grad_output
        # For MSE loss with prediction = W @ h, grad_output = 2 * (prediction - values)
        # So d_loss/d_W = 2 * h^T @ (prediction - values)
        batch_size, seq_len, dim = keys.shape
        h_flat = h.view(-1, dim)
        grad_flat = grad_output.view(-1, grad_output.shape[-1])

        grad_weight = grad_flat.t() @ h_flat

        return None, None, grad_weight, None


def compute_memory_gradients_efficient(
    keys: Tensor,
    values: Tensor,
    memory_weights: list[Tensor],
    activation: str = "silu",
) -> list[Tensor]:
    """Compute memory gradients efficiently using analytical computation.

    For a single-layer memory (linear), computes gradients analytically.
    For multi-layer memory, falls back to autograd.

    Args:
        keys: Key vectors (batch, seq, dim)
        values: Value vectors (batch, seq, dim)
        memory_weights: List of memory weight matrices
        activation: Activation function name

    Returns:
        List of gradient tensors for each weight matrix
    """
    if len(memory_weights) == 1:
        # Single layer - use analytical gradient
        weight = memory_weights[0]

        # Compute activation
        if activation == "silu":
            h = F.silu(keys)
        elif activation == "gelu":
            h = F.gelu(keys)
        else:
            h = F.relu(keys)

        # Prediction: M(k) = W @ h
        prediction = F.linear(h, weight)

        # Loss gradient: d_loss/d_pred = 2 * (pred - values) for MSE
        diff = prediction - values

        # Weight gradient: d_loss/d_W = diff^T @ h
        batch_size, seq_len, dim = keys.shape
        h_flat = h.view(-1, dim)
        diff_flat = diff.view(-1, dim)

        # Compute gradient: (batch*seq, out_dim)^T @ (batch*seq, in_dim)
        grad_weight = diff_flat.t() @ h_flat
        grad_weight = grad_weight * (2.0 / (batch_size * seq_len))

        return [grad_weight]
    else:
        # Multi-layer - use autograd (more complex analytical gradient)
        return _compute_gradients_autograd(keys, values, memory_weights, activation)


def _compute_gradients_autograd(
    keys: Tensor,
    values: Tensor,
    memory_weights: list[Tensor],
    activation: str,
) -> list[Tensor]:
    """Compute gradients using autograd for multi-layer memory.

    Optimized to avoid cloning weights - uses requires_grad directly.
    """
    # Ensure weights require grad temporarily
    original_requires_grad = [w.requires_grad for w in memory_weights]
    for w in memory_weights:
        w.requires_grad_(True)

    # Forward pass with no_grad disabled
    with torch.enable_grad():
        h = keys.detach()  # Detach input to avoid backprop through it
        for i, w in enumerate(memory_weights):
            h = F.linear(h, w)
            if i < len(memory_weights) - 1:
                if activation == "silu":
                    h = F.silu(h)
                elif activation == "gelu":
                    h = F.gelu(h)
                else:
                    h = F.relu(h)

        # Compute loss
        loss = F.mse_loss(h, values.detach())

        # Compute gradients without creating graph
        grads = torch.autograd.grad(loss, memory_weights, create_graph=False)

    # Restore original requires_grad state
    for w, req in zip(memory_weights, original_requires_grad):
        w.requires_grad_(req)

    return list(grads)


# =============================================================================
# Batched Memory Updates
# =============================================================================


def batched_memory_update(
    weights: list[Tensor],
    momentum: list[Tensor],
    gradients: list[Tensor],
    alpha: Tensor | float,
    eta: Tensor | float,
    theta: Tensor | float,
) -> tuple[list[Tensor], list[Tensor]]:
    """Perform batched memory updates efficiently.

    Vectorizes the update equations:
        S_t = eta * S_{t-1} - theta * grad
        M_t = (1 - alpha) * M_{t-1} + S_t

    Args:
        weights: List of weight tensors
        momentum: List of momentum tensors
        gradients: List of gradient tensors
        alpha: Decay factor (scalar or tensor)
        eta: Momentum coefficient
        theta: Learning rate

    Returns:
        Tuple of (new_weights, new_momentum)
    """
    new_weights = []
    new_momentum = []

    # Keep as tensors - no .item() needed, operations broadcast correctly
    # This avoids CPU-GPU sync entirely
    one_minus_alpha = 1.0 - alpha

    for w, m, g in zip(weights, momentum, gradients, strict=True):
        # Fused update: S_t = eta * S_{t-1} - theta * grad
        # Use mul_ and add_ for potential in-place optimization by compiler
        new_m = eta * m - theta * g

        # Fused update: M_t = (1 - alpha) * M_{t-1} + S_t
        new_w = one_minus_alpha * w + new_m

        new_weights.append(new_w)
        new_momentum.append(new_m)

    return new_weights, new_momentum


# =============================================================================
# torch.compile Wrappers
# =============================================================================


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> nn.Module:
    """Apply torch.compile to a model for JIT optimization.

    Args:
        model: Model to compile
        mode: Compilation mode:
            - "default": Good balance of speed and compile time
            - "reduce-overhead": Best for small batches, reduces kernel launch overhead
            - "max-autotune": Slowest compile, fastest execution
        fullgraph: Whether to require full graph compilation
        dynamic: Whether to allow dynamic shapes

    Returns:
        Compiled model (or original if torch.compile unavailable)
    """
    if not HAS_TORCH_COMPILE:
        return model

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return compiled
    except Exception as e:
        print(f"Warning: torch.compile failed: {e}")
        return model


def compile_function(
    fn: Callable,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    dynamic: bool = True,
) -> Callable:
    """Apply torch.compile to a function.

    Args:
        fn: Function to compile
        mode: Compilation mode
        fullgraph: Whether to require full graph compilation
        dynamic: Whether to allow dynamic shapes

    Returns:
        Compiled function (or original if unavailable)
    """
    if not HAS_TORCH_COMPILE:
        return fn

    try:
        return torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    except Exception:
        return fn


# =============================================================================
# Prefetching and Data Pipeline Optimization
# =============================================================================


class CUDAPrefetcher:
    """Prefetches data to GPU using CUDA streams.

    Overlaps data transfer with computation for better GPU utilization.
    """

    def __init__(self, loader: Any, device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if CUDA_AVAILABLE else None
        self.batch: dict[str, Tensor] | None = None

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> dict[str, Tensor]:
        if self.batch is None:
            raise StopIteration

        # Wait for current batch to be ready
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        batch = self.batch
        self._preload()
        return batch

    def _preload(self) -> None:
        """Preload next batch to GPU asynchronously."""
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                    if isinstance(v, Tensor)
                }
        else:
            self.batch = {
                k: v.to(self.device) for k, v in batch.items() if isinstance(v, Tensor)
            }


# =============================================================================
# Memory Pool Management
# =============================================================================


def configure_memory_pool(
    max_split_size_mb: int = 128,
    garbage_collection_threshold: float = 0.8,
) -> None:
    """Configure CUDA memory allocator for optimal performance.

    Args:
        max_split_size_mb: Maximum size for memory block splitting
        garbage_collection_threshold: Threshold for triggering GC
    """
    if not CUDA_AVAILABLE:
        return

    # Set memory allocator configuration
    torch.cuda.set_per_process_memory_fraction(0.95)

    # Configure expandable segments for better memory utilization
    try:
        torch.cuda.memory._set_allocator_settings(
            f"max_split_size_mb:{max_split_size_mb},"
            f"garbage_collection_threshold:{garbage_collection_threshold}"
        )
    except Exception:
        pass  # Older PyTorch versions may not support this


def empty_cache_if_needed(threshold: float = 0.9) -> None:
    """Empty CUDA cache if memory usage exceeds threshold.

    Args:
        threshold: Memory usage threshold (0-1)
    """
    if not CUDA_AVAILABLE:
        return

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    if reserved > 0 and allocated / reserved > threshold:
        torch.cuda.empty_cache()


# =============================================================================
# Inference Optimization Utilities
# =============================================================================


@torch.inference_mode()
def optimized_generate_step(
    model: nn.Module,
    input_ids: Tensor,
    past_key_values: list[tuple[Tensor, Tensor]] | None = None,
    use_cache: bool = True,
) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | None]:
    """Optimized single generation step.

    Uses inference_mode and optional KV caching for faster generation.

    Args:
        model: The model to use for generation
        input_ids: Input token IDs
        past_key_values: Cached key-value pairs from previous steps
        use_cache: Whether to use/return KV cache

    Returns:
        Tuple of (logits, new_past_key_values)
    """
    # Forward pass
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=use_cache)

    if isinstance(outputs, tuple):
        logits = outputs[0]
        new_past = outputs[1] if len(outputs) > 1 else None
    else:
        logits = outputs
        new_past = None

    return logits, new_past


def speculative_decode(
    model: nn.Module,
    draft_model: nn.Module,
    input_ids: Tensor,
    num_speculative_tokens: int = 4,
    temperature: float = 1.0,
) -> Tensor:
    """Speculative decoding for faster inference.

    Uses a smaller draft model to predict multiple tokens,
    then verifies with the main model.

    Args:
        model: Main (larger) model
        draft_model: Draft (smaller) model
        input_ids: Input token IDs
        num_speculative_tokens: Number of tokens to speculate
        temperature: Sampling temperature

    Returns:
        Generated token IDs
    """
    device = input_ids.device
    generated = input_ids.clone()

    with torch.inference_mode():
        # Generate speculative tokens with draft model
        draft_tokens = []
        draft_input = input_ids

        for _ in range(num_speculative_tokens):
            draft_logits, _ = draft_model(draft_input)
            next_token_logits = draft_logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(next_token)
            draft_input = torch.cat([draft_input, next_token], dim=1)

        draft_sequence = torch.cat(draft_tokens, dim=1)

        # Verify with main model (single forward pass for all tokens)
        full_input = torch.cat([input_ids, draft_sequence], dim=1)
        main_logits, _ = model(full_input)

        # Compare and accept/reject tokens
        accepted = []
        for i in range(num_speculative_tokens):
            pos = input_ids.shape[1] + i - 1
            main_probs = F.softmax(main_logits[:, pos, :] / temperature, dim=-1)
            draft_token = draft_tokens[i]

            # Accept if probability is high enough
            token_prob = main_probs.gather(1, draft_token)
            if token_prob.item() > 0.1:  # Acceptance threshold
                accepted.append(draft_token)
            else:
                # Sample from main model instead
                new_token = torch.multinomial(main_probs, num_samples=1)
                accepted.append(new_token)
                break  # Stop accepting after first rejection

        if accepted:
            generated = torch.cat([generated] + accepted, dim=1)

    return generated


# =============================================================================
# Utility Functions
# =============================================================================


def get_optimal_batch_size(
    model: nn.Module,
    seq_len: int,
    target_memory_usage: float = 0.8,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate optimal batch size for given model and sequence length.

    Args:
        model: Model to estimate for
        seq_len: Sequence length
        target_memory_usage: Target GPU memory usage fraction
        dtype: Data type for computation

    Returns:
        Estimated optimal batch size
    """
    if not CUDA_AVAILABLE:
        return 1

    # Get available memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory * target_memory_usage

    # Estimate model memory (rough approximation)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_memory = param_bytes * 2  # Parameters + gradients

    # Estimate activation memory per sample
    # Rough estimate: 4 * model_dim * seq_len * num_layers * bytes_per_element
    num_params = sum(p.numel() for p in model.parameters())
    model_dim = int((num_params / 1e6) ** 0.5 * 100)  # Rough estimate
    activation_memory_per_sample = (
        4 * model_dim * seq_len * (2 if dtype == torch.float16 else 4)
    )

    # Calculate batch size
    remaining_memory = available_memory - param_memory
    batch_size = max(1, int(remaining_memory / activation_memory_per_sample))

    return batch_size


def benchmark_model(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Benchmark model throughput and latency.

    Args:
        model: Model to benchmark
        batch_size: Batch size for benchmarking
        seq_len: Sequence length
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(input_ids)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    import time

    times = []

    with torch.inference_mode():
        for _ in range(num_iterations):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids)

            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_second = (batch_size * seq_len) / avg_time

    return {
        "avg_latency_ms": avg_time * 1000,
        "throughput_tokens_per_sec": tokens_per_second,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
    }
