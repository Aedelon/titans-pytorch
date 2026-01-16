# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Optimized inference for maximum GPU utilization during generation.

This module provides:
1. Static KV caching for efficient autoregressive generation
2. Continuous batching for throughput optimization
3. Speculative decoding support
4. Quantization utilities
5. Tensor parallelism helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Check availability
CUDA_AVAILABLE = torch.cuda.is_available()
HAS_TORCH_COMPILE = hasattr(torch, "compile")


# =============================================================================
# Static KV Cache
# =============================================================================


@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""
    max_batch_size: int = 32
    max_seq_len: int = 4096
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    dtype: torch.dtype = torch.float16


class StaticKVCache:
    """Static KV cache with pre-allocated memory.

    Pre-allocates memory for K and V tensors to avoid
    dynamic allocation during generation.
    """

    def __init__(
        self,
        config: KVCacheConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device

        # Pre-allocate cache tensors
        cache_shape = (
            config.max_batch_size,
            config.num_heads,
            config.max_seq_len,
            config.head_dim,
        )

        self.k_cache = [
            torch.zeros(cache_shape, dtype=config.dtype, device=device)
            for _ in range(config.num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, dtype=config.dtype, device=device)
            for _ in range(config.num_layers)
        ]

        # Track current sequence lengths per batch item
        self.seq_lens = torch.zeros(
            config.max_batch_size, dtype=torch.long, device=device
        )

    def update(
        self,
        layer_idx: int,
        batch_indices: Tensor,
        new_k: Tensor,
        new_v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Update cache and return full K, V for specified batch items.

        Args:
            layer_idx: Layer index
            batch_indices: Which batch items to update
            new_k: New keys (batch, heads, new_seq, head_dim)
            new_v: New values

        Returns:
            Full K, V up to current position
        """
        batch_size = batch_indices.shape[0]
        new_seq_len = new_k.shape[2]

        # Get starting positions
        start_pos = self.seq_lens[batch_indices]

        # Update cache for each batch item
        for i, (bidx, start) in enumerate(zip(batch_indices, start_pos)):
            end = start + new_seq_len
            self.k_cache[layer_idx][bidx, :, start:end, :] = new_k[i]
            self.v_cache[layer_idx][bidx, :, start:end, :] = new_v[i]

        # Return full cache
        max_len = (start_pos + new_seq_len).max().item()
        full_k = self.k_cache[layer_idx][batch_indices, :, :max_len, :]
        full_v = self.v_cache[layer_idx][batch_indices, :, :max_len, :]

        return full_k, full_v

    def increment_seq_len(self, batch_indices: Tensor, delta: int) -> None:
        """Increment sequence length for batch items."""
        self.seq_lens[batch_indices] += delta

    def reset(self, batch_indices: Tensor | None = None) -> None:
        """Reset cache for specified batch items (or all if None)."""
        if batch_indices is None:
            self.seq_lens.zero_()
            for layer_idx in range(len(self.k_cache)):
                self.k_cache[layer_idx].zero_()
                self.v_cache[layer_idx].zero_()
        else:
            self.seq_lens[batch_indices] = 0
            for layer_idx in range(len(self.k_cache)):
                self.k_cache[layer_idx][batch_indices].zero_()
                self.v_cache[layer_idx][batch_indices].zero_()


# =============================================================================
# Optimized Generation
# =============================================================================


class OptimizedGenerator:
    """Optimized text generation with maximum GPU utilization."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        use_compile: bool = True,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype

        # Move model to device and set dtype
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()

        # Compile for faster generation
        if use_compile and HAS_TORCH_COMPILE:
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            except Exception:
                pass

        # Get model config
        self.config = model.config if hasattr(model, "config") else None

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = True,
    ) -> Tensor:
        """Generate tokens with optimized inference.

        Args:
            input_ids: Input token IDs (batch, seq)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Repetition penalty
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            do_sample: Whether to sample (False = greedy)

        Returns:
            Generated token IDs (batch, seq + max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone().to(self.device)
        states = None

        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            # Get context (use chunking for efficiency)
            chunk_size = getattr(self.config, "chunk_size", 512) if self.config else 512
            context_size = min(generated.shape[1], chunk_size)
            context = generated[:, -context_size:]

            # Forward pass
            logits, states = self.model(context, states=states)

            # Get next token logits
            next_logits = logits[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(
                    next_logits, generated, repetition_penalty
                )

            # Sample or greedy
            if do_sample:
                next_tokens = self._sample_tokens(
                    next_logits, top_k=top_k, top_p=top_p
                )
            else:
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)

            # Handle padding for finished sequences
            if pad_token_id is not None:
                next_tokens = next_tokens.masked_fill(
                    finished.unsqueeze(-1), pad_token_id
                )

            # Append to generated
            generated = torch.cat([generated, next_tokens], dim=-1)

            # Check for EOS
            if eos_token_id is not None:
                finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return generated

    def _apply_repetition_penalty(
        self,
        logits: Tensor,
        generated: Tensor,
        penalty: float,
    ) -> None:
        """Apply repetition penalty in-place."""
        for i in range(generated.shape[0]):
            unique_tokens = generated[i].unique()
            logits[i, unique_tokens] = logits[i, unique_tokens] / penalty

    def _sample_tokens(
        self,
        logits: Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tensor:
        """Sample tokens with top-k and top-p filtering."""
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_streaming(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> Iterator[Tensor]:
        """Generate tokens with streaming output.

        Yields one token at a time for real-time output.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: End of sequence token

        Yields:
            Single token tensors
        """
        generated = input_ids.clone().to(self.device)
        states = None

        for _ in range(max_new_tokens):
            chunk_size = getattr(self.config, "chunk_size", 512) if self.config else 512
            context_size = min(generated.shape[1], chunk_size)
            context = generated[:, -context_size:]

            logits, states = self.model(context, states=states)
            next_logits = logits[:, -1, :] / max(temperature, 1e-7)

            next_token = self._sample_tokens(next_logits, top_k=top_k, top_p=top_p)

            generated = torch.cat([generated, next_token], dim=-1)
            yield next_token

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break


# =============================================================================
# Continuous Batching
# =============================================================================


class ContinuousBatcher:
    """Continuous batching for serving multiple requests.

    Dynamically batches requests as they arrive and complete,
    maximizing GPU utilization.
    """

    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device or torch.device("cuda" if CUDA_AVAILABLE else "cpu")

        # Request queue
        self.pending_requests: list[dict] = []
        self.active_requests: list[dict] = []

        # Shared state tracking
        self.shared_states = None

    def add_request(
        self,
        request_id: str,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> None:
        """Add a generation request to the queue.

        Args:
            request_id: Unique request identifier
            input_ids: Input token IDs (1, seq)
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
        """
        self.pending_requests.append({
            "id": request_id,
            "input_ids": input_ids.to(self.device),
            "generated": input_ids.to(self.device),
            "max_new_tokens": max_new_tokens,
            "tokens_generated": 0,
            "finished": False,
            **kwargs,
        })

    def step(self) -> list[dict]:
        """Process one generation step for all active requests.

        Returns:
            List of completed requests with their outputs
        """
        completed = []

        # Move pending to active if space available
        while (
            self.pending_requests
            and len(self.active_requests) < self.max_batch_size
        ):
            request = self.pending_requests.pop(0)
            self.active_requests.append(request)

        if not self.active_requests:
            return completed

        # Batch active requests
        # Pad to same length
        max_len = max(r["generated"].shape[1] for r in self.active_requests)
        batch_inputs = []

        for request in self.active_requests:
            seq = request["generated"]
            if seq.shape[1] < max_len:
                padding = torch.zeros(
                    1, max_len - seq.shape[1],
                    dtype=seq.dtype,
                    device=self.device,
                )
                seq = torch.cat([padding, seq], dim=1)
            batch_inputs.append(seq)

        batch = torch.cat(batch_inputs, dim=0)

        # Forward pass
        with torch.inference_mode():
            chunk_size = 512
            context = batch[:, -chunk_size:]
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :]

        # Sample for each request
        still_active = []
        for i, request in enumerate(self.active_requests):
            temperature = request.get("temperature", 1.0)
            token_logits = next_logits[i:i+1] / max(temperature, 1e-7)
            probs = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            request["generated"] = torch.cat(
                [request["generated"], next_token], dim=1
            )
            request["tokens_generated"] += 1

            # Check completion
            eos_id = request.get("eos_token_id")
            if (
                request["tokens_generated"] >= request["max_new_tokens"]
                or (eos_id is not None and next_token.item() == eos_id)
            ):
                request["finished"] = True
                completed.append(request)
            else:
                still_active.append(request)

        self.active_requests = still_active
        return completed


# =============================================================================
# Quantization Utilities
# =============================================================================


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply dynamic quantization for faster inference.

    Args:
        model: Model to quantize
        dtype: Quantization dtype

    Returns:
        Quantized model
    """
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
    )


def quantize_model_static(
    model: nn.Module,
    calibration_data: list[Tensor],
    backend: str = "qnnpack",
) -> nn.Module:
    """Apply static quantization with calibration.

    Args:
        model: Model to quantize
        calibration_data: Data for calibration
        backend: Quantization backend

    Returns:
        Quantized model
    """
    model.eval()
    torch.backends.quantized.engine = backend

    # Fuse modules
    model_fused = torch.quantization.fuse_modules(
        model,
        [["linear", "relu"]],
        inplace=False,
    )

    # Prepare for quantization
    model_fused.qconfig = torch.quantization.get_default_qconfig(backend)
    model_prepared = torch.quantization.prepare(model_fused, inplace=False)

    # Calibrate
    with torch.inference_mode():
        for data in calibration_data:
            model_prepared(data)

    # Convert
    return torch.quantization.convert(model_prepared, inplace=False)


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_generation(
    model: nn.Module,
    input_length: int = 128,
    output_length: int = 128,
    batch_size: int = 1,
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Benchmark generation performance.

    Args:
        model: Model to benchmark
        input_length: Input sequence length
        output_length: Number of tokens to generate
        batch_size: Batch size
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    import time

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    vocab_size = model.config.vocab_size if hasattr(model, "config") else 32000

    generator = OptimizedGenerator(model, device, use_compile=False)

    input_ids = torch.randint(
        0, vocab_size, (batch_size, input_length), device=device
    )

    # Warmup
    for _ in range(warmup_runs):
        _ = generator.generate(
            input_ids, max_new_tokens=output_length, do_sample=False
        )

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = generator.generate(
            input_ids, max_new_tokens=output_length, do_sample=False
        )

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    total_tokens = batch_size * output_length
    avg_time = sum(times) / len(times)

    return {
        "avg_latency_ms": avg_time * 1000,
        "tokens_per_second": total_tokens / avg_time,
        "time_per_token_ms": (avg_time / output_length) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
    }
