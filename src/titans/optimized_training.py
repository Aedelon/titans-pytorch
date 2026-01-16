# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Optimized training utilities for maximum GPU utilization.

This module provides:
1. Optimized training loop with torch.compile integration
2. Efficient gradient accumulation with proper synchronization
3. CUDA graph capture for reduced kernel launch overhead
4. Mixed precision training with automatic loss scaling
5. Efficient data prefetching
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

# Check for optional dependencies
CUDA_AVAILABLE = torch.cuda.is_available()
HAS_TORCH_COMPILE = hasattr(torch, "compile")

try:
    from titans.cuda_optimizations import (
        CUDAPrefetcher,
        compile_model,
        configure_memory_pool,
        empty_cache_if_needed,
        get_stream_manager,
    )
    HAS_CUDA_OPTS = True
except ImportError:
    HAS_CUDA_OPTS = False


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class OptimizedTrainingConfig:
    """Configuration for optimized training."""

    # Compilation
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune

    # CUDA optimizations
    use_cuda_graphs: bool = False  # Experimental
    use_prefetching: bool = True
    num_prefetch_batches: int = 2

    # Memory optimization
    gradient_checkpointing: bool = False
    empty_cache_frequency: int = 100  # Steps between cache clearing

    # Mixed precision
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16

    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    sync_grads_every_step: bool = False  # For DDP

    # Performance monitoring
    profile_enabled: bool = False
    profile_steps: int = 10


# =============================================================================
# Optimized Forward Pass
# =============================================================================


def create_optimized_forward(
    model: nn.Module,
    config: OptimizedTrainingConfig,
) -> Callable:
    """Create an optimized forward function.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        Optimized forward function
    """
    # Apply torch.compile if available
    if config.use_torch_compile and HAS_TORCH_COMPILE:
        try:
            model = torch.compile(
                model,
                mode=config.compile_mode,
                fullgraph=False,
                dynamic=True,
            )
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

    def forward_fn(input_ids: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Optimized forward pass with optional loss computation."""
        logits, states = model(input_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return logits, loss

    return forward_fn


# =============================================================================
# CUDA Graph Capture
# =============================================================================


class CUDAGraphRunner:
    """Runs captured CUDA graphs for reduced kernel launch overhead.

    CUDA graphs capture a sequence of operations and replay them
    with minimal CPU overhead. Best for fixed-size inputs.
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: Tensor,
        sample_labels: Tensor,
        warmup_steps: int = 3,
    ) -> None:
        self.model = model
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_input: Tensor | None = None
        self.static_labels: Tensor | None = None
        self.static_output: Tensor | None = None
        self.static_loss: Tensor | None = None

        if CUDA_AVAILABLE:
            self._capture_graph(sample_input, sample_labels, warmup_steps)

    def _capture_graph(
        self,
        sample_input: Tensor,
        sample_labels: Tensor,
        warmup_steps: int,
    ) -> None:
        """Capture CUDA graph."""
        device = sample_input.device

        # Allocate static tensors
        self.static_input = sample_input.clone()
        self.static_labels = sample_labels.clone()

        # Warmup
        for _ in range(warmup_steps):
            logits, _ = self.model(self.static_input)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                self.static_labels.view(-1),
            )
            loss.backward()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        self.model.zero_grad()

        with torch.cuda.graph(self.graph):
            logits, _ = self.model(self.static_input)
            self.static_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                self.static_labels.view(-1),
            )
            self.static_loss.backward()

        self.static_output = logits

    def run(self, input_ids: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Run the captured graph.

        Args:
            input_ids: Input tensor (must match captured shape)
            labels: Labels tensor (must match captured shape)

        Returns:
            Tuple of (logits, loss)
        """
        if self.graph is None:
            # Fallback to regular forward
            logits, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss.backward()
            return logits, loss

        # Copy inputs to static buffers
        self.static_input.copy_(input_ids)
        self.static_labels.copy_(labels)

        # Replay graph
        self.graph.replay()

        return self.static_output.clone(), self.static_loss.clone()


# =============================================================================
# Optimized Training Step
# =============================================================================


class OptimizedTrainer:
    """Optimized trainer with maximum GPU utilization."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: OptimizedTrainingConfig,
        device: torch.device,
        scheduler: Any = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Mixed precision
        self.scaler = None
        self.autocast_ctx = nullcontext()
        if config.use_amp and device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
            self.autocast_ctx = torch.autocast(
                device_type="cuda",
                dtype=config.amp_dtype,
            )

        # Compile model
        if config.use_torch_compile and HAS_TORCH_COMPILE:
            try:
                self.model = torch.compile(
                    self.model,
                    mode=config.compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        # CUDA graph runner (optional)
        self.cuda_graph_runner: CUDAGraphRunner | None = None

        # Configure memory pool
        if HAS_CUDA_OPTS and CUDA_AVAILABLE:
            configure_memory_pool()

        # State
        self.global_step = 0
        self.accumulation_step = 0

    def setup_cuda_graphs(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
    ) -> None:
        """Setup CUDA graph capture for fixed-size inputs.

        Args:
            batch_size: Fixed batch size
            seq_len: Fixed sequence length
            vocab_size: Vocabulary size
        """
        if not self.config.use_cuda_graphs or not CUDA_AVAILABLE:
            return

        sample_input = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=self.device
        )
        sample_labels = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=self.device
        )

        self.cuda_graph_runner = CUDAGraphRunner(
            self.model, sample_input, sample_labels
        )

    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
    ) -> dict[str, float]:
        """Perform optimized training step.

        Args:
            input_ids: Input token IDs
            labels: Target labels

        Returns:
            Dictionary of metrics
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Use CUDA graphs if available and shapes match
        if self.cuda_graph_runner is not None:
            logits, loss = self.cuda_graph_runner.run(input_ids, labels)
        else:
            # Standard forward pass with autocast
            with self.autocast_ctx:
                logits, states = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        self.accumulation_step += 1

        # Optimizer step after accumulation
        metrics = {"loss": loss.item(), "ppl": math.exp(min(loss.item(), 10))}

        if self.accumulation_step >= self.config.gradient_accumulation_steps:
            self._optimizer_step()
            self.accumulation_step = 0
            metrics["lr"] = self._get_lr()

        # Periodic cache clearing
        if (
            self.config.empty_cache_frequency > 0
            and self.global_step % self.config.empty_cache_frequency == 0
            and HAS_CUDA_OPTS
        ):
            empty_cache_if_needed()

        return metrics

    def _optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping and scaling."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _get_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def train_epoch(
        self,
        dataloader: DataLoader,
        max_steps: int = -1,
    ) -> Iterator[dict[str, float]]:
        """Train for one epoch with prefetching.

        Args:
            dataloader: Training dataloader
            max_steps: Maximum steps (-1 for full epoch)

        Yields:
            Metrics dictionary for each step
        """
        self.model.train()

        # Setup prefetching
        if self.config.use_prefetching and HAS_CUDA_OPTS and CUDA_AVAILABLE:
            data_iter = CUDAPrefetcher(dataloader, self.device)
        else:
            data_iter = dataloader

        step = 0
        for batch in data_iter:
            if max_steps > 0 and step >= max_steps:
                break

            input_ids = batch["input_ids"]
            labels = batch["labels"]

            metrics = self.train_step(input_ids, labels)
            yield metrics

            step += 1


# =============================================================================
# Gradient Checkpointing Utilities
# =============================================================================


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing for memory efficiency.

    This trades compute for memory by recomputing activations
    during backward pass instead of storing them.

    Args:
        model: Model to enable checkpointing on
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return

    # Manual checkpointing for blocks
    for module in model.modules():
        if hasattr(module, "blocks"):
            for i, block in enumerate(module.blocks):
                # Wrap forward method
                original_forward = block.forward

                def make_checkpointed_forward(orig_fwd):
                    def checkpointed_forward(*args, **kwargs):
                        def custom_forward(*inputs):
                            return orig_fwd(*inputs, **kwargs)
                        return torch.utils.checkpoint.checkpoint(
                            custom_forward, *args, use_reentrant=False
                        )
                    return checkpointed_forward

                block.forward = make_checkpointed_forward(original_forward)


# =============================================================================
# Memory-Efficient Attention
# =============================================================================


def memory_efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> Tensor:
    """Memory-efficient attention using PyTorch SDPA.

    Uses the most efficient backend available:
    - Flash Attention (if available)
    - Memory-efficient attention
    - Math fallback

    Args:
        query: Query tensor (batch, heads, seq, head_dim)
        key: Key tensor
        value: Value tensor
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal mask
        scale: Optional scale factor

    Returns:
        Attention output
    """
    # Use PyTorch 2.0 SDPA if available
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    # Fallback to manual implementation
    if scale is None:
        scale = query.shape[-1] ** -0.5

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    if is_causal:
        seq_len = query.shape[-2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    return torch.matmul(attn_weights, value)


# =============================================================================
# Profiling Utilities
# =============================================================================


@contextmanager
def profile_cuda(name: str = "profile", enabled: bool = True):
    """Context manager for CUDA profiling.

    Args:
        name: Name for the profile
        enabled: Whether profiling is enabled

    Yields:
        Profile context
    """
    if not enabled or not CUDA_AVAILABLE:
        yield
        return

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"[{name}] {elapsed_ms:.2f} ms")


def get_gpu_memory_stats() -> dict[str, float]:
    """Get current GPU memory statistics.

    Returns:
        Dictionary with memory stats in GB
    """
    if not CUDA_AVAILABLE:
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def print_gpu_utilization() -> None:
    """Print current GPU utilization."""
    if not CUDA_AVAILABLE:
        print("CUDA not available")
        return

    stats = get_gpu_memory_stats()
    print(f"GPU Memory: {stats['allocated_gb']:.2f} GB allocated, "
          f"{stats['reserved_gb']:.2f} GB reserved, "
          f"{stats['max_allocated_gb']:.2f} GB max")

    # Try to get GPU utilization via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            utilization = result.stdout.strip()
            print(f"GPU Utilization: {utilization}%")
    except Exception:
        pass
