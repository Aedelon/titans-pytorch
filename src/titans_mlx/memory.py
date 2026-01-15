# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Neural Long-term Memory Module for Titans (MLX Implementation).

This module implements the core innovation of Titans: a neural memory that
learns to memorize at test time using gradient descent with momentum and
weight decay.

Key equations from the paper:
    Memory update: M_t = (1 - alpha_t) * M_{t-1} + S_t
    Surprise: S_t = eta_t * S_{t-1} - theta_t * grad(loss(M_{t-1}; x_t))
    Loss: loss(M; x) = ||M(k) - v||^2

MLX-specific optimizations:
- Lazy evaluation for efficient computation graphs
- Unified memory (no CPU/GPU transfers)
- Vectorized operations for Apple Silicon
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.config import TitansConfig


def get_activation(name: str) -> Callable[[mx.array], mx.array]:
    """Get activation function by name."""
    activations = {
        "silu": nn.silu,
        "gelu": nn.gelu,
        "relu": nn.relu,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


@dataclass
class MemoryState:
    """State of the neural long-term memory.

    Attributes:
        weights: List of weight matrices for each memory layer
        momentum: Accumulated surprise momentum (S_t in paper)
    """

    weights: list[mx.array]
    momentum: list[mx.array]

    def detach(self) -> MemoryState:
        """Detach state (stop gradients)."""
        return MemoryState(
            weights=[mx.stop_gradient(w) for w in self.weights],
            momentum=[mx.stop_gradient(m) for m in self.momentum],
        )

    def clone(self) -> MemoryState:
        """Clone the memory state."""
        return MemoryState(
            weights=[mx.array(w) for w in self.weights],
            momentum=[mx.array(m) for m in self.momentum],
        )


class MemoryMLP(nn.Module):
    """MLP architecture for the neural memory.

    This is the actual memory module that stores information in its weights.
    For L_M = 1 (linear memory), this is equivalent to a matrix-valued memory.
    For L_M >= 2 (deep memory), this provides more expressive power.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_memory_layers
        self.dim = config.dim
        self.hidden_dim = config.memory_hidden_dim
        self.activation = get_activation(config.activation)

        # Build MLP layers
        self.layers: list[nn.Linear] = []

        if self.num_layers == 1:
            # Linear memory: single linear layer
            self.layers.append(nn.Linear(self.dim, self.dim, bias=False))
        else:
            # Deep memory: MLP with hidden layers
            self.layers.append(nn.Linear(self.dim, self.hidden_dim, bias=False))

            for _ in range(self.num_layers - 2):
                self.layers.append(
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
                )

            self.layers.append(nn.Linear(self.hidden_dim, self.dim, bias=False))

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        for layer in self.layers:
            # MLX uses different initialization
            layer.weight = mx.random.normal(layer.weight.shape) * self.config.init_std

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through memory MLP.

        Args:
            x: Input tensor of shape (batch, seq, dim)

        Returns:
            Output tensor of shape (batch, seq, dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Apply activation for all but last layer
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[mx.array]:
        """Get current weight matrices."""
        return [mx.array(layer.weight) for layer in self.layers]

    def set_weights(self, weights: list[mx.array]) -> None:
        """Set weight matrices."""
        for layer, w in zip(self.layers, weights):
            layer.weight = w

    def compute_loss(self, keys: mx.array, values: mx.array) -> mx.array:
        """Compute associative memory loss.

        Loss: ||M(k) - v||^2

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)

        Returns:
            Scalar loss value
        """
        predictions = self(keys)
        diff = predictions - values
        return mx.mean(diff * diff)


class NeuralLongTermMemory(nn.Module):
    """Neural Long-term Memory Module (MLX Implementation).

    This is the main memory component of Titans. It learns to memorize
    at test time by treating training as an online learning problem.

    The memory is updated using gradient descent with:
    - Momentum (for past surprise)
    - Weight decay (for forgetting)

    MLX-specific optimizations:
    - Uses mx.grad for efficient gradient computation
    - Lazy evaluation for optimal memory usage
    - Vectorized operations for Apple Silicon
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim

        # Projections for keys, values, and queries
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)

        # Optional 1D convolution
        self.use_conv = config.use_conv
        if self.use_conv:
            self.conv_k = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_v = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )
            self.conv_q = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=config.conv_kernel_size,
                padding=config.conv_kernel_size - 1,
                groups=config.dim,
            )

        # The actual memory module
        self.memory = MemoryMLP(config)

        # Data-dependent gates
        self.gate_decay = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            lambda x: mx.sigmoid(x),
        )
        self.gate_lr = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            lambda x: mx.sigmoid(x),
        )
        self.gate_momentum = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            lambda x: mx.sigmoid(x),
        )

        # Output projection
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_k, self.proj_v, self.proj_q, self.proj_out]:
            module.weight = mx.random.normal(module.weight.shape) * self.config.init_std

    def _apply_conv(
        self, k: mx.array, v: mx.array, q: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Apply 1D convolution to K, V, Q."""
        if not self.use_conv:
            return k, v, q

        # Reshape for conv: (batch, seq, dim) -> (batch, dim, seq)
        k = mx.transpose(k, (0, 2, 1))
        v = mx.transpose(v, (0, 2, 1))
        q = mx.transpose(q, (0, 2, 1))

        # Apply causal convolution
        seq_len = k.shape[-1]
        k = self.conv_k(k)[..., :seq_len]
        v = self.conv_v(v)[..., :seq_len]
        q = self.conv_q(q)[..., :seq_len]

        # Reshape back: (batch, dim, seq) -> (batch, seq, dim)
        k = mx.transpose(k, (0, 2, 1))
        v = mx.transpose(v, (0, 2, 1))
        q = mx.transpose(q, (0, 2, 1))

        return k, v, q

    def _compute_gradients(
        self,
        keys: mx.array,
        values: mx.array,
        weights: list[mx.array],
    ) -> list[mx.array]:
        """Compute gradients for memory update analytically.

        Optimized implementation that minimizes Python overhead.

        For the loss ||M(k) - v||^2, we compute gradients analytically
        to avoid nested mx.grad calls which cause VJP issues.

        Args:
            keys: Key vectors (batch, seq, dim)
            values: Value vectors (batch, seq, dim)
            weights: Current memory weights

        Returns:
            List of gradient tensors for each memory layer
        """
        num_layers = len(weights)

        # Fast path for linear memory (1 layer) - most common case
        if num_layers == 1:
            return self._compute_gradients_linear(keys, values, weights[0])

        # Multi-layer case: use optimized computation
        return self._compute_gradients_deep(keys, values, weights)

    def _compute_gradients_linear(
        self,
        keys: mx.array,
        values: mx.array,
        weight: mx.array,
    ) -> list[mx.array]:
        """Optimized gradient computation for linear (1-layer) memory.

        For M(k) = W @ k, the gradient is:
            dL/dW = 2/n * sum((W @ k - v) @ k^T)

        Uses matmul instead of expand_dims for efficiency.
        """
        # Forward pass: predictions = keys @ W^T
        predictions = keys @ weight.T

        # Error and gradient scale
        error = mx.clip(predictions - values, -10.0, 10.0)
        scale = 2.0 / float(error.size)

        # Efficient gradient via matmul: (D_out, B*S) @ (B*S, D_in) -> (D_out, D_in)
        # Flatten batch and seq dims, then use matmul instead of outer product
        batch_seq = error.shape[0] * error.shape[1]
        error_flat = error.reshape(batch_seq, -1)  # (B*S, D_out)
        keys_flat = keys.reshape(batch_seq, -1)    # (B*S, D_in)
        grad_w = scale * (error_flat.T @ keys_flat)  # (D_out, D_in)

        return [mx.clip(grad_w, -1.0, 1.0)]

    def _compute_gradients_deep(
        self,
        keys: mx.array,
        values: mx.array,
        weights: list[mx.array],
    ) -> list[mx.array]:
        """Optimized gradient computation for deep (multi-layer) memory.

        Uses matmul instead of expand_dims for efficient gradient computation.
        """
        num_layers = len(weights)
        batch_size, seq_len = keys.shape[0], keys.shape[1]
        batch_seq = batch_size * seq_len

        # Forward pass - collect activations
        activations = [keys]
        pre_activations = []
        h = keys

        for i in range(num_layers):
            h_pre = h @ weights[i].T
            pre_activations.append(h_pre)
            if i < num_layers - 1:
                h = self.memory.activation(h_pre)
                activations.append(h)
            else:
                h = h_pre

        # Error computation
        error = mx.clip(h - values, -10.0, 10.0)
        scale = 2.0 / float(error.size)
        delta = scale * error

        # Backward pass - compute gradients using efficient matmul
        grads = [None] * num_layers

        for i in range(num_layers - 1, -1, -1):
            act = activations[i]

            # Efficient gradient via matmul: (D_out, B*S) @ (B*S, D_in) -> (D_out, D_in)
            delta_flat = delta.reshape(batch_seq, -1)  # (B*S, D_out)
            act_flat = act.reshape(batch_seq, -1)      # (B*S, D_in)
            grad_w = delta_flat.T @ act_flat           # (D_out, D_in)
            grads[i] = mx.clip(grad_w, -1.0, 1.0)

            # Propagate gradient to previous layer
            if i > 0:
                delta = delta @ weights[i]
                # SiLU gradient: sig * (1 + x * (1 - sig))
                x = pre_activations[i - 1]
                sig = mx.sigmoid(x)
                delta = delta * sig * (1.0 + x * (1.0 - sig))

        return grads

    def init_state(self, batch_size: int) -> MemoryState:
        """Initialize memory state.

        Args:
            batch_size: Batch size (reserved for future use)

        Returns:
            Initial memory state
        """
        # Get initial weights from memory module
        weights = self.memory.get_weights()

        # Initialize momentum to zeros
        momentum = [mx.zeros_like(w) for w in weights]

        return MemoryState(weights=weights, momentum=momentum)

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
        return_state: bool = True,
    ) -> tuple[mx.array, MemoryState | None]:
        """Forward pass with memory update.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Previous memory state (optional)
            return_state: Whether to return updated state

        Returns:
            Tuple of (output, state) where output is (batch, seq, dim)
        """
        batch_size = x.shape[0]

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size)

        # Set memory weights from state
        self.memory.set_weights(state.weights)

        # Project to keys, values, queries
        k = self.proj_k(x)
        v = self.proj_v(x)
        q = self.proj_q(x)

        # Apply convolution
        k, v, q = self._apply_conv(k, v, q)

        # Apply SiLU activation
        k = nn.silu(k)
        v = nn.silu(v)
        q = nn.silu(q)

        # Normalize using L2-norm
        q = q / (mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True)) + 1e-8)
        k = k / (mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True)) + 1e-8)

        # Retrieve from memory
        retrieved = self.memory(q)

        # Compute data-dependent gates
        x_mean = mx.mean(x, axis=1, keepdims=True)
        alpha = mx.mean(self._apply_gate_decay(x_mean))
        theta = mx.mean(self._apply_gate_lr(x_mean)) * self.config.memory_lr
        eta = mx.mean(self._apply_gate_momentum(x_mean)) * self.config.memory_momentum

        # Compute gradients
        grads = self._compute_gradients(k, v, state.weights)

        # Update momentum: S_t = eta * S_{t-1} - theta * grad
        new_momentum = []
        for m, g in zip(state.momentum, grads):
            s = eta * m - theta * g
            new_momentum.append(s)

        # Update weights: M_t = (1 - alpha) * M_{t-1} + S_t
        new_weights = []
        for w, s in zip(state.weights, new_momentum):
            w_new = (1 - alpha) * w + s
            new_weights.append(w_new)

        # Output projection
        output = self.proj_out(retrieved)

        # Create new state
        new_state = MemoryState(weights=new_weights, momentum=new_momentum)

        if return_state:
            return output, new_state.detach()
        return output, None

    def _apply_gate_decay(self, x: mx.array) -> mx.array:
        """Apply decay gate."""
        h = self.gate_decay.layers[0](x)
        return mx.sigmoid(h)

    def _apply_gate_lr(self, x: mx.array) -> mx.array:
        """Apply learning rate gate."""
        h = self.gate_lr.layers[0](x)
        return mx.sigmoid(h)

    def _apply_gate_momentum(self, x: mx.array) -> mx.array:
        """Apply momentum gate."""
        h = self.gate_momentum.layers[0](x)
        return mx.sigmoid(h)

    def retrieve(
        self,
        queries: mx.array,
        state: MemoryState,
    ) -> mx.array:
        """Retrieve from memory without updating.

        Args:
            queries: Query vectors (batch, seq, dim)
            state: Memory state to query

        Returns:
            Retrieved values (batch, seq, dim)
        """
        # Set weights from state
        self.memory.set_weights(state.weights)

        # Project queries
        q = self.proj_q(queries)

        if self.use_conv:
            q = mx.transpose(q, (0, 2, 1))
            q = self.conv_q(q)[..., : q.shape[-1]]
            q = mx.transpose(q, (0, 2, 1))

        q = nn.silu(q)
        q = q / (mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True)) + 1e-8)

        # Retrieve
        retrieved = self.memory(q)
        return self.proj_out(retrieved)
