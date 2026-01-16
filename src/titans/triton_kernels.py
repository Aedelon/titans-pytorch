# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Triton kernels for optimized Titans memory operations.

This module provides GPU-optimized kernels using Triton for:
1. Memory update with momentum and weight decay
2. Fused gradient computation for associative memory loss
3. Efficient key-value retrieval

Triton (Tillet et al., 2019) enables writing high-performance GPU
kernels in Python with near-CUDA performance.

Installation:
    pip install triton

Requirements:
    - NVIDIA GPU with compute capability >= 7.0
    - CUDA 11.4+
    - PyTorch 2.0+
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Check for Triton availability
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return HAS_TRITON


if HAS_TRITON:
    # =============================================================================
    # Triton Kernels
    # =============================================================================

    @triton.jit
    def fused_add_rms_norm_kernel(
        # Pointers
        x_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        residual_out_ptr,  # Store x + residual for skip connection
        # Sizes
        n_rows,
        n_cols,
        eps,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused residual add + RMS normalization kernel.

        Computes:
            hidden = x + residual
            output = hidden / sqrt(mean(hidden^2) + eps) * weight

        Returns both hidden (for residual) and normalized output.
        """
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load x and residual
        x_ptrs = x_ptr + row_idx * n_cols + col_offsets
        res_ptrs = residual_ptr + row_idx * n_cols + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        res = tl.load(res_ptrs, mask=mask, other=0.0)

        # Add residual
        hidden = x + res

        # Store hidden for skip connection
        hidden_out_ptrs = residual_out_ptr + row_idx * n_cols + col_offsets
        tl.store(hidden_out_ptrs, hidden, mask=mask)

        # Compute RMS norm
        hidden_sq = hidden * hidden
        mean_sq = tl.sum(hidden_sq, axis=0) / n_cols
        rms = tl.sqrt(mean_sq + eps)
        hidden_norm = hidden / rms

        # Apply weight
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        output = hidden_norm * w

        # Store normalized output
        output_ptrs = output_ptr + row_idx * n_cols + col_offsets
        tl.store(output_ptrs, output, mask=mask)

    @triton.jit
    def rms_norm_kernel(
        # Pointers
        x_ptr,
        weight_ptr,
        output_ptr,
        # Sizes
        n_rows,
        n_cols,
        eps,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMS normalization kernel.

        Computes: output = x / sqrt(mean(x^2) + eps) * weight
        More efficient than separate operations.
        """
        # Get row index
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        # Compute offsets for this row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load row
        x_ptrs = x_ptr + row_idx * n_cols + col_offsets
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Compute RMS
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / n_cols
        rms = tl.sqrt(mean_sq + eps)

        # Normalize
        x_norm = x / rms

        # Load weight and apply
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        output = x_norm * w

        # Store
        output_ptrs = output_ptr + row_idx * n_cols + col_offsets
        tl.store(output_ptrs, output, mask=mask)

    @triton.jit
    def fused_silu_mul_kernel(
        # Pointers
        gate_ptr,
        up_ptr,
        output_ptr,
        # Size
        n_elements,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU(gate) * up kernel."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

        # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        # Cast to fp32 for exp() which doesn't support bf16
        gate_f32 = gate.to(tl.float32)
        neg_gate = -gate_f32
        exp_neg = tl.exp(neg_gate)
        sigmoid_gate = 1.0 / (1.0 + exp_neg)
        silu_gate = gate_f32 * sigmoid_gate

        # Multiply (keep in fp32 for accuracy, will be cast on store)
        up_f32 = up.to(tl.float32)
        output = silu_gate * up_f32

        # Store (auto-casts back to original dtype)
        tl.store(output_ptr + offsets, output, mask=mask)

    @triton.jit
    def memory_update_kernel(
        # Pointers
        weights_ptr,
        momentum_ptr,
        gradients_ptr,
        output_weights_ptr,
        output_momentum_ptr,
        # Scalars
        alpha,  # Decay/forgetting factor
        eta,  # Momentum coefficient
        theta,  # Learning rate
        # Sizes
        n_elements,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused memory update kernel.

        Implements:
            S_t = eta * S_{t-1} - theta * grad
            M_t = (1 - alpha) * M_{t-1} + S_t

        All operations are fused into a single kernel for efficiency.
        """
        # Get program ID
        pid = tl.program_id(0)

        # Compute block start and offsets
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load inputs
        weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
        momentum = tl.load(momentum_ptr + offsets, mask=mask, other=0.0)
        gradients = tl.load(gradients_ptr + offsets, mask=mask, other=0.0)

        # Compute new momentum: S_t = eta * S_{t-1} - theta * grad
        new_momentum = eta * momentum - theta * gradients

        # Compute new weights: M_t = (1 - alpha) * M_{t-1} + S_t
        new_weights = (1.0 - alpha) * weights + new_momentum

        # Store outputs
        tl.store(output_weights_ptr + offsets, new_weights, mask=mask)
        tl.store(output_momentum_ptr + offsets, new_momentum, mask=mask)

    @triton.jit
    def associative_memory_loss_kernel(
        # Pointers
        predictions_ptr,
        targets_ptr,
        output_ptr,
        # Sizes
        batch_size,
        seq_len,
        dim,
        # Block size
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute associative memory loss: ||M(k) - v||^2.

        Returns per-element squared differences for gradient computation.
        """
        # Get program ID
        pid = tl.program_id(0)

        # Compute position
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        n_elements = batch_size * seq_len * dim
        mask = offsets < n_elements

        # Load predictions and targets
        predictions = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
        targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

        # Compute squared difference
        diff = predictions - targets
        squared_diff = diff * diff

        # Store result
        tl.store(output_ptr + offsets, squared_diff, mask=mask)

    @triton.jit
    def fused_linear_forward_kernel(
        # Pointers
        input_ptr,
        weight_ptr,
        output_ptr,
        # Sizes
        batch_seq,
        in_features,
        out_features,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused linear layer forward pass.

        Computes output = input @ weight.T
        """
        # Get program IDs
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute starting positions
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Main loop over K dimension
        for k in range(0, in_features, BLOCK_K):
            # Load input block
            k_offs = k + offs_k
            input_ptrs = input_ptr + offs_m[:, None] * in_features + k_offs[None, :]
            mask_input = (offs_m[:, None] < batch_seq) & (k_offs[None, :] < in_features)
            input_block = tl.load(input_ptrs, mask=mask_input, other=0.0)

            # Load weight block (transposed)
            weight_ptrs = weight_ptr + offs_n[:, None] * in_features + k_offs[None, :]
            mask_weight = (offs_n[:, None] < out_features) & (
                k_offs[None, :] < in_features
            )
            weight_block = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

            # Accumulate
            acc += tl.dot(input_block, tl.trans(weight_block))

        # Store output
        output_ptrs = output_ptr + offs_m[:, None] * out_features + offs_n[None, :]
        mask_output = (offs_m[:, None] < batch_seq) & (offs_n[None, :] < out_features)
        tl.store(output_ptrs, acc, mask=mask_output)

    # =============================================================================
    # Python Wrappers
    # =============================================================================

    def triton_fused_add_rms_norm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual add + RMS normalization.

        Args:
            x: Input tensor (batch, seq, dim)
            residual: Residual tensor (same shape as x)
            weight: RMSNorm weight (dim,)
            eps: Epsilon for numerical stability

        Returns:
            Tuple of (hidden, normalized) where:
                hidden = x + residual (for next residual)
                normalized = rmsnorm(hidden, weight)
        """
        original_shape = x.shape
        x = x.contiguous()
        residual = residual.contiguous()

        # Reshape to 2D if needed
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))
            residual = residual.view(-1, residual.size(-1))

        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        hidden = torch.empty_like(x)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        fused_add_rms_norm_kernel[(n_rows,)](
            x, residual, weight, output, hidden,
            n_rows, n_cols, eps,
            BLOCK_SIZE,
        )

        return hidden.view(original_shape), output.view(original_shape)

    def triton_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """RMS normalization using Triton kernel.

        Args:
            x: Input tensor (batch, seq, dim) or (batch*seq, dim)
            weight: Weight tensor (dim,)
            eps: Epsilon for numerical stability

        Returns:
            Normalized tensor same shape as input
        """
        original_shape = x.shape
        x = x.contiguous()

        # Reshape to 2D if needed
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))

        n_rows, n_cols = x.shape
        output = torch.empty_like(x)

        # Block size must be >= n_cols for reduction
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        # Launch kernel
        rms_norm_kernel[(n_rows,)](
            x, weight, output,
            n_rows, n_cols, eps,
            BLOCK_SIZE,
        )

        return output.view(original_shape)

    def triton_fused_silu_mul(
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        """Fused SiLU(gate) * up using Triton kernel.

        Args:
            gate: Gate tensor
            up: Up tensor (same shape as gate)

        Returns:
            SiLU(gate) * up
        """
        gate = gate.contiguous()
        up = up.contiguous()

        n_elements = gate.numel()
        output = torch.empty_like(gate)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_silu_mul_kernel[grid](
            gate, up, output,
            n_elements,
            BLOCK_SIZE,
        )

        return output

    def triton_memory_update(
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        gradients: list[torch.Tensor],
        alpha: torch.Tensor | float,
        eta: torch.Tensor | float,
        theta: torch.Tensor | float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Update memory weights using Triton kernel.

        Args:
            weights: List of weight tensors
            momentum: List of momentum tensors
            gradients: List of gradient tensors
            alpha: Decay factor (tensor or float)
            eta: Momentum coefficient (tensor or float)
            theta: Learning rate (tensor or float)

        Returns:
            Tuple of (new_weights, new_momentum)
        """
        new_weights = []
        new_momentum = []

        BLOCK_SIZE = 1024

        # Convert tensors to floats - use mean if batched, extract scalar if 0-dim
        # This sync happens once per forward pass, not per element
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.flatten()[0].item() if alpha.numel() > 0 else 0.01
        if isinstance(eta, torch.Tensor):
            eta = eta.flatten()[0].item() if eta.numel() > 0 else 0.9
        if isinstance(theta, torch.Tensor):
            theta = theta.flatten()[0].item() if theta.numel() > 0 else 0.1

        for w, m, g in zip(weights, momentum, gradients, strict=True):
            # Ensure contiguous
            w = w.contiguous()
            m = m.contiguous()
            g = g.contiguous()

            # Allocate outputs
            out_w = torch.empty_like(w)
            out_m = torch.empty_like(m)

            n_elements = w.numel()
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            # Launch kernel
            memory_update_kernel[grid](
                w,
                m,
                g,
                out_w,
                out_m,
                alpha,
                eta,
                theta,
                n_elements,
                BLOCK_SIZE,
            )

            new_weights.append(out_w)
            new_momentum.append(out_m)

        return new_weights, new_momentum

    def triton_associative_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute associative memory loss using Triton.

        Args:
            predictions: Model predictions (batch, seq, dim)
            targets: Target values (batch, seq, dim)

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, dim = predictions.shape
        n_elements = predictions.numel()

        # Ensure contiguous
        predictions = predictions.contiguous()
        targets = targets.contiguous()

        # Allocate output
        output = torch.empty_like(predictions)

        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        # Launch kernel
        associative_memory_loss_kernel[grid](
            predictions,
            targets,
            output,
            batch_size,
            seq_len,
            dim,
            BLOCK_SIZE,
        )

        # Return mean loss
        return output.mean()

else:
    # Fallback implementations when Triton is not available

    def triton_memory_update(
        weights: list[torch.Tensor],
        momentum: list[torch.Tensor],
        gradients: list[torch.Tensor],
        alpha: float,
        eta: float,
        theta: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """PyTorch fallback for memory update."""
        new_weights = []
        new_momentum = []

        for w, m, g in zip(weights, momentum, gradients, strict=True):
            # S_t = eta * S_{t-1} - theta * grad
            new_m = eta * m - theta * g
            # M_t = (1 - alpha) * M_{t-1} + S_t
            new_w = (1 - alpha) * w + new_m

            new_weights.append(new_w)
            new_momentum.append(new_m)

        return new_weights, new_momentum

    def triton_associative_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorch fallback for associative loss."""
        return torch.nn.functional.mse_loss(predictions, targets, reduction="mean")


# =============================================================================
# Optimized Memory Module
# =============================================================================


class TritonMemoryMLP(nn.Module):
    """Memory MLP with optional Triton optimization.

    Falls back to standard PyTorch when Triton is unavailable.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "silu",
        use_triton: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_triton = use_triton and HAS_TRITON

        # Build layers
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(dim, dim, bias=False))
        else:
            self.layers.append(nn.Linear(dim, hidden_dim, bias=False))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.layers.append(nn.Linear(hidden_dim, dim, bias=False))

        # Activation
        activations = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
        }
        self.activation = activations.get(activation, nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def update_with_triton(
        self,
        gradients: list[torch.Tensor],
        momentum: list[torch.Tensor],
        alpha: float,
        eta: float,
        theta: float,
    ) -> list[torch.Tensor]:
        """Update weights using Triton-optimized kernel.

        Args:
            gradients: Gradients for each layer
            momentum: Current momentum for each layer
            alpha: Decay factor
            eta: Momentum coefficient
            theta: Learning rate

        Returns:
            New momentum values
        """
        weights = [layer.weight.data for layer in self.layers]

        new_weights, new_momentum = triton_memory_update(
            weights, momentum, gradients, alpha, eta, theta
        )

        # Update layer weights
        for layer, w in zip(self.layers, new_weights, strict=True):
            layer.weight.data = w

        return new_momentum
