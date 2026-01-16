# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans Model Architectures.

This module implements the three variants of Titans:
1. MAC (Memory as Context): Memory retrieval concatenated with input before attention
2. MAG (Memory as Gate): Memory and attention combined via gating
3. MAL (Memory as Layer): Memory used as a layer before attention

Plus the standalone LMM (Long-term Memory Module) without attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans.attention import SegmentedAttention, SlidingWindowAttention
from titans.config import TitansConfig
from titans.memory import MemoryState, NeuralLongTermMemory
from titans.persistent import PersistentMemory


class FeedForward(nn.Module):
    """Feed-forward network with gating (following recent architectures).

    Uses Triton fused kernel when available for better performance.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.ffn_dim

        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self._use_triton: bool | None = None

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        """Check if we should use Triton kernel."""
        if self._use_triton is None:
            try:
                from titans.triton_kernels import is_triton_available, triton_fused_silu_mul
                self._use_triton = is_triton_available()
                self._triton_silu_mul = triton_fused_silu_mul
            except ImportError:
                self._use_triton = False
        return self._use_triton and x.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SiLU gating."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Fused SiLU + multiply
        if self._should_use_triton(x):
            try:
                hidden = self._triton_silu_mul(gate, up)
            except Exception:
                hidden = F.silu(gate) * up
        else:
            hidden = F.silu(gate) * up

        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Uses Triton kernel when available for better performance.
    Supports fused residual add + norm for efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._use_triton: bool | None = None  # Lazy init
        self._triton_rms_norm = None
        self._triton_fused_add_rms_norm = None

    def _init_triton(self, x: torch.Tensor) -> bool:
        """Initialize Triton kernels if available."""
        if self._use_triton is None:
            try:
                from titans.triton_kernels import (
                    is_triton_available,
                    triton_rms_norm,
                    triton_fused_add_rms_norm,
                )
                self._use_triton = is_triton_available() and x.is_cuda
                self._triton_rms_norm = triton_rms_norm
                self._triton_fused_add_rms_norm = triton_fused_add_rms_norm
            except ImportError:
                self._use_triton = False
        return self._use_triton and x.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        if self._init_triton(x) and x.size(-1) == self.dim:
            try:
                return self._triton_rms_norm(x, self.weight, self.eps)
            except Exception:
                pass  # Fallback to PyTorch
        # PyTorch fallback
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def forward_with_residual(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused residual add + RMS normalization.

        Args:
            x: Input tensor
            residual: Residual tensor to add

        Returns:
            Tuple of (hidden, normalized) where hidden = x + residual
        """
        if self._init_triton(x) and x.size(-1) == self.dim:
            try:
                return self._triton_fused_add_rms_norm(x, residual, self.weight, self.eps)
            except Exception:
                pass
        # PyTorch fallback
        hidden = x + residual
        rms = torch.sqrt(torch.mean(hidden**2, dim=-1, keepdim=True) + self.eps)
        return hidden, hidden / rms * self.weight


# =============================================================================
# MAC: Memory as Context
# =============================================================================


class MACBlock(nn.Module):
    """Memory as Context Block.

    Architecture:
    1. Retrieve from long-term memory using input as query
    2. Concatenate: [persistent] || [memory] || [input]
    3. Apply segmented attention
    4. Feed-forward network

    At test time:
    - Persistent memory parameters are fixed
    - Attention performs in-context learning
    - Long-term memory continues learning (weight updates)
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Segmented attention (Core module)
        self.attention = SegmentedAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm_mem = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass for MAC block.

        Following the paper (Section 4.1, Eq. 21-25):
        1. h_t = M*_{t-1}(q_t) - Retrieve from memory using input as query (Eq. 21)
        2. S̃^(t) = [persistent] || h_t || x - Concatenate (Eq. 22)
        3. y_t = Attn(S̃^(t)) - Attention (Eq. 23)
        4. M_t = M_{t-1}(y_t) - Update memory with attention output (Eq. 24)
        5. o_t = y_t ⊗ M*_t(y_t) - Final output (Eq. 25)

        Args:
            x: Input tensor (batch, seq, dim) - single chunk/segment
            state: Memory state from previous chunk

        Returns:
            Tuple of (output, new_state)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize memory state if needed
        if state is None:
            state = self.memory.init_state(batch_size, x.device)

        # Step 1 (Eq. 21): Retrieve from memory using input as query
        # h_t = M*_{t-1}(q_t) - forward pass without weight update
        memory_retrieved = self.memory.retrieve(x, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        # Get persistent memory tokens
        persistent = self.persistent(batch_size)

        # Steps 2-3 (Eq. 22-23): Attention with [persistent || memory || input]
        normed = self.norm1(x)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)
        y_t = x + self.dropout(attn_out)  # y_t is the attention output

        # Step 4 (Eq. 24): Update memory with attention output
        # M_t = M_{t-1}(y_t) - this updates memory weights
        _, new_state = self.memory(y_t, state=state)

        # Step 5 (Eq. 25): Final output o_t = y_t ⊗ M*_t(y_t)
        # Retrieve from updated memory
        mem_out = self.memory.retrieve(y_t, new_state)
        output = y_t * mem_out  # Element-wise product

        # Feed-forward
        normed = self.norm2(output)
        ffn_out = self.ffn(normed)
        output = output + self.dropout(ffn_out)

        return output, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context.

    Segments the sequence into chunks and processes each with MAC blocks.
    Long-term memory persists across chunks within a sequence.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAC blocks
        self.blocks = nn.ModuleList(
            [MACBlock(config) for _ in range(config.num_layers)]
        )

        # Output normalization and head
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states for each layer

        Returns:
            Tuple of (logits, new_states)
        """
        batch_size, seq_len = input_ids.shape
        chunk_size = self.config.chunk_size

        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process in chunks
        outputs = []
        new_states = [None] * len(self.blocks)

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk = x[:, chunk_start:chunk_end]

            # Process through blocks
            chunk_states = states
            for i, block in enumerate(self.blocks):
                chunk, new_state = block(chunk, state=chunk_states[i])
                new_states[i] = new_state

            outputs.append(chunk)

            # Update states for next chunk
            states = new_states

        # Concatenate outputs
        x = torch.cat(outputs, dim=1)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# MAG: Memory as Gate
# =============================================================================


class MAGBlock(nn.Module):
    """Memory as Gate Block.

    Architecture (Section 4.2, Eq. 26-28):
    1. y_t = Attn(x) - Sliding window attention (Eq. 26)
    2. M_t = M_{t-1}(x_t) - Update memory with input (Eq. 27)
    3. o_t = y_t ⊗ M*_t(x_t) - Element-wise product (Eq. 28)

    The attention handles precise local dependencies,
    while memory provides fading long-range context.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Persistent memory (prepended to input)
        self.persistent = PersistentMemory(config)

        # Sliding window attention
        self.attention = SlidingWindowAttention(config)

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass for MAG block.

        Following the paper (Section 4.2, Eq. 26-28):
        1. y_t = Attn(x) - Attention on input (Eq. 26)
        2. M_t = M_{t-1}(x_t) - Update memory with input (Eq. 27)
        3. o_t = y_t ⊗ M*_t(x_t) - Output is element-wise product (Eq. 28)

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Get persistent memory as prefix for attention
        persistent = self.persistent(batch_size)

        # Eq. 26: y_t = Attn(x) - Attention branch
        normed = self.norm1(x)
        attn_out = self.attention(normed, prefix=persistent)
        y_t = x + self.dropout(attn_out)

        # Eq. 27: M_t = M_{t-1}(x_t) - Memory update with input
        mem_out, new_state = self.memory(normed, state=state)

        # Eq. 28: o_t = y_t ⊗ M*_t(x_t) - Element-wise product
        # Use memory output (which is M*(x)) as the gate
        output = y_t * mem_out

        # Feed-forward
        normed = self.norm2(output)
        ffn_out = self.ffn(normed)
        output = output + self.dropout(ffn_out)

        return output, new_state


class TitansMAG(nn.Module):
    """Titans with Memory as Gate.

    Uses sliding window attention and long-term memory in parallel,
    combined via a gating mechanism.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAG blocks
        self.blocks = nn.ModuleList(
            [MAGBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# MAL: Memory as Layer
# =============================================================================


class MALBlock(nn.Module):
    """Memory as Layer Block.

    Architecture:
    1. Long-term memory processes input
    2. Sliding window attention on memory output
    3. Feed-forward network

    Memory acts as a preprocessing layer before attention.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Persistent memory
        self.persistent = PersistentMemory(config)

        # Long-term memory (first layer)
        self.memory = NeuralLongTermMemory(config)

        # Sliding window attention (second layer)
        self.attention = SlidingWindowAttention(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)
        self.norm3 = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass for MAL block.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        batch_size = x.shape[0]

        # Get persistent memory
        persistent = self.persistent(batch_size)

        # Memory layer
        normed = self.norm1(x)
        mem_out, new_state = self.memory(normed, state=state)
        x = x + self.dropout(mem_out)

        # Attention layer with persistent prefix
        normed = self.norm2(x)
        attn_out = self.attention(normed, prefix=persistent)
        x = x + self.dropout(attn_out)

        # Feed-forward
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, new_state


class TitansMAL(nn.Module):
    """Titans with Memory as Layer.

    Memory processes input before attention in a sequential manner.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAL blocks
        self.blocks = nn.ModuleList(
            [MALBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# LMM: Long-term Memory Module (standalone)
# =============================================================================


class LMMBlock(nn.Module):
    """Standalone Long-term Memory Block (no attention).

    Uses only the neural memory module as a sequence model.
    This tests the memory's ability to work independently.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Long-term memory
        self.memory = NeuralLongTermMemory(config)

        # Feed-forward
        self.ffn = FeedForward(config)

        # Layer norms
        self.norm1 = RMSNorm(config.dim)
        self.norm2 = RMSNorm(config.dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
    ) -> tuple[torch.Tensor, MemoryState]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq, dim)
            state: Memory state

        Returns:
            Tuple of (output, new_state)
        """
        # Memory
        normed = self.norm1(x)
        mem_out, new_state = self.memory(normed, state=state)
        x = x + self.dropout(mem_out)

        # Feed-forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x, new_state


class TitansLMM(nn.Module):
    """Titans with only Long-term Memory (no attention).

    A sequence model using only the neural memory module.
    Tests memory's standalone capability.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of LMM blocks
        self.blocks = nn.ModuleList(
            [LMMBlock(config) for _ in range(config.num_layers)]
        )

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie weights
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, list[MemoryState]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            states: List of memory states

        Returns:
            Tuple of (logits, new_states)
        """
        # Initialize states if needed
        if states is None:
            states = [None] * len(self.blocks)

        # Embed
        x = self.embed(input_ids)

        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, state=states[i])
            new_states.append(new_state)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states
