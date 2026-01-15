# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Titans Model Architectures (MLX Implementation).

This module implements the three variants of Titans:
1. MAC (Memory as Context): Memory retrieval concatenated with input before attention
2. MAG (Memory as Gate): Memory and attention combined via gating
3. MAL (Memory as Layer): Memory used as a layer before attention

Plus the standalone LMM (Long-term Memory Module) without attention.

MLX-specific optimizations:
- Vectorized operations for Apple Silicon
- Unified memory architecture
- Lazy evaluation for optimal computation graphs
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from titans_mlx.attention import SegmentedAttention, SlidingWindowAttention
from titans_mlx.config import TitansConfig
from titans_mlx.memory import MemoryState, NeuralLongTermMemory
from titans_mlx.persistent import PersistentMemory


class FeedForward(nn.Module):
    """Feed-forward network with gating (following recent architectures) - MLX."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.hidden_dim = config.ffn_dim

        self.gate_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout_p = config.dropout

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SiLU gating."""
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout_p > 0:
            hidden = nn.Dropout(self.dropout_p)(hidden)
        return self.down_proj(hidden)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - MLX."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization."""
        rms = mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# MAC: Memory as Context
# =============================================================================


class MACBlock(nn.Module):
    """Memory as Context Block - MLX.

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
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
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
        batch_size = x.shape[0]

        # Initialize memory state if needed
        if state is None:
            state = self.memory.init_state(batch_size)

        # Step 1 (Eq. 21): Retrieve from memory using input as query
        # h_t = M*_{t-1}(q_t) - forward pass without weight update
        memory_retrieved = self.memory.retrieve(x, state)
        memory_tokens = self.norm_mem(memory_retrieved)

        # Get persistent memory tokens
        persistent = self.persistent(batch_size)

        # Steps 2-3 (Eq. 22-23): Attention with [persistent || memory || input]
        normed = self.norm1(x)
        attn_out = self.attention(normed, persistent=persistent, memory=memory_tokens)

        # Apply dropout
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)
        y_t = x + attn_out  # y_t is the attention output

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
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        output = output + ffn_out

        return output, new_state


class TitansMAC(nn.Module):
    """Titans with Memory as Context - MLX.

    Segments the sequence into chunks and processes each with MAC blocks.
    Long-term memory persists across chunks within a sequence.

    Optimized for Apple Silicon with:
    - JIT compiled block processing
    - Minimized intermediate evaluations
    - Efficient chunk concatenation
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAC blocks
        self.blocks = [MACBlock(config) for _ in range(config.num_layers)]

        # Output normalization and head
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Note: MLX doesn't support weight tying the same way as PyTorch
        # We'll handle it differently if needed

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def _process_single_chunk(
        self,
        chunk: mx.array,
        states: list[MemoryState | None],
    ) -> tuple[mx.array, list[MemoryState]]:
        """Process a single chunk through all blocks."""
        new_states = []
        for i, block in enumerate(self.blocks):
            chunk, new_state = block(chunk, state=states[i])
            new_states.append(new_state)
        return chunk, new_states

    def _process_all_chunks_compiled(
        self,
        x: mx.array,
        states: list[MemoryState | None],
        chunk_size: int,
    ) -> tuple[mx.array, list[MemoryState]]:
        """Process all chunks with minimal Python overhead.

        This function is designed to be JIT compiled.
        """
        seq_len = x.shape[1]
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Process chunks and collect outputs
        outputs = []
        current_states = states

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk = x[:, start:end]

            # Process through all blocks
            for j, block in enumerate(self.blocks):
                chunk, new_state = block(chunk, state=current_states[j])
                if i == 0:
                    current_states = list(current_states)  # Copy on first iteration
                current_states[j] = new_state

            outputs.append(chunk)

        # Concatenate all outputs
        return mx.concatenate(outputs, axis=1), current_states

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
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

        # Fast path: if sequence fits in one chunk, skip chunking overhead
        if seq_len <= chunk_size:
            x, new_states = self._process_single_chunk(x, states)
            x = self.norm(x)
            logits = self.head(x)
            return logits, new_states

        # Process in chunks - collect outputs without intermediate evaluations
        outputs = []
        new_states = list(states)  # Make a copy

        # Calculate number of chunks upfront
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk = x[:, chunk_start:chunk_end]

            # Process through blocks
            chunk, new_states = self._process_single_chunk(chunk, new_states)
            outputs.append(chunk)

        # Concatenate all outputs at once
        x = mx.concatenate(outputs, axis=1)

        # Output projection
        x = self.norm(x)
        logits = self.head(x)

        return logits, new_states


# =============================================================================
# MAG: Memory as Gate
# =============================================================================


class MAGBlock(nn.Module):
    """Memory as Gate Block - MLX.

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
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
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
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)
        y_t = x + attn_out

        # Eq. 27: M_t = M_{t-1}(x_t) - Memory update with input
        mem_out, new_state = self.memory(normed, state=state)

        # Eq. 28: o_t = y_t ⊗ M*_t(x_t) - Element-wise product
        # Use memory output (which is M*(x)) as the gate
        output = y_t * mem_out

        # Feed-forward
        normed = self.norm2(output)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        output = output + ffn_out

        return output, new_state


class TitansMAG(nn.Module):
    """Titans with Memory as Gate - MLX.

    Uses sliding window attention and long-term memory in parallel,
    combined via a gating mechanism.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAG blocks
        self.blocks = [MAGBlock(config) for _ in range(config.num_layers)]

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
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
    """Memory as Layer Block - MLX.

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
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
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
        if self.dropout_p > 0:
            mem_out = nn.Dropout(self.dropout_p)(mem_out)
        x = x + mem_out

        # Attention layer with persistent prefix
        normed = self.norm2(x)
        attn_out = self.attention(normed, prefix=persistent)
        if self.dropout_p > 0:
            attn_out = nn.Dropout(self.dropout_p)(attn_out)
        x = x + attn_out

        # Feed-forward
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        x = x + ffn_out

        return x, new_state


class TitansMAL(nn.Module):
    """Titans with Memory as Layer - MLX.

    Memory processes input before attention in a sequential manner.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of MAL blocks
        self.blocks = [MALBlock(config) for _ in range(config.num_layers)]

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
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
    """Standalone Long-term Memory Block (no attention) - MLX.

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
        self.dropout_p = config.dropout

    def __call__(
        self,
        x: mx.array,
        state: MemoryState | None = None,
    ) -> tuple[mx.array, MemoryState]:
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
        if self.dropout_p > 0:
            mem_out = nn.Dropout(self.dropout_p)(mem_out)
        x = x + mem_out

        # Feed-forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        if self.dropout_p > 0:
            ffn_out = nn.Dropout(self.dropout_p)(ffn_out)
        x = x + ffn_out

        return x, new_state


class TitansLMM(nn.Module):
    """Titans with only Long-term Memory (no attention) - MLX.

    A sequence model using only the neural memory module.
    Tests memory's standalone capability.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        # Stack of LMM blocks
        self.blocks = [LMMBlock(config) for _ in range(config.num_layers)]

        # Output
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape) * self.config.init_std
        )

    def __call__(
        self,
        input_ids: mx.array,
        states: list[MemoryState] | None = None,
    ) -> tuple[mx.array, list[MemoryState]]:
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
