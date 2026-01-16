# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Attention modules for Titans architecture.

This module implements:
1. Sliding Window Attention (SWA) - for MAG and MAL variants
2. Segmented Attention - for MAC variant with full causal attention per segment
3. Rotary Position Embeddings (RoPE)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from titans.config import TitansConfig


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary position embeddings to queries and keys.
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(
        self, dim: int, max_seq_len: int = 8192, base: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(
        self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Build cos/sin cache for given sequence length, device, and dtype."""
        inv_freq = self.inv_freq
        if device is not None:
            inv_freq = inv_freq.to(device)

        positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq.float())

        # Compute cos and sin in target dtype
        cos = freqs.cos()
        sin = freqs.sin()

        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self._cache_dtype = dtype
        self._cache_device = device

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.

        Args:
            q: Queries (batch, heads, seq, head_dim)
            k: Keys (batch, heads, seq, head_dim)
            seq_offset: Offset for position indices

        Returns:
            Tuple of rotated (q, k)
        """
        seq_len = q.shape[2]
        device = q.device

        # Rebuild cache if needed (length, device, or dtype changed)
        need_rebuild = (
            seq_offset + seq_len > self.cos_cached.shape[0]
            or getattr(self, "_cache_device", None) != device
            or getattr(self, "_cache_dtype", None) != q.dtype
        )
        if need_rebuild:
            self._build_cache(max(seq_offset + seq_len, self.max_seq_len), device, q.dtype)

        # Get cached cos/sin - already in correct dtype/device
        cos = self.cos_cached[seq_offset : seq_offset + seq_len]
        sin = self.sin_cached[seq_offset : seq_offset + seq_len]

        # Apply rotation
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embedding to tensor.

        Args:
            x: Input tensor (batch, heads, seq, head_dim)
            cos: Cosine values (seq, head_dim // 2)
            sin: Sine values (seq, head_dim // 2)

        Returns:
            Rotated tensor
        """
        # Split into even and odd parts
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Expand cos/sin for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        rotated = torch.stack(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            dim=-1,
        )
        return rotated.flatten(-2)


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention (SWA).

    Implements local attention with a fixed window size.
    Each position can only attend to positions within the window.
    Used in MAG and MAL variants of Titans.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.scale = self.head_dim**-0.5

        # Projections
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Rotary embeddings
        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
            )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create sliding window causal mask.

        Args:
            seq_len: Sequence length
            device: Device for mask

        Returns:
            Boolean mask where True = attend, False = mask out
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)

        # Compute distances
        row_idx = positions.unsqueeze(1)  # (seq, 1)
        col_idx = positions.unsqueeze(0)  # (1, seq)

        # Causal: can only attend to past (including self)
        causal_mask = col_idx <= row_idx

        # Window: can only attend within window
        window_mask = (row_idx - col_idx) < self.window_size

        # Combine
        mask = causal_mask & window_mask

        return mask

    def forward(
        self,
        x: torch.Tensor,
        prefix: torch.Tensor | None = None,
        seq_offset: int = 0,
    ) -> torch.Tensor:
        """Forward pass with sliding window attention.

        Args:
            x: Input tensor (batch, seq, dim)
            prefix: Optional prefix tokens that can be attended to (batch, prefix_len, dim)
            seq_offset: Offset for rotary embeddings

        Returns:
            Output tensor (batch, seq, dim)
        """
        batch_size, seq_len, _ = x.shape

        # If prefix provided, concatenate it
        if prefix is not None:
            full_x = torch.cat([prefix, x], dim=1)
            prefix_len = prefix.shape[1]
        else:
            full_x = x
            prefix_len = 0

        full_len = full_x.shape[1]

        # Project Q, K, V
        q = self.proj_q(x)  # Only from x, not prefix
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE
        if self.rope is not None:
            # For k/v, we need to account for prefix
            q, _ = self.rope(q, q, seq_offset=prefix_len + seq_offset)
            k, _ = self.rope(k, k, seq_offset=seq_offset)

        # Create attention mask for sliding window
        # For queries in x, we create a mask for attending to full_x
        mask = self._create_extended_mask(seq_len, full_len, prefix_len, x.device)
        # Convert to additive mask for SDPA (0 = attend, -inf = mask)
        attn_mask = torch.zeros_like(mask, dtype=q.dtype)
        attn_mask.masked_fill_(~mask, float("-inf"))

        # Use PyTorch SDPA for efficiency
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,  # We provide explicit mask
            scale=self.scale,
        )

        # Reshape back
        output = rearrange(output, "b h s d -> b s (h d)")

        # Output projection
        output = self.proj_out(output)

        return output

    def _create_extended_mask(
        self,
        query_len: int,
        key_len: int,
        prefix_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create mask for queries attending to keys (including prefix).

        Args:
            query_len: Length of query sequence
            key_len: Length of key sequence (prefix + query)
            prefix_len: Length of prefix
            device: Device for mask

        Returns:
            Boolean mask (1, 1, query_len, key_len)
        """
        # Queries can always attend to all prefix tokens
        prefix_mask = torch.ones(query_len, prefix_len, dtype=torch.bool, device=device)

        # For non-prefix positions, use sliding window causal mask
        if key_len > prefix_len:
            main_mask = self._create_sliding_window_mask(query_len, device)
        else:
            main_mask = torch.empty(query_len, 0, dtype=torch.bool, device=device)

        # Combine
        mask = torch.cat([prefix_mask, main_mask], dim=1)

        return mask.unsqueeze(0).unsqueeze(0)


class SegmentedAttention(nn.Module):
    """Segmented/Chunked Attention for MAC variant.

    Implements full causal attention within each segment/chunk.
    The segment includes:
    1. Persistent memory tokens (fixed)
    2. Retrieved long-term memory tokens
    3. Current input chunk

    This is the "Core" module in the MAC architecture.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        # Projections
        self.proj_q = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_k = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_v = nn.Linear(config.dim, config.dim, bias=False)
        self.proj_out = nn.Linear(config.dim, config.dim, bias=False)

        # Rotary embeddings
        self.rope: RotaryPositionEmbedding | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbedding(
                dim=config.head_dim,
                max_seq_len=config.max_seq_len,
            )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj_out]:
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(
        self,
        x: torch.Tensor,
        persistent: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with segmented attention.

        The full sequence is: [persistent] || [memory] || [input]

        Args:
            x: Input tensor (batch, seq, dim)
            persistent: Persistent memory tokens (batch, num_persistent, dim)
            memory: Retrieved long-term memory (batch, num_memory, dim)

        Returns:
            Output tensor (batch, seq, dim) - only for input positions
        """
        batch_size, seq_len, _ = x.shape

        # Build full sequence
        components = []
        prefix_lens = []

        if persistent is not None:
            components.append(persistent)
            prefix_lens.append(persistent.shape[1])

        if memory is not None:
            components.append(memory)
            prefix_lens.append(memory.shape[1])

        components.append(x)

        full_x = torch.cat(components, dim=1)
        full_len = full_x.shape[1]
        prefix_len = sum(prefix_lens)

        # Project Q, K, V
        q = self.proj_q(full_x)
        k = self.proj_k(full_x)
        v = self.proj_v(full_x)

        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        # Apply RoPE
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Use PyTorch SDPA for efficiency (Flash Attention when available)
        # SDPA handles causal masking internally with is_causal=True
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )

        # Reshape back
        output = rearrange(output, "b h s d -> b s (h d)")

        # Output projection
        output = self.proj_out(output)

        # Return only the input positions (not persistent/memory)
        return output[:, prefix_len:]

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create full causal mask.

        Args:
            seq_len: Sequence length
            device: Device for mask

        Returns:
            Boolean mask (1, 1, seq, seq) where True = attend
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
