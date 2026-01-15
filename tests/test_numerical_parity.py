# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Numerical parity tests between MLX and PyTorch implementations.

These tests verify that MLX and PyTorch produce identical outputs
when initialized with the same weights.
"""

import numpy as np
import pytest

# Skip all tests if PyTorch or MLX is not available
torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")

from titans import TitansConfig as TorchConfig
from titans import TitansLMM as TorchLMM
from titans import TitansMAC as TorchMAC
from titans import TitansMAG as TorchMAG
from titans.memory import NeuralLongTermMemory as TorchMemory

from titans_mlx import TitansConfig as MLXConfig
from titans_mlx import TitansLMM as MLXLMM
from titans_mlx import TitansMAC as MLXMAC
from titans_mlx import TitansMAG as MLXMAG
from titans_mlx.memory import NeuralLongTermMemory as MLXMemory


def sync_memory_weights(mlx_mem, torch_mem):
    """Synchronize weights from PyTorch memory to MLX memory."""
    # Projections
    for proj in ["proj_k", "proj_v", "proj_q", "proj_out"]:
        torch_w = getattr(torch_mem, proj).weight.detach().numpy()
        getattr(mlx_mem, proj).weight = mx.array(torch_w)

    # Memory MLP layers
    for i, (mlx_layer, torch_layer) in enumerate(
        zip(mlx_mem.memory.layers, torch_mem.memory.layers)
    ):
        mlx_layer.weight = mx.array(torch_layer.weight.detach().numpy())

    # Gates
    for gate_name in ["gate_decay", "gate_lr", "gate_momentum"]:
        torch_gate = getattr(torch_mem, gate_name)
        mlx_gate = getattr(mlx_mem, gate_name)
        mlx_gate.layers[0].weight = mx.array(torch_gate[0].weight.detach().numpy())
        mlx_gate.layers[0].bias = mx.array(torch_gate[0].bias.detach().numpy())


def sync_lmm_weights(mlx_model, torch_model):
    """Synchronize weights from PyTorch LMM to MLX LMM."""
    mlx_model.embed.weight = mx.array(torch_model.embed.weight.detach().numpy())
    mlx_model.head.weight = mx.array(torch_model.head.weight.detach().numpy())
    mlx_model.norm.weight = mx.array(torch_model.norm.weight.detach().numpy())

    for mlx_block, torch_block in zip(mlx_model.blocks, torch_model.blocks):
        mlx_block.norm1.weight = mx.array(torch_block.norm1.weight.detach().numpy())
        mlx_block.norm2.weight = mx.array(torch_block.norm2.weight.detach().numpy())

        mlx_block.ffn.gate_proj.weight = mx.array(
            torch_block.ffn.gate_proj.weight.detach().numpy()
        )
        mlx_block.ffn.up_proj.weight = mx.array(
            torch_block.ffn.up_proj.weight.detach().numpy()
        )
        mlx_block.ffn.down_proj.weight = mx.array(
            torch_block.ffn.down_proj.weight.detach().numpy()
        )

        sync_memory_weights(mlx_block.memory, torch_block.memory)


def sync_mag_weights(mlx_model, torch_model):
    """Synchronize weights from PyTorch MAG to MLX MAG."""
    mlx_model.embed.weight = mx.array(torch_model.embed.weight.detach().numpy())
    mlx_model.head.weight = mx.array(torch_model.head.weight.detach().numpy())
    mlx_model.norm.weight = mx.array(torch_model.norm.weight.detach().numpy())

    for mlx_block, torch_block in zip(mlx_model.blocks, torch_model.blocks):
        mlx_block.norm1.weight = mx.array(torch_block.norm1.weight.detach().numpy())
        mlx_block.norm2.weight = mx.array(torch_block.norm2.weight.detach().numpy())

        mlx_block.persistent.tokens = mx.array(
            torch_block.persistent.tokens.detach().numpy()
        )

        mlx_block.ffn.gate_proj.weight = mx.array(
            torch_block.ffn.gate_proj.weight.detach().numpy()
        )
        mlx_block.ffn.up_proj.weight = mx.array(
            torch_block.ffn.up_proj.weight.detach().numpy()
        )
        mlx_block.ffn.down_proj.weight = mx.array(
            torch_block.ffn.down_proj.weight.detach().numpy()
        )

        # Attention
        attn_mlx = mlx_block.attention
        attn_torch = torch_block.attention
        attn_mlx.proj_q.weight = mx.array(attn_torch.proj_q.weight.detach().numpy())
        attn_mlx.proj_k.weight = mx.array(attn_torch.proj_k.weight.detach().numpy())
        attn_mlx.proj_v.weight = mx.array(attn_torch.proj_v.weight.detach().numpy())
        attn_mlx.proj_out.weight = mx.array(attn_torch.proj_out.weight.detach().numpy())

        sync_memory_weights(mlx_block.memory, torch_block.memory)


def sync_mac_weights(mlx_model, torch_model):
    """Synchronize weights from PyTorch MAC to MLX MAC."""
    mlx_model.embed.weight = mx.array(torch_model.embed.weight.detach().numpy())
    mlx_model.head.weight = mx.array(torch_model.head.weight.detach().numpy())
    mlx_model.norm.weight = mx.array(torch_model.norm.weight.detach().numpy())

    for mlx_block, torch_block in zip(mlx_model.blocks, torch_model.blocks):
        mlx_block.norm1.weight = mx.array(torch_block.norm1.weight.detach().numpy())
        mlx_block.norm2.weight = mx.array(torch_block.norm2.weight.detach().numpy())
        mlx_block.norm_mem.weight = mx.array(
            torch_block.norm_mem.weight.detach().numpy()
        )

        mlx_block.persistent.tokens = mx.array(
            torch_block.persistent.tokens.detach().numpy()
        )

        mlx_block.ffn.gate_proj.weight = mx.array(
            torch_block.ffn.gate_proj.weight.detach().numpy()
        )
        mlx_block.ffn.up_proj.weight = mx.array(
            torch_block.ffn.up_proj.weight.detach().numpy()
        )
        mlx_block.ffn.down_proj.weight = mx.array(
            torch_block.ffn.down_proj.weight.detach().numpy()
        )

        # Segmented Attention
        attn_mlx = mlx_block.attention
        attn_torch = torch_block.attention
        attn_mlx.proj_q.weight = mx.array(attn_torch.proj_q.weight.detach().numpy())
        attn_mlx.proj_k.weight = mx.array(attn_torch.proj_k.weight.detach().numpy())
        attn_mlx.proj_v.weight = mx.array(attn_torch.proj_v.weight.detach().numpy())
        attn_mlx.proj_out.weight = mx.array(attn_torch.proj_out.weight.detach().numpy())

        sync_memory_weights(mlx_block.memory, torch_block.memory)


class TestNumericalParity:
    """Test numerical parity between MLX and PyTorch implementations."""

    @pytest.fixture
    def config_kwargs(self):
        """Common configuration for all tests."""
        return {
            "vocab_size": 1000,
            "dim": 64,
            "num_heads": 2,
            "num_layers": 1,
            "num_memory_layers": 1,
            "num_persistent_tokens": 4,
            "chunk_size": 16,
            "window_size": 32,
            "use_conv": False,
            "dropout": 0.0,
        }

    @pytest.fixture
    def input_ids(self):
        """Common input for all tests."""
        np.random.seed(42)
        return np.random.randint(0, 1000, (2, 32))

    def test_memory_parity(self, config_kwargs):
        """Test NeuralLongTermMemory produces identical outputs."""
        mlx_config = MLXConfig(**config_kwargs)
        torch_config = TorchConfig(**config_kwargs)

        mlx_mem = MLXMemory(mlx_config)
        torch_mem = TorchMemory(torch_config)

        sync_memory_weights(mlx_mem, torch_mem)
        mx.eval(mlx_mem.parameters())

        np.random.seed(42)
        input_np = np.random.randn(2, 16, 64).astype(np.float32)

        mlx_input = mx.array(input_np)
        torch_input = torch.tensor(input_np, dtype=torch.float32)

        mlx_out, _ = mlx_mem(mlx_input, state=None)
        mx.eval(mlx_out)

        with torch.no_grad():
            torch_out, _ = torch_mem(torch_input, state=None)

        max_diff = np.max(np.abs(np.array(mlx_out) - torch_out.numpy()))
        assert max_diff < 1e-5, f"Memory output diff too large: {max_diff}"

    def test_lmm_parity(self, config_kwargs, input_ids):
        """Test TitansLMM produces identical outputs."""
        mlx_config = MLXConfig(**config_kwargs)
        torch_config = TorchConfig(**config_kwargs)

        mlx_model = MLXLMM(mlx_config)
        torch_model = TorchLMM(torch_config)

        sync_lmm_weights(mlx_model, torch_model)
        mx.eval(mlx_model.parameters())

        mlx_input = mx.array(input_ids)
        torch_input = torch.tensor(input_ids, dtype=torch.long)

        mlx_out, _ = mlx_model(mlx_input)
        mx.eval(mlx_out)

        with torch.no_grad():
            torch_out, _ = torch_model(torch_input)

        max_diff = np.max(np.abs(np.array(mlx_out) - torch_out.numpy()))
        assert max_diff < 1e-4, f"LMM output diff too large: {max_diff}"

    def test_mag_parity(self, config_kwargs, input_ids):
        """Test TitansMAG produces identical outputs."""
        mlx_config = MLXConfig(**config_kwargs)
        torch_config = TorchConfig(**config_kwargs)

        mlx_model = MLXMAG(mlx_config)
        torch_model = TorchMAG(torch_config)

        sync_mag_weights(mlx_model, torch_model)
        mx.eval(mlx_model.parameters())

        mlx_input = mx.array(input_ids)
        torch_input = torch.tensor(input_ids, dtype=torch.long)

        mlx_out, _ = mlx_model(mlx_input)
        mx.eval(mlx_out)

        with torch.no_grad():
            torch_out, _ = torch_model(torch_input)

        max_diff = np.max(np.abs(np.array(mlx_out) - torch_out.numpy()))
        assert max_diff < 1e-4, f"MAG output diff too large: {max_diff}"

    def test_mac_parity(self, config_kwargs, input_ids):
        """Test TitansMAC produces identical outputs."""
        mlx_config = MLXConfig(**config_kwargs)
        torch_config = TorchConfig(**config_kwargs)

        mlx_model = MLXMAC(mlx_config)
        torch_model = TorchMAC(torch_config)

        sync_mac_weights(mlx_model, torch_model)
        mx.eval(mlx_model.parameters())

        mlx_input = mx.array(input_ids)
        torch_input = torch.tensor(input_ids, dtype=torch.long)

        mlx_out, _ = mlx_model(mlx_input)
        mx.eval(mlx_out)

        with torch.no_grad():
            torch_out, _ = torch_model(torch_input)

        max_diff = np.max(np.abs(np.array(mlx_out) - torch_out.numpy()))
        assert max_diff < 1e-4, f"MAC output diff too large: {max_diff}"
