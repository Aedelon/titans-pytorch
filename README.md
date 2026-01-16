# Titans: Learning to Memorize at Test Time

<p align="center">
<img src="assets/hero.png" alt="Titans Hero" width="100%"/>
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MLX](https://img.shields.io/badge/mlx-apple%20silicon-black.svg)](https://ml-explore.github.io/mlx/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-105%20passed-brightgreen.svg)](tests/)

A complete **PyTorch** and **MLX** (Apple Silicon) implementation of the Titans architecture from Google Research.

Titans introduce a **Neural Long-term Memory (LMM)** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. This enables attention mechanisms to focus on local context while utilizing long-range information through neural memory.

---

## Table of Contents

- [Paper References](#paper-references)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
  - [Memory Perspective](#memory-perspective)
  - [Architecture Variants](#architecture-variants)
  - [Neural Long-term Memory](#neural-long-term-memory)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [PyTorch](#pytorch-quick-start)
  - [MLX (Apple Silicon)](#mlx-quick-start)
- [Pretraining](#pretraining)
  - [PyTorch Pretraining](#pytorch-pretraining)
  - [MLX Pretraining](#mlx-pretraining)
- [Inference](#inference)
- [Benchmarks](#benchmarks)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [MLX Optimizations](#mlx-optimizations)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Paper References

> **Original Paper**: Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv preprint arXiv:2501.00663

> **Analysis Paper**: Di Nepi, G., Siciliano, F., & Silvestri, F. (2025). *Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model*. arXiv preprint arXiv:2510.09551

---

## Features

### Core Features

| Feature | PyTorch | MLX |
|---------|---------|-----|
| MAC (Memory as Context) | ✅ | ✅ |
| MAG (Memory as Gate) | ✅ | ✅ |
| MAL (Memory as Layer) | ✅ | ✅ |
| LMM (Memory Only) | ✅ | ✅ |
| Deep Memory (L_M >= 1) | ✅ | ✅ |
| Data-dependent Gating | ✅ | ✅ |
| RoPE (Rotary Embeddings) | ✅ | ✅ |
| 1D Depthwise Convolution | ✅ | ✅ |
| Mixed Precision Training | ✅ bf16/fp16 | ✅ fp16/bf16 |
| Gradient Accumulation | ✅ | ✅ |
| Streaming Datasets | ✅ | ✅ |
| W&B Logging | ✅ | ✅ |

### Backend-Specific Features

| Feature | PyTorch | MLX |
|---------|---------|-----|
| Flash Attention 2 | ✅ (CUDA) | N/A |
| Triton Kernels | ✅ (CUDA) | N/A |
| Metal Kernels | N/A | ✅ |
| MPS Backend | ✅ | N/A |
| Unified Memory | N/A | ✅ |
| Numerical Parity | Reference | ✅ < 1e-4 |

### Test Coverage

- **105 unit tests** covering all modules
- **Numerical parity tests** ensuring MLX matches PyTorch outputs
- **Integration tests** for all model variants

---

## Architecture Overview

<p align="center">
<img src="assets/figures/fig1_memory_training.png" alt="Neural Memory Training" width="600"/>
</p>
<p align="center"><em>Figure 1: Neural memory training with efficient parallelization via matmul operations (from paper)</em></p>

### Memory Perspective

Titans are designed around a **memory perspective** inspired by human cognition (Section 1 of paper):

| Memory Type | Module | Behavior at Test Time | Characteristics |
|-------------|--------|----------------------|-----------------|
| **Short-term** | Attention (limited window) | In-context learning (fixed weights) | Precise, limited capacity |
| **Long-term** | Neural Memory (LMM) | **Still learning** (weight updates via gradient descent) | Fading, unlimited capacity |
| **Persistent** | Learnable tokens | Fixed (task knowledge) | Stable, task-specific |

### Architecture Variants

#### Quick Comparison

| Aspect | MAC | MAG | MAL | LMM |
|--------|-----|-----|-----|-----|
| **Architecture** | Memory → Attention → Memory | Attention ⊗ Memory | Memory → Attention | Memory only |
| **Attention Type** | Segmented (full causal per chunk) | Sliding Window | Sliding Window | None |
| **Memory-Attention** | Bidirectional | Parallel (gating) | Sequential | N/A |
| **Chunking Required** | Yes | No | No | No |
| **Long-context** | ⭐⭐⭐ Best | ⭐⭐ Good | ⭐ Baseline | ⭐⭐ Good |
| **Training Speed** | Medium | Fast | Fastest | Fast |

#### When to Use Each Variant

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Needle-in-haystack retrieval | **MAC** | Attention decides when to query long-term memory |
| Long document QA (>100K tokens) | **MAC** | Best BABILong benchmark results (97.95%) |
| Language modeling (perplexity) | **MAG** | Slightly better perplexity than MAC |
| Real-time / streaming inference | **MAG** | No chunking, constant memory footprint |
| Maximum training throughput | **MAL** | Leverages FlashAttention optimizations |
| Existing hybrid model replacement | **MAL** | Same architecture as Griffin/Samba |
| Pure sequence modeling | **LMM** | Tests memory capability alone |

#### MAC: Memory as Context (Section 4.1)

<p align="center">
<img src="assets/figures/fig2_mac.png" alt="MAC Architecture" width="700"/>
</p>
<p align="center"><em>Figure 2: MAC (Memory as Context) - Bidirectional interaction between memory and attention</em></p>

```
h_t = M*_{t-1}(q_t)                              # Eq. 21: Retrieve from memory
S̃^(t) = [persistent] || h_t || x                # Eq. 22: Concatenate
y_t = Attn(S̃^(t))                               # Eq. 23: Segmented attention
M_t = M_{t-1}(y_t)                               # Eq. 24: Update memory
o_t = y_t ⊗ M*_t(y_t)                            # Eq. 25: Output gating
```

**Advantages**: Best long-context performance, bidirectional memory-attention interaction
**Disadvantages**: Requires chunking, slightly slower training

#### MAG: Memory as Gate (Section 4.2)

<p align="center">
<img src="assets/figures/fig4_mag_mal.png" alt="MAG and MAL Architecture" width="700"/>
</p>
<p align="center"><em>Figure 4-5: MAG (Memory as Gate) and MAL (Memory as Layer) architectures</em></p>

```
x̃ = [persistent] || x                           # Eq. 26: Add persistent tokens
y = SW-Attn*(x̃)                                  # Eq. 27: Sliding window attention
o = y ⊗ M(x̃)                                     # Eq. 28: Element-wise gating
```

**Advantages**: No chunking, best perplexity, good balance
**Disadvantages**: Memory and attention don't directly communicate

#### MAL: Memory as Layer (Section 4.3)

```
x̃ = [persistent] || x                           # Eq. 29: Add persistent tokens
y = M(x̃)                                         # Eq. 30: Memory layer
o = SW-Attn(y)                                   # Eq. 31: Attention on memory output
```

**Advantages**: Fastest training, simplest architecture
**Disadvantages**: Weaker long-context performance

### Neural Long-term Memory

<p align="center">
<img src="assets/figures/fig3a_lstm_forget.png" alt="LSTM-inspired Gating" width="400"/>
<img src="assets/figures/fig3b_lstm_update.png" alt="Memory Update" width="400"/>
</p>
<p align="center"><em>Figure 3: LSTM-inspired gating mechanism for memory forgetting (left) and update (right)</em></p>

#### Core Equations (Section 3.1)

**Associative Memory Loss** (Eq. 12):
```
ℓ(M; x_t) = ||M(k_t) - v_t||²
```

**Memory Update with Forgetting** (Eq. 13):
```
M_t = (1 - α_t) · M_{t-1} + S_t
```

**Surprise with Momentum** (Eq. 14):
```
S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; x_t)
      \_________/   \____________________/
      Past Surprise   Momentary Surprise
```

Where:
- `α_t` ∈ [0,1]: Forgetting/decay factor (data-dependent)
- `η_t` ∈ [0,1): Surprise decay / momentum coefficient (data-dependent)
- `θ_t` > 0: Learning rate for momentary surprise (data-dependent)

#### Key Innovations

1. **Momentum-based surprise**: Unlike DeltaNet/TTT which use momentary surprise only
2. **Forgetting mechanism**: Weight decay for memory management on long sequences
3. **Deep memory**: MLP with L_M >= 2 layers for more expressive power
4. **Data-dependent gates**: α, η, θ are functions of input, not fixed hyperparameters

---

## Installation

### Basic Installation (PyTorch)

```bash
git clone https://github.com/yourusername/Google-Titans-replication.git
cd Google-Titans-replication
uv sync
```

### With Training Dependencies

```bash
uv sync --extra train
```

### With All Extras (Development)

```bash
uv sync --all-extras
```

### MLX Requirements

MLX requires macOS 13.5+ and Apple Silicon (M1/M2/M3/M4):

```bash
# MLX is included in default dependencies
uv sync
```

---

## Quick Start

### PyTorch Quick Start

```python
import torch
from titans import TitansConfig, TitansMAC, TitansMAG, TitansMAL

# Configuration
config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    vocab_size=32000,
    chunk_size=512,           # For MAC
    window_size=512,          # For MAG/MAL
    num_persistent_tokens=16,
    num_memory_layers=2,      # Deep memory
)

# Create model
model = TitansMAC(config)  # or TitansMAG, TitansMAL

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)

# Continue with states for next segment
input_ids_next = torch.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

### MLX Quick Start

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansMAC, TitansMAG, TitansMAL

# Configuration (same as PyTorch)
config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    vocab_size=32000,
    chunk_size=512,
    window_size=512,
    num_persistent_tokens=16,
    num_memory_layers=2,
)

# Create model
model = TitansMAC(config)
mx.eval(model.parameters())  # Evaluate parameters

# Forward pass
input_ids = mx.random.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
mx.eval(logits)  # Force evaluation

# Continue with states
input_ids_next = mx.random.randint(0, config.vocab_size, (2, 512))
logits_next, states = model(input_ids_next, states=states)
```

### Standalone Neural Memory

```python
# PyTorch
from titans import TitansConfig, NeuralLongTermMemory
import torch

config = TitansConfig(dim=512, num_memory_layers=2)
memory = NeuralLongTermMemory(config)

x = torch.randn(2, 100, 512)
output, state = memory(x)
output2, state2 = memory(x, state=state)  # Continue with state

# MLX
from titans_mlx import TitansConfig, NeuralLongTermMemory
import mlx.core as mx

config = TitansConfig(dim=512, num_memory_layers=2)
memory = NeuralLongTermMemory(config)
mx.eval(memory.parameters())

x = mx.random.normal((2, 100, 512))
output, state = memory(x)
mx.eval(output)
```

---

## Pretraining

### PyTorch Pretraining

#### Pre-tokenization (Recommended for Large Datasets)

Pre-tokenize your dataset once to eliminate tokenization bottleneck during training:

```bash
# Pre-tokenize FineWeb-Edu
uv run python scripts/pretokenize.py \
    --dataset HuggingFaceFW/fineweb-edu \
    --subset sample-10BT \
    --tokenizer NousResearch/Llama-2-7b-hf \
    --output data/fineweb-tokenized \
    --seq-len 4096 \
    --num-proc 8

# Then train with pre-tokenized data (much faster)
uv run python scripts/pretrain.py --model mac \
    --local-dataset data/fineweb-tokenized \
    --tokenizer NousResearch/Llama-2-7b-hf \
    --dim 512 --num-layers 12
```

#### Training Commands

```bash
# Demo with synthetic data (quick test)
uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

# Train with pre-tokenized local dataset (fastest)
uv run python scripts/pretrain.py --model mac \
    --local-dataset data/fineweb-tokenized \
    --tokenizer NousResearch/Llama-2-7b-hf \
    --dim 512 --num-layers 12

# Train with streaming dataset (no pre-tokenization)
uv run python scripts/pretrain.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 512 --num-layers 12

# Full training with paper hyperparameters (340M params)
uv run python scripts/pretrain.py --model mac \
    --local-dataset data/fineweb-tokenized \
    --tokenizer NousResearch/Llama-2-7b-hf \
    --dim 1024 --num-layers 24 --num-heads 16 \
    --batch-size 8 --gradient-accumulation-steps 32 \
    --lr 4e-4 --weight-decay 0.1 \
    --mixed-precision bf16 --wandb

# Train with local text file
uv run python scripts/pretrain.py --model mag \
    --data path/to/corpus.txt \
    --tokenizer gpt2

# Resume from checkpoint
uv run python scripts/pretrain.py --model mac \
    --resume checkpoints/latest.pt
```

#### PyTorch Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `mac` | Model variant: mac, mag, mal, lmm |
| `--dim` | `512` | Model dimension |
| `--num-heads` | `8` | Attention heads |
| `--num-layers` | `12` | Number of layers |
| `--local-dataset` | - | Pre-tokenized local dataset (Arrow format) |
| `--dataset` | - | HuggingFace dataset name (streaming) |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer |
| `--batch-size` | `4` | Per-device batch size |
| `--gradient-accumulation-steps` | `32` | Gradient accumulation |
| `--lr` | `4e-4` | Learning rate |
| `--weight-decay` | `0.1` | Weight decay |
| `--mixed-precision` | `bf16` | none, fp16, bf16 |
| `--use-torch-compile` | `False` | Enable torch.compile (PyTorch 2.0+) |
| `--wandb` | `False` | Enable W&B logging |

#### CUDA Optimizations

The training scripts include automatic CUDA optimizations:

- **TF32 Precision**: Enabled by default on Ampere+ GPUs for faster matmul
- **cuDNN Benchmark**: Auto-tunes convolution algorithms
- **Fused AdamW**: Optimizer runs entirely on GPU
- **BFloat16 Native**: Model initialized in bf16 (no autocast overhead)
- **Non-blocking Transfers**: CPU→GPU transfers overlap with computation

### Distributed Training (Multi-GPU)

```bash
# Multi-GPU with DDP (auto-detects GPUs)
uv run accelerate launch scripts/pretrain_distributed.py \
    --model mac --dim 512 \
    --local-dataset data/fineweb-tokenized

# Multi-GPU with custom config
uv run accelerate launch --config_file configs/fsdp_config.yaml \
    scripts/pretrain_distributed.py --model mac --dim 1024
```

### MLX Pretraining

```bash
# Demo with synthetic data
uv run python scripts/pretrain_mlx.py --model mac --dim 256 --epochs 10

# Train with FineWeb-Edu
uv run python scripts/pretrain_mlx.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 512 --num-layers 12

# Full training
uv run python scripts/pretrain_mlx.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 1024 --num-layers 24 --num-heads 16 \
    --batch-size 4 --gradient-accumulation-steps 32 \
    --dtype float16 --wandb

# Train with local text
uv run python scripts/pretrain_mlx.py --model mag \
    --data path/to/corpus.txt

# Resume from checkpoint
uv run python scripts/pretrain_mlx.py --model mac \
    --resume checkpoints_mlx/latest.safetensors
```

#### MLX Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `mac` | Model variant: mac, mag, mal, lmm |
| `--dim` | `512` | Model dimension |
| `--num-heads` | `8` | Attention heads |
| `--num-layers` | `12` | Number of layers |
| `--dataset` | - | HuggingFace dataset name |
| `--tokenizer` | `gpt2` | HuggingFace tokenizer |
| `--batch-size` | `4` | Batch size |
| `--gradient-accumulation-steps` | `32` | Gradient accumulation |
| `--lr` | `4e-4` | Learning rate |
| `--dtype` | `float16` | float32, float16, bfloat16 |
| `--wandb` | `False` | Enable W&B logging |

---

## Inference

### PyTorch Inference

```bash
# Generate text
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --max-tokens 100

# Interactive mode
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --interactive

# With sampling parameters
uv run python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The meaning of life is" \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-tokens 200
```

### MLX Inference

```bash
# Generate text
uv run python scripts/inference_mlx.py \
    --checkpoint checkpoints_mlx/best_model.safetensors \
    --prompt "Once upon a time" \
    --max-tokens 100

# Interactive mode
uv run python scripts/inference_mlx.py \
    --checkpoint checkpoints_mlx/best_model.safetensors \
    --interactive
```

---

## Benchmarks

### Model Quality (from Paper Table 1 & 5)

**Language Modeling (340M params, 15B tokens)**:

| Model | Wiki ppl ↓ | Avg Accuracy ↑ |
|-------|------------|----------------|
| MAC | 25.43 | 47.36 |
| MAG | **25.07** | **47.54** |
| MAL | 24.69 | 46.55 |
| LMM | 26.18 | 46.17 |

**Long Context (BABILong benchmark)**:

| Model | Accuracy ↑ |
|-------|------------|
| MAC | **97.95** |
| MAG | 96.70 |
| MAL | 96.91 |
| LMM | 92.68 |

### Inference Speed (Apple M3 Max)

Configuration: batch=4, seq_len=256, dim=256, 4 layers

| Model | MLX (ms) | PyTorch MPS (ms) | PyTorch CPU (ms) | MLX Speedup vs MPS |
|-------|----------|------------------|------------------|-------------------|
| MAC | **19.89** | 24.94 | 90.30 | **1.25x** |
| MAG | **9.72** | 16.66 | 43.45 | **1.71x** |
| MAL | **9.75** | 16.89 | 45.05 | **1.73x** |
| LMM | **7.11** | 11.88 | 28.73 | **1.67x** |

**All MLX implementations are faster than PyTorch MPS on Apple Silicon.**

### Numerical Parity

MLX and PyTorch implementations produce identical outputs:

| Model | Max Difference |
|-------|---------------|
| Memory | < 1e-5 |
| LMM | < 1e-4 |
| MAG | < 1e-4 |
| MAC | < 1e-4 |

---

## Configuration Reference

### TitansConfig Parameters

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| **Model Architecture** |
| `dim` | 512 | Model dimension (d_in) | - |
| `num_heads` | 8 | Number of attention heads | - |
| `num_layers` | 12 | Number of Titans blocks | Stackable |
| `vocab_size` | 32000 | Vocabulary size | - |
| `max_seq_len` | 8192 | Maximum sequence length | - |
| **Memory** |
| `num_memory_layers` | 2 | Memory MLP depth (L_M >= 1) | Section 3.1 |
| `memory_hidden_mult` | 4.0 | Memory hidden dim multiplier | - |
| `memory_lr` | 0.1 | Learning rate θ_t (scaled by gate) | Eq. 14 |
| `memory_momentum` | 0.9 | Momentum η_t (scaled by gate) | Eq. 14 |
| `memory_decay` | 0.01 | Forgetting α_t (scaled by gate) | Eq. 13 |
| **Attention** |
| `num_persistent_tokens` | 16 | Persistent memory tokens (N_p) | Eq. 19 |
| `chunk_size` | 512 | Segment size for MAC | Section 4.1 |
| `window_size` | 512 | Sliding window for MAG/MAL | Section 4.2-4.3 |
| **Architecture Options** |
| `use_conv` | True | 1D depthwise convolution | Section 4.4 |
| `conv_kernel_size` | 4 | Convolution kernel size | Section 4.4 |
| `use_rope` | True | Rotary Position Embeddings | - |
| `activation` | "silu" | Activation function | Section 4.4 |
| `dropout` | 0.0 | Dropout rate | - |
| **FFN** |
| `ffn_mult` | 4.0 | FFN hidden dim multiplier | - |
| `init_std` | 0.02 | Weight initialization std | - |

---

## API Reference

### PyTorch API

```python
from titans import (
    # Configuration
    TitansConfig,

    # Models
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansLMM,

    # Components
    NeuralLongTermMemory,
    MemoryState,
    SlidingWindowAttention,
    SegmentedAttention,
    PersistentMemory,
)

# Model forward signature
logits, states = model(input_ids, states=None)
# input_ids: (batch, seq_len) - Token IDs
# states: Optional list of MemoryState
# Returns: logits (batch, seq_len, vocab_size), new states

# Memory forward signature
output, state = memory(x, state=None, return_state=True)
# x: (batch, seq_len, dim)
# state: Optional MemoryState
# Returns: output (batch, seq_len, dim), new state
```

### MLX API

```python
from titans_mlx import (
    # Configuration
    TitansConfig,

    # Models
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansLMM,

    # Components
    NeuralLongTermMemory,
    MemoryState,
    SlidingWindowAttention,
    SegmentedAttention,
    PersistentMemory,

    # Optimizations
    compile_model,      # Note: Limited support
    compile_function,
    get_device_info,

    # Metal Kernels (benchmarking only)
    metal_silu_gate,
    metal_memory_update,
    metal_rope,
)
```

---

## MLX Optimizations

### Gradient Computation

The MLX implementation uses **analytical gradients** instead of `mx.grad` for the memory update:

```python
# Efficient gradient via matmul (avoids huge intermediate tensors)
# Instead of: expand_dims + outer product + sum
# We use: reshape + matmul

delta_flat = delta.reshape(batch_seq, -1)  # (B*S, D_out)
act_flat = act.reshape(batch_seq, -1)      # (B*S, D_in)
grad_w = delta_flat.T @ act_flat           # (D_out, D_in)
```

This optimization provides **5x speedup** for MAC.

### Why Not mx.compile?

`mx.compile` cannot compile full Titans models because:

1. **MemoryState**: Dataclasses are not supported
2. **Dynamic loops**: Python for-loops for chunk processing
3. **Mutable state**: Memory state updates

Individual components (FFN, attention) can be compiled for marginal gains.

### Metal Kernels

Custom Metal kernels are available but **not faster** than native MLX for typical tensor sizes:

| Operation | Metal Kernel | Native MLX | Verdict |
|-----------|--------------|------------|---------|
| SiLU Gate | 0.44ms | 0.26ms | Native faster |
| Memory Update | 0.20ms | 0.23ms | ~Equal |
| RoPE | 0.23ms | 0.23ms | ~Equal |

MLX already optimizes well for Apple Silicon. Use native operations.

### Recommended Practices

```python
import mlx.core as mx
from titans_mlx import TitansConfig, TitansMAC

# 1. Use float16 for training (default)
# Saves memory, marginal speed difference on Apple Silicon

# 2. Evaluate parameters after creation
model = TitansMAC(config)
mx.eval(model.parameters())

# 3. Evaluate outputs when needed
logits, states = model(input_ids)
mx.eval(logits)  # Force computation

# 4. Use larger batches to amortize overhead
# batch_size=4 or higher recommended

# 5. Disable convolution if dimensions mismatch
config = TitansConfig(..., use_conv=False)
```

---

## Troubleshooting

### Common Issues

#### PyTorch

**Issue**: Out of memory on GPU
```bash
# Reduce batch size or use gradient accumulation
--batch-size 2 --gradient-accumulation-steps 64
```

**Issue**: NaN loss during training
```bash
# Use bf16 instead of fp16, or reduce learning rate
--mixed-precision bf16 --lr 2e-4
```

#### MLX

**Issue**: `ValueError: conv1d groups` error
```python
# Disable convolution
config = TitansConfig(..., use_conv=False)
```

**Issue**: Slow first iteration
```python
# Normal - MLX compiles on first call
# Subsequent iterations will be faster
```

**Issue**: Memory not releasing
```python
# Force garbage collection
import gc
gc.collect()
mx.metal.clear_cache()  # If available
```

### Numerical Differences

Small numerical differences (< 1e-4) between PyTorch and MLX are expected due to:
- Different floating-point implementations
- Different reduction orders
- Platform-specific optimizations

For exact reproducibility, use the same backend.

---

## Development

### Project Structure

```
titans-pytorch/
├── src/
│   ├── titans/                 # PyTorch implementation
│   │   ├── __init__.py
│   │   ├── config.py           # TitansConfig
│   │   ├── memory.py           # Neural Long-term Memory
│   │   ├── attention.py        # Attention modules
│   │   ├── persistent.py       # Persistent Memory
│   │   ├── models.py           # MAC, MAG, MAL, LMM
│   │   └── triton_kernels.py   # Triton optimizations
│   │
│   └── titans_mlx/             # MLX implementation
│       ├── __init__.py
│       ├── config.py
│       ├── memory.py
│       ├── attention.py
│       ├── persistent.py
│       ├── models.py
│       ├── optimizations.py    # MLX optimizations
│       └── metal_kernels.py    # Metal kernels
│
├── scripts/
│   ├── pretrain.py             # PyTorch training (optimized)
│   ├── pretrain_distributed.py # Multi-GPU training (Accelerate)
│   ├── pretrain_mlx.py         # MLX training
│   ├── pretokenize.py          # Dataset pre-tokenization
│   ├── inference.py            # PyTorch inference
│   └── inference_mlx.py        # MLX inference
│
├── tests/
│   ├── test_memory.py
│   ├── test_attention.py
│   ├── test_models.py
│   ├── test_persistent.py
│   └── test_numerical_parity.py  # MLX vs PyTorch
│
├── examples/
│   ├── basic_usage.py
│   └── long_sequence.py
│
└── pyproject.toml
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_numerical_parity.py -v

# With coverage
uv run pytest tests/ --cov=titans --cov=titans_mlx --cov-report=term-missing
```

### Linting

```bash
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

---

## Citation

```bibtex
@article{behrouz2024titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2024}
}

@article{dinepi2025titans,
  title={Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model},
  author={Di Nepi, Gavriel and Siciliano, Federico and Silvestri, Fabrizio},
  journal={arXiv preprint arXiv:2510.09551},
  year={2025}
}
```

---

## License

Apache License 2.0

Copyright (c) 2024 Delanoe Pirard / Aedelon
