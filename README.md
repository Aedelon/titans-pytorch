# Titans: Learning to Memorize at Test Time

A PyTorch implementation of the Titans architecture from Google Research.

Titans introduce a **Neural Long-term Memory (LMM)** module that learns to memorize historical context at test time using gradient descent with momentum and weight decay. This enables attention mechanisms to focus on local context while utilizing long-range information through neural memory.

## Paper References

> **Original Paper**: Behrouz, A., Zhong, P., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv preprint arXiv:2501.00663

> **Analysis Paper**: Di Nepi, G., Siciliano, F., & Silvestri, F. (2025). *Titans Revisited: A Lightweight Reimplementation and Critical Analysis of a Test-Time Memory Model*. arXiv preprint arXiv:2510.09551

## Features

- Complete implementation of all three Titans variants (MAC, MAG, MAL)
- Standalone Neural Long-term Memory module (LMM)
- Deep memory support (L_M >= 1 layers)
- Data-dependent gating for learning rate, momentum, and decay
- 1D depthwise convolution following Mamba2/GatedDeltaNet
- Rotary Position Embeddings (RoPE)
- 86 unit tests with comprehensive coverage

## Memory Perspective

Titans are designed around a **memory perspective** inspired by human cognition (Section 1 of paper):

| Memory Type | Module | Behavior at Test Time |
|-------------|--------|----------------------|
| **Short-term** | Attention (limited window) | In-context learning (fixed weights) |
| **Long-term** | Neural Memory (LMM) | **Still learning** (weight updates via gradient descent) |
| **Persistent** | Learnable tokens | Fixed (task knowledge) |

This separation allows each component to operate independently while working together.

## Architecture Variants

### Quick Comparison

| Aspect | MAC | MAG | MAL | LMM |
|--------|-----|-----|-----|-----|
| **Architecture** | Memory → Attention → Memory | Attention ⊗ Memory | Memory → Attention | Memory only |
| **Attention Type** | Segmented (full causal per chunk) | Sliding Window | Sliding Window | None |
| **Memory-Attention Interaction** | Bidirectional | Parallel (gating) | Sequential | N/A |
| **Chunking Required** | Yes | No | No | No |
| **Long-context Performance** | ⭐⭐⭐ Best | ⭐⭐ Good | ⭐ Baseline | ⭐⭐ Good |
| **Training Speed** | Medium | Fast | Fastest | Fast |
| **Complexity** | High | Medium | Low | Low |

### When to Use Each Variant

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Needle-in-haystack retrieval** | MAC | Attention decides when to query long-term memory |
| **Long document QA (>100K tokens)** | MAC | Best BABILong benchmark results (97.95% accuracy) |
| **Language modeling (perplexity)** | MAG | Slightly better perplexity than MAC |
| **Real-time / streaming inference** | MAG | No chunking, constant memory footprint |
| **Maximum training throughput** | MAL | Leverages FlashAttention optimizations |
| **Existing hybrid model replacement** | MAL | Same architecture as Griffin/Samba |
| **Pure sequence modeling (no attention)** | LMM | Tests memory capability alone |

### MAC: Memory as Context (Section 4.1)

**How it works**: The input queries the memory, then attention sees both the retrieved memory and the current context. After attention, the memory is updated with the attention output.

```
h_t = M*_{t-1}(q_t)                              # Eq. 21: Retrieve from memory
S̃^(t) = [persistent] || h_t || x                # Eq. 22: Concatenate
y_t = Attn(S̃^(t))                               # Eq. 23: Segmented attention
M_t = M_{t-1}(y_t)                               # Eq. 24: Update memory
o_t = y_t ⊗ M*_t(y_t)                            # Eq. 25: Output gating
```

**Key insight**: Attention acts as a "gatekeeper" - it decides:
1. Whether long-term memory is relevant for the current context
2. What information from the current context should be stored in memory

**Advantages**:
- Best performance on long-context reasoning (BABILong: 97.95%)
- Memory stores only attention-filtered useful information
- Bidirectional memory-attention interaction

**Disadvantages**:
- Requires chunking (adds complexity)
- Slightly slower training than MAG/MAL

**Best for**: Document QA, multi-hop reasoning, needle-in-haystack tasks

### MAG: Memory as Gate (Section 4.2)

**How it works**: Attention and memory process the input in parallel. Their outputs are combined via element-wise multiplication (gating).

```
x̃ = [persistent] || x                           # Eq. 26: Add persistent tokens
y = SW-Attn*(x̃)                                  # Eq. 27: Sliding window attention
o = y ⊗ M(x̃)                                     # Eq. 28: Element-wise gating
```

**Key insight**:
- Sliding window attention captures **precise local dependencies**
- Neural memory provides **fading long-range context**
- The gating mechanism lets the model decide how much to rely on each

**Advantages**:
- No chunking required (simpler implementation)
- Best perplexity on language modeling benchmarks
- Good balance between performance and speed

**Disadvantages**:
- Slightly weaker on very long-context tasks than MAC
- Memory and attention don't directly communicate

**Best for**: Language modeling, general-purpose use, streaming applications

### MAL: Memory as Layer (Section 4.3)

**How it works**: Memory processes the input first (as a preprocessing layer), then attention operates on the memory output. This is the "standard" hybrid architecture used by Griffin, Samba, etc.

```
x̃ = [persistent] || x                           # Eq. 29: Add persistent tokens
y = M(x̃)                                         # Eq. 30: Memory layer
o = SW-Attn(y)                                   # Eq. 31: Attention on memory output
```

**Key insight**: Memory compresses historical context before attention sees it. This is computationally efficient but limits the interaction between memory and attention.

**Advantages**:
- Fastest training (benefits from FlashAttention)
- Simplest architecture (drop-in replacement for existing hybrids)
- Well-understood design pattern

**Disadvantages**:
- Weaker long-context performance than MAC/MAG
- Sequential design limits expressivity
- Memory can't benefit from attention's filtering

**Best for**: When you need maximum training speed, or replacing existing Griffin/Samba models

### LMM: Memory Only

**How it works**: Pure neural memory without attention. Tests the memory module's standalone capability.

**Advantages**:
- Simplest architecture
- Useful for understanding memory behavior
- Can still achieve good results (46.17 avg on benchmarks)

**Best for**: Research, ablation studies, memory-only applications

### Performance Summary (from Paper Table 1 & 5)

**Language Modeling (340M params, 15B tokens)**:
| Model | Wiki ppl ↓ | Avg Accuracy ↑ |
|-------|------------|----------------|
| MAC | 25.43 | 47.36 |
| MAG | 25.07 | 47.54 |
| MAL | 24.69 | 46.55 |
| LMM | 26.18 | 46.17 |

**Long Context (BABILong benchmark)**:
| Model | Accuracy ↑ |
|-------|------------|
| MAC | 97.95 |
| MAG | 96.70 |
| MAL | 96.91 |
| LMM | 92.68 |

**Recommendation**: Start with **MAG** for general use, switch to **MAC** if you need the best long-context performance.

### MLX vs PyTorch Benchmark (Apple M3 Max)

Inference speed comparison on batch=4, seq_len=256, dim=256, 4 layers:

| Model | MLX (ms) | PyTorch MPS (ms) | MLX Speedup |
|-------|----------|------------------|-------------|
| MAC | 19.98 | 24.58 | **1.23x** |
| MAG | 9.49 | 16.73 | **1.76x** |
| MAL | 9.81 | 16.91 | **1.72x** |
| LMM | 7.15 | 11.99 | **1.68x** |

All MLX implementations are faster than PyTorch MPS on Apple Silicon, with MAG showing the best speedup.

## Neural Long-term Memory

### Core Equations (Section 3.1)

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
- `k_t, v_t`: Key-value pairs from input projections

### Key Innovations

1. **Momentum-based surprise**: Unlike DeltaNet/TTT which use momentary surprise only, Titans incorporate token flow via momentum
2. **Forgetting mechanism**: Weight decay allows memory management for very long sequences
3. **Deep memory**: MLP with L_M >= 2 layers provides more expressive power than matrix-valued memory
4. **Data-dependent gates**: α, η, θ are functions of input, not fixed hyperparameters

### Architectural Details (Section 4.4)

Following modern linear recurrent models:

- **Activation**: SiLU for query, key, value projections
- **Normalization**: L2-norm for queries and keys
- **Convolution**: 1D depthwise-separable convolution after Q/K/V projections
- **Residual**: All blocks use residual connections
- **Gating**: Output gating with learnable normalization

## Installation

```bash
uv sync
```

For development:

```bash
uv sync --all-extras
```

For training with HuggingFace:

```bash
uv sync --extra train
```

## Quick Start

### Training

The training script supports HuggingFace tokenizers, streaming datasets, mixed precision, and gradient accumulation.

```bash
# Train with FineWeb-Edu (streaming dataset, recommended)
uv run python scripts/pretrain.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 512 --num-layers 12 --epochs 1

# Train with custom text file
uv run python scripts/pretrain.py --model mag \
    --data path/to/corpus.txt \
    --tokenizer gpt2 \
    --dim 256 --epochs 10

# Full training with paper hyperparameters (340M params)
uv run python scripts/pretrain.py --model mac \
    --dataset HuggingFaceFW/fineweb-edu \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --dim 1024 --num-layers 24 --num-heads 16 \
    --batch-size 8 --grad-accum 32 \
    --lr 4e-4 --weight-decay 0.1 \
    --precision bf16 --wandb

# Resume training from checkpoint
uv run python scripts/pretrain.py --model mac --resume checkpoints/latest.pt
```

**Training Features**:
- HuggingFace tokenizers (LLaMA 2, GPT-2, etc.)
- Streaming datasets (FineWeb-Edu, SlimPajama, etc.)
- Mixed precision (bf16/fp16)
- Gradient accumulation (default: 32 steps)
- Cosine annealing with warmup
- Wandb logging (optional)

### Inference

```bash
# Generate text from trained model
uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --prompt "Hello" --max-tokens 100

# Interactive mode
uv run python scripts/inference.py --checkpoint checkpoints/best_model.pt --interactive
```

### Examples

```bash
# Run basic usage examples
uv run python examples/basic_usage.py

# Run long sequence processing examples
uv run python examples/long_sequence.py
```

## Usage

### TitansMAC (Recommended for long-context)

```python
import torch
from titans import TitansConfig, TitansMAC

config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    vocab_size=32000,
    chunk_size=512,           # Segment size
    num_persistent_tokens=16, # N_p persistent tokens
    num_memory_layers=2,      # L_M >= 1 for deep memory
    memory_lr=0.1,            # θ: learning rate
    memory_momentum=0.9,      # η: momentum
    memory_decay=0.01,        # α: forgetting factor
)

model = TitansMAC(config)

# Forward pass - processes in chunks, memory persists across chunks
input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)

# Continue with states for next segment
input_ids_next = torch.randint(0, config.vocab_size, (2, 512))
logits_next, states_next = model(input_ids_next, states=states)
```

### TitansMAG (Memory as Gate)

```python
from titans import TitansConfig, TitansMAG

config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
    window_size=256,  # Sliding window size
)

model = TitansMAG(config)
input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
```

### TitansMAL (Memory as Layer)

```python
from titans import TitansConfig, TitansMAL

config = TitansConfig(
    dim=512,
    num_heads=8,
    num_layers=6,
)

model = TitansMAL(config)
input_ids = torch.randint(0, config.vocab_size, (2, 1024))
logits, states = model(input_ids)
```

### Standalone Neural Memory

```python
from titans import TitansConfig, NeuralLongTermMemory

config = TitansConfig(
    dim=512,
    num_memory_layers=2,      # Deep memory
    memory_hidden_mult=4.0,
    memory_lr=0.1,
    memory_momentum=0.9,
    memory_decay=0.01,
)

memory = NeuralLongTermMemory(config)

# Forward pass with memory update
x = torch.randn(2, 100, 512)  # (batch, seq, dim)
output, state = memory(x)

# Continue with state
x_next = torch.randn(2, 50, 512)
output2, state2 = memory(x_next, state=state)

# Retrieve without updating (inference)
queries = torch.randn(2, 10, 512)
retrieved = memory.retrieve(queries, state2)
```

## Configuration Reference

| Parameter | Default | Description | Paper Reference |
|-----------|---------|-------------|-----------------|
| `dim` | 512 | Model dimension (d_in) | - |
| `num_heads` | 8 | Attention heads | - |
| `num_layers` | 12 | Number of Titans blocks | Stackable |
| `num_memory_layers` | 2 | Memory MLP depth (L_M >= 1) | Section 3.1 |
| `memory_hidden_mult` | 4.0 | Memory hidden dim multiplier | - |
| `num_persistent_tokens` | 16 | Persistent memory tokens (N_p) | Eq. 19 |
| `chunk_size` | 512 | Segment size for MAC | Section 4.1 |
| `window_size` | 512 | Sliding window for MAG/MAL | Section 4.2-4.3 |
| `memory_lr` | 0.1 | Learning rate θ_t (scaled) | Eq. 14 |
| `memory_momentum` | 0.9 | Momentum η_t (scaled) | Eq. 14 |
| `memory_decay` | 0.01 | Forgetting α_t (scaled) | Eq. 13 |
| `use_conv` | True | 1D depthwise convolution | Section 4.4 |
| `conv_kernel_size` | 4 | Convolution kernel size | Section 4.4 |
| `use_rope` | True | Rotary Position Embeddings | - |
| `activation` | "silu" | Activation function | Section 4.4 |

## Comparison with Related Models

| Model | Memory Type | Forgetting | Momentum | Deep Memory |
|-------|-------------|------------|----------|-------------|
| Linear Transformer | Matrix-valued | No | No | No |
| Mamba/Mamba2 | Matrix-valued | Yes (gate) | No | No |
| DeltaNet | Matrix-valued | No | No | No |
| Gated DeltaNet | Matrix-valued | Yes | No | No |
| TTT | Matrix/MLP | No | No | Yes |
| **Titans (LMM)** | **Deep MLP** | **Yes** | **Yes** | **Yes** |

Titans generalize Gated DeltaNet with: (1) momentum-based update, (2) deep memory, (3) non-linear recurrence.

## Resolved Ambiguities

Based on "Titans Revisited" (arXiv:2510.09551), this implementation resolves:

| Ambiguity | Resolution |
|-----------|------------|
| Output from last chunk only? | **All chunks** - outputs concatenated |
| Dimensionality reduction after concat? | **Input positions only** - persistent/memory excluded from output |
| MAC block stackable? | **Yes** - via `num_layers` parameter |
| Attention structure details? | **Configurable** - heads, layers, RoPE via TitansConfig |

## Project Structure

```
titans/
├── src/titans/
│   ├── __init__.py         # Public API exports
│   ├── config.py           # TitansConfig dataclass
│   ├── memory.py           # Neural Long-term Memory (Eq. 12-14)
│   ├── attention.py        # SWA (MAG/MAL) + Segmented (MAC)
│   ├── persistent.py       # Persistent Memory (Eq. 19)
│   └── models.py           # MAC, MAG, MAL, LMM implementations
├── scripts/
│   ├── pretrain.py         # Training script
│   └── inference.py        # Generation script
├── examples/
│   ├── basic_usage.py      # Basic API examples
│   └── long_sequence.py    # Long sequence processing
└── tests/                  # 86 unit tests
```

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### With Coverage

```bash
uv run pytest tests/ --cov=titans --cov-report=term-missing
```

### Linting

```bash
uv run ruff check src/titans/ tests/ scripts/ examples/
uv run ruff format src/titans/ tests/ scripts/ examples/
```

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

## License

Apache License 2.0

Copyright (c) 2024 Delanoe Pirard / Aedelon
