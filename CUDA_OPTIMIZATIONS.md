# CUDA Optimizations for Titans Pretraining

This document describes the CUDA optimizations implemented for maximum GPU utilization during pretraining.

## Performance Results

Benchmarked on NVIDIA L4 (22GB VRAM):

| Configuration | Before | After | Speedup |
|---------------|--------|-------|---------|
| bs=4, seq=4096 | 725ms, 22.5K tok/s | **384ms, 42.6K tok/s** | **1.9x** |
| bs=4, seq=2048 | 379ms, 21.6K tok/s | **190ms, 43.1K tok/s** | **2.0x** |
| bs=2, seq=4096 | 700ms, 11.7K tok/s | **359ms, 22.8K tok/s** | **1.9x** |

## Optimizations Applied

### 1. CPU-GPU Synchronization Elimination

**Problem**: `loss.item()` was called on every training step, causing a CPU-GPU sync.

**Solution** (`scripts/pretrain.py`):
- Accumulate loss as tensors without `.item()`
- Only sync at logging intervals (every `log_every` steps)

```python
# Before (sync every step):
return loss, {"loss": loss.item()}

# After (sync only at logging):
return loss, {"loss_tensor": loss.detach()}
```

### 2. PyTorch SDPA for Attention

**Problem**: Manual attention computation with separate matmul, softmax, dropout ops.

**Solution** (`src/titans/attention.py`):
- Use `F.scaled_dot_product_attention()` which uses Flash Attention backend
- Reduces memory usage by 47% and improves speed by 27%

```python
# Before:
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
attn_weights = F.softmax(attn_scores.masked_fill(~mask, -inf), dim=-1)
output = torch.matmul(attn_weights, v)

# After:
output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
```

### 3. Triton Kernels for RMSNorm

**Problem**: RMSNorm uses multiple elementwise operations that don't fuse well.

**Solution** (`src/titans/triton_kernels.py`):
- Custom Triton kernel `rms_norm_kernel` fuses the entire operation
- Automatically used in `RMSNorm` when Triton is available

### 4. Triton Fused SiLU*Mul Kernel

**Problem**: FFN gating uses `silu(gate) * up` which is two separate kernels.

**Solution** (`src/titans/triton_kernels.py`):
- Custom `fused_silu_mul_kernel` combines both operations
- Reduces kernel launch overhead and memory traffic

### 5. Optimized Memory Gradient Computation

**Problem**: Memory module clones weights for gradient computation.

**Solution** (`src/titans/cuda_optimizations.py`):
- Compute gradients in-place without cloning
- Use analytical gradients for single-layer memory
- Efficient autograd for multi-layer memory

### 6. Efficient Batched Memory Updates

**Problem**: Memory update equations used individual tensor operations.

**Solution** (`src/titans/cuda_optimizations.py`):
- Vectorized memory updates without `.item()` sync
- All operations stay on GPU tensors

```python
# Efficient tensor operations, no sync
one_minus_alpha = 1.0 - alpha
new_m = eta * m - theta * g
new_w = one_minus_alpha * w + new_m
```

## Usage

Run optimized pretraining:

```bash
uv run python scripts/pretrain.py \
    --model mac \
    --dim 320 \
    --num-layers 6 \
    --num-heads 8 \
    --chunk-size 512 \
    --vocab-size 32000 \
    --seq-len 4096 \
    --dataset HuggingFaceFW/fineweb-edu \
    --dataset-subset sample-10BT \
    --tokenizer NousResearch/Llama-2-7b-hf \
    --batch-size 4 \
    --gradient-accumulation-steps 32 \
    --lr 4e-4 \
    --weight-decay 0.1 \
    --mixed-precision bf16 \
    --wandb \
    --wandb-project titans-mac \
    --max-steps 10000
```

## Benchmark Script

Run the optimization benchmark:

```bash
uv run python scripts/benchmark_optimizations.py
```

## Requirements

- PyTorch 2.0+ (for SDPA)
- Triton (for custom kernels)
- NVIDIA GPU with compute capability >= 7.0

## Known Limitations

1. **Gradient checkpointing incompatible**: The Titans memory module computes gradients during the forward pass, which conflicts with gradient checkpointing.

2. **Triton kernels require CUDA**: Automatic fallback to PyTorch when Triton is unavailable.

3. **SDPA mask limitation**: SlidingWindowAttention uses explicit mask which may prevent Flash Attention backend.
