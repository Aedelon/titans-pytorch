#!/usr/bin/env python3
"""Benchmark CUDA optimizations for Titans pretraining."""

import time
import torch
import torch.nn as nn
import sys

sys.path.insert(0, "src")

from titans.config import TitansConfig
from titans.models import TitansMAC


def benchmark_config(batch_size: int, seq_len: int, num_iters: int = 10):
    """Benchmark a specific configuration."""
    device = torch.device("cuda")

    config = TitansConfig(
        dim=320,
        num_heads=8,
        num_layers=6,
        chunk_size=512,
        vocab_size=32000,
        max_seq_len=seq_len,
    )

    model = TitansMAC(config).to(device)
    model.train()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    tokens_per_iter = batch_size * seq_len

    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, 32000), labels.view(-1)
            )
            loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, 32000), labels.view(-1)
            )
            loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    throughput = tokens_per_iter * num_iters / elapsed

    return {
        "params_m": num_params / 1e6,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_iter": tokens_per_iter,
        "avg_time_ms": elapsed / num_iters * 1000,
        "throughput_tokens_sec": throughput,
        "peak_memory_gb": peak_memory,
    }


def main():
    print("=" * 70)
    print("Titans CUDA Optimization Benchmark")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA not available")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    configs = [
        # (batch_size, seq_len)
        (1, 2048),
        (2, 2048),
        (4, 2048),
        (1, 4096),
        (2, 4096),
        (4, 4096),
    ]

    print(f"{'Config':<20} {'Time (ms)':<12} {'Tokens/sec':<15} {'Memory (GB)':<12}")
    print("-" * 70)

    for batch_size, seq_len in configs:
        try:
            torch.cuda.empty_cache()
            result = benchmark_config(batch_size, seq_len)
            print(
                f"bs={batch_size}, seq={seq_len:<5} "
                f"{result['avg_time_ms']:<12.1f} "
                f"{result['throughput_tokens_sec']:<15,.0f} "
                f"{result['peak_memory_gb']:<12.2f}"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"bs={batch_size}, seq={seq_len:<5} OOM")
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("Optimizations Applied:")
    print("  - PyTorch SDPA (Scaled Dot-Product Attention)")
    print("  - No CPU-GPU sync during training (.item() deferred)")
    print("  - Efficient batched memory updates")
    print("  - BFloat16 mixed precision")
    print("=" * 70)


if __name__ == "__main__":
    main()
