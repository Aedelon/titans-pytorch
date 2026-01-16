#!/usr/bin/env python3
"""Quick test to verify CUDA optimizations are working."""

import time
import torch

# Set up the path
import sys
sys.path.insert(0, "src")

from titans.config import TitansConfig
from titans.models import TitansMAC


def test_forward_backward():
    """Test a forward and backward pass."""
    print("=" * 60)
    print("CUDA Optimization Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Real config (21M params target)
    config = TitansConfig(
        dim=320,
        num_heads=8,
        num_layers=6,
        chunk_size=512,
        vocab_size=32000,
        max_seq_len=4096,
    )

    model = TitansMAC(config).to(device)
    model.train()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Test input (pretraining config)
    batch_size = 4
    seq_len = 4096
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        logits, _ = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()

    # Benchmark
    print("\nBenchmarking 10 iterations...")
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for i in range(10):
        logits, _ = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )
        loss.backward()
        model.zero_grad()

        if i == 0:
            torch.cuda.synchronize()
            first_iter = time.perf_counter() - start

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\nResults:")
    print(f"  First iteration: {first_iter*1000:.1f} ms")
    print(f"  Average (10 iters): {total_time/10*1000:.1f} ms")
    print(f"  Peak GPU memory: {peak_memory:.2f} GB")
    print(f"  Throughput: {batch_size * seq_len * 10 / total_time:.0f} tokens/sec")

    # Check for sync issues
    print("\nChecking for CPU-GPU sync issues...")

    # Profile a single forward pass
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        logits, _ = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )
        loss.backward()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Check for sync events
    events = prof.key_averages()
    sync_events = [e for e in events if 'sync' in e.key.lower() or 'cudaStreamSynchronize' in e.key]

    if sync_events:
        print(f"  WARNING: Found {len(sync_events)} sync events!")
        for e in sync_events[:5]:
            print(f"    - {e.key}: {e.cpu_time_total/1000:.2f} ms")
    else:
        print("  No explicit sync events detected (good!)")

    print(f"\n  Profiled iteration: {elapsed*1000:.1f} ms")
    print("=" * 60)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_forward_backward()
