#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
GPU optimization benchmark for Titans models.

This script benchmarks:
1. Training throughput with various optimizations
2. Inference throughput and latency
3. Memory efficiency
4. GPU utilization metrics

Usage:
    # Run all benchmarks
    uv run python scripts/benchmark_gpu.py

    # Benchmark specific model
    uv run python scripts/benchmark_gpu.py --model mag --batch-size 8

    # Compare optimized vs baseline
    uv run python scripts/benchmark_gpu.py --compare
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from titans import (
    TitansConfig,
    TitansMAC,
    TitansMAG,
    TitansMAL,
    TitansLMM,
    OptimizedTrainer,
    OptimizedTrainingConfig,
    OptimizedGenerator,
    compile_model,
    benchmark_model,
    benchmark_generation,
)

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    model_type: str = "mag"
    dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    vocab_size: int = 32000
    seq_len: int = 512
    batch_size: int = 4
    num_warmup: int = 3
    num_iterations: int = 10


def create_model(config: BenchmarkConfig) -> nn.Module:
    """Create model based on config."""
    model_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.seq_len,
        window_size=config.seq_len,
    )

    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }

    return models[config.model_type](model_config)


def get_device() -> torch.device:
    """Get the best available device."""
    if CUDA_AVAILABLE:
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_gpu_info() -> None:
    """Print GPU information."""
    if not CUDA_AVAILABLE:
        print("CUDA not available")
        return

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}")
    print(f"Total memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"Multi-processor count: {props.multi_processor_count}")


def clear_memory() -> None:
    """Clear GPU memory cache."""
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_training_baseline(
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark baseline training (no optimizations)."""
    print("\n[Baseline Training]")
    clear_memory()

    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy data
    input_ids = torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
    )
    labels = torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
    )

    # Warmup
    model.train()
    for _ in range(config.num_warmup):
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        loss.backward()
        optimizer.step()

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config.num_iterations):
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()

        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            labels.view(-1),
        )
        loss.backward()
        optimizer.step()

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = (config.batch_size * config.seq_len) / avg_time

    # Memory usage
    memory_allocated = 0
    memory_reserved = 0
    if CUDA_AVAILABLE:
        memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        memory_reserved = torch.cuda.max_memory_reserved() / 1e9

    results = {
        "avg_step_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
    }

    print(f"  Average step time: {results['avg_step_ms']:.2f} ms")
    print(f"  Throughput: {results['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Memory allocated: {results['memory_allocated_gb']:.2f} GB")

    del model, optimizer
    clear_memory()

    return results


def benchmark_training_optimized(
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark optimized training."""
    print("\n[Optimized Training]")
    clear_memory()

    model = create_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Only use torch.compile on CUDA where it works best
    use_compile = CUDA_AVAILABLE and device.type == "cuda"

    training_config = OptimizedTrainingConfig(
        use_torch_compile=use_compile,
        compile_mode="reduce-overhead",
        use_amp=device.type == "cuda",
        amp_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        gradient_accumulation_steps=1,
    )

    trainer = OptimizedTrainer(
        model=model,
        optimizer=optimizer,
        config=training_config,
        device=device,
    )

    # Create dummy data
    input_ids = torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
    )
    labels = torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
    )

    # Warmup
    for _ in range(config.num_warmup):
        trainer.train_step(input_ids, labels)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    times = []
    for _ in range(config.num_iterations):
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        trainer.train_step(input_ids, labels)

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = (config.batch_size * config.seq_len) / avg_time

    # Memory usage
    memory_allocated = 0
    memory_reserved = 0
    if CUDA_AVAILABLE:
        memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        memory_reserved = torch.cuda.max_memory_reserved() / 1e9

    results = {
        "avg_step_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
    }

    print(f"  Average step time: {results['avg_step_ms']:.2f} ms")
    print(f"  Throughput: {results['tokens_per_sec']:.0f} tokens/sec")
    print(f"  Memory allocated: {results['memory_allocated_gb']:.2f} GB")

    del model, optimizer, trainer
    clear_memory()

    return results


def benchmark_inference_baseline(
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark baseline inference."""
    print("\n[Baseline Inference]")
    clear_memory()

    model = create_model(config).to(device)
    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
    )

    # Warmup
    # Note: Titans requires gradients even at inference (learns at test time)
    # so we use no_grad() but don't disable gradient computation in memory module
    with torch.no_grad():
        for _ in range(config.num_warmup):
            _ = model(input_ids)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(config.num_iterations):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids)

            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = (config.batch_size * config.seq_len) / avg_time

    results = {
        "avg_latency_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
    }

    print(f"  Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {results['tokens_per_sec']:.0f} tokens/sec")

    del model
    clear_memory()

    return results


def benchmark_inference_optimized(
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark optimized inference with torch.compile."""
    print("\n[Optimized Inference]")
    clear_memory()

    model = create_model(config).to(device)

    # Apply optimizations - only on CUDA where it works best
    if hasattr(torch, "compile") and CUDA_AVAILABLE and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"  Warning: torch.compile failed: {e}")

    model.eval()

    input_ids = torch.randint(
        0, config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
    )

    # Warmup (more iterations needed for compilation)
    # Note: Titans requires gradients even at inference (learns at test time)
    with torch.no_grad():
        for _ in range(config.num_warmup + 5):
            _ = model(input_ids)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(config.num_iterations):
            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_ids)

            if CUDA_AVAILABLE:
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = (config.batch_size * config.seq_len) / avg_time

    results = {
        "avg_latency_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
    }

    print(f"  Average latency: {results['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {results['tokens_per_sec']:.0f} tokens/sec")

    del model
    clear_memory()

    return results


def benchmark_generation_speed(
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark text generation speed."""
    print("\n[Generation Benchmark]")
    clear_memory()

    model = create_model(config).to(device)
    # Only use compile on CUDA
    use_compile = CUDA_AVAILABLE and device.type == "cuda"
    generator = OptimizedGenerator(model, device, use_compile=use_compile)

    input_length = 32
    output_length = 64

    input_ids = torch.randint(
        0, config.vocab_size,
        (1, input_length),
        device=device,
    )

    # Warmup
    for _ in range(config.num_warmup):
        _ = generator.generate(input_ids, max_new_tokens=output_length, do_sample=False)

    if CUDA_AVAILABLE:
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(config.num_iterations):
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = generator.generate(input_ids, max_new_tokens=output_length, do_sample=False)

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = output_length / avg_time
    time_per_token = (avg_time / output_length) * 1000

    results = {
        "avg_generation_ms": avg_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": time_per_token,
    }

    print(f"  Generation time ({output_length} tokens): {results['avg_generation_ms']:.2f} ms")
    print(f"  Throughput: {results['tokens_per_sec']:.1f} tokens/sec")
    print(f"  Time per token: {results['ms_per_token']:.2f} ms")

    del model, generator
    clear_memory()

    return results


def run_full_benchmark(config: BenchmarkConfig) -> dict[str, dict]:
    """Run full benchmark suite."""
    device = get_device()

    print("=" * 60)
    print("TITANS GPU OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print_gpu_info()
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_type.upper()}")
    print(f"  Dimensions: {config.dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Device: {device}")

    results = {}

    # Training benchmarks
    results["training_baseline"] = benchmark_training_baseline(config, device)
    results["training_optimized"] = benchmark_training_optimized(config, device)

    # Inference benchmarks
    results["inference_baseline"] = benchmark_inference_baseline(config, device)
    results["inference_optimized"] = benchmark_inference_optimized(config, device)

    # Generation benchmark
    results["generation"] = benchmark_generation_speed(config, device)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results["training_baseline"]["avg_step_ms"] > 0:
        training_speedup = (
            results["training_baseline"]["avg_step_ms"]
            / results["training_optimized"]["avg_step_ms"]
        )
        print(f"\nTraining Speedup: {training_speedup:.2f}x")
        print(f"  Baseline: {results['training_baseline']['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Optimized: {results['training_optimized']['tokens_per_sec']:.0f} tokens/sec")

    if results["inference_baseline"]["avg_latency_ms"] > 0:
        inference_speedup = (
            results["inference_baseline"]["avg_latency_ms"]
            / results["inference_optimized"]["avg_latency_ms"]
        )
        print(f"\nInference Speedup: {inference_speedup:.2f}x")
        print(f"  Baseline: {results['inference_baseline']['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Optimized: {results['inference_optimized']['tokens_per_sec']:.0f} tokens/sec")

    print(f"\nGeneration: {results['generation']['tokens_per_sec']:.1f} tokens/sec")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GPU optimizations for Titans"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mag",
        choices=["mac", "mag", "mal", "lmm"],
        help="Model variant to benchmark",
    )
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")

    args = parser.parse_args()

    config = BenchmarkConfig(
        model_type=args.model,
        dim=args.dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iterations=args.iterations,
        num_warmup=args.warmup,
    )

    run_full_benchmark(config)


if __name__ == "__main__":
    main()
