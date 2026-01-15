#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Inference script for Titans MLX models on Apple Silicon.

Optimized for M1/M2/M3/M4 GPUs with unified memory and lazy evaluation.

Usage:
    # Generate from trained model
    uv run python scripts/inference_mlx.py --checkpoint checkpoints_mlx/best_model.safetensors --prompt "Hello"

    # Generate with HuggingFace tokenizer
    uv run python scripts/inference_mlx.py --checkpoint model.safetensors --tokenizer meta-llama/Llama-2-7b-hf

    # Interactive mode with streaming
    uv run python scripts/inference_mlx.py --checkpoint model.safetensors --interactive --stream

    # Quantized inference (4-bit or 8-bit)
    uv run python scripts/inference_mlx.py --checkpoint model.safetensors --quantize 4
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans_mlx.memory import MemoryState

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Optional HuggingFace transformers
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Tokenizers
# =============================================================================


class SimpleTokenizer:
    """Simple character-level tokenizer for demo purposes."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self.char_to_id = {chr(i): i for i in range(vocab_size)}
        self.id_to_char = {i: chr(i) for i in range(vocab_size)}
        self.eos_token_id = 0
        self.pad_token_id = 0

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        """Encode text to token IDs."""
        ids = [self.char_to_id.get(c, 0) for c in text]
        if return_tensors == "mlx":
            return {"input_ids": mx.array([ids])}
        return ids

    def decode(
        self, ids: list[int] | mx.array, skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(ids, mx.array):
            ids = ids.tolist()
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def __call__(self, text: str, return_tensors: str | None = None) -> Any:
        return self.encode(text, return_tensors=return_tensors)


def load_tokenizer(
    tokenizer_name: str | None,
    vocab_size: int,
) -> SimpleTokenizer | PreTrainedTokenizerBase:
    """Load tokenizer - HuggingFace or simple character-level."""
    if tokenizer_name and HAS_TRANSFORMERS:
        logger.info(f"Loading HuggingFace tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    elif tokenizer_name and not HAS_TRANSFORMERS:
        logger.warning("transformers not installed, using simple tokenizer")

    logger.info("Using simple character-level tokenizer")
    return SimpleTokenizer(vocab_size)


# =============================================================================
# Model Loading
# =============================================================================


def create_model(model_type: str, config: TitansConfig) -> nn.Module:
    """Create Titans model based on type."""
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(models.keys())}"
        )
    return models[model_type](config)


def load_model(
    checkpoint_path: Path,
    quantize: int | None = None,
) -> tuple[nn.Module, TitansConfig, str, str | None]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        quantize: Quantization bits (None, 4, or 8)

    Returns:
        Tuple of (model, config, model_type, tokenizer_name)
    """
    # Load metadata
    meta_path = checkpoint_path.with_suffix(".meta.npz")
    if meta_path.exists():
        meta = np.load(str(meta_path))
        model_type = str(meta["model_type"][0])
        dim = int(meta["dim"][0])
        num_heads = int(meta["num_heads"][0])
        num_layers = int(meta["num_layers"][0])
        vocab_size = int(meta["vocab_size"][0])
        chunk_size = int(meta.get("chunk_size", [512])[0])
        window_size = int(meta.get("window_size", [512])[0])
        num_persistent_tokens = int(meta.get("num_persistent_tokens", [16])[0])
        num_memory_layers = int(meta.get("num_memory_layers", [2])[0])
        tokenizer_name = str(meta.get("tokenizer_name", [None])[0])
        if tokenizer_name == "None":
            tokenizer_name = None
    else:
        # Try to infer from checkpoint name
        logger.warning("No metadata file found, using default configuration")
        model_type = "mac"
        dim = 512
        num_heads = 8
        num_layers = 12
        vocab_size = 32000
        chunk_size = 512
        window_size = 512
        num_persistent_tokens = 16
        num_memory_layers = 2
        tokenizer_name = None

    config = TitansConfig(
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        chunk_size=chunk_size,
        window_size=window_size,
        num_persistent_tokens=num_persistent_tokens,
        num_memory_layers=num_memory_layers,
        dropout=0.0,
        use_conv=False,  # Disable conv for compatibility
    )

    model = create_model(model_type, config)

    # Load weights
    weights_path = checkpoint_path.with_suffix(".safetensors")
    if weights_path.exists():
        model.load_weights(str(weights_path))
    elif checkpoint_path.suffix == ".safetensors" and checkpoint_path.exists():
        model.load_weights(str(checkpoint_path))
    else:
        # Try npz format
        checkpoint = np.load(str(checkpoint_path), allow_pickle=True)
        weights = {}
        for k in checkpoint.files:
            if not k.startswith("_"):
                weights[k] = mx.array(checkpoint[k])
        model.update(weights)

    # Apply quantization if requested
    if quantize:
        model = quantize_model(model, quantize)
        logger.info(f"Applied {quantize}-bit quantization")

    mx.eval(model.parameters())

    logger.info(f"Loaded {model_type.upper()} model from {checkpoint_path}")

    return model, config, model_type, tokenizer_name


def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    """Apply quantization to model.

    Args:
        model: Model to quantize
        bits: Quantization bits (4 or 8)

    Returns:
        Quantized model
    """
    if bits not in (4, 8):
        logger.warning(f"Unsupported quantization bits: {bits}, using 8")
        bits = 8

    # MLX supports quantization via mlx.nn.quantize
    try:
        nn.quantize(model, bits=bits)
        logger.info(f"Quantized model to {bits} bits")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")

    return model


# =============================================================================
# Text Generation
# =============================================================================


def sample_top_p(probs: mx.array, p: float) -> mx.array:
    """Sample from top-p (nucleus) distribution."""
    sorted_indices = mx.argsort(-probs)
    sorted_probs = mx.take(probs, sorted_indices)
    cumulative_probs = mx.cumsum(sorted_probs)

    # Find cutoff
    cutoff_mask = cumulative_probs <= p
    # Always keep at least one token
    cutoff_mask = mx.concatenate([mx.array([True]), cutoff_mask[:-1]])

    # Zero out low probability tokens
    filtered_probs = mx.where(
        mx.take(cutoff_mask, mx.argsort(sorted_indices)),
        probs,
        mx.array(0.0),
    )

    # Renormalize
    filtered_probs = filtered_probs / (mx.sum(filtered_probs) + 1e-10)

    # Sample
    return mx.random.categorical(mx.log(filtered_probs + 1e-10))


def generate(
    model: nn.Module,
    input_ids: mx.array,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    states: list[MemoryState] | None = None,
    stream: bool = False,
) -> (
    Iterator[tuple[mx.array, list[MemoryState] | None]]
    | tuple[mx.array, list[MemoryState] | None]
):
    """Generate tokens autoregressively.

    Args:
        model: Titans model
        input_ids: Input token IDs (batch, seq)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Keep only top k tokens for sampling
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        eos_token_id: Stop generation at this token
        states: Initial memory states
        stream: If True, yield tokens one by one

    Returns:
        Generated token IDs and final memory states
    """
    generated = input_ids

    def _generate_step():
        nonlocal generated, states

        for _ in range(max_new_tokens):
            # Forward pass (use last chunk_size tokens for efficiency)
            context_size = min(generated.shape[1], model.config.chunk_size)
            context = generated[:, -context_size:]

            logits, states = model(context, states=states)
            mx.eval(logits)

            # Get logits for last position
            next_logits = logits[:, -1, :] / max(temperature, 1e-7)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                # Get unique tokens from generated sequence
                gen_list = generated[0].tolist() if generated.shape[0] == 1 else []
                for token_id in set(gen_list):
                    next_logits = next_logits.at[0, token_id].set(
                        next_logits[0, token_id] / repetition_penalty
                    )

            # Apply top-k filtering
            if top_k > 0 and top_k < next_logits.shape[-1]:
                # Get top-k values
                topk_values = mx.sort(next_logits)[:, -top_k:][:, 0]
                # Mask out tokens below threshold
                mask = next_logits < topk_values[:, None]
                next_logits = mx.where(mask, mx.array(-float("inf")), next_logits)

            # Softmax to get probabilities
            probs = mx.softmax(next_logits, axis=-1)

            # Apply top-p (nucleus) filtering and sample
            if top_p < 1.0:
                next_token = sample_top_p(probs[0], top_p).reshape(1, 1)
            else:
                # Standard categorical sampling
                next_token = mx.random.categorical(mx.log(probs + 1e-10)).reshape(1, 1)

            mx.eval(next_token)

            # Append to generated
            generated = mx.concatenate([generated, next_token], axis=1)

            # Check for EOS
            if eos_token_id is not None and int(next_token[0, 0]) == eos_token_id:
                break

            if stream:
                yield generated, states

        if not stream:
            yield generated, states

    if stream:
        return _generate_step()
    else:
        return next(_generate_step())


def generate_streaming(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    states: list[MemoryState] | None = None,
) -> Iterator[str]:
    """Generate tokens with streaming output.

    Yields decoded text incrementally.
    """
    # Encode prompt
    if hasattr(tokenizer, "__call__"):
        if HAS_TRANSFORMERS and hasattr(tokenizer, "encode"):
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = mx.array(encoded["input_ids"].numpy())
        else:
            encoded = tokenizer(prompt, return_tensors="mlx")
            input_ids = encoded["input_ids"]
    else:
        input_ids = mx.array([tokenizer.encode(prompt)])

    generator = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        states=states,
        stream=True,
    )

    for generated, _ in generator:
        # Decode only the new token
        new_token = generated[0, -1:].tolist()
        text = tokenizer.decode(new_token, skip_special_tokens=True)
        yield text


# =============================================================================
# Benchmark
# =============================================================================


def benchmark_generation(
    model: nn.Module,
    tokenizer: Any,
    prompt: str = "Hello, world!",
    num_tokens: int = 100,
    warmup: int = 2,
    repeat: int = 5,
) -> dict[str, float]:
    """Benchmark generation speed."""
    # Encode prompt
    if HAS_TRANSFORMERS and hasattr(tokenizer, "encode"):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = mx.array(encoded["input_ids"].numpy())
    else:
        encoded = tokenizer(prompt, return_tensors="mlx")
        input_ids = encoded["input_ids"]

    # Warmup
    for _ in range(warmup):
        output, _ = generate(model, input_ids, max_new_tokens=10)
        mx.eval(output)

    # Benchmark
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        output, _ = generate(model, input_ids, max_new_tokens=num_tokens)
        mx.eval(output)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_per_sec = num_tokens / avg_time

    return {
        "avg_time_s": avg_time,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": avg_time / num_tokens * 1000,
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with Titans MLX models")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (e.g., meta-llama/Llama-2-7b-hf)",
    )

    # Generation arguments
    parser.add_argument("--prompt", type=str, default="", help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.0, help="Repetition penalty"
    )

    # Mode arguments
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--stream", action="store_true", help="Stream output token by token"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run generation benchmark"
    )

    # Optimization arguments
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantization bits (4 or 8)",
    )

    args = parser.parse_args()

    # Load model
    model, config, model_type, saved_tokenizer = load_model(
        Path(args.checkpoint), args.quantize
    )

    # Load tokenizer (prefer command line, fallback to saved, then simple)
    tokenizer_name = args.tokenizer or saved_tokenizer
    tokenizer = load_tokenizer(tokenizer_name, config.vocab_size)

    if args.benchmark:
        # Run benchmark
        logger.info("Running generation benchmark...")
        results = benchmark_generation(
            model, tokenizer, args.prompt or "Hello, world!", args.max_tokens
        )
        print("\n" + "=" * 50)
        print("  Generation Benchmark Results")
        print("=" * 50)
        print(f"  Tokens generated: {args.max_tokens}")
        print(f"  Average time: {results['avg_time_s']:.3f}s")
        print(f"  Tokens/sec: {results['tokens_per_sec']:.1f}")
        print(f"  ms/token: {results['ms_per_token']:.2f}")
        print("=" * 50)
        return

    if args.interactive:
        # Interactive mode
        logger.info("Interactive mode. Type 'quit' to exit, 'reset' to clear memory.")
        states = None

        while True:
            try:
                prompt = input("\nYou: ")
                if prompt.lower() == "quit":
                    break
                if prompt.lower() == "reset":
                    states = None
                    logger.info("Memory cleared.")
                    continue

                if args.stream:
                    # Streaming output
                    print("Model: ", end="", flush=True)
                    start_time = time.time()
                    token_count = 0

                    for text in generate_streaming(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        states=states,
                    ):
                        print(text, end="", flush=True)
                        token_count += 1

                    elapsed = time.time() - start_time
                    print(
                        f"\n[{token_count} tokens in {elapsed:.2f}s = {token_count / elapsed:.1f} tok/s]"
                    )
                else:
                    # Normal generation
                    if HAS_TRANSFORMERS and hasattr(tokenizer, "encode"):
                        encoded = tokenizer(prompt, return_tensors="pt")
                        input_ids = mx.array(encoded["input_ids"].numpy())
                    else:
                        encoded = tokenizer(prompt, return_tensors="mlx")
                        input_ids = encoded["input_ids"]

                    start_time = time.time()
                    output_ids, states = generate(
                        model,
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        eos_token_id=getattr(tokenizer, "eos_token_id", None),
                        states=states,
                    )
                    mx.eval(output_ids)
                    elapsed = time.time() - start_time

                    generated_text = tokenizer.decode(
                        output_ids[0].tolist(),
                        skip_special_tokens=True,
                    )
                    new_tokens = output_ids.shape[1] - input_ids.shape[1]
                    print(f"Model: {generated_text}")
                    print(
                        f"[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s]"
                    )

            except KeyboardInterrupt:
                break

        logger.info("Goodbye!")

    else:
        # Single generation
        if not args.prompt:
            logger.warning("No prompt provided. Using empty prompt.")

        if HAS_TRANSFORMERS and hasattr(tokenizer, "encode"):
            encoded = tokenizer(args.prompt, return_tensors="pt")
            input_ids = mx.array(encoded["input_ids"].numpy())
        else:
            encoded = tokenizer(args.prompt, return_tensors="mlx")
            input_ids = encoded["input_ids"]

        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generating {args.max_tokens} tokens...")

        start_time = time.time()

        if args.stream:
            print("\n" + "=" * 50)
            print("Generated text:")
            print("=" * 50)
            print(args.prompt, end="", flush=True)

            token_count = 0
            for text in generate_streaming(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                print(text, end="", flush=True)
                token_count += 1

            elapsed = time.time() - start_time
            print(f"\n{'=' * 50}")
            print(
                f"[{token_count} tokens in {elapsed:.2f}s = {token_count / elapsed:.1f} tok/s]"
            )
        else:
            output_ids, _ = generate(
                model,
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
            mx.eval(output_ids)
            elapsed = time.time() - start_time

            generated_text = tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True,
            )

            print("\n" + "=" * 50)
            print("Generated text:")
            print("=" * 50)
            print(generated_text)
            print("=" * 50)

            new_tokens = output_ids.shape[1] - input_ids.shape[1]
            print(
                f"[{new_tokens} tokens in {elapsed:.2f}s = {new_tokens / elapsed:.1f} tok/s]"
            )


if __name__ == "__main__":
    main()
