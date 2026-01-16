#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pre-tokenize a HuggingFace dataset for faster training.

Tokenizes all texts once and saves to disk, eliminating the
tokenization bottleneck during training.

Usage:
    # Pre-tokenize FineWeb-Edu sample
    uv run python scripts/pretokenize.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --subset sample-10BT \
        --tokenizer NousResearch/Llama-2-7b-hf \
        --output data/fineweb-tokenized \
        --seq-len 4096 \
        --num-proc 8

    # Then train with:
    uv run python scripts/pretrain.py \
        --local-dataset data/fineweb-tokenized \
        --tokenizer NousResearch/Llama-2-7b-hf \
        ...
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., HuggingFaceFW/fineweb-edu)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/config (e.g., sample-10BT)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to tokenize",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="NousResearch/Llama-2-7b-hf",
        help="HuggingFace tokenizer to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenized dataset",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Sequence length for chunking tokens",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for parallel tokenization",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for tokenization (higher = faster but more RAM)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} (subset={args.subset})")
    ds = load_dataset(
        args.dataset,
        args.subset,
        split=args.split,
    )
    logger.info(f"Dataset size: {len(ds):,} samples")

    # Limit samples if requested
    if args.max_samples is not None and args.max_samples < len(ds):
        logger.info(f"Limiting to {args.max_samples:,} samples")
        ds = ds.select(range(args.max_samples))

    # Tokenization function
    seq_len = args.seq_len

    def tokenize_and_chunk(examples: dict) -> dict:
        """Tokenize texts and chunk into fixed-length sequences."""
        all_input_ids = []

        for text in examples.get("text", examples.get("content", [])):
            if not text:
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)

            # Chunk into sequences of seq_len + 1 (for input/label shift)
            for i in range(0, len(tokens) - seq_len, seq_len):
                chunk = tokens[i : i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    all_input_ids.append(chunk)

        return {"input_ids": all_input_ids}

    # Process dataset
    logger.info(f"Tokenizing with {args.num_proc} processes...")
    logger.info(f"Sequence length: {args.seq_len} tokens")

    # Remove original columns, keep only input_ids
    columns_to_remove = ds.column_names

    ds_tokenized = ds.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=columns_to_remove,
        desc="Tokenizing",
    )

    logger.info(f"Created {len(ds_tokenized):,} training sequences")

    # Save to disk
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_path}...")
    ds_tokenized.save_to_disk(str(output_path))

    # Calculate stats
    total_tokens = len(ds_tokenized) * (args.seq_len + 1)
    size_gb = total_tokens * 8 / (1024**3)  # int64 = 8 bytes

    logger.info("=" * 50)
    logger.info("Pre-tokenization complete!")
    logger.info(f"  Sequences: {len(ds_tokenized):,}")
    logger.info(f"  Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B)")
    logger.info(f"  Estimated size: {size_gb:.2f} GB")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 50)
    logger.info("")
    logger.info("To train with this dataset:")
    logger.info(f"  uv run python scripts/pretrain.py \\")
    logger.info(f"      --local-dataset {output_path} \\")
    logger.info(f"      --tokenizer {args.tokenizer} \\")
    logger.info(f"      --seq-len {args.seq_len} \\")
    logger.info(f"      ...")


if __name__ == "__main__":
    main()
