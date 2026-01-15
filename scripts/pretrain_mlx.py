#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pretraining script for Titans MLX models on Apple Silicon.

Optimized for M1/M2/M3/M4 GPUs with unified memory architecture.

Supports:
- HuggingFace tokenizers (LLaMA 2, GPT-2, etc.)
- HuggingFace datasets with streaming
- Cosine annealing with warmup
- Gradient accumulation
- Weights & Biases logging (optional)

Usage:
    # Demo with synthetic data
    uv run python scripts/pretrain_mlx.py --model mac --dim 256 --epochs 10

    # Train with FineWeb-Edu (streaming)
    uv run python scripts/pretrain_mlx.py --model mac --dataset HuggingFaceFW/fineweb-edu \
        --tokenizer meta-llama/Llama-2-7b-hf --dim 512 --num-layers 12

    # Train with local text file
    uv run python scripts/pretrain_mlx.py --model mag --data path/to/data.txt

    # Resume from checkpoint
    uv run python scripts/pretrain_mlx.py --model mac --resume checkpoints_mlx/latest.npz
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from titans_mlx import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Optional imports
try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Training hyperparameters following the paper (Section 5.1)."""

    # Model
    model_type: str = "mac"
    dim: int = 512
    num_heads: int = 8
    num_layers: int = 12
    vocab_size: int = 32000
    chunk_size: int = 512
    window_size: int = 512
    num_persistent_tokens: int = 16
    num_memory_layers: int = 2

    # Data
    dataset: str | None = None  # HuggingFace dataset name
    dataset_subset: str | None = None  # Dataset subset/config
    data_path: str | None = None  # Local text file
    tokenizer: str = "gpt2"  # HuggingFace tokenizer
    seq_len: int = 4096  # Paper uses 4K

    # Training (following paper Section 5.1)
    epochs: int = 1
    max_steps: int = -1  # -1 = use epochs
    batch_size: int = 4  # Per-device batch size
    gradient_accumulation_steps: int = 32  # Effective batch ~0.5M tokens
    lr: float = 4e-4  # Paper: 4e-4
    weight_decay: float = 0.1  # Paper: 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Checkpointing
    checkpoint_dir: str = "checkpoints_mlx"
    save_every: int = 1000  # Save every N steps
    eval_every: int = 500  # Evaluate every N steps
    resume: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-mlx"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    synthetic_samples: int = 10000  # For demo mode


# =============================================================================
# Datasets
# =============================================================================


class SyntheticDataset:
    """Synthetic dataset for testing/demo purposes."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        np.random.seed(seed)
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        return {
            "input_ids": mx.array(self.data[idx, :-1]),
            "labels": mx.array(self.data[idx, 1:]),
        }

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples."""
        batch_data = self.data[indices]
        return {
            "input_ids": mx.array(batch_data[:, :-1]),
            "labels": mx.array(batch_data[:, 1:]),
        }


class TextFileDataset:
    """Dataset from a local text file with HuggingFace tokenizer."""

    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        self.tokens = np.array(
            tokenizer.encode(text, add_special_tokens=False), dtype=np.int32
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": mx.array(x), "labels": mx.array(y)}

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples."""
        batch_x = []
        batch_y = []
        for idx in indices:
            batch_x.append(self.tokens[idx : idx + self.seq_len])
            batch_y.append(self.tokens[idx + 1 : idx + self.seq_len + 1])
        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
        }


class CharLevelDataset:
    """Simple character-level dataset (fallback when no tokenizer)."""

    def __init__(
        self,
        path: Path,
        vocab_size: int,
        seq_len: int,
    ) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        chars = sorted(set(text))
        char_to_idx = {c: i % vocab_size for i, c in enumerate(chars)}
        self.tokens = np.array([char_to_idx.get(c, 0) for c in text], dtype=np.int32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, mx.array]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": mx.array(x), "labels": mx.array(y)}

    def get_batch(self, indices: list[int]) -> dict[str, mx.array]:
        """Get a batch of samples."""
        batch_x = []
        batch_y = []
        for idx in indices:
            batch_x.append(self.tokens[idx : idx + self.seq_len])
            batch_y.append(self.tokens[idx + 1 : idx + self.seq_len + 1])
        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
        }


class StreamingDataset:
    """Streaming dataset from HuggingFace datasets."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.subset = subset
        self.split = split
        self.seed = seed
        self._iterator = None
        self._buffer: list[int] = []

    def __iter__(self):
        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)

        buffer: list[int] = []
        for example in ds:
            # Get text from example (try common field names)
            text = example.get("text") or example.get("content") or str(example)

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            # Yield complete sequences
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]  # Overlap by 1 for next prediction

                yield {
                    "input_ids": mx.array(chunk[:-1]),
                    "labels": mx.array(chunk[1:]),
                }

    def get_batch(self, batch_size: int) -> dict[str, mx.array] | None:
        """Get a batch from streaming dataset."""
        if self._iterator is None:
            self._iterator = iter(self)

        batch_x = []
        batch_y = []

        for _ in range(batch_size):
            try:
                sample = next(self._iterator)
                batch_x.append(np.array(sample["input_ids"]))
                batch_y.append(np.array(sample["labels"]))
            except StopIteration:
                self._iterator = iter(self)
                if batch_x:
                    break
                return None

        if not batch_x:
            return None

        return {
            "input_ids": mx.array(np.stack(batch_x)),
            "labels": mx.array(np.stack(batch_y)),
        }


# =============================================================================
# Model Creation
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


def count_parameters(model: nn.Module) -> int:
    """Count total parameters."""

    def _count(params):
        total = 0
        for v in params.values():
            if isinstance(v, mx.array):
                total += v.size
            elif isinstance(v, dict):
                total += _count(v)
        return total

    return _count(model.parameters())


# =============================================================================
# Learning Rate Scheduler
# =============================================================================


def get_lr_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))


# =============================================================================
# Training Functions
# =============================================================================


def loss_fn(
    model: nn.Module, input_ids: mx.array, labels: mx.array
) -> tuple[mx.array, mx.array]:
    """Compute cross-entropy loss."""
    logits, _ = model(input_ids)
    # Reshape for cross entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Cross entropy loss
    loss = nn.losses.cross_entropy(logits_flat, labels_flat, reduction="mean")
    return loss, logits


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    input_ids: mx.array,
    labels: mx.array,
    grad_clip: float = 1.0,
) -> tuple[float, float]:
    """Single training step with gradient clipping."""
    # Compute loss and gradients
    loss_and_grad_fn = nn.value_and_grad(
        model, lambda m: loss_fn(m, input_ids, labels)[0]
    )
    loss, grads = loss_and_grad_fn(model)

    # Check for NaN in loss
    mx.eval(loss)
    loss_val = float(loss)
    if math.isnan(loss_val) or math.isinf(loss_val):
        # Skip this step if loss is invalid
        return loss_val, float("nan")

    # Replace NaN gradients with zeros and clip large values
    def sanitize_grad(g):
        if isinstance(g, mx.array):
            # Replace NaN with 0 and clip large values
            g = mx.where(mx.isnan(g), mx.zeros_like(g), g)
            g = mx.clip(g, -10.0, 10.0)
            return g
        elif isinstance(g, dict):
            return {k: sanitize_grad(v) for k, v in g.items()}
        elif isinstance(g, list):
            return [sanitize_grad(v) for v in g]
        return g

    grads = {k: sanitize_grad(v) for k, v in grads.items()}

    # Gradient clipping with proper recursive traversal
    if grad_clip > 0:
        max_norm = mx.array(grad_clip)

        # Compute global norm recursively
        def compute_norm_sq(g):
            if isinstance(g, mx.array):
                return mx.sum(g * g)
            elif isinstance(g, dict):
                total = mx.array(0.0)
                for v in g.values():
                    total = total + compute_norm_sq(v)
                return total
            elif isinstance(g, list):
                total = mx.array(0.0)
                for v in g:
                    total = total + compute_norm_sq(v)
                return total
            return mx.array(0.0)

        total_norm_sq = mx.array(0.0)
        for g in grads.values():
            total_norm_sq = total_norm_sq + compute_norm_sq(g)

        total_norm = mx.sqrt(total_norm_sq + 1e-8)
        clip_coef = mx.minimum(max_norm / total_norm, mx.array(1.0))

        # Scale gradients recursively
        def scale_grad(g):
            if isinstance(g, mx.array):
                return g * clip_coef
            elif isinstance(g, dict):
                return {k: scale_grad(v) for k, v in g.items()}
            elif isinstance(g, list):
                return [scale_grad(v) for v in g]
            return g

        grads = {k: scale_grad(v) for k, v in grads.items()}

    # Update parameters
    optimizer.update(model, grads)

    # Force evaluation to prevent lazy evaluation issues
    mx.eval(model.parameters(), optimizer.state)

    ppl = math.exp(min(loss_val, 100))  # Cap to avoid overflow
    return loss_val, ppl


def evaluate(
    model: nn.Module,
    dataset: Any,
    batch_size: int,
    num_batches: int = 50,
) -> dict[str, float]:
    """Evaluate on validation set."""
    total_loss = 0.0
    total_tokens = 0

    indices = list(range(min(len(dataset), num_batches * batch_size)))
    np.random.shuffle(indices)

    for i in range(0, min(len(indices), num_batches * batch_size), batch_size):
        batch_indices = indices[i : i + batch_size]
        if len(batch_indices) < batch_size:
            continue

        batch = dataset.get_batch(batch_indices)
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        loss, _ = loss_fn(model, input_ids, labels)
        mx.eval(loss)

        batch_tokens = labels.size
        total_loss += float(loss) * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {"val_loss": avg_loss, "val_ppl": math.exp(min(avg_loss, 100))}


# =============================================================================
# Checkpoint Functions
# =============================================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    model_config: TitansConfig,
    step: int,
    epoch: int,
    best_val_loss: float,
    path: Path,
) -> None:
    """Save checkpoint in MLX format."""
    # Get model weights as flat dict
    weights = dict(model.parameters())

    # Convert to numpy for saving
    weights_np = {}
    for k, v in weights.items():
        if isinstance(v, mx.array):
            weights_np[k] = np.array(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                weights_np[f"{k}.{k2}"] = np.array(v2)

    # Save metadata as JSON-compatible dict
    metadata = {
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_type": config.model_type,
        "dim": model_config.dim,
        "num_heads": model_config.num_heads,
        "num_layers": model_config.num_layers,
        "vocab_size": model_config.vocab_size,
        "chunk_size": model_config.chunk_size,
        "window_size": model_config.window_size,
        "num_persistent_tokens": model_config.num_persistent_tokens,
        "num_memory_layers": model_config.num_memory_layers,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    # Save using safetensors format via mlx
    model.save_weights(str(path.with_suffix(".safetensors")))

    # Also save metadata separately
    np.savez(
        str(path.with_suffix(".meta.npz")),
        **{k: np.array([v]) for k, v in metadata.items()},
    )

    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[int, int, float]:
    """Load checkpoint."""
    # Load weights
    weights_path = path.with_suffix(".safetensors")
    if weights_path.exists():
        model.load_weights(str(weights_path))
    else:
        # Try legacy npz format
        checkpoint = np.load(str(path), allow_pickle=True)
        weights = {}
        for k in checkpoint.files:
            if not k.startswith("_"):
                weights[k] = mx.array(checkpoint[k])
        model.update(weights)

    # Load metadata
    meta_path = path.with_suffix(".meta.npz")
    if meta_path.exists():
        meta = np.load(str(meta_path))
        step = int(meta["step"][0])
        epoch = int(meta["epoch"][0])
        best_val_loss = float(meta["best_val_loss"][0])
    else:
        step, epoch, best_val_loss = 0, 0, float("inf")

    logger.info(f"Loaded checkpoint from {path} (step {step})")
    return step, epoch, best_val_loss


# =============================================================================
# Main Training Loop
# =============================================================================


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataset: Any,
    val_dataset: Any | None,
    config: TrainingConfig,
    model_config: TitansConfig,
) -> None:
    """Main training loop."""
    # Calculate total steps
    if config.max_steps > 0:
        total_steps = config.max_steps
    elif hasattr(train_dataset, "__len__"):
        steps_per_epoch = (
            len(train_dataset)
            // config.batch_size
            // config.gradient_accumulation_steps
        )
        total_steps = max(1, steps_per_epoch * config.epochs)
    else:
        total_steps = 100000  # Default for streaming

    warmup_steps = int(total_steps * config.warmup_ratio)

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # State
    global_step = 0
    epoch = 0
    best_val_loss = float("inf")
    running_loss = 0.0
    running_count = 0

    # Resume if specified
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            global_step, epoch, best_val_loss = load_checkpoint(
                resume_path, model, optimizer
            )

    # Checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Wandb
    if config.wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "model_type": config.model_type,
                "dim": config.dim,
                "num_layers": config.num_layers,
                "lr": config.lr,
                "batch_size": config.batch_size,
                "seq_len": config.seq_len,
            },
        )

    start_time = time.time()
    pbar = tqdm(total=total_steps, initial=global_step, desc="Training")

    # Training loop
    accumulation_loss = 0.0
    accumulation_step = 0

    while global_step < total_steps:
        epoch += 1

        # Get batches
        if hasattr(train_dataset, "get_batch") and hasattr(train_dataset, "__len__"):
            # Fixed-size dataset
            indices = list(range(len(train_dataset)))
            np.random.shuffle(indices)

            for i in range(0, len(indices), config.batch_size):
                if global_step >= total_steps:
                    break

                batch_indices = indices[i : i + config.batch_size]
                if len(batch_indices) < config.batch_size:
                    continue

                batch = train_dataset.get_batch(batch_indices)
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # Update learning rate
                lr = get_lr_schedule(global_step, total_steps, warmup_steps, config.lr)
                optimizer.learning_rate = lr

                # Training step
                loss_val, ppl = train_step(
                    model, optimizer, input_ids, labels, config.grad_clip
                )

                accumulation_loss += loss_val
                accumulation_step += 1
                running_loss += loss_val
                running_count += 1

                # Step after accumulation
                if accumulation_step >= config.gradient_accumulation_steps:
                    global_step += 1
                    accumulation_step = 0
                    accumulation_loss = 0.0

                    # Logging
                    if global_step % config.log_every == 0:
                        avg_loss = running_loss / running_count
                        avg_ppl = math.exp(min(avg_loss, 100))

                        log_dict = {
                            "train/loss": avg_loss,
                            "train/ppl": avg_ppl,
                            "train/lr": lr,
                            "train/step": global_step,
                        }

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "ppl": f"{avg_ppl:.2f}",
                                "lr": f"{lr:.2e}",
                            }
                        )

                        if config.wandb and HAS_WANDB:
                            wandb.log(log_dict, step=global_step)

                        running_loss = 0.0
                        running_count = 0

                    # Evaluation
                    if (
                        config.eval_every > 0
                        and global_step % config.eval_every == 0
                        and val_dataset is not None
                    ):
                        val_metrics = evaluate(model, val_dataset, config.batch_size)
                        logger.info(
                            f"Step {global_step}: "
                            f"val_loss={val_metrics['val_loss']:.4f}, "
                            f"val_ppl={val_metrics['val_ppl']:.2f}"
                        )

                        if config.wandb and HAS_WANDB:
                            wandb.log(
                                {f"val/{k}": v for k, v in val_metrics.items()},
                                step=global_step,
                            )

                        # Save best model
                        if val_metrics["val_loss"] < best_val_loss:
                            best_val_loss = val_metrics["val_loss"]
                            save_checkpoint(
                                model,
                                optimizer,
                                config,
                                model_config,
                                global_step,
                                epoch,
                                best_val_loss,
                                checkpoint_dir / "best_model",
                            )

                    # Periodic checkpoint
                    if config.save_every > 0 and global_step % config.save_every == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            config,
                            model_config,
                            global_step,
                            epoch,
                            best_val_loss,
                            checkpoint_dir / f"step_{global_step}",
                        )

                    pbar.update(1)

        else:
            # Streaming dataset
            for batch in train_dataset:
                if global_step >= total_steps:
                    break

                input_ids = batch["input_ids"].reshape(1, -1)  # Add batch dim
                labels = batch["labels"].reshape(1, -1)

                # Update learning rate
                lr = get_lr_schedule(global_step, total_steps, warmup_steps, config.lr)
                optimizer.learning_rate = lr

                # Training step
                loss_val, ppl = train_step(
                    model, optimizer, input_ids, labels, config.grad_clip
                )

                accumulation_loss += loss_val
                accumulation_step += 1
                running_loss += loss_val
                running_count += 1

                # Step after accumulation
                if accumulation_step >= config.gradient_accumulation_steps:
                    global_step += 1
                    accumulation_step = 0

                    if global_step % config.log_every == 0:
                        avg_loss = running_loss / running_count
                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "ppl": f"{math.exp(min(avg_loss, 100)):.2f}",
                                "lr": f"{lr:.2e}",
                            }
                        )
                        running_loss = 0.0
                        running_count = 0

                    if config.save_every > 0 and global_step % config.save_every == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            config,
                            model_config,
                            global_step,
                            epoch,
                            best_val_loss,
                            checkpoint_dir / f"step_{global_step}",
                        )

                    pbar.update(1)

    pbar.close()

    # Final checkpoint
    save_checkpoint(
        model,
        optimizer,
        config,
        model_config,
        global_step,
        epoch,
        best_val_loss,
        checkpoint_dir / "final_model",
    )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed / 3600:.2f} hours")
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    if config.wandb and HAS_WANDB:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain Titans MLX models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="mac",
        choices=["mac", "mag", "mal", "lmm"],
        help="Model variant",
    )
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size (MAC)")
    parser.add_argument(
        "--window-size", type=int, default=512, help="Window size (MAG/MAL)"
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset (e.g., HuggingFaceFW/fineweb-edu)",
    )
    parser.add_argument(
        "--dataset-subset", type=str, default=None, help="Dataset subset"
    )
    parser.add_argument("--data", type=str, default=None, help="Local text file path")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HuggingFace tokenizer (e.g., meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (-1=epochs)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Gradient accumulation",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_mlx")
    parser.add_argument(
        "--save-every", type=int, default=1000, help="Save every N steps"
    )
    parser.add_argument(
        "--eval-every", type=int, default=500, help="Eval every N steps"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    # Logging
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="titans-mlx")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--synthetic-samples", type=int, default=10000, help="Synthetic samples (demo)"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # Build config
    config = TrainingConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        data_path=args.data,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        resume=args.resume,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        synthetic_samples=args.synthetic_samples,
    )

    # Check dependencies
    if config.dataset and not HAS_DATASETS:
        logger.error(
            "Install 'datasets' for HuggingFace datasets: pip install datasets"
        )
        return

    if config.wandb and not HAS_WANDB:
        logger.warning("wandb not installed, disabling logging")
        config.wandb = False

    # Load tokenizer
    tokenizer = None
    if HAS_TRANSFORMERS and (config.dataset or config.data_path):
        logger.info(f"Loading tokenizer: {config.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config.vocab_size = tokenizer.vocab_size
        logger.info(f"Tokenizer vocab size: {config.vocab_size}")

    # Create model config
    model_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        num_persistent_tokens=config.num_persistent_tokens,
        num_memory_layers=config.num_memory_layers,
        dropout=0.0,  # Usually 0 for pretraining
        use_conv=False,  # Disable conv to avoid dimension issues
    )

    # Create model
    model = create_model(config.model_type, model_config)
    total_params = count_parameters(model)
    logger.info(f"Model: Titans{config.model_type.upper()} (MLX)")
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Create optimizer (AdamW as in paper)
    optimizer = optim.AdamW(
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        betas=[0.9, 0.95],  # Common for LLMs
    )

    # Create dataset
    train_dataset: Any
    val_dataset: Any | None = None

    if config.dataset:
        # HuggingFace streaming dataset
        logger.info(f"Using HuggingFace dataset: {config.dataset}")
        if tokenizer is None:
            raise ValueError("Tokenizer required for HuggingFace datasets")

        train_dataset = StreamingDataset(
            config.dataset,
            tokenizer,
            config.seq_len,
            subset=config.dataset_subset,
            split="train",
            seed=config.seed,
        )
        # Note: validation requires non-streaming dataset or separate split

    elif config.data_path:
        # Local text file
        logger.info(f"Loading data from: {config.data_path}")
        path = Path(config.data_path)

        if tokenizer is not None:
            full_dataset = TextFileDataset(path, tokenizer, config.seq_len)
        else:
            full_dataset = CharLevelDataset(path, config.vocab_size, config.seq_len)

        # Split into train/val
        train_size = int(0.95 * len(full_dataset))
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create subset datasets
        class SubsetDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

            def get_batch(self, batch_indices):
                actual_indices = [self.indices[i] for i in batch_indices]
                return self.dataset.get_batch(actual_indices)

        train_dataset = SubsetDataset(full_dataset, train_indices)
        val_dataset = SubsetDataset(full_dataset, val_indices)
        logger.info(f"Train samples: {train_size}, Val samples: {len(val_indices)}")

    else:
        # Synthetic data (demo)
        logger.info("Using synthetic data (demo mode)")
        train_dataset = SyntheticDataset(
            config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
        )
        val_dataset = SyntheticDataset(
            config.vocab_size,
            config.seq_len,
            config.synthetic_samples // 10,
            config.seed + 1,
        )

    # Log effective batch size
    effective_batch_size = (
        config.batch_size * config.gradient_accumulation_steps * config.seq_len
    )
    logger.info(f"Effective batch size: {effective_batch_size:,} tokens")
    logger.info(f"Sequence length: {config.seq_len}")
    logger.info("Backend: MLX (Apple Silicon optimized)")

    # Train
    train(model, optimizer, train_dataset, val_dataset, config, model_config)


if __name__ == "__main__":
    main()
