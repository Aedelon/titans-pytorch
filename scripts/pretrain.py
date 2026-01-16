#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Pretraining script for Titans models.

Supports:
- HuggingFace tokenizers (LLaMA 2, GPT-2, etc.)
- HuggingFace datasets (FineWeb-Edu, etc.) with streaming
- Mixed precision training (fp16/bf16)
- Gradient accumulation for large effective batch sizes
- Cosine annealing with warmup
- Weights & Biases logging (optional)
- Multi-GPU with Accelerate (optional)

Usage:
    # Demo with synthetic data
    uv run python scripts/pretrain.py --model mac --dim 256 --epochs 10

    # Train with FineWeb-Edu (streaming)
    uv run python scripts/pretrain.py --model mac --dataset HuggingFaceFW/fineweb-edu \\
        --tokenizer meta-llama/Llama-2-7b-hf --dim 512 --num-layers 12

    # Train with local text file
    uv run python scripts/pretrain.py --model mag --data path/to/data.txt

    # Resume from checkpoint
    uv run python scripts/pretrain.py --model mac --resume checkpoints/latest.pt

    # With wandb logging
    uv run python scripts/pretrain.py --model mac --wandb --wandb-project titans
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

# CUDA optimizations
try:
    from titans.cuda_optimizations import (
        CUDAPrefetcher,
        configure_memory_pool,
        empty_cache_if_needed,
    )
    HAS_CUDA_OPTS = True
except ImportError:
    HAS_CUDA_OPTS = False

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
    local_dataset: str | None = None  # Local HuggingFace dataset (from save_to_disk)
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

    # Mixed precision
    mixed_precision: str = "bf16"  # none, fp16, bf16

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000  # Save every N steps
    eval_every: int = 500  # Evaluate every N steps
    resume: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    num_workers: int = 4
    synthetic_samples: int = 10000  # For demo mode

    # CUDA optimizations
    use_torch_compile: bool = False  # Enable torch.compile for faster training
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory efficiency


# =============================================================================
# Datasets
# =============================================================================


class SyntheticDataset(Dataset):
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

        generator = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size, (num_samples, seq_len + 1), generator=generator
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.data[idx, :-1],
            "labels": self.data[idx, 1:],
        }


class TextFileDataset(Dataset):
    """Dataset from a local text file with HuggingFace tokenizer."""

    def __init__(
        self,
        path: Path,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
    ) -> None:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


class CharLevelDataset(Dataset):
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
        self.tokens = torch.tensor(
            [char_to_idx.get(c, 0) for c in text], dtype=torch.long
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


class LocalHFDataset(Dataset):
    """Dataset from a local HuggingFace dataset saved with save_to_disk().

    Supports two formats:
    1. Pre-tokenized: dataset has 'input_ids' field (from pretokenize.py)
    2. Raw text: dataset has 'text' field (will tokenize on load)
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerBase | None,
        seq_len: int,
        seed: int = 42,
    ) -> None:
        from datasets import load_from_disk

        self.seq_len = seq_len

        # Load dataset from disk
        logger.info(f"Loading local dataset from {path}")
        self.dataset = load_from_disk(path)

        # Shuffle
        self.dataset = self.dataset.shuffle(seed=seed)
        logger.info(f"Loaded {len(self.dataset)} samples")

        # Check if already pre-tokenized
        first_example = self.dataset[0]
        self.is_pretokenized = "input_ids" in first_example and isinstance(
            first_example["input_ids"], (list, tuple)
        )

        if self.is_pretokenized:
            # Already tokenized - use directly (fast path)
            logger.info("Dataset is pre-tokenized, using directly (fast path)")
            self.tokens_list = None  # Use self.dataset directly
        else:
            # Need to tokenize (slow path)
            if tokenizer is None:
                raise ValueError("Tokenizer required for non-pretokenized datasets")

            logger.info("Dataset is raw text, tokenizing...")
            self.tokens_list = []
            buffer = []

            for i, example in enumerate(self.dataset):
                text = example.get("text") or example.get("content") or str(example)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(tokens)

                # Create samples from buffer
                while len(buffer) >= seq_len + 1:
                    self.tokens_list.append(
                        torch.tensor(buffer[: seq_len + 1], dtype=torch.long)
                    )
                    buffer = buffer[seq_len:]  # Non-overlapping for efficiency

                if (i + 1) % 50000 == 0:
                    logger.info(
                        f"  Processed {i+1} samples, created {len(self.tokens_list)} sequences"
                    )

            logger.info(f"Created {len(self.tokens_list)} training sequences")

    def __len__(self) -> int:
        if self.is_pretokenized:
            return len(self.dataset)
        return len(self.tokens_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.is_pretokenized:
            # Pre-tokenized: input_ids already contains seq_len+1 tokens
            tokens = self.dataset[idx]["input_ids"]
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            return {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }
        else:
            # Tokenized on load
            tokens = self.tokens_list[idx]
            return {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }


class StreamingDataset(IterableDataset):
    """Streaming dataset from HuggingFace datasets with multi-worker support."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        subset: str | None = None,
        split: str = "train",
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.subset = subset
        self.split = split
        self.seed = seed
        self.num_workers = num_workers

    def __iter__(self):
        # Get worker info for multi-process data loading
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Load dataset in streaming mode
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed + worker_id, buffer_size=10000)

        buffer = []
        sample_idx = 0
        for example in ds:
            # Shard across workers: each worker processes every Nth example
            if sample_idx % num_workers != worker_id:
                sample_idx += 1
                continue
            sample_idx += 1

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
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
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


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Learning Rate Scheduler
# =============================================================================


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup (as used in the paper)."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training Loop
# =============================================================================


class Trainer:
    """Training loop with mixed precision and gradient accumulation."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = self.model.to(self.device)

        # Apply torch.compile for faster training (PyTorch 2.0+)
        if config.use_torch_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            logger.info(f"Applying torch.compile (mode={config.compile_mode})")
            try:
                self.model = torch.compile(
                    self.model,
                    mode=config.compile_mode,
                    fullgraph=False,
                    dynamic=True,
                )
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Configure CUDA memory pool
        if HAS_CUDA_OPTS and self.device.type == "cuda":
            configure_memory_pool()

        # Note: Gradient checkpointing is incompatible with Titans memory module
        # which computes gradients during forward pass. Skip for now.
        if config.gradient_checkpointing:
            logger.warning(
                "Gradient checkpointing is not compatible with Titans memory module. "
                "The memory module computes gradients during forward pass which conflicts "
                "with checkpointing. Ignoring --gradient-checkpointing flag."
            )

        # Optimizer (AdamW as in paper)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),  # Common for LLMs
        )

        # Calculate total steps
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            steps_per_epoch = (
                len(train_dataloader) // config.gradient_accumulation_steps
            )
            self.total_steps = steps_per_epoch * config.epochs

        # Scheduler
        num_warmup_steps = int(self.total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps, self.total_steps
        )

        # Mixed precision
        self.scaler = None
        self.autocast_dtype = None
        if config.mixed_precision == "fp16" and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
            self.autocast_dtype = torch.float16
        elif config.mixed_precision == "bf16":
            self.autocast_dtype = torch.bfloat16

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Wandb
        if config.wandb and HAS_WANDB:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

    def train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Single training step with mixed precision."""
        # Use non_blocking=True to overlap CPU->GPU transfer with computation
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        # Forward pass with autocast
        if self.autocast_dtype is not None:
            with torch.autocast(
                device_type=self.device.type, dtype=self.autocast_dtype
            ):
                logits, _ = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
        else:
            logits, _ = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Return loss tensor directly - avoid .item() sync on every step
        # Only convert to scalar when actually logging (every log_every steps)
        return loss, {"loss_tensor": loss.detach()}

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all transformer blocks."""
        from torch.utils.checkpoint import checkpoint

        # Find all blocks with forward method
        for name, module in self.model.named_modules():
            if hasattr(module, "blocks") and isinstance(module.blocks, nn.ModuleList):
                for i, block in enumerate(module.blocks):
                    original_forward = block.forward

                    def make_checkpointed_forward(orig_fwd):
                        def checkpointed_forward(*args, **kwargs):
                            # Use use_reentrant=False for better compatibility
                            def forward_fn(*fwd_args):
                                return orig_fwd(*fwd_args, **kwargs)
                            return checkpoint(forward_fn, *args, use_reentrant=False)
                        return checkpointed_forward

                    block.forward = make_checkpointed_forward(original_forward)

    def optimizer_step(self) -> None:
        """Optimizer step with gradient clipping and scaling."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            if self.autocast_dtype is not None:
                with torch.autocast(
                    device_type=self.device.type, dtype=self.autocast_dtype
                ):
                    logits, _ = self.model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )
            else:
                logits, _ = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            batch_tokens = labels.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        self.model.train()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return {"val_loss": avg_loss, "val_ppl": math.exp(avg_loss)}

    def save_checkpoint(self, name: str = "checkpoint") -> Path:
        """Save training checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": vars(self.config),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self) -> None:
        """Main training loop."""
        self.model.train()
        self.optimizer.zero_grad()

        accumulation_step = 0
        running_loss_tensor = None  # Keep as tensor to avoid sync
        running_count = 0

        start_time = time.time()
        pbar = tqdm(total=self.total_steps, initial=self.global_step, desc="Training")

        # Use CUDA prefetching for better GPU utilization
        use_prefetch = (
            HAS_CUDA_OPTS
            and self.device.type == "cuda"
            and not isinstance(self.train_dataloader.dataset, IterableDataset)
        )

        while self.global_step < self.total_steps:
            self.epoch += 1

            # Setup data iterator with optional prefetching
            if use_prefetch:
                data_iter = CUDAPrefetcher(self.train_dataloader, self.device)
            else:
                data_iter = self.train_dataloader

            for batch in data_iter:
                if self.global_step >= self.total_steps:
                    break

                # Training step
                loss, metrics = self.train_step(batch)
                # Accumulate loss tensor without sync - sync only at logging time
                if running_count == 0:
                    running_loss_tensor = metrics["loss_tensor"]
                else:
                    running_loss_tensor = running_loss_tensor + metrics["loss_tensor"]
                running_count += 1
                accumulation_step += 1

                # Optimizer step after accumulation
                if accumulation_step >= self.config.gradient_accumulation_steps:
                    self.optimizer_step()
                    accumulation_step = 0

                    # Logging - only sync here (every log_every steps)
                    if self.global_step % self.config.log_every == 0:
                        # Single sync point: convert accumulated tensor to scalar
                        avg_loss = (running_loss_tensor / running_count).item()
                        current_lr = self.scheduler.get_last_lr()[0]

                        log_dict = {
                            "train/loss": avg_loss,
                            "train/ppl": math.exp(min(avg_loss, 20)),  # Clamp to avoid overflow
                            "train/lr": current_lr,
                            "train/step": self.global_step,
                        }

                        pbar.set_postfix(
                            {
                                "loss": f"{avg_loss:.4f}",
                                "ppl": f"{math.exp(min(avg_loss, 20)):.2f}",
                                "lr": f"{current_lr:.2e}",
                            }
                        )

                        if self.config.wandb and HAS_WANDB:
                            wandb.log(log_dict, step=self.global_step)

                        running_loss_tensor = None
                        running_count = 0

                    # Evaluation
                    if (
                        self.config.eval_every > 0
                        and self.global_step % self.config.eval_every == 0
                    ):
                        val_metrics = self.evaluate()
                        if val_metrics:
                            logger.info(
                                f"Step {self.global_step}: "
                                f"val_loss={val_metrics['val_loss']:.4f}, "
                                f"val_ppl={val_metrics['val_ppl']:.2f}"
                            )

                            if self.config.wandb and HAS_WANDB:
                                wandb.log(
                                    {f"val/{k}": v for k, v in val_metrics.items()},
                                    step=self.global_step,
                                )

                            # Save best model
                            if val_metrics["val_loss"] < self.best_val_loss:
                                self.best_val_loss = val_metrics["val_loss"]
                                self.save_checkpoint("best_model")

                    # Periodic checkpoint
                    if (
                        self.config.save_every > 0
                        and self.global_step % self.config.save_every == 0
                    ):
                        self.save_checkpoint(f"step_{self.global_step}")

                    pbar.update(1)

        pbar.close()

        # Final checkpoint
        self.save_checkpoint("final_model")

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed / 3600:.2f} hours")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.config.wandb and HAS_WANDB:
            wandb.finish()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain Titans models",
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
        "--local-dataset",
        type=str,
        default=None,
        help="Local HuggingFace dataset path (from save_to_disk)",
    )
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

    # Mixed precision
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
        help="Mixed precision mode",
    )

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
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
    parser.add_argument("--wandb-project", type=str, default="titans")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--synthetic-samples", type=int, default=10000, help="Synthetic samples (demo)"
    )

    # CUDA optimizations
    parser.add_argument(
        "--torch-compile", action="store_true", help="Enable torch.compile for faster training"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

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
        local_dataset=args.local_dataset,
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
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        resume=args.resume,
        log_every=args.log_every,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        num_workers=args.num_workers,
        synthetic_samples=args.synthetic_samples,
        use_torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        gradient_checkpointing=args.gradient_checkpointing,
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
    if HAS_TRANSFORMERS and (config.dataset or config.data_path or config.local_dataset):
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
    )

    # Create model
    model = create_model(config.model_type, model_config)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model: Titans{config.model_type.upper()}")
    logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create dataset
    train_dataset: Dataset | IterableDataset
    val_dataset: Dataset | None = None

    if config.local_dataset:
        # Local HuggingFace dataset (from save_to_disk)
        logger.info(f"Using local HuggingFace dataset: {config.local_dataset}")
        if tokenizer is None:
            raise ValueError("Tokenizer required for local datasets")

        full_dataset = LocalHFDataset(
            config.local_dataset,
            tokenizer,
            config.seq_len,
            seed=config.seed,
        )

        # Split into train/val
        train_size = int(0.95 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        else:
            train_dataset = full_dataset
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {val_size}")

    elif config.dataset:
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
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        logger.info(f"Train samples: {train_size}, Val samples: {val_size}")

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

    # Create dataloaders with optimized settings
    is_streaming = isinstance(train_dataset, IterableDataset)
    use_workers = config.num_workers > 0

    # DataLoader optimization settings
    loader_kwargs = {
        "batch_size": config.batch_size,
        "pin_memory": True,  # Fast CPU->GPU transfer
        "num_workers": config.num_workers,
        "drop_last": True,  # Avoid incomplete batches for consistent GPU perf
    }

    # Add prefetch and persistent workers only if using workers
    if use_workers:
        loader_kwargs["prefetch_factor"] = 4  # Prefetch 4 batches per worker
        loader_kwargs["persistent_workers"] = True  # Keep workers alive between epochs

    if is_streaming:
        loader_kwargs["shuffle"] = False  # Streaming handles its own shuffling
    else:
        loader_kwargs["shuffle"] = True

    logger.info(
        f"DataLoader: num_workers={config.num_workers}, pin_memory=True, "
        f"prefetch_factor={4 if use_workers else 'N/A'}, "
        f"persistent_workers={use_workers}, drop_last=True"
    )

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    val_loader = None
    if val_dataset is not None:
        val_loader_kwargs = {
            "batch_size": config.batch_size,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": config.num_workers,
        }
        if use_workers:
            val_loader_kwargs["prefetch_factor"] = 2
            val_loader_kwargs["persistent_workers"] = True
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # Log effective batch size
    effective_batch_size = (
        config.batch_size * config.gradient_accumulation_steps * config.seq_len
    )
    logger.info(f"Effective batch size: {effective_batch_size:,} tokens")
    logger.info(f"Sequence length: {config.seq_len}")
    logger.info(f"Mixed precision: {config.mixed_precision}")

    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    # Resume if specified
    if config.resume:
        trainer.load_checkpoint(Path(config.resume))

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
