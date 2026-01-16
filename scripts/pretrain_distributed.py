#!/usr/bin/env python3
# Copyright 2024 Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0

"""
Distributed pretraining script for Titans models using Accelerate.

Supports:
- Multi-GPU training with DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel) for large models
- DeepSpeed ZeRO stages 1, 2, 3
- Mixed precision (fp16/bf16)
- Gradient accumulation
- Gradient checkpointing

Usage:
    # Single GPU (same as pretrain.py)
    uv run python scripts/pretrain_distributed.py --model mac --dim 256

    # Multi-GPU with DDP (auto-detects available GPUs)
    uv run accelerate launch scripts/pretrain_distributed.py --model mac --dim 512

    # Multi-GPU with FSDP
    uv run accelerate launch --config_file configs/fsdp_config.yaml \
        scripts/pretrain_distributed.py --model mac --dim 1024

    # Multi-node training
    uv run accelerate launch --multi_gpu --num_machines 2 --machine_rank 0 \
        scripts/pretrain_distributed.py --model mac
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

# CUDA optimizations - set before any CUDA operations
if torch.cuda.is_available():
    # Auto-tune convolution algorithms for hardware
    torch.backends.cudnn.benchmark = True
    # Use TF32 for faster matmul on Ampere+ GPUs (slight precision loss, big speedup)
    torch.set_float32_matmul_precision('high')
    # Optimize memory allocator
    torch.cuda.set_per_process_memory_fraction(0.95)

# torch.compile optimizations
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.suppress_errors = True
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", ".cache/torch_compile")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

from titans import TitansConfig, TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Optional imports
try:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedTokenizerBase = Any  # type: ignore

try:
    from datasets import load_dataset, load_from_disk

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
# Configuration
# =============================================================================


@dataclass
class DistributedTrainingConfig:
    """Training configuration with distributed training options."""

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
    dataset: str | None = None
    dataset_subset: str | None = None
    local_dataset: str | None = None  # Pre-tokenized local dataset
    data_path: str | None = None
    tokenizer: str = "gpt2"
    seq_len: int = 4096

    # Training
    epochs: int = 1
    max_steps: int = -1
    batch_size: int = 4
    gradient_accumulation_steps: int = 32
    lr: float = 4e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03

    # Mixed precision (handled by Accelerate)
    mixed_precision: str = "bf16"

    # Distributed
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    gradient_checkpointing: bool = False

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    resume: str | None = None

    # Logging
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "titans-distributed"
    wandb_run_name: str | None = None

    # Other
    seed: int = 42
    num_workers: int = 4
    synthetic_samples: int = 10000


# =============================================================================
# Datasets (same as pretrain.py)
# =============================================================================


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""

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
    """Dataset from local text file."""

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


class StreamingDataset(IterableDataset):
    """Streaming dataset from HuggingFace."""

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

    def __iter__(self):
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)

        buffer = []
        for example in ds:
            text = example.get("text", example.get("content", ""))
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]

                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }


class LocalHFDataset(Dataset):
    """Dataset from pre-tokenized local HuggingFace dataset (Arrow format)."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.seq_len = seq_len

        # Load dataset (Arrow memory-mapped)
        logger.info(f"Loading local dataset from {path}")
        self.dataset = load_from_disk(path)
        num_samples = len(self.dataset)
        logger.info(f"Loaded {num_samples} samples (Arrow memory-mapped)")

        # Check if already pre-tokenized
        first_example = self.dataset[0]
        self.is_pretokenized = "input_ids" in first_example and isinstance(
            first_example["input_ids"], (list, tuple)
        )

        if self.is_pretokenized:
            logger.info("Dataset is pre-tokenized, ready for training")
        else:
            raise ValueError(
                "Raw text datasets not supported. Use pretokenize.py first."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = torch.tensor(self.dataset[idx]["input_ids"], dtype=torch.long)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


# =============================================================================
# Model Creation
# =============================================================================


def create_model(model_type: str, config: TitansConfig) -> nn.Module:
    """Create Titans model."""
    models = {
        "mac": TitansMAC,
        "mag": TitansMAG,
        "mal": TitansMAL,
        "lmm": TitansLMM,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    return models[model_type](config)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Distributed Trainer
# =============================================================================


class DistributedTrainer:
    """Trainer with Accelerate for distributed training."""

    def __init__(
        self,
        model: nn.Module,
        config: DistributedTrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ) -> None:
        self.config = config

        # Initialize Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision
            if config.mixed_precision != "none"
            else "no",
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb" if config.wandb and HAS_WANDB else None,
        )

        # Model, optimizer, scheduler, dataloaders
        self.model = model

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                logger.warning("Model does not support gradient checkpointing")

        # Optimizer (use fused=True for faster GPU execution)
        optimizer_kwargs = {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "betas": (0.9, 0.95),
        }
        if torch.cuda.is_available():
            optimizer_kwargs["fused"] = True
            logger.info("Using fused AdamW optimizer (GPU-accelerated)")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_kwargs,
        )

        # Calculate total steps
        if config.max_steps > 0:
            self.total_steps = config.max_steps
        else:
            num_batches = len(train_dataloader)
            num_update_steps = num_batches // config.gradient_accumulation_steps
            self.total_steps = num_update_steps * config.epochs

        # Scheduler
        num_warmup_steps = int(self.total_steps * config.warmup_ratio)
        self.scheduler = self._get_cosine_schedule(num_warmup_steps, self.total_steps)

        # Prepare with Accelerate
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            train_dataloader,
            self.scheduler,
        )

        self.val_dataloader = val_dataloader
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        if config.wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=vars(config),
                init_kwargs={"wandb": {"name": config.wandb_run_name}},
            )

    def _get_cosine_schedule(
        self,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine schedule with warmup."""

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        self.model.train()

        with self.accelerator.accumulate(self.model):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            logits, _ = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.config.grad_clip > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        return {"loss": loss.item(), "ppl": math.exp(loss.item())}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(
            self.val_dataloader,
            desc="Evaluating",
            leave=False,
            disable=not self.accelerator.is_main_process,
        ):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            logits, _ = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            batch_tokens = labels.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        # Gather across processes
        total_loss = torch.tensor([total_loss], device=self.accelerator.device)
        total_tokens = torch.tensor([total_tokens], device=self.accelerator.device)

        total_loss = self.accelerator.gather(total_loss).sum().item()
        total_tokens = self.accelerator.gather(total_tokens).sum().item()

        self.model.train()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return {"val_loss": avg_loss, "val_ppl": math.exp(avg_loss)}

    def save_checkpoint(self, name: str = "checkpoint") -> None:
        """Save checkpoint."""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            path = self.checkpoint_dir / f"{name}"

            # Unwrap model
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # Save
            self.accelerator.save_state(str(path))

            # Also save model weights separately for inference
            torch.save(
                {
                    "model_state_dict": unwrapped_model.state_dict(),
                    "config": vars(self.config),
                    "model_type": self.config.model_type,
                    "global_step": self.global_step,
                },
                path / "model.pt",
            )

            logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load checkpoint."""
        self.accelerator.load_state(str(path))
        logger.info(f"Loaded checkpoint from {path}")

    def train(self) -> None:
        """Main training loop."""
        self.model.train()

        running_loss = 0.0
        running_count = 0

        start_time = time.time()
        pbar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )

        while self.global_step < self.total_steps:
            self.epoch += 1

            for batch in self.train_dataloader:
                if self.global_step >= self.total_steps:
                    break

                metrics = self.train_step(batch)

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    running_loss += metrics["loss"]
                    running_count += 1

                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        avg_loss = running_loss / running_count
                        current_lr = self.scheduler.get_last_lr()[0]

                        if self.accelerator.is_main_process:
                            pbar.set_postfix(
                                {
                                    "loss": f"{avg_loss:.4f}",
                                    "ppl": f"{math.exp(avg_loss):.2f}",
                                    "lr": f"{current_lr:.2e}",
                                }
                            )

                            self.accelerator.log(
                                {
                                    "train/loss": avg_loss,
                                    "train/ppl": math.exp(avg_loss),
                                    "train/lr": current_lr,
                                },
                                step=self.global_step,
                            )

                        running_loss = 0.0
                        running_count = 0

                    # Evaluation
                    if (
                        self.config.eval_every > 0
                        and self.global_step % self.config.eval_every == 0
                    ):
                        val_metrics = self.evaluate()
                        if val_metrics and self.accelerator.is_main_process:
                            logger.info(
                                f"Step {self.global_step}: "
                                f"val_loss={val_metrics['val_loss']:.4f}, "
                                f"val_ppl={val_metrics['val_ppl']:.2f}"
                            )
                            self.accelerator.log(val_metrics, step=self.global_step)

                            if val_metrics["val_loss"] < self.best_val_loss:
                                self.best_val_loss = val_metrics["val_loss"]
                                self.save_checkpoint("best_model")

                    # Checkpoint
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
        if self.accelerator.is_main_process:
            logger.info(f"Training completed in {elapsed / 3600:.2f} hours")
            logger.info(f"Total steps: {self.global_step}")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        self.accelerator.end_training()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    if not HAS_ACCELERATE:
        raise ImportError(
            "accelerate required for distributed training. "
            "Install with: pip install accelerate"
        )

    parser = argparse.ArgumentParser(description="Distributed pretraining for Titans")

    # Model
    parser.add_argument(
        "--model", type=str, default="mac", choices=["mac", "mag", "mal", "lmm"]
    )
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=32000)

    # Data
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-subset", type=str, default=None)
    parser.add_argument("--local-dataset", type=str, default=None, help="Pre-tokenized local dataset (Arrow format)")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--seq-len", type=int, default=4096)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)

    # Precision
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"]
    )

    # Distributed
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)

    # Logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="titans-distributed")
    parser.add_argument("--wandb-run-name", type=str, default=None)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create config
    config = DistributedTrainingConfig(
        model_type=args.model,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        local_dataset=args.local_dataset,
        data_path=args.data,
        tokenizer=args.tokenizer,
        seq_len=args.seq_len,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        mixed_precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing,
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
    )

    # Load tokenizer
    tokenizer = None
    if config.tokenizer and HAS_TRANSFORMERS:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        config.vocab_size = tokenizer.vocab_size

    # Create dataset (prioritize local pre-tokenized dataset)
    if config.local_dataset and HAS_DATASETS:
        train_dataset = LocalHFDataset(config.local_dataset, config.seq_len)
    elif config.dataset and HAS_DATASETS:
        train_dataset = StreamingDataset(
            config.dataset,
            tokenizer,
            config.seq_len,
            config.dataset_subset,
        )
    elif config.data_path:
        if tokenizer:
            train_dataset = TextFileDataset(
                Path(config.data_path), tokenizer, config.seq_len
            )
        else:
            from scripts.pretrain import CharLevelDataset

            train_dataset = CharLevelDataset(
                Path(config.data_path), config.vocab_size, config.seq_len
            )
    else:
        logger.info("Using synthetic data")
        train_dataset = SyntheticDataset(
            config.vocab_size, config.seq_len, config.synthetic_samples, config.seed
        )

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
        if not isinstance(train_dataset, IterableDataset)
        else 0,
        pin_memory=True,
        shuffle=not isinstance(train_dataset, IterableDataset),
    )

    # Create model config
    titans_config = TitansConfig(
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=config.vocab_size,
        chunk_size=config.chunk_size,
        window_size=config.window_size,
        num_persistent_tokens=config.num_persistent_tokens,
        num_memory_layers=config.num_memory_layers,
    )

    # Create model
    model = create_model(config.model_type, titans_config)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model: {config.model_type.upper()}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
    )

    # Resume if needed
    if config.resume:
        trainer.load_checkpoint(Path(config.resume))

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
