"""Minimal training script: single macro-block H-UoE, stages 1--2, small data.

Real data: put pre-tokenized tokens in --data-dir (tokens.npy with shape (N, L) or *.npy per sequence).
  python -m huoe.scripts.train_minimal --data-dir ./data --seq-len 1024 --batch-size 16 --epochs 3

Dummy data (default):
  python -m huoe.scripts.train_minimal --epochs 2

2×RTX 5000:
  torchrun --nproc_per_node=2 -m huoe.scripts.train_minimal --data-dir ./data --seq-len 1024 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from huoe.model import HUoEModel
from huoe.train import (
    StageConfig,
    train_epoch,
    run_rewire_step,
)

from huoe.scripts.data_utils import PreTokenizedDataset


class DummyTokenDataset(Dataset):
    """Minimal in-memory token dataset for prototyping (replace with real data)."""

    def __init__(self, num_seqs: int = 100, seq_len: int = 256, vocab_size: int = 50257):
        self.data = torch.randint(0, vocab_size, (num_seqs, seq_len))
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.data[i], "labels": self.data[i].clone()}


def main() -> None:
    p = argparse.ArgumentParser(description="Train minimal H-UoE (single macro-block)")
    p.add_argument("--data-dir", type=str, default=None, help="Data dir: tokens.npy (N,L) or *.npy/*.bin per seq")
    p.add_argument("--num-seqs", type=int, default=200, help="Dummy dataset size (ignored if --data-dir set)")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--stage1-epochs", type=int, default=1, help="Warmup without rewire")
    p.add_argument("--rewire-every", type=int, default=100, help="SET rewire every N steps (stage 2)")
    p.add_argument("--output-dir", type=str, default="./huoe_out")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = HUoEModel.minimal_prototype(
        vocab_size=args.vocab_size,
        d_model=512,
        d_h=128,
        num_heads=8,
        num_groups=4,
        num_experts_per_group=4,
        d_ff=1024,
        expert_density=0.1,
        window_size=32,
        dropout=0.1,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    if args.data_dir:
        dataset = PreTokenizedDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
        )
        print(f"Real data: {len(dataset)} sequences from {args.data_dir} (seq_len={args.seq_len})")
    else:
        dataset = DummyTokenDataset(num_seqs=args.num_seqs, seq_len=args.seq_len, vocab_size=args.vocab_size)
        print(f"Dummy data: {len(dataset)} sequences (seq_len={args.seq_len})")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def rewire_cb(m: HUoEModel, frac: float) -> None:
        run_rewire_step(m, prune_frac=frac)  # rewires all SparseLinear layers in model

    stage1 = StageConfig(
        rewire_every_steps=0,
        load_balance_group_weight=0.01,
        load_balance_expert_weight=0.01,
        group_sparsity_weight=0.001,
        stability_weight=0.01,
    )
    stage2 = StageConfig(
        rewire_every_steps=args.rewire_every,
        rewire_prune_frac=0.2,
        load_balance_group_weight=0.01,
        load_balance_expert_weight=0.01,
        group_sparsity_weight=0.001,
        stability_weight=0.01,
    )

    step = 0
    for epoch in range(args.epochs):
        stage = stage2 if epoch >= args.stage1_epochs else stage1
        cb = rewire_cb if stage.rewire_every_steps > 0 else None
        metrics = train_epoch(
            model,
            dataloader,
            optimizer,
            stage,
            device,
            step_offset=step,
            rewire_callback=cb,
        )
        step += len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        with open(Path(args.output_dir) / "metrics.json", "w") as f:
            json.dump({"epoch": epoch + 1, **metrics}, f, indent=2)

    torch.save(model.state_dict(), Path(args.output_dir) / "model.pt")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
