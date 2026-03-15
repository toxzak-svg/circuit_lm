"""Benchmarks for H-UoE: perplexity, task-switch, and ablations.

Usage:
  # Perplexity on pre-tokenized data
  python -m huoe.scripts.run_benchmark perplexity --checkpoint ./huoe_out/model.pt --data-dir ./data --seq-len 512

  # Task-switch: synthetic alternating segments (report loss per segment type)
  python -m huoe.scripts.run_benchmark task_switch --checkpoint ./huoe_out/model.pt --seq-len 512 --segment-len 256 --num-segments 8

  # Ablation: no controller (zero state, no update)
  python -m huoe.scripts.run_benchmark perplexity --checkpoint ./huoe_out/model.pt --data-dir ./data --ablation no_controller
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from huoe.model import HUoEModel
from huoe.scripts.data_utils import PreTokenizedDataset


def _load_model(checkpoint_path: str | Path, device: torch.device) -> HUoEModel:
    model = HUoEModel.minimal_prototype(
        vocab_size=50257,
        d_model=512,
        d_h=128,
        num_heads=8,
        num_groups=4,
        num_experts_per_group=4,
        d_ff=1024,
        expert_density=0.1,
        window_size=32,
        dropout=0.0,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def run_perplexity(
    model: HUoEModel,
    dataloader: DataLoader,
    device: torch.device,
    ablation_no_controller: bool = False,
) -> float:
    """Compute mean CE loss and return perplexity = exp(loss)."""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        h_controllers = None
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            B, S = input_ids.shape
            if ablation_no_controller:
                # Zero controller, do not use returned state (ablation: no controller)
                h_controllers = [
                    torch.zeros(B, model.d_h, device=device, dtype=model.embed.weight.dtype)
                    for _ in range(model.num_layers)
                ]
            logits, h_list, aux = model(input_ids, h_controllers, labels)
            if not ablation_no_controller:
                h_controllers = [h.detach() for h in h_list]
            loss = aux["loss"]
            if loss is not None:
                n = (labels != -100).sum().item()
                if n > 0:
                    total_loss += loss.item() * n
                    total_tokens += n
    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_loss)).item())


def run_task_switch(
    model: HUoEModel,
    segment_len: int,
    num_segments: int,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
) -> dict[str, float]:
    """Synthetic task-switch: alternate two token distributions every segment_len tokens.
    Returns mean loss and per-segment-type losses (even vs odd segments).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    seq_len = segment_len * num_segments
    # Two "tasks": different token distributions (e.g. low vs high token ids)
    task_a = torch.randint(0, vocab_size // 2, (batch_size, seq_len), device=device, generator=gen)
    task_b = torch.randint(vocab_size // 2, vocab_size, (batch_size, seq_len), device=device, generator=gen)
    input_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    for i in range(num_segments):
        s, e = i * segment_len, (i + 1) * segment_len
        if i % 2 == 0:
            input_ids[:, s:e] = task_a[:, s:e]
        else:
            input_ids[:, s:e] = task_b[:, s:e]
    labels = input_ids.clone()

    model.eval()
    with torch.no_grad():
        logits, _, aux = model(input_ids, None, labels)
        loss = aux["loss"]
        if loss is None:
            ce = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, vocab_size),
                labels[..., 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        else:
            ce = loss
        overall_loss = ce.item()
    # Per-segment loss (approximate: take slice of logits/labels)
    seg_losses = []
    with torch.no_grad():
        for i in range(num_segments):
            s, e = i * segment_len, (i + 1) * segment_len
            if e - 1 <= 0:
                continue
            logits_s = logits[:, s : e - 1].contiguous().view(-1, vocab_size)
            labels_s = labels[:, s + 1 : e].contiguous().view(-1)
            seg_losses.append(
                torch.nn.functional.cross_entropy(logits_s, labels_s, ignore_index=-100).item()
            )
    return {
        "overall_loss": overall_loss,
        "overall_perplexity": float(torch.exp(torch.tensor(overall_loss)).item()),
        "segment_losses": seg_losses,
        "mean_even_segment_loss": sum(seg_losses[0::2]) / max(1, len(seg_losses[0::2])),
        "mean_odd_segment_loss": sum(seg_losses[1::2]) / max(1, len(seg_losses[1::2])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="H-UoE benchmarks: perplexity, task-switch, ablations")
    sub = p.add_subparsers(dest="command", required=True)
    # perplexity
    pp = sub.add_parser("perplexity", help="Perplexity on pre-tokenized data")
    pp.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    pp.add_argument("--data-dir", type=str, required=True, help="Directory with tokens.npy or *.npy")
    pp.add_argument("--seq-len", type=int, default=512)
    pp.add_argument("--batch-size", type=int, default=8)
    pp.add_argument("--ablation", type=str, default="none", choices=["none", "no_controller"], help="Ablation: no_controller = zero state")
    pp.add_argument("--output", type=str, default=None, help="Write metrics JSON here")
    # task_switch
    ts = sub.add_parser("task_switch", help="Synthetic alternating segments")
    ts.add_argument("--checkpoint", type=str, required=True, help="Path to model.pt")
    ts.add_argument("--seq-len", type=int, default=512, help="Total length (segment_len * num_segments)")
    ts.add_argument("--segment-len", type=int, default=256)
    ts.add_argument("--num-segments", type=int, default=8)
    ts.add_argument("--batch-size", type=int, default=4)
    ts.add_argument("--vocab-size", type=int, default=50257)
    ts.add_argument("--output", type=str, default=None)

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "perplexity":
        model = _load_model(args.checkpoint, device)
        dataset = PreTokenizedDataset(data_dir=args.data_dir, seq_len=args.seq_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        ppl = run_perplexity(
            model,
            dataloader,
            device,
            ablation_no_controller=(args.ablation == "no_controller"),
        )
        metrics = {"perplexity": ppl, "ablation": args.ablation}
        print(f"Perplexity: {ppl:.4f}  (ablation={args.ablation})")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
        return

    if args.command == "task_switch":
        model = _load_model(args.checkpoint, device)
        metrics = run_task_switch(
            model,
            segment_len=args.segment_len,
            num_segments=args.num_segments,
            vocab_size=args.vocab_size,
            batch_size=args.batch_size,
            device=device,
        )
        print("Task-switch metrics:", json.dumps(metrics, indent=2))
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(metrics, f, indent=2)
        return

    p.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
