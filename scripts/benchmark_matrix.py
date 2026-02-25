"""Run a small integer-only benchmark matrix for circuit_lm.

Usage
-----
    py -3 scripts/benchmark_matrix.py
    py -3 scripts/benchmark_matrix.py --csv-out bench.csv
    py -3 scripts/benchmark_matrix.py --tsv-out bench.tsv
    py -3 scripts/benchmark_matrix.py --snapshot-dir benchmark_runs

Purpose
-------
- Produces a repeatable train/eval benchmark table across a small grid.
- Uses only integer timing (`time.monotonic_ns`) and integer metrics.
- Helps track regressions in speed/accuracy while CP-SAT training evolves.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import tempfile
import time

# Allow running directly from repo root without installation.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.data import load_sequences
from circuit_lm.eval import evaluate
from circuit_lm.metrics import format_accuracy
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat import train


BENCHMARK_TEXT: str = (
    "the quick brown fox jumps over the lazy dog. " * 30
    + "hello world hello world hello world. " * 20
    + "abcdef ghijklm nopqrs tuvwxyz. " * 10
    + "circuit language model integer only no floats. " * 15
)

VOCAB_SIZE: int = 64
BPE_MERGES: int = 32

# Small default grid so the script stays fast enough for routine checks.
TOKENIZER_MODES: tuple[str, ...] = ("char", "bpe")
STATE_BITS_VALUES: tuple[int, ...] = (2, 3, 4)
STEPS_VALUES: tuple[int, ...] = (1, 2, 5)

COLUMN_NAMES: tuple[str, ...] = (
    "tokenizer",
    "state_bits",
    "steps",
    "effective_vocab_size",
    "num_sequences",
    "total_tokens",
    "tok_ms",
    "train_ms",
    "eval_ms",
    "correct",
    "total",
    "accuracy",
)


def _ns_to_ms(ns: int) -> int:
    return ns // 1_000_000


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a small benchmark matrix and print a table."
    )
    parser.add_argument(
        "--csv-out",
        default="",
        metavar="PATH",
        help="Optional CSV output path for benchmark rows.",
    )
    parser.add_argument(
        "--tsv-out",
        default="",
        metavar="PATH",
        help="Optional TSV output path for benchmark rows.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="",
        metavar="DIR",
        help=(
            "Optional output directory for timestamped CSV/TSV exports. "
            "Writes both formats using an integer timestamp tag."
        ),
    )
    parser.add_argument(
        "--snapshot-prefix",
        default="benchmark_matrix",
        metavar="NAME",
        help="Filename prefix used with --snapshot-dir (default: benchmark_matrix).",
    )
    return parser


def _run_one(
    *,
    data_path: str,
    tokenizer_mode: str,
    state_bits: int,
    steps: int,
) -> dict[str, int | str]:
    tok_t0 = time.monotonic_ns()
    tokenizer = Tokenizer.from_text(
        BENCHMARK_TEXT,
        vocab_size=VOCAB_SIZE,
        mode=tokenizer_mode,
        bpe_merges=BPE_MERGES,
    )
    tok_t1 = time.monotonic_ns()

    sequences = load_sequences(data_path, tokenizer)

    train_t0 = time.monotonic_ns()
    model = train(
        sequences=sequences,
        vocab_size=tokenizer.vocab_size,
        state_bits=state_bits,
        steps=steps,
    )
    train_t1 = time.monotonic_ns()

    eval_t0 = time.monotonic_ns()
    results = evaluate(model, sequences)
    eval_t1 = time.monotonic_ns()

    correct = results["correct"]
    total = results["total"]

    return {
        "tokenizer": tokenizer_mode,
        "state_bits": state_bits,
        "steps": steps,
        "effective_vocab_size": tokenizer.vocab_size,
        "num_sequences": len(sequences),
        "total_tokens": sum(len(seq) for seq in sequences),
        "tok_ms": _ns_to_ms(tok_t1 - tok_t0),
        "train_ms": _ns_to_ms(train_t1 - train_t0),
        "eval_ms": _ns_to_ms(eval_t1 - eval_t0),
        "correct": correct,
        "total": total,
        "accuracy": format_accuracy(correct, total),
    }


def _print_rows(rows: list[dict[str, int | str]]) -> None:
    print("circuit_lm benchmark matrix")
    print(
        "grid:"
        f" tokenizer={TOKENIZER_MODES}"
        f" state_bits={STATE_BITS_VALUES}"
        f" steps={STEPS_VALUES}"
    )
    print(
        "columns:"
        " tokenizer | state_bits | steps | vocab | seqs | tokens |"
        " tok_ms | train_ms | eval_ms | correct | total | accuracy"
    )

    for row in rows:
        print(
            f"{row['tokenizer']} | {row['state_bits']} | {row['steps']} | "
            f"{row['effective_vocab_size']} | {row['num_sequences']} | "
            f"{row['total_tokens']} | {row['tok_ms']} | {row['train_ms']} | "
            f"{row['eval_ms']} | {row['correct']} | {row['total']} | "
            f"{row['accuracy']}"
        )


def _write_delimited(
    path_str: str,
    rows: list[dict[str, int | str]],
    *,
    delimiter: str,
) -> None:
    path = pathlib.Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMN_NAMES), delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {len(rows)} rows -> {path}")


def _timestamp_tag() -> str:
    # Integer-only timestamp string (YYYYMMDD-HHMMSS) for stable file naming.
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _collect_rows(data_path: str) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for tokenizer_mode in TOKENIZER_MODES:
        for state_bits in STATE_BITS_VALUES:
            for steps in STEPS_VALUES:
                rows.append(
                    _run_one(
                        data_path=data_path,
                        tokenizer_mode=tokenizer_mode,
                        state_bits=state_bits,
                        steps=steps,
                    )
                )
    return rows


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(BENCHMARK_TEXT)
        data_path = fh.name

    try:
        rows = _collect_rows(data_path)
        _print_rows(rows)
        if args.csv_out:
            _write_delimited(args.csv_out, rows, delimiter=",")
        if args.tsv_out:
            _write_delimited(args.tsv_out, rows, delimiter="\t")
        if args.snapshot_dir:
            stamp = _timestamp_tag()
            out_dir = pathlib.Path(args.snapshot_dir)
            csv_path = out_dir / f"{args.snapshot_prefix}_{stamp}.csv"
            tsv_path = out_dir / f"{args.snapshot_prefix}_{stamp}.tsv"
            _write_delimited(str(csv_path), rows, delimiter=",")
            _write_delimited(str(tsv_path), rows, delimiter="\t")
    finally:
        tmp_path = pathlib.Path(data_path)
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()
