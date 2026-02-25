"""Run a small integer-only benchmark matrix for circuit_lm.

Usage
-----
    py -3 scripts/benchmark_matrix.py

Purpose
-------
- Produces a repeatable train/eval benchmark table across a small grid.
- Uses only integer timing (`time.monotonic_ns`) and integer metrics.
- Helps track regressions in speed/accuracy while CP-SAT training evolves.
"""

from __future__ import annotations

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


def _ns_to_ms(ns: int) -> int:
    return ns // 1_000_000


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


def main() -> None:
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

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    ) as fh:
        fh.write(BENCHMARK_TEXT)
        data_path = fh.name

    for tokenizer_mode in TOKENIZER_MODES:
        for state_bits in STATE_BITS_VALUES:
            for steps in STEPS_VALUES:
                row = _run_one(
                    data_path=data_path,
                    tokenizer_mode=tokenizer_mode,
                    state_bits=state_bits,
                    steps=steps,
                )
                print(
                    f"{row['tokenizer']} | {row['state_bits']} | {row['steps']} | "
                    f"{row['effective_vocab_size']} | {row['num_sequences']} | "
                    f"{row['total_tokens']} | {row['tok_ms']} | {row['train_ms']} | "
                    f"{row['eval_ms']} | {row['correct']} | {row['total']} | "
                    f"{row['accuracy']}"
                )


if __name__ == "__main__":
    main()
