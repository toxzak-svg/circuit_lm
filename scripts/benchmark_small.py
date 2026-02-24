"""Benchmark circuit_lm on a small synthetic dataset.

Usage
-----
    python scripts/benchmark_small.py

No external dependencies beyond OR-Tools.  No floats.
Timing uses time.monotonic_ns() which returns an integer (nanoseconds).
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

# Allow running directly from the repo root without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.data import load_sequences
from circuit_lm.eval import evaluate
from circuit_lm.infer import greedy_decode, sample_tokens
from circuit_lm.metrics import format_accuracy
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat import train

# ---------------------------------------------------------------------------
# Benchmark configuration (all integers)
# ---------------------------------------------------------------------------

BENCHMARK_TEXT: str = (
    "the quick brown fox jumps over the lazy dog. " * 30
    + "hello world hello world hello world. " * 20
    + "abcdef ghijklm nopqrs tuvwxyz. " * 10
    + "circuit language model integer only no floats. " * 15
)

VOCAB_SIZE:   int = 64
STATE_BITS:   int = 3   # 8 states
STEPS:        int = 5   # CP-SAT time limit in seconds
SAMPLE_SEED:  int = 42
SAMPLE_LEN:   int = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns_to_ms(ns: int) -> int:
    """Convert nanoseconds (int) to milliseconds (int) via integer division."""
    return ns // 1_000_000


def _section(title: str) -> None:
    sep = "-" * 50
    print(f"\n{sep}\n{title}\n{sep}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    _section("circuit_lm small benchmark")
    print(f"text_len={len(BENCHMARK_TEXT)} chars")
    print(f"vocab_size={VOCAB_SIZE}  state_bits={STATE_BITS}  steps={STEPS}")

    # ------------------------------------------------------------------
    # Tokenise
    # ------------------------------------------------------------------
    _section("Tokenise")
    t0 = time.monotonic_ns()
    tok = Tokenizer.from_text(BENCHMARK_TEXT, vocab_size=VOCAB_SIZE)
    t1 = time.monotonic_ns()
    print(f"effective vocab_size={tok.vocab_size}  time={_ns_to_ms(t1 - t0)}ms")

    # Write to a temp file so load_sequences can read it
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(BENCHMARK_TEXT)
        tmp_path = fh.name

    sequences = load_sequences(tmp_path, tok)
    total_tokens = sum(len(s) for s in sequences)
    print(f"sequences={len(sequences)}  total_tokens={total_tokens}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    _section("Train (CP-SAT)")
    t0 = time.monotonic_ns()
    model = train(
        sequences=sequences,
        vocab_size=tok.vocab_size,
        state_bits=STATE_BITS,
        steps=STEPS,
    )
    t1 = time.monotonic_ns()
    print(f"num_states={model.num_states}  train_time={_ns_to_ms(t1 - t0)}ms")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    _section("Evaluate")
    t0 = time.monotonic_ns()
    results = evaluate(model, sequences)
    t1 = time.monotonic_ns()
    correct = results["correct"]
    total   = results["total"]
    print(
        f"correct={correct}  total={total}  "
        f"accuracy={format_accuracy(correct, total)}  "
        f"time={_ns_to_ms(t1 - t0)}ms"
    )

    # ------------------------------------------------------------------
    # Sample (greedy)
    # ------------------------------------------------------------------
    _section("Sample – greedy")
    prompt_str = "the quick"
    prompt_ids = tok.encode(prompt_str)
    t0 = time.monotonic_ns()
    greedy_ids = greedy_decode(model, prompt_ids, max_tokens=SAMPLE_LEN)
    t1 = time.monotonic_ns()
    generated_greedy = tok.decode(greedy_ids[len(prompt_ids):])
    print(f"prompt:    {prompt_str!r}")
    print(f"generated: {generated_greedy!r}")
    print(f"time={_ns_to_ms(t1 - t0)}ms")

    # ------------------------------------------------------------------
    # Sample (stochastic, integer-weighted)
    # ------------------------------------------------------------------
    _section("Sample – stochastic (integer-weighted, seed=" + str(SAMPLE_SEED) + ")")
    t0 = time.monotonic_ns()
    sampled_ids = sample_tokens(model, prompt_ids, max_tokens=SAMPLE_LEN, seed=SAMPLE_SEED)
    t1 = time.monotonic_ns()
    generated_sampled = tok.decode(sampled_ids[len(prompt_ids):])
    print(f"prompt:    {prompt_str!r}")
    print(f"generated: {generated_sampled!r}")
    print(f"time={_ns_to_ms(t1 - t0)}ms")

    _section("Done")


if __name__ == "__main__":
    main()
