"""Reproduce the PDA depth-generalization experiment.

Thesis
------
A CP-SAT-trained pushdown automaton (PDA) generalizes perfectly to balanced
parentheses deeper than its training distribution, because the stack explicitly
represents the structural feature (nesting depth).  An FSM without a stack
cannot generalise beyond its context window; a PPM model degrades similarly.

Experiment
----------
  - Vocabulary  : OPEN='(', CLOSE=')', EOS  (3 tokens, vocab_size=3)
  - Training    : balanced strings with max nesting depth <= MAX_TRAIN_DEPTH
  - Test        : balanced strings stratified by exact nesting depth d in
                  TEST_DEPTHS.  Depths > MAX_TRAIN_DEPTH are out-of-distribution.
  - Models      : PDA (CP-SAT), FSM (CircuitLM CP-SAT), PPM (n-gram)
  - Metric      : next-token prediction accuracy (integer basis-points)

All arithmetic is integer-only.  Seeds are deterministic.

Usage
-----
    python scripts/reproduce_depth_generalization.py
    python scripts/reproduce_depth_generalization.py --seed 0 --train-seqs 300
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.eval import evaluate, evaluate_pda, evaluate_ppm
from circuit_lm.metrics import format_accuracy, accuracy_pct_times100
from circuit_lm.train_cpsat import train as train_fsm
from circuit_lm.train_pda_cpsat import train_pda
from circuit_lm.train_ppm import train_ppm

# ---------------------------------------------------------------------------
# Token vocabulary (synthetic – no Tokenizer needed)
# ---------------------------------------------------------------------------

OPEN:       int = 0   # '(' – triggers PUSH in a correctly-learned PDA
CLOSE:      int = 1   # ')' – triggers POP in a correctly-learned PDA
EOS:        int = 2   # end-of-sequence sentinel
VOCAB_SIZE: int = 3

# ---------------------------------------------------------------------------
# Experiment parameters (overrideable via CLI)
# ---------------------------------------------------------------------------

MAX_TRAIN_DEPTH: int = 3
TEST_DEPTHS: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
DEFAULT_TRAIN_SEQS: int = 300
DEFAULT_TEST_SEQS_PER_DEPTH: int = 100
DEFAULT_SEED: int = 42

# Model hyper-parameters (fixed for reproducibility)
PDA_STATE_BITS: int = 2       # 4 states
PDA_STACK_DEPTH: int = 10     # supports up to depth 10
PDA_STEPS: int = 20           # CP-SAT wall-clock budget in seconds
PDA_MAX_PUSH: int = 1         # only ONE push token expected ('(')
PDA_MAX_POP: int = 1          # only ONE pop  token expected (')')

FSM_STATE_BITS: int = 4       # 16 states
FSM_STEPS: int = 20           # CP-SAT wall-clock budget in seconds
FSM_CONTEXT_LEN: int = 4      # rolling-hash context window

PPM_ORDER: int = 6            # n-gram order for PPM baseline

# ---------------------------------------------------------------------------
# Balanced-parentheses data generation (integer-only, deterministic seeds)
# ---------------------------------------------------------------------------


def _gen_one_balanced(
    max_depth: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
) -> list[int]:
    """Generate one balanced parentheses sequence of token IDs.

    Returns a flat list:  OPEN/CLOSE tokens ... EOS

    The maximum nesting depth is bounded by *max_depth*.
    The total number of open-close pairs is drawn uniformly from
    [min_pairs, max_pairs].

    The generator uses a greedy walk with lookahead feasibility:
      - ``can_open``  requires opens_left > 0, depth < max_depth,
                       and closes_left > depth  (so we can close all open parens)
      - ``can_close`` requires closes_left > 0 and depth > 0
    """
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n

    while opens_left + closes_left > 0:
        can_open = (
            opens_left > 0
            and depth < max_depth
            and closes_left > depth        # ensures we can close all open parens
        )
        can_close = closes_left > 0 and depth > 0

        if not can_open and not can_close:
            break  # exhausted (balanced and at depth 0)

        if can_open and can_close:
            choice = OPEN if rng.randint(0, 1) == 0 else CLOSE
        elif can_open:
            choice = OPEN
        else:
            choice = CLOSE

        if choice == OPEN:
            tokens.append(OPEN)
            depth += 1
            opens_left -= 1
        else:
            tokens.append(CLOSE)
            depth -= 1
            closes_left -= 1

    tokens.append(EOS)
    return tokens


def _max_depth_of(tokens: list[int]) -> int:
    """Return the maximum nesting depth reached by a token sequence."""
    depth = 0
    max_d = 0
    for tok in tokens:
        if tok == OPEN:
            depth += 1
            if depth > max_d:
                max_d = depth
        elif tok == CLOSE:
            depth -= 1
    return max_d


def gen_train_seqs(
    max_depth: int,
    num_seqs: int,
    seed: int,
    min_pairs: int = 2,
    max_pairs: int = 8,
) -> list[list[int]]:
    """Generate *num_seqs* balanced sequences with max nesting depth <= *max_depth*."""
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    while len(seqs) < num_seqs:
        seq = _gen_one_balanced(max_depth, min_pairs, max_pairs, rng)
        seqs.append(seq)
    return seqs


def gen_test_seqs_at_depth(
    target_depth: int,
    num_seqs: int,
    seed: int,
    min_pairs: int | None = None,
    max_pairs: int = 16,
) -> list[list[int]]:
    """Generate balanced sequences where max nesting depth == *target_depth*.

    Uses rejection sampling: generate with max_depth=target_depth, keep only
    sequences that actually reach that depth.

    Args:
        target_depth: Exact nesting depth required.
        num_seqs:     Number of sequences to produce.
        seed:         RNG seed for reproducibility.
        min_pairs:    Minimum bracket pairs per sequence (defaults to target_depth,
                      ensuring the sequence is long enough to reach the depth).
        max_pairs:    Maximum bracket pairs per sequence.
    """
    if min_pairs is None:
        min_pairs = target_depth
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    max_attempts = num_seqs * 10_000
    attempts = 0
    while len(seqs) < num_seqs and attempts < max_attempts:
        seq = _gen_one_balanced(target_depth, min_pairs, max_pairs, rng)
        if _max_depth_of(seq) == target_depth:
            seqs.append(seq)
        attempts += 1
    return seqs


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _eval_pda_on_seqs(model, seqs: list[list[int]]) -> tuple[int, int]:
    r = evaluate_pda(model, seqs)
    return r["correct"], r["total"]  # type: ignore[return-value]


def _eval_fsm_on_seqs(model, seqs: list[list[int]]) -> tuple[int, int]:
    r = evaluate(model, seqs)
    return r["correct"], r["total"]  # type: ignore[return-value]


def _eval_ppm_on_seqs(model, seqs: list[list[int]]) -> tuple[int, int]:
    r = evaluate_ppm(model, seqs)
    return r["correct"], r["total"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Printing helpers (integer-only)
# ---------------------------------------------------------------------------

_SEP = "-" * 72
_HDR = f"{'depth':>7}  {'PDA':>10}  {'FSM':>10}  {'PPM':>10}  {'seqs':>6}"


def _row(depth: int, pda: tuple[int, int], fsm: tuple[int, int], ppm: tuple[int, int], n: int) -> str:
    ood = "*" if depth > MAX_TRAIN_DEPTH else " "
    return (
        f"{depth:>6}{ood}"
        f"  {format_accuracy(*pda):>10}"
        f"  {format_accuracy(*fsm):>10}"
        f"  {format_accuracy(*ppm):>10}"
        f"  {n:>6}"
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run(
    seed: int,
    train_seqs: int,
    test_seqs_per_depth: int,
    quiet: bool = False,
) -> dict[int, dict[str, tuple[int, int]]]:
    """Run the full depth-generalisation experiment.

    Returns:
        dict mapping depth -> {"pda": (correct, total), "fsm": ..., "ppm": ...}
    """
    if not quiet:
        print()
        print("=== PDA Depth Generalisation Experiment ===")
        print(f"  train_depth_max={MAX_TRAIN_DEPTH}  seed={seed}")
        print(f"  train_seqs={train_seqs}  test_seqs_per_depth={test_seqs_per_depth}")
        print(f"  vocab_size={VOCAB_SIZE}  tokens: OPEN={OPEN} CLOSE={CLOSE} EOS={EOS}")
        print()

    # ------------------------------------------------------------------
    # 1. Generate training data
    # ------------------------------------------------------------------
    if not quiet:
        print(f"Generating {train_seqs} training sequences (depth <= {MAX_TRAIN_DEPTH}) ...")
    train_data = gen_train_seqs(
        max_depth=MAX_TRAIN_DEPTH,
        num_seqs=train_seqs,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 2. Train models
    # ------------------------------------------------------------------
    if not quiet:
        print(f"Training PDA  (state_bits={PDA_STATE_BITS}, stack_depth={PDA_STACK_DEPTH}, steps={PDA_STEPS}s) ...")
    pda_model = train_pda(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        state_bits=PDA_STATE_BITS,
        stack_depth=PDA_STACK_DEPTH,
        steps=PDA_STEPS,
        max_push=PDA_MAX_PUSH,
        max_pop=PDA_MAX_POP,
        top_k_coverage=VOCAB_SIZE,
    )

    if not quiet:
        print(f"  push_tokens={sorted(pda_model.push_tokens)}  pop_tokens={sorted(pda_model.pop_tokens)}")
        print(f"Training FSM  (state_bits={FSM_STATE_BITS}, context_len={FSM_CONTEXT_LEN}, steps={FSM_STEPS}s) ...")
    fsm_model = train_fsm(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        state_bits=FSM_STATE_BITS,
        steps=FSM_STEPS,
        context_len=FSM_CONTEXT_LEN,
    )

    if not quiet:
        print(f"Training PPM  (order={PPM_ORDER}) ...")
    ppm_model = train_ppm(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        order=PPM_ORDER,
    )

    # ------------------------------------------------------------------
    # 3. Evaluate at each depth
    # ------------------------------------------------------------------
    if not quiet:
        print()
        print("Evaluating ...")
        print(_SEP)
        print(_HDR)
        print(_SEP)
        print(f"  (* = out-of-distribution depth, not seen during training)")

    results: dict[int, dict[str, tuple[int, int]]] = {}

    for depth in TEST_DEPTHS:
        test_data = gen_test_seqs_at_depth(
            target_depth=depth,
            num_seqs=test_seqs_per_depth,
            seed=seed + depth,
        )
        n = len(test_data)
        pda_res = _eval_pda_on_seqs(pda_model, test_data)
        fsm_res = _eval_fsm_on_seqs(fsm_model, test_data)
        ppm_res = _eval_ppm_on_seqs(ppm_model, test_data)

        results[depth] = {"pda": pda_res, "fsm": fsm_res, "ppm": ppm_res}

        if not quiet:
            print(_row(depth, pda_res, fsm_res, ppm_res, n))

    if not quiet:
        print(_SEP)
        print()
        print("Basis-points table (10000 = 100%):  PDA / FSM / PPM")
        for depth in TEST_DEPTHS:
            r = results[depth]
            pda_bp = accuracy_pct_times100(*r["pda"])
            fsm_bp = accuracy_pct_times100(*r["fsm"])
            ppm_bp = accuracy_pct_times100(*r["ppm"])
            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            print(f"  depth {depth}{ood}: PDA={pda_bp:5d}  FSM={fsm_bp:5d}  PPM={ppm_bp:5d}")
        print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Integer RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--train-seqs",
        type=int,
        default=DEFAULT_TRAIN_SEQS,
        help=f"Number of training sequences (default: {DEFAULT_TRAIN_SEQS})",
    )
    parser.add_argument(
        "--test-seqs-per-depth",
        type=int,
        default=DEFAULT_TEST_SEQS_PER_DEPTH,
        help=f"Test sequences per depth band (default: {DEFAULT_TEST_SEQS_PER_DEPTH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages; print only the results table.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run(
        seed=args.seed,
        train_seqs=args.train_seqs,
        test_seqs_per_depth=args.test_seqs_per_depth,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
