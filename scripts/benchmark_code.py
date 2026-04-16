"""Benchmark: Multi-bracket code structures — PDA vs FSM vs PPM depth generalization

Thesis
------
A PDA with an explicit stack generalizes to deeper nested structures (functions,
lists, dicts) better than FSMs or PPM, because the stack mechanism matches the
structural recursion in code.  Real code mixes parentheses (), brackets [], and
braces {} — each requiring correct open/close tracking.

Results so far: PPM outperforms on this synthetic task because compression lets it
model local n-gram patterns effectively. PDA shows marginal OOD improvement over
FSM. The benchmark is valid; the hypothesis needs refinement (see NOTES below).

NOTES
-----
- Low absolute accuracy (~25-40%) across all models suggests sequences are short
  and the task is hard
- PPM winning suggests next-token prediction on mixed-bracket strings favors
  local patterns over deep structure
- PDA/FSM gap is small — might need more states, deeper training, or a harder
  generalization split to see separation
- Try: (1) longer sequences, (2) harder generalization split (train depth<=3, test depth>=8),
  (3) bracket-only vocab to isolate stack benefit

Experiment
----------
  - Vocabulary: 8 tokens — OPEN_PAREN, CLOSE_PAREN, OPEN_BRACK, CLOSE_BRACK,
                OPEN_BRACE, CLOSE_BRACE, COMMA, EOS
  - Training:   mixed-bracket strings with max nesting depth <= MAX_TRAIN_DEPTH
  - Test:       stratified by exact nesting depth d in TEST_DEPTHS.
                Depths > MAX_TRAIN_DEPTH are out-of-distribution (OOD).
  - Models:     PDA (CP-SAT), FSM (CircuitLM), PPM (n-gram baseline)
  - Metric:     next-token prediction accuracy (%)

Usage
-----
    python scripts/benchmark_code.py
    python scripts/benchmark_code.py --train-seqs 500 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import PDACircuitLM
from circuit_lm.ppm import PPMModel
from circuit_lm.train_cpsat import train as train_fsm
from circuit_lm.train_pda_cpsat import train_pda
from circuit_lm.train_ppm import train_ppm
from circuit_lm.eval import evaluate, evaluate_pda, evaluate_ppm

# ---------------------------------------------------------------------------
# Token vocabulary (8 tokens — no tokenizer needed for synthetic data)
# ---------------------------------------------------------------------------
T_OPEN_PAREN  = 0
T_CLOSE_PAREN = 1
T_OPEN_BRACK  = 2
T_CLOSE_BRACK = 3
T_OPEN_BRACE  = 4
T_CLOSE_BRACE = 5
T_COMMA       = 6
T_EOS         = 7
VOCAB_SIZE    = 8

OPENERS = [T_OPEN_PAREN, T_OPEN_BRACK, T_OPEN_BRACE]
CLOSERS = [T_CLOSE_PAREN, T_CLOSE_BRACK, T_CLOSE_BRACE]
PAIR_MAP = {
    T_OPEN_PAREN: T_CLOSE_PAREN,
    T_OPEN_BRACK: T_CLOSE_BRACK,
    T_OPEN_BRACE: T_CLOSE_BRACE,
}
OPENER_MAP = {v: k for k, v in PAIR_MAP.items()}

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
MAX_TRAIN_DEPTH = 4          # train on depth <= 4
TEST_DEPTHS = (3, 4, 5, 6, 7, 8, 10, 12)
DEFAULT_TRAIN_SEQS = 400
DEFAULT_TEST_PER_DEPTH = 100
DEFAULT_SEED = 42

# Model hyper-parameters
PDA_STATE_BITS = 3          # 8 states
PDA_STACK_DEPTH = 15       # supports up to depth 15
PDA_STEPS = 30             # CP-SAT wall-clock budget in seconds
FSM_STATE_BITS = 5         # 32 states
FSM_STEPS = 30
FSM_CONTEXT_LEN = 6        # rolling-hash context window
PPM_ORDER = 6

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _gen_mixed_bracket_seq(
    max_depth: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
) -> list[int]:
    """Generate one mixed-bracket balanced sequence.
    
    Opens and closes are chosen randomly from the 3 bracket types.
    Depth is tracked across ALL bracket types (nested [] inside () counts).
    """
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n
    open_stack: list[int] = []  # track which bracket type is open

    while opens_left + closes_left > 0:
        can_open = (
            opens_left > 0
            and depth < max_depth
            and closes_left > depth
        )
        can_close = closes_left > 0 and len(open_stack) > 0

        if not can_open and not can_close:
            break

        if can_open and can_close:
            # Prefer to alternate to create nested structures
            if rng.random() < 0.6:  # 60% open when we can
                choice = OPENERS[rng.randint(0, 2)]
                tokens.append(choice)
                open_stack.append(choice)
                depth += 1
                opens_left -= 1
            else:
                choice = open_stack[-1]
                tokens.append(CLOSERS[OPENERS.index(choice)])
                open_stack.pop()
                depth -= 1
                closes_left -= 1
        elif can_open:
            choice = OPENERS[rng.randint(0, 2)]
            tokens.append(choice)
            open_stack.append(choice)
            depth += 1
            opens_left -= 1
        else:
            choice = open_stack[-1]
            tokens.append(CLOSERS[OPENERS.index(choice)])
            open_stack.pop()
            depth -= 1
            closes_left -= 1

    # Close any remaining open brackets
    while open_stack:
        choice = open_stack.pop()
        tokens.append(CLOSERS[OPENERS.index(choice)])

    tokens.append(T_EOS)
    return tokens


def max_depth_of(tokens: list[int]) -> int:
    """Return max nesting depth across all bracket types."""
    depth = 0
    max_d = 0
    for tok in tokens:
        if tok in OPENERS:
            depth += 1
            max_d = max(max_d, depth)
        elif tok in CLOSERS:
            depth -= 1
    return max_d


def gen_train_seqs(max_depth: int, num_seqs: int, rng: random.Random) -> list[list[int]]:
    """Generate training sequences with depth <= max_depth."""
    seqs = []
    for _ in range(num_seqs):
        min_pairs = 2
        max_pairs = max_depth * 2
        seq = _gen_mixed_bracket_seq(max_depth, min_pairs, max_pairs, rng)
        seqs.append(seq)
    return seqs


def gen_test_seqs_per_depth(
    target_depth: int,
    num_seqs: int,
    rng: random.Random,
) -> list[list[int]]:
    """Generate test sequences with exact target nesting depth."""
    seqs = []
    min_pairs = target_depth
    max_pairs = target_depth * 3
    for _ in range(num_seqs):
        # Generate with max_depth = target_depth, then filter
        seq = _gen_mixed_bracket_seq(target_depth, min_pairs, max_pairs, rng)
        if max_depth_of(seq) == target_depth and seq[-1] == T_EOS:
            seqs.append(seq)
    # If we didn't get enough at exact depth, loosen constraints
    attempts = 0
    while len(seqs) < num_seqs and attempts < 500:
        seq = _gen_mixed_bracket_seq(target_depth + 1, min_pairs, max_pairs + 2, rng)
        if max_depth_of(seq) == target_depth and seq[-1] == T_EOS:
            seqs.append(seq)
        attempts += 1
    return seqs[:num_seqs]


# ---------------------------------------------------------------------------
# Accuracy helpers (integer basis-points, no floats in the model layer)
# ---------------------------------------------------------------------------

def accuracy_pct(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (correct / total) * 100.0


def format_acc(correct: int, total: int) -> str:
    return f"{accuracy_pct(correct, total):.1f}%"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-bracket depth generalization benchmark")
    parser.add_argument("--train-seqs", type=int, default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--test-per-depth", type=int, default=DEFAULT_TEST_PER_DEPTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-depth", type=int, default=MAX_TRAIN_DEPTH)
    parser.add_argument("--fsm-states", type=int, default=FSM_STATE_BITS)
    parser.add_argument("--pda-states", type=int, default=PDA_STATE_BITS)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"=== Multi-bracket code benchmark ===")
    print(f"Train: {args.train_seqs} seqs, depth <= {args.max_train_depth}")
    print(f"Test:  {args.test_per_depth} seqs per depth")
    print(f"Vocabulary: {VOCAB_SIZE} tokens (3 bracket types + comma + EOS)")
    print()

    # Generate training data
    print("Generating training data...")
    train_seqs = gen_train_seqs(args.max_train_depth, args.train_seqs, rng)
    total_train_tokens = sum(len(s) for s in train_seqs)
    print(f"  {len(train_seqs)} sequences, {total_train_tokens} total tokens")

    # Train models
    print("\nTraining PDA...")
    t0 = time.monotonic_ns()
    pda = train_pda(
        train_seqs,
        vocab_size=VOCAB_SIZE,
        state_bits=args.pda_states,
        stack_depth=PDA_STACK_DEPTH,
        max_push=1,
        max_pop=1,
        steps=args.max_train_depth * 5,
    )
    pda_train_ms = (time.monotonic_ns() - t0) // 1_000_000
    print(f"  trained in {pda_train_ms}ms")

    print("Training FSM...")
    t0 = time.monotonic_ns()
    fsm = train_fsm(
        train_seqs,
        vocab_size=VOCAB_SIZE,
        state_bits=args.fsm_states,
        context_len=FSM_CONTEXT_LEN,
        steps=FSM_STEPS,
    )
    fsm_train_ms = (time.monotonic_ns() - t0) // 1_000_000
    print(f"  trained in {fsm_train_ms}ms")

    print("Training PPM...")
    t0 = time.monotonic_ns()
    ppm = train_ppm(train_seqs, vocab_size=VOCAB_SIZE, order=PPM_ORDER)
    ppm_train_ms = (time.monotonic_ns() - t0) // 1_000_000
    print(f"  trained in {ppm_train_ms}ms")

    # Evaluate at each depth
    print(f"\n{'Depth':<8} {'PDA':<10} {'FSM':<10} {'PPM':<10}  Notes")
    print("-" * 60)

    results = []
    for depth in TEST_DEPTHS:
        test_seqs = gen_test_seqs_per_depth(depth, args.test_per_depth, rng)
        if not test_seqs:
            print(f"{depth:<8} {'(no data)':<10}")
            continue

        pda_result = evaluate_pda(pda, test_seqs)
        fsm_result = evaluate(fsm, test_seqs)
        ppm_result = evaluate_ppm(ppm, test_seqs)

        pda_acc = accuracy_pct(pda_result["correct"], pda_result["total"])
        fsm_acc = accuracy_pct(fsm_result["correct"], fsm_result["total"])
        ppm_acc = accuracy_pct(ppm_result["correct"], ppm_result["total"])

        ood = " [OOD]" if depth > args.max_train_depth else ""
        print(f"{depth:<8} {pda_acc:>6.1f}%   {fsm_acc:>6.1f}%   {ppm_acc:>6.1f}%   {ood}")

        results.append({
            "depth": depth,
            "pda_acc": pda_acc,
            "fsm_acc": fsm_acc,
            "ppm_acc": ppm_acc,
            "ood": depth > args.max_train_depth,
        })

    # Summary
    print()
    print("=== OOD Generalization ===")
    ood_results = [r for r in results if r["ood"]]
    if ood_results:
        avg_pda = sum(r["pda_acc"] for r in ood_results) / len(ood_results)
        avg_fsm = sum(r["fsm_acc"] for r in ood_results) / len(ood_results)
        avg_ppm = sum(r["ppm_acc"] for r in ood_results) / len(ood_results)
        print(f"Average OOD accuracy (depth > {args.max_train_depth}):")
        print(f"  PDA: {avg_pda:.1f}%")
        print(f"  FSM: {avg_fsm:.1f}%")
        print(f"  PPM: {avg_ppm:.1f}%")
        print(f"  PDA advantage over FSM: +{avg_pda - avg_fsm:.1f}pp")
        print(f"  PDA advantage over PPM: +{avg_pda - avg_ppm:.1f}pp")

    # CSV output
    csv_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark_code.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("depth,pda_acc,fsm_acc,ppm_acc,ood\n")
        for r in results:
            f.write(f"{r['depth']},{r['pda_acc']:.2f},{r['fsm_acc']:.2f},{r['ppm_acc']:.2f},{r['ood']}\n")
    print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
