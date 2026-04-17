"""Benchmark: Hard generalization — train shallow, test deep

Purpose
-------
Train on depth <= 3, test on depth 7-12.
Classic out-of-distribution stress test for structural reasoning.

Hypothesis: PDA stack should show clear advantage when
FSM context window can't cover the deep structures.

Results
-------
v1 (200 train, 2-8 pairs, depth<=4): PPM dominates
v2 (400 train, 4-12 pairs, depth<=4): PPM still wins
v3 (400 train, 4-12 pairs, bracket-only): Same as v2
v4 (HARD SPLIT): Train depth<=3, test depth>=7
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
# Bracket-only vocabulary
# ---------------------------------------------------------------------------
T_OPEN_PAREN  = 0
T_CLOSE_PAREN = 1
T_OPEN_BRACK  = 2
T_CLOSE_BRACK = 3
T_OPEN_BRACE  = 4
T_CLOSE_BRACE = 5
T_EOS         = 6
VOCAB_SIZE    = 7

OPENERS = [T_OPEN_PAREN, T_OPEN_BRACK, T_OPEN_BRACE]
CLOSERS = [T_CLOSE_PAREN, T_CLOSE_BRACK, T_CLOSE_BRACE]

# ---------------------------------------------------------------------------
MAX_TRAIN_DEPTH = 3  # HARD SPLIT: train on shallow only
TEST_DEPTHS = (7, 8, 9, 10, 11, 12)  # deep OOD only
DEFAULT_SEED = 42

# Model configs (slightly larger since task is harder)
PDA_STATE_BITS = 4
PDA_STACK_DEPTH = 15
PDA_STEPS = 60
FSM_STATE_BITS = 6
FSM_STEPS = 60
FSM_CONTEXT_LEN = 8
PPM_ORDER = 8


def _gen_mixed_bracket_seq(min_pairs: int, max_pairs: int, rng: random.Random) -> list[int]:
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n
    open_stack: list[int] = []

    while opens_left + closes_left > 0:
        can_open = opens_left > 0 and closes_left > depth
        can_close = closes_left > 0 and len(open_stack) > 0

        if not can_open and not can_close:
            break

        if can_open and can_close:
            choice = OPENERS[rng.randint(0, 2)] if rng.random() < 0.55 else CLOSERS[OPENERS.index(open_stack[-1])]
        elif can_open:
            choice = OPENERS[rng.randint(0, 2)]
        else:
            choice = CLOSERS[OPENERS.index(open_stack[-1])]

        if choice in OPENERS:
            tokens.append(choice)
            open_stack.append(choice)
            depth += 1
            opens_left -= 1
        else:
            tokens.append(choice)
            open_stack.pop()
            depth -= 1
            closes_left -= 1

    while open_stack:
        choice = open_stack.pop()
        tokens.append(CLOSERS[OPENERS.index(choice)])

    tokens.append(T_EOS)
    return tokens


def max_depth_of(tokens: list[int]) -> int:
    depth = 0
    max_d = 0
    for tok in tokens:
        if tok in OPENERS:
            depth += 1
            max_d = max(max_d, depth)
        elif tok in CLOSERS:
            depth -= 1
    return max_d


def gen_train_seqs(max_depth: int, num_seqs: int, rng: random.Random, min_pairs: int = 4, max_pairs: int = 10) -> list[list[int]]:
    seqs = []
    for _ in range(num_seqs):
        while True:
            seq = _gen_mixed_bracket_seq(min_pairs, max_pairs, rng)
            if max_depth_of(seq) <= max_depth and len(seq) >= min_pairs * 2:
                seqs.append(seq)
                break
    return seqs


def gen_test_seqs_per_depth(target_depth: int, num_seqs: int, rng: random.Random, min_pairs: int = 8, max_pairs: int = 16) -> list[list[int]]:
    seqs = []
    for _ in range(num_seqs * 5):
        if len(seqs) >= num_seqs:
            break
        seq = _gen_mixed_bracket_seq(max(target_depth, min_pairs), max_pairs, rng)
        if max_depth_of(seq) == target_depth and seq[-1] == T_EOS:
            seqs.append(seq)
    return seqs[:num_seqs]


def accuracy_pct(correct: int, total: int) -> float:
    return (correct / total) * 100.0 if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Hard split: train depth<=3, test deep (v4)")
    parser.add_argument("--train-seqs", type=int, default=500)
    parser.add_argument("--test-per-depth", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-pairs", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--fsm-states", type=int, default=FSM_STATE_BITS)
    parser.add_argument("--pda-states", type=int, default=PDA_STATE_BITS)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"=== Hard split benchmark v4 (train depth<={MAX_TRAIN_DEPTH}, test depth>=7) ===")
    print(f"Train: {args.train_seqs} seqs, depth <= {MAX_TRAIN_DEPTH}, {args.min_pairs}-{args.max_pairs} pairs")
    print(f"Test:  {args.test_per_depth} seqs per depth (depth 7-12 only)")
    print()

    print("Generating training data...")
    train_seqs = gen_train_seqs(MAX_TRAIN_DEPTH, args.train_seqs, rng, args.min_pairs, args.max_pairs)
    total_train_tokens = sum(len(s) for s in train_seqs)
    avg_len = total_train_tokens / len(train_seqs)
    max_train_depth_seen = max(max_depth_of(s) for s in train_seqs)
    print(f"  {len(train_seqs)} sequences, {total_train_tokens} tokens, avg len={avg_len:.1f}")
    print(f"  max depth in training: {max_train_depth_seen}")

    print("\nTraining PDA...")
    t0 = time.monotonic_ns()
    pda = train_pda(train_seqs, vocab_size=VOCAB_SIZE, state_bits=args.pda_states, stack_depth=PDA_STACK_DEPTH, max_push=1, max_pop=1, steps=PDA_STEPS)
    print(f"  trained in {(time.monotonic_ns() - t0) // 1_000_000}ms")

    print("Training FSM...")
    t0 = time.monotonic_ns()
    fsm = train_fsm(train_seqs, vocab_size=VOCAB_SIZE, state_bits=args.fsm_states, context_len=FSM_CONTEXT_LEN, steps=FSM_STEPS)
    print(f"  trained in {(time.monotonic_ns() - t0) // 1_000_000}ms")

    print("Training PPM...")
    t0 = time.monotonic_ns()
    ppm = train_ppm(train_seqs, vocab_size=VOCAB_SIZE, order=PPM_ORDER)
    print(f"  trained in {(time.monotonic_ns() - t0) // 1_000_000}ms")

    print(f"\n{'Depth':<8} {'PDA':<10} {'FSM':<10} {'PPM':<10}  Notes")
    print("-" * 60)

    results = []
    for depth in TEST_DEPTHS:
        test_seqs = gen_test_seqs_per_depth(depth, args.test_per_depth, rng)
        if not test_seqs:
            continue

        pda_result = evaluate_pda(pda, test_seqs)
        fsm_result = evaluate(fsm, test_seqs)
        ppm_result = evaluate_ppm(ppm, test_seqs)

        pda_acc = accuracy_pct(pda_result["correct"], pda_result["total"])
        fsm_acc = accuracy_pct(fsm_result["correct"], fsm_result["total"])
        ppm_acc = accuracy_pct(ppm_result["correct"], ppm_result["total"])

        print(f"{depth:<8} {pda_acc:>6.1f}%   {fsm_acc:>6.1f}%   {ppm_acc:>6.1f}%   [OOD depth>{MAX_TRAIN_DEPTH}]")
        results.append({"depth": depth, "pda_acc": pda_acc, "fsm_acc": fsm_acc, "ppm_acc": ppm_acc, "ood": True})

    print()
    print("=== Hard OOD Generalization (all depths > 3) ===")
    avg_pda = sum(r["pda_acc"] for r in results) / len(results)
    avg_fsm = sum(r["fsm_acc"] for r in results) / len(results)
    avg_ppm = sum(r["ppm_acc"] for r in results) / len(results)
    print(f"Average accuracy (depth 7-12):")
    print(f"  PDA: {avg_pda:.1f}%")
    print(f"  FSM: {avg_fsm:.1f}%")
    print(f"  PPM: {avg_ppm:.1f}%")
    print(f"  PDA vs FSM: {avg_pda - avg_fsm:+.1f}pp")
    print(f"  PDA vs PPM: {avg_pda - avg_ppm:+.1f}pp")

    if avg_pda > avg_fsm and avg_pda > avg_ppm:
        print("\n*** PDA WINS ***")
    elif avg_fsm > avg_ppm:
        print("\n*** FSM wins over PDA ***")
    else:
        print("\n*** PPM still wins — stack provides no advantage on this task ***")

    csv_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark_code_v4_hard_split.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("depth,pda_acc,fsm_acc,ppm_acc,ood\n")
        for r in results:
            f.write(f"{r['depth']},{r['pda_acc']:.2f},{r['fsm_acc']:.2f},{r['ppm_acc']:.2f},{r['ood']}\n")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()