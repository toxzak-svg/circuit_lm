"""Benchmark: Alternating mismatched bracket types

Purpose
-------
Pairs of brackets must match type: () with (), [] with [], {} with {}.
Interleaved different types test whether the stack can track type+depth simultaneously.

Example: ( [ ) ] — the ] can't close the ( 
This requires the PDA to track BOTH bracket type AND depth.
PPM can't do this as cleanly since it only sees local n-grams.

Hypothesis: PDA should WIN this one — stack tracks type+depth, PPM gets confused.
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
VOCAB_SIZE = 7
T_OPEN_PAREN = 0; T_CLOSE_PAREN = 1
T_OPEN_BRACK = 2; T_CLOSE_BRACK = 3
T_OPEN_BRACE = 4; T_CLOSE_BRACE = 5
T_EOS = 6
OPENERS = [T_OPEN_PAREN, T_OPEN_BRACK, T_OPEN_BRACE]
CLOSERS = [T_CLOSE_PAREN, T_CLOSE_BRACK, T_CLOSE_BRACE]
PAIR_MAP = {T_OPEN_PAREN: T_CLOSE_PAREN, T_OPEN_BRACK: T_CLOSE_BRACK, T_OPEN_BRACE: T_CLOSE_BRACE}

MAX_TRAIN_DEPTH = 4
TEST_DEPTHS = (5, 6, 7, 8, 10, 12)
DEFAULT_SEED = 42

PDA_STATE_BITS = 4; PDA_STACK_DEPTH = 15; PDA_STEPS = 60
FSM_STATE_BITS = 6; FSM_STEPS = 60; FSM_CONTEXT_LEN = 8
PPM_ORDER = 8


def _gen_balanced_interleaved(min_pairs: int, max_pairs: int, rng: random.Random) -> list[int]:
    """Generate balanced string where each pair's type is chosen randomly.
    Must close in LIFO order — but types can interleave: ( [ ) ] is invalid.
    Valid sequence: push A, push B, pop B, pop A (types match on close).
    """
    n = rng.randint(min_pairs, max_pairs)
    tokens = []
    stack = []  # stack of opener types
    remaining = n
    
    # Strategy: generate valid close sequence first, then insert opens
    # Simple random walk that guarantees validity
    while remaining > 0 or stack:
        can_open = remaining > 0
        can_close = len(stack) > 0
        
        if can_open and (not can_close or rng.random() < 0.5):
            opener = OPENERS[rng.randint(0, 2)]
            tokens.append(opener)
            stack.append(opener)
            remaining -= 1
        elif can_close:
            closer = CLOSERS[OPENERS.index(stack[-1])]
            tokens.append(closer)
            stack.pop()
        else:
            break
    
    tokens.append(T_EOS)
    return tokens


def max_depth_and_type(tokens: list[int]) -> tuple[int, int]:
    """Return (max_depth, max_type_interleaving)"""
    depth = 0
    max_d = 0
    type_changes = 0
    last_opener_type = None
    for tok in tokens:
        if tok in OPENERS:
            depth += 1
            max_d = max(max_d, depth)
            if last_opener_type is not None and last_opener_type != tok:
                type_changes += 1
            last_opener_type = tok
        elif tok in CLOSERS:
            depth -= 1
    return max_d


def gen_train_seqs(max_depth: int, num_seqs: int, rng: random.Random, min_pairs: int = 4, max_pairs: int = 10) -> list[list[int]]:
    seqs = []
    for _ in range(num_seqs):
        while True:
            seq = _gen_balanced_interleaved(min_pairs, max_pairs, rng)
            d = max_depth_and_type(seq)
            if d <= max_depth and seq[-1] == T_EOS and len(seq) >= min_pairs * 2:
                seqs.append(seq)
                break
    return seqs


def gen_test_seqs_per_depth(target_depth: int, num_seqs: int, rng: random.Random) -> list[list[int]]:
    seqs = []
    for _ in range(num_seqs * 5):
        if len(seqs) >= num_seqs:
            break
        seq = _gen_balanced_interleaved(target_depth, target_depth + 6, rng)
        if max_depth_and_type(seq) == target_depth and seq[-1] == T_EOS:
            seqs.append(seq)
    return seqs[:num_seqs]


def accuracy_pct(correct: int, total: int) -> float:
    return (correct / total) * 100.0 if total > 0 else 0.0


def run(
    train_seqs: int = 400,
    test_per_depth: int = 80,
    seed: int = DEFAULT_SEED,
    fsm_states: int = FSM_STATE_BITS,
    pda_states: int = PDA_STATE_BITS,
    quiet: bool = False,
) -> dict[str, float]:
    """Run the mismatched-bracket benchmark and return OOD averages.

    Returns:
        dict with keys: pda_ood, fsm_ood, ppm_ood, pda_fsm_delta, pda_ppm_delta, winner
    """
    rng = random.Random(seed)

    if not quiet:
        print(f"=== Mismatched bracket types (v5): stack must track type+depth ===")
        print(f"Train: {train_seqs} seqs, depth <= {MAX_TRAIN_DEPTH}")
        print(f"Test:  {test_per_depth} seqs per depth")
        print()

    if not quiet:
        print("Generating training data...")
    train_seqs_list = gen_train_seqs(MAX_TRAIN_DEPTH, train_seqs, rng)
    total_train_tokens = sum(len(s) for s in train_seqs_list)
    if not quiet:
        print(f"  {len(train_seqs_list)} sequences, {total_train_tokens} tokens, avg len={total_train_tokens/len(train_seqs_list):.1f}")

    if not quiet:
        print("\nTraining PDA...")
    t0 = time.monotonic_ns()
    pda = train_pda(train_seqs_list, vocab_size=VOCAB_SIZE, state_bits=pda_states, stack_depth=PDA_STACK_DEPTH, max_push=1, max_pop=1, steps=PDA_STEPS)
    if not quiet:
        print(f"  {(time.monotonic_ns() - t0) // 1_000_000}ms")

    if not quiet:
        print("Training FSM...")
    t0 = time.monotonic_ns()
    fsm = train_fsm(train_seqs_list, vocab_size=VOCAB_SIZE, state_bits=fsm_states, context_len=FSM_CONTEXT_LEN, steps=FSM_STEPS)
    if not quiet:
        print(f"  {(time.monotonic_ns() - t0) // 1_000_000}ms")

    if not quiet:
        print("Training PPM...")
    t0 = time.monotonic_ns()
    ppm = train_ppm(train_seqs_list, vocab_size=VOCAB_SIZE, order=PPM_ORDER)
    if not quiet:
        print(f"  {(time.monotonic_ns() - t0) // 1_000_000}ms")

    if not quiet:
        print(f"\n{'Depth':<8} {'PDA':<10} {'FSM':<10} {'PPM':<10}")
        print("-" * 60)

    results = []
    for depth in TEST_DEPTHS:
        test_seqs = gen_test_seqs_per_depth(depth, test_per_depth, rng)
        if not test_seqs:
            continue

        p = evaluate_pda(pda, test_seqs)
        f = evaluate(fsm, test_seqs)
        pp = evaluate_ppm(ppm, test_seqs)

        pa = accuracy_pct(p["correct"], p["total"])
        fa = accuracy_pct(f["correct"], f["total"])
        ppa = accuracy_pct(pp["correct"], pp["total"])

        ood = depth > MAX_TRAIN_DEPTH
        if not quiet:
            ood_str = " [OOD]" if ood else ""
            print(f"{depth:<8} {pa:>6.1f}%   {fa:>6.1f}%   {ppa:>6.1f}%{ood_str}")
        results.append({"depth": depth, "pda": pa, "fsm": fa, "ppm": ppa, "ood": ood})

    ood_r = [r for r in results if r["ood"]]
    pda_ood = sum(r['pda'] for r in ood_r) / len(ood_r) if ood_r else 0.0
    fsm_ood = sum(r['fsm'] for r in ood_r) / len(ood_r) if ood_r else 0.0
    ppm_ood = sum(r['ppm'] for r in ood_r) / len(ood_r) if ood_r else 0.0

    if not quiet and ood_r:
        print(f"\nOOD avg: PDA={pda_ood:.1f}%, FSM={fsm_ood:.1f}%, PPM={ppm_ood:.1f}%")
        print(f"  PDA vs FSM: {pda_ood - fsm_ood:+.1f}pp | PDA vs PPM: {pda_ood - ppm_ood:+.1f}pp")

    winner = "PDA" if pda_ood > fsm_ood and pda_ood > ppm_ood else ("PPM" if ppm_ood > fsm_ood else "FSM")
    if not quiet:
        print(f"*** {winner} wins ***")

    # Write CSV
    csv_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark_code_v5_mismatched_types.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("depth,pda_acc,fsm_acc,ppm_acc,ood\n")
        for r in results:
            f.write(f"{r['depth']},{r['pda']:.2f},{r['fsm']:.2f},{r['ppm']:.2f},{r['ood']}\n")

    return {
        "pda_ood": pda_ood,
        "fsm_ood": fsm_ood,
        "ppm_ood": ppm_ood,
        "pda_fsm_delta": pda_ood - fsm_ood,
        "pda_ppm_delta": pda_ood - ppm_ood,
        "winner": winner,
        "ood_depths": len(ood_r),
    }


def main():
    parser = argparse.ArgumentParser(description="Mismatched bracket types benchmark (v5)")
    parser.add_argument("--train-seqs", type=int, default=400)
    parser.add_argument("--test-per-depth", type=int, default=80)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fsm-states", type=int, default=FSM_STATE_BITS)
    parser.add_argument("--pda-states", type=int, default=PDA_STATE_BITS)
    args = parser.parse_args()

    result = run(
        train_seqs=args.train_seqs,
        test_per_depth=args.test_per_depth,
        seed=args.seed,
        fsm_states=args.fsm_states,
        pda_states=args.pda_states,
        quiet=False,
    )
    csv_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark_code_v5_mismatched_types.csv"
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()