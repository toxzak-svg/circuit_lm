"""Benchmark: Multi-bracket code structures — PDA vs FSM vs PPM depth generalization

Thesis
------
A PDA with an explicit stack generalizes to deeper nested structures (functions,
lists, dicts) better than FSMs or PPM, because the stack mechanism matches the
structural recursion in code.

Results history
---------------
v1 (200 train seqs, min 2 pairs): PPM dominates (~32% vs 24%)
v2 (400 train seqs, min 4 pairs): TESTING

NOTES
-----
- Longer sequences = harder next-token prediction = more stress on structure
- PPM wins on short sequences because local n-grams suffice
- PDA should pull ahead when prediction requires tracking deep state
- Try: (1) longer sequences [CURRENT], (2) harder split (train depth<=3), (3) bracket-only vocab
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
MAX_TRAIN_DEPTH = 4
TEST_DEPTHS = (3, 4, 5, 6, 7, 8, 10, 12)
DEFAULT_SEED = 42

# Model hyper-parameters (scaled up for harder task)
PDA_STATE_BITS = 4          # 16 states (was 3)
PDA_STACK_DEPTH = 15
PDA_STEPS = 60             # more time for harder problem
FSM_STATE_BITS = 6         # 64 states (was 5)
FSM_STEPS = 60
FSM_CONTEXT_LEN = 8        # was 6
PPM_ORDER = 8              # was 6

# ---------------------------------------------------------------------------
# Data generation — longer sequences version
# ---------------------------------------------------------------------------

def _gen_mixed_bracket_seq(
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
) -> list[int]:
    """Generate one mixed-bracket balanced sequence with configurable length.
    
    min_pairs/max_pairs control sequence length directly.
    """
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n
    open_stack: list[int] = []

    while opens_left + closes_left > 0:
        can_open = (
            opens_left > 0
            and closes_left > depth  # can always close remaining
        )
        can_close = closes_left > 0 and len(open_stack) > 0

        if not can_open and not can_close:
            break

        if can_open and can_close:
            # 55% open to create nesting
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

    # Close remaining
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


def gen_train_seqs(max_depth: int, num_seqs: int, rng: random.Random, min_pairs: int = 4, max_pairs: int = 12) -> list[list[int]]:
    """Generate training sequences with depth <= max_depth and longer lengths."""
    seqs = []
    for _ in range(num_seqs):
        while True:
            seq = _gen_mixed_bracket_seq(min_pairs, max_pairs, rng)
            if max_depth_of(seq) <= max_depth and len(seq) >= min_pairs * 2:
                seqs.append(seq)
                break
    return seqs


def gen_test_seqs_per_depth(
    target_depth: int,
    num_seqs: int,
    rng: random.Random,
    min_pairs: int = 4,
    max_pairs: int = 12,
) -> list[list[int]]:
    """Generate test sequences with exact target nesting depth."""
    seqs = []
    for _ in range(num_seqs * 3):  # generate more, filter
        if len(seqs) >= num_seqs:
            break
        seq = _gen_mixed_bracket_seq(max(target_depth, min_pairs), max_pairs + 4, rng)
        if max_depth_of(seq) == target_depth and seq[-1] == T_EOS:
            seqs.append(seq)
    return seqs[:num_seqs]


def accuracy_pct(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (correct / total) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Multi-bracket depth generalization benchmark (v2: longer sequences)")
    parser.add_argument("--train-seqs", type=int, default=400)
    parser.add_argument("--test-per-depth", type=int, default=80)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-depth", type=int, default=MAX_TRAIN_DEPTH)
    parser.add_argument("--min-pairs", type=int, default=4, help="Minimum bracket pairs per sequence (v2: 4 vs v1: 2)")
    parser.add_argument("--max-pairs", type=int, default=12, help="Maximum bracket pairs per sequence (v2: 12 vs v1: 8)")
    parser.add_argument("--fsm-states", type=int, default=FSM_STATE_BITS)
    parser.add_argument("--pda-states", type=int, default=PDA_STATE_BITS)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"=== Multi-bracket code benchmark v2 (longer sequences) ===")
    print(f"Train: {args.train_seqs} seqs, depth <= {args.max_train_depth}, {args.min_pairs}-{args.max_pairs} pairs")
    print(f"Test:  {args.test_per_depth} seqs per depth")
    print(f"PDA: {2**args.pda_states} states, FSM: {2**args.fsm_states} states")
    print()

    # Generate training data
    print("Generating training data...")
    train_seqs = gen_train_seqs(args.max_train_depth, args.train_seqs, rng, args.min_pairs, args.max_pairs)
    total_train_tokens = sum(len(s) for s in train_seqs)
    avg_len = total_train_tokens / len(train_seqs)
    print(f"  {len(train_seqs)} sequences, {total_train_tokens} total tokens, avg len={avg_len:.1f}")

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
        steps=PDA_STEPS,
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
        test_seqs = gen_test_seqs_per_depth(depth, args.test_per_depth, rng, args.min_pairs, args.max_pairs)
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
    id_results = [r for r in results if not r["ood"]]
    if ood_results:
        avg_pda = sum(r["pda_acc"] for r in ood_results) / len(ood_results)
        avg_fsm = sum(r["fsm_acc"] for r in ood_results) / len(ood_results)
        avg_ppm = sum(r["ppm_acc"] for r in ood_results) / len(ood_results)
        print(f"Average OOD accuracy (depth > {args.max_train_depth}):")
        print(f"  PDA: {avg_pda:.1f}%")
        print(f"  FSM: {avg_fsm:.1f}%")
        print(f"  PPM: {avg_ppm:.1f}%")
        print(f"  PDA vs FSM: {avg_pda - avg_fsm:+.1f}pp")
        print(f"  PDA vs PPM: {avg_pda - avg_ppm:+.1f}pp")
    if id_results:
        avg_pda = sum(r["pda_acc"] for r in id_results) / len(id_results)
        avg_fsm = sum(r["fsm_acc"] for r in id_results) / len(id_results)
        avg_ppm = sum(r["ppm_acc"] for r in id_results) / len(id_results)
        print(f"\nAverage ID accuracy (depth <= {args.max_train_depth}):")
        print(f"  PDA: {avg_pda:.1f}%")
        print(f"  FSM: {avg_fsm:.1f}%")
        print(f"  PPM: {avg_ppm:.1f}%")

    # CSV output
    csv_path = pathlib.Path(__file__).parent.parent / "results" / "benchmark_code_v2_longer.csv"
    csv_path.parent.mkdir(exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("depth,pda_acc,fsm_acc,ppm_acc,ood\n")
        for r in results:
            f.write(f"{r['depth']},{r['pda_acc']:.2f},{r['fsm_acc']:.2f},{r['ppm_acc']:.2f},{r['ood']}\n")
    print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
