"""Robustness experiments for PDA depth-generalization.

Four axes:

1. Seed stability      – same settings, 5 different RNG seeds.
2. Token-ID invariance – permute which integer is OPEN / CLOSE / EOS;
                         the co-occurrence objective should be label-agnostic.
3. Noise robustness    – corrupt a fraction of training sequences by randomly
                         swapping OPEN/CLOSE tokens; check graceful degradation.
4. Stack-alphabet size – two matched bracket types require push={OPEN1,OPEN2}
                         and pop={CLOSE1,CLOSE2}; stack_top distinguishes them.

For every configuration the key metric is whether PDA-2ph outperforms FSM at
the deepest OOD depth (8), confirming depth-invariant stack encoding.

All arithmetic is integer-only.

Usage
-----
    python scripts/robustness_experiment.py
    python scripts/robustness_experiment.py --steps 10 --train-seqs 200
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
# Experiment-wide constants
# ---------------------------------------------------------------------------

MAX_TRAIN_DEPTH:       int = 3
DEEP_TEST_DEPTH:       int = 8   # primary OOD depth used for comparisons
TEST_DEPTHS:           tuple[int, ...] = (3, 6, 8)
DEFAULT_TRAIN_SEQS:    int = 300
DEFAULT_TEST_SEQS:     int = 100
DEFAULT_STEPS:         int = 20

PDA_STATE_BITS:        int = 2   # 4 states
PDA_STACK_DEPTH:       int = 10
FSM_STATE_BITS:        int = 4   # 16 states
FSM_CONTEXT_LEN:       int = 4
PPM_ORDER:             int = 6

# Axis 1 – seeds
SEEDS: list[int] = [0, 1, 2, 42, 100]

# Axis 2 – token-ID permutations of (OPEN, CLOSE, EOS) in {0, 1, 2}
TOKEN_PERMS: list[tuple[int, int, int]] = [
    (0, 1, 2),   # canonical
    (1, 2, 0),   # cyclic +1
    (2, 0, 1),   # cyclic +2
    (0, 2, 1),   # swap CLOSE↔EOS
    (1, 0, 2),   # swap OPEN↔CLOSE
    (2, 1, 0),   # reverse
]

# Axis 3 – noise rates (integer percentages, 0–100)
NOISE_RATES: list[int] = [0, 10, 20, 30]

# ---------------------------------------------------------------------------
# Single-bracket sequence generators (reused from reproduce_depth_generalization)
# ---------------------------------------------------------------------------


def _gen_one_balanced(
    max_depth: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
    open_tok: int = 0,
    close_tok: int = 1,
    eos_tok: int = 2,
) -> list[int]:
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n

    while opens_left + closes_left > 0:
        can_open  = (opens_left > 0 and depth < max_depth and closes_left > depth)
        can_close = closes_left > 0 and depth > 0

        if not can_open and not can_close:
            break

        if can_open and can_close:
            choice = open_tok if rng.randint(0, 1) == 0 else close_tok
        elif can_open:
            choice = open_tok
        else:
            choice = close_tok

        if choice == open_tok:
            tokens.append(open_tok)
            depth += 1
            opens_left -= 1
        else:
            tokens.append(close_tok)
            depth -= 1
            closes_left -= 1

    tokens.append(eos_tok)
    return tokens


def _max_depth_of(tokens: list[int], open_tok: int = 0) -> int:
    depth = 0
    max_d = 0
    for tok in tokens:
        if tok == open_tok:
            depth += 1
            max_d = max(max_d, depth)
        elif tok != open_tok:
            depth = max(0, depth - 1)
    return max_d


def _max_depth_multi(tokens: list[int], open_tokens: set[int]) -> int:
    """Max nesting depth for multi-bracket sequences."""
    depth = 0
    max_d = 0
    for tok in tokens:
        if tok in open_tokens:
            depth += 1
            max_d = max(max_d, depth)
        else:
            depth = max(0, depth - 1)
    return max_d


def gen_train_seqs(
    max_depth: int,
    num_seqs: int,
    seed: int,
    open_tok: int = 0,
    close_tok: int = 1,
    eos_tok: int = 2,
    min_pairs: int = 2,
    max_pairs: int = 8,
) -> list[list[int]]:
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    while len(seqs) < num_seqs:
        seqs.append(_gen_one_balanced(max_depth, min_pairs, max_pairs, rng,
                                      open_tok, close_tok, eos_tok))
    return seqs


def gen_test_seqs_at_depth(
    target_depth: int,
    num_seqs: int,
    seed: int,
    open_tok: int = 0,
    close_tok: int = 1,
    eos_tok: int = 2,
    min_pairs: int | None = None,
    max_pairs: int = 16,
) -> list[list[int]]:
    if min_pairs is None:
        min_pairs = target_depth
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    max_attempts = num_seqs * 10_000
    attempts = 0
    while len(seqs) < num_seqs and attempts < max_attempts:
        seq = _gen_one_balanced(target_depth, min_pairs, max_pairs, rng,
                                open_tok, close_tok, eos_tok)
        if seq.count(open_tok) >= target_depth:
            d = 0
            m = 0
            for t in seq:
                if t == open_tok:
                    d += 1
                    m = max(m, d)
                elif t == close_tok:
                    d -= 1
            if m == target_depth:
                seqs.append(seq)
        attempts += 1
    return seqs


# ---------------------------------------------------------------------------
# Noise corruption helper (axis 3)
# ---------------------------------------------------------------------------


def _corrupt_seq(
    seq: list[int],
    open_tok: int,
    close_tok: int,
    noise_rate: int,
    rng: random.Random,
) -> list[int]:
    """Swap OPEN↔CLOSE in a random subset of positions.

    *noise_rate* is an integer percentage 0–100.  Each OPEN or CLOSE token
    is independently flipped with probability noise_rate / 100.
    Uses only integer random ops (``rng.randint``), no floats.
    """
    result = list(seq)
    for i, tok in enumerate(result):
        if (tok == open_tok or tok == close_tok) and rng.randint(0, 99) < noise_rate:
            result[i] = close_tok if tok == open_tok else open_tok
    return result


def corrupt_train_seqs(
    seqs: list[list[int]],
    open_tok: int,
    close_tok: int,
    noise_rate: int,
    seed: int,
) -> list[list[int]]:
    if noise_rate == 0:
        return seqs
    rng = random.Random(seed + 9999)
    return [_corrupt_seq(s, open_tok, close_tok, noise_rate, rng) for s in seqs]


# ---------------------------------------------------------------------------
# Multi-bracket sequence generator (axis 4)
# ---------------------------------------------------------------------------


def _gen_one_multi_bracket(
    max_depth: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
    open_tokens: list[int],
    close_map: dict[int, int],
    eos: int,
) -> list[int]:
    """Generate properly nested multi-bracket sequences.

    Each open bracket is chosen uniformly from *open_tokens*.
    Each close bracket must match the topmost open bracket on the stack.
    """
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n
    bracket_stack: list[int] = []

    while opens_left + closes_left > 0:
        can_open  = (opens_left > 0 and depth < max_depth and closes_left > depth)
        can_close = closes_left > 0 and depth > 0

        if not can_open and not can_close:
            break

        if can_open and can_close:
            action = "open" if rng.randint(0, 1) == 0 else "close"
        elif can_open:
            action = "open"
        else:
            action = "close"

        if action == "open":
            open_tok = open_tokens[rng.randint(0, len(open_tokens) - 1)]
            tokens.append(open_tok)
            bracket_stack.append(open_tok)
            depth += 1
            opens_left -= 1
        else:
            open_tok = bracket_stack[-1]
            tokens.append(close_map[open_tok])
            bracket_stack.pop()
            depth -= 1
            closes_left -= 1

    tokens.append(eos)
    return tokens


def gen_multi_train_seqs(
    max_depth: int,
    num_seqs: int,
    seed: int,
    open_tokens: list[int],
    close_map: dict[int, int],
    eos: int,
    min_pairs: int = 2,
    max_pairs: int = 8,
) -> list[list[int]]:
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    while len(seqs) < num_seqs:
        seqs.append(_gen_one_multi_bracket(
            max_depth, min_pairs, max_pairs, rng, open_tokens, close_map, eos
        ))
    return seqs


def gen_multi_test_seqs_at_depth(
    target_depth: int,
    num_seqs: int,
    seed: int,
    open_tokens: list[int],
    close_map: dict[int, int],
    eos: int,
    max_pairs: int = 16,
) -> list[list[int]]:
    min_pairs = target_depth
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    max_attempts = num_seqs * 10_000
    attempts = 0
    open_set = set(open_tokens)
    while len(seqs) < num_seqs and attempts < max_attempts:
        seq = _gen_one_multi_bracket(
            target_depth, min_pairs, max_pairs, rng, open_tokens, close_map, eos
        )
        if _max_depth_multi(seq, open_set) == target_depth:
            seqs.append(seq)
        attempts += 1
    return seqs


# ---------------------------------------------------------------------------
# Train + evaluate helpers
# ---------------------------------------------------------------------------


def _train_all(
    train_data: list[list[int]],
    vocab_size: int,
    steps: int,
    max_push: int = 1,
    max_pop: int = 1,
) -> tuple:
    pda_model = train_pda(
        sequences=train_data,
        vocab_size=vocab_size,
        state_bits=PDA_STATE_BITS,
        stack_depth=PDA_STACK_DEPTH,
        steps=steps,
        max_push=max_push,
        max_pop=max_pop,
        top_k_coverage=vocab_size,
    )
    fsm_model = train_fsm(
        sequences=train_data,
        vocab_size=vocab_size,
        state_bits=FSM_STATE_BITS,
        steps=steps,
        context_len=FSM_CONTEXT_LEN,
    )
    ppm_model = train_ppm(
        sequences=train_data,
        vocab_size=vocab_size,
        order=PPM_ORDER,
    )
    return pda_model, fsm_model, ppm_model


def _eval_all(pda_model, fsm_model, ppm_model, test_data):
    pda_r = evaluate_pda(pda_model, test_data)
    fsm_r = evaluate(fsm_model, test_data)
    ppm_r = evaluate_ppm(ppm_model, test_data)
    return (
        (pda_r["correct"], pda_r["total"]),
        (fsm_r["correct"], fsm_r["total"]),
        (ppm_r["correct"], ppm_r["total"]),
    )


# ---------------------------------------------------------------------------
# Axis 1 – Seed stability
# ---------------------------------------------------------------------------


def run_seed_stability(
    train_seqs: int,
    test_seqs: int,
    steps: int,
) -> None:
    print()
    print("=" * 72)
    print("AXIS 1: Seed stability")
    print(f"  vocab=(OPEN=0,CLOSE=1,EOS=2)  train_seqs={train_seqs}  steps={steps}s")
    print("=" * 72)
    print(f"{'seed':>6}  {'depth':>5}  {'PDA-2ph':>10}  {'FSM':>10}  {'PPM':>10}  {'push':>6}  {'pop':>6}")
    print("-" * 72)

    pda_wins = 0
    total_runs = 0

    for seed in SEEDS:
        train_data = gen_train_seqs(MAX_TRAIN_DEPTH, train_seqs, seed)
        pda_model, fsm_model, ppm_model = _train_all(train_data, 3, steps)

        for depth in TEST_DEPTHS:
            test_data = gen_test_seqs_at_depth(depth, test_seqs, seed + depth)
            pda_r, fsm_r, ppm_r = _eval_all(pda_model, fsm_model, ppm_model, test_data)

            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            if depth == DEEP_TEST_DEPTH:
                pda_bp = accuracy_pct_times100(*pda_r)
                fsm_bp = accuracy_pct_times100(*fsm_r)
                if pda_bp > fsm_bp:
                    pda_wins += 1
                total_runs += 1

            print(
                f"{seed:>6}  {depth:>4}{ood}"
                f"  {format_accuracy(*pda_r):>10}"
                f"  {format_accuracy(*fsm_r):>10}"
                f"  {format_accuracy(*ppm_r):>10}"
                f"  {sorted({tok for (_, tok, _) in pda_model.push_configs})!s:>6}"
                f"  {sorted({tok for (_, tok, _) in pda_model.pop_configs})!s:>6}"
            )

        print()

    print(f"PDA-2ph > FSM at depth {DEEP_TEST_DEPTH}: {pda_wins}/{total_runs} seeds")


# ---------------------------------------------------------------------------
# Axis 2 – Token-ID invariance
# ---------------------------------------------------------------------------


def run_token_permutations(
    train_seqs: int,
    test_seqs: int,
    steps: int,
    seed: int = 42,
) -> None:
    print()
    print("=" * 72)
    print("AXIS 2: Token-ID invariance  (seed=42 fixed)")
    print(f"  train_seqs={train_seqs}  steps={steps}s")
    print("=" * 72)
    print(
        f"{'perm':>12}  {'depth':>5}  {'PDA-2ph':>10}  {'FSM':>10}  {'PPM':>10}"
        f"  {'push':>8}  {'pop':>8}"
    )
    print("-" * 72)

    pda_wins = 0
    correct_discovery = 0
    total_runs = 0

    for open_t, close_t, eos_t in TOKEN_PERMS:
        perm_label = f"({open_t},{close_t},{eos_t})"
        train_data = gen_train_seqs(
            MAX_TRAIN_DEPTH, train_seqs, seed,
            open_tok=open_t, close_tok=close_t, eos_tok=eos_t,
        )
        pda_model, fsm_model, ppm_model = _train_all(train_data, 3, steps)

        for depth in TEST_DEPTHS:
            test_data = gen_test_seqs_at_depth(
                depth, test_seqs, seed + depth,
                open_tok=open_t, close_tok=close_t, eos_tok=eos_t,
            )
            pda_r, fsm_r, ppm_r = _eval_all(pda_model, fsm_model, ppm_model, test_data)

            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            if depth == DEEP_TEST_DEPTH:
                pda_bp = accuracy_pct_times100(*pda_r)
                fsm_bp = accuracy_pct_times100(*fsm_r)
                if pda_bp > fsm_bp:
                    pda_wins += 1
                # Check push/pop discovery
                if (
                    {tok for (_, tok, _) in pda_model.push_configs} == {open_t}
                    and {tok for (_, tok, _) in pda_model.pop_configs} == {close_t}
                ):
                    correct_discovery += 1
                total_runs += 1

            print(
                f"{perm_label:>12}  {depth:>4}{ood}"
                f"  {format_accuracy(*pda_r):>10}"
                f"  {format_accuracy(*fsm_r):>10}"
                f"  {format_accuracy(*ppm_r):>10}"
                f"  {sorted({tok for (_, tok, _) in pda_model.push_configs})!s:>8}"
                f"  {sorted({tok for (_, tok, _) in pda_model.pop_configs})!s:>8}"
            )

        print()

    print(f"PDA-2ph > FSM at depth {DEEP_TEST_DEPTH}: {pda_wins}/{total_runs} permutations")
    print(f"Correct push/pop discovery: {correct_discovery}/{total_runs} permutations")


# ---------------------------------------------------------------------------
# Axis 3 – Noise robustness
# ---------------------------------------------------------------------------


def run_noise_robustness(
    train_seqs: int,
    test_seqs: int,
    steps: int,
    seed: int = 42,
) -> None:
    print()
    print("=" * 72)
    print("AXIS 3: Noise robustness  (seed=42 fixed)")
    print(f"  vocab=(OPEN=0,CLOSE=1,EOS=2)  train_seqs={train_seqs}  steps={steps}s")
    print("  Noise: each OPEN/CLOSE token independently flipped with prob noise_rate/100")
    print("=" * 72)
    print(
        f"{'noise%':>7}  {'depth':>5}  {'PDA-2ph':>10}  {'FSM':>10}  {'PPM':>10}"
        f"  {'push':>6}  {'pop':>6}"
    )
    print("-" * 72)

    pda_wins = 0
    total_runs = 0

    for noise_rate in NOISE_RATES:
        train_data = gen_train_seqs(MAX_TRAIN_DEPTH, train_seqs, seed)
        noisy_data = corrupt_train_seqs(train_data, 0, 1, noise_rate, seed)
        pda_model, fsm_model, ppm_model = _train_all(noisy_data, 3, steps)

        for depth in TEST_DEPTHS:
            test_data = gen_test_seqs_at_depth(depth, test_seqs, seed + depth)
            pda_r, fsm_r, ppm_r = _eval_all(pda_model, fsm_model, ppm_model, test_data)

            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            if depth == DEEP_TEST_DEPTH:
                pda_bp = accuracy_pct_times100(*pda_r)
                fsm_bp = accuracy_pct_times100(*fsm_r)
                if pda_bp > fsm_bp:
                    pda_wins += 1
                total_runs += 1

            print(
                f"{noise_rate:>7}  {depth:>4}{ood}"
                f"  {format_accuracy(*pda_r):>10}"
                f"  {format_accuracy(*fsm_r):>10}"
                f"  {format_accuracy(*ppm_r):>10}"
                f"  {sorted({tok for (_, tok, _) in pda_model.push_configs})!s:>6}"
                f"  {sorted({tok for (_, tok, _) in pda_model.pop_configs})!s:>6}"
            )

        print()

    print(f"PDA-2ph > FSM at depth {DEEP_TEST_DEPTH}: {pda_wins}/{total_runs} noise levels")


# ---------------------------------------------------------------------------
# Axis 4 – Multi-bracket (stack alphabet > 1)
# ---------------------------------------------------------------------------


def run_multi_bracket(
    train_seqs: int,
    test_seqs: int,
    steps: int,
    seed: int = 42,
) -> None:
    """Two matched bracket types.

    Vocab (size 5):
      OPEN1=0  CLOSE1=1  OPEN2=2  CLOSE2=3  EOS=4

    Correct stack policy:
      push_tokens = {0, 2}   (both OPEN types)
      pop_tokens  = {1, 3}   (both CLOSE types)

    stack_top distinguishes which bracket type we're inside:
      config (state, stack_top=0) → should predict CLOSE1=1 eventually
      config (state, stack_top=2) → should predict CLOSE2=3 eventually
    """
    OPEN1, CLOSE1 = 0, 1
    OPEN2, CLOSE2 = 2, 3
    EOS             = 4
    VOCAB_SIZE      = 5

    open_tokens = [OPEN1, OPEN2]
    close_map   = {OPEN1: CLOSE1, OPEN2: CLOSE2}
    open_set    = set(open_tokens)

    print()
    print("=" * 72)
    print("AXIS 4: Multi-bracket  (2 bracket types, stack alphabet > 1)")
    print(f"  vocab=5 (OPEN1=0,CLOSE1=1,OPEN2=2,CLOSE2=3,EOS=4)")
    print(f"  train_seqs={train_seqs}  steps={steps}s  seed={seed}")
    print(f"  Expected: push={{0,2}}  pop={{1,3}}")
    print("=" * 72)
    print(f"{'depth':>5}  {'PDA-2ph':>10}  {'FSM':>10}  {'PPM':>10}  {'push':>8}  {'pop':>8}")
    print("-" * 72)

    train_data = gen_multi_train_seqs(
        MAX_TRAIN_DEPTH, train_seqs, seed, open_tokens, close_map, EOS,
    )
    pda_model = train_pda(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        state_bits=PDA_STATE_BITS,
        stack_depth=PDA_STACK_DEPTH,
        steps=steps,
        max_push=2,   # two push tokens expected
        max_pop=2,    # two pop  tokens expected
        top_k_coverage=VOCAB_SIZE,
    )
    fsm_model = train_fsm(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        state_bits=FSM_STATE_BITS,
        steps=steps,
        context_len=FSM_CONTEXT_LEN,
    )
    ppm_model = train_ppm(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        order=PPM_ORDER,
    )

    pda_win_d8 = False

    for depth in TEST_DEPTHS:
        test_data = gen_multi_test_seqs_at_depth(
            depth, test_seqs, seed + depth, open_tokens, close_map, EOS,
        )
        pda_r = evaluate_pda(pda_model, test_data)
        fsm_r = evaluate(fsm_model, test_data)
        ppm_r = evaluate_ppm(ppm_model, test_data)

        pda_t = (pda_r["correct"], pda_r["total"])
        fsm_t = (fsm_r["correct"], fsm_r["total"])
        ppm_t = (ppm_r["correct"], ppm_r["total"])

        ood = "*" if depth > MAX_TRAIN_DEPTH else " "
        if depth == DEEP_TEST_DEPTH:
            pda_win_d8 = accuracy_pct_times100(*pda_t) > accuracy_pct_times100(*fsm_t)

        print(
            f"{depth:>4}{ood}"
            f"  {format_accuracy(*pda_t):>10}"
            f"  {format_accuracy(*fsm_t):>10}"
            f"  {format_accuracy(*ppm_t):>10}"
            f"  {sorted({tok for (_, tok, _) in pda_model.push_configs})!s:>8}"
            f"  {sorted({tok for (_, tok, _) in pda_model.pop_configs})!s:>8}"
        )

    correct_push = {tok for (_, tok, _) in pda_model.push_configs} == {OPEN1, OPEN2}
    correct_pop  = {tok for (_, tok, _) in pda_model.pop_configs}  == {CLOSE1, CLOSE2}
    print()
    print(f"Correct push discovery {{0,2}}: {correct_push}")
    print(f"Correct pop  discovery {{1,3}}: {correct_pop}")
    print(f"PDA-2ph > FSM at depth {DEEP_TEST_DEPTH}: {pda_win_d8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"CP-SAT wall-clock budget per model in seconds (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--train-seqs",
        type=int,
        default=DEFAULT_TRAIN_SEQS,
        help=f"Training sequences per run (default: {DEFAULT_TRAIN_SEQS})",
    )
    parser.add_argument(
        "--test-seqs",
        type=int,
        default=DEFAULT_TEST_SEQS,
        help=f"Test sequences per depth band (default: {DEFAULT_TEST_SEQS})",
    )
    parser.add_argument(
        "--axis",
        choices=["seeds", "tokens", "noise", "multi", "all"],
        default="all",
        help="Which robustness axis to run (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    kw = dict(train_seqs=args.train_seqs, test_seqs=args.test_seqs, steps=args.steps)

    if args.axis in ("seeds", "all"):
        run_seed_stability(**kw)
    if args.axis in ("tokens", "all"):
        run_token_permutations(**kw)
    if args.axis in ("noise", "all"):
        run_noise_robustness(**kw)
    if args.axis in ("multi", "all"):
        run_multi_bracket(**kw)


if __name__ == "__main__":
    main()
