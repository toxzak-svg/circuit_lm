"""Small-scale joint-PDA depth-generalization verifier (Task 6).

Runs the same depth-generalization comparison as reproduce_depth_generalization.py
but with 100 training sequences (T_total ≈ 1100 < 2000 solver limit), so that
the joint-PDA solver has enough budget to discover the stack.

Usage
-----
    py -3 scripts/verify_joint_pda_small.py
    py -3 scripts/verify_joint_pda_small.py --train-seqs 80 --steps 60
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.eval    import evaluate_pda
from circuit_lm.metrics import format_accuracy, accuracy_pct_times100
from circuit_lm.train_joint_pda_cpsat import train_joint_pda
from circuit_lm.train_pda_cpsat       import train_pda

# ---------------------------------------------------------------------------
# Vocabulary (same as reproduce_depth_generalization.py)
# ---------------------------------------------------------------------------

OPEN:       int = 0
CLOSE:      int = 1
EOS:        int = 2
VOCAB_SIZE: int = 3

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

MAX_TRAIN_DEPTH:        int = 3
TEST_DEPTHS: tuple[int, ...] = (3, 4, 5, 6)
TRAIN_SEQS:             int = 100
TEST_SEQS_PER_DEPTH:    int = 30

PDA_STATE_BITS:  int = 2
PDA_STACK_DEPTH: int = 10
PDA_STEPS:       int = 20
PDA_MAX_PUSH:    int = 1
PDA_MAX_POP:     int = 1

JPDA_NUM_STATES:  int = 4
JPDA_STACK_DEPTH: int = 10
JPDA_STEPS:       int = 30
JPDA_MAX_PUSH:    int = 1
JPDA_MAX_POP:     int = 1

# ---------------------------------------------------------------------------
# Data generation (duplicated from reproduce_depth_generalization.py)
# ---------------------------------------------------------------------------


def _gen_one_balanced(
    max_depth: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
) -> list[int]:
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n
    while opens_left + closes_left > 0:
        can_open  = opens_left > 0 and depth < max_depth and closes_left > depth
        can_close = closes_left > 0 and depth > 0
        if not can_open and not can_close:
            break
        if can_open and can_close:
            choice = OPEN if rng.randint(0, 1) == 0 else CLOSE
        elif can_open:
            choice = OPEN
        else:
            choice = CLOSE
        if choice == OPEN:
            tokens.append(OPEN);  depth += 1;  opens_left -= 1
        else:
            tokens.append(CLOSE); depth -= 1; closes_left -= 1
    tokens.append(EOS)
    return tokens


def _max_depth_of(tokens: list[int]) -> int:
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
    max_depth: int, num_seqs: int, seed: int,
    min_pairs: int = 2, max_pairs: int = 8,
) -> list[list[int]]:
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    while len(seqs) < num_seqs:
        seqs.append(_gen_one_balanced(max_depth, min_pairs, max_pairs, rng))
    return seqs


def gen_test_seqs_at_depth(
    target_depth: int, num_seqs: int, seed: int,
    min_pairs: int | None = None, max_pairs: int = 16,
) -> list[list[int]]:
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
# Eval helpers
# ---------------------------------------------------------------------------


def _eval(model, seqs: list[list[int]]) -> tuple[int, int]:
    r = evaluate_pda(model, seqs)
    return r["correct"], r["total"]  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Core run function (importable for tests)
# ---------------------------------------------------------------------------


def run_small(
    seed:                int = 42,
    train_seqs:          int = TRAIN_SEQS,
    test_seqs_per_depth: int = TEST_SEQS_PER_DEPTH,
    steps:               int = JPDA_STEPS,
    quiet:               bool = False,
) -> dict:
    """Run the small-scale joint-PDA verification experiment.

    Returns a plain dict so callers (including tests) can inspect results
    without parsing stdout.

    Keys:
        t_total (int)          — total tokens in training data
        push_tokens (list[int])— distinct token IDs in jpda push_configs
        pop_tokens  (list[int])— distinct token IDs in jpda pop_configs
        stack_discovered (bool)— True iff push_configs is non-empty
        results (dict)         — {depth: {"pda": (correct, total),
                                          "jpda": (correct, total)}}
    """
    if not quiet:
        print()
        print("=== Joint-PDA Small-Scale Verification ===")
        print(f"  max_train_depth={MAX_TRAIN_DEPTH}  seed={seed}")
        print(f"  train_seqs={train_seqs}  test_seqs_per_depth={test_seqs_per_depth}")
        print(f"  jpda_steps={steps}s  pda_steps={PDA_STEPS}s")
        print()

    train_data = gen_train_seqs(MAX_TRAIN_DEPTH, train_seqs, seed)
    t_total: int = sum(len(s) for s in train_data)

    if not quiet:
        print(f"  T_total={t_total}  (limit: 2000)")
        print(f"  Training joint-PDA (num_states={JPDA_NUM_STATES}, steps={steps}s) ...")

    jpda_model = train_joint_pda(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        num_states=JPDA_NUM_STATES,
        stack_depth=JPDA_STACK_DEPTH,
        steps=steps,
        max_push=JPDA_MAX_PUSH,
        max_pop=JPDA_MAX_POP,
        top_k_coverage=VOCAB_SIZE,
    )

    push_tokens = sorted({tok for (_, tok, _) in jpda_model.push_configs})
    pop_tokens  = sorted({tok for (_, tok, _) in jpda_model.pop_configs})
    stack_discovered = len(jpda_model.push_configs) > 0

    if not quiet:
        print(f"  jpda push_tokens={push_tokens}  pop_tokens={pop_tokens}")
        if stack_discovered:
            print("  [PASS] joint-PDA discovered stack")
        else:
            print("  [WARN] joint-PDA found no stack — push_configs empty")
        print(f"  Training PDA-2ph (state_bits={PDA_STATE_BITS}, steps={PDA_STEPS}s) ...")

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
        _pp = sorted({tok for (_, tok, _) in pda_model.push_configs})
        _pop = sorted({tok for (_, tok, _) in pda_model.pop_configs})
        print(f"  pda  push_tokens={_pp}  pop_tokens={_pop}")
        print()
        print("  Evaluating ...")
        _SEP = "-" * 52
        _HDR = f"{'depth':>7}  {'PDA-2ph':>10}  {'PDA-jt':>10}  {'seqs':>6}"
        print(_SEP)
        print(_HDR)
        print(_SEP)

    results: dict[int, dict[str, tuple[int, int]]] = {}

    for depth in TEST_DEPTHS:
        test_data = gen_test_seqs_at_depth(depth, test_seqs_per_depth, seed + depth)
        n = len(test_data)
        pda_res  = _eval(pda_model,  test_data)
        jpda_res = _eval(jpda_model, test_data)
        results[depth] = {"pda": pda_res, "jpda": jpda_res}

        if not quiet:
            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            print(
                f"{depth:>6}{ood}"
                f"  {format_accuracy(*pda_res):>10}"
                f"  {format_accuracy(*jpda_res):>10}"
                f"  {n:>6}"
            )

    if not quiet:
        print(_SEP)
        print()
        print("  Basis-points (10000 = 100%):  PDA-2ph / PDA-jt")
        for depth in TEST_DEPTHS:
            r = results[depth]
            pda_bp  = accuracy_pct_times100(*r["pda"])
            jpda_bp = accuracy_pct_times100(*r["jpda"])
            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            print(f"    depth {depth}{ood}: PDA-2ph={pda_bp:5d}  PDA-jt={jpda_bp:5d}")
        print()

    return {
        "t_total":          t_total,
        "push_tokens":      push_tokens,
        "pop_tokens":       pop_tokens,
        "stack_discovered": stack_discovered,
        "results":          results,
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed",                 type=int, default=42)
    p.add_argument("--train-seqs",           type=int, default=TRAIN_SEQS)
    p.add_argument("--test-seqs-per-depth",  type=int, default=TEST_SEQS_PER_DEPTH)
    p.add_argument("--steps",                type=int, default=JPDA_STEPS)
    p.add_argument("--quiet",                action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_small(
        seed=args.seed,
        train_seqs=args.train_seqs,
        test_seqs_per_depth=args.test_seqs_per_depth,
        steps=args.steps,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
