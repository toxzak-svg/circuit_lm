"""Joint-PDA push/pop discovery parameter sweep.

Calls run_small() from verify_joint_pda_small with each (train_seqs, steps)
combination and reports whether push/pop/full stack were discovered.

Usage
-----
    py -3 scripts/sweep_jpda_budget.py
    py -3 scripts/sweep_jpda_budget.py --train-seqs 50 100 --steps 30 60 120
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from verify_joint_pda_small import run_small  # noqa: E402


SWEEP_GRID: list[tuple[int, int]] = [
    (50, 30),
    (50, 60),
    (50, 120),
    (100, 60),
    (100, 120),
]


def run_sweep(
    param_grid: list[tuple[int, int]],
    seed: int = 42,
    test_seqs_per_depth: int = 10,
    quiet: bool = False,
) -> list[dict]:
    """Run a (train_seqs, steps) sweep and return row dicts."""
    rows: list[dict] = []

    for train_seqs, steps in param_grid:
        if not quiet:
            print(
                f"  [{len(rows) + 1}/{len(param_grid)}]"
                f" train_seqs={train_seqs} steps={steps}s ..."
            )
        result = run_small(
            seed=seed,
            train_seqs=train_seqs,
            test_seqs_per_depth=test_seqs_per_depth,
            steps=steps,
            quiet=True,
        )
        push_tokens = list(result["push_tokens"])
        pop_tokens = list(result["pop_tokens"])
        push_discovered = len(push_tokens) > 0
        pop_discovered = len(pop_tokens) > 0

        rows.append(
            {
                "train_seqs": train_seqs,
                "steps": steps,
                "t_total": int(result["t_total"]),
                "push_discovered": push_discovered,
                "pop_discovered": pop_discovered,
                "full_stack_discovered": push_discovered and pop_discovered,
                "push_tokens": push_tokens,
                "pop_tokens": pop_tokens,
            }
        )
    return rows


_HDR = (
    f"{'train_seqs':>10}  {'steps':>5}  {'T_total':>7}"
    f"  {'push':>5}  {'pop':>5}  {'full':>5}"
    f"  {'push_toks':>10}  {'pop_toks':>10}"
)
_SEP = "-" * len(_HDR)


def _print_table(rows: list[dict]) -> None:
    print(_SEP)
    print(_HDR)
    print(_SEP)
    for row in rows:
        push_s = "YES" if row["push_discovered"] else "no"
        pop_s = "YES" if row["pop_discovered"] else "no"
        full_s = "YES" if row["full_stack_discovered"] else "no"
        print(
            f"{row['train_seqs']:>10}  {row['steps']:>5}  {row['t_total']:>7}"
            f"  {push_s:>5}  {pop_s:>5}  {full_s:>5}"
            f"  {str(row['push_tokens']):>10}  {str(row['pop_tokens']):>10}"
        )
    print(_SEP)


def _parse_grid(train_seqs: list[int], steps: list[int]) -> list[tuple[int, int]]:
    return [(n, s) for n in train_seqs for s in steps]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-seqs-per-depth", type=int, default=10)
    parser.add_argument(
        "--train-seqs",
        type=int,
        nargs="+",
        default=[],
        help="Override train sequence counts (space-separated list)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[],
        help="Override CP-SAT second budgets (space-separated list)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.train_seqs or args.steps:
        train_vals = args.train_seqs if args.train_seqs else sorted({n for n, _ in SWEEP_GRID})
        step_vals = args.steps if args.steps else sorted({s for _, s in SWEEP_GRID})
        grid = _parse_grid(train_vals, step_vals)
    else:
        grid = SWEEP_GRID

    if not args.quiet:
        print()
        print("=== Joint-PDA Sweep ===")
        print(f"  runs={len(grid)}  seed={args.seed}  test_seqs_per_depth={args.test_seqs_per_depth}")
        print()

    rows = run_sweep(
        param_grid=grid,
        seed=args.seed,
        test_seqs_per_depth=args.test_seqs_per_depth,
        quiet=args.quiet,
    )
    _print_table(rows)

    if not args.quiet:
        full = sum(1 for row in rows if row["full_stack_discovered"])
        print(f"  full-stack discoveries: {full}/{len(rows)}")
        print()


if __name__ == "__main__":
    main()
