"""Run all benchmarks and emit a single summary table.

Runs:
  1. Depth generalization (PDA vs FSM vs PPM) — reproduce_depth_generalization
  2. Serialization (JSON vs MessagePack) — benchmark_serialization

Output: Markdown table to stdout; optional --csv-out for machine-readable.

Usage
-----
    py -3 scripts/run_all_benchmarks.py
    py -3 scripts/run_all_benchmarks.py --csv-out results/bench.csv --seed 42
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))


def _run_depth_gen(seed: int, train_seqs: int, test_seqs_per_depth: int):
    """Import and run depth generalization by file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "reproduce_depth_generalization",
        _REPO / "scripts" / "reproduce_depth_generalization.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run(
        seed=seed,
        train_seqs=train_seqs,
        test_seqs_per_depth=test_seqs_per_depth,
        quiet=True,
    )


def _run_serialization_benchmark(seed: int):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "benchmark_serialization",
        pathlib.Path(__file__).parent / "benchmark_serialization.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_benchmark(seed=seed)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--csv-out", type=pathlib.Path, default=None, metavar="PATH")
    parser.add_argument("--quiet", action="store_true", help="Minimal stdout")
    args = parser.parse_args()

    rows: list[dict] = []

    # --- Depth generalization ---
    if not args.quiet:
        print("Running depth generalization ...")
    depth_results = _run_depth_gen(
        seed=args.seed,
        train_seqs=300,
        test_seqs_per_depth=100,
    )
    from circuit_lm.metrics import accuracy_pct_times100

    max_depth = max(depth_results)
    ood_depths = [d for d in depth_results if d > 3]  # MAX_TRAIN_DEPTH=3
    if ood_depths:
        pda_ood = sum(accuracy_pct_times100(*depth_results[d]["pda"]) for d in ood_depths) // len(ood_depths)
        jpda_ood = sum(accuracy_pct_times100(*depth_results[d]["jpda"]) for d in ood_depths) // len(ood_depths)
        fsm_ood = sum(accuracy_pct_times100(*depth_results[d]["fsm"]) for d in ood_depths) // len(ood_depths)
        ppm_ood = sum(accuracy_pct_times100(*depth_results[d]["ppm"]) for d in ood_depths) // len(ood_depths)
    else:
        pda_ood = jpda_ood = fsm_ood = ppm_ood = 0

    rows.append({
        "benchmark": "depth_gen",
        "metric": "accuracy_bp_ood_avg",
        "model": "PDA-2ph",
        "value": pda_ood,
        "note": "out-of-distribution depths 4-8",
    })
    rows.append({
        "benchmark": "depth_gen",
        "metric": "accuracy_bp_ood_avg",
        "model": "PDA-jt",
        "value": jpda_ood,
        "note": "out-of-distribution depths 4-8",
    })
    rows.append({
        "benchmark": "depth_gen",
        "metric": "accuracy_bp_ood_avg",
        "model": "FSM",
        "value": fsm_ood,
        "note": "out-of-distribution depths 4-8",
    })
    rows.append({
        "benchmark": "depth_gen",
        "metric": "accuracy_bp_ood_avg",
        "model": "PPM",
        "value": ppm_ood,
        "note": "out-of-distribution depths 4-8",
    })

    # --- Serialization ---
    if not args.quiet:
        print("Running serialization benchmark ...")
    ser_rows = _run_serialization_benchmark(seed=args.seed)
    for r in ser_rows:
        rows.append({
            "benchmark": "serialization",
            "metric": "bytes",
            "model": f"{r['label']}_{r['format']}",
            "value": r["bytes"],
            "note": f"states={r['states']} vocab={r['vocab']}",
        })

    # --- Output ---
    if not args.quiet:
        print()
        print("=== CircuitLM benchmark summary ===")
        print(f"  seed={args.seed}")
        print()
    print("| benchmark     | metric              | model    | value  | note |")
    print("|---------------|---------------------|----------|--------|------|")
    for r in rows:
        print(f"| {r['benchmark']:<13} | {r['metric']:<19} | {r['model']:<8} | {r['value']:<6} | {r['note'][:30]} |")
    print()

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8") as f:
            f.write("benchmark,metric,model,value,note\n")
            for r in rows:
                f.write(f"{r['benchmark']},{r['metric']},{r['model']},{r['value']},{r['note']}\n")
        if not args.quiet:
            print(f"CSV written to {args.csv_out}")


if __name__ == "__main__":
    _main()
