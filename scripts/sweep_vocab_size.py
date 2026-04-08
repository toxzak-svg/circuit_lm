"""Sweep vocabulary sizes and report next-token accuracy.

Usage:
    py -3.12 scripts/sweep_vocab_size.py --data combined_data.txt --vocab_sizes 256 512 1024 2048 4096 8192 --automaton fsm --steps 20
    py -3.12 scripts/sweep_vocab_size.py --data combined_data.txt --vocab_sizes 256 512 1024 2048 --automaton pda --steps 40 --stack_depth 4
"""

import argparse
import pathlib
import time
import sys

import circuit_lm.data as data_module
from circuit_lm.io import save_model
from circuit_lm.tokenizer import Tokenizer


def _int_ge_1(value: str) -> int:
    i = int(value)
    if i < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {i})")
    return i


def run_sweep(
    data_path: str,
    vocab_sizes: list[int],
    automaton: str,
    steps: int,
    state_bits: int,
    stack_depth: int,
    context_len: int,
    refinement_rounds: int,
    output_csv: str | None,
) -> None:
    """Train and evaluate at each vocab size, print a table."""

    rows: list[dict] = []
    headers = [
        "vocab_size",
        "bpe_merges",
        "effective_vocab",
        "train_seqs",
        "total_tokens",
        "train_sec",
        "eval_sec",
        "correct",
        "total",
        "accuracy",
    ]
    print(",".join(headers))
    all_ok = True

    for V in vocab_sizes:
        print(f"\n{'='*60}")
        print(f"  vocab_size={V}  automaton={automaton}  steps={steps}")
        print(f"{'='*60}")

        try:
            t0 = time.perf_counter_ns()
            text = data_module.load_text(data_path)
            tok = Tokenizer.from_text(text, vocab_size=V, mode="bpe", bpe_merges=V)
            sequences = data_module.load_sequences(data_path, tok)
            t_load = (time.perf_counter_ns() - t0) // 1_000_000

            total_tokens = sum(len(s) for s in sequences)
            print(
                f"  tokenizer: mode={tok.mode}  effective_vocab={tok.vocab_size}"
                f"  bpe_merges={V}  seqs={len(sequences)}  tokens={total_tokens}"
            )

            if not sequences:
                print(f"  SKIP: no sequences loaded")
                continue

            t1 = time.perf_counter_ns()
            if automaton == "pda":
                from circuit_lm.train_pda_cpsat import train_pda

                # Budget split: stack=40%, transition=30%, emission=30%
                phase1 = max(1, steps // 3)
                phase2 = steps - phase1
                transition_steps = phase2 // 2
                emission_steps = phase2 - transition_steps
                model = train_pda(
                    sequences=sequences,
                    vocab_size=tok.vocab_size,
                    state_bits=state_bits,
                    stack_depth=stack_depth,
                    steps=steps,
                    context_len=context_len,
                    max_push=16,
                    max_pop=16,
                    top_k_pairs=256,
                    top_k_coverage=16,
                    stack_steps=phase1,
                    transition_steps=transition_steps,
                    emission_steps=emission_steps,
                    refinement_rounds=refinement_rounds,
                )
            else:
                from circuit_lm.train_cpsat import train

                transition_steps = steps // 2
                emission_steps = steps - transition_steps
                model = train(
                    sequences=sequences,
                    vocab_size=tok.vocab_size,
                    state_bits=state_bits,
                    steps=steps,
                    context_len=context_len,
                    top_k_coverage=16,
                    transition_steps=transition_steps,
                    emission_steps=emission_steps,
                    refinement_rounds=refinement_rounds,
                )
            t_train = (time.perf_counter_ns() - t1) // 1_000_000

            t2 = time.perf_counter_ns()
            from circuit_lm.eval import evaluate_any
            from circuit_lm.metrics import format_accuracy

            results = evaluate_any(model, sequences)
            t_eval = (time.perf_counter_ns() - t2) // 1_000_000

            correct = results["correct"]
            total = results["total"]
            acc = format_accuracy(correct, total)

            print(
                f"  train={t_train}ms  eval={t_eval}ms"
                f"  correct={correct}  total={total}  accuracy={acc}"
            )

            row = {
                "vocab_size": V,
                "bpe_merges": V,
                "effective_vocab": tok.vocab_size,
                "train_seqs": len(sequences),
                "total_tokens": total_tokens,
                "train_sec": t_train,
                "eval_sec": t_eval,
                "correct": correct,
                "total": total,
                "accuracy": acc,
            }
            rows.append(row)
            print(",".join(str(row[h]) for h in headers))

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            all_ok = False
            continue

    if output_csv and rows:
        import csv

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written -> {output_csv}")

    if not all_ok:
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep vocabulary sizes for CircuitLM.")
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Training text file.",
    )
    parser.add_argument(
        "--vocab_sizes",
        type=_int_ge_1,
        nargs="+",
        required=True,
        metavar="V",
        help="Vocabulary sizes to sweep.",
    )
    parser.add_argument(
        "--automaton",
        choices=["fsm", "pda"],
        default="fsm",
        help="Automaton type (default: fsm).",
    )
    parser.add_argument(
        "--steps",
        type=_int_ge_1,
        default=20,
        metavar="K",
        help="Total CP-SAT budget in seconds (default: 20).",
    )
    parser.add_argument(
        "--state_bits",
        type=_int_ge_1,
        default=5,
        metavar="S",
        help="State bits (2**S states, default: 5 => 32 states).",
    )
    parser.add_argument(
        "--stack_depth",
        type=_int_ge_1,
        default=4,
        metavar="D",
        help="Stack depth for PDA (default: 4).",
    )
    parser.add_argument(
        "--context_len",
        type=_int_ge_1,
        default=4,
        metavar="N",
        help="Context length (default: 4).",
    )
    parser.add_argument(
        "--refinement_rounds",
        type=_int_ge_1,
        default=1,
        metavar="R",
        help="Refinement rounds (default: 1).",
    )
    parser.add_argument(
        "--csv_out",
        default=None,
        metavar="PATH",
        help="Write results to CSV file.",
    )
    args = parser.parse_args()

    run_sweep(
        data_path=args.data,
        vocab_sizes=args.vocab_sizes,
        automaton=args.automaton,
        steps=args.steps,
        state_bits=args.state_bits,
        stack_depth=args.stack_depth,
        context_len=args.context_len,
        refinement_rounds=args.refinement_rounds,
        output_csv=args.csv_out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
