"""Command-line interface for circuit_lm.

Commands
--------
  train   Train a new model (FSM or PDA) from a plain-text file.
  eval    Evaluate next-token accuracy on a data file.
  sample  Generate text from a trained model.

All subcommands import only stdlib + circuit_lm modules; no floats are
introduced at the CLI layer.
"""

from __future__ import annotations

import argparse
import sys


# ---------------------------------------------------------------------------
# Argparse validators
# ---------------------------------------------------------------------------


def _int_ge_0(value: str) -> int:
    """Argparse type: integer >= 0."""
    try:
        out = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if out < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0 (got {out})")
    return out


def _int_ge_1(value: str) -> int:
    """Argparse type: integer >= 1."""
    try:
        out = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if out < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {out})")
    return out


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Train a CircuitLM (FSM or PDA) and save it to a JSON file."""
    from circuit_lm.data import load_sequences, load_text
    from circuit_lm.io import save_model
    from circuit_lm.tokenizer import Tokenizer

    automaton = args.automaton  # "fsm", "pda", or "ppm"
    transition_steps: int | None = None
    emission_steps: int | None = None
    stack_steps: int | None = None

    if automaton in ("fsm", "pda"):
        if args.transition_steps is None and args.emission_steps is None:
            if automaton == "fsm":
                transition_steps = args.steps // 2
                emission_steps = args.steps - transition_steps
            else:
                phase1 = args.steps // 2
                phase2 = args.steps - phase1
                stack_steps = phase1
                transition_steps = phase2 // 2
                emission_steps = phase2 - transition_steps
        elif args.transition_steps is None or args.emission_steps is None:
            print(
                "[train] ERROR: --transition_steps and --emission_steps must be "
                "provided together.",
                file=sys.stderr,
            )
            return 1
        else:
            transition_steps = args.transition_steps
            emission_steps = args.emission_steps

    if automaton == "pda":
        if args.stack_steps is None and args.transition_steps is None and args.emission_steps is None:
            # Already resolved from legacy --steps fallback above.
            pass
        elif (
            args.stack_steps is None
            or args.transition_steps is None
            or args.emission_steps is None
        ):
            print(
                "[train] ERROR: PDA explicit budgets require all of "
                "--stack_steps, --transition_steps, and --emission_steps.",
                file=sys.stderr,
            )
            return 1
        else:
            stack_steps = args.stack_steps

    print(f"[train] automaton={automaton}  data={args.data!r}  out={args.out!r}")
    print(
        f"[train] vocab_size={args.vocab_size}  state_bits={args.state_bits}"
        f"  steps={args.steps}"
        + (f"  stack_depth={args.stack_depth}" if automaton == "pda" else "")
    )
    if automaton in ("fsm", "pda"):
        print(
            f"[train] context_len={args.context_len}"
            f"  top_k_coverage={args.top_k_coverage}"
            f"  transition_steps={transition_steps}"
            f"  emission_steps={emission_steps}"
            f"  refinement_rounds={args.refinement_rounds}"
        )
    if automaton == "pda":
        print(
            f"[train] max_push={args.max_push}  max_pop={args.max_pop}"
            f"  top_k_pairs={args.top_k_pairs}"
            f"  stack_steps={stack_steps}"
        )

    text = load_text(args.data)
    tokenizer = Tokenizer.from_text(
        text,
        vocab_size=args.vocab_size,
        mode=args.tokenizer,
        bpe_merges=args.bpe_merges,
    )
    print(
        f"[train] tokenizer={tokenizer.mode}  effective vocab_size={tokenizer.vocab_size}"
        + (f"  bpe_merges={args.bpe_merges}" if tokenizer.mode == "bpe" else "")
    )

    sequences = load_sequences(args.data, tokenizer)
    print(f"[train] loaded {len(sequences)} sequences")
    if not sequences:
        print("[train] ERROR: no sequences found – is the file long enough?",
              file=sys.stderr)
        return 1

    if automaton == "pda":
        from circuit_lm.train_pda_cpsat import train_pda
        model = train_pda(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            stack_depth=args.stack_depth,
            steps=args.steps,
            context_len=args.context_len,
            max_push=args.max_push,
            max_pop=args.max_pop,
            top_k_pairs=args.top_k_pairs,
            top_k_coverage=args.top_k_coverage,
            stack_steps=stack_steps,
            transition_steps=transition_steps,
            emission_steps=emission_steps,
            refinement_rounds=args.refinement_rounds,
        )
        push_n = len(model.push_tokens)
        pop_n  = len(model.pop_tokens)
        print(f"[train] PDA configs={len(model.config_counts)}"
              f"  push_tokens={push_n}  pop_tokens={pop_n}")
    elif automaton == "ppm":
        from circuit_lm.train_ppm import train_ppm
        print(f"[train] PPM order={args.order}  (no CP-SAT; pure counting)")
        model = train_ppm(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            order=args.order,
        )
        print(f"[train] PPM contexts={len(model.counts)}")
    else:
        from circuit_lm.train_cpsat import train
        model = train(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            steps=args.steps,
            context_len=args.context_len,
            top_k_coverage=args.top_k_coverage,
            transition_steps=transition_steps,
            emission_steps=emission_steps,
            refinement_rounds=args.refinement_rounds,
        )

    save_model(model, tokenizer, args.out)
    print(f"[train] model saved -> {args.out!r}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate next-token prediction accuracy."""
    from circuit_lm.eval import evaluate_any
    from circuit_lm.io import load_model
    from circuit_lm.metrics import format_accuracy
    from circuit_lm.data import load_sequences

    print(f"[eval] model={args.model!r}  data={args.data!r}")

    model, tokenizer = load_model(args.model)
    sequences = load_sequences(args.data, tokenizer)
    print(f"[eval] {len(sequences)} sequences  vocab_size={tokenizer.vocab_size}")

    if not sequences:
        print("[eval] ERROR: no sequences found.", file=sys.stderr)
        return 1

    results = evaluate_any(model, sequences, per_token=args.per_token)
    correct = results["correct"]
    total   = results["total"]
    print(f"[eval] correct={correct}  total={total}  accuracy={format_accuracy(correct, total)}")
    if args.per_token:
        per_token = results.get("per_token", {})
        if isinstance(per_token, dict) and per_token:
            ranked_tokens = sorted(
                per_token,
                key=lambda tok: (-per_token[tok]["total"], tok),
            )
            limit = args.per_token_limit if args.per_token_limit > 0 else len(ranked_tokens)
            print(f"[eval] per-token breakdown (top {min(limit, len(ranked_tokens))} by frequency)")
            for tok in ranked_tokens[:limit]:
                stats = per_token[tok]
                tok_text = repr(tokenizer.decode([tok]))
                tok_correct = stats["correct"]
                tok_total = stats["total"]
                tok_acc = format_accuracy(tok_correct, tok_total)
                print(
                    f"[eval] tok={tok}  text={tok_text}  "
                    f"correct={tok_correct}  total={tok_total}  accuracy={tok_acc}"
                )
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Sample tokens from a trained model."""
    from circuit_lm.infer import decode_sample
    from circuit_lm.io import load_model

    print(
        f"[sample] model={args.model!r}  max_tokens={args.max_tokens}  seed={args.seed}"
        f"  top_k={args.top_k}  repeat_penalty_div={args.repeat_penalty_div}"
        f"  repeat_window={args.repeat_window}"
    )

    model, tokenizer = load_model(args.model)
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"[sample] prompt={args.prompt!r}  ({len(prompt_ids)} tokens)")

    out_ids = decode_sample(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_k=args.top_k,
        repeat_penalty_div=args.repeat_penalty_div,
        repeat_window=args.repeat_window,
    )
    generated = tokenizer.decode(out_ids[len(prompt_ids):])
    print(f"[sample] {args.prompt}{generated}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuit-lm",
        description=(
            "Circuit language model – integer FSM/PDA trained with OR-Tools CP-SAT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train ---------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train a new model.")
    p_train.add_argument("--data", required=True, metavar="PATH",
                         help="Path to training text file (UTF-8).")
    p_train.add_argument("--out", default="model.json", metavar="PATH",
                         help="Output model JSON path (default: model.json).")
    p_train.add_argument("--vocab_size", type=_int_ge_0, default=256, metavar="N",
                         help="Maximum vocabulary size (default: 256).")
    p_train.add_argument("--state_bits", type=_int_ge_0, default=4, metavar="S",
                         help="State width in bits; num_states = 2**S (default: 4).")
    p_train.add_argument("--steps", type=_int_ge_0, default=10, metavar="K",
                         help="Legacy total CP-SAT time budget in integer seconds (default: 10).")
    p_train.add_argument(
        "--transition_steps", type=_int_ge_0, default=None, metavar="K",
        help=(
            "Total transition CP-SAT budget in integer seconds across all "
            "passes (FSM/PDA). Must be paired with --emission_steps."
        ),
    )
    p_train.add_argument(
        "--emission_steps", type=_int_ge_0, default=None, metavar="K",
        help=(
            "Total emission CP-SAT budget in integer seconds across all "
            "passes (FSM/PDA). Must be paired with --transition_steps."
        ),
    )
    p_train.add_argument(
        "--refinement_rounds", type=_int_ge_0, default=1, metavar="R",
        help=(
            "Additional EM-like transition/emission refinement rounds after "
            "the initial hashed pass (default: 1, ignored for PPM)."
        ),
    )
    p_train.add_argument(
        "--context_len", type=_int_ge_0, default=4, metavar="N",
        help=(
            "Context window length for hashed FSM state assignment "
            "(default: 4, ignored for PPM)."
        ),
    )
    p_train.add_argument(
        "--top_k_coverage", type=_int_ge_0, default=16, metavar="K",
        help=(
            "Top-K token coverage constraint for CP-SAT emission optimisation "
            "(default: 16, ignored for PPM)."
        ),
    )
    p_train.add_argument(
        "--tokenizer", choices=["char", "bpe"], default="char",
        help="Tokenizer mode: character-level (char) or simple BPE (bpe).",
    )
    p_train.add_argument(
        "--bpe_merges", type=_int_ge_0, default=256, metavar="N",
        help="Maximum BPE merge operations (default: 256, ignored for char tokenizer).",
    )
    p_train.add_argument(
        "--automaton", choices=["fsm", "pda", "ppm"], default="fsm",
        help=(
            "Automaton type: 'fsm' (plain FSM, default), 'pda' (stack-augmented), "
            "or 'ppm' (context-tree, no CP-SAT)."
        ),
    )
    p_train.add_argument(
        "--stack_depth", type=_int_ge_0, default=4, metavar="D",
        help="Maximum stack depth for PDA training (default: 4, ignored for FSM/PPM).",
    )
    p_train.add_argument(
        "--stack_steps", type=_int_ge_0, default=None, metavar="K",
        help=(
            "Total stack-policy (PDA Phase 1) CP-SAT budget in integer seconds. "
            "When set, use with --transition_steps and --emission_steps."
        ),
    )
    p_train.add_argument(
        "--max_push", type=_int_ge_0, default=16, metavar="N",
        help="Maximum number of PUSH tokens for PDA Phase 1 (default: 16).",
    )
    p_train.add_argument(
        "--max_pop", type=_int_ge_0, default=16, metavar="N",
        help="Maximum number of POP tokens for PDA Phase 1 (default: 16).",
    )
    p_train.add_argument(
        "--top_k_pairs", type=_int_ge_0, default=256, metavar="K",
        help="Top co-occurrence pairs considered in PDA Phase 1 (default: 256).",
    )
    p_train.add_argument(
        "--order", type=_int_ge_0, default=4, metavar="N",
        help="Context order for PPM training (default: 4, ignored for FSM/PDA).",
    )
    p_train.set_defaults(func=cmd_train)

    # -- eval ----------------------------------------------------------------
    p_eval = sub.add_parser("eval", help="Evaluate a trained model.")
    p_eval.add_argument("--data", required=True, metavar="PATH",
                        help="Path to evaluation text file.")
    p_eval.add_argument("--model", default="model.json", metavar="PATH",
                        help="Model JSON path (default: model.json).")
    p_eval.add_argument(
        "--per_token", action="store_true",
        help="Print per-token (gold-token) accuracy breakdown.",
    )
    p_eval.add_argument(
        "--per_token_limit", type=int, default=20, metavar="N",
        help="Max tokens to print in per-token breakdown (<=0 prints all).",
    )
    p_eval.set_defaults(func=cmd_eval)

    # -- sample --------------------------------------------------------------
    p_sample = sub.add_parser("sample", help="Sample text from a trained model.")
    p_sample.add_argument("--prompt", default="", metavar="TEXT",
                          help="Prompt string (default: empty).")
    p_sample.add_argument("--model", default="model.json", metavar="PATH",
                          help="Model JSON path (default: model.json).")
    p_sample.add_argument("--max_tokens", type=_int_ge_0, default=64, metavar="M",
                          help="Number of tokens to generate (default: 64).")
    p_sample.add_argument("--seed", type=int, default=42, metavar="SEED",
                          help="Integer random seed (default: 42).")
    p_sample.add_argument(
        "--top_k", type=_int_ge_0, default=0, metavar="K",
        help="Keep only top-K integer weights during sampling (0 disables).",
    )
    p_sample.add_argument(
        "--repeat_penalty_div", type=_int_ge_1, default=1, metavar="D",
        help="Divide repeated-token weights by D (1 disables).",
    )
    p_sample.add_argument(
        "--repeat_window", type=_int_ge_0, default=0, metavar="N",
        help="Penalise tokens seen in the last N tokens (0 = full history if penalty enabled).",
    )
    p_sample.set_defaults(func=cmd_sample)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
