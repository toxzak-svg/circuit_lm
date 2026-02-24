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
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Train a CircuitLM (FSM or PDA) and save it to a JSON file."""
    from circuit_lm.data import load_sequences, load_text
    from circuit_lm.io import save_model
    from circuit_lm.tokenizer import Tokenizer

    automaton = args.automaton  # "fsm" or "pda"
    print(f"[train] automaton={automaton}  data={args.data!r}  out={args.out!r}")
    print(
        f"[train] vocab_size={args.vocab_size}  state_bits={args.state_bits}"
        f"  steps={args.steps}"
        + (f"  stack_depth={args.stack_depth}" if automaton == "pda" else "")
    )

    text = load_text(args.data)
    tokenizer = Tokenizer.from_text(text, vocab_size=args.vocab_size)
    print(f"[train] effective vocab_size={tokenizer.vocab_size}")

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

    results = evaluate_any(model, sequences)
    correct = results["correct"]
    total   = results["total"]
    print(f"[eval] correct={correct}  total={total}  accuracy={format_accuracy(correct, total)}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Sample tokens from a trained model."""
    from circuit_lm.infer import decode_sample
    from circuit_lm.io import load_model

    print(f"[sample] model={args.model!r}  max_tokens={args.max_tokens}  seed={args.seed}")

    model, tokenizer = load_model(args.model)
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"[sample] prompt={args.prompt!r}  ({len(prompt_ids)} tokens)")

    out_ids = decode_sample(
        model=model,
        prompt_ids=prompt_ids,
        max_tokens=args.max_tokens,
        seed=args.seed,
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
    p_train.add_argument("--vocab_size", type=int, default=256, metavar="N",
                         help="Maximum vocabulary size (default: 256).")
    p_train.add_argument("--state_bits", type=int, default=4, metavar="S",
                         help="State width in bits; num_states = 2**S (default: 4).")
    p_train.add_argument("--steps", type=int, default=10, metavar="K",
                         help="Total CP-SAT time budget in integer seconds (default: 10).")
    p_train.add_argument(
        "--automaton", choices=["fsm", "pda", "ppm"], default="fsm",
        help=(
            "Automaton type: 'fsm' (plain FSM, default), 'pda' (stack-augmented), "
            "or 'ppm' (context-tree, no CP-SAT)."
        ),
    )
    p_train.add_argument(
        "--stack_depth", type=int, default=4, metavar="D",
        help="Maximum stack depth for PDA training (default: 4, ignored for FSM/PPM).",
    )
    p_train.add_argument(
        "--order", type=int, default=4, metavar="N",
        help="Context order for PPM training (default: 4, ignored for FSM/PDA).",
    )
    p_train.set_defaults(func=cmd_train)

    # -- eval ----------------------------------------------------------------
    p_eval = sub.add_parser("eval", help="Evaluate a trained model.")
    p_eval.add_argument("--data", required=True, metavar="PATH",
                        help="Path to evaluation text file.")
    p_eval.add_argument("--model", default="model.json", metavar="PATH",
                        help="Model JSON path (default: model.json).")
    p_eval.set_defaults(func=cmd_eval)

    # -- sample --------------------------------------------------------------
    p_sample = sub.add_parser("sample", help="Sample text from a trained model.")
    p_sample.add_argument("--prompt", default="", metavar="TEXT",
                          help="Prompt string (default: empty).")
    p_sample.add_argument("--model", default="model.json", metavar="PATH",
                          help="Model JSON path (default: model.json).")
    p_sample.add_argument("--max_tokens", type=int, default=64, metavar="M",
                          help="Number of tokens to generate (default: 64).")
    p_sample.add_argument("--seed", type=int, default=42, metavar="SEED",
                          help="Integer random seed (default: 42).")
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
