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
import pathlib
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
        push_n = len({tok for (_, tok, _) in model.push_configs})
        pop_n  = len({tok for (_, tok, _) in model.pop_configs})
        print(f"[train] PDA configs={len(model.config_counts)}"
              f"  push_token_ids={push_n}  pop_token_ids={pop_n}")
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


def cmd_trace(args: argparse.Namespace) -> int:
    """Print step-by-step trace of state and top-k predictions."""
    import json as _json
    from circuit_lm.io import load_model
    from circuit_lm.trace import trace_steps, TraceStep

    print(f"[trace] model={args.model!r}  prompt={args.prompt!r}  top_k={args.top_k}")
    model, tokenizer = load_model(args.model)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        print("[trace] empty prompt")
        return 0

    steps: list[dict] = []
    for s in trace_steps(model, prompt_ids, top_k=args.top_k):
        tok_repr = repr(tokenizer.decode([s.token_id]))
        stack_str = f" stack_top={s.stack_top}" if s.stack_top is not None else ""
        top_k_repr = tokenizer.decode(s.top_k_token_ids)
        line = f"  step={s.step} token_id={s.token_id} token={tok_repr} state={s.state}{stack_str} top_k={top_k_repr!r}"
        print(line)
        steps.append({
            "step": s.step,
            "token_id": s.token_id,
            "token": tokenizer.decode([s.token_id]),
            "state": s.state,
            "stack_top": s.stack_top,
            "top_k_token_ids": s.top_k_token_ids,
            "top_k_tokens": tokenizer.decode(s.top_k_token_ids),
        })

    if args.json_out:
        args.json_out.write_text(_json.dumps(steps, indent=2), encoding="utf-8")
        print(f"[trace] JSON written to {args.json_out}")
    return 0


def _import_hybrid(*names: str):
    """Import from src.hybrid; ensure repo root is on path when running as installed cmd."""
    import pathlib

    def _get(mod):  # noqa: B008
        return tuple(getattr(mod, n) for n in names)

    try:
        from src import hybrid as _mod
        return _get(_mod)
    except ImportError:
        pass
    for candidate in [pathlib.Path.cwd(), pathlib.Path.cwd().parent]:
        if (candidate / "src" / "hybrid.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            from src import hybrid as _mod
            return _get(_mod)
    try:
        import hybrid as _mod
        return _get(_mod)
    except ImportError:
        return None


def cmd_hybrid_train(args: argparse.Namespace) -> int:
    """Train the neural corrector on top of an existing circuit model."""
    out = _import_hybrid("train_hybrid")
    if out is None:
        print(
            "[hybrid-train] ERROR: hybrid module not found. Run from repo root or "
            "install with: pip install -e '.[dev]' and run from repo root.",
            file=sys.stderr,
        )
        return 1
    (train_hybrid,) = out

    print(
        f"[hybrid-train] circuit={args.circuit!r}  data={args.data!r}  out={args.out!r}"
        f"  epochs={args.epochs}  max_examples={args.max_examples}"
    )
    use_ssd = args.ssd and not args.lstm
    train_hybrid(
        circuit_path=args.circuit,
        data_path=args.data,
        output_path=args.out,
        num_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        circuit_weight=args.circuit_weight,
        max_examples=args.max_examples,
        max_context_len=args.max_context_len,
        use_ssd=use_ssd,
    )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive chat using a trained model (User: / Assistant: format)."""
    from circuit_lm.chat import (
        ASSISTANT_PREFIX,
        USER_PREFIX,
        generate_reply,
        prompt_for_assistant_reply,
    )
    from circuit_lm.io import load_model

    use_hybrid = getattr(args, 'corrector', None) is not None

    if use_hybrid:
        out = _import_hybrid("HybridModel", "generate_reply_hybrid")
        if out is None:
            print(
                "[chat] ERROR: hybrid module not found. Run from repo root.",
                file=sys.stderr,
            )
            return 1
        HybridModel, generate_reply_hybrid = out
        hybrid, tokenizer = HybridModel.load(args.model, args.corrector)
        model = hybrid
    else:
        model, tokenizer = load_model(args.model)

    print(
        f"[chat] model={args.model!r}  max_tokens={args.max_tokens}  seed={args.seed}"
        + (f"  corrector={args.corrector!r}" if use_hybrid else "")
    )

    turns: list[tuple[str, str]] = []
    print(USER_PREFIX, end="", flush=True)

    while True:
        try:
            user_line = input()
        except EOFError:
            break
        if not user_line.strip():
            continue
        turns.append(("user", user_line))

        prompt_str = prompt_for_assistant_reply(turns, system_preamble=args.system)
        prompt_ids = tokenizer.encode(prompt_str)

        stop_sequence = tokenizer.encode("\n\n") if args.paragraph else None
        if use_hybrid:
            reply_ids = generate_reply_hybrid(
                hybrid=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                max_tokens=args.max_tokens,
                stop_token_ids=tokenizer.encode("\n"),
                stop_sequence=stop_sequence,
            )
        else:
            reply_ids = generate_reply(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                max_tokens=args.max_tokens,
                seed=args.seed,
                stop_sequence=stop_sequence,
                top_k=args.top_k,
                repeat_penalty_div=args.repeat_penalty_div,
                repeat_window=args.repeat_window,
            )
        reply_text = tokenizer.decode(reply_ids)
        turns.append(("assistant", reply_text))

        print(ASSISTANT_PREFIX + reply_text)
        print()
        print(USER_PREFIX, end="", flush=True)
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

    # -- trace ----------------------------------------------------------------
    p_trace = sub.add_parser("trace", help="Step-by-step trace of state and top-k predictions.")
    p_trace.add_argument("--prompt", default="", metavar="TEXT", help="Prompt string.")
    p_trace.add_argument("--model", default="model.json", metavar="PATH", help="Model path.")
    p_trace.add_argument("--top_k", type=_int_ge_0, default=5, metavar="K", help="Number of top predictions per step (default: 5).")
    p_trace.add_argument("--json-out", type=pathlib.Path, default=None, metavar="PATH", help="Write trace to JSON file.")
    p_trace.set_defaults(func=cmd_trace)

    # -- hybrid-train --------------------------------------------------------
    p_hybrid = sub.add_parser(
        "hybrid-train",
        help="Train neural corrector on top of an existing circuit model.",
    )
    p_hybrid.add_argument("--circuit", required=True, metavar="PATH",
                         help="Path to trained circuit model .json.")
    p_hybrid.add_argument("--data", required=True, metavar="PATH",
                         help="Path to training text file.")
    p_hybrid.add_argument("--out", default="corrector.pt", metavar="PATH",
                         help="Output path for corrector checkpoint (default: corrector.pt).")
    p_hybrid.add_argument("--epochs", type=_int_ge_1, default=3, metavar="N",
                         help="Training epochs (default: 3).")
    p_hybrid.add_argument("--batch", type=_int_ge_1, default=64, metavar="N",
                         help="Batch size (default: 64).")
    p_hybrid.add_argument("--lr", type=float, default=1e-3, metavar="R",
                         help="Learning rate (default: 1e-3).")
    p_hybrid.add_argument("--circuit-weight", type=float, default=0.5, metavar="W",
                         help="Circuit weight in blend (default: 0.5).")
    p_hybrid.add_argument("--max-examples", type=_int_ge_0, default=50000, metavar="N",
                         help="Max training examples (default: 50000).")
    p_hybrid.add_argument("--max-context-len", type=_int_ge_0, default=32, metavar="N",
                         help="Context length for corrector (default: 32).")
    p_hybrid.add_argument("--ssd", action="store_true",
                         help="Use SSD context encoder instead of LSTM (faster at large vocab).")
    p_hybrid.add_argument("--lstm", action="store_true",
                         help="Force LSTM context encoder (overrides --ssd).")
    p_hybrid.set_defaults(func=cmd_hybrid_train)

    # -- chat ----------------------------------------------------------------
    p_chat = sub.add_parser("chat", help="Interactive chat (User: / Assistant: format).")
    p_chat.add_argument("--model", default="model.json", metavar="PATH",
                        help="Model JSON path (default: model.json).")
    p_chat.add_argument("--corrector", default=None, metavar="PATH",
                        help="Path to corrector .pt for hybrid chat (optional).")
    p_chat.add_argument("--max_tokens", type=_int_ge_0, default=128, metavar="M",
                        help="Max tokens per reply (default: 128).")
    p_chat.add_argument("--seed", type=int, default=42, metavar="SEED",
                        help="Random seed (default: 42).")
    p_chat.add_argument(
        "--top_k", type=_int_ge_0, default=0, metavar="K",
        help="Sampling top-k (0 disables).",
    )
    p_chat.add_argument(
        "--repeat_penalty_div", type=_int_ge_1, default=1, metavar="D",
        help="Repetition penalty divisor (1 disables).",
    )
    p_chat.add_argument(
        "--repeat_window", type=_int_ge_0, default=0, metavar="N",
        help="Repetition penalty window (0 = full history).",
    )
    p_chat.add_argument(
        "--system", default=None, metavar="TEXT",
        help="System preamble for conversationalist behavior (default: brief helpful assistant). Use '' to disable.",
    )
    p_chat.add_argument(
        "--paragraph", action="store_true",
        help="Allow multi-sentence replies (stop at double newline).",
    )
    p_chat.add_argument(
        "--system", default=None, metavar="TEXT",
        help="System preamble for conversationalist behavior (default: brief helpful assistant). Use '' to disable.",
    )
    p_chat.set_defaults(func=cmd_chat)

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
