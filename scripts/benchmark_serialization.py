"""Serialization benchmark for CircuitLM and PDACircuitLM models.

Measures round-trip byte size, save time, and load time for three model
configurations across JSON and MessagePack formats. Each model is trained
once, then reused for both format rows (no duplicate CP-SAT solves).

All timings are integer milliseconds.  No floats anywhere.

Usage
-----
    py -3 scripts/benchmark_serialization.py
    py -3 scripts/benchmark_serialization.py --csv-out results/ser_bench.csv
"""
from __future__ import annotations

import argparse
import pathlib
import random as _random
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.circuits  import CircuitLM
from circuit_lm.io        import (
    has_msgpack,
    load_model,
    load_msgpack,
    save_model,
    save_msgpack,
)
from circuit_lm.pda       import PDACircuitLM
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat      import train as train_fsm
from circuit_lm.train_pda_cpsat  import train_pda

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_random_seqs(
    num_seqs: int, seq_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    """Random integer token sequences — no linguistic structure needed for a serialisation bench."""
    rng = _random.Random(seed)
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
        for _ in range(num_seqs)
    ]


def _make_tokenizer(vocab_size: int) -> Tokenizer:
    """Build a minimal char tokenizer with *vocab_size* printable characters."""
    chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789!@#$%^&*()-+=[]{}|;:,.<>?"
    )
    text = chars[:vocab_size] * 20
    return Tokenizer.from_text(text, vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# Benchmark row
# ---------------------------------------------------------------------------


def _train_one(
    model_type: str,
    num_states: int,
    vocab_size: int,
    seed: int,
):
    """Train one model and tokenizer used by all serialization formats."""
    seqs = _make_random_seqs(num_seqs=30, seq_len=20, vocab_size=vocab_size, seed=seed)
    tokenizer = _make_tokenizer(vocab_size)
    state_bits = num_states.bit_length() - 1

    if model_type == "fsm":
        model = train_fsm(
            sequences=seqs,
            vocab_size=vocab_size,
            state_bits=state_bits,
            steps=3,
        )
    else:
        model = train_pda(
            sequences=seqs,
            vocab_size=vocab_size,
            state_bits=state_bits,
            stack_depth=1,
            steps=3,
        )
    return model, tokenizer


def _bench_one(
    label: str,
    model_type: str,
    num_states: int,
    vocab_size: int,
    model,
    tokenizer: Tokenizer,
    ser_format: str,
    tmp_dir: pathlib.Path,
) -> dict:
    """Save, load, and compare one model for one serialization format."""
    suffix = ".json" if ser_format == "json" else ".msgpack"
    out_path = tmp_dir / f"{label}-{ser_format}{suffix}"

    # --- save ---
    t0 = time.perf_counter_ns()
    if ser_format == "json":
        save_model(model, tokenizer, out_path)
    else:
        save_msgpack(model, tokenizer, out_path)
    save_ms: int = (time.perf_counter_ns() - t0) // 1_000_000

    file_bytes: int = len(out_path.read_bytes())

    # --- load ---
    t1 = time.perf_counter_ns()
    if ser_format == "json":
        loaded_model, _ = load_model(out_path)
    else:
        loaded_model, _ = load_msgpack(out_path)
    load_ms: int = (time.perf_counter_ns() - t1) // 1_000_000

    # --- roundtrip check ---
    roundtrip_ok: int = 1 if _models_equal(model, loaded_model) else 0

    return {
        "label":        label,
        "type":         model_type,
        "format":       ser_format,
        "states":       num_states,
        "vocab":        vocab_size,
        "bytes":        file_bytes,
        "save_ms":      save_ms,
        "load_ms":      load_ms,
        "roundtrip_ok": roundtrip_ok,
    }


def _models_equal(a, b) -> bool:
    """Field-by-field equality for CircuitLM and PDACircuitLM."""
    if type(a) is not type(b):
        return False
    if isinstance(a, PDACircuitLM):
        return (
            a.vocab_size             == b.vocab_size
            and a.num_states         == b.num_states
            and a.transitions        == b.transitions
            and a.push_configs       == b.push_configs
            and a.pop_configs        == b.pop_configs
            and a.config_pred_tokens == b.config_pred_tokens
        )
    # CircuitLM (FSM)
    return (
        a.vocab_size      == b.vocab_size
        and a.num_states  == b.num_states
        and a.transitions == b.transitions
        and a.pred_tokens == b.pred_tokens
    )


# ---------------------------------------------------------------------------
# BENCH_CONFIGS: (label, type, num_states, vocab_size)
# ---------------------------------------------------------------------------

BENCH_CONFIGS: list[tuple[str, str, int, int]] = [
    ("fsm-sm", "fsm",  8, 30),
    ("pda-sm", "pda",  8, 30),
    ("pda-md", "pda", 16, 30),
]

# ---------------------------------------------------------------------------
# Public run function (importable for tests)
# ---------------------------------------------------------------------------


def run_benchmark(
    seed: int = 42,
    tmp_dir: pathlib.Path | None = None,
) -> list[dict]:
    """Run the serialization benchmark and return a list of row dicts."""
    formats: list[str] = ["json"]
    if has_msgpack():
        formats.append("msgpack")

    rows: list[dict] = []

    def _run_with_tmp(td_path: pathlib.Path) -> None:
        for i, (label, mtype, ns, vs) in enumerate(BENCH_CONFIGS):
            model, tokenizer = _train_one(
                model_type=mtype,
                num_states=ns,
                vocab_size=vs,
                seed=seed + i,
            )
            for ser_format in formats:
                rows.append(
                    _bench_one(
                        label=label,
                        model_type=mtype,
                        num_states=ns,
                        vocab_size=vs,
                        model=model,
                        tokenizer=tokenizer,
                        ser_format=ser_format,
                        tmp_dir=td_path,
                    )
                )

    if tmp_dir is not None:
        _run_with_tmp(tmp_dir)
        return rows

    with tempfile.TemporaryDirectory() as td:
        _run_with_tmp(pathlib.Path(td))
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_HDR = (
    f"{'label':<8}  {'type':<4}  {'fmt':<7}  {'states':>6}  {'vocab':>5}  "
    f"{'bytes':>8}  {'save_ms':>7}  {'load_ms':>7}  {'ok':>2}"
)
_SEP = "-" * len(_HDR)


def _print_table(rows: list[dict]) -> None:
    print(_SEP)
    print(_HDR)
    print(_SEP)
    for r in rows:
        print(
            f"{r['label']:<8}  {r['type']:<4}  {r['format']:<7}  {r['states']:>6}  {r['vocab']:>5}"
            f"  {r['bytes']:>8}  {r['save_ms']:>7}  {r['load_ms']:>7}  {r['roundtrip_ok']:>2}"
        )
    print(_SEP)


def _write_csv(rows: list[dict], path: pathlib.Path) -> None:
    header = "label,type,format,states,vocab,bytes,save_ms,load_ms,roundtrip_ok"
    lines  = [header]
    for r in rows:
        lines.append(
            f"{r['label']},{r['type']},{r['format']},{r['states']},{r['vocab']},"
            f"{r['bytes']},{r['save_ms']},{r['load_ms']},{r['roundtrip_ok']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--csv-out", type=pathlib.Path, default=None,
                   metavar="PATH", help="Write results to CSV file")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    print()
    print("=== Serialization Benchmark (JSON + MessagePack) ===")
    print(f"  seed={args.seed}  configs={len(BENCH_CONFIGS)}")
    if not has_msgpack():
        print("  note: msgpack not installed, running JSON rows only")
    print()
    rows = run_benchmark(seed=args.seed)
    _print_table(rows)
    if args.csv_out:
        _write_csv(rows, args.csv_out)
        print(f"  CSV written to {args.csv_out}")
    print()


if __name__ == "__main__":
    main()
