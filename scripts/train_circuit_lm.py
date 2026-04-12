"""
train_circuit_lm.py

Train a hybrid CircuitLM (FSM/PDA + neural corrector) on the personal dataset.
CPU-friendly defaults. Use --vocab-size 4096 for a bigger vocabulary.

Usage:
    py -3.12 scripts/train_circuit_lm.py --data research_evolver_data.txt --vocab-size 4096 --epochs 5
"""

import argparse
import sys
import time
from pathlib import Path
import os

# Allow imports from src/ and circuit_lm/ subdirs when run as script
_script_dir = Path(__file__).parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
os.chdir(_repo_root)

# Ensure src in path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from circuit_lm.data import load_text, load_sequences
from circuit_lm.io import save_model, load_model
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat import train as train_fsm
from circuit_lm.train_joint_pda_cpsat import train_joint_pda as train_pda
from src.hybrid import train_hybrid


def main():
    parser = argparse.ArgumentParser(description="Train CircuitLM + hybrid on personal dataset")
    parser.add_argument("--data", default="research_evolver_data.txt",
                        help="Training data .txt file (default: research_evolver_data.txt)")
    parser.add_argument("--vocab-size", type=int, default=4096,
                        help="Target vocab size (default: 4096)")
    parser.add_argument("--bpe-merges", type=int, default=None,
                        help="BPE merge count (default: same as vocab-size)")
    parser.add_argument("--automaton", default="pda", choices=["fsm", "pda"],
                        help="Circuit type (default: pda)")
    parser.add_argument("--state-bits", type=int, default=6,
                        help="State bits — states = 2^bits (default: 6 = 64 states)")
    parser.add_argument("--stack-depth", type=int, default=4,
                        help="PDA stack depth (default: 4)")
    parser.add_argument("--steps", type=int, default=60,
                        help="CP-SAT total solver steps (default: 60)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Hybrid corrector epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--max-examples", type=int, default=50000,
                        help="Max training examples for corrector (default: 50000)")
    parser.add_argument("--circuit-weight", type=float, default=0.5,
                        help="Circuit weight in blend (default: 0.5)")
    parser.add_argument("--out-circuit", default="circuit.json",
                        help="Output circuit JSON path")
    parser.add_argument("--out-corrector", default="corrector.pt",
                        help="Output corrector.pt path")
    parser.add_argument("--max-train-lines", type=int, default=5000,
                        help="Max lines to use from data file (for CPU speed, default: 5000)")
    args = parser.parse_args()

    bpe_merges = args.bpe_merges or args.vocab_size

    print("=" * 60)
    print("TRAIN CircuitLM — Personal Dataset")
    print("=" * 60)
    print(f"Data:          {args.data}")
    print(f"Automaton:     {args.automaton.upper()}")
    print(f"Vocab size:    {args.vocab_size}")
    print(f"BPE merges:    {bpe_merges}")
    print(f"State bits:    {args.state_bits}  ({2**args.state_bits} states)")
    if args.automaton == "pda":
        print(f"Stack depth:   {args.stack_depth}")
    print(f"Solver steps:  {args.steps}")
    print(f"Corrector:    {args.epochs} epochs, batch={args.batch_size}, max_examples={args.max_examples}")
    print()

    # ── Step 1: Build tokenizer ─────────────────────────────────────────
    print("STEP 1: Building BPE tokenizer...")
    t0 = time.time()

    # Load a subset of data for tokenizer (speed)
    text_lines = open(args.data, encoding="utf-8", errors="replace").readlines()[:args.max_train_lines]
    text_sample = " ".join(text_lines[:1000])  # first 1K lines for speed
    full_text = " ".join(text_lines)

    tokenizer = Tokenizer.from_text(
        full_text,
        vocab_size=args.vocab_size,
        mode="bpe",
        bpe_merges=bpe_merges,
    )
    print(f"Tokenizer done in {time.time()-t0:.1f}s — vocab={tokenizer.vocab_size}")

    # ── Step 2: Train circuit ──────────────────────────────────────────
    print(f"\nSTEP 2: Training {args.automaton.upper()} circuit...")
    t0 = time.time()

    # Use subset for circuit training too
    sequences = load_sequences("\n".join(text_lines[:args.max_train_lines]), tokenizer)
    print(f"  {len(sequences)} sequences, {sum(len(s) for s in sequences)} total tokens")

    total_tokens = sum(len(s) for s in sequences)
    print(f"  Total tokens: {total_tokens:,}")

    if args.automaton == "pda":
        stack_steps = max(1, args.steps // 4)
        remaining = args.steps - stack_steps
        transition_steps = remaining // 2
        emission_steps = remaining - transition_steps
        print(f"  PDA config: stack={stack_steps}, trans={transition_steps}, emit={emission_steps}")
        circuit = train_pda(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            stack_depth=args.stack_depth,
            stack_steps=stack_steps,
            transition_steps=transition_steps,
            emission_steps=emission_steps,
        )
    else:
        transition_steps = args.steps // 2
        emission_steps = args.steps - transition_steps
        print(f"  FSM config: trans={transition_steps}, emit={emission_steps}")
        circuit = train_fsm(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            steps=args.steps,
        )

    circuit_time = time.time() - t0
    print(f"  Circuit trained in {circuit_time:.1f}s")

    # Save circuit
    save_model(circuit, tokenizer, args.out_circuit)
    print(f"  Saved: {args.out_circuit}")

    # ── Step 3: Train hybrid corrector ─────────────────────────────────
    print(f"\nSTEP 3: Training hybrid corrector...")
    t0 = time.time()

    corrector_path = args.out_corrector
    hybrid = train_hybrid(
        circuit_path=args.out_circuit,
        data_path=args.data,
        output_path=corrector_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        circuit_weight=args.circuit_weight,
        max_examples=args.max_examples,
    )

    corrector_time = time.time() - t0
    print(f"  Corrector trained in {corrector_time:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Circuit:    {args.out_circuit}  ({Path(args.out_circuit).stat().st_size / 1024:.0f} KB)")
    print(f"Corrector:  {corrector_path}  ({Path(corrector_path).stat().st_size / 1024:.0f} KB)")
    print(f"Tokenizer:  BPE {tokenizer.vocab_size} vocab, {bpe_merges} merges")
    print(f"Times:      circuit={circuit_time:.1f}s  corrector={corrector_time:.1f}s")
    print(f"\nTotal time: {circuit_time + corrector_time:.1f}s")
    print("\nTo chat:")
    print(f"  py -3.12 -m circuit_lm.cli chat --model {args.out_circuit} --corrector {corrector_path}")


if __name__ == "__main__":
    main()
