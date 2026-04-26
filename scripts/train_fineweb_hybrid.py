"""Train BPE circuit + corrected hybrid corrector on FineWeb-Edu.

Downloads a ~10M token slice of FineWeb-Edu from HuggingFace, then:
1. Builds a BPE tokenizer
2. Trains a PDA/FSM circuit via CP-SAT
3. Trains the CorrectedCorrector (gated delta, stacked SSD)

Usage:
    py -3.12 scripts/train_fineweb_hybrid.py --data-name FineWeb-Edu --subset 10M
        --circuit-out circuit_fineweb.json --corrector-out corrector_fineweb.pt

    py -3.12 scripts/train_fineweb_hybrid.py --data-name FineWeb-Edu --subset 10M
        --circuit-out circuit_fineweb.json --corrector-out corrector_fineweb.pt
        --vocab-size 4096 --bpe-merges 2048 --state-bits 12 --epochs 5
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from circuit_lm.data import load_text, load_sequences
from circuit_lm.io import save_model, load_model
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_joint_pda_cpsat import train_joint_pda as train_pda
from circuit_lm.train_cpsat import train as train_fsm
from hybrid import train_corrected_hybrid


def download_fineweb_subset(dataset_name: str, subset: str, output_path: str) -> str:
    """Download a subset of FineWeb-Edu from HuggingFace and save as .txt.

    Uses streaming to avoid loading the full dataset into memory.
    Writes directly to file as rows are streamed.

    Args:
        dataset_name: HuggingFace dataset (e.g. "HuggingFaceFW/fineweb-edu")
        subset: Token count hint (e.g. "10M", "1M", "100K"). Determines how many rows to write.
        output_path: Local .txt output path

    Returns:
        Path to saved text file
    """
    print(f"Streaming {dataset_name} ({subset}) from HuggingFace...")
    t0 = time.time()

    from datasets import load_dataset

    if subset.endswith("M"):
        num_rows = int(subset[:-1]) * 1000 * 1000  # Convert "10M" to 10_000_000
    elif subset.endswith("K"):
        num_rows = int(subset[:-1]) * 1000  # Convert "100K" to 100_000
    else:
        num_rows = int(subset)

    ds = load_dataset(dataset_name, split="train", streaming=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= num_rows:
                break
            if i > 0 and i % 10000 == 0:
                print(f"  Written {i:,}/{num_rows:,} rows...")
            f.write(row["text"] + "\n")

    size = Path(output_path).stat().st_size
    print(f"  Wrote {i+1:,} rows, {size:,} bytes in {time.time()-t0:.1f}s -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train BPE circuit + hybrid on FineWeb-Edu")
    parser.add_argument("--data-name", default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--subset", default="10M",
                        help="Subset to download (e.g. '10M', '1M', '100K')")
    parser.add_argument("--data-cache", default="fineweb_cache.txt",
                        help="Local .txt cache path (skip download if exists)")
    parser.add_argument("--circuit-out", required=True, help="Output circuit .json")
    parser.add_argument("--corrector-out", required=True, help="Output corrector .pt")
    parser.add_argument("--automaton", default="pda", choices=["fsm", "pda"],
                        help="Circuit type")
    parser.add_argument("--vocab-size", type=int, default=4096, help="Target vocab size")
    parser.add_argument("--bpe-merges", type=int, default=2048, help="BPE merge count")
    parser.add_argument("--state-bits", type=int, default=12,
                        help="State bits (states = 2^bits), use 12 for 4096 states")
    parser.add_argument("--stack-depth", type=int, default=4, help="PDA stack depth")
    parser.add_argument("--steps", type=int, default=60, help="CP-SAT solver steps")
    parser.add_argument("--epochs", type=int, default=5, help="Hybrid training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-examples", type=int, default=500000,
                        help="Max training examples for corrector")
    parser.add_argument("--embed-dim", type=int, default=256, help="Corrector embed dim")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Corrector hidden dim")
    parser.add_argument("--num-ssd-layers", type=int, default=2, help="SSD layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use cached --data-cache file")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 0: Download / Cache FineWeb-Edu")
    print("=" * 60)

    if args.skip_download and Path(args.data_cache).exists():
        print(f"Using cached data: {args.data_cache}")
        data_path = args.data_cache
    else:
        data_path = download_fineweb_subset(
            args.data_name, args.subset, args.data_cache
        )

    print(f"\n{'=' * 60}")
    print("STEP 1: Build BPE Tokenizer")
    print("=" * 60)

    t0 = time.time()
    text = load_text(data_path)
    tokenizer = Tokenizer.from_text(
        text,
        vocab_size=args.vocab_size,
        mode="bpe",
        bpe_merges=args.bpe_merges,
    )
    print(f"Tokenizer: mode={tokenizer.mode}, vocab_size={tokenizer.vocab_size}")
    print(f"BPE merges performed: {args.bpe_merges}")
    print(f"Text: {len(text):,} chars, took {time.time()-t0:.1f}s")

    print(f"\n{'=' * 60}")
    print("STEP 2: Train Circuit (CP-SAT)")
    print("=" * 60)

    sequences = load_sequences(data_path, tokenizer)
    print(f"Loaded {len(sequences):,} sequences")
    total_tokens = sum(len(s) for s in sequences)
    print(f"Total tokens: {total_tokens:,}")

    t0 = time.time()
    stack_steps = args.steps // 3
    remaining = args.steps - stack_steps
    trans_steps = remaining // 2
    emit_steps = remaining - trans_steps

    if args.automaton == "pda":
        print(f"Training PDA: stack={stack_steps}, trans={trans_steps}, emit={emit_steps}")
        circuit = train_pda(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            stack_depth=args.stack_depth,
            stack_steps=stack_steps,
            transition_steps=trans_steps,
            emission_steps=emit_steps,
        )
    else:
        print(f"Training FSM: trans={trans_steps}, emit={emit_steps}")
        circuit = train_fsm(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            steps=args.steps,
        )

    circuit_train_time = time.time() - t0
    print(f"Circuit training took {circuit_train_time:.1f}s")

    print(f"\n{'=' * 60}")
    print("STEP 3: Save Circuit")
    print("=" * 60)
    save_model(circuit, tokenizer, args.circuit_out)
    print(f"Saved circuit to {args.circuit_out}")

    print(f"\n{'=' * 60}")
    print("STEP 4: Train CorrectedCorrector")
    print("=" * 60)

    t0 = time.time()
    hybrid = train_corrected_hybrid(
        circuit_path=args.circuit_out,
        data_path=data_path,
        output_path=args.corrector_out,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_examples=args.max_examples,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_ssd_layers=args.num_ssd_layers,
    )
    corrector_train_time = time.time() - t0
    print(f"Corrector training took {corrector_train_time:.1f}s")

    print(f"\n{'=' * 60}")
    print("DONE!")
    print("=" * 60)
    print(f"Circuit:   {args.circuit_out}")
    print(f"Corrector: {args.corrector_out}")
    print(f"Tokenizer: BPE vocab={tokenizer.vocab_size}, {args.bpe_merges} merges")
    print(f"Circuit states: {circuit.num_states} ({args.state_bits} bits)")
    print(f"Total time: {circuit_train_time + corrector_train_time:.1f}s")
    print(f"\nTo chat:")
    print(f"  py -3.12 -m circuit_lm.cli chat --model {args.circuit_out} --corrector {args.corrector_out}")


if __name__ == "__main__":
    main()
