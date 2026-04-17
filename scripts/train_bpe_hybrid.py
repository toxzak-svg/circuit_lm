"""Train BPE circuit + hybrid for a kickass CPU chat bot.

Usage:
    py -3.12 scripts/train_bpe_hybrid.py --data training_data.txt --circuit-out circuit.json --corrector-out corrector.pt
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from circuit_lm.data import load_text
from circuit_lm.io import save_model, load_model
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat import train as train_fsm
from circuit_lm.train_joint_pda_cpsat import train_joint_pda as train_pda
from hybrid import train_hybrid, train_hybrid_streaming, HybridModel


def main():
    parser = argparse.ArgumentParser(description="Train BPE circuit + hybrid")
    parser.add_argument("--data", required=True, help="Training data .txt file")
    parser.add_argument("--circuit-out", required=True, help="Output circuit .json")
    parser.add_argument("--corrector-out", required=True, help="Output corrector .pt")
    parser.add_argument("--automaton", default="pda", choices=["fsm", "pda"], help="Circuit type")
    parser.add_argument("--vocab-size", type=int, default=1024, help="Target vocab size")
    parser.add_argument("--bpe-merges", type=int, default=512, help="BPE merge count")
    parser.add_argument("--state-bits", type=int, default=5, help="State bits (states = 2^bits)")
    parser.add_argument("--stack-depth", type=int, default=4, help="PDA stack depth")
    parser.add_argument("--steps", type=int, default=60, help="CP-SAT solver steps")
    parser.add_argument("--epochs", type=int, default=5, help="Hybrid training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-examples", type=int, default=100000, help="Max training examples")
    parser.add_argument("--circuit-weight", type=float, default=0.5, help="Circuit weight in blend")
    parser.add_argument("--streaming", action="store_true", help="Use streaming data loader (for large corpora)")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Examples per streaming chunk")
    parser.add_argument("--embed-dim", type=int, default=128, help="Corrector embedding dim (default 128, larger=256+)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Corrector hidden dim (default 256, larger=512+)")
    parser.add_argument("--num-layers", type=int, default=3, help="Corrector layers (default 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Build BPE Tokenizer")
    print("=" * 60)
    
    text = load_text(args.data)
    tokenizer = Tokenizer.from_text(
        text,
        vocab_size=args.vocab_size,
        mode="bpe",
        bpe_merges=args.bpe_merges,
    )
    print(f"Tokenizer: mode={tokenizer.mode}, vocab_size={tokenizer.vocab_size}")
    print(f"BPE merges performed: {args.bpe_merges}")

    print("\n" + "=" * 60)
    print("STEP 2: Train Circuit (CP-SAT)")
    print("=" * 60)
    
    from circuit_lm.data import load_sequences
    sequences = load_sequences(args.data, tokenizer)
    print(f"Loaded {len(sequences)} sequences")

    # Calculate budget
    total_tokens = sum(len(s) for s in sequences)
    print(f"Total tokens: {total_tokens}")

    t0 = time.time()
    if args.automaton == "pda":
        # PDA: stack_steps + transition_steps + emission_steps
        stack_steps = args.steps // 3
        remaining = args.steps - stack_steps
        transition_steps = remaining // 2
        emission_steps = remaining - transition_steps
        print(f"Training PDA: stack={stack_steps}, trans={transition_steps}, emit={emission_steps}")
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
        # FSM
        transition_steps = args.steps // 2
        emission_steps = args.steps - transition_steps
        print(f"Training FSM: trans={transition_steps}, emit={emission_steps}")
        circuit = train_fsm(
            sequences=sequences,
            vocab_size=tokenizer.vocab_size,
            state_bits=args.state_bits,
            steps=args.steps,
        )
    
    train_time = time.time() - t0
    print(f"Circuit training took {train_time:.1f}s")

    # Save circuit
    save_model(circuit, tokenizer, args.circuit_out)
    print(f"Saved circuit to {args.circuit_out}")

    print("\n" + "=" * 60)
    print("STEP 3: Train Hybrid Corrector")
    print("=" * 60)
    
    # Reload circuit for hybrid training
    circuit, tokenizer = load_model(args.circuit_out)
    print(f"Reloaded circuit: vocab={circuit.vocab_size}, states={circuit.num_states}")

    t0 = time.time()
    if args.streaming:
        print(f"Using streaming loader (chunk_size={args.chunk_size})")
        hybrid = train_hybrid_streaming(
            circuit_path=args.circuit_out,
            data_path=args.data,
            output_path=args.corrector_out,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            circuit_weight=args.circuit_weight,
            max_examples=args.max_examples,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            chunk_size=args.chunk_size,
        )
    else:
        hybrid = train_hybrid(
            circuit_path=args.circuit_out,
            data_path=args.data,
            output_path=args.corrector_out,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            circuit_weight=args.circuit_weight,
            max_examples=args.max_examples,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    corrector_time = time.time() - t0
    print(f"Corrector training took {corrector_time:.1f}s")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Circuit:   {args.circuit_out}")
    print(f"Corrector: {args.corrector_out}")
    print(f"Tokenizer: BPE with {tokenizer.vocab_size} vocab, {args.bpe_merges} merges")
    print(f"Total time: {train_time + corrector_time:.1f}s")
    print("\nTo chat:")
    print(f"  py -3.12 -m circuit_lm.cli chat --model {args.circuit_out} --corrector {args.corrector_out}")


if __name__ == "__main__":
    main()
