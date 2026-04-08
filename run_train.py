import sys
sys.path.insert(0, "C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm/src")

import time
from circuit_lm.data import load_text, load_sequences
from circuit_lm.io import save_model, load_model
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_joint_pda_cpsat import train_joint_pda as train_pda
from hybrid import train_hybrid

DATA = "C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm/research_evolver_data.txt"
VOCAB = 4096
LINES = 3000
STEPS = 60
EPOCHS = 5

print("STEP 1: Tokenizer...")
t0 = time.time()
lines = open(DATA, encoding="utf-8", errors="replace").readlines()[:LINES]
full_text = " ".join(lines)
tok = Tokenizer.from_text(full_text, vocab_size=VOCAB, mode="bpe", bpe_merges=VOCAB)
print(f"  vocab={tok.vocab_size}, took {time.time()-t0:.1f}s")

print("STEP 2: Load sequences...")
seqs = load_sequences("\n".join(lines), tok)
print(f"  {len(seqs)} sequences, {sum(len(s) for s in seqs)} tokens")

print("STEP 3: Train PDA circuit...")
t0 = time.time()
stack_steps = STEPS // 4
rem = STEPS - stack_steps
trans_steps = rem // 2
emit_steps = rem - trans_steps
print(f"  stack={stack_steps}, trans={trans_steps}, emit={emit_steps}")
circuit = train_pda(
    sequences=seqs,
    vocab_size=tok.vocab_size,
    state_bits=6,
    stack_depth=4,
    stack_steps=stack_steps,
    transition_steps=trans_steps,
    emission_steps=emit_steps,
)
print(f"  done in {time.time()-t0:.1f}s")

CIRCUIT_OUT = "C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm/circuit_4k.json"
save_model(circuit, tok, CIRCUIT_OUT)
print(f"  saved: {CIRCUIT_OUT}")

print("STEP 4: Train corrector...")
t0 = time.time()
corrector = train_hybrid(
    circuit_path=CIRCUIT_OUT,
    data_path=DATA,
    output_path="C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm/corrector_4k.pt",
    num_epochs=EPOCHS,
    batch_size=64,
    circuit_weight=0.5,
    max_examples=50000,
)
print(f"  done in {time.time()-t0:.1f}s")
print("DONE")
