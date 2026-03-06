# Training API

CircuitLM provides multiple training modules for different model types and training approaches.

## FSM Training

### `train_cpsat.train`

Train a CircuitLM using CP-SAT optimization.

```python
from circuit_lm.train_cpsat import train
```

```python
def train(
    sequences: list[list[int]],
    vocab_size: int,
    state_bits: int,
    steps: int,
    context_len: int = 4,
    top_k_coverage: int = 16,
    transition_steps: int | None = None,
    emission_steps: int | None = None,
    refinement_rounds: int = 1,
    joint_transition_state_steps: int | None = None,
) -> CircuitLM
```

**Parameters:**
- `sequences` (list[list[int]]): List of integer token-ID sequences
- `vocab_size` (int): Number of distinct token IDs
- `state_bits` (int): State width in bits; num_states = 2^state_bits
- `steps` (int): Legacy total CP-SAT time budget (seconds)
- `context_len` (int): Context window for state hashing (default: 4)
- `top_k_coverage` (int): Top-K token coverage constraint (default: 16)
- `transition_steps` (int | None): Transition CP-SAT budget
- `emission_steps` (int | None): Emission CP-SAT budget
- `refinement_rounds` (int): Additional refinement rounds (default: 1)
- `joint_transition_state_steps` (int | None): Joint transition/state bootstrap budget

**Returns:**
- `CircuitLM`: Trained FSM model

### `train_joint_cpsat.train_joint`

True joint FSM training with states as CP-SAT variables.

```python
from circuit_lm.train_joint_cpsat import train_joint
```

**Parameters:**
- `sequences` (list[list[int]]): Training sequences
- `vocab_size` (int): Vocabulary size
- `state_bits` (int): State bits
- `steps` (int): CP-SAT time budget
- `context_len` (int): Context window length
- `top_k_coverage` (int): Coverage constraint
- `refinement_rounds` (int): Refinement rounds

**Returns:**
- `CircuitLM`: Trained model with jointly optimized states

## PDA Training

### `train_pda_cpsat.train_pda`

Two-phase CP-SAT training for PDA.

```python
from circuit_lm.train_pda_cpsat import train_pda
```

```python
def train_pda(
    sequences: list[list[int]],
    vocab_size: int,
    state_bits: int,
    stack_depth: int,
    steps: int,
    context_len: int = 4,
    max_push: int = 16,
    max_pop: int = 16,
    top_k_pairs: int = 256,
    top_k_coverage: int = 16,
    stack_steps: int | None = None,
    transition_steps: int | None = None,
    emission_steps: int | None = None,
    refinement_rounds: int = 1,
) -> PDACircuitLM
```

**Parameters:**
- `sequences` (list[list[int]]): Training sequences
- `vocab_size` (int): Vocabulary size
- `state_bits` (int): State bits
- `stack_depth` (int): Maximum stack depth
- `steps` (int): Legacy total CP-SAT budget
- `context_len` (int): Context window length
- `max_push` (int): Maximum PUSH tokens (default: 16)
- `max_pop` (int): Maximum POP tokens (default: 16)
- `top_k_pairs` (int): Top co-occurrence pairs (default: 256)
- `top_k_coverage` (int): Coverage constraint
- `stack_steps` (int | None): Phase 1 stack policy budget
- `transition_steps` (int | None): Transition budget
- `emission_steps` (int | None): Emission budget
- `refinement_rounds` (int): Refinement rounds

**Returns:**
- `PDACircuitLM`: Trained PDA model

### `train_joint_pda_cpsat.train_joint_pda`

True joint PDA training with states, stack ops, and emissions as CP-SAT variables.

```python
from circuit_lm.train_joint_pda_cpsat import train_joint_pda
```

## PPM Training

### `train_ppm.train_ppm`

Train a PPM model (pure counting, no CP-SAT).

```python
from circuit_lm.train_ppm import train_ppm
```

```python
def train_ppm(
    sequences: list[list[int]],
    vocab_size: int,
    order: int,
) -> PPMModel
```

**Parameters:**
- `sequences` (list[list[int]]): Training sequences
- `vocab_size` (int): Vocabulary size
- `order` (int): Maximum context order

**Returns:**
- `PPMModel`: Trained PPM model

## Training Process

### FSM Training Process

1. **Hash-based Bootstrap**: Compute states from context via rolling hash
2. **Transition Collection**: Build (state, token) → successor-state histograms
3. **Emission Collection**: Build state → next-token histograms
4. **CP-SAT Optimization**: 
   - Optimize transitions to maximize successor agreement
   - Optimize emissions to maximize prediction accuracy
5. **Refinement**: EM-like passes over runtime states

### PDA Training Process (Two-Phase)

**Phase 1 - Stack Policy:**
- Compute co-occurrence scores for token pairs
- CP-SAT selects push/pop tokens to maximize co-occurrence weight

**Phase 2 - Config Emissions:**
- Simulate PDA with learned stack policy
- Collect (state, stack_top) → next-token histograms
- CP-SAT optimizes emissions per configuration

### Joint Training

True joint optimization where all components (states, transitions, emissions) are simultaneously optimized as CP-SAT decision variables to directly maximize prediction accuracy.

## Usage Example

```python
from circuit_lm.train_cpsat import train
from circuit_lm.data import load_sequences
from circuit_lm.tokenizer import Tokenizer

# Prepare data
tokenizer = Tokenizer.from_text(text, vocab_size=128, mode="char")
sequences = load_sequences("data.txt", tokenizer)

# Train FSM
model = train(
    sequences=sequences,
    vocab_size=tokenizer.vocab_size,
    state_bits=4,
    steps=10,
    context_len=4,
    top_k_coverage=16,
    refinement_rounds=1,
)
```
