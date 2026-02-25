# Joint-PDA Verification + Config-Conditioned Stack Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** (1) Verify joint-PDA stack discovery at T_total ≤ 2000 via a new focused script; (2) replace `push_tokens`/`pop_tokens` (per-token-ID) with `push_configs`/`pop_configs` (`frozenset[tuple[int,int,int]]`) throughout the codebase so stack operations can be conditioned on `(src_state, token, stack_top)`.

**Architecture:** `pda.py` is the root data model — change it first, then fix every consumer top-down. `train_pda_cpsat.py`'s internal simulation helpers keep `frozenset[int]` parameters; only the final `PDACircuitLM(...)` constructor call expands tokens → config triples. `train_joint_pda_cpsat.py` gets a fully config-conditioned CP-SAT formulation using `add_element` to look up `is_push_flat[src * V * ST_RANGE + tok * ST_RANGE + st_prev]` per occurrence.

**Tech Stack:** Python 3.12, OR-Tools CP-SAT (`ortools.sat.python.cp_model`), pytest, stdlib only (no floats anywhere).

---

### Task 1: pda.py — Replace push/pop fields and update stack_op + step

**Files:**
- Modify: `circuit_lm/pda.py`
- Modify: `tests/test_pda.py`

#### Step 1: Write failing tests for the new API

Add to `tests/test_pda.py` (at the end of the file, before the save/load section):

```python
# ---------------------------------------------------------------------------
# Config-conditioned push/pop fields
# ---------------------------------------------------------------------------

def test_pda_has_push_configs_field() -> None:
    """PDACircuitLM should have push_configs, not push_tokens."""
    m = PDACircuitLM(
        vocab_size=3, num_states=4, state_bits=2, stack_depth=1,
        push_configs=frozenset([(0, 0, -1)]),
        pop_configs=frozenset([(0, 1, 0)]),
    )
    assert isinstance(m.push_configs, frozenset)
    assert (0, 0, -1) in m.push_configs


def test_pda_stack_op_three_arg() -> None:
    """stack_op(state, token, stack_top) dispatches via push_configs/pop_configs."""
    m = PDACircuitLM(
        vocab_size=3, num_states=4, state_bits=2, stack_depth=1,
        push_configs=frozenset([(0, 0, -1)]),   # push tok=0 when state=0, stack=EMPTY
        pop_configs=frozenset([(0, 1, 0)]),      # pop  tok=1 when state=0, stack=0
    )
    assert m.stack_op(0, 0, STACK_EMPTY) == OP_PUSH
    assert m.stack_op(0, 1, 0)           == OP_POP
    assert m.stack_op(0, 2, STACK_EMPTY) == OP_NOP
    # Config not in push_configs, even if same token
    assert m.stack_op(1, 0, STACK_EMPTY) == OP_NOP  # state=1, not in push_configs


def test_pda_step_uses_source_state() -> None:
    """step() resolves op with src state, not dst state."""
    # State 0 → state 1 on tok=0.  Push only fires on (src=0, tok=0, stack=EMPTY).
    transitions = {(0, 0): 1, (1, 0): 0}
    m = PDACircuitLM(
        vocab_size=3, num_states=2, state_bits=1, stack_depth=2,
        push_configs=frozenset([(0, 0, STACK_EMPTY)]),
        pop_configs=frozenset(),
        transitions=transitions,
    )
    # First step: state=0 (src), tok=0, stack=[] → push should fire
    next_s, new_stack = m.step(0, [], 0)
    assert next_s == 1
    assert new_stack == [0]   # tok=0 was pushed

    # Second step: state=1 (src), tok=0, stack=[0] → push should NOT fire
    next_s2, new_stack2 = m.step(1, [0], 0)
    assert next_s2 == 0
    assert new_stack2 == [0]  # unchanged
```

#### Step 2: Run to verify they fail

```
py -3.12 -m pytest tests/test_pda.py::test_pda_has_push_configs_field tests/test_pda.py::test_pda_stack_op_three_arg tests/test_pda.py::test_pda_step_uses_source_state -v
```

Expected: `FAILED — TypeError` or `AttributeError: 'PDACircuitLM' has no field 'push_configs'`

#### Step 3: Implement changes in circuit_lm/pda.py

**3a — Replace dataclass fields** (lines 100–101):

```python
# Remove:
push_tokens:   frozenset[int] = field(default_factory=frozenset)
pop_tokens:    frozenset[int] = field(default_factory=frozenset)

# Add:
push_configs: frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
pop_configs:  frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
```

**3b — Replace stack_op method** (lines 110–116):

```python
def stack_op(self, state: int, token: int, stack_top: int) -> int:
    """Return the stack operation for the given (state, token, stack_top) triple.

    Uses push_configs / pop_configs; returns OP_NOP if the triple appears in
    neither set.  All arguments are integers; STACK_EMPTY (-1) is the sentinel
    for an empty stack.
    """
    key = (state, token, stack_top)
    if key in self.push_configs:
        return OP_PUSH
    if key in self.pop_configs:
        return OP_POP
    return OP_NOP
```

**3c — Update step method** (lines 133–159) to compute stack_top first and pass all three args:

```python
def step(
    self, state: int, stack: list[int], token: int
) -> tuple[int, list[int]]:
    """Execute one PDA step and return (next_state, new_stack).

    Stack operation is resolved using (state, token, stack_top) before the
    FSM transition so that the source state governs the op decision.
    """
    stack_top = stack[-1] if stack else STACK_EMPTY
    op = self.stack_op(state, token, stack_top)
    next_s = self.next_state(state, token)

    new_stack = list(stack)
    if op == OP_PUSH and len(new_stack) < self.stack_depth:
        new_stack.append(token)
    elif op == OP_POP and new_stack:
        new_stack.pop()

    return next_s, new_stack
```

**3d — Update docstring fields** in the class docstring (lines 89–93):

```
push_configs:  Frozen set of (state, token, stack_top) triples that trigger OP_PUSH.
pop_configs:   Frozen set of (state, token, stack_top) triples that trigger OP_POP.
```

#### Step 4: Update existing tests in test_pda.py that reference push_tokens/pop_tokens

Search for any usage of `push_tokens` or `pop_tokens` in `test_pda.py` and update:
- All `PDACircuitLM(push_tokens=..., pop_tokens=...)` constructor calls → `push_configs=frozenset(), pop_configs=frozenset()`
- Any assertions on `model.push_tokens` → `model.push_configs`
- Any assertions on `model.pop_tokens` → `model.pop_configs`
- Any call to `model.stack_op(token)` → `model.stack_op(state, token, stack_top)`

Run: `py -3.12 -m pytest tests/test_pda.py -v`

Expected: all tests PASS (or at most failures from train_pda not yet updated — those are in Tasks 3–4).

#### Step 5: Commit

```bash
git add circuit_lm/pda.py tests/test_pda.py
git commit -m "feat: replace push/pop token-ID fields with config triples in PDACircuitLM"
```

---

### Task 2: io.py — Serialization with migration shim

**Files:**
- Modify: `circuit_lm/io.py`
- Modify: `tests/test_pda.py` (add serialization round-trip tests)

#### Step 1: Write failing serialization tests

Add to `tests/test_pda.py`:

```python
# ---------------------------------------------------------------------------
# Serialization round-trip for push_configs / pop_configs
# ---------------------------------------------------------------------------

def test_pda_save_load_push_configs_round_trip(tmp_path: pathlib.Path) -> None:
    """push_configs / pop_configs survive a JSON save/load cycle."""
    from circuit_lm.io import save_model, load_model
    from circuit_lm.tokenizer import Tokenizer
    push = frozenset([(0, 0, STACK_EMPTY), (1, 0, 0)])
    pop  = frozenset([(0, 1, 0)])
    m = PDACircuitLM(
        vocab_size=3, num_states=2, state_bits=1, stack_depth=2,
        push_configs=push, pop_configs=pop,
        transitions={(0, 0): 1, (0, 1): 0, (0, 2): 0,
                     (1, 0): 0, (1, 1): 1, (1, 2): 0},
        config_counts={(0, STACK_EMPTY): [0, 5, 3]},
        config_pred_tokens={(0, STACK_EMPTY): 1},
    )
    tok = Tokenizer.from_text("( ) x ( ) x", vocab_size=4)
    path = tmp_path / "model.json"
    save_model(m, tok, path)
    loaded, _ = load_model(path)
    assert isinstance(loaded, PDACircuitLM)
    assert loaded.push_configs == push
    assert loaded.pop_configs == pop


def test_pda_load_old_push_tokens_migrates(tmp_path: pathlib.Path) -> None:
    """Old JSON with push_tokens=[0] / pop_tokens=[1] loads as degenerate push_configs."""
    import json
    from circuit_lm.io import load_model
    old_payload = {
        "model_type": "pda",
        "vocab_size": 3, "num_states": 2, "state_bits": 1, "stack_depth": 1,
        "push_tokens": [0],
        "pop_tokens":  [1],
        "transitions": {"0,0": 1, "0,1": 0, "0,2": 0,
                        "1,0": 0, "1,1": 1, "1,2": 0},
        "config_counts": {"0,-1": [0, 3, 1]},
        "config_pred_tokens": {"0,-1": 1},
        "tokenizer": {"mode": "char", "chars": ["<PAD>", "<UNK>", "(", ")", "x"]},
    }
    p = tmp_path / "old_model.json"
    p.write_text(json.dumps(old_payload), encoding="utf-8")
    loaded, _ = load_model(p)
    assert isinstance(loaded, PDACircuitLM)
    # Every (state, tok=0, any_stack_top) must be in push_configs
    for s in range(2):
        for st in [STACK_EMPTY, 0, 1, 2]:
            assert (s, 0, st) in loaded.push_configs
    # No push_tokens attribute
    assert not hasattr(loaded, "push_tokens")
```

Run: `py -3.12 -m pytest tests/test_pda.py::test_pda_save_load_push_configs_round_trip tests/test_pda.py::test_pda_load_old_push_tokens_migrates -v`

Expected: FAIL (io.py still uses push_tokens)

#### Step 2: Update _save_pda in circuit_lm/io.py

Replace lines 130–131:
```python
# Remove:
"push_tokens":   sorted(model.push_tokens),
"pop_tokens":    sorted(model.pop_tokens),

# Add:
"push_configs":  sorted([s, tok, st] for (s, tok, st) in model.push_configs),
"pop_configs":   sorted([s, tok, st] for (s, tok, st) in model.pop_configs),
```

Also update the module docstring comment (lines 31–32) to reflect the new format:
```
"push_configs":  [[state, token, stack_top], ...],
"pop_configs":   [[state, token, stack_top], ...],
```

#### Step 3: Update _load_pda in circuit_lm/io.py

Replace lines 268–269 with:

```python
# New format: push_configs / pop_configs as [[s, tok, st], ...]
if "push_configs" in data:
    push_configs: frozenset[tuple[int, int, int]] = frozenset(
        (int(triple[0]), int(triple[1]), int(triple[2]))
        for triple in data["push_configs"]
    )
    pop_configs: frozenset[tuple[int, int, int]] = frozenset(
        (int(triple[0]), int(triple[1]), int(triple[2]))
        for triple in data["pop_configs"]
    )
else:
    # Migration shim: old format with push_tokens / pop_tokens as int lists.
    vocab_size_v = int(data["vocab_size"])
    num_states_v = int(data["num_states"])
    all_stack_tops = [STACK_EMPTY] + list(range(vocab_size_v))
    push_tokens_old = [int(t) for t in data.get("push_tokens", [])]
    pop_tokens_old  = [int(t) for t in data.get("pop_tokens", [])]
    push_configs = frozenset(
        (s, tok, st)
        for tok in push_tokens_old
        for s in range(num_states_v)
        for st in all_stack_tops
    )
    pop_configs = frozenset(
        (s, tok, st)
        for tok in pop_tokens_old
        for s in range(num_states_v)
        for st in all_stack_tops
    )
```

Update the PDACircuitLM() constructor call at line 263 to use `push_configs=push_configs, pop_configs=pop_configs`.

#### Step 4: Run the new tests

```
py -3.12 -m pytest tests/test_pda.py -v
```

Expected: all tests PASS.

#### Step 5: Commit

```bash
git add circuit_lm/io.py tests/test_pda.py
git commit -m "feat: update PDA serialization to push_configs/pop_configs + migration shim"
```

---

### Task 3: train_pda_cpsat.py — Degenerate expansion at output

**Files:**
- Modify: `circuit_lm/train_pda_cpsat.py`
- Modify: `tests/test_pda.py` (update `tiny_pda` fixture-based tests)

**Key insight:** `_simulate_and_collect` and `_simulate_and_collect_runtime` keep their `push_tokens: frozenset[int]` / `pop_tokens: frozenset[int]` parameters unchanged — they are internal functions that never construct a `PDACircuitLM`. Only the final `return PDACircuitLM(...)` call needs updating.

#### Step 1: Write a failing test for degenerate expansion

Add to `tests/test_pda.py`:

```python
def test_train_pda_push_configs_cover_all_states_and_stack_tops() -> None:
    """Two-phase PDA expands each push token to all (state, tok, stack_top) triples."""
    seqs = [[0, 1, 0, 1, 0, 1]] * 20   # tok 0 = push, tok 1 = pop
    from circuit_lm.train_pda_cpsat import train_pda
    m = train_pda(
        sequences=seqs, vocab_size=2, state_bits=1, stack_depth=1,
        steps=3, max_push=1, max_pop=1,
    )
    # If token 0 was chosen as push, ALL (state, 0, stack_top) triples must exist
    push_toks = set(tok for (_, tok, _) in m.push_configs)
    for tok in push_toks:
        for s in range(m.num_states):
            for st in [STACK_EMPTY] + list(range(m.vocab_size)):
                assert (s, tok, st) in m.push_configs, (
                    f"Missing ({s}, {tok}, {st}) in push_configs"
                )
```

Run: `py -3.12 -m pytest tests/test_pda.py::test_train_pda_push_configs_cover_all_states_and_stack_tops -v`

Expected: FAIL — `PDACircuitLM` construction in `train_pda` still passes `push_tokens=`.

#### Step 2: Update train_pda in circuit_lm/train_pda_cpsat.py

At the end of `train_pda` (lines 570–580), replace the PDACircuitLM constructor call:

```python
# Expand per-token push/pop sets to full (state, token, stack_top) config triples.
# Internal simulation helpers (_simulate_and_collect, _simulate_and_collect_runtime)
# still receive frozenset[int] and are unchanged.
all_stack_tops = [STACK_EMPTY] + list(range(vocab_size))
push_configs: frozenset[tuple[int, int, int]] = frozenset(
    (s, tok, st)
    for tok in push_tokens
    for s in range(num_states)
    for st in all_stack_tops
)
pop_configs: frozenset[tuple[int, int, int]] = frozenset(
    (s, tok, st)
    for tok in pop_tokens
    for s in range(num_states)
    for st in all_stack_tops
)

return PDACircuitLM(
    vocab_size=vocab_size,
    num_states=num_states,
    state_bits=state_bits,
    stack_depth=stack_depth,
    push_configs=push_configs,
    pop_configs=pop_configs,
    transitions=transitions,
    config_counts=config_counts,
    config_pred_tokens=config_pred_tokens,
)
```

#### Step 3: Update existing tiny_pda fixture tests in test_pda.py

Search for any test that asserts `tiny_pda.push_tokens` or `tiny_pda.pop_tokens`. Update:
- `assert isinstance(tiny_pda.push_tokens, frozenset)` →
  `assert isinstance(tiny_pda.push_configs, frozenset)`
- Any count assertions: `len(tiny_pda.push_tokens)` is no longer meaningful; replace with
  `len({tok for (_, tok, _) in tiny_pda.push_configs})` to count distinct push token IDs.
- Save/load round-trip tests: ensure they use `push_configs`.

Run: `py -3.12 -m pytest tests/test_pda.py -v`

Expected: all PASS.

#### Step 4: Commit

```bash
git add circuit_lm/train_pda_cpsat.py tests/test_pda.py
git commit -m "feat: expand push/pop tokens to config triples in train_pda output"
```

---

### Task 4: train_joint_pda_cpsat.py — Config-conditioned CP-SAT formulation

**Files:**
- Modify: `circuit_lm/train_joint_pda_cpsat.py`
- Modify: `tests/test_joint_pda.py`

#### Step 1: Write failing tests for the new formulation

Add to `tests/test_joint_pda.py` (replace or extend existing push/pop tests):

```python
def test_train_joint_pda_push_configs_are_triples(tiny_joint_pda: PDACircuitLM) -> None:
    """push_configs must be a frozenset of (int, int, int) triples."""
    for triple in tiny_joint_pda.push_configs:
        s, tok, st = triple
        assert isinstance(s,   int)
        assert isinstance(tok, int)
        assert isinstance(st,  int)
        assert 0 <= s < tiny_joint_pda.num_states
        assert 0 <= tok < tiny_joint_pda.vocab_size
        assert st == STACK_EMPTY or (0 <= st < tiny_joint_pda.vocab_size)


def test_train_joint_pda_push_pop_configs_disjoint(tiny_joint_pda: PDACircuitLM) -> None:
    """No triple may appear in both push_configs and pop_configs."""
    assert tiny_joint_pda.push_configs.isdisjoint(tiny_joint_pda.pop_configs)


def test_train_joint_pda_max_push_token_semantics() -> None:
    """max_push=1 means at most 1 distinct token ID appears in push_configs."""
    model = train_joint_pda(
        sequences=BRACKET_SEQS, vocab_size=3, num_states=4,
        stack_depth=1, steps=4, max_push=1,
    )
    push_tok_ids = {tok for (_, tok, _) in model.push_configs}
    assert len(push_tok_ids) <= 1


def test_train_joint_pda_max_push_zero_means_no_push_configs() -> None:
    """max_push=0 means push_configs is empty."""
    model = train_joint_pda(
        sequences=BRACKET_SEQS, vocab_size=3, num_states=4,
        stack_depth=1, steps=4, max_push=0,
    )
    assert len(model.push_configs) == 0
```

Update existing tests that reference `tiny_joint_pda.push_tokens`:
- `test_train_joint_pda_push_pop_disjoint` → use `push_configs`, `pop_configs`
- `test_train_joint_pda_push_tokens_in_vocab` → check `{tok for (_, tok, _) in m.push_configs}`
- `test_train_joint_pda_pop_tokens_in_vocab` → similar

Run: `py -3.12 -m pytest tests/test_joint_pda.py -v`

Expected: many FAIL (push_tokens attribute gone, push_configs not yet produced).

#### Step 2: Update variable declarations in train_joint_pda_cpsat.py

**Replace** the `is_push` / `is_pop` variable blocks (lines 208–217):

```python
# Flat index helper (inline, no lambda — avoids float-scan false positive)
# idx = s * vocab_size * ST_RANGE  +  tok * ST_RANGE  +  st_enc
PUSH_STRIDE_S: int = vocab_size * ST_RANGE
PUSH_SIZE:     int = num_states * PUSH_STRIDE_S

is_push_flat: list[cp_model.BoolVar] = [
    model.new_bool_var(f"push_{s}_{tok}_{st_enc}")
    for s in range(num_states)
    for tok in range(vocab_size)
    for st_enc in range(ST_RANGE)
]
is_pop_flat: list[cp_model.BoolVar] = [
    model.new_bool_var(f"pop_{s}_{tok}_{st_enc}")
    for s in range(num_states)
    for tok in range(vocab_size)
    for st_enc in range(ST_RANGE)
]

# Mutual exclusivity per (state, token, stack_top) triple
for _s in range(num_states):
    for _tok in range(vocab_size):
        for _st_enc in range(ST_RANGE):
            _i = _s * PUSH_STRIDE_S + _tok * ST_RANGE + _st_enc
            model.add(is_push_flat[_i] + is_pop_flat[_i] <= 1)
```

#### Step 3: Replace max_push / max_pop budget constraint

**Remove** the old `if max_push is not None: model.add(sum(is_push) <= max_push)` block.

**Add** auxiliary "token ever pushes" variables (after mutual exclusivity block):

```python
# Budget constraints: max_push / max_pop count distinct token IDs, not triples.
if max_push is not None or max_pop is not None:
    tok_ever_pushes: list[cp_model.BoolVar] = [
        model.new_bool_var(f"tok_push_{_tok}") for _tok in range(vocab_size)
    ]
    tok_ever_pops: list[cp_model.BoolVar] = [
        model.new_bool_var(f"tok_pop_{_tok}") for _tok in range(vocab_size)
    ]
    for _tok in range(vocab_size):
        for _s in range(num_states):
            for _st_enc in range(ST_RANGE):
                _i = _s * PUSH_STRIDE_S + _tok * ST_RANGE + _st_enc
                model.add(is_push_flat[_i] <= tok_ever_pushes[_tok])
                model.add(is_pop_flat[_i]  <= tok_ever_pops[_tok])
    if max_push is not None:
        model.add(sum(tok_ever_pushes) <= max_push)
    if max_pop is not None:
        model.add(sum(tok_ever_pops) <= max_pop)
```

#### Step 4: Update the per-occurrence constraint loop (lines 251–283)

Replace the block that sets `push_tok = is_push[tok]` and `pop_tok = is_pop[tok]` with per-occurrence lookups:

```python
for occ_idx, (prev_occ_idx, tok, _next_tok) in enumerate(occurrences):
    src = src_vars[occ_idx]
    dst = dst_vars[occ_idx]
    st  = st_vars[occ_idx]

    # --- FSM chain ---
    if prev_occ_idx < 0:
        model.add(src == 0)
    else:
        model.add(src == dst_vars[prev_occ_idx])

    # --- FSM transition ---
    model.add_element(src, delta_cols[tok], dst)

    # --- Stack op lookup ---
    tok_offset: int = tok * ST_RANGE   # constant for this occurrence

    if prev_occ_idx < 0:
        # Sequence start: src=0 (fixed), st_prev=EMPTY_ENC (fixed) → constant index
        _start_idx: int = 0 * PUSH_STRIDE_S + tok_offset + EMPTY_ENC
        push_at_k: cp_model.BoolVar = is_push_flat[_start_idx]
        pop_at_k:  cp_model.BoolVar = is_pop_flat[_start_idx]

        model.add(st == tok).only_enforce_if(push_at_k)
        model.add(st == EMPTY_ENC).only_enforce_if(push_at_k.Not())

    else:
        # Non-start: src and st_prev are variables → add_element lookup
        st_prev = st_vars[prev_occ_idx]

        op_idx = model.new_int_var(0, PUSH_SIZE - 1, f"op_idx_{occ_idx}")
        model.add(op_idx == src * PUSH_STRIDE_S + tok_offset + st_prev)

        push_at_k = model.new_bool_var(f"push_at_{occ_idx}")
        model.add_element(op_idx, is_push_flat, push_at_k)

        pop_at_k = model.new_bool_var(f"pop_at_{occ_idx}")
        model.add_element(op_idx, is_pop_flat, pop_at_k)

        model.add(st == tok).only_enforce_if(push_at_k)
        model.add(st == EMPTY_ENC).only_enforce_if(pop_at_k)
        model.add(st == st_prev).only_enforce_if([push_at_k.Not(), pop_at_k.Not()])

    # --- Config index ---
    model.add(cfg_vars[occ_idx] == dst * ST_RANGE + st)
```

#### Step 5: Update solution extraction (lines 360–411)

Replace `learned_push` / `learned_pop` extraction:

```python
learned_push_configs: frozenset[tuple[int, int, int]] = frozenset(
    (s, tok, STACK_EMPTY if st_enc == EMPTY_ENC else st_enc)
    for s in range(num_states)
    for tok in range(vocab_size)
    for st_enc in range(ST_RANGE)
    if solver.value(is_push_flat[s * PUSH_STRIDE_S + tok * ST_RANGE + st_enc])
)
learned_pop_configs: frozenset[tuple[int, int, int]] = frozenset(
    (s, tok, STACK_EMPTY if st_enc == EMPTY_ENC else st_enc)
    for s in range(num_states)
    for tok in range(vocab_size)
    for st_enc in range(ST_RANGE)
    if solver.value(is_pop_flat[s * PUSH_STRIDE_S + tok * ST_RANGE + st_enc])
)
```

Update the `PDACircuitLM(...)` constructor (lines 402–412):
```python
return PDACircuitLM(
    ...
    push_configs=learned_push_configs,
    pop_configs=learned_pop_configs,
    ...
)
```

#### Step 6: Update _hash_fallback

Replace:
```python
push_tokens=frozenset(),
pop_tokens=frozenset(),
```
With:
```python
push_configs=frozenset(),
pop_configs=frozenset(),
```

#### Step 7: Run the joint PDA tests

```
py -3.12 -m pytest tests/test_joint_pda.py -v
```

Expected: all PASS.

Also run full suite to catch any regressions:
```
py -3.12 -m pytest -q
```

Expected: all green.

#### Step 8: Commit

```bash
git add circuit_lm/train_joint_pda_cpsat.py tests/test_joint_pda.py
git commit -m "feat: config-conditioned push/pop in joint PDA CP-SAT formulation"
```

---

### Task 5: Fix scripts and cli.py

**Files:**
- Modify: `circuit_lm/cli.py`
- Modify: `scripts/reproduce_depth_generalization.py`
- Modify: `scripts/robustness_experiment.py`

No new tests needed — these are display-only changes. Run the full suite after to confirm nothing regressed.

#### Step 1: Update cli.py (lines 160–163)

```python
# Remove:
push_n = len(model.push_tokens)
pop_n  = len(model.pop_tokens)
print(f"  push_tokens={push_n}  pop_tokens={pop_n}")

# Add:
push_n = len({tok for (_, tok, _) in model.push_configs})
pop_n  = len({tok for (_, tok, _) in model.pop_configs})
print(f"  push_token_ids={push_n}  pop_token_ids={pop_n}")
```

#### Step 2: Update reproduce_depth_generalization.py (lines 307, 321)

```python
# Remove:
print(f"  push_tokens={sorted(pda_model.push_tokens)}  pop_tokens={sorted(pda_model.pop_tokens)}")
print(f"  push_tokens={sorted(jpda_model.push_tokens)}  pop_tokens={sorted(jpda_model.pop_tokens)}")

# Add (extract unique token IDs from configs):
_push = sorted({tok for (_, tok, _) in pda_model.push_configs})
_pop  = sorted({tok for (_, tok, _) in pda_model.pop_configs})
print(f"  push_tokens={_push}  pop_tokens={_pop}")

_jpush = sorted({tok for (_, tok, _) in jpda_model.push_configs})
_jpop  = sorted({tok for (_, tok, _) in jpda_model.pop_configs})
print(f"  push_tokens={_jpush}  pop_tokens={_jpop}")
```

#### Step 3: Update robustness_experiment.py

Find all occurrences of `pda_model.push_tokens` and `pda_model.pop_tokens`:

```python
# Pattern to replace everywhere in the file:
# pda_model.push_tokens  →  frozenset(tok for (_, tok, _) in pda_model.push_configs)
# pda_model.pop_tokens   →  frozenset(tok for (_, tok, _) in pda_model.pop_configs)
# sorted(pda_model.push_tokens) → sorted({tok for (_, tok, _) in pda_model.push_configs})
# sorted(pda_model.pop_tokens)  → sorted({tok for (_, tok, _) in pda_model.pop_configs})
```

Lines to change (from grep output):
- 425–426: display lines → use set comprehension
- 483–484: correctness check `pda_model.push_tokens == frozenset({open_t})` →
  `{tok for (_, tok, _) in pda_model.push_configs} == {open_t}`
- 494–495: display lines
- 552–553: display lines
- 657–658: correctness check for multi-bracket case

#### Step 4: Run full test suite

```
py -3.12 -m pytest -q
```

Expected: 175+ tests, all PASS.

#### Step 5: Commit

```bash
git add circuit_lm/cli.py scripts/reproduce_depth_generalization.py scripts/robustness_experiment.py
git commit -m "fix: update scripts and cli to use push_configs/pop_configs API"
```

---

### Task 6: Verification script

**Files:**
- Create: `scripts/verify_joint_pda_small.py`

No unit tests — the script itself IS the experiment. Its output documents the research finding.

#### Step 1: Write the script

```python
"""Verify that train_joint_pda discovers the stack at T_total <= 2000.

Research question: the joint CP-SAT PDA solver timed out at 300 training
sequences (T_total ~3000) in the depth-generalization experiment.  Does it
recover push/pop configs when the corpus is kept within the 2000-token limit?

Settings: 50 balanced-paren sequences, vocab=3 (OPEN=0 CLOSE=1 EOS=2),
max_depth=3, 4 states, 20s budget, max_push=1, max_pop=1.

Usage
-----
    py -3.12 scripts/verify_joint_pda_small.py
    py -3.12 scripts/verify_joint_pda_small.py --train-seqs 40 --seed 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.eval import evaluate_pda
from circuit_lm.metrics import format_accuracy
from circuit_lm.train_joint_pda_cpsat import train_joint_pda

# Re-use sequence generators from the depth-generalization experiment
from reproduce_depth_generalization import gen_train_seqs, gen_test_seqs_at_depth  # type: ignore[import]

OPEN:       int = 0
CLOSE:      int = 1
EOS:        int = 2
VOCAB_SIZE: int = 3

MAX_TRAIN_DEPTH: int = 3
TEST_DEPTHS:     tuple[int, ...] = (3, 5, 8)

DEFAULT_TRAIN_SEQS: int = 50
DEFAULT_STEPS:      int = 20
DEFAULT_SEED:       int = 42

T_TOTAL_LIMIT: int = 2000


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-seqs", type=int, default=DEFAULT_TRAIN_SEQS)
    parser.add_argument("--steps",      type=int, default=DEFAULT_STEPS)
    args = parser.parse_args(argv)

    print()
    print("=== Joint-PDA Small-Scale Stack Verification ===")
    print(f"  seed={args.seed}  train_seqs={args.train_seqs}  steps={args.steps}s")
    print(f"  vocab: OPEN={OPEN} CLOSE={CLOSE} EOS={EOS}  max_depth={MAX_TRAIN_DEPTH}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Generate training data and assert T_total within solver limit    #
    # ------------------------------------------------------------------ #
    train_data = gen_train_seqs(
        max_depth=MAX_TRAIN_DEPTH, num_seqs=args.train_seqs, seed=args.seed
    )
    t_total = sum(len(seq) for seq in train_data)
    print(f"T_total = {t_total}  (limit: {T_TOTAL_LIMIT})")
    if t_total > T_TOTAL_LIMIT:
        print(f"WARNING: T_total {t_total} exceeds limit {T_TOTAL_LIMIT}. "
              f"Reduce --train-seqs.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Train joint PDA                                                   #
    # ------------------------------------------------------------------ #
    print(f"Training joint-PDA (num_states=4, max_push=1, max_pop=1) ...")
    model = train_joint_pda(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        num_states=4,
        stack_depth=10,
        steps=args.steps,
        max_push=1,
        max_pop=1,
        top_k_coverage=VOCAB_SIZE,
    )

    push_tok_ids = sorted({tok for (_, tok, _) in model.push_configs})
    pop_tok_ids  = sorted({tok for (_, tok, _) in model.pop_configs})
    print(f"  push token IDs: {push_tok_ids}")
    print(f"  pop  token IDs: {pop_tok_ids}")
    print(f"  push_configs count: {len(model.push_configs)}")
    print(f"  pop_configs  count: {len(model.pop_configs)}")

    stack_found = bool(model.push_configs) and bool(model.pop_configs)
    correct_push = push_tok_ids == [OPEN]
    correct_pop  = pop_tok_ids  == [CLOSE]

    if stack_found and correct_push and correct_pop:
        print("  PASS: stack correctly discovered (push=OPEN, pop=CLOSE)")
    elif stack_found:
        print(f"  PARTIAL: stack found but unexpected tokens "
              f"(expected push=[{OPEN}] pop=[{CLOSE}])")
    else:
        print("  WARN: no stack discovered within time budget")

    # ------------------------------------------------------------------ #
    # 3. Evaluate at test depths                                           #
    # ------------------------------------------------------------------ #
    print()
    sep = "-" * 48
    hdr = f"{'depth':>7}  {'joint-PDA':>12}  {'seqs':>6}"
    print(sep)
    print(hdr)
    print(sep)
    print("  (* = out-of-distribution depth)")

    for depth in TEST_DEPTHS:
        test_data = gen_test_seqs_at_depth(
            target_depth=depth, num_seqs=100, seed=args.seed + depth
        )
        r = evaluate_pda(model, test_data)
        ood = "*" if depth > MAX_TRAIN_DEPTH else " "
        print(f"{depth:>6}{ood}  {format_accuracy(r['correct'], r['total']):>12}"
              f"  {len(test_data):>6}")

    print(sep)
    print()


if __name__ == "__main__":
    main()
```

#### Step 2: Run the script

```
py -3.12 scripts/verify_joint_pda_small.py
```

Expected: prints T_total, discovered push/pop configs, and a 3-row depth table. Either `PASS: stack correctly discovered` or `WARN: no stack discovered` depending on solver luck within 20s.

#### Step 3: Commit

```bash
git add scripts/verify_joint_pda_small.py
git commit -m "feat: add joint-PDA small-scale stack verification script"
```

---

### Task 7: Full test suite + STATUS.md update

**Files:**
- Run: `py -3.12 -m pytest -q`
- Run: `py -3.12 scripts/verify_joint_pda_small.py`
- Modify: `STATUS.md`

#### Step 1: Run the full test suite

```
py -3.12 -m pytest -q
```

Expected: all tests pass (175+ tests). If any fail, diagnose and fix before proceeding.

#### Step 2: Run the verification script and note the result

```
py -3.12 scripts/verify_joint_pda_small.py
```

Record: T_total, push/pop token IDs discovered, PASS/WARN status, accuracy at depths 3/5/8.

#### Step 3: Update STATUS.md

Add a new section after "Robustness Experiment":

```markdown
## Joint-PDA Small-Scale Verification (2026-02-25)

Command:

    py -3.12 scripts/verify_joint_pda_small.py

Settings: 50 train seqs, vocab=3, max_depth=3, 4 states, 20s budget, seed=42.
T_total: [fill in] (limit: 2000).

Result:
- push token IDs: [fill in]
- pop  token IDs: [fill in]
- [PASS/WARN]: [fill in]

| depth | joint-PDA | seqs |
|-------|-----------|------|
| 3     | [fill in] | 100  |
| 5*    | [fill in] | 100  |
| 8*    | [fill in] | 100  |

## Config-Conditioned Stack Operations (2026-02-25)

- `push_tokens`/`pop_tokens` replaced by `push_configs`/`pop_configs` throughout.
- `PDACircuitLM.stack_op(state, token, stack_top)` — now takes full config triple.
- Two-phase PDA (`train_pda`) uses degenerate expansion: token-ID sets → all-(state, stack_top) triples.
- Joint PDA (`train_joint_pda`) uses `S × V × ST_RANGE` CP-SAT booleans for per-config ops.
- JSON migration shim: old `push_tokens`/`pop_tokens` fields automatically expand on load.
```

#### Step 4: Final commit

```bash
git add STATUS.md
git commit -m "docs: record joint-PDA verification result and config-ops migration in STATUS.md"
```

---

## Execution Notes

- The plan is strictly TDD: write failing test → implement → green → commit.
- `_simulate_and_collect` and `_simulate_and_collect_runtime` are intentionally left unchanged — they are private to `train_pda_cpsat.py` and never construct a `PDACircuitLM`.
- The no-float invariant (`tests/test_no_floats.py`) covers all new code automatically via its static regex scan.
- The `verify_joint_pda_small.py` script imports from `reproduce_depth_generalization.py` via a relative `sys.path` insert — same pattern as other scripts in this repo.
