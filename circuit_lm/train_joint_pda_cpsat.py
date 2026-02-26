"""True joint CP-SAT PDA learning – states, stack ops, and emissions as decision variables.

Problem Statement
-----------------
Unlike ``train_pda_cpsat.train_pda`` (two sequential CP-SAT phases with
hash-anchored FSM states), this module treats **states, stack operations
(push/pop/noop per token), and config emissions** as integer decision variables
in a single CP-SAT model whose objective is next-token prediction accuracy.

The stack is modelled at depth 1: only the topmost element is tracked.
A POP on an empty stack is a silent no-op (stack top remains STACK_EMPTY).

CP-SAT Formulation
------------------
Given:
  N sequences x^(i) of length T_i,  vocab_size V,  num_states S
  T_total = Σ_i T_i
  ST_RANGE  = V + 1          (stack-top values: 0..V-1 are tokens; V = EMPTY_ENC)
  CONFIG_RANGE = S * ST_RANGE

Variables (all integer):
  delta[(s, tok)]  ∈ {0..S-1}           FSM transition table (S × V vars)
  is_push[tok]     ∈ {0, 1}             1 if tok triggers PUSH        (V bool vars)
  is_pop[tok]      ∈ {0, 1}             1 if tok triggers POP         (V bool vars)
  pred_flat[idx]   ∈ {0..V-1}           emission per config, flattened (CONFIG_RANGE)
                                         idx = s * ST_RANGE + st_enc
  src[k]           ∈ {0..S-1}           FSM state before consuming tok[k]
  dst[k]           ∈ {0..S-1}           FSM state after  consuming tok[k]
  st[k]            ∈ {0..V}             stack top (enc.) after consuming tok[k]
  cfg[k]           ∈ {0..CONFIG_RANGE-1} = dst[k] * ST_RANGE + st[k]
  ph[k]            ∈ {0..V-1}           = pred_flat[cfg[k]]
  match[k]         ∈ {0, 1}             1 iff ph[k] == next_tok[k]

Constraints:
  src[seq_start] = 0                               (fixed initial state)
  src[k+1] = dst[k]                                (chain within sequence)
  dst[k] = delta[src[k], tok[k]]                   (via add_element)
  is_push[tok] + is_pop[tok] ≤ 1                   (mutually exclusive ops)
  Stack top (sequence start – initial stack is EMPTY):
    st[k] = tok       if is_push[tok]
    st[k] = EMPTY_ENC otherwise           (pop-from-empty = noop)
  Stack top (non-start – previous st = st[prev_k]):
    st[k] = tok       if is_push[tok]
    st[k] = EMPTY_ENC if is_pop[tok]
    st[k] = st[prev_k] otherwise           (noop)
  cfg[k] = dst[k] * ST_RANGE + st[k]               (linear arithmetic)
  ph[k] = pred_flat[cfg[k]]                         (via add_element)
  match[k] ↔ (ph[k] == next_tok[k])

Objective:
  maximize Σ_k match[k]

Symmetry breaking:
  First-appearance ordering on src[k] (same scheme as train_joint_cpsat).

Scalability
-----------
The shared is_push/is_pop variables couple ALL occurrences of each token,
making this harder than the FSM joint formulation.  The solver must commit
to a global stack policy simultaneously with state and emission assignments.

Empirically tractable:
  T_total  ≤ 2_000   (tighter than FSM joint due to token-policy coupling)
  S = num_states ≤ 8
  V = vocab_size ≤ 32

For larger corpora use ``train_pda_cpsat.train_pda`` with hash-bootstrapped
states and the two-phase sequential approach.

Differences from train_pda_cpsat
---------------------------------
Phase 1 of ``train_pda_cpsat`` selects push/pop tokens by maximising
co-occurrence scores – a proxy for structural utility.  Phase 2 then solves
config emissions with states still hash-anchored.  This module is different:
  - push/pop tokens are decision variables whose selection is governed by
    prediction accuracy, not co-occurrence
  - FSM states are free decision variables (not hash-fixed)
  - all three (transitions, stack policy, emissions) are solved simultaneously
    in a single model with a single objective
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from circuit_lm.circuits import HASH_PRIME
from circuit_lm.pda import PDACircuitLM, STACK_EMPTY


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


def train_joint_pda(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    stack_depth: int,
    steps: int,
    max_push: int | None = None,
    max_pop: int | None = None,
    top_k_coverage: int = 0,
    sym_break: bool = True,
) -> PDACircuitLM:
    """Train a PDACircuitLM via true joint CP-SAT.

    States, FSM transitions, per-token stack operations (push / pop / noop),
    and per-config emissions are all integer decision variables solved jointly.
    The objective is next-token prediction accuracy on the training corpus.

    Args:
        sequences:       List of integer token-ID sequences.
        vocab_size:      Number of distinct token IDs (V).
        num_states:      Number of FSM states (S).  Must be a power of 2.
                         Keep S ≤ 8 for tractable solve times.
        stack_depth:     Passed through to the returned PDACircuitLM.  The
                         joint formulation models a depth-1 stack internally;
                         set stack_depth=1 for full structural fidelity.
        steps:           CP-SAT wall-clock budget in integer seconds.
        max_push:        If given, at most this many tokens may be PUSH tokens.
        max_pop:         If given, at most this many tokens may be POP tokens.
        top_k_coverage:  If > 0, the top-K globally frequent tokens must each
                         be predicted by at least one config (state, stack_top).
        sym_break:       If True (default), add first-appearance ordering
                         constraints on src states to break label-permutation
                         symmetry.  Strongly recommended.

    Returns:
        A trained :class:`~circuit_lm.pda.PDACircuitLM`.

    Notes:
        - ``num_states`` must be a power of 2.
        - T_total ≤ 2_000 and S ≤ 8 are recommended for tractability.
        - The solver may return FEASIBLE (not OPTIMAL) within the time limit.
        - Falls back to a hash-FSM baseline (no push/pop tokens) if CP-SAT
          finds no solution within the time budget.
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if num_states & (num_states - 1):
        raise ValueError("num_states must be a power of 2")
    if stack_depth < 0:
        raise ValueError("stack_depth must be non-negative")

    state_bits: int = num_states.bit_length() - 1

    # EMPTY_ENC: CP-SAT-safe encoding of STACK_EMPTY.  Must be ≥ 0.
    # Converted back to STACK_EMPTY (-1) when building PDACircuitLM output.
    EMPTY_ENC: int = vocab_size

    # ST_RANGE: number of distinct encoded stack-top values (token IDs + empty).
    ST_RANGE: int = vocab_size + 1

    # CONFIG_RANGE: total number of (state, stack_top_enc) pairs.
    CONFIG_RANGE: int = num_states * ST_RANGE

    # ------------------------------------------------------------------
    # Flatten sequences into a linear occurrence list.
    # Each entry: (prev_occ_idx, tok, next_tok_or_neg1)
    # prev_occ_idx == -1 marks the first token in each sequence.
    # ------------------------------------------------------------------
    occurrences: list[tuple[int, int, int]] = []

    for seq in sequences:
        if len(seq) < 2:
            continue
        prev_idx = -1
        for pos, tok in enumerate(seq):
            if not (0 <= tok < vocab_size):
                continue
            next_tok = seq[pos + 1] if pos + 1 < len(seq) else -1
            if next_tok != -1 and not (0 <= next_tok < vocab_size):
                next_tok = -1
            occurrences.append((prev_idx, tok, next_tok))
            prev_idx = len(occurrences) - 1

    if not occurrences:
        return _hash_fallback(sequences, vocab_size, num_states, state_bits, stack_depth)

    model = cp_model.CpModel()

    # ------------------------------------------------------------------
    # Decision variables: FSM transition table
    # delta[(s, tok)] = next_state after consuming tok from state s
    # ------------------------------------------------------------------
    delta: dict[tuple[int, int], cp_model.IntVar] = {
        (s, tok): model.new_int_var(0, num_states - 1, f"delta_{s}_{tok}")
        for s in range(num_states)
        for tok in range(vocab_size)
    }

    # Pre-build delta columns indexed by token for add_element lookups.
    # delta_cols[tok] = [delta[(0, tok)], delta[(1, tok)], ..., delta[(S-1, tok)]]
    # add_element(src, delta_cols[tok], dst) encodes dst = delta[src, tok].
    delta_cols: list[list[cp_model.IntVar]] = [
        [delta[(s, tok)] for s in range(num_states)]
        for tok in range(vocab_size)
    ]

    # ------------------------------------------------------------------
    # Decision variables: per-config stack operations
    # Flat index: s * PUSH_STRIDE_S + tok * ST_RANGE + st_enc
    # ------------------------------------------------------------------
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

    # Mutual exclusivity per (state, token, stack_top) triple.
    for _s in range(num_states):
        for _tok in range(vocab_size):
            for _st_enc in range(ST_RANGE):
                _i = _s * PUSH_STRIDE_S + _tok * ST_RANGE + _st_enc
                model.add(is_push_flat[_i] + is_pop_flat[_i] <= 1)

    # Budget constraints: max_push / max_pop count distinct token IDs.
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

    # ------------------------------------------------------------------
    # Decision variables: emission table (flattened over configs)
    # pred_flat[s * ST_RANGE + st_enc] = predicted next-token for config (s, st_enc)
    # ------------------------------------------------------------------
    pred_flat: list[cp_model.IntVar] = [
        model.new_int_var(0, vocab_size - 1, f"pred_{idx}")
        for idx in range(CONFIG_RANGE)
    ]

    # ------------------------------------------------------------------
    # Per-occurrence variables
    # ------------------------------------------------------------------
    src_vars: list[cp_model.IntVar] = []
    dst_vars: list[cp_model.IntVar] = []
    st_vars:  list[cp_model.IntVar] = []   # stack top (encoded) after consuming tok
    cfg_vars: list[cp_model.IntVar] = []   # = dst * ST_RANGE + st

    for occ_idx in range(len(occurrences)):
        src_vars.append(model.new_int_var(0, num_states - 1, f"src_{occ_idx}"))
        dst_vars.append(model.new_int_var(0, num_states - 1, f"dst_{occ_idx}"))
        st_vars.append(model.new_int_var(0, EMPTY_ENC, f"st_{occ_idx}"))
        cfg_vars.append(model.new_int_var(0, CONFIG_RANGE - 1, f"cfg_{occ_idx}"))

    # ------------------------------------------------------------------
    # Constraints: initial state, chain, FSM transition, stack top update
    # ------------------------------------------------------------------
    for occ_idx, (prev_occ_idx, tok, _next_tok) in enumerate(occurrences):
        src = src_vars[occ_idx]
        dst = dst_vars[occ_idx]
        st  = st_vars[occ_idx]

        # FSM initial state / chain
        if prev_occ_idx < 0:
            model.add(src == 0)
        else:
            model.add(src == dst_vars[prev_occ_idx])

        # FSM transition: dst = delta[src, tok]
        model.add_element(src, delta_cols[tok], dst)

        # Stack op lookup and stack top update
        tok_offset: int = tok * ST_RANGE

        if prev_occ_idx < 0:
            # Sequence start: src=0 (fixed), st_prev=EMPTY_ENC (fixed) → constant index.
            _start_idx: int = 0 * PUSH_STRIDE_S + tok_offset + EMPTY_ENC
            push_at_k: cp_model.BoolVar = is_push_flat[_start_idx]
            pop_at_k:  cp_model.BoolVar = is_pop_flat[_start_idx]

            model.add(st == tok).only_enforce_if(push_at_k)
            model.add(st == EMPTY_ENC).only_enforce_if(push_at_k.Not())
        else:
            # Non-start: src and st_prev are variables — use add_element.
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

        # Config index (linear: constant * IntVar + IntVar is supported)
        model.add(cfg_vars[occ_idx] == dst * ST_RANGE + st)

    # ------------------------------------------------------------------
    # Symmetry breaking: first-appearance ordering on src states
    # Constraint: src[k] ≤ max(src[0..k-1]) + 1
    # Encoded with a running maximum variable.
    # ------------------------------------------------------------------
    if sym_break and len(occurrences) > 1:
        msf_prev: cp_model.IntVar | None = src_vars[0]  # == 0 (fixed above)
        for k in range(1, len(occurrences)):
            msf_k = model.new_int_var(0, num_states - 1, f"msf_{k}")
            model.add_max_equality(msf_k, [msf_prev, src_vars[k - 1]])  # type: ignore[list-item]
            model.add(src_vars[k] <= msf_k + 1)
            msf_prev = msf_k

    # ------------------------------------------------------------------
    # Prediction: ph[k] = pred_flat[cfg[k]], match iff ph[k] == next_tok[k]
    # ------------------------------------------------------------------
    match_vars: list[cp_model.BoolVar] = []

    for occ_idx, (_prev, _tok, next_tok) in enumerate(occurrences):
        if next_tok < 0:
            # End of sequence — no prediction target; contributes 0.
            b = model.new_bool_var(f"match_{occ_idx}")
            model.add(b == 0)
            match_vars.append(b)
            continue

        ph = model.new_int_var(0, vocab_size - 1, f"ph_{occ_idx}")
        model.add_element(cfg_vars[occ_idx], pred_flat, ph)

        match = model.new_bool_var(f"match_{occ_idx}")
        model.add(ph == next_tok).only_enforce_if(match)
        model.add(ph != next_tok).only_enforce_if(match.Not())
        match_vars.append(match)

    # ------------------------------------------------------------------
    # Optional top-K coverage constraint:
    # each of the globally most-frequent top_k_coverage tokens must be
    # predicted by at least one config.
    # ------------------------------------------------------------------
    if top_k_coverage > 0:
        global_counts: list[int] = [0] * vocab_size
        for _, _, nt in occurrences:
            if 0 <= nt < vocab_size:
                global_counts[nt] += 1
        ranked = sorted(range(vocab_size), key=global_counts.__getitem__, reverse=True)
        k_cov = min(top_k_coverage, CONFIG_RANGE)
        top_tokens = [t for t in ranked[:k_cov] if global_counts[t] > 0]

        for t in top_tokens:
            covers: list[cp_model.BoolVar] = []
            for idx in range(CONFIG_RANGE):
                b = model.new_bool_var(f"cov_{idx}_{t}")
                model.add(pred_flat[idx] == t).only_enforce_if(b)
                covers.append(b)
            model.add(sum(covers) >= 1)

    # ------------------------------------------------------------------
    # Objective: maximise total correct next-token predictions
    # ------------------------------------------------------------------
    model.maximize(sum(match_vars))

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = steps
    solver.parameters.log_search_progress = False
    status = solver.solve(model)

    # ------------------------------------------------------------------
    # Extract solution or fall back to hash-FSM baseline (no push/pop)
    # ------------------------------------------------------------------
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _hash_fallback(sequences, vocab_size, num_states, state_bits, stack_depth)

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

    learned_delta: dict[tuple[int, int], int] = {
        (s, tok): solver.value(delta[(s, tok)])
        for s in range(num_states)
        for tok in range(vocab_size)
    }

    # Emission table: decode EMPTY_ENC back to STACK_EMPTY (-1)
    config_pred_tokens: dict[tuple[int, int], int] = {}
    for s in range(num_states):
        for st_enc in range(ST_RANGE):
            idx = s * ST_RANGE + st_enc
            pt = solver.value(pred_flat[idx])
            st_key = STACK_EMPTY if st_enc == EMPTY_ENC else st_enc
            config_pred_tokens[(s, st_key)] = pt

    # Collect config_counts from solved runtime (state, stack_top) assignments
    config_counts: dict[tuple[int, int], list[int]] = {}
    for occ_idx, (_prev, _tok, next_tok) in enumerate(occurrences):
        if next_tok >= 0:
            dst_s   = solver.value(dst_vars[occ_idx])
            st_enc  = solver.value(st_vars[occ_idx])
            st_key  = STACK_EMPTY if st_enc == EMPTY_ENC else st_enc
            cfg_key = (dst_s, st_key)
            if cfg_key not in config_counts:
                config_counts[cfg_key] = [0] * vocab_size
            config_counts[cfg_key][next_tok] += 1

    # Full transition table: learned pairs override hash fallback
    transitions: dict[tuple[int, int], int] = {
        (s, t): (s * HASH_PRIME + t + 1) % num_states
        for s in range(num_states)
        for t in range(vocab_size)
    }
    transitions.update(learned_delta)

    return PDACircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=state_bits,
        stack_depth=stack_depth,
        push_configs=learned_push_configs,
        pop_configs=learned_pop_configs,
        transitions=transitions,
        config_counts=config_counts,
        config_pred_tokens=config_pred_tokens,
    )


# ---------------------------------------------------------------------------
# Hash-FSM fallback (no push/pop tokens)
# ---------------------------------------------------------------------------


def _hash_fallback(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    state_bits: int,
    stack_depth: int,
) -> PDACircuitLM:
    """Return a hash-FSM PDA baseline with no push/pop tokens.

    Used when CP-SAT finds no solution within the time budget.  All configs
    degenerate to (state, STACK_EMPTY) since the stack is never pushed.
    """
    transitions: dict[tuple[int, int], int] = {
        (s, t): (s * HASH_PRIME + t + 1) % num_states
        for s in range(num_states)
        for t in range(vocab_size)
    }
    config_counts: dict[tuple[int, int], list[int]] = {}

    for seq in sequences:
        state = 0
        for pos, tok in enumerate(seq):
            if not (0 <= tok < vocab_size):
                continue
            next_s = transitions[(state, tok)]
            if pos + 1 < len(seq):
                nt = seq[pos + 1]
                if 0 <= nt < vocab_size:
                    cfg = (next_s, STACK_EMPTY)
                    if cfg not in config_counts:
                        config_counts[cfg] = [0] * vocab_size
                    config_counts[cfg][nt] += 1
            state = next_s

    config_pred_tokens: dict[tuple[int, int], int] = {}
    for cfg, counts in config_counts.items():
        config_pred_tokens[cfg] = max(range(vocab_size), key=counts.__getitem__)

    return PDACircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=state_bits,
        stack_depth=stack_depth,
        push_configs=frozenset(),
        pop_configs=frozenset(),
        transitions=transitions,
        config_counts=config_counts,
        config_pred_tokens=config_pred_tokens,
    )
