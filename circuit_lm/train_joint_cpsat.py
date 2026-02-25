"""True joint CP-SAT FSM learning – states as decision variables.

Problem Statement
-----------------
Unlike ``train_cpsat.train`` (which fixes states via a rolling hash and then
optimises emissions), this module treats **states themselves** as integer
decision variables and jointly solves for transitions, states, and emissions
in a single CP-SAT model whose objective is next-token prediction accuracy.

CP-SAT Formulation
------------------
Given:
  N sequences x^(i) of length T_i,  vocab_size V,  num_states S
  T_total = Σ_i T_i

Variables (all integer):
  src[k]        ∈ {0..S-1}        state BEFORE consuming the k-th occurrence
  dst[k]        ∈ {0..S-1}        state AFTER  consuming the k-th occurrence
  delta[s, tok] ∈ {0..S-1}        transition table (S × V integer vars)
  pred[s]       ∈ {0..V-1}        emission table (S integer vars)
  pred_here[k]  ∈ {0..V-1}        = pred[dst[k]]  (auxiliary)
  match[k]      ∈ {0, 1}          1 iff prediction is correct at k

Variable count: O(T_total + S·V)
  – State vars:      2 × T_total  (src + dst per occurrence)
  – Transition:      S × V
  – Emission:        S
  – Auxiliary:       2 × T_total  (pred_here + match per occurrence)

Constraints:
  src[seq_start] = 0                                  (fixed initial state)
  src[k+1] = dst[k]                                   (chain within sequence)
  dst[k] = delta[src[k], seq[k]]
    ↑ encoded via add_element(src[k], delta_row_for_tok, dst[k])
  pred_here[k] = pred[dst[k]]
    ↑ encoded via add_element(dst[k], pred_vars, pred_here[k])
  match[k] = 1 ↔ pred_here[k] == next_tok[k]

Objective:
  maximize Σ_k match[k]           (total correct next-token predictions)

This is a pure integer program – no floating-point values anywhere.

Scalability
-----------
Variable count: O(T_total + S·V).  Runtime is *not* linear in variable count.
The dominant cost is **symmetry and branching** in the CP-SAT search.

Without symmetry breaking, equivalent solutions (state-label permutations)
flood the search tree and the solver stalls well before T_total=4000.  The
``sym_break=True`` parameter (default) adds first-appearance ordering
constraints that eliminate this issue — they are **not optional** for
non-trivial problems.

Empirically tractable with symmetry breaking:
  T_total ≤ 4_000   (e.g. 200 seqs × avg 20 tokens)
  S = num_states ≤ 16
  V = vocab_size ≤ 64

For larger corpora, use ``train_cpsat.train`` with hash-bootstrapped states.

Symmetry Breaking: First-Appearance Ordering
--------------------------------------------
State labels are permutable: swapping all occurrences of labels 3 and 7 gives
an identical-objective solution.  Without breaking this symmetry, the solver
wastes time rediscovering equivalent solutions.

Constraint: state[k] ≤ max(state[0..k-1]) + 1

Encoded incrementally using a running maximum:
  msf[k] = max(msf[k-1], state[k-1])
  state[k] ≤ msf[k] + 1

This forces new state labels to appear in ascending order of first visit:
label 0 is always first (initial state = 0), label 1 appears before 2, etc.
The shared delta/pred tables ensure this applies globally across sequences.

Differences from existing joint bootstrap
-----------------------------------------
``train_cpsat._joint_bootstrap_transition_state_cpsat`` maximises *agreement
with hash-based state assignments* as a proxy objective.  This module's
objective is **prediction accuracy directly** – no hash anchoring, no proxy.

The solver is free to discover any state partitioning that maximises the
next-token prediction count, subject only to FSM semantics and symmetry
breaking.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from circuit_lm.circuits import CircuitLM, HASH_PRIME


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


def train_joint(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    steps: int,
    top_k_coverage: int = 0,
    sym_break: bool = True,
) -> CircuitLM:
    """Train a CircuitLM via true joint CP-SAT (states as decision variables).

    States, transitions, and emissions are all integer decision variables.
    The objective is next-token prediction accuracy – not agreement with any
    hash prior.

    Args:
        sequences:       List of integer token-ID sequences.
        vocab_size:      Number of distinct token IDs (V).
        num_states:      Number of FSM states (S).  Must be a power of 2.
                         Keep S ≤ 16 for tractability.
        steps:           CP-SAT wall-clock budget in integer seconds.
        top_k_coverage:  If > 0, add a coverage constraint: the top-K globally
                         frequent tokens must each be predicted by at least one
                         state.  0 disables.
        sym_break:       If True (default), add first-appearance ordering
                         constraints to break state-label symmetry.  Strongly
                         recommended – disabling it will stall the solver on any
                         non-trivial input.

    Returns:
        A trained :class:`~circuit_lm.circuits.CircuitLM`.

    Notes:
        - ``num_states`` must be a power of 2 (state_bits = log2(num_states)).
        - T_total (sum of all sequence lengths) should be ≤ 4_000 for
          tractable solve times; S·V should be ≤ 1_024.
        - The solver may return FEASIBLE (not OPTIMAL) within the time limit;
          the best solution found so far is used.
        - Falls back to hash-based argmax when CP-SAT finds no solution.
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if num_states & (num_states - 1):
        raise ValueError("num_states must be a power of 2")

    # state_bits: num_states = 1 << state_bits  (valid because num_states is 2^k)
    state_bits: int = num_states.bit_length() - 1

    # ------------------------------------------------------------------
    # Flatten sequences into a linear occurrence list.
    # Each entry: (prev_occ_idx, tok, next_tok_or_neg1)
    # prev_occ_idx == -1 marks the first token in each sequence (state resets to 0).
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
        return CircuitLM(
            vocab_size=vocab_size,
            num_states=num_states,
            state_bits=state_bits,
        )

    model = cp_model.CpModel()

    # ------------------------------------------------------------------
    # Decision variables: transition table and emission table
    # ------------------------------------------------------------------

    # delta[(s, tok)] = next_state   (shared across all sequences)
    delta: dict[tuple[int, int], cp_model.IntVar] = {
        (s, tok): model.new_int_var(0, num_states - 1, f"delta_{s}_{tok}")
        for s in range(num_states)
        for tok in range(vocab_size)
    }

    # pred[s] = predicted next token emitted from state s
    pred: list[cp_model.IntVar] = [
        model.new_int_var(0, vocab_size - 1, f"pred_{s}")
        for s in range(num_states)
    ]

    # ------------------------------------------------------------------
    # Per-occurrence variables: src, dst, pred_here, match
    # ------------------------------------------------------------------
    src_vars: list[cp_model.IntVar] = []
    dst_vars: list[cp_model.IntVar] = []

    for occ_idx in range(len(occurrences)):
        src_vars.append(model.new_int_var(0, num_states - 1, f"src_{occ_idx}"))
        dst_vars.append(model.new_int_var(0, num_states - 1, f"dst_{occ_idx}"))

    # ------------------------------------------------------------------
    # Constraints: initial state, chain, transition
    # ------------------------------------------------------------------

    # Pre-build delta rows indexed by token for add_element lookup
    delta_rows: list[list[cp_model.IntVar]] = [
        [delta[(s, tok)] for s in range(num_states)]
        for tok in range(vocab_size)
    ]

    for occ_idx, (prev_occ_idx, tok, _next_tok) in enumerate(occurrences):
        src = src_vars[occ_idx]
        dst = dst_vars[occ_idx]

        # Initial state for each new sequence
        if prev_occ_idx < 0:
            model.add(src == 0)
        else:
            # src[k] == dst[k-1]  (chain within same sequence)
            model.add(src == dst_vars[prev_occ_idx])

        # dst[k] = delta[src[k], tok]
        # add_element(index_var, array_of_vars, result_var)
        # → result_var = array_of_vars[index_var]
        model.add_element(src, delta_rows[tok], dst)

    # ------------------------------------------------------------------
    # Symmetry breaking: first-appearance ordering (incremental max)
    # ------------------------------------------------------------------
    # Constraint: src[k] ≤ max(src[0..k-1]) + 1
    # Encoded via running maximum: msf[k] = max(msf[k-1], src[k-1])
    # This forces new state labels to appear in ascending order.
    # Without this, state-label permutations flood the search tree.
    # ------------------------------------------------------------------
    if sym_break and len(occurrences) > 1:
        msf_prev: cp_model.IntVar | None = src_vars[0]  # = 0 (initial, constrained above)
        for k in range(1, len(occurrences)):
            msf_k = model.new_int_var(0, num_states - 1, f"msf_{k}")
            # msf_k = max(msf_prev, src[k-1])
            model.add_max_equality(msf_k, [msf_prev, src_vars[k - 1]])  # type: ignore[list-item]
            # src[k] can be an existing state or exactly the next new label
            model.add(src_vars[k] <= msf_k + 1)
            msf_prev = msf_k

    # ------------------------------------------------------------------
    # Prediction matches: pred_here[k] = pred[dst[k]], match iff == next_tok
    # ------------------------------------------------------------------
    match_vars: list[cp_model.BoolVar] = []

    for occ_idx, (_prev, _tok, next_tok) in enumerate(occurrences):
        if next_tok < 0:
            # End of sequence – no prediction target, contribute zero to objective
            b = model.new_bool_var(f"match_{occ_idx}")
            model.add(b == 0)
            match_vars.append(b)
            continue

        ph = model.new_int_var(0, vocab_size - 1, f"ph_{occ_idx}")
        # ph = pred[dst[k]]  (emission after consuming the current token)
        model.add_element(dst_vars[occ_idx], pred, ph)

        match = model.new_bool_var(f"match_{occ_idx}")
        model.add(ph == next_tok).only_enforce_if(match)
        model.add(ph != next_tok).only_enforce_if(match.Not())
        match_vars.append(match)

    # ------------------------------------------------------------------
    # Objective: maximise total correct next-token predictions
    # ------------------------------------------------------------------
    model.maximize(sum(match_vars))

    # ------------------------------------------------------------------
    # Optional coverage constraint
    # ------------------------------------------------------------------
    if top_k_coverage > 0:
        global_counts: list[int] = [0] * vocab_size
        for _, _, nt in occurrences:
            if 0 <= nt < vocab_size:
                global_counts[nt] += 1
        ranked = sorted(range(vocab_size), key=global_counts.__getitem__, reverse=True)
        k_cov = min(top_k_coverage, num_states)
        top_tokens = [t for t in ranked[:k_cov] if global_counts[t] > 0]

        for t in top_tokens:
            covers: list[cp_model.BoolVar] = []
            for s in range(num_states):
                b = model.new_bool_var(f"cov_{s}_{t}")
                model.add(pred[s] == t).only_enforce_if(b)
                covers.append(b)
            model.add(sum(covers) >= 1)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = steps
    solver.parameters.log_search_progress = False
    status = solver.solve(model)

    # ------------------------------------------------------------------
    # Extract solution or fall back to greedy hash baseline
    # ------------------------------------------------------------------
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        learned_delta: dict[tuple[int, int], int] = {
            (s, tok): solver.value(delta[(s, tok)])
            for s in range(num_states)
            for tok in range(vocab_size)
        }
        learned_pred: dict[int, int] = {
            s: solver.value(pred[s]) for s in range(num_states)
        }
        # Collect state_counts from solved runtime states for inference
        state_counts: dict[int, list[int]] = {s: [0] * vocab_size for s in range(num_states)}
        for occ_idx, (_prev, _tok, next_tok) in enumerate(occurrences):
            if next_tok >= 0:
                dst_s = solver.value(dst_vars[occ_idx])
                if 0 <= dst_s < num_states:
                    state_counts[dst_s][next_tok] += 1
    else:
        # Fallback: hash-based state assignment + greedy argmax per state
        state_counts = {s: [0] * vocab_size for s in range(num_states)}
        for seq in sequences:
            state = 0
            for pos, tok in enumerate(seq):
                if not (0 <= tok < vocab_size):
                    continue
                next_s = (state * HASH_PRIME + tok + 1) % num_states
                if pos + 1 < len(seq):
                    nt = seq[pos + 1]
                    if 0 <= nt < vocab_size:
                        state_counts[next_s][nt] += 1
                state = next_s

        learned_delta = {
            (s, tok): (s * HASH_PRIME + tok + 1) % num_states
            for s in range(num_states)
            for tok in range(vocab_size)
        }
        learned_pred = {
            s: max(range(vocab_size), key=state_counts[s].__getitem__)
            for s in range(num_states)
        }

    return CircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=state_bits,
        transitions=learned_delta,
        state_counts=state_counts,
        pred_tokens=learned_pred,
    )
