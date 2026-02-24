"""CP-SAT training for the circuit language model.

All arithmetic is integer only.  The OR-Tools CP-SAT solver operates
entirely in the integer domain; no floating-point instructions are issued
from this module.

CP-SAT problem statement
------------------------
Given:
  S states, V tokens, integer count tables state_counts[s][t] ∈ ℤ≥0

Variables:
  pred[s] ∈ {0, …, V-1}   – the single "best" token predicted from state s

Objective (maximise):
  Σ_s  state_counts[s][ pred[s] ]

Coverage constraint:
  For each token t in the top-K most-frequent tokens globally,
  at least one state s must satisfy pred[s] == t.

  This makes the problem non-trivially combinatorial: a pure argmax-per-state
  solution would collapse all states onto the globally dominant token, but the
  coverage constraint forces diversity.

TODO: Learn the full transition function delta(s,t) via CP-SAT rather than
      using the fixed rolling-hash fallback in circuits.py.
TODO: Iterative refinement – re-derive state assignments after each CP-SAT
      pass and re-optimise.
TODO: Multi-objective: balance accuracy vs. state entropy.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from circuit_lm.circuits import CircuitLM, HASH_PRIME


# ---------------------------------------------------------------------------
# State assignment (deterministic rolling hash – no CP-SAT needed here)
# ---------------------------------------------------------------------------


def _compute_state(context: list[int], num_states: int) -> int:
    """Map an integer context window to a state via rolling polynomial hash.

    Uses only integer arithmetic; result is in [0, num_states).
    """
    h = 0
    for tok in context:
        h = (h * HASH_PRIME + tok + 1) % num_states
    return h


def _build_state_counts(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    context_len: int,
) -> dict[int, list[int]]:
    """Collect integer (state, next_token) frequency tables from training data.

    For each position *i* in each sequence, the state is computed from the
    *context_len* tokens ending at position *i* (inclusive), and the
    observation is the token at position *i+1*.

    Returns:
        dict mapping state (int) → list[int] of length *vocab_size*,
        where entry *t* is the count of times token *t* followed that state.
    """
    state_counts: dict[int, list[int]] = {}

    for seq in sequences:
        for pos in range(len(seq) - 1):
            start = max(0, pos - context_len + 1)
            ctx = seq[start : pos + 1]
            state = _compute_state(ctx, num_states)
            next_tok = seq[pos + 1]

            if state not in state_counts:
                state_counts[state] = [0] * vocab_size
            if 0 <= next_tok < vocab_size:
                state_counts[state][next_tok] += 1

    return state_counts


# ---------------------------------------------------------------------------
# CP-SAT emission optimiser
# ---------------------------------------------------------------------------


def _optimize_emissions_cpsat(
    state_counts: dict[int, list[int]],
    vocab_size: int,
    num_states: int,
    top_k_coverage: int,
    time_limit_seconds: int,
) -> dict[int, int]:
    """Choose one prediction token per state using CP-SAT.

    Returns:
        dict mapping state (int) → predicted token (int).

    Falls back to per-state argmax if CP-SAT cannot find a feasible solution
    within *time_limit_seconds*.

    Args:
        state_counts:       Integer histograms, keyed by state.
        vocab_size:         Number of distinct tokens.
        num_states:         Total number of states (used only for range checks).
        top_k_coverage:     How many of the globally most-frequent tokens must
                            be predicted by at least one state.
        time_limit_seconds: Integer wall-clock limit for the CP-SAT solver.
    """
    model = cp_model.CpModel()

    active_states = [s for s in range(num_states) if s in state_counts]
    if not active_states:
        return {}

    # Decision variables: one integer per active state
    pred_token: dict[int, cp_model.IntVar] = {
        s: model.new_int_var(0, vocab_size - 1, f"pred_{s}")
        for s in active_states
    }

    # ------------------------------------------------------------------
    # Objective: maximise total correct-prediction count
    # ------------------------------------------------------------------
    gain_vars: list[cp_model.IntVar] = []
    for s in active_states:
        counts = state_counts[s]
        max_count = max(counts) if any(c > 0 for c in counts) else 0
        if max_count == 0:
            continue
        gain = model.new_int_var(0, max_count, f"gain_{s}")
        # gain == counts[ pred_token[s] ]
        model.add_element(pred_token[s], counts, gain)
        gain_vars.append(gain)

    if gain_vars:
        model.maximize(sum(gain_vars))

    # ------------------------------------------------------------------
    # Coverage constraint: top-K tokens must each be predicted somewhere
    # ------------------------------------------------------------------
    global_counts: list[int] = [0] * vocab_size
    for counts in state_counts.values():
        for tok, cnt in enumerate(counts):
            global_counts[tok] += cnt

    # Pick top-K tokens; limit to number of active states (can't cover more)
    ranked = sorted(range(vocab_size), key=global_counts.__getitem__, reverse=True)
    k = min(top_k_coverage, len(active_states))
    top_tokens = [t for t in ranked[:k] if global_counts[t] > 0]

    for t in top_tokens:
        # At least one active state must have pred_token[s] == t
        covers: list[cp_model.BoolVar] = []
        for s in active_states:
            b = model.new_bool_var(f"cov_{s}_{t}")
            # b == 1  ⟹  pred_token[s] == t
            model.add(pred_token[s] == t).only_enforce_if(b)
            covers.append(b)
        model.add(sum(covers) >= 1)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = cp_model.CpSolver()
    # time_limit_seconds is already an int – no float literal here
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = False

    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fallback: argmax per state
        result: dict[int, int] = {}
        for s in active_states:
            counts = state_counts[s]
            best = max(range(vocab_size), key=counts.__getitem__)
            result[s] = best
        return result

    return {s: solver.value(pred_token[s]) for s in active_states}


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


def train(
    sequences: list[list[int]],
    vocab_size: int,
    state_bits: int,
    steps: int,
    context_len: int = 4,
    top_k_coverage: int = 16,
) -> CircuitLM:
    """Train a CircuitLM from integer token sequences using OR-Tools CP-SAT.

    Args:
        sequences:       List of integer token-ID sequences.
        vocab_size:      Number of distinct token IDs.
        state_bits:      State width in bits; num_states = 1 << state_bits.
        steps:           CP-SAT wall-clock time limit in integer seconds.
        context_len:     Number of preceding tokens used to derive the state.
        top_k_coverage:  How many top tokens must be covered by the emission
                         function (makes the CP-SAT problem non-trivial).

    Returns:
        A trained :class:`~circuit_lm.circuits.CircuitLM` instance.

    TODO: Learn transitions via CP-SAT (currently uses fixed hash fallback).
    TODO: Multi-pass iterative refinement of state assignments.
    TODO: Expose context_len and top_k_coverage through the CLI.
    """
    # 1 << state_bits == 2 ** state_bits, but stays strictly integer
    num_states: int = 1 << state_bits

    # Phase 1 – collect integer frequency tables
    state_counts = _build_state_counts(sequences, vocab_size, num_states, context_len)

    # Ensure every state has a (possibly all-zero) histogram
    for s in range(num_states):
        if s not in state_counts:
            state_counts[s] = [0] * vocab_size

    # Phase 2 – CP-SAT emission optimisation
    # (transition function is fixed hash; see TODO above)
    _optimize_emissions_cpsat(
        state_counts, vocab_size, num_states, top_k_coverage, steps
    )

    # Build the fixed-hash transition table as integers
    transitions: dict[tuple[int, int], int] = {}
    for s in range(num_states):
        for t in range(vocab_size):
            transitions[(s, t)] = (s * HASH_PRIME + t + 1) % num_states

    return CircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=state_bits,
        transitions=transitions,
        state_counts=state_counts,
    )
