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

TODO: Jointly learn transitions and state assignments via CP-SAT (current
      transition CP-SAT pass learns observed delta(s,t) under fixed
      assignments within each pass; refinement re-assigns between passes).
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


def _build_transition_counts(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    context_len: int,
) -> dict[tuple[int, int], list[int]]:
    """Collect integer successor-state histograms for observed (state, token).

    States are the hashed-context assignments from :func:`_compute_state`.
    For each consumed token ``seq[pos]`` we add an observation:

        ``(src_state, seq[pos]) -> dst_state``

    where ``dst_state`` is the hashed state of the context ending at ``pos``,
    and ``src_state`` is:
      - ``0`` for ``pos == 0`` (initial runtime state), or
      - the hashed state of the context ending at ``pos - 1``.

    Returns:
        ``transition_counts[(state, token)] = [count_next_state_0, ...]``
        histograms of length ``num_states``.
    """
    transition_counts: dict[tuple[int, int], list[int]] = {}

    for seq in sequences:
        if not seq:
            continue

        hashed_states: list[int] = []
        for pos in range(len(seq)):
            start = max(0, pos - context_len + 1)
            ctx = seq[start : pos + 1]
            hashed_states.append(_compute_state(ctx, num_states))

        for pos, tok in enumerate(seq):
            if not (0 <= tok < vocab_size):
                continue
            src_state = 0 if pos == 0 else hashed_states[pos - 1]
            dst_state = hashed_states[pos]
            key = (src_state, tok)
            if key not in transition_counts:
                transition_counts[key] = [0] * num_states
            transition_counts[key][dst_state] += 1

    return transition_counts


def _next_state_with_fallback(
    state: int,
    token: int,
    num_states: int,
    learned_transitions: dict[tuple[int, int], int],
) -> int:
    """Apply a sparse learned transition table with hash fallback."""
    key = (state, token)
    if key in learned_transitions:
        return learned_transitions[key]
    return (state * HASH_PRIME + token + 1) % num_states


def _collect_runtime_counts(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    learned_transitions: dict[tuple[int, int], int],
) -> tuple[dict[int, list[int]], dict[tuple[int, int], list[int]]]:
    """Collect emission/transition counts from runtime states under ``delta``.

    This is the E-like step for iterative refinement: re-run the FSM using the
    current transition function (sparse learned pairs + hash fallback), then
    rebuild the integer count tables used by the CP-SAT M-step.
    """
    state_counts: dict[int, list[int]] = {}
    transition_counts: dict[tuple[int, int], list[int]] = {}

    for seq in sequences:
        state = 0
        for pos, tok in enumerate(seq):
            next_state = _next_state_with_fallback(
                state, tok, num_states, learned_transitions
            )

            if 0 <= tok < vocab_size:
                key = (state, tok)
                if key not in transition_counts:
                    transition_counts[key] = [0] * num_states
                transition_counts[key][next_state] += 1

            if pos + 1 < len(seq):
                next_tok = seq[pos + 1]
                if 0 <= next_tok < vocab_size:
                    if next_state not in state_counts:
                        state_counts[next_state] = [0] * vocab_size
                    state_counts[next_state][next_tok] += 1

            state = next_state

    return state_counts, transition_counts


def _optimize_transitions_cpsat(
    transition_counts: dict[tuple[int, int], list[int]],
    num_states: int,
    time_limit_seconds: int,
) -> dict[tuple[int, int], int]:
    """Learn observed transition outputs ``delta(s,t)`` using CP-SAT.

    For each observed ``(state, token)`` pair, chooses one integer next-state
    ``delta(s,t)`` that maximises empirical agreement with the successor-state
    histogram in ``transition_counts``.

    Returns:
        Sparse mapping ``{(state, token): next_state}`` for observed pairs.
        Unobserved pairs should be filled by the caller (e.g. hash fallback).
    """
    active_pairs = sorted(
        key for key, counts in transition_counts.items() if any(c > 0 for c in counts)
    )
    if not active_pairs:
        return {}

    model = cp_model.CpModel()

    delta_vars: dict[tuple[int, int], cp_model.IntVar] = {
        key: model.new_int_var(0, num_states - 1, f"delta_{key[0]}_{key[1]}")
        for key in active_pairs
    }

    gain_vars: list[cp_model.IntVar] = []
    for key in active_pairs:
        counts = transition_counts[key]
        max_count = max(counts) if any(c > 0 for c in counts) else 0
        if max_count == 0:
            continue
        gain = model.new_int_var(0, max_count, f"gain_delta_{key[0]}_{key[1]}")
        model.add_element(delta_vars[key], counts, gain)
        gain_vars.append(gain)

    if gain_vars:
        model.maximize(sum(gain_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = False
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        result: dict[tuple[int, int], int] = {}
        for key in active_pairs:
            counts = transition_counts[key]
            result[key] = max(range(num_states), key=counts.__getitem__)
        return result

    return {key: solver.value(delta_vars[key]) for key in active_pairs}


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

    active_states = [
        s for s in range(num_states)
        if s in state_counts and any(c > 0 for c in state_counts[s])
    ]
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


def _resolve_transition_emission_budgets(
    steps: int,
    transition_steps: int | None,
    emission_steps: int | None,
) -> tuple[int, int]:
    """Resolve total transition/emission budgets, keeping ``steps`` fallback."""
    if transition_steps is None and emission_steps is None:
        transition_seconds = steps // 2
        emission_seconds = steps - transition_seconds
        return transition_seconds, emission_seconds

    if transition_steps is None or emission_steps is None:
        raise ValueError(
            "transition_steps and emission_steps must be provided together"
        )

    return transition_steps, emission_steps


def _split_budget_across_passes(total_seconds: int, num_passes: int) -> list[int]:
    """Distribute an integer time budget across refinement passes."""
    if num_passes <= 0:
        return []
    base = total_seconds // num_passes
    rem = total_seconds - (base * num_passes)
    return [base + (1 if i < rem else 0) for i in range(num_passes)]


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
    transition_steps: int | None = None,
    emission_steps: int | None = None,
    refinement_rounds: int = 1,
) -> CircuitLM:
    """Train a CircuitLM from integer token sequences using OR-Tools CP-SAT.

    Args:
        sequences:       List of integer token-ID sequences.
        vocab_size:      Number of distinct token IDs.
        state_bits:      State width in bits; num_states = 1 << state_bits.
        steps:           Legacy total CP-SAT wall-clock budget (integer
                         seconds). Used only when explicit per-phase budgets
                         are not provided.
        context_len:     Number of preceding tokens used to derive the state.
        top_k_coverage:  How many top tokens must be covered by the emission
                         function (makes the CP-SAT problem non-trivial).
        transition_steps: Total transition-optimisation budget across all
                         passes (initial pass + refinement passes). Must be
                         provided together with *emission_steps*.
        emission_steps:  Total emission-optimisation budget across all passes.
        refinement_rounds: Number of additional EM-like state-assignment
                         refinement rounds after the initial hashed pass.

    Returns:
        A trained :class:`~circuit_lm.circuits.CircuitLM` instance.

    TODO: Jointly learn transitions and state assignments (current transition
          CP-SAT pass still optimises under fixed assignments within each pass,
          but assignments are now re-derived between passes).
    """
    # 1 << state_bits == 2 ** state_bits, but stays strictly integer
    num_states: int = 1 << state_bits

    if refinement_rounds < 0:
        raise ValueError("refinement_rounds must be >= 0")
    num_passes = 1 + refinement_rounds

    transition_budget_total, emission_budget_total = _resolve_transition_emission_budgets(
        steps,
        transition_steps,
        emission_steps,
    )
    transition_pass_budgets = _split_budget_across_passes(
        transition_budget_total, num_passes
    )
    emission_pass_budgets = _split_budget_across_passes(
        emission_budget_total, num_passes
    )

    # Phase 1 – collect integer frequency tables and transition observations
    state_counts = _build_state_counts(sequences, vocab_size, num_states, context_len)
    transition_counts = _build_transition_counts(
        sequences, vocab_size, num_states, context_len
    )

    learned_transitions: dict[tuple[int, int], int] = {}
    pred_tokens: dict[int, int] = {}

    for pass_idx in range(num_passes):
        learned_transitions = _optimize_transitions_cpsat(
            transition_counts, num_states, transition_pass_budgets[pass_idx]
        )
        pred_tokens = _optimize_emissions_cpsat(
            state_counts,
            vocab_size,
            num_states,
            top_k_coverage,
            emission_pass_budgets[pass_idx],
        )

        if pass_idx + 1 < num_passes:
            state_counts, transition_counts = _collect_runtime_counts(
                sequences,
                vocab_size,
                num_states,
                learned_transitions,
            )

    # Ensure every state has a (possibly all-zero) histogram for inference /
    # serialisation after optimisation (zero-count states should not satisfy
    # the coverage constraint during the solve).
    for s in range(num_states):
        if s not in state_counts:
            state_counts[s] = [0] * vocab_size

    # Build the transition table: learned observed pairs + hash fallback for
    # unseen pairs.
    transitions: dict[tuple[int, int], int] = {}
    for s in range(num_states):
        for t in range(vocab_size):
            transitions[(s, t)] = (s * HASH_PRIME + t + 1) % num_states
    transitions.update(learned_transitions)

    return CircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=state_bits,
        transitions=transitions,
        state_counts=state_counts,
        pred_tokens=pred_tokens,
    )
