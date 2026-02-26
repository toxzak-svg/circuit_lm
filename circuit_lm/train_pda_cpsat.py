"""Two-phase CP-SAT training for the Pushdown Automaton Circuit LM.

All arithmetic is integer-only.

Phase 1 – Stack Policy Learning (CP-SAT)
==========================================
Which tokens should PUSH to the stack, and which should POP?

CP-SAT problem
--------------
Given integer co-occurrence scores ``pair_score[(t1, t2)]`` = count of t1
appearing within distance 1..stack_depth of t2 in training data:

Variables
    ``is_push[t]``           ∈ {0,1}  for each token t
    ``is_pop[t]``            ∈ {0,1}  for each token t
    ``pair_active[t1, t2]``  ∈ {0,1}  for each top-scored pair

Constraints
    ``is_push[t] + is_pop[t] ≤ 1``           (mutually exclusive)
    ``pair_active[t1,t2] ≤ is_push[t1]``     (pair only active if t1 pushes)
    ``pair_active[t1,t2] ≤ is_pop[t2]``      (pair only active if t2 pops)
    ``Σ is_push ≤ max_push``                 (budget on push tokens)
    ``Σ is_pop  ≤ max_pop``                  (budget on pop tokens)

Objective (maximise)
    ``Σ_{(t1,t2)} pair_active[t1,t2] × pair_score[(t1,t2)]``

This incentivises choosing push/pop tokens that genuinely co-occur within
the stack depth window – the defining property of a useful bracket pair.
The budget constraints prevent degenerate solutions.

Phase 2 – Config Emission Optimisation (CP-SAT)
================================================
Identical to the FSM emission optimisation in ``train_cpsat.py``, but
now operating over the extended configuration space
``(state, stack_top)`` produced by the Phase-1 stack policy.

See ``train_cpsat._optimize_emissions_cpsat`` for the problem statement.

TODO: Joint optimisation of Phase 1 and Phase 2 in a single CP-SAT model.
TODO: Learn per-(state, token, stack_top) stack operations instead of the
      current token-only assignment.
TODO: Multi-level stack policy (PUSH vs PUSH+REPLACE vs POP).
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from circuit_lm.circuits import HASH_PRIME
from circuit_lm.pda import STACK_EMPTY, PDACircuitLM
from circuit_lm.train_cpsat import (
    _build_transition_counts,
    _next_state_with_fallback,
    _optimize_transitions_cpsat,
    _split_budget_across_passes,
)


# ---------------------------------------------------------------------------
# Helpers shared by both phases
# ---------------------------------------------------------------------------


def _compute_state(context: list[int], num_states: int) -> int:
    """Rolling polynomial hash of a token context window → integer state."""
    h = 0
    for tok in context:
        h = (h * HASH_PRIME + tok + 1) % num_states
    return h


# ---------------------------------------------------------------------------
# Phase 1 helpers – co-occurrence statistics
# ---------------------------------------------------------------------------


def _collect_pair_scores(
    sequences: list[list[int]],
    vocab_size: int,
    max_dist: int,
) -> dict[tuple[int, int], int]:
    """Count co-occurrences of (t1, t2) within distance 1..max_dist.

    Returns a sparse dict mapping ``(t1, t2) → total_count`` (integers only).
    Self-pairs (t1 == t2) are excluded so that a token cannot be both
    its own push and its own pop.

    Complexity: O(T × max_dist) where T = total tokens in training.
    """
    counts: dict[tuple[int, int], int] = {}
    for seq in sequences:
        n = len(seq)
        for pos in range(n):
            t1 = seq[pos]
            if not (0 <= t1 < vocab_size):
                continue
            for d in range(1, min(max_dist + 1, n - pos)):
                t2 = seq[pos + d]
                if 0 <= t2 < vocab_size and t2 != t1:
                    key = (t1, t2)
                    counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Phase 1 – Stack policy learning
# ---------------------------------------------------------------------------


def _learn_push_pop_cpsat(
    pair_scores: dict[tuple[int, int], int],
    vocab_size: int,
    max_push: int,
    max_pop: int,
    top_k_pairs: int,
    time_limit_seconds: int,
) -> tuple[frozenset[int], frozenset[int]]:
    """CP-SAT Phase 1: learn which tokens trigger PUSH and which trigger POP.

    Returns:
        ``(push_tokens, pop_tokens)`` as frozensets of integer token IDs.

    Falls back to empty sets (no-stack behaviour, same as plain FSM) if the
    solver cannot find a feasible solution within *time_limit_seconds*.

    Args:
        pair_scores:       Sparse dict of integer co-occurrence scores.
        vocab_size:        Number of token IDs.
        max_push:          Maximum number of PUSH tokens allowed.
        max_pop:           Maximum number of POP tokens allowed.
        top_k_pairs:       Consider only this many highest-scored (t1,t2) pairs.
        time_limit_seconds: Integer solver time limit in seconds.
    """
    # Nothing to optimise
    if not pair_scores:
        return frozenset(), frozenset()

    model = cp_model.CpModel()

    # Decision variables ---------------------------------------------------
    is_push = [model.new_bool_var(f"push_{t}") for t in range(vocab_size)]
    is_pop  = [model.new_bool_var(f"pop_{t}")  for t in range(vocab_size)]

    # Mutual exclusion: a token cannot be both push and pop
    for t in range(vocab_size):
        model.add(is_push[t] + is_pop[t] <= 1)

    # Budget constraints
    if max_push < vocab_size:
        model.add(sum(is_push) <= max_push)
    if max_pop < vocab_size:
        model.add(sum(is_pop) <= max_pop)

    # Select top-K pairs by score -----------------------------------------
    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_pairs = ranked[:top_k_pairs]

    # Pair-activation variables and objective ------------------------------
    obj_terms: list = []
    for (t1, t2), score in top_pairs:
        b = model.new_bool_var(f"pair_{t1}_{t2}")
        # b can be 1 only if t1 is a push token AND t2 is a pop token
        model.add(b <= is_push[t1])
        model.add(b <= is_pop[t2])
        obj_terms.append(score * b)

    if obj_terms:
        model.maximize(sum(obj_terms))

    # Solve ----------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = False
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return frozenset(), frozenset()

    push_set = frozenset(t for t in range(vocab_size) if solver.value(is_push[t]))
    pop_set  = frozenset(t for t in range(vocab_size) if solver.value(is_pop[t]))
    return push_set, pop_set


# ---------------------------------------------------------------------------
# Phase 2 helpers – PDA simulation & statistics
# ---------------------------------------------------------------------------


def _simulate_and_collect(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    stack_depth: int,
    push_tokens: frozenset[int],
    pop_tokens: frozenset[int],
    context_len: int,
) -> dict[tuple[int, int], list[int]]:
    """Simulate the PDA on training sequences and collect config counts.

    For each position *i* in each sequence the current configuration is
    ``(state, stack_top)`` and the observation is ``seq[i+1]``.

    Returns:
        ``config_counts[(state, stack_top)] = list[int]`` of length
        *vocab_size*.  Only observed configs appear as keys.
    """
    config_counts: dict[tuple[int, int], list[int]] = {}

    for seq in sequences:
        state = 0
        stack: list[int] = []

        for pos in range(len(seq) - 1):
            tok      = seq[pos]
            next_tok = seq[pos + 1]

            # Derive state from context window (rolling hash)
            start   = max(0, pos - context_len + 1)
            ctx     = seq[start : pos + 1]
            state   = _compute_state(ctx, num_states)

            # Update stack before recording config so that config keys match
            # the post-update stack top used by evaluate_pda (which steps
            # first and then calls predict_token on the resulting stack).
            if tok in push_tokens and len(stack) < stack_depth:
                stack.append(tok)
            elif tok in pop_tokens and stack:
                stack.pop()

            stack_top = stack[-1] if stack else STACK_EMPTY
            config    = (state, stack_top)

            if config not in config_counts:
                config_counts[config] = [0] * vocab_size
            if 0 <= next_tok < vocab_size:
                config_counts[config][next_tok] += 1

    return config_counts


def _simulate_and_collect_runtime(
    sequences: list[list[int]],
    vocab_size: int,
    num_states: int,
    stack_depth: int,
    push_tokens: frozenset[int],
    pop_tokens: frozenset[int],
    learned_transitions: dict[tuple[int, int], int],
) -> tuple[dict[tuple[int, int], list[int]], dict[tuple[int, int], list[int]]]:
    """Re-simulate the PDA under learned transitions and rebuild count tables.

    Returns:
        ``(config_counts, transition_counts)`` where:
          - ``config_counts[(state, stack_top)]`` stores next-token histograms
          - ``transition_counts[(src_state, token)]`` stores successor-state
            histograms used by transition re-optimisation

    The state component is updated using the current sparse transition table
    with hash fallback; the stack policy (push/pop token sets) stays fixed.
    """
    config_counts: dict[tuple[int, int], list[int]] = {}
    transition_counts: dict[tuple[int, int], list[int]] = {}

    for seq in sequences:
        state = 0
        stack: list[int] = []

        for pos, tok in enumerate(seq):
            next_state = _next_state_with_fallback(
                state, tok, num_states, learned_transitions
            )

            if 0 <= tok < vocab_size:
                t_key = (state, tok)
                if t_key not in transition_counts:
                    transition_counts[t_key] = [0] * num_states
                transition_counts[t_key][next_state] += 1

            # Update stack before recording config so that config keys match
            # the post-update stack top used by evaluate_pda.
            if tok in push_tokens and len(stack) < stack_depth:
                stack.append(tok)
            elif tok in pop_tokens and stack:
                stack.pop()

            if pos + 1 < len(seq):
                next_tok = seq[pos + 1]
                stack_top = stack[-1] if stack else STACK_EMPTY
                cfg = (next_state, stack_top)
                if cfg not in config_counts:
                    config_counts[cfg] = [0] * vocab_size
                if 0 <= next_tok < vocab_size:
                    config_counts[cfg][next_tok] += 1

            state = next_state

    return config_counts, transition_counts


# ---------------------------------------------------------------------------
# Phase 2 – Config emission optimisation (same structure as FSM phase 2)
# ---------------------------------------------------------------------------


def _optimize_config_emissions_cpsat(
    config_counts: dict[tuple[int, int], list[int]],
    vocab_size: int,
    top_k_coverage: int,
    time_limit_seconds: int,
) -> dict[tuple[int, int], int]:
    """CP-SAT Phase 2: choose one prediction token per (state, stack_top) config.

    Maximises total correct predictions subject to a coverage constraint:
    each of the top-K globally frequent tokens must be predicted by at least
    one config.  Analogous to ``train_cpsat._optimize_emissions_cpsat`` but
    operating over the richer PDA configuration space.

    Returns:
        ``{(state, stack_top): predicted_token}`` dict of integers.
    """
    active_configs = sorted(
        cfg for cfg, counts in config_counts.items() if any(c > 0 for c in counts)
    )
    if not active_configs:
        return {}

    model = cp_model.CpModel()

    pred_tok: dict[tuple[int, int], cp_model.IntVar] = {
        cfg: model.new_int_var(0, vocab_size - 1, f"pred_{cfg[0]}_{cfg[1]}")
        for cfg in active_configs
    }

    # Objective: maximise total correct-prediction count ------------------
    gain_vars: list[cp_model.IntVar] = []
    for cfg in active_configs:
        counts = config_counts[cfg]
        max_cnt = max(counts) if any(c > 0 for c in counts) else 0
        if max_cnt == 0:
            continue
        gain = model.new_int_var(0, max_cnt, f"gain_{cfg[0]}_{cfg[1]}")
        model.add_element(pred_tok[cfg], counts, gain)
        gain_vars.append(gain)

    if gain_vars:
        model.maximize(sum(gain_vars))

    # Coverage constraint -------------------------------------------------
    global_counts: list[int] = [0] * vocab_size
    for counts in config_counts.values():
        for t, c in enumerate(counts):
            global_counts[t] += c

    ranked_toks = sorted(range(vocab_size), key=global_counts.__getitem__, reverse=True)
    k = min(top_k_coverage, len(active_configs))
    top_tokens = [t for t in ranked_toks[:k] if global_counts[t] > 0]

    for t in top_tokens:
        covers: list[cp_model.BoolVar] = []
        for cfg in active_configs:
            b = model.new_bool_var(f"cov_{cfg[0]}_{cfg[1]}_{t}")
            model.add(pred_tok[cfg] == t).only_enforce_if(b)
            covers.append(b)
        model.add(sum(covers) >= 1)

    # Solve ---------------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = False
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Fallback: argmax per config
        result: dict[tuple[int, int], int] = {}
        for cfg in active_configs:
            counts = config_counts[cfg]
            result[cfg] = max(range(vocab_size), key=counts.__getitem__)
        return result

    return {cfg: solver.value(pred_tok[cfg]) for cfg in active_configs}


def _resolve_pda_budgets(
    steps: int,
    stack_steps: int | None,
    transition_steps: int | None,
    emission_steps: int | None,
) -> tuple[int, int, int]:
    """Resolve total CP-SAT budgets for PDA phase1/transition/emission."""
    if (
        stack_steps is None
        and transition_steps is None
        and emission_steps is None
    ):
        phase1_seconds = steps // 2
        phase2_seconds = steps - phase1_seconds
        transition_seconds = phase2_seconds // 2
        emission_seconds = phase2_seconds - transition_seconds
        return phase1_seconds, transition_seconds, emission_seconds

    if (
        stack_steps is None
        or transition_steps is None
        or emission_steps is None
    ):
        raise ValueError(
            "stack_steps, transition_steps, and emission_steps must be provided together"
        )

    return stack_steps, transition_steps, emission_steps


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


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
) -> PDACircuitLM:
    """Train a PDACircuitLM from integer token sequences using OR-Tools CP-SAT.

    Two-phase CP-SAT training:

    Phase 1 (stack_steps seconds)
        Learns which tokens trigger PUSH and POP by maximising the total
        co-occurrence weight of chosen push/pop pairs within distance
        1..stack_depth.  This is a set-selection CP-SAT problem.

    Phase 2 (transition_steps + emission_steps total, distributed across
    refinement passes)
        Simulates the PDA on training data using the Phase-1 stack policy,
        collects integer (state, stack_top) → next-token frequency tables,
        then solves the emission-prediction CP-SAT problem subject to a
        token-coverage constraint.

    Args:
        sequences:       List of integer token-ID sequences.
        vocab_size:      Number of distinct token IDs.
        state_bits:      FSM state width in bits; num_states = 1 << state_bits.
        stack_depth:     Maximum integer stack depth (0 disables the stack).
        steps:           Legacy total CP-SAT wall-clock budget in integer
                         seconds. Used only when explicit phase budgets are
                         not provided.
        context_len:     Context window for FSM state hashing.
        max_push:        Maximum number of PUSH tokens (Phase 1 budget).
        max_pop:         Maximum number of POP tokens (Phase 1 budget).
        top_k_pairs:     How many top co-occurrence pairs to consider in Phase 1.
        top_k_coverage:  Tokens that must be covered in Phase 2.
        stack_steps:     Total Phase-1 (stack-policy) CP-SAT budget.
        transition_steps: Total transition CP-SAT budget across the initial
                         pass plus refinement passes.
        emission_steps:  Total config-emission CP-SAT budget across the
                         initial pass plus refinement passes.
        refinement_rounds: Number of additional EM-like transition/emission
                         re-estimation rounds after the initial hashed pass.

    Returns:
        A trained :class:`~circuit_lm.pda.PDACircuitLM` instance.

    TODO: Joint phase optimisation in a single CP-SAT model.
    """
    num_states: int = 1 << state_bits
    if refinement_rounds < 0:
        raise ValueError("refinement_rounds must be >= 0")
    num_passes = 1 + refinement_rounds

    phase1_seconds, transition_budget_total, emission_budget_total = _resolve_pda_budgets(
        steps,
        stack_steps,
        transition_steps,
        emission_steps,
    )
    transition_pass_budgets = _split_budget_across_passes(
        transition_budget_total, num_passes
    )
    emission_pass_budgets = _split_budget_across_passes(
        emission_budget_total, num_passes
    )

    # ------------------------------------------------------------------
    # Phase 1: stack policy
    # ------------------------------------------------------------------
    push_tokens: frozenset[int] = frozenset()
    pop_tokens:  frozenset[int] = frozenset()

    if stack_depth > 0 and phase1_seconds > 0:
        pair_scores = _collect_pair_scores(sequences, vocab_size, stack_depth)
        push_tokens, pop_tokens = _learn_push_pop_cpsat(
            pair_scores,
            vocab_size,
            max_push,
            max_pop,
            top_k_pairs,
            phase1_seconds,
        )

    # ------------------------------------------------------------------
    # Phase 2: transition learning + config emissions (+ refinement)
    # ------------------------------------------------------------------
    transition_counts = _build_transition_counts(
        sequences, vocab_size, num_states, context_len
    )

    config_counts = _simulate_and_collect(
        sequences,
        vocab_size,
        num_states,
        stack_depth,
        push_tokens,
        pop_tokens,
        context_len,
    )

    learned_transitions: dict[tuple[int, int], int] = {}
    config_pred_tokens: dict[tuple[int, int], int] = {}

    for pass_idx in range(num_passes):
        learned_transitions = _optimize_transitions_cpsat(
            transition_counts,
            num_states,
            transition_pass_budgets[pass_idx],
        )
        config_pred_tokens = _optimize_config_emissions_cpsat(
            config_counts,
            vocab_size,
            top_k_coverage,
            emission_pass_budgets[pass_idx],
        )

        if pass_idx + 1 < num_passes:
            config_counts, transition_counts = _simulate_and_collect_runtime(
                sequences,
                vocab_size,
                num_states,
                stack_depth,
                push_tokens,
                pop_tokens,
                learned_transitions,
            )

    # Fill in all (state, STACK_EMPTY) configs after optimisation so that
    # unseen zero-count configs cannot satisfy the coverage constraint.
    for s in range(num_states):
        if (s, STACK_EMPTY) not in config_counts:
            config_counts[(s, STACK_EMPTY)] = [0] * vocab_size

    # Build transition table: learned observed pairs + hash fallback.
    transitions: dict[tuple[int, int], int] = {}
    for s in range(num_states):
        for t in range(vocab_size):
            transitions[(s, t)] = (s * HASH_PRIME + t + 1) % num_states
    transitions.update(learned_transitions)

    # Expand per-token push/pop sets to full (state, token, stack_top) config triples.
    # _simulate_and_collect and _simulate_and_collect_runtime are internal functions
    # that still receive frozenset[int] and are left unchanged.
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
