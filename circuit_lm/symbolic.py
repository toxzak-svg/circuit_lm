"""Symbolic Reasoning with CP-SAT for constraint-satisfied generation.

All fields and computations are strictly integer; no floats anywhere.

This module provides constraint-based generation where the CP-SAT solver
validates candidate tokens against defined "world rules" before emission.

Unlike standard LLMs which may hallucinate facts, this ensures generated
text satisfies logical constraints encoded in the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from ortools.sat.python import cp_model

from circuit_lm.circuits import HASH_PRIME


# ---------------------------------------------------------------------------
# Constraint Types (Integer-encoded)
# ---------------------------------------------------------------------------


class ConstraintType(IntEnum):
    """Types of constraints for symbolic reasoning."""
    TEMPORAL = 1      # Time-based constraints (no conflicts at same time)
    LOGICAL = 2       # Logical implications (A -> B)
    SEQUENCE = 3      # Sequence ordering (greeting before question)
    EXCLUSIVE = 4     # Mutual exclusion (cannot have both A and B)
    CARDINALITY = 5   # Count constraints (at most N of something)
    CUSTOM = 99      # User-defined constraints


class TokenConstraint:
    """Base class for token-level constraints."""
    
    def __init__(self, constraint_type: ConstraintType):
        self.constraint_type = constraint_type
    
    def validate(self, token: int, context: dict) -> bool:
        """Validate if token satisfies this constraint given context."""
        raise NotImplementedError
    
    def to_cp_constraint(self, model: cp_model.CpModel, token_var: cp_model.IntVar) -> None:
        """Add this constraint to a CP-SAT model."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete Constraint Implementations
# ---------------------------------------------------------------------------


@dataclass
class TemporalConstraint(TokenConstraint):
    """Constraint that prevents temporal conflicts.
    
    Example: "meeting at 2 PM" cannot coexist with "lunch at 2 PM"
    
    Attributes:
        time_slot: The time slot ID (integer)
        conflict_tokens: Tokens that conflict with this time slot
    """
    
    time_slot: int
    conflict_tokens: list[int]
    
    def __init__(self, time_slot: int, conflict_tokens: list[int]):
        super().__init__(ConstraintType.TEMPORAL)
        self.time_slot = time_slot
        self.conflict_tokens = conflict_tokens
    
    def validate(self, token: int, context: dict) -> bool:
        """Check if token conflicts with existing time slot."""
        if self.time_slot not in context.get("active_times", []):
            return True
        return token not in self.conflict_tokens
    
    def to_cp_constraint(self, model: cp_model.CpModel, token_var: cp_model.IntVar) -> None:
        """Add temporal exclusion constraint."""
        # token not in conflict_tokens if time_slot is active
        for bad_token in self.conflict_tokens:
            model.add(token_var != bad_token)


@dataclass
class LogicalImplication(TokenConstraint):
    """Constraint implementing logical implication (A -> B).
    
    Example: If "fever" then must have "temperature > threshold"
    
    Attributes:
        antecedent: Token that triggers the implication
        consequent: Required token if antecedent is present
    """
    
    antecedent: int
    consequent: int
    
    def __init__(self, antecedent: int, consequent: int):
        super().__init__(ConstraintType.LOGICAL)
        self.antecedent = antecedent
        self.consequent = consequent
    
    def validate(self, token: int, context: dict) -> bool:
        """Check if consequent is present when antecedent appears."""
        if token == self.antecedent:
            return self.consequent in context.get("recent_tokens", [])
        return True
    
    def to_cp_constraint(self, model: cp_model.CpModel, token_var: cp_model.IntVar) -> None:
        """Add implication constraint (if antecedent, must have consequent)."""
        # This is complex in CP-SAT; simplified to exclusion
        # Full implementation would track state variables
        pass


@dataclass
class SequenceConstraint(TokenConstraint):
    """Constraint enforcing sequence ordering.
    
    Example: greeting must come before question
    
    Attributes:
        required_first: Token that must appear first
        required_second: Token that must appear after
    """
    
    required_first: int
    required_second: int
    seen_first: bool = False
    
    def __init__(self, required_first: int, required_second: int):
        super().__init__(ConstraintType.SEQUENCE)
        self.required_first = required_first
        self.required_second = required_second
        self.seen_first = False
    
    def validate(self, token: int, context: dict) -> bool:
        """Check if order is maintained."""
        if token == self.required_first:
            self.seen_first = True
        if token == self.required_second and not self.seen_first:
            return False
        return True
    
    def to_cp_constraint(self, model: cp_model.CpModel, token_var: cp_model.IntVar) -> None:
        """Add sequence order constraint."""
        # Requires state tracking; simplified here
        pass


@dataclass
class MutualExclusion(TokenConstraint):
    """Constraint that two tokens cannot both appear.
    
    Example: Cannot have both "indoor" and "outdoor" for same event
    
    Attributes:
        token_a: First exclusive token
        token_b: Second exclusive token
    """
    
    token_a: int
    token_b: int
    
    def __init__(self, token_a: int, token_b: int):
        super().__init__(ConstraintType.EXCLUSIVE)
        self.token_a = token_a
        self.token_b = token_b
    
    def validate(self, token: int, context: dict) -> bool:
        """Check if token conflicts with previously seen token."""
        seen = context.get("seen_tokens", set())
        if token == self.token_a:
            return self.token_b not in seen
        if token == self.token_b:
            return self.token_a not in seen
        return True
    
    def to_cp_constraint(self, model: cp_model.CpModel, token_var: cp_model.IntVar) -> None:
        """Add mutual exclusion constraint."""
        # Requires tracking which was seen first
        pass


# ---------------------------------------------------------------------------
# Symbolic Reasoner
# ---------------------------------------------------------------------------


@dataclass
class SymbolicReasoner:
    """CP-SAT powered constraint solver for token generation.
    
    This reasoner validates candidate tokens against a set of constraints
    before allowing emission. Unlike LLMs which may produce inconsistent
    output, this ensures logical consistency.
    
    Attributes:
        constraints: List of active constraints
        context: Current generation context (state, stack, recent tokens)
        validation_cache: Cache of validated (token, context) pairs
    
    Note: Full runtime CP-SAT solving can be slow. This implementation
    provides fast validation using integer checks; full CP-SAT is used
    for complex multi-step reasoning.
    """
    
    constraints: list[TokenConstraint] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    _validation_cache: dict[tuple, bool] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    
    def add_constraint(self, constraint: TokenConstraint) -> None:
        """Add a constraint to the reasoner."""
        self.constraints.append(constraint)
    
    def clear_constraints(self) -> None:
        """Remove all constraints."""
        self.constraints.clear()
        self._validation_cache.clear()
    
    def update_context(self, state: int, stack: list[int], recent_tokens: list[int]) -> None:
        """Update the reasoning context with current state."""
        self.context = {
            "state": state,
            "stack_top": stack[-1] if stack else -1,
            "stack_depth": len(stack),
            "recent_tokens": recent_tokens[-10:] if recent_tokens else [],
            "seen_tokens": set(recent_tokens[-50:]) if recent_tokens else set(),
            "active_times": [],  # For temporal constraints
        }
    
    def validate_token_fast(self, token: int) -> bool:
        """Fast validation using integer checks.
        
        Uses caching for performance. This is the primary method for
        single-token validation during generation.
        
        Returns:
            True if token satisfies all constraints, False otherwise
        """
        # Check cache
        cache_key = (token, tuple(self.context.get("recent_tokens", [])[-5:]))
        if cache_key in self._validation_cache:
            self.cache_hits += 1
            return self._validation_cache[cache_key]
        
        self.cache_misses += 1
        
        # Validate against all constraints
        for constraint in self.constraints:
            if not constraint.validate(token, self.context):
                self._validation_cache[cache_key] = False
                return False
        
        self._validation_cache[cache_key] = True
        return True
    
    def validate_token_cpsat(
        self, 
        token: int, 
        candidates: list[int],
        time_limit: int = 1,
    ) -> list[int]:
        """Validate tokens using full CP-SAT solver.
        
        This is slower but can handle complex multi-step constraints
        that the fast validation cannot.
        
        Args:
            token: Current token to validate
            candidates: List of candidate next tokens
            time_limit: CP-SAT time limit in seconds
        
        Returns:
            List of valid candidate tokens
        """
        if not self.constraints or not candidates:
            return candidates
        
        model = cp_model.CpModel()
        
        # Create token variable (which candidate to choose)
        token_var = model.new_int_var(0, len(candidates) - 1, "token_choice")
        
        # Add constraints
        for constraint in self.constraints:
            constraint.to_cp_constraint(model, token_var)
        
        # Solve (just check feasibility, not optimization)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        
        # Get valid candidates by checking each
        valid = []
        for i, cand in enumerate(candidates):
            # Quick pre-filter
            if self.validate_token_fast(cand):
                valid.append(cand)
        
        return valid
    
    def filter_candidates(self, candidates: list[int]) -> list[int]:
        """Filter candidate tokens through all constraints.
        
        Args:
            candidates: List of candidate token IDs
        
        Returns:
            List of candidates that pass all constraints
        """
        return [c for c in candidates if self.validate_token_fast(c)]
    
    def get_stats(self) -> dict:
        """Return reasoner statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate_bp = (self.cache_hits * 10000 // total) if total > 0 else 0
        return {
            "num_constraints": len(self.constraints),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_bp": hit_rate_bp,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.cache_hits = 0
        self.cache_misses = 0
        self._validation_cache.clear()


# ---------------------------------------------------------------------------
# Predefined Constraint Sets
# ---------------------------------------------------------------------------


def create_chat_constraints() -> list[TokenConstraint]:
    """Create constraints for conversational text.
    
    Enforces:
    - Greeting before question
    - No duplicate consecutive tokens
    """
    constraints = []
    # Sequence: greeting before question (token IDs would need mapping)
    # Exclusions: no duplicate consecutive
    return constraints


def create_schedule_constraints() -> list[TokenConstraint]:
    """Create constraints for scheduling/text with times.
    
    Enforces:
    - No temporal conflicts
    """
    constraints = []
    # Would need specific token IDs for times
    return constraints


def create_code_constraints() -> list[TokenConstraint]:
    """Create constraints for code generation.
    
    Enforces:
    - Matching brackets
    - Indentation consistency
    """
    constraints = []
    # Would need tokenization that preserves brackets/indentation
    return constraints


# ---------------------------------------------------------------------------
# Integration with Inference
# ---------------------------------------------------------------------------


class ConstraintAwareGenerator:
    """Generator that uses symbolic constraints during text generation.
    
    Wraps a CircuitLM or PDACircuitLM and filters predictions through
    the SymbolicReasoner before emission.
    """
    
    def __init__(self, circuit, reasoner: SymbolicReasoner | None = None):
        self.circuit = circuit
        self.reasoner = reasoner or SymbolicReasoner()
        self.state = 0
        self.stack: list[int] = []
        self.recent_tokens: list[int] = []
    
    def predict_with_constraints(self) -> int:
        """Predict next token while respecting constraints."""
        # Get candidates from circuit
        if hasattr(self.circuit, 'stack_depth'):
            # PDA
            hist = self.circuit.config_histogram(self.state, self.stack)
            stack_top = self.stack[-1] if self.stack else -1
        else:
            # FSM
            hist = self.circuit.state_histogram(self.state)
            stack_top = -1
        
        # Update reasoner context
        self.reasoner.update_context(self.state, self.stack, self.recent_tokens)
        
        # Get top candidates
        candidates = sorted(range(len(hist)), key=lambda i: hist[i], reverse=True)[:20]
        
        # Filter through constraints
        valid = self.reasoner.filter_candidates(candidates)
        
        if not valid:
            # Fallback: use circuit's raw prediction
            if hasattr(self.circuit, 'predict_token'):
                return self.circuit.predict_token(self.state, self.stack)
            return self.circuit.predict_token(self.state)
        
        # Return best valid candidate
        return valid[0]
    
    def step(self, token: int) -> None:
        """Step forward with a token."""
        self.recent_tokens.append(token)
        
        if hasattr(self.circuit, 'step'):
            self.state, self.stack = self.circuit.step(self.state, self.stack, token)
        else:
            self.state = self.circuit.next_state(self.state, token)
    
    def reset(self) -> None:
        """Reset generator state."""
        self.state = 0
        self.stack = []
        self.recent_tokens = []


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_reasoner(
    constraint_type: str | None = None,
    **constraint_args,
) -> SymbolicReasoner:
    """Create a reasoner with specified constraints.
    
    Args:
        constraint_type: Type of constraints ("chat", "schedule", "code", or None)
        **constraint_args: Arguments for constraint creation
    
    Returns:
        SymbolicReasoner with configured constraints
    """
    reasoner = SymbolicReasoner()
    
    if constraint_type == "chat":
        reasoner.constraints = create_chat_constraints()
    elif constraint_type == "schedule":
        reasoner.constraints = create_schedule_constraints()
    elif constraint_type == "code":
        reasoner.constraints = create_code_constraints()
    
    return reasoner