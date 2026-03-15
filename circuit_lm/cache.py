"""LRU Cache for FSM state transitions.

All fields and computations are strictly integer; no floats anywhere.

This module provides caching for common state transition paths to speed up
inference on CPU. The cache stores (state, token) -> next_state mappings
that are accessed frequently, avoiding repeated hash computations.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generic, TypeVar


T = TypeVar('T')


@dataclass
class LRUCache(Generic[T]):
    """Least Recently Used cache with integer-only operations.
    
    Attributes:
        capacity: Maximum number of entries (must be > 0)
        _cache: OrderedDict mapping keys to values
        hits: Number of cache hits (integer)
        misses: Number of cache misses (integer)
    
    Note: Uses OrderedDict for O(1) LRU operations on Python 3.7+.
    """
    
    capacity: int = 1024
    _cache: OrderedDict = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0
    
    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
    
    def get(self, key: tuple[int, int]) -> T | None:
        """Get value for key, updating LRU position.
        
        Returns:
            Cached value if found, None otherwise.
        """
        if key in self._cache:
            self.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None
    
    def put(self, key: tuple[int, int], value: T) -> None:
        """Put key-value pair into cache, evicting LRU if at capacity.
        
        If key already exists, updates value and moves to end.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)  # Remove oldest (first) item
        
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear all entries and reset stats."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
    
    def hit_rate(self) -> int:
        """Return cache hit rate in basis-points (1/100 of percent).
        
        Returns 0 if no lookups have occurred.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0
        # Return as basis-points: (hits / total) * 10000
        return (self.hits * 10000) // total
    
    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)


@dataclass
class TransitionCache:
    """Specialized cache for FSM state transitions.
    
    Provides caching for (state, token) -> next_state lookups,
    which are the most common operation during inference.
    
    Attributes:
        capacity: Maximum number of cached transitions
        cache: The underlying LRU cache
    """
    
    capacity: int = 4096
    cache: LRUCache[int] = field(default_factory=lambda: LRUCache(capacity=4096))
    
    def __post_init__(self):
        self.cache = LRUCache(capacity=self.capacity)
    
    def get_next_state(self, state: int, token: int, fallback_fn) -> int:
        """Get next state with caching.
        
        Args:
            state: Current FSM state (integer)
            token: Current input token (integer)
            fallback_fn: Function to compute next state on cache miss
        
        Returns:
            Next state (integer)
        """
        key = (state, token)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # Compute and cache
        next_state = fallback_fn(state, token)
        self.cache.put(key, next_state)
        return next_state
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "hit_rate_bp": self.cache.hit_rate(),
        }
    
    def clear(self) -> None:
        """Clear cache and reset stats."""
        self.cache.clear()


@dataclass
class ConfigCache:
    """Specialized cache for PDA configurations.
    
    Caches (state, stack_top, token) -> (next_state, stack_operation) tuples.
    
    Attributes:
        capacity: Maximum number of cached configurations
        cache: The underlying LRU cache
    """
    
    capacity: int = 2048
    cache: LRUCache[tuple[int, int]] = field(default_factory=lambda: LRUCache(capacity=2048))
    
    def __post_init__(self):
        self.cache = LRUCache(capacity=self.capacity)
    
    def get_config(
        self, 
        state: int, 
        stack_top: int, 
        token: int,
        fallback_fn
    ) -> tuple[int, int]:
        """Get next config with caching.
        
        Args:
            state: Current FSM state (integer)
            stack_top: Current stack top (integer, -1 for empty)
            token: Current input token (integer)
            fallback_fn: Function to compute (next_state, stack_op) on cache miss
        
        Returns:
            Tuple of (next_state, stack_operation)
        """
        key = (state, stack_top, token)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        result = fallback_fn(state, stack_top, token)
        self.cache.put(key, result)
        return result
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.cache.hits,
            "misses": self.cache.misses,
            "hit_rate_bp": self.cache.hit_rate(),
        }
    
    def clear(self) -> None:
        """Clear cache and reset stats."""
        self.cache.clear()


# ---------------------------------------------------------------------------
# Integration with CircuitLM
# ---------------------------------------------------------------------------


def add_transition_cache_to_circuit(circuit, capacity: int = 4096):
    """Add transition cache to an existing CircuitLM.
    
    This monkey-patches the circuit's next_state method to use caching.
    
    Args:
        circuit: A CircuitLM or PDACircuitLM instance
        capacity: Cache size (default 4096)
    
    Returns:
        The same circuit with cache attached (modified in place)
    """
    if hasattr(circuit, '_transition_cache'):
        # Already has cache
        return circuit
    
    # Create cache
    cache = TransitionCache(capacity=capacity)
    
    # Store original method
    original_next_state = circuit.next_state
    
    def cached_next_state(state: int, token: int) -> int:
        return cache.get_next_state(state, token, original_next_state)
    
    # Replace method
    circuit.next_state = cached_next_state
    
    # Attach cache for stats
    circuit._transition_cache = cache
    
    return circuit


def add_config_cache_to_pda(pda, capacity: int = 2048):
    """Add config cache to an existing PDACircuitLM.
    
    This monkey-patches the PDA's step method to use caching.
    
    Args:
        pda: A PDACircuitLM instance
        capacity: Cache size (default 2048)
    
    Returns:
        The same PDA with cache attached (modified in place)
    """
    if hasattr(pda, '_config_cache'):
        # Already has cache
        return pda
    
    # Create cache
    cache = ConfigCache(capacity=capacity)
    
    # Store original methods
    original_step = pda.step
    original_next_state = pda.next_state
    original_stack_op = pda.stack_op
    
    def cached_step(state: int, stack: list[int], token: int):
        stack_top = stack[-1] if stack else -1
        
        def fallback(s: int, st: int, t: int):
            op = original_stack_op(s, t, st)
            ns = original_next_state(s, t)
            return ns, op
        
        next_state, stack_op = cache.get_config(state, stack_top, token, fallback)
        
        new_stack = list(stack)
        if stack_op == 1 and len(new_stack) < pda.stack_depth:  # PUSH
            new_stack.append(token)
        elif stack_op == 2 and new_stack:  # POP
            new_stack.pop()
        
        return next_state, new_stack
    
    # Replace method
    pda.step = cached_step
    
    # Attach cache for stats
    pda._config_cache = cache
    
    return pda


def get_cache_stats(circuit) -> dict | None:
    """Get cache statistics from a circuit or PDA.
    
    Returns None if no cache is attached.
    """
    if hasattr(circuit, '_transition_cache'):
        return circuit._transition_cache.get_stats()
    if hasattr(circuit, '_config_cache'):
        return circuit._config_cache.get_stats()
    return None


# ---------------------------------------------------------------------------
# Batch operations for SIMD-like performance
# ---------------------------------------------------------------------------


def batch_get_next_states(
    cache: TransitionCache,
    states: list[int],
    tokens: list[int],
    fallback_fn,
) -> list[int]:
    """Get multiple next states efficiently.
    
    This is designed for CPU-friendly batch processing without SIMD,
    but organizes work to minimize cache misses.
    
    Args:
        cache: TransitionCache instance
        states: List of current states
        tokens: List of input tokens (same length as states)
        fallback_fn: Function (state, token) -> next_state
    
    Returns:
        List of next states
    """
    if len(states) != len(tokens):
        raise ValueError("states and tokens must have same length")
    
    results = []
    for s, t in zip(states, tokens):
        results.append(cache.get_next_state(s, t, fallback_fn))
    
    return results