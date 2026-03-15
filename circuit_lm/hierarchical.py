"""Hierarchical (Multi-Level) FSM for large state-space modeling.

All fields and computations are strictly integer; no floats anywhere.

Architecture
------------
A hierarchical circuit consists of:
1. Global Circuit: Tracks high-level topics/intents (fewer states)
2. Local Circuits: Per-topic FSMs that handle token-to-token generation

This allows millions of virtual states while only storing active paths.

Example:
  Global: topic "weather" -> local states 0-15
  Global: topic "sports"  -> local states 16-31
  Virtual state space = num_topics * states_per_topic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from circuit_lm.circuits import CircuitLM, HASH_PRIME


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Special topic IDs
TOPIC_DEFAULT: int = 0  # Default/initial topic
TOPIC_UNKNOWN: int = -1  # Unknown topic (should not occur in normal operation)


# ---------------------------------------------------------------------------
# Hierarchical Circuit
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalCircuitLM:
    """Two-level hierarchical FSM: global topic tracker + local context FSMs.
    
    The hierarchical approach dramatically increases the effective state space:
    - num_topics * states_per_local = virtual states
    - Only stores active (topic, state) pairs in local circuits
    
    Attributes:
        vocab_size:     Number of distinct token IDs.
        global_bits:    Number of bits for global topic state.
        local_bits:     Number of bits for local context state.
        global_circuit: High-level FSM tracking topics/intents.
        local_circuits: Dict mapping topic_id -> local CircuitLM.
        topic_transitions: Learned (topic, token) -> next_topic mappings.
        global_transitions: Dict for topic switching decisions.
    
    Virtual state calculation:
        effective_states = (2 ** global_bits) * (2 ** local_bits)
        For global_bits=8, local_bits=4: 256 * 16 = 4096 virtual states
    """
    
    vocab_size: int
    global_bits: int
    local_bits: int
    global_circuit: CircuitLM = field(default=None)
    local_circuits: dict[int, CircuitLM] = field(default_factory=dict)
    topic_transitions: dict[tuple[int, int], int] = field(default_factory=dict)
    global_transitions: dict[tuple[int, int], int] = field(default_factory=dict)
    
    # Derived properties
    num_topics: int = field(init=False)
    states_per_local: int = field(init=False)
    _topic_hash_fn: Callable[[int], int] = field(init=False)
    
    def __post_init__(self):
        """Initialize derived properties."""
        self.num_topics = 1 << self.global_bits  # 2 ** global_bits
        self.states_per_local = 1 << self.local_bits  # 2 ** local_bits
        
        # Initialize global circuit if not provided
        if self.global_circuit is None:
            self.global_circuit = CircuitLM(
                vocab_size=self.vocab_size,
                num_states=self.num_topics,
                state_bits=self.global_bits,
            )
        
        # Initialize topic hash function
        self._topic_hash_fn = lambda t: (t * HASH_PRIME + 1) % self.num_topics
    
    # ------------------------------------------------------------------------
    # Topic Operations
    # ------------------------------------------------------------------------
    
    def next_topic(self, current_topic: int, token: int) -> int:
        """Get next topic based on current topic and token.
        
        Uses learned transitions with hash fallback.
        """
        key = (current_topic, token)
        if key in self.topic_transitions:
            return self.topic_transitions[key]
        # Fallback: deterministic hash
        return self._topic_hash_fn(current_topic)
    
    def get_or_create_local_circuit(self, topic: int) -> CircuitLM:
        """Get existing local circuit for topic or create new one.
        
        Creates a fresh CircuitLM with local_bits states if not exists.
        """
        if topic not in self.local_circuits:
            self.local_circuits[topic] = CircuitLM(
                vocab_size=self.vocab_size,
                num_states=self.states_per_local,
                state_bits=self.local_bits,
            )
        return self.local_circuits[topic]
    
    # ------------------------------------------------------------------------
    # State Operations (Virtual State = topic * states_per_local + local_state)
    # ------------------------------------------------------------------------
    
    def next_virtual_state(
        self, 
        virtual_state: int, 
        token: int
    ) -> int:
        """Compute next virtual state (topic, local) from current.
        
        Args:
            virtual_state: Current virtual state encoding
            token: Current input token
        
        Returns:
            Next virtual state
        """
        # Decode virtual state
        topic = virtual_state // self.states_per_local
        local_state = virtual_state % self.states_per_local
        
        # Update topic
        next_topic = self.next_topic(topic, token)
        
        # Update local state
        local_circuit = self.get_or_create_local_circuit(next_topic)
        next_local_state = local_circuit.next_state(local_state, token)
        
        # Encode back to virtual state
        return next_topic * self.states_per_local + next_local_state
    
    def predict_token(self, virtual_state: int) -> int:
        """Predict next token from virtual state.
        
        Args:
            virtual_state: Current virtual state
        
        Returns:
            Predicted token ID
        """
        topic = virtual_state // self.states_per_local
        local_state = virtual_state % self.states_per_local
        
        local_circuit = self.get_or_create_local_circuit(topic)
        return local_circuit.predict_token(local_state)
    
    def state_histogram(self, virtual_state: int) -> list[int]:
        """Get integer count histogram for virtual state.
        
        Args:
            virtual_state: Current virtual state
        
        Returns:
            List of length vocab_size with integer counts
        """
        topic = virtual_state // self.states_per_local
        local_state = virtual_state % self.states_per_local
        
        local_circuit = self.get_or_create_local_circuit(topic)
        return local_circuit.state_histogram(local_state)
    
    # ------------------------------------------------------------------------
    # Sequence Operations
    # ------------------------------------------------------------------------
    
    def run(self, tokens: list[int], initial_topic: int = 0) -> list[int]:
        """Run hierarchical FSM over tokens.
        
        Args:
            tokens: List of integer token IDs
            initial_topic: Starting topic (default 0)
        
        Returns:
            List of virtual states (one per token)
        """
        states = []
        topic = initial_topic
        local_state = 0
        
        for tok in tokens:
            # Encode current virtual state
            virtual_state = topic * self.states_per_local + local_state
            states.append(virtual_state)
            
            # Update topic
            topic = self.next_topic(topic, tok)
            
            # Update local state
            local_circuit = self.get_or_create_local_circuit(topic)
            local_state = local_circuit.next_state(local_state, tok)
        
        return states
    
    def train_local_on_sequence(
        self, 
        tokens: list[int],
        initial_topic: int = 0,
    ) -> None:
        """Train local circuits on a token sequence.
        
        This updates the state histograms for each (topic, local_state) pair.
        """
        topic = initial_topic
        local_state = 0
        
        for i, tok in enumerate(tokens):
            # Get current config
            local_circuit = self.get_or_create_local_circuit(topic)
            
            # Update histogram: count next token from this config
            if local_state not in local_circuit.state_counts:
                local_circuit.state_counts[local_state] = [0] * self.vocab_size
            
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if 0 <= next_tok < self.vocab_size:
                    local_circuit.state_counts[local_state][next_tok] += 1
            
            # Move to next state
            topic = self.next_topic(topic, tok)
            local_state = local_circuit.next_state(local_state, tok)
    
    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------
    
    def num_virtual_states(self) -> int:
        """Return number of virtual states used so far."""
        return len(self.local_circuits) * self.states_per_local
    
    def memory_footprint(self) -> int:
        """Estimate memory usage in integer units.
        
        Counts: transitions + state_counts across all local circuits.
        """
        total = 0
        # Global circuit
        total += len(self.global_circuit.transitions)
        for s, counts in self.global_circuit.state_counts.items():
            total += len(counts)
        # Local circuits
        for circuit in self.local_circuits.values():
            total += len(circuit.transitions)
            for s, counts in circuit.state_counts.items():
                total += len(counts)
        return total
    
    # ------------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------------
    
    def to_dict(self) -> dict:
        """Serialize to dictionary (for JSON/MsgPack)."""
        return {
            "model_type": "hierarchical",
            "vocab_size": self.vocab_size,
            "global_bits": self.global_bits,
            "local_bits": self.local_bits,
            "global_circuit": {
                "vocab_size": self.global_circuit.vocab_size,
                "num_states": self.global_circuit.num_states,
                "state_bits": self.global_circuit.state_bits,
                "transitions": {
                    f"{k[0]},{k[1]}": v 
                    for k, v in self.global_circuit.transitions.items()
                },
                "state_counts": {
                    str(s): counts 
                    for s, counts in self.global_circuit.state_counts.items()
                },
                "pred_tokens": self.global_circuit.pred_tokens,
            },
            "local_circuits": {
                str(topic): {
                    "vocab_size": c.vocab_size,
                    "num_states": c.num_states,
                    "state_bits": c.state_bits,
                    "transitions": {
                        f"{k[0]},{k[1]}": v 
                        for k, v in c.transitions.items()
                    },
                    "state_counts": {
                        str(s): counts 
                        for s, counts in c.state_counts.items()
                    },
                    "pred_tokens": c.pred_tokens,
                }
                for topic, c in self.local_circuits.items()
            },
            "topic_transitions": {
                f"{k[0]},{k[1]}": v 
                for k, v in self.topic_transitions.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HierarchicalCircuitLM':
        """Deserialize from dictionary."""
        global_data = data["global_circuit"]
        global_circuit = CircuitLM(
            vocab_size=global_data["vocab_size"],
            num_states=global_data["num_states"],
            state_bits=global_data["state_bits"],
            transitions={
                (int(k.split(",")[0]), int(k.split(",")[1])): v
                for k, v in global_data.get("transitions", {}).items()
            },
            state_counts={
                int(s): counts 
                for s, counts in global_data.get("state_counts", {}).items()
            },
            pred_tokens={
                int(s): t for s, t in global_data.get("pred_tokens", {}).items()
            },
        )
        
        local_circuits = {}
        for topic_str, c_data in data.get("local_circuits", {}).items():
            topic = int(topic_str)
            local_circuits[topic] = CircuitLM(
                vocab_size=c_data["vocab_size"],
                num_states=c_data["num_states"],
                state_bits=c_data["state_bits"],
                transitions={
                    (int(k.split(",")[0]), int(k.split(",")[1])): v
                    for k, v in c_data.get("transitions", {}).items()
                },
                state_counts={
                    int(s): counts 
                    for s, counts in c_data.get("state_counts", {}).items()
                },
                pred_tokens={
                    int(s): t for s, t in c_data.get("pred_tokens", {}).items()
                },
            )
        
        return cls(
            vocab_size=data["vocab_size"],
            global_bits=data["global_bits"],
            local_bits=data["local_bits"],
            global_circuit=global_circuit,
            local_circuits=local_circuits,
            topic_transitions={
                (int(k.split(",")[0]), int(k.split(",")[1])): v
                for k, v in data.get("topic_transitions", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Hierarchical PDA (Stack Machine at Each Topic)
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalPDACircuitLM:
    """Hierarchical PDA: topics with independent stack machines.
    
    Each topic has its own PDA, allowing nested context within topics.
    """
    
    vocab_size: int
    global_bits: int
    local_bits: int
    stack_depth: int
    
    # Per-topic PDAs (stored as their configs/dicts)
    topic_pdas: dict[int, dict] = field(default_factory=dict)
    global_circuit: CircuitLM = field(default=None)
    
    # Derived
    num_topics: int = field(init=False)
    states_per_local: int = field(init=False)
    
    def __post_init__(self):
        self.num_topics = 1 << self.global_bits
        self.states_per_local = 1 << self.local_bits
        
        if self.global_circuit is None:
            self.global_circuit = CircuitLM(
                vocab_size=self.vocab_size,
                num_states=self.num_topics,
                state_bits=self.global_bits,
            )
    
    def get_or_create_pda(self, topic: int) -> dict:
        """Get or create PDA config for topic."""
        if topic not in self.topic_pdas:
            self.topic_pdas[topic] = {
                "transitions": {},
                "config_counts": {},
                "push_configs": set(),
                "pop_configs": set(),
            }
        return self.topic_pdas[topic]
    
    # Note: Full implementation would mirror HierarchicalCircuitLM
    # but with (state, stack_top) configurations instead of just state


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_hierarchical_circuit(
    vocab_size: int,
    global_bits: int = 8,
    local_bits: int = 4,
) -> HierarchicalCircuitLM:
    """Create a hierarchical circuit with specified parameters.
    
    Args:
        vocab_size: Size of vocabulary
        global_bits: Bits for topic state (default 8 = 256 topics)
        local_bits: Bits for local state per topic (default 4 = 16 states)
    
    Returns:
        HierarchicalCircuitLM instance
    """
    return HierarchicalCircuitLM(
        vocab_size=vocab_size,
        global_bits=global_bits,
        local_bits=local_bits,
    )


def create_hierarchical_pda(
    vocab_size: int,
    global_bits: int = 8,
    local_bits: int = 4,
    stack_depth: int = 4,
) -> HierarchicalPDACircuitLM:
    """Create a hierarchical PDA with specified parameters."""
    return HierarchicalPDACircuitLM(
        vocab_size=vocab_size,
        global_bits=global_bits,
        local_bits=local_bits,
        stack_depth=stack_depth,
    )