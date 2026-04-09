"""Hybrid model: CircuitLM + Neural Corrector.

The idea:
1. CircuitLM handles structural patterns (state transitions)
2. A small neural network learns to "correct" predictions where CircuitLM is weak
3. Blend circuit + neural predictions

CircuitLM is great at spaces (78%) but terrible at most letters (1-33%).
The neural should focus on where CircuitLM fails.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import PDACircuitLM
from circuit_lm.io import load_model
from circuit_lm.tokenizer import Tokenizer


@dataclass
class TrainingExample:
    """A single training example for the neural corrector."""
    circuit_state: int
    stack_top: int  # -1 for FSM, >=0 for PDA
    context_ids: list[int]  # Previous tokens
    target_token: int  # What the next token actually is
    circuit_histogram: list[int]  # What CircuitLM predicted


class HybridDataset(Dataset):
    """Dataset that pairs CircuitLM predictions with actual next tokens."""

    def __init__(
        self,
        examples: list[TrainingExample],
        vocab_size: int,
        max_context_len: int = 32,
    ):
        self.examples = examples
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        
        # Pad context to fixed length
        context = ex.context_ids[-self.max_context_len:] if ex.context_ids else []
        context = context + [0] * (self.max_context_len - len(context))
        
        # Normalize histogram to probabilities (for neural network)
        hist_sum = sum(ex.circuit_histogram) if ex.circuit_histogram else 1
        if hist_sum == 0:
            hist_sum = 1
        circuit_probs = [h / hist_sum for h in ex.circuit_histogram]
        # Pad to vocab_size
        circuit_probs = circuit_probs + [0.0] * (self.vocab_size - len(circuit_probs))
        
        return {
            'circuit_state': torch.tensor(ex.circuit_state, dtype=torch.long),
            'stack_top': torch.tensor(ex.stack_top, dtype=torch.long),
            'context': torch.tensor(context, dtype=torch.long),
            'circuit_probs': torch.tensor(circuit_probs, dtype=torch.float32),
            'target': torch.tensor(ex.target_token, dtype=torch.long),
        }


class SSDContext(nn.Module):
    """SSD-style linear recurrence for context encoding.
    
    Replaces bidirectional LSTM with a structured linear recurrence:
        h_{t+1} = A @ h_t + B @ token_embed(t)
    
    No gates. No forget/input/reset. Two learnable matrices A and B.
    The hidden state is a fixed-size vector that's actually inspectable.
    
    Based on Mamba-3's insight that structured state spaces are compiler-friendly
    and more efficient than LSTM while being competitive on sequence tasks.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # State transition matrix A — (embed_dim, embed_dim)
        self.A = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        # Input projection B — (embed_dim, embed_dim)
        self.B = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        # Output projection — maps hidden state to context vector
        self.C = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        # Learnable initial state
        self.h0 = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Process sequence of token embeddings with SSD recurrence.
        
        Args:
            token_embeddings: (batch, seq_len, embed_dim)
        
        Returns:
            (batch, embed_dim) — final hidden state as context vector
        """
        batch, seq_len, _ = token_embeddings.shape
        
        # Initialize hidden state
        h = self.h0.unsqueeze(0).expand(batch, -1)  # (batch, embed_dim)
        
        # SSD recurrence — no gates, just linear update + input
        for t in range(seq_len):
            h = torch.matmul(h, self.A.t()) + torch.matmul(token_embeddings[:, t], self.B.t())
            # Normalize for stability (like layer norm but simpler)
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Final hidden state as context representation
        return torch.matmul(h, self.C.t())


class ResidualCorrector(nn.Module):
    """Residual neural corrector that predicts delta to add to circuit output.
    
    Architecture:
      1. Circuit provides base integer prediction
      2. This network predicts delta (correction) to shift that prediction
      3. Final = base + delta (residual connection)
    
    This is more principled than weighted blending - the network explicitly
    learns where the circuit is wrong and by how much.
    
    Input: (circuit_state, stack_top, context, circuit_histogram_counts)
    Output: Delta logits to ADD to circuit histogram
    
    Uses SSD-style context encoder instead of LSTM for efficiency at large vocab.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_states: int = 16,
        stack_depth: int = 4,
        max_context_len: int = 32,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_quantization: bool = False,
        use_ssd: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_states = num_states
        self.stack_depth = stack_depth
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_quantization = use_quantization
        self.use_ssd = use_ssd
        
        # Embeddings for discrete inputs
        self.state_embed = nn.Embedding(num_states + 1, embed_dim)
        self.stack_embed = nn.Embedding(stack_depth + 1, embed_dim)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Circuit histogram projection (use counts directly, not normalized)
        self.hist_proj = nn.Sequential(
            nn.Linear(vocab_size, embed_dim),
            nn.ReLU(),
        )
        
        # Context encoder — SSD (linear recurrence) or LSTM
        if use_ssd:
            self.context_layer = SSDContext(embed_dim)
            context_out = embed_dim  # single final state vector
        else:
            self.context_lstm = nn.LSTM(
                embed_dim, embed_dim, num_layers=1,
                batch_first=True, bidirectional=True
            )
            context_out = embed_dim * 2  # bidirectional concat
        
        # Combined feature dimension
        combined_dim = embed_dim * 3 + context_out
        
        # Deep MLP that outputs DELTA (not final prediction)
        layers = []
        in_dim = combined_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        # Output is delta logits (can be negative)
        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.delta_mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        circuit_state: torch.Tensor,
        stack_top: torch.Tensor,
        context: torch.Tensor,
        circuit_counts: torch.Tensor,  # Integer counts, not probabilities
    ) -> torch.Tensor:
        """Forward pass computing delta to add to circuit prediction.
        
        Args:
            circuit_state: (batch,) - current FSM state
            stack_top: (batch,) - current stack top
            context: (batch, max_context_len) - previous tokens
            circuit_counts: (batch, vocab_size) - Integer counts from circuit
        
        Returns:
            (batch, vocab_size) - Delta logits to ADD to circuit
        """
        batch_size = circuit_state.size(0)
        
        # Embed discrete features
        stack_idx = torch.clamp(stack_top + 1, min=0, max=self.stack_depth)
        state_emb = self.state_embed(circuit_state)
        stack_emb = self.stack_embed(stack_idx)
        
        # Token embeddings for context
        token_emb = self.token_embed(context)  # (batch, context_len, embed_dim)
        
        # Context encoding — SSD (linear recurrence) or LSTM
        if self.use_ssd:
            context_enc = self.context_layer(token_emb)  # (batch, embed_dim)
        else:
            lstm_out, (h_n, _) = self.context_lstm(token_emb)
            context_enc = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, embed_dim*2)
        
        # Project circuit counts (integer histogram -> features)
        circuit_enc = self.hist_proj(circuit_counts.float())  # (batch, embed_dim)
        
        # Combine
        combined = torch.cat([
            state_emb,
            stack_emb,
            context_enc,
            circuit_enc,
        ], dim=1)
        
        # Compute delta
        delta = self.delta_mlp(combined)
        
        return delta


class QuantizedCorrector(nn.Module):
    """Quantized MLP using small float weights for CPU efficiency.
    
    Uses small float weights (initialized from small int range) for CPU efficiency.
    For true integer-only inference, would need custom kernels.
    """
    
    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Small float weights (initialized small for quantization effect)
        # w1: input_dim -> hidden_dim
        # w2: hidden_dim -> vocab_size
        self.w1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.w2 = nn.Parameter(torch.randn(hidden_dim, vocab_size) * 0.1)
        
        # Biases
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with small float weights."""
        h = torch.relu(x @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return out


class NeuralCorrector(nn.Module):
    """Neural network that learns to correct CircuitLM predictions.
    
    Input: (circuit_state, stack_top, context, circuit_probs)
    Output: Logits to add to circuit predictions
    
    Configs (use larger for better quality):
      small:  embed_dim=32,  hidden_dim=64,  num_layers=1
      medium: embed_dim=64,  hidden_dim=128, num_layers=2  
      large:  embed_dim=128, hidden_dim=256, num_layers=3
    
    Uses SSD context encoder by default (use_ssd=False for LSTM).
    """

    def __init__(
        self,
        vocab_size: int,
        num_states: int = 16,
        stack_depth: int = 4,
        max_context_len: int = 32,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_ssd: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_states = num_states
        self.stack_depth = stack_depth
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_ssd = use_ssd

        # Embeddings for discrete inputs
        self.state_embed = nn.Embedding(num_states + 1, embed_dim)  # +1 for padding
        self.stack_embed = nn.Embedding(stack_depth + 1, embed_dim)  # +1 for empty
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Circuit probability projection (with reduction for large vocab)
        if vocab_size > 512:
            self.circuit_proj = nn.Sequential(
                nn.Linear(vocab_size, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.circuit_proj = nn.Linear(vocab_size, embed_dim)

        # Context encoder — SSD (linear recurrence) or LSTM
        if use_ssd:
            self.context_layer = SSDContext(embed_dim)
            context_out = embed_dim
        else:
            self.context_lstm = nn.LSTM(
                embed_dim, embed_dim, num_layers=1, 
                batch_first=True, bidirectional=True
            )
            context_out = embed_dim * 2  # bidirectional
        
        # Combined hidden layer
        combined_dim = embed_dim * 3 + context_out  # state + stack + context + circuit
        
        # Deep MLP with skip connections
        layers = []
        in_dim = combined_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        circuit_state: torch.Tensor,
        stack_top: torch.Tensor,
        context: torch.Tensor,
        circuit_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            circuit_state: (batch,) - current FSM state
            stack_top: (batch,) - current stack top (-1 = empty)
            context: (batch, max_context_len) - previous tokens
            circuit_probs: (batch, vocab_size) - CircuitLM's prediction probs
        
        Returns:
            (batch, vocab_size) - correction logits to add to circuit
        """
        batch_size = circuit_state.size(0)

        # Embed discrete features
        # Handle stack_top < 0 by mapping to index num_states
        stack_idx = torch.clamp(stack_top + 1, min=0, max=self.stack_depth)
        state_emb = self.state_embed(circuit_state)  # (batch, embed_dim)
        stack_emb = self.stack_embed(stack_idx)       # (batch, embed_dim)

        # Token embeddings for context
        token_emb = self.token_embed(context)  # (batch, context_len, embed_dim)
        
        # Context encoding — SSD or LSTM
        if self.use_ssd:
            context_enc = self.context_layer(token_emb)  # (batch, embed_dim)
        else:
            lstm_out, (h_n, c_n) = self.context_lstm(token_emb)
            context_enc = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, embed_dim * 2)

        # Project circuit probabilities
        circuit_enc = self.circuit_proj(circuit_probs)  # (batch, embed_dim)

        # Combine all features
        combined = torch.cat([
            state_emb,
            stack_emb,
            context_enc,
            circuit_enc,
        ], dim=1)

        # Deep MLP
        logits = self.mlp(combined)  # (batch, vocab_size)

        return logits


class ResidualHybridModel:
    """Hybrid using residual correction (base + delta).
    
    This approach is more principled than weighted blending:
    - Circuit provides base integer prediction (histogram counts)
    - Neural network predicts DELTA to add to the histogram
    - Final prediction = argmax(base + delta)
    
    The residual approach explicitly learns where the circuit is wrong.
    """
    
    def __init__(
        self,
        circuit: CircuitLM | PDACircuitLM,
        corrector: ResidualCorrector,
    ):
        self.circuit = circuit
        self.corrector = corrector
    
    def predict(
        self,
        context_ids: list[int],
    ) -> tuple[int, dict]:
        """Predict using residual correction."""
        device = next(self.corrector.parameters()).device
        
        # Get circuit state
        if isinstance(self.circuit, PDACircuitLM):
            state = 0
            stack = []
            for tok in context_ids:
                state, stack = self.circuit.step(state, stack, tok)
            
            circuit_hist = self.circuit.config_histogram(state, stack)
            stack_top = stack[-1] if stack else -1
        else:
            state = 0
            for tok in context_ids:
                state = self.circuit.next_state(state, tok)
            
            circuit_hist = self.circuit.state_histogram(state)
            stack_top = -1
        
        # Prepare inputs
        # Use raw counts (not normalized) for residual learning
        circuit_counts = torch.tensor(
            list(circuit_hist) + [0] * (self.corrector.vocab_size - len(circuit_hist)),
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        context_padded = context_ids[-self.corrector.max_context_len:]
        context_padded = context_padded + [0] * (self.corrector.max_context_len - len(context_padded))
        
        with torch.no_grad():
            delta = self.corrector(
                circuit_state=torch.tensor([state], dtype=torch.long).to(device),
                stack_top=torch.tensor([stack_top], dtype=torch.long).to(device),
                context=torch.tensor([context_padded], dtype=torch.long).to(device),
                circuit_counts=circuit_counts,
            )
        
        # Add delta to base counts
        base = circuit_counts.squeeze(0)
        adjusted = base + delta.squeeze(0)
        
        # Argmax
        predicted_token = adjusted.argmax().item()
        
        info = {
            'circuit_state': state,
            'stack_top': stack_top,
            'circuit_histogram': circuit_hist,
            'delta': delta.cpu().numpy().tolist(),
            'adjusted': adjusted.cpu().numpy().tolist(),
        }
        
        return predicted_token, info
    
    def save(self, path: str) -> None:
        """Save residual hybrid model."""
        torch.save({
            'corrector_state_dict': self.corrector.state_dict(),
            'corrector_type': 'residual',
            'embed_dim': self.corrector.embed_dim,
            'hidden_dim': self.corrector.hidden_dim,
            'num_layers': self.corrector.num_layers,
        }, path)
    
    @classmethod
    def load(
        cls,
        circuit_path: str,
        corrector_path: str,
    ) -> tuple['ResidualHybridModel', Tokenizer]:
        """Load residual hybrid model."""
        circuit, tokenizer = load_model(circuit_path)
        
        stack_depth = getattr(circuit, 'stack_depth', 4)
        
        checkpoint = torch.load(corrector_path, map_location='cpu')
        embed_dim = checkpoint.get('embed_dim', 64)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_layers = checkpoint.get('num_layers', 2)
        
        corrector = ResidualCorrector(
            vocab_size=circuit.vocab_size,
            num_states=circuit.num_states,
            stack_depth=stack_depth,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        corrector.load_state_dict(checkpoint['corrector_state_dict'])
        
        return cls(circuit, corrector), tokenizer


class HybridModel:
    """Hybrid: CircuitLM + Neural Corrector."""

    def __init__(
        self,
        circuit: CircuitLM | PDACircuitLM,
        corrector: NeuralCorrector,
        circuit_weight: float = 0.5,
    ):
        self.circuit = circuit
        self.corrector = corrector
        self.circuit_weight = circuit_weight  # Weight for circuit in final blend
        self.corrector_weight = 1.0 - circuit_weight

    def predict(
        self,
        context_ids: list[int],
    ) -> tuple[int, dict]:
        """Predict next token using hybrid approach.
        
        Returns:
            (predicted_token, info_dict)
        """
        device = next(self.corrector.parameters()).device
        
        # Get CircuitLM prediction
        if isinstance(self.circuit, PDACircuitLM):
            # PDA: track state and stack
            state = 0
            stack = []
            for tok in context_ids:
                state, stack = self.circuit.step(state, stack, tok)
            
            circuit_hist = self.circuit.config_histogram(state, stack)
            stack_top = stack[-1] if stack else -1
        else:
            # FSM
            state = 0
            for tok in context_ids:
                state = self.circuit.next_state(state, tok)
            
            circuit_hist = self.circuit.state_histogram(state)
            stack_top = -1

        # Convert histogram to tensor
        hist_sum = sum(circuit_hist) if circuit_hist else 1
        if hist_sum == 0:
            hist_sum = 1
        circuit_probs = torch.tensor(
            [h / hist_sum for h in circuit_hist] + [0.0] * (self.corrector.vocab_size - len(circuit_hist)),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        # Get corrector prediction
        context_padded = context_ids[-self.corrector.max_context_len:]
        context_padded = context_padded + [0] * (self.corrector.max_context_len - len(context_padded))
        
        with torch.no_grad():
            correction_logits = self.corrector(
                circuit_state=torch.tensor([state], dtype=torch.long).to(device),
                stack_top=torch.tensor([stack_top], dtype=torch.long).to(device),
                context=torch.tensor([context_padded], dtype=torch.long).to(device),
                circuit_probs=circuit_probs,
            )

        # Blend circuit probs + corrector logits
        circuit_probs = circuit_probs.squeeze(0)
        circuit_logits = torch.log(circuit_probs + 1e-8)
        
        combined_logits = (
            self.circuit_weight * circuit_logits + 
            self.corrector_weight * correction_logits.squeeze(0)
        )

        # Argmax
        predicted_token = combined_logits.argmax().item()

        info = {
            'circuit_state': state,
            'stack_top': stack_top,
            'circuit_histogram': circuit_hist,
            'correction_logits': correction_logits.cpu().numpy().tolist(),
            'combined_logits': combined_logits.cpu().numpy().tolist(),
        }

        return predicted_token, info

    def save(self, path: str) -> None:
        """Save hybrid model (circuit + corrector state dict)."""
        torch.save({
            'corrector_state_dict': self.corrector.state_dict(),
            'circuit_weight': self.circuit_weight,
            'circuit_type': 'pda' if isinstance(self.circuit, PDACircuitLM) else 'fsm',
            'embed_dim': self.corrector.embed_dim,
            'hidden_dim': self.corrector.hidden_dim,
            'num_layers': self.corrector.num_layers,
        }, path)

    @classmethod
    def load(
        cls,
        circuit_path: str,
        corrector_path: str,
        circuit_weight: float = 0.5,
    ) -> tuple['HybridModel', Tokenizer]:
        """Load hybrid model and its tokenizer (from the circuit file).

        Returns:
            (HybridModel, Tokenizer) so the caller can decode generated tokens.
        """
        circuit, tokenizer = load_model(circuit_path)

        stack_depth = getattr(circuit, 'stack_depth', 4)

        # Load checkpoint to get architecture params
        checkpoint = torch.load(corrector_path, map_location='cpu')
        embed_dim = checkpoint.get('embed_dim', 64)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_layers = checkpoint.get('num_layers', 2)
        
        corrector = NeuralCorrector(
            vocab_size=circuit.vocab_size,
            num_states=circuit.num_states,
            stack_depth=stack_depth,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        checkpoint = torch.load(corrector_path, map_location='cpu')
        corrector.load_state_dict(checkpoint['corrector_state_dict'])

        weight = checkpoint.get('circuit_weight', circuit_weight)
        return cls(circuit, corrector, weight), tokenizer


def generate_reply_hybrid(
    hybrid: HybridModel,
    tokenizer: Tokenizer,
    prompt_ids: list[int],
    max_tokens: int,
    stop_token_ids: list[int] | None = None,
    stop_sequence: list[int] | None = None,
) -> list[int]:
    """Generate reply token IDs using the hybrid model (greedy).

    Stops at first stop token, at stop_sequence, or at max_tokens.
    Returns only the generated reply tokens (no prompt).
    """
    if stop_token_ids is None:
        stop_token_ids = tokenizer.encode("\n")
    if stop_sequence is None:
        stop_sequence = []

    context_ids = list(prompt_ids)
    reply: list[int] = []

    for _ in range(max_tokens):
        next_tok, _ = hybrid.predict(context_ids)
        if stop_sequence and len(reply) + 1 >= len(stop_sequence):
            suffix = (reply + [next_tok])[-len(stop_sequence):]
            if suffix == stop_sequence:
                break
        if next_tok in stop_token_ids:
            break
        reply.append(next_tok)
        context_ids.append(next_tok)

    return reply


def build_dataset(
    circuit: CircuitLM | PDACircuitLM,
    tokenizer: Tokenizer,
    data_path: str,
    max_examples: int = 10000,
) -> list[TrainingExample]:
    """Build training dataset by running CircuitLM on data and comparing to true next tokens.
    
    This is the key: we collect cases where CircuitLM was WRONG, because that's
    what the neural needs to learn.
    """
    # Load text data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    token_ids = tokenizer.encode(text)
    
    examples = []
    is_pda = isinstance(circuit, PDACircuitLM)
    
    # Track state across the text
    state = 0
    stack = []
    
    for i, target_token in enumerate(token_ids[1:], 1):
        prev_token = token_ids[i - 1] if i > 0 else 0
        
        # Get CircuitLM's prediction BEFORE seeing the target
        if is_pda:
            circuit_hist = circuit.config_histogram(state, stack)
        else:
            circuit_hist = circuit.state_histogram(state)
        
        # Store example
        context = token_ids[max(0, i-32):i]
        
        example = TrainingExample(
            circuit_state=state,
            stack_top=stack[-1] if stack else -1,
            context_ids=context,
            target_token=target_token,
            circuit_histogram=circuit_hist,
        )
        examples.append(example)
        
        # Update state for next step
        if is_pda:
            state, stack = circuit.step(state, stack, target_token)
        else:
            state = circuit.next_state(state, target_token)
        
        if len(examples) >= max_examples:
            break
    
    return examples


def train_hybrid(
    circuit_path: str,
    data_path: str,
    output_path: str,
    num_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    circuit_weight: float = 0.5,
    max_examples: int = 50000,
    max_context_len: int = 32,
    # Larger corrector options:
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    use_ssd: bool = True,
):
    """Train the neural corrector.

    The circuit (and its tokenizer/vocab) must already be trained. Use
    circuit-lm train with --tokenizer bpe --bpe_merges 512 for a larger
    vocabulary, then run hybrid-train on the same data.
    """
    print(f"Loading circuit from {circuit_path}...")
    circuit, tokenizer = load_model(circuit_path)
    
    print(f"Building dataset from {data_path}...")
    examples = build_dataset(circuit, tokenizer, data_path, max_examples)
    print(f"Built {len(examples)} training examples")
    
    # Count how many CircuitLM got wrong
    wrong_count = 0
    for ex in examples:
        hist = ex.circuit_histogram
        if hist:
            pred = max(range(len(hist)), key=lambda i: hist[i])
            if pred != ex.target_token:
                wrong_count += 1
    print(f"CircuitLM accuracy: {100*(len(examples)-wrong_count)/len(examples):.2f}%")
    
    # Create dataset and dataloader
    dataset = HybridDataset(examples, circuit.vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    stack_depth = getattr(circuit, 'stack_depth', 4)

    corrector = NeuralCorrector(
        vocab_size=circuit.vocab_size,
        num_states=circuit.num_states,
        stack_depth=stack_depth,
        max_context_len=max_context_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_ssd=use_ssd,
    )
    
    ssd_note = " (SSD)" if use_ssd else " (LSTM)"
    num_params = sum(p.numel() for p in corrector.parameters())
    print(f"NeuralCorrector{ssd_note}: {num_params:,} parameters")
    
    # Training setup
    optimizer = torch.optim.Adam(corrector.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    corrector.to(device)
    circuit_weight_t = torch.tensor(circuit_weight, dtype=torch.float32).to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_count = 0
        
        for batch in dataloader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get corrector logits
            correction_logits = corrector(
                batch['circuit_state'],
                batch['stack_top'],
                batch['context'],
                batch['circuit_probs'],
            )
            
            # Blend with circuit (in log-space for numerical stability)
            circuit_logits = torch.log(batch['circuit_probs'] + 1e-8)
            combined_logits = (
                circuit_weight_t * circuit_logits + 
                (1 - circuit_weight_t) * correction_logits
            )
            
            # Train on cross-entropy
            loss = criterion(combined_logits, batch['target'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = combined_logits.argmax(dim=1)
            total_correct += (preds == batch['target']).sum().item()
            total_count += batch['target'].size(0)
        
        accuracy = 100 * total_correct / total_count
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")
    
    # Save
    hybrid = HybridModel(circuit, corrector, circuit_weight)
    torch.save({
        'corrector_state_dict': corrector.state_dict(),
        'circuit_weight': circuit_weight,
    }, output_path)
    print(f"Saved to {output_path}")
    
    return hybrid


def train_residual_hybrid(
    circuit_path: str,
    data_path: str,
    output_path: str,
    num_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_examples: int = 50000,
    max_context_len: int = 32,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    use_ssd: bool = True,
) -> ResidualHybridModel:
    """Train the residual neural corrector.
    
    This trains the network to predict DELTA to add to circuit predictions,
    rather than blending circuit + neural outputs.
    """
    print(f"Loading circuit from {circuit_path}...")
    circuit, tokenizer = load_model(circuit_path)
    
    print(f"Building dataset from {data_path}...")
    examples = build_dataset(circuit, tokenizer, data_path, max_examples)
    print(f"Built {len(examples)} training examples")
    
    # Count how many CircuitLM got wrong
    wrong_count = 0
    for ex in examples:
        hist = ex.circuit_histogram
        if hist:
            pred = max(range(len(hist)), key=lambda i: hist[i])
            if pred != ex.target_token:
                wrong_count += 1
    print(f"CircuitLM accuracy: {100*(len(examples)-wrong_count)/len(examples):.2f}%")
    
    # Create dataset - use counts directly for residual learning
    class ResidualDataset(Dataset):
        def __init__(self, examples, vocab_size, max_context_len):
            self.examples = examples
            self.vocab_size = vocab_size
            self.max_context_len = max_context_len
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            ex = self.examples[idx]
            context = ex.context_ids[-self.max_context_len:] if ex.context_ids else []
            context = context + [0] * (self.max_context_len - len(context))
            
            # Use raw counts (not normalized)
            counts = ex.circuit_histogram + [0] * (self.vocab_size - len(ex.circuit_histogram))
            
            return {
                'circuit_state': torch.tensor(ex.circuit_state, dtype=torch.long),
                'stack_top': torch.tensor(ex.stack_top, dtype=torch.long),
                'context': torch.tensor(context, dtype=torch.long),
                'circuit_counts': torch.tensor(counts, dtype=torch.float32),
                'target': torch.tensor(ex.target_token, dtype=torch.long),
            }
    
    dataset = ResidualDataset(examples, circuit.vocab_size, max_context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    stack_depth = getattr(circuit, 'stack_depth', 4)
    
    corrector = ResidualCorrector(
        vocab_size=circuit.vocab_size,
        num_states=circuit.num_states,
        stack_depth=stack_depth,
        max_context_len=max_context_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_ssd=use_ssd,
    )
    
    ssd_note = " (SSD)" if use_ssd else " (LSTM)"
    num_params = sum(p.numel() for p in corrector.parameters())
    print(f"ResidualCorrector{ssd_note}: {num_params:,} parameters")
    
    optimizer = torch.optim.Adam(corrector.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    corrector.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0
        total_count = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get delta from corrector
            delta = corrector(
                batch['circuit_state'],
                batch['stack_top'],
                batch['context'],
                batch['circuit_counts'],
            )
            
            # Add delta to base counts
            base = batch['circuit_counts']
            adjusted = base + delta
            
            # Cross-entropy loss
            loss = criterion(adjusted, batch['target'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = adjusted.argmax(dim=1)
            total_correct += (preds == batch['target']).sum().item()
            total_count += batch['target'].size(0)
        
        accuracy = 100 * total_correct / total_count
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.2f}%")
    
    # Save
    hybrid = ResidualHybridModel(circuit, corrector)
    hybrid.save(output_path)
    print(f"Saved to {output_path}")
    
    return hybrid


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--circuit', required=True, help='Path to circuit model .json')
    parser.add_argument('--data', required=True, help='Path to training data .txt')
    parser.add_argument('--output', required=True, help='Output path for corrector')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--circuit-weight', type=float, default=0.5)
    parser.add_argument('--max-examples', type=int, default=50000)
    parser.add_argument('--use-ssd', action='store_true', help='Use SSD context encoder (default: True, use --no-ssd to disable')
    parser.add_argument('--no-ssd', dest='use_ssd', action='store_false', help='Disable SSD context encoder')

    
    args = parser.parse_args()
    
    train_hybrid(
        args.circuit,
        args.data,
        args.output,
        args.epochs,
        args.batch,
        args.lr,
        args.circuit_weight,
        args.max_examples,
        use_ssd=args.use_ssd,
    )
