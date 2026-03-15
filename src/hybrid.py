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


class NeuralCorrector(nn.Module):
    """Small neural network that learns to correct CircuitLM predictions.
    
    Input: (circuit_state, stack_top, context, circuit_probs)
    Output: Logits to add to circuit predictions
    """

    def __init__(
        self,
        vocab_size: int,
        num_states: int = 16,
        stack_depth: int = 4,
        max_context_len: int = 32,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_states = num_states
        self.stack_depth = stack_depth
        self.max_context_len = max_context_len

        # Embeddings for discrete inputs
        self.state_embed = nn.Embedding(num_states + 1, embed_dim)  # +1 for padding
        self.stack_embed = nn.Embedding(stack_depth + 1, embed_dim)  # +1 for empty
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Circuit probability projection
        self.circuit_proj = nn.Linear(vocab_size, embed_dim)

        # Context encoder (simple CNN over embeddings)
        self.context_conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=3, padding=1
        )

        # Combined hidden layer
        combined_dim = embed_dim * 4  # state + stack + context + circuit
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

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
        
        # Encode context with conv
        token_emb_t = token_emb.transpose(1, 2)  # (batch, embed_dim, context_len)
        context_enc = self.context_conv(token_emb_t)  # (batch, embed_dim, context_len)
        context_enc = context_enc.mean(dim=2)  # (batch, embed_dim) - average pooling

        # Project circuit probabilities
        circuit_enc = self.circuit_proj(circuit_probs)  # (batch, embed_dim)

        # Combine all features
        combined = torch.cat([
            state_emb,
            stack_emb,
            context_enc,
            circuit_enc,
        ], dim=1)  # (batch, embed_dim * 4)

        # Hidden layers
        hidden = F.relu(self.fc1(combined))
        logits = self.fc2(hidden)  # (batch, vocab_size)

        return logits


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

        corrector = NeuralCorrector(
            vocab_size=circuit.vocab_size,
            num_states=circuit.num_states,
            stack_depth=stack_depth,
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
    )
    
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
    )
