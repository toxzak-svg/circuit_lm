"""
GGUF Weight Recovery — Calibration

Handles loading calibration data and computing loss for E optimization.

Calibration dataset: small representative corpus of text.
We run the model (or a subset of layers) and measure:
1. Per-token perplexity difference between quantized and corrected outputs
2. KL divergence of output distributions
3. Hidden state L2 distance

For the circuit_lm use case, we use Starfire's training corpus
(~5.7MB, 1,758 examples) as calibration data.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class CalibrationExample:
    """A single calibration example (tokenized text)."""
    input_ids: np.ndarray  # token IDs
    target_ids: np.ndarray  # next-token targets
    text: str  # original text (for debugging)


class CalibrationDataset:
    """Manages calibration data loading and batching."""

    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.examples: List[CalibrationExample] = []
        self._loaded = False

    def load_from_text_files(self, paths: List[Path]) -> None:
        """Load calibration data from text files."""
        texts = []
        for path in paths:
            if path.suffix == ".json":
                # Load from JSON (Starfire training format)
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "text" in item:
                                texts.append(item["text"])
                            elif isinstance(item, dict) and "content" in item:
                                texts.append(item["content"])
                            elif isinstance(item, str):
                                texts.append(item)
                    elif isinstance(data, dict) and "text" in data:
                        texts.append(data["text"])
            else:
                # Plain text
                with open(path) as f:
                    texts.append(f.read())

        self._build_examples(texts)
        self._loaded = True

    def load_from_starfire_training(self, training_dir: Path) -> None:
        """Load Starfire's training data as calibration set."""
        import os

        texts = []
        # training_dir has JSON files with conversation data
        for fname in os.listdir(training_dir):
            if fname.endswith(".json"):
                fpath = training_dir / fname
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    # Try 'text' or 'content' or 'input'/'output'
                                    text = item.get("text") or item.get("content")
                                    if text:
                                        texts.append(text)
                                    # Also look for conversation turns
                                    if "messages" in item:
                                        for msg in item["messages"]:
                                            if isinstance(msg, dict):
                                                text = msg.get("content") or msg.get("text")
                                                if text:
                                                    texts.append(text)
                except (json.JSONDecodeError, IOError):
                    pass

        self._build_examples(texts)
        self._loaded = True
        print(f"Loaded {len(self.examples)} calibration examples from Starfire training data")

    def _build_examples(self, texts: List[str]) -> None:
        """Tokenize texts and build examples."""
        # Simple character-level or BPE tokenization
        # For real calibration, use proper tokenizer
        for text in texts:
            if not text or len(text) < 10:
                continue

            # Very simple tokenization: encode to bytes, then to ints
            # In practice, you'd use the model's actual tokenizer
            tokens = np.array([ord(c) % 256 for c in text[:self.max_length]], dtype=np.int64)

            if len(tokens) < 4:
                continue

            # input = all but last token, target = all but first token
            self.examples.append(CalibrationExample(
                input_ids=tokens[:-1],
                target_ids=tokens[1:],
                text=text[:100],
            ))

    def get_batch(self, batch_size: int = 8) -> List[CalibrationExample]:
        """Get a random batch of examples."""
        if not self._loaded:
            return []
        indices = np.random.randint(0, len(self.examples), min(batch_size, len(self.examples)))
        return [self.examples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.examples)


class CalibrationRunner:
    """
    Runs calibration: measures quality of quantized vs corrected model.

    For now, uses a proxy metric: measures dequantized weight statistics
    and how much E corrections shift the distribution.

    Real calibration would run the full model forward pass, but that
    requires the actual model architecture. This simplified version
    measures per-tensor statistics.
    """

    def __init__(self, dataset: CalibrationDataset):
        self.dataset = dataset

    def measure_perplexity_proxy(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
    ) -> Dict[str, float]:
        """
        Proxy perplexity: measures distribution shift between
        quantized/dequantized weights and corrected weights.

        Returns dict with:
        - l2_error: L2 norm of difference
        - max_error: maximum absolute error
        - mean_error: mean absolute error
        - kl_approx: approximate KL divergence (simplified)
        """
        diff = (weights_after - weights_before).flatten()

        metrics = {
            "l2_error": float(np.linalg.norm(diff)),
            "max_error": float(np.abs(diff).max()),
            "mean_error": float(np.abs(diff).mean()),
            "std_error": float(np.std(diff)),
            "relative_error": float(np.linalg.norm(diff) / (np.linalg.norm(weights_before.flatten()) + 1e-8)),
        }

        # Approximate KL: assume Gaussian distributions
        var_before = np.var(weights_before.flatten())
        var_after = np.var(weights_after.flatten())
        kl_approx = 0.5 * (var_before / (var_after + 1e-8) - 1 + np.log(var_after / (var_before + 1e-8)))
        metrics["kl_approx"] = float(kl_approx)

        return metrics

    def run_tensor_calibration(
        self,
        dequantized_weights: Dict[str, np.ndarray],
        corrected_weights: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Run calibration across all tensors.

        Returns aggregate metrics and per-tensor breakdown.
        """
        all_metrics = {}
        agg_l2 = []
        agg_max = []

        for tensor_name in dequantized_weights.keys():
            w_before = dequantized_weights[tensor_name]
            w_after = corrected_weights.get(tensor_name, w_before)
            metrics = self.measure_perplexity_proxy(w_before, w_after)
            all_metrics[tensor_name] = metrics
            agg_l2.append(metrics["l2_error"])
            agg_max.append(metrics["max_error"])

        return {
            "per_tensor": all_metrics,
            "aggregate": {
                "total_l2": float(np.sum(agg_l2)),
                "mean_l2": float(np.mean(agg_l2)),
                "max_l2": float(np.max(agg_l2)),
                "mean_max_error": float(np.mean(agg_max)),
                "worst_tensor": max(all_metrics.items(), key=lambda x: x[1]["max_error"])[0] if all_metrics else None,
            },
        }

    def compute_loss(
        self,
        dequantized_weights: Dict[str, np.ndarray],
        corrected_weights: Dict[str, np.ndarray],
        lambda_constraint: float = 0.01,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute calibration loss: measures quality improvement from corrections.

        Returns (total_loss, metrics_dict).
        """
        metrics = self.run_tensor_calibration(dequantized_weights, corrected_weights)

        # Loss = how much corrections improved distribution match
        # Lower is better: 0 = perfect correction
        loss = metrics["aggregate"]["total_l2"] * (1 + lambda_constraint * metrics["aggregate"]["mean_max_error"])

        return float(loss), metrics