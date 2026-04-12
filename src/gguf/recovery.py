"""
GGUF Weight Recovery — Residual Parameter Layer

Core idea: W_approx = dequantize(quantized) + E

E is a learned correction that compensates for quantization error.
E is constrained so that re-quantization lands in valid codebook regions.

Supported E parameterizations:
- Per-block affine: E_block = alpha * block_scale + beta
- Low-rank: E = U @ V.T (very small param count)
- Per-tensor scalar: E_tensor = alpha * residual_scale


This module defines the E parameter structures and the forward pass
that computes W_approx from quantized weights + E corrections.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class EParameterization(Enum):
    """How E is structured / parameterized."""
    PER_TENSOR_SCALAR = "per_tensor_scalar"    # 1 alpha per tensor
    PER_BLOCK_AFFINE = "per_block_affine"      # alpha + beta per block
    LOW_RANK = "low_rank"                      # U @ V.T per tensor
    CODEBOOK_OFFSET = "codebook_offset"         # shift cluster assignment


@dataclass
class BlockCorrection:
    """Per-block correction parameters for one tensor block."""
    alpha: np.ndarray  # shape (1,) or (block_size,)
    beta: np.ndarray   # shape (1,) or (block_size,)


@dataclass
class TensorCorrection:
    """Correction parameters for a single tensor."""
    tensor_name: str
    n_blocks: int
    block_size: int

    # Parameterization type
    param_type: EParameterization

    # For PER_BLOCK_AFFINE:
    alphas: Optional[np.ndarray] = None  # shape (n_blocks,) or (n_blocks, block_size)
    betas: Optional[np.ndarray] = None   # shape (n_blocks,) or (n_blocks, block_size)

    # For LOW_RANK:
    U: Optional[np.ndarray] = None  # shape (n_blocks, rank) or (n_elements, rank)
    V: Optional[np.ndarray] = None  # shape (n_blocks, rank) or (n_elements, rank)

    # For PER_TENSOR_SCALAR:
    alpha: Optional[float] = None
    residual_scale: Optional[np.ndarray] = None  # precomputed residual direction

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        if self.param_type == EParameterization.PER_TENSOR_SCALAR:
            return 1
        elif self.param_type == EParameterization.PER_BLOCK_AFFINE:
            return 2 * self.n_blocks
        elif self.param_type == EParameterization.LOW_RANK:
            rank = self.U.shape[1] if self.U is not None else 4
            return rank * (self.U.shape[0] + self.V.shape[0])
        elif self.param_type == EParameterization.CODEBOOK_OFFSET:
            return self.n_blocks
        return 0

    def to_dict(self) -> dict:
        """Serialize to dict for saving."""
        return {
            "tensor_name": self.tensor_name,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "param_type": self.param_type.value,
            "n_params": self.n_params,
        }


@dataclass
class RecoveryConfig:
    """Configuration for the recovery process."""
    param_type: EParameterization = EParameterization.PER_BLOCK_AFFINE
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_iterations: int = 1000
    early_stop_patience: int = 50
    calibration_split: float = 0.1  # fraction of calibration data for validation
    lambda_constraint: float = 0.01  # strength of re-quantization constraint
    verbose: bool = True

    @classmethod
    def from_json(cls, path: str) -> "RecoveryConfig":
        with open(path) as f:
            d = json.load(f)
        d["param_type"] = EParameterization(d["param_type"])
        return cls(**d)


class ResidualRecovery:
    """
    Manages E corrections for all tensors in a model.

    Usage:
        recovery = ResidualRecovery(config)
        recovery.attach_tensors(tensors_info)  # tensors from GGUF parser

        # Forward pass: W_approx = W_quant_dequant + E
        W_approx = recovery.apply_corrections(dequantized_weights, tensor_name)

        # Optimization
        for batch in calibration_data:
            loss = compute_loss(recovery, batch)
            recovery.step(loss)

        # After optimization: save E, then re-quantize
        recovery.save("recovery_params.json")
    """

    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.tensor_corrections: Dict[str, TensorCorrection] = {}
        self._is_attached = False

    def attach_tensors(self, tensor_infos: List[dict]) -> None:
        """Initialize E parameters for each tensor.

        Args:
            tensor_infos: List of dicts with keys: name, n_elements, block_size
        """
        self.tensor_corrections = {}

        for tinfo in tensor_infos:
            n_elements = tinfo["n_elements"]
            block_size = tinfo.get("block_size", 32)
            n_blocks = (n_elements + block_size - 1) // block_size
            dtype = tinfo.get("dtype", 10)  # default Q4_K

            if self.config.param_type == EParameterization.PER_BLOCK_AFFINE:
                # Each block gets alpha + beta (scalar per block)
                alphas = np.zeros(n_blocks, dtype=np.float32)
                betas = np.zeros(n_blocks, dtype=np.float32)
                correction = TensorCorrection(
                    tensor_name=tinfo["name"],
                    n_blocks=n_blocks,
                    block_size=block_size,
                    param_type=EParameterization.PER_BLOCK_AFFINE,
                    alphas=alphas,
                    betas=betas,
                )
            elif self.config.param_type == EParameterization.PER_TENSOR_SCALAR:
                correction = TensorCorrection(
                    tensor_name=tinfo["name"],
                    n_blocks=n_blocks,
                    block_size=block_size,
                    param_type=EParameterization.PER_TENSOR_SCALAR,
                    alpha=0.0,
                )
            elif self.config.param_type == EParameterization.LOW_RANK:
                rank = 4  # default low rank
                # U: (n_blocks, rank), V: (n_blocks, rank)
                U = np.random.randn(n_blocks, rank).astype(np.float32) * 1e-3
                V = np.random.randn(n_blocks, rank).astype(np.float32) * 1e-3
                correction = TensorCorrection(
                    tensor_name=tinfo["name"],
                    n_blocks=n_blocks,
                    block_size=block_size,
                    param_type=EParameterization.LOW_RANK,
                    U=U,
                    V=V,
                )
            else:
                raise NotImplementedError(self.config.param_type)

            self.tensor_corrections[tinfo["name"]] = correction

        self._is_attached = True

    def apply_corrections(
        self,
        dequantized: np.ndarray,
        tensor_name: str,
        block_size: int,
    ) -> np.ndarray:
        """Apply E corrections to a dequantized tensor.

        W_approx = W_dequant + E

        Args:
            dequantized: float32 array, already reshaped to tensor.shape
            tensor_name: name of the tensor
            block_size: elements per block for this dtype

        Returns:
            Corrected float32 array with same shape as dequantized
        """
        if tensor_name not in self.tensor_corrections:
            return dequantized

        corr = self.tensor_corrections[tensor_name]
        n_elements = dequantized.size
        n_blocks = (n_elements + block_size - 1) // block_size
        flat = dequantized.flatten()

        if corr.param_type == EParameterization.PER_BLOCK_AFFINE:
            result = flat.copy()
            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, n_elements)
                block = result[start:end]
                alpha = corr.alphas[i] if corr.alphas[i] != 0 else 1.0
                beta = corr.betas[i] if corr.betas[i] != 0 else 0.0
                result[start:end] = alpha * block + beta
            return result.reshape(dequantized.shape)

        elif corr.param_type == EParameterization.PER_TENSOR_SCALAR:
            alpha = corr.alpha if corr.alpha is not None else 0.0
            if alpha == 0:
                return dequantized
            result = dequantized + alpha * (dequantized - dequantized.mean())
            return result

        elif corr.param_type == EParameterization.LOW_RANK:
            # E = U @ V.T per block
            result = flat.copy()
            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, n_elements)
                block_flat = flat[start:end]
                # Low rank: use block index as embedding
                e_block = corr.U[i] @ corr.V[i].T  # (block_size,)
                if e_block.shape[0] != block_flat.shape[0]:
                    # Tile if mismatch
                    e_block = np.tile(e_block, (block_flat.shape[0] // e_block.shape[0] + 1))[:block_flat.shape[0]]
                result[start:end] = block_flat + e_block
            return result.reshape(dequantized.shape)

        return dequantized

    def compute_constraint_penalty(
        self,
        dequantized: np.ndarray,
        tensor_name: str,
        block_size: int,
    ) -> float:
        """
        Compute penalty for re-quantization deviation.
        
        This measures how much E causes the re-quantized weights to differ
        from the original quantized values.
        """
        corr = self.tensor_corrections.get(tensor_name)
        if corr is None:
            return 0.0

        # Simplified: just measure E magnitude
        if corr.param_type == EParameterization.PER_BLOCK_AFFINE:
            alpha_deviation = np.abs(corr.alphas - 1.0).mean()
            beta_magnitude = np.abs(corr.betas).mean()
            return alpha_deviation + beta_magnitude
        elif corr.param_type == EParameterization.PER_TENSOR_SCALAR:
            return abs(corr.alpha or 0.0)
        elif corr.param_type == EParameterization.LOW_RANK:
            return np.abs(corr.U).mean() + np.abs(corr.V).mean()

        return 0.0

    def get_trainable_params(self) -> List[Tuple[str, np.ndarray]]:
        """Return all trainable parameter arrays for gradient descent."""
        params = []
        for name, corr in self.tensor_corrections.items():
            if corr.param_type == EParameterization.PER_BLOCK_AFFINE:
                params.append((f"{name}.alphas", corr.alphas))
                params.append((f"{name}.betas", corr.betas))
            elif corr.param_type == EParameterization.PER_TENSOR_SCALAR:
                if corr.alpha is not None:
                    params.append((f"{name}.alpha", np.array([corr.alpha])))
            elif corr.param_type == EParameterization.LOW_RANK:
                params.append((f"{name}.U", corr.U))
                params.append((f"{name}.V", corr.V))
        return params

    def save(self, path: str) -> None:
        """Save E parameters to JSON (numpy arrays as lists)."""
        data = {}
        for name, corr in self.tensor_corrections.items():
            d = {
                "tensor_name": corr.tensor_name,
                "n_blocks": corr.n_blocks,
                "block_size": corr.block_size,
                "param_type": corr.param_type.value,
            }
            if corr.param_type == EParameterization.PER_BLOCK_AFFINE:
                d["alphas"] = corr.alphas.tolist()
                d["betas"] = corr.betas.tolist()
            elif corr.param_type == EParameterization.PER_TENSOR_SCALAR:
                d["alpha"] = corr.alpha
            elif corr.param_type == EParameterization.LOW_RANK:
                d["U"] = corr.U.tolist()
                d["V"] = corr.V.tolist()
            data[name] = d

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load E parameters from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.tensor_corrections = {}
        for name, d in data.items():
            corr = TensorCorrection(
                tensor_name=d["tensor_name"],
                n_blocks=d["n_blocks"],
                block_size=d["block_size"],
                param_type=EParameterization(d["param_type"]),
            )
            if corr.param_type == EParameterization.PER_BLOCK_AFFINE:
                corr.alphas = np.array(d["alphas"], dtype=np.float32)
                corr.betas = np.array(d["betas"], dtype=np.float32)
            elif corr.param_type == EParameterization.PER_TENSOR_SCALAR:
                corr.alpha = d.get("alpha", 0.0)
            elif corr.param_type == EParameterization.LOW_RANK:
                corr.U = np.array(d["U"], dtype=np.float32)
                corr.V = np.array(d["V"], dtype=np.float32)
            self.tensor_corrections[name] = corr

    def summary(self) -> str:
        """Return a human-readable summary of corrections."""
        lines = [f"Recovery method: {self.config.param_type.value}"]
        total_params = sum(c.n_params for c in self.tensor_corrections.values())
        lines.append(f"Tensors with corrections: {len(self.tensor_corrections)}")
        lines.append(f"Total correction parameters: {total_params:,}")
        return "\n".join(lines)