"""Data loading utilities for H-UoE training: pre-tokenized .npy / .bin from --data-dir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_token_array(path: Path) -> np.ndarray:
    """Load tokens from a single file. Returns (N, L) or (L,) int array."""
    p = path.as_posix()
    if p.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
    elif p.endswith(".bin"):
        # Raw uint16 or uint32; assume one contiguous stream, shape inferred from file size
        raw = np.fromfile(path, dtype=np.uint32)
        arr = raw
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .npy or .bin")
    return np.asarray(arr, dtype=np.int64)


class PreTokenizedDataset(Dataset):
    """Dataset from --data-dir: tokens.npy (N, L) or directory of .npy/.bin per sequence.

    - Single file: data_dir/tokens.npy with shape (num_seqs, seq_len) or (num_seqs, L).
      If L > seq_len we take the first seq_len tokens per row; if L < seq_len we pad with 0.
    - Multiple files: data_dir/*.npy (or *.bin), each file one sequence (1D); truncated/padded to seq_len.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        vocab_size: int = 50257,
        pad_id: int = 0,
    ) -> None:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        single = data_dir / "tokens.npy"
        if single.exists():
            data = _load_token_array(single)
            if data.ndim == 1:
                # One long stream: reshape into (num_seqs, seq_len) by truncation
                n = len(data) // seq_len
                data = data[: n * seq_len].reshape(n, seq_len)
            elif data.ndim == 2:
                pass  # (N, L)
            else:
                raise ValueError(f"tokens.npy must be 1D or 2D, got shape {data.shape}")
            self._data = torch.from_numpy(np.asarray(data, dtype=np.int64))
            self._single_file = True
        else:
            files = sorted(data_dir.glob("*.npy")) + sorted(data_dir.glob("*.bin"))
            if not files:
                raise FileNotFoundError(
                    f"No tokens.npy or *.npy/*.bin found in {data_dir}"
                )
            self._files = files
            self._single_file = False
            self._seq_len = seq_len
            self._pad_id = pad_id
            self._vocab_size = vocab_size

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def __len__(self) -> int:
        if self._single_file:
            return self._data.size(0)
        return len(self._files)

    def _get_single_file(self, i: int) -> tuple[torch.Tensor, int]:
        row = self._data[i]
        valid = min(row.size(0), self.seq_len)
        if row.size(0) >= self.seq_len:
            return row[: self.seq_len].clone(), self.seq_len
        out = torch.full((self.seq_len,), self.pad_id, dtype=row.dtype)
        out[:valid] = row[:valid]
        return out, valid

    def _get_multi_file(self, i: int) -> tuple[torch.Tensor, int]:
        path = self._files[i]
        arr = _load_token_array(path)
        if arr.ndim > 1:
            arr = arr.ravel()
        seq = torch.from_numpy(np.asarray(arr, dtype=np.int64))
        valid = min(seq.size(0), self.seq_len)
        if seq.size(0) >= self.seq_len:
            return seq[: self.seq_len].clone(), self.seq_len
        out = torch.full((self.seq_len,), self.pad_id, dtype=seq.dtype)
        out[:valid] = seq[:valid]
        return out, valid

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if self._single_file:
            seq, valid_len = self._get_single_file(i)
        else:
            seq, valid_len = self._get_multi_file(i)
        labels = seq.clone()
        labels[valid_len:] = -100  # CE loss ignore_index=-100
        return {"input_ids": seq, "labels": labels}
