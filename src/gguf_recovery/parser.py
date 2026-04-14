"""
GGUF Parser — wraps the gguf library for Qwen2.5 compatibility.

GGUF version 3 format: keys are null-terminated C strings with u8 type + 8-byte aligned values.
Tensor infos stored before tensor data at data_offset.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import sys

# Get the pre-loaded system gguf (run_recovery.py preloads this as 'system_gguf')
# This avoids collision with the local src/gguf package which Python caches as 'gguf'
gguf_lib = sys.modules.get('system_gguf')
if gguf_lib is None or not hasattr(gguf_lib, 'GGUFReader'):
    import gguf as gguf_lib


QUANT_TYPES = {
    # From gguf.GGMLQuantizationType enum — GGUF v3
    0: ('F32', 1, 4),       # Float32
    1: ('F16', 1, 2),       # Float16
    2: ('Q4_0', 32, 2),     # 4-bit, 32 elem/block, 2 bytes meta + 16 bytes data
    3: ('Q4_1', 32, 2),     # 4-bit with offset, 32 elem/block, 4 bytes meta + 16 bytes data
    6: ('Q5_0', 32, 2),     # 5-bit
    7: ('Q5_1', 32, 2),     # 5-bit with offset
    8: ('Q8_0', 32, 1),     # 8-bit, 32 elem/block, 4 bytes scale + 32 bytes data
    9: ('Q8_1', 32, 1),     # 8-bit with offset
    10: ('Q2_K', 256, 2),   # 2-bit K-quantization, 256 elem/block
    11: ('Q3_K', 256, 2),   # 3-bit K-quantization, 256 elem/block
    12: ('Q4_K', 256, 2),   # 4-bit K-quantization, 256 elem/block
    13: ('Q5_K', 256, 2),   # 5-bit K-quantization, 256 elem/block
    14: ('Q6_K', 256, 1),   # 6-bit K-quantization, 256 elem/block
    15: ('Q8_K', 256, 1),   # 8-bit K-quantization, 256 elem/block
    16: ('IQ2_XXS', 32, 2), # Incremental 2-bit
    17: ('IQ2_XS', 32, 2),
    18: ('IQ3_XXS', 32, 2),
    19: ('IQ1_S', 32, 2),
    20: ('IQ4_NL', 32, 2),   # 4-bit non-linear
    21: ('IQ3_S', 32, 2),
    22: ('IQ2_S', 32, 2),
    23: ('IQ4_XS', 32, 2),
    24: ('I8', 1, 1),        # Int8
    25: ('I16', 1, 2),       # Int16
    26: ('I32', 1, 4),       # Int32
    27: ('I64', 1, 8),       # Int64
    28: ('F64', 1, 8),       # Float64
    29: ('IQ1_M', 32, 2),
    30: ('BF16', 1, 2),      # Brain Float16
    34: ('TQ1_0', 32, 2),
    35: ('TQ2_0', 32, 2),
    39: ('MXFP4', 32, 2),
}


@dataclass
class GGUFMetadata:
    """Metadata from GGUF file header."""
    version: int = 3
    tensor_count: int = 0
    metadata_len: int = 0
    alignment: int = 32
    arch: str = 'qwen2'
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    context_length: int = 32768
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorInfo:
    """Metadata about a single tensor."""
    name: str
    shape: List[int]
    dtype: int  # GGML type enum
    n_elements: int
    bytes_size: int
    offset: int  # offset in file where tensor data starts
    tensor_type: str = ''  # human-readable type name

    @property
    def dtype_name(self) -> str:
        return self.tensor_type or QUANT_TYPES.get(self.dtype, ('UNKNOWN', 32, 2))[0]

    def __repr__(self):
        return f"TensorInfo({self.name}, shape={self.shape}, dtype={self.dtype}({self.dtype_name}))"


def _get_field_uint32(reader, key: str, default: int = 0) -> int:
    """Get uint32 field from gguf reader."""
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        part = f.parts[-1]
        if hasattr(part, 'tobytes'):
            val_bytes = part.tobytes()
        else:
            val_bytes = bytes(part)
        if len(val_bytes) == 4:
            return int(np.frombuffer(val_bytes, dtype='<u4')[0])
        elif len(val_bytes) == 8:
            return int(np.frombuffer(val_bytes, dtype='<u8')[0])
        return default
    except:
        return default


def _get_arch(reader) -> str:
    """Get architecture name."""
    f = reader.fields.get('general.architecture')
    if f is None:
        return 'qwen2'
    try:
        for part in reversed(f.parts):
            if hasattr(part, 'dtype') and part.dtype == np.uint8:
                return bytes(part).decode('utf-8', errors='replace').strip()
        for part in f.parts:
            if hasattr(part, 'dtype') and part.dtype == np.uint8 and len(part) > 2:
                return bytes(part).decode('utf-8', errors='replace').strip()
    except:
        pass
    return 'qwen2'


class GGUFReader:
    """Wrapper around gguf.GGUFReader with our interface."""

    def __init__(self, path: str):
        self._reader = gguf_lib.GGUFReader(path)
        self.meta, self.tensors = self._parse()

    def _parse(self):
        reader = self._reader

        # Build metadata
        meta = GGUFMetadata(
            version=_get_field_uint32(reader, 'GGUF.version', 3),
            tensor_count=_get_field_uint32(reader, 'GGUF.tensor_count', 339),
            metadata_len=_get_field_uint32(reader, 'GGUF.kv_count', 0),
            alignment=reader.alignment,
            arch=_get_arch(reader),
            hidden_size=_get_field_uint32(reader, 'qwen2.embedding_length', 1536),
            intermediate_size=_get_field_uint32(reader, 'qwen2.feed_forward_length', 8960),
            num_hidden_layers=_get_field_uint32(reader, 'qwen2.block_count', 28),
            num_attention_heads=_get_field_uint32(reader, 'qwen2.attention.head_count', 12),
            num_key_value_heads=_get_field_uint32(reader, 'qwen2.attention.head_count_kv', 2),
            vocab_size=_get_field_uint32(reader, 'tokenizer.ggml.tokens', 151936),
            context_length=_get_field_uint32(reader, 'qwen2.context_length', 32768),
            bos_token_id=_get_field_uint32(reader, 'tokenizer.ggml.bos_token_id', 151643),
            eos_token_id=_get_field_uint32(reader, 'tokenizer.ggml.eos_token_id', 151645),
        )

        # Build tensor infos
        tensors = []
        for t in reader.tensors:
            shape = [int(d) for d in t.shape]
            n_elements = 1
            for dim in shape:
                n_elements *= dim

            dtype = int(t.tensor_type)
            qt = QUANT_TYPES.get(dtype, ('UNKNOWN', 32, 2))
            type_name, block_size, _ = qt

            # Rough byte size estimate for quantized tensors
            if dtype in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
                bytes_size = (n_elements // block_size + 1) * 16
            elif dtype in (12, 13):
                bytes_size = n_elements * 2
            elif dtype == 14:
                bytes_size = n_elements * 4
            elif dtype == 15:
                bytes_size = n_elements * 8
            else:
                bytes_size = n_elements * 4

            tensors.append(TensorInfo(
                name=t.name,
                shape=shape,
                dtype=dtype,
                n_elements=n_elements,
                bytes_size=bytes_size,
                offset=t.data_offset,
                tensor_type=type_name,
            ))

        return meta, tensors

    @property
    def data_offset(self):
        return self._reader.data_offset

    @property
    def alignment(self):
        return self._reader.alignment

    def __getitem__(self, key):
        return self.tensors[key]


def parse_gguf(path: str) -> tuple:
    """Parse GGUF file. Returns (meta, tensors)."""
    reader = GGUFReader(path)
    return reader.meta, reader.tensors