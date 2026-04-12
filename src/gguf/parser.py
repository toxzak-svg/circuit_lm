"""
GGUF Binary Parser

Parses GGUF format files ( llama.cpp's model format ) and extracts:
- Metadata (version, tensor count, etc.)
- Tensor information (name, shape, dtype, offsets)
- Quantized tensor data

Based on llama.cpp GGUF format specification:
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

import struct
import mmap
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json


# GGUF Magic: "GGUF" in ASCII
GGUF_MAGIC = 0x47475546
GGUF_SUPPORTED_VERSIONS = [1, 2, 3]

# Quantization types (from ggml/ggml.h)
@dataclass
class QuantType:
    name: str
    block_size: int  # elements per block
    bytes_per_block: int
    type_id: int

QUANT_TYPES: Dict[int, QuantType] = {
    0: QuantType("F32", 1, 4, 0),        # float32
    1: QuantType("F16", 1, 2, 1),        # float16
    2: QuantType("Q4_0", 32, 18, 2),    # 4-bit, quantized, 0.5 bit/element + 4B scale
    3: QuantType("Q4_1", 32, 20, 3),    # 4-bit, quantized, 0.5 bit/elem + 4B scale + 4B offset
    4: QuantType("Q5_0", 32, 22, 4),    # 5-bit, quantized
    5: QuantType("Q5_1", 32, 24, 5),    # 5-bit, quantized with scale+offset
    6: QuantType("Q8_0", 32, 36, 6),    # 8-bit, quantized
    7: QuantType("Q8_1", 32, 40, 7),    # 8-bit with scale+offset
    8: QuantType("Q2_K", 256, 256/16 + 2 + 256/64, 8),  # Q2_K - mixed
    9: QuantType("Q3_K", 256, 256/8 + 3 + 256/64, 9),   # Q3_K - mixed
    10: QuantType("Q4_K", 256, 256/2 + 2 + 12 + 256/64, 10),  # Q4_K - mixed
    11: QuantType("Q5_K", 256, 256/2 + 3 + 12 + 256/64, 11),  # Q5_K - mixed
    12: QuantType("Q6_K", 256, 256/2 + 256/4 + 2, 12),  # Q6_K - mixed
    13: QuantType("Q8_0_VEC", 32, 34, 13),  # Q8_0 with F16 scales
    14: QuantType("IQ4_NL", 32, 18, 14),  # Incremental 4-bit with NL scale
    15: QuantType("IQ5_NL", 32, 22, 15),  # Incremental 5-bit NL
    16: QuantType("IQ3_NL", 32, 12, 16),  # Incremental 3-bit NL
    17: QuantType("IQ2_NL", 32, 8, 17),  # Incremental 2-bit NL
    18: QuantType("IQ4_XS", 32, 14, 18), # Incremental 4-bit with extended scale
    19: QuantType("IQ3_S", 32, 12, 19), # Incremental 3-bit with scale
    20: QuantType("IQ2_S", 32, 8, 20),  # Incremental 2-bit with scale
    21: QuantType("IQ4_SH", 32, 14, 21), # Incremental 4-bit with extended scale (Hermite)
    22: QuantType("IQ3_SH", 32, 12, 22), # Incremental 3-bit with scale (Hermite)
    23: QuantType("IQ2_SH", 32, 8, 23),  # Incremental 2-bit with scale (Hermite)
    24: QuantType("IQ8_NL", 32, 36, 24), # Incremental 8-bit NL
    25: QuantType("Q8_1", 32, 40, 25),   # Same as type 7
    26: QuantType("Q4_NL", 32, 18, 26),  # Q4 with NL scale
    27: QuantType("Q5_NL", 32, 22, 27),  # Q5 with NL scale
    28: QuantType("Q2_K_S", 256, 256/16 + 2 + 256/64, 28), # Q2_K with F16 scales
    29: QuantType("Q3_K_S", 256, 256/8 + 3 + 256/64, 29),   # Q3_K with F16 scales
    30: QuantType("Q4_K_S", 256, 256/2 + 2 + 12 + 256/64, 30),  # Q4_K with F16 scales
    31: QuantType("Q5_K_S", 256, 256/2 + 3 + 12 + 256/64, 31),  # Q5_K with F16 scales
}


@dataclass
class TensorInfo:
    """Information about a single tensor in the GGUF file."""
    name: str
    n_dims: int  # number of dimensions (1-4)
    shape: Tuple[int, ...]  # e.g. (4096, 4096) for a weight matrix
    dtype: int  # quantization type ID
    offset: int  # byte offset in file where tensor data starts
    n_elements: int  # total number of elements
    bytes_size: int  # total bytes for this tensor

    @property
    def dtype_name(self) -> str:
        return QUANT_TYPES.get(self.dtype, QuantType("UNKNOWN", 1, 1, self.dtype)).name

    def __repr__(self):
        return (f"TensorInfo(name={self.name!r}, shape={self.shape}, "
                f"dtype={self.dtype_name}, offset={self.offset}, "
                f"n_elements={self.n_elements}, bytes_size={self.bytes_size})")


@dataclass
class GGUFMetadata:
    """Metadata extracted from the GGUF header and kv data."""
    version: int
    tensor_count: int
    metadata_len: int
    alignment: int
    # KV pairs
    arch: Optional[str] = None
    general_name: Optional[str] = None
    general_quant_version: Optional[int] = None
    general_file_type: Optional[int] = None
    # Model params
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    context_length: Optional[int] = None
    # Tokenizer
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)


class GGUFReader:
    """Reads GGUF binary files and extracts tensor information and data."""

    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.metadata: Optional[GGUFMetadata] = None
        self.tensors: List[TensorInfo] = []

    def read_uint32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.mm, offset)[0]

    def read_uint64(self, offset: int) -> int:
        return struct.unpack_from("<Q", self.mm, offset)[0]

    def read_float32(self, offset: int) -> float:
        return struct.unpack_from("<f", self.mm, offset)[0]

    def read_string(self, offset: int) -> Tuple[str, int]:
        """Read a length-prefixed string. Returns (string, bytes_read)."""
        length = self.read_uint64(offset)
        if length > 8192:
            raise ValueError(f"String too long: {length}")
        # Strings are 8-byte aligned
        str_offset = offset + 8
        data = self.mm[str_offset:str_offset + length]
        return data.decode("utf-8", errors="replace"), 8 + length

    def read_tensor_info(self, offset: int) -> Tuple[TensorInfo, int]:
        """Read one tensor info entry. Returns (TensorInfo, bytes_consumed)."""
        start = offset

        # Read name
        name, consumed = self.read_string(offset)
        offset += consumed

        # Read n_dims
        n_dims = self.read_uint32(offset)
        offset += 4

        # Read shape (n_dims uint32s)
        shape = tuple(self.read_uint32(offset + i * 4) for i in range(n_dims))
        offset += n_dims * 4

        # Read dtype
        dtype = self.read_uint32(offset)
        offset += 4

        # Read offset (byte offset in file where tensor data starts)
        tensor_offset = self.read_uint64(offset)
        offset += 8

        # Calculate n_elements and bytes_size
        n_elements = 1
        for dim in shape:
            n_elements *= dim

        qt = QUANT_TYPES.get(dtype)
        if qt is None:
            raise ValueError(f"Unknown dtype {dtype}")
        bytes_size = qt.bytes_per_block * (n_elements // qt.block_size)
        if n_elements % qt.block_size != 0:
            bytes_size += qt.bytes_per_block

        tensor = TensorInfo(
            name=name,
            n_dims=n_dims,
            shape=shape,
            dtype=dtype,
            offset=tensor_offset,
            n_elements=n_elements,
            bytes_size=bytes_size,
        )
        return tensor, offset - start

    def read_metadata_value(self, offset: int) -> Tuple[Any, int]:
        """Read a metadata key-value pair value. Returns (value, bytes_consumed)."""
        value_type = self.read_uint32(offset)
        offset += 4

        if value_type == 0:  # UINT32
            return self.read_uint32(offset), 8
        elif value_type == 1:  # INT32
            return struct.unpack_from("<i", self.mm, offset)[0], 8
        elif value_type == 2:  # FLOAT32
            return self.read_float32(offset), 8
        elif value_type == 3:  # BOOL
            return self.read_uint32(offset) != 0, 8
        elif value_type == 4:  # STRING
            result, consumed = self.read_string(offset)
            return result, consumed + 4
        elif value_type == 5:  # ARRAY
            # Read array type
            arr_type = self.read_uint32(offset)
            offset += 4
            arr_len = self.read_uint64(offset)
            offset += 8
            # Read array elements (simplified - just read as uint32 for now)
            elems = []
            for _ in range(arr_len):
                elem, _ = self.read_metadata_value(offset - 4)  # back up to type
                elems.append(elem)
                offset += 4
            return elems, offset - start
        elif value_type == 6:  # UINT64
            return self.read_uint64(offset), 12
        else:
            raise ValueError(f"Unknown metadata type: {value_type}")

    def parse_header(self) -> GGUFMetadata:
        """Parse the GGUF header."""
        # Magic (4 bytes)
        magic = self.read_uint32(0)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}, expected {hex(GGUF_MAGIC)}")

        # Version (4 bytes)
        version = self.read_uint32(4)
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Tensor count (8 bytes)
        tensor_count = self.read_uint64(8)

        # Metadata length (8 bytes)
        metadata_len = self.read_uint64(16)

        # Alignment (4 bytes, v3+)
        alignment = 32
        if version >= 3:
            alignment = self.read_uint32(24)

        meta = GGUFMetadata(
            version=version,
            tensor_count=tensor_count,
            metadata_len=metadata_len,
            alignment=alignment,
        )

        return meta

    def parse_kv_data(self, meta: GGUFMetadata) -> int:
        """Parse key-value metadata. Returns bytes consumed."""
        # KV data starts after header (32 bytes)
        kv_start = 32
        kv_end = meta.metadata_len
        offset = kv_start

        known_keys = {
            "general.architecture": "arch",
            "general.name": "general_name",
            "general.quantization_version": "general_quant_version",
            "general.file_type": "general_file_type",
            "{arch}.hidden_size": "hidden_size",
            "{arch}.intermediate_size": "intermediate_size",
            "{arch}.num_hidden_layers": "num_hidden_layers",
            "{arch}.num_attention_heads": "num_attention_heads",
            "{arch}.num.key_value_heads": "num_key_value_heads",
            "{arch}.vocab_size": "vocab_size",
            "{arch}.context_length": "context_length",
            "tokenizer.bos_token_id": "bos_token_id",
            "tokenizer.eos_token_id": "eos_token_id",
            "tokenizer.pad_token_id": "pad_token_id",
        }

        while offset < kv_end:
            # Read key
            key, consumed = self.read_string(offset)
            offset += consumed

            # Read value
            value, consumed = self.read_metadata_value(offset)
            offset += consumed

            # Map to known fields
            if key in known_keys:
                field_name = known_keys[key]
                setattr(meta, field_name, value)
            else:
                meta.extra[key] = value

        return offset - kv_start

    def parse(self) -> Tuple[GGUFMetadata, List[TensorInfo]]:
        """Parse the full GGUF file."""
        # Parse header
        meta = self.parse_header()

        # Parse KV data
        self.parse_kv_data(meta)

        # Parse tensor infos (they come after metadata)
        tensor_info_start = 32 + meta.metadata_len
        # Align to 32 bytes (or alignment value)
        align = meta.alignment
        tensor_info_start = (tensor_info_start + align - 1) // align * align

        offset = tensor_info_start
        for _ in range(meta.tensor_count):
            tensor, consumed = self.read_tensor_info(offset)
            self.tensors.append(tensor)
            offset += consumed

        self.metadata = meta
        return meta, self.tensors

    def read_tensor_data(self, tensor: TensorInfo) -> np.ndarray:
        """Read raw bytes for a tensor. Does NOT dequantize."""
        self.file.seek(tensor.offset)
        data = self.file.read(tensor.bytes_size)
        return np.frombuffer(data, dtype=np.uint8)

    def close(self):
        self.mm.close()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def parse_gguf(path: Path) -> Tuple[GGUFMetadata, List[TensorInfo]]:
    """Convenience function to parse a GGUF file."""
    with GGUFReader(path) as reader:
        meta, tensors = reader.parse()
        return meta, tensors


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to.gguf>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Parsing GGUF: {path}")
    print("=" * 60)

    meta, tensors = parse_gguf(path)

    print(f"Version: {meta.version}")
    print(f"Tensors: {meta.tensor_count}")
    print(f"Metadata len: {meta.metadata_len}")
    print(f"Alignment: {meta.alignment}")
    print()
    print("Model Parameters:")
    print(f"  Architecture: {meta.arch}")
    print(f"  Hidden size: {meta.hidden_size}")
    print(f"  Intermediate size: {meta.intermediate_size}")
    print(f"  Layers: {meta.num_hidden_layers}")
    print(f"  Heads: {meta.num_attention_heads}")
    print(f"  KV heads: {meta.num_key_value_heads}")
    print(f"  Vocab size: {meta.vocab_size}")
    print(f"  Context: {meta.context_length}")
    print()
    print(f"Tokenizer:")
    print(f"  BOS token: {meta.bos_token_id}")
    print(f"  EOS token: {meta.eos_token_id}")
    print(f"  PAD token: {meta.pad_token_id}")
    print()
    print(f"Tensors ({len(tensors)}):")
    print("-" * 60)

    total_bytes = 0
    for i, t in enumerate(tensors):
        print(f"  [{i:3d}] {t.name}")
        print(f"         shape={t.shape}, dtype={t.dtype_name}, "
              f"offset={t.offset}, {t.n_elements:,} elem, {t.bytes_size:,} bytes")
        total_bytes += t.bytes_size

    print("-" * 60)
    print(f"Total tensor data: {total_bytes / 1024 / 1024:.2f} MB")

    # Group by dtype
    dtype_counts: Dict[str, int] = {}
    for t in tensors:
        name = t.dtype_name
        dtype_counts[name] = dtype_counts.get(name, 0) + 1
    print()
    print("Tensors by dtype:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"  {dtype}: {count}")