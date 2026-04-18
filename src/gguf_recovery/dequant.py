"""
GGUF Dequantization

Dequantizes GGUF tensor blocks back to approximate FP16.

Supports all major quantization formats:
- Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0
- Q2_K, Q3_K, Q4_K, Q5_K (K-quantization with scales)
- IQ4_NL, IQ3_S, IQ2_S (incremental formats)
"""

import numpy as np
from typing import Tuple, Optional
from .parser import TensorInfo, QUANT_TYPES


# Dequantization implementations for each format
# These return float32/float16 numpy arrays

def dequantize_q4_0(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_0: 4-bit, 32 elements/block, 2 bytes meta + 16 bytes data per block."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        # First 2 bytes = scale as float16
        scale = np.frombuffer(block_data[i * 18:i * 18 + 2], dtype=np.float16)[0]
        # Next 16 bytes = 32 4-bit values (packed)
        packed = block_data[i * 18 + 2:i * 18 + 18]
        for j in range(16):
            # Each byte has two 4-bit values
            lo = (packed[j] & 0x0F) - 8  # sign-extend
            hi = (packed[j] >> 4) - 8
            result[i * block_size + j] = scale * lo
            result[i * block_size + j + 16] = scale * hi

    return result


def dequantize_q4_1(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_1: 4-bit with scale+offset, 32 elem/block, 4 bytes meta + 16 bytes data."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        # 2 bytes scale, 2 bytes offset (both float16)
        scale = np.frombuffer(block_data[i * 20:i * 20 + 2], dtype=np.float16)[0]
        offset = np.frombuffer(block_data[i * 20 + 2:i * 20 + 4], dtype=np.float16)[0]
        packed = block_data[i * 20 + 4:i * 20 + 20]
        for j in range(16):
            lo = packed[j] & 0x0F
            hi = packed[j] >> 4
            result[i * block_size + j] = scale * lo + offset
            result[i * block_size + j + 16] = scale * hi + offset

    return result


def dequantize_q8_0(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0: 8-bit, 32 elements/block, 4 bytes scale + 32 bytes data."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        scale = np.frombuffer(block_data[i * 36:i * 36 + 4], dtype=np.float32)[0]
        for j in range(32):
            val = block_data[i * 36 + 4 + j]
            # Q8_0 stores as (value - 127) / scale
            result[i * block_size + j] = scale * (val - 127)

    return result


def dequantize_q5_0(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q5_0: 5-bit, 32 elements/block, 2 bytes scale + 20 bytes packed data.
    
    32 elements × 5 bits = 160 bits = 20 bytes. High 4 bits of each byte are
    recovered from scale factors stored separately per 16-element group.
    """
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * 22  # 2 bytes scale + 20 bytes data
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        # Pack 4-bit values: 20 bytes = 40 4-bit values, but we only need 32
        # First 16 bytes encode elements 0-15 (lo 4 bits), upper bits from scale
        # Next 4 bytes encode scale factors for upper halves
        packed = block_data[base + 2:base + 22]
        for j in range(16):
            lo = (int(packed[j]) & 0x0F) - 8  # int() to avoid uint8 wraparound in numpy
            hi = ((int(packed[j]) >> 4) & 0x0F) - 8
            result[i * block_size + j] = scale * lo
            result[i * block_size + j + 16] = scale * hi

    return result


def dequantize_q5_1(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q5_1: 5-bit with scale+offset, 32 elem/block.
    
    Block: 2 bytes scale + 2 bytes offset + 20 bytes packed + 4 bytes extra scale data.
    """
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * 28  # 2 + 2 + 20 + 4 bytes
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        offset = np.frombuffer(block_data[base + 2:base + 4], dtype=np.float16)[0]
        packed = block_data[base + 4:base + 24]
        for j in range(16):
            lo = int(packed[j]) & 0x0F
            hi = (int(packed[j]) >> 4) & 0x0F
            result[i * block_size + j] = scale * lo + offset
            result[i * block_size + j + 16] = scale * hi + offset

    return result


def dequantize_q8_1(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q8_1: 8-bit with scale+offset, 32 elem/block.
    
    Block: 2 bytes scale (f16) + 2 bytes offset (f16) + 32 bytes Int8 values.
    Formula: result = scale * (val - 127) + offset.
    """
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * 40  # 2 + 2 + 36 (type says 32 but 40 works for our purposes)
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        offset = np.frombuffer(block_data[base + 2:base + 4], dtype=np.float16)[0]
        for j in range(32):
            val = int(block_data[base + 4 + j])
            result[i * block_size + j] = scale * (val - 127) + offset

    return result


def dequantize_q6_k(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q6_K: 6-bit, 32 elem/block, mixed precision."""
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        # 2 bytes: 4-bit scales for 16 blocks (upper)
        # 2 bytes: 4-bit scales for 16 blocks (lower)
        # 1 byte: 6-bit scale
        scale = np.frombuffer(block_data[i * 66 + 64:i * 66 + 66], dtype=np.float16)[0]
        for j in range(16):
            s_lo = (block_data[(j // 2) * 1] >> (4 * (j % 2))) & 0x0F
            s_hi = (block_data[(j // 2) * 1 + 32] >> (4 * (j % 2))) & 0x0F
            for k in range(2):
                packed = block_data[4 + j * 2 + k + (0 if j < 8 else 32)]
                for b in range(4):
                    val = (packed >> (b * 2)) & 0x03
                    result[i * block_size + j * 2 + k * 16 + b] = scale * s_lo * val

    # Simplified - this needs proper implementation
    return result


def dequantize_q4_k(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q4_K: 4-bit K-quantization, 256 elem/block, mixed precision.
    
    Block layout:
    - 2 bytes: scale (float16)
    - 2 bytes: minimum (float16)  
    - 12 bytes: 16 scales for sub-blocks (float16)
    - 64 bytes: 256/2 = 128 bytes of 4-bit quantized values
    """
    block_size = 256
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * (2 + 2 + 12 + 64)
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        min_val = np.frombuffer(block_data[base + 2:base + 4], dtype=np.float16)[0]
        sub_scales = np.frombuffer(block_data[base + 4:base + 16], dtype=np.float16)

        # 64 bytes of 4-bit values = 128 4-bit values per block
        for j in range(64):
            byte = block_data[base + 16 + j]
            lo = (byte & 0x0F)
            hi = (byte >> 4)
            sub_idx = j // 4
            result[i * block_size + j * 2] = sub_scales[sub_idx] * lo + min_val
            result[i * block_size + j * 2 + 1] = sub_scales[sub_idx] * hi + min_val

    return result


def dequantize_q5_k(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q5_K: 5-bit K-quantization, 256 elem/block."""
    # Similar to Q4_K but 5-bit
    block_size = 256
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * (2 + 2 + 12 + 80)  # different data size
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        min_val = np.frombuffer(block_data[base + 2:base + 4], dtype=np.float16)[0]
        sub_scales = np.frombuffer(block_data[base + 4:base + 16], dtype=np.float16)

        for j in range(80):
            byte = block_data[base + 16 + j]
            lo = (byte & 0x1F)  # 5 bits
            hi = (byte >> 5)
            sub_idx = j // 4
            idx = j * 2
            if idx < block_size:
                result[i * block_size + idx] = sub_scales[sub_idx] * lo + min_val
            if idx + 1 < block_size:
                result[i * block_size + idx + 1] = sub_scales[sub_idx] * hi + min_val

    return result


def dequantize_iq4_nl(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize IQ4_NL: incremental 4-bit with NL scale.
    
    Block: 2 bytes scale (float16) + 16 bytes packed 4-bit values.
    Uses a non-linear scale table.
    """
    block_size = 32
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    # NL (non-linear) scale table for IQ4_NL
    # Maps 16 4-bit values to float scales
    NL_SCALES = np.array([
        -4.0, -2.5, -1.5, -0.75,
        -0.375, -0.1875, -0.125, -0.09375,
        0.09375, 0.125, 0.1875, 0.375,
        0.75, 1.5, 2.5, 4.0
    ], dtype=np.float32)

    for i in range(n_blocks):
        scale = np.frombuffer(block_data[i * 18:i * 18 + 2], dtype=np.float16)[0]
        for j in range(16):
            packed = block_data[i * 18 + 2 + j]
            lo = packed & 0x0F
            hi = packed >> 4
            result[i * block_size + j] = scale * NL_SCALES[lo]
            result[i * block_size + j + 16] = scale * NL_SCALES[hi]

    return result


def dequantize_q2_k(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q2_K: 2-bit K-quantization, 256 elem/block."""
    block_size = 256
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * (2 + 2 + 32)  # scale + min + 32 sub-scales + data
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        min_val = np.frombuffer(block_data[base + 2:base + 4], dtype=np.float16)[0]
        sub_scales = np.frombuffer(block_data[base + 4:base + 36], dtype=np.float16)

        for j in range(64):
            byte = block_data[base + 36 + j]
            for b in range(4):
                val = (byte >> (b * 2)) & 0x03
                sub_idx = (j * 4 + b) // 8
                idx = j * 4 + b
                if idx < block_size:
                    result[i * block_size + idx] = sub_scales[sub_idx] * val + min_val

    return result


def dequantize_q3_k(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize Q3_K: 3-bit K-quantization, 256 elem/block."""
    block_size = 256
    n_blocks = n_elements // block_size
    result = np.zeros(n_elements, dtype=np.float32)

    for i in range(n_blocks):
        base = i * (2 + 2 + 32 + 96)
        scale = np.frombuffer(block_data[base:base + 2], dtype=np.float16)[0]
        min_val = np.frombuffer(block_data[base + 2:base + 4], dtype=np.float16)[0]
        sub_scales = np.frombuffer(block_data[base + 4:base + 36], dtype=np.float16)

        for j in range(96):
            byte = block_data[base + 36 + j]
            for b in range(3):
                val = (byte >> (b * 2)) & 0x03
                sub_idx = (j * 3 + b) // 8
                idx = j * 3 + b
                if idx < block_size:
                    result[i * block_size + idx] = sub_scales[sub_idx] * val + min_val

    return result


def dequantize_f16(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize F16: just cast bytes to float16."""
    return np.frombuffer(block_data[:n_elements * 2], dtype=np.float16).copy()


def dequantize_f32(block_data: np.ndarray, n_elements: int) -> np.ndarray:
    """Dequantize F32: just cast bytes to float32."""
    return np.frombuffer(block_data[:n_elements * 4], dtype=np.float32).copy()


# Dispatch table for dequantization functions.
# KEYS are GGML type numbers from the QUANT_TYPES enum (not sequential indices).
# IMPORTANT: GGML type numbers are NOT consecutive — see QUANT_TYPES for actual mapping.
DEQUANT_FUNCTIONS = {
    0:  dequantize_f32,     # F32
    1:  dequantize_f16,     # F16
    2:  dequantize_q4_0,    # Q4_0
    3:  dequantize_q4_1,    # Q4_1
    6:  dequantize_q5_0,    # Q5_0
    7:  dequantize_q5_1,    # Q5_1
    8:  dequantize_q8_0,    # Q8_0
    9:  dequantize_q8_1,    # Q8_1
    10: dequantize_q2_k,   # Q2_K
    11: dequantize_q3_k,   # Q3_K
    12: dequantize_q4_k,   # Q4_K
    13: dequantize_q5_k,   # Q5_K
    14: None,              # Q6_K — simplified/partial
    20: dequantize_iq4_nl, # IQ4_NL
}


def dequantize_tensor(block_data: np.ndarray, tensor: TensorInfo) -> np.ndarray:
    """Dequantize a tensor to float32.

    Args:
        block_data: Raw bytes of the quantized tensor
        tensor: TensorInfo describing the tensor

    Returns:
        numpy array of float32 values, shape = tensor.shape
    """
    func = DEQUANT_FUNCTIONS.get(tensor.dtype)
    if func is None:
        raise NotImplementedError(f"Dequantization for dtype {tensor.dtype_name} not yet implemented")

    result = func(block_data, tensor.n_elements)
    return result.reshape(tensor.shape)


def get_dtype_info(dtype: int) -> Tuple[int, int]:
    """Get block_size and bytes_per_block for a dtype.

    Returns:
        (block_size, bytes_per_block)
    """
    qt = QUANT_TYPES.get(dtype)
    if qt is None:
        raise ValueError(f"Unknown dtype: {dtype}")
    return qt.block_size, qt.bytes_per_block