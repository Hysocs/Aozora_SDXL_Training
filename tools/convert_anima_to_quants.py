"""
GUI-only converter for Anima/DiT safetensors checkpoints to ComfyUI quantized weights.

This is a post-training inference checkpoint converter launched through its Tk GUI. It can write ComfyUI
scaled FP8, tensorwise INT8, NVFP4, or dynamic mixed-precision checkpoints.

Simple mode applies one selected quant format to every compatible matrix weight,
excluding protected tensors. Dynamic mode is compression-first: choose the lowest
allowed quant tier, then it tries to use that tier on as many compatible Linear
layers as possible, promoting only high-risk layers to 8-bit or keep-dtype.
It does not create MXFP8 block-scaled checkpoints.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import struct
import sys
import time
import uuid
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from safetensors import safe_open


FP8_DTYPES = {
    "e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
    "int8": "int8_tensorwise",
    "nvfp4": "nvfp4",
}

QUANT_STORAGE_DTYPES = {
    "int8_tensorwise": torch.int8,
    "nvfp4": torch.uint8,
}

KEEP_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "same": None,
}

SAFETENSORS_DTYPE_NAMES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _name, _code in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e4m3fnuz", "F8_E4M3FNUZ"),
    ("float8_e5m2", "F8_E5M2"),
    ("float8_e5m2fnuz", "F8_E5M2FNUZ"),
    ("uint64", "U64"),
    ("uint32", "U32"),
    ("uint16", "U16"),
):
    _dtype = getattr(torch, _name, None)
    if _dtype is not None:
        SAFETENSORS_DTYPE_NAMES[_dtype] = _code


TARGET_HINTS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "output_proj.weight",
    "to_q.weight",
    "to_k.weight",
    "to_v.weight",
    "to_out",
    "mlp.0.weight",
    "mlp.2.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
    "linear",
    "proj",
    "ffn",
)

PROTECTED_HINTS = (
    ".bias",
    "bias",
    "norm",
    "ln",
    "embed",
    "embedding",
    "patch_embed",
    "pos_embed",
    "t_embedder",
    "time_embed",
    "final_layer",
    "adaln",
)


DEFAULT_NVFP4_SCALE_MULTIPLIERS = "0.50,0.60,0.70,0.80,0.90,1.00,1.10,1.25,1.50,1.75,2.00,2.50,3.00"


def normalized_mode(mode: str) -> str:
    mode = (mode or "simple").lower().replace("-", "_").replace(" ", "_")
    if mode == "fast":
        return "simple"
    if mode in {"fast_dynamic", "quick_dynamic", "local_dynamic"}:
        return "fast_dynamic"
    if mode in {"compression_first", "compress_first", "compression"}:
        return "compression_first"
    if mode in {"mixed", "calibrated"}:
        return "dynamic"
    if mode in {"simple", "dynamic", "fast_dynamic", "compression_first"}:
        return mode
    raise ValueError(f"Unsupported mode: {mode}")


def dtype_name(dtype: torch.dtype) -> str:
    name = SAFETENSORS_DTYPE_NAMES.get(dtype)
    if name is None:
        raise TypeError(f"Unsupported safetensors dtype: {dtype}")
    return name


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def pattern_matches(patterns: list[re.Pattern[str]], key: str) -> bool:
    return any(p.search(key) for p in patterns)


def quant_format_name(choice: str) -> str:
    return FP8_DTYPES[choice]


def quant_storage_dtype(format_name: str) -> torch.dtype:
    dtype = QUANT_STORAGE_DTYPES.get(format_name)
    if dtype is not None:
        return dtype
    dtype = getattr(torch, format_name, None)
    if dtype is None:
        raise RuntimeError(f"This PyTorch build does not expose torch.{format_name}.")
    return dtype


def quant_label(choice: str) -> str:
    if choice == "int8":
        return "INT8"
    if choice == "nvfp4":
        return "NVFP4"
    return "FP8"


def is_protected_name(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in PROTECTED_HINTS)


def is_full_quant_compatible_name(key: str) -> bool:
    k = key.lower()
    blocked = (
        "norm",
        "embed",
        "embedding",
        "patch_embed",
        "pos_embed",
        "t_embedder",
        "time_embed",
        "final_layer",
        ".bias",
    )
    return not any(h in k for h in blocked)


def is_target_name(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in TARGET_HINTS)

def _block_index_from_key(key: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:blocks|double_blocks|single_blocks)\.(\d+)\.", key.lower())
    return int(match.group(1)) if match else None


def is_composition_sensitive_weight(key: str, total_blocks: int | None = None) -> bool:
    """Layers where 4-bit drift often shows up as composition/layout damage."""
    k = key.lower()
    sensitive_hints = (
        "cross_attn",
        "cross_attention",
        "attn2",
        "context",
        "q_proj.weight",
        "k_proj.weight",
        "to_q.weight",
        "to_k.weight",
        "adaln",
        "modulation",
        "time_embed",
        "t_embedder",
        "final_layer",
        "patch_embed",
        "pos_embed",
    )
    if any(h in k for h in sensitive_hints):
        return True
    block_index = _block_index_from_key(k)
    if block_index is not None:
        if block_index <= 1:
            return True
        if total_blocks is not None and block_index >= max(total_blocks - 2, 0):
            return True
    return False


def choose_save_dtype(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    keep_dtype: torch.dtype | None,
    preset: str,
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
    min_elements: int,
) -> tuple[torch.dtype, str]:
    if not torch.is_floating_point(tensor):
        return tensor.dtype, "non_float"
    if pattern_matches(exclude_patterns, key):
        return keep_dtype or tensor.dtype, "excluded"
    if pattern_matches(include_patterns, key):
        return fp8_dtype, "included"

    if preset == "all" and tensor.ndim >= 2 and is_full_quant_compatible_name(key):
        return fp8_dtype, "all"

    if tensor.numel() < min_elements:
        return keep_dtype or tensor.dtype, "small"

    if preset == "broad":
        if tensor.ndim >= 2 and not is_protected_name(key):
            return fp8_dtype, "broad"
        return keep_dtype or tensor.dtype, "protected"

    if tensor.ndim >= 2 and is_target_name(key) and not is_protected_name(key):
        return fp8_dtype, "target"
    return keep_dtype or tensor.dtype, "kept"


def should_prune_tensor(key: str, tensor: torch.Tensor, args: SimpleNamespace) -> bool:
    if not getattr(args, "prune_small", False):
        return False
    max_elements = int(getattr(args, "prune_max_elements", 0) or 0)
    if max_elements <= 0:
        return False
    if not torch.is_floating_point(tensor):
        return False
    if tensor.numel() > max_elements:
        return False
    return not is_protected_name(key)


def calibrated_key_matches(key: str, quantize_keys: set[str]) -> bool:
    if key in quantize_keys:
        return True
    return any(key.endswith(f".{candidate}") for candidate in quantize_keys)


def calibrated_format_for_key(key: str, quantize_formats: dict[str, str]) -> str | None:
    if key in quantize_formats:
        return quantize_formats[key]
    for candidate, format_name in quantize_formats.items():
        if key.endswith(f".{candidate}"):
            return format_name
    return None


def normalized_quant_spec(spec) -> dict[str, object]:
    if isinstance(spec, str):
        return {"format": spec}
    if isinstance(spec, dict):
        if "format" not in spec:
            raise ValueError(f"Dynamic quant spec is missing 'format': {spec!r}")
        return dict(spec)
    raise TypeError(f"Unsupported dynamic quant spec: {spec!r}")


def calibrated_spec_for_key(key: str, quantize_specs: dict[str, object]) -> dict[str, object] | None:
    if key in quantize_specs:
        return normalized_quant_spec(quantize_specs[key])
    for candidate, spec in quantize_specs.items():
        if key.endswith(f".{candidate}"):
            return normalized_quant_spec(spec)
    return None


def calibrated_extra_dtype(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    keep_dtype: torch.dtype | None,
    preset: str,
    min_elements: int,
) -> tuple[torch.dtype, str]:
    if not torch.is_floating_point(tensor):
        return tensor.dtype, "kept"
    if preset == "all" and tensor.ndim >= 2 and is_full_quant_compatible_name(key):
        return fp8_dtype, "all_extra"
    if (
        preset == "broad"
        and tensor.ndim >= 2
        and tensor.numel() >= min_elements
        and not is_protected_name(key)
    ):
        return fp8_dtype, "broad_extra"
    return keep_dtype or tensor.dtype, "kept"


def stream_tensor_to_file(handle, tensor: torch.Tensor) -> None:
    tensor.reshape(-1).view(torch.uint8).numpy().tofile(handle)


def comfy_quant_key_for_weight(key: str) -> str:
    return key[:-7] + ".comfy_quant" if key.endswith(".weight") else key + ".comfy_quant"


def comfy_scale_key_for_weight(key: str) -> str:
    return key[:-7] + ".weight_scale" if key.endswith(".weight") else key + "_scale"


def comfy_scale2_key_for_weight(key: str) -> str:
    return key[:-7] + ".weight_scale_2" if key.endswith(".weight") else key + "_scale_2"


def comfy_quant_info_tensor(format_name: str, **metadata) -> torch.Tensor:
    payload = json.dumps({"format": format_name, **metadata}, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor, flatten: bool = False) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    step = blocked.reshape(-1, 32, 16)
    step = step.reshape(-1, 32, 4, 4).transpose(1, 2)
    step = step.reshape(n_row_blocks, n_col_blocks, 128, 4).permute(0, 2, 1, 3)
    unblocked = step.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols].contiguous()


NVFP4_BLOCK_SIZE = 16
NVFP4_F4_E2M1_MAX = 6.0
NVFP4_F8_E4M3_MAX = 448.0
_EBITS_F32 = 8
_MBITS_F32 = 23
_F32_EXP_BIAS = (1 << (_EBITS_F32 - 1)) - 1
E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _n_ones(n: int) -> int:
    return (1 << n) - 1


def _float8_round(x: torch.Tensor) -> torch.Tensor:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn, required for NVFP4 scales.")
    return x.to(fp8_dtype).to(torch.float32)


def f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    if x.dtype != torch.float32:
        raise ValueError("f32_to_floatx_unpacked requires float32 input")
    if 1 + ebits + mbits > 8:
        raise ValueError("sub-byte float must fit in a byte")
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(_MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32, device=x.device).view(torch.float32)

    x_int = x.view(torch.int32)
    sign = x_int & 0x80000000
    x_abs = (x_int ^ sign).view(torch.float32)

    saturate_mask = x_abs >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x_abs < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = (x_abs + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x_abs.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder
    normal_x = normal_x + val_to_add + mant_odd
    normal_x = (normal_x >> (_MBITS_F32 - mbits)).to(torch.uint8)

    out = torch.full_like(x_abs, max_int, dtype=torch.uint8)
    out = torch.where(denormal_mask, denormal_x, out)
    out = torch.where(normal_mask, normal_x, out)
    sign_lp = (sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)).to(torch.uint8) & sign_mask
    return (out | sign_lp).to(torch.uint8)


def pack_uint4(nibbles: torch.Tensor) -> torch.Tensor:
    shape = nibbles.shape
    if shape[-1] % 2 != 0:
        raise ValueError("pack_uint4 requires an even last dimension")
    flat = nibbles.contiguous().view(-1)
    packed = (flat[::2] << 4) | flat[1::2]
    return packed.view(*shape[:-1], shape[-1] // 2)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    shape = packed.shape
    hi = (packed >> 4).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.uint8)
    return torch.stack([hi, lo], dim=-1).view(*shape[:-1], shape[-1] * 2)


def e2m1_to_f32(codes: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return lut[codes.long()]


def is_nvfp4_compatible_weight(key: str, tensor: torch.Tensor) -> bool:
    return (
        key.endswith(".weight")
        and torch.is_floating_point(tensor)
        and tensor.ndim == 2
        and int(tensor.shape[1]) % NVFP4_BLOCK_SIZE == 0
    )


def quantize_nvfp4_tensor(
    weight: torch.Tensor,
    scale_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn, required for NVFP4 scales.")

    scale_multiplier = float(scale_multiplier)
    if not torch.isfinite(torch.tensor(scale_multiplier)) or scale_multiplier <= 0.0:
        raise ValueError(f"NVFP4 scale_multiplier must be a positive finite number, got {scale_multiplier!r}")

    w = weight.detach()
    if w.ndim != 2:
        raise ValueError("NVFP4 export requires a rank-2 weight tensor")
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    if in_f % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"NVFP4 in_features {in_f} must be a multiple of {NVFP4_BLOCK_SIZE}")

    wf = w.to(torch.float32)
    base_per_tensor = wf.abs().amax() / (NVFP4_F8_E4M3_MAX * NVFP4_F4_E2M1_MAX)
    per_tensor = base_per_tensor * scale_multiplier
    per_tensor_div = per_tensor.clamp(min=2.0**-126)

    xb = wf.reshape(out_f, in_f // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_amax = xb.abs().amax(dim=-1)
    block_scale = block_amax / NVFP4_F4_E2M1_MAX
    scaled_fp8 = (block_scale / per_tensor_div).clamp(max=NVFP4_F8_E4M3_MAX)

    total = per_tensor_div * _float8_round(scaled_fp8)
    total_safe = torch.where(total == 0, torch.ones_like(total), total)
    data_scaled = xb / total_safe.unsqueeze(-1)
    data_scaled = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)
    data_scaled = data_scaled.reshape(out_f, in_f).clamp(-NVFP4_F4_E2M1_MAX, NVFP4_F4_E2M1_MAX)

    nibbles = f32_to_floatx_unpacked(data_scaled.contiguous(), 2, 1)
    weight_uint8 = pack_uint4(nibbles)
    weight_scale = to_blocked(scaled_fp8.to(fp8_dtype), flatten=False)
    weight_scale_2 = per_tensor.to(torch.float32)
    return weight_uint8.cpu().contiguous(), weight_scale.cpu().contiguous(), weight_scale_2.cpu().contiguous()


def dequantize_nvfp4_tensor(
    weight_uint8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_f: int,
    in_f: int,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    nibbles = unpack_uint4(weight_uint8.to(device=device))[:, :in_f]
    block_cols = in_f // NVFP4_BLOCK_SIZE
    block_scale = from_blocked(weight_scale.to(device=device), out_f, block_cols).to(torch.float32)
    total = block_scale * weight_scale_2.to(device=device, dtype=torch.float32)
    values = e2m1_to_f32(nibbles).reshape(out_f, block_cols, NVFP4_BLOCK_SIZE)
    return (values * total.unsqueeze(-1)).reshape(out_f, in_f).to(dtype=compute_dtype)


def quantize_ternary_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack per-output-channel absmean ternary weights into four 2-bit codes per byte."""
    w = weight.detach().to(dtype=torch.float32)
    if w.ndim != 2:
        raise ValueError("Ternary 1.58-bit export requires a rank-2 weight tensor")
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1.0e-12)
    ternary = (w / scale).round().clamp(-1, 1).to(dtype=torch.int8)
    codes = (ternary + 1).to(dtype=torch.uint8)
    out_f, in_f = int(codes.shape[0]), int(codes.shape[1])
    padded_in = _ceil_div(in_f, 4) * 4
    if padded_in != in_f:
        padded = torch.ones((out_f, padded_in), dtype=torch.uint8, device=codes.device)
        padded[:, :in_f] = codes
        codes = padded
    codes = codes.reshape(out_f, padded_in // 4, 4)
    packed = (
        codes[:, :, 0]
        | (codes[:, :, 1] << 2)
        | (codes[:, :, 2] << 4)
        | (codes[:, :, 3] << 6)
    )
    return packed.cpu().contiguous(), scale.cpu().contiguous()


def dequantize_ternary_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_f: int,
    in_f: int,
    compute_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    packed = packed.to(device=device, dtype=torch.uint8)
    codes = torch.stack(
        tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)),
        dim=-1,
    ).reshape(out_f, -1)[:, :in_f]
    if torch.any(codes > 2):
        raise ValueError("Invalid reserved 2-bit code in ternary weight tensor")
    values = codes.to(dtype=torch.float32) - 1.0
    row_scale = scale.to(device=device, dtype=torch.float32).reshape(out_f, 1)
    return (values * row_scale).to(dtype=compute_dtype)


def comfy_quant_records_for_weight(
    key: str,
    tensor: torch.Tensor,
    format_name: str,
    storage_dtype: torch.dtype,
    nvfp4_scale_multiplier: float = 1.0,
) -> tuple[tuple[str, torch.Tensor, torch.dtype], ...]:
    if format_name == "ternary_1_58":
        packed, scale = quantize_ternary_tensor(tensor)
        quant_info = comfy_quant_info_tensor(
            format_name,
            logical_bits_per_weight=math.log2(3.0),
            storage_bits_per_weight=2,
            packing="ternary_2bit_four_per_byte",
            original_shape=[int(tensor.shape[0]), int(tensor.shape[1])],
            scale="per_output_absmean",
        )
        return (
            (key, packed, torch.uint8),
            (comfy_scale_key_for_weight(key), scale, torch.float32),
            (comfy_quant_key_for_weight(key), quant_info, torch.uint8),
        )

    if format_name == "nvfp4":
        weight_uint8, weight_scale, weight_scale_2 = quantize_nvfp4_tensor(
            tensor,
            scale_multiplier=nvfp4_scale_multiplier,
        )
        quant_info = comfy_quant_info_tensor(format_name)
        return (
            (key, weight_uint8, torch.uint8),
            (comfy_scale_key_for_weight(key), weight_scale, getattr(torch, "float8_e4m3fn")),
            (comfy_scale2_key_for_weight(key), weight_scale_2, torch.float32),
            (comfy_quant_key_for_weight(key), quant_info, torch.uint8),
        )

    quant_tensor, scale_tensor = scaled_quant_tensor(tensor, storage_dtype, format_name)
    quant_info = comfy_quant_info_tensor(format_name)
    return (
        (key, quant_tensor, storage_dtype),
        (comfy_scale_key_for_weight(key), scale_tensor, torch.float32),
        (comfy_quant_key_for_weight(key), quant_info, torch.uint8),
    )


def format_compatible_with_weight(format_name: str, key: str, tensor: torch.Tensor) -> bool:
    if format_name == "ternary_1_58":
        return key.endswith(".weight") and torch.is_floating_point(tensor) and tensor.ndim == 2
    if format_name == "nvfp4":
        return is_nvfp4_compatible_weight(key, tensor)
    return key.endswith(".weight") and torch.is_floating_point(tensor) and tensor.ndim >= 2


def scaled_quant_tensor(tensor: torch.Tensor, storage_dtype: torch.dtype, format_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if format_name == "int8_tensorwise":
        data = tensor.detach().float()
        if data.ndim >= 2:
            scale = (data.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1.0e-30).to(dtype=torch.float32)
        else:
            scale = (data.abs().max() / 127.0).clamp(min=1.0e-30).reshape(()).to(dtype=torch.float32)
        quantized = (
            (data / scale)
            .round()
            .clamp(-128.0, 127.0)
            .to(dtype=torch.int8)
            .cpu()
            .contiguous()
        )
        return quantized, scale
    max_value = float(tensor.detach().abs().max().float().item()) if tensor.numel() else 0.0
    quant_max = float(torch.finfo(storage_dtype).max)
    scale_value = max(max_value / quant_max, 1.0e-12)
    scale = torch.tensor(scale_value, dtype=torch.float32)
    quantized = (tensor.detach().float() / scale_value).to(dtype=storage_dtype).cpu().contiguous()
    return quantized, scale


def simulated_quant_weight(
    weight: torch.Tensor,
    storage_dtype: torch.dtype,
    format_name: str,
    compute_dtype: torch.dtype,
    nvfp4_scale_multiplier: float = 1.0,
) -> torch.Tensor:
    w = weight.detach().float()
    if format_name == "ternary_1_58":
        packed, scale = quantize_ternary_tensor(weight)
        return dequantize_ternary_tensor(
            packed,
            scale,
            int(weight.shape[0]),
            int(weight.shape[1]),
            compute_dtype,
            weight.device,
        )
    if format_name == "int8_tensorwise":
        if w.ndim >= 2:
            scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1.0e-30)
        else:
            scale = (w.abs().max() / 127.0).clamp(min=1.0e-30)
        quantized = (w / scale).round().clamp(-128.0, 127.0).to(dtype=torch.int8)
        return (quantized.float() * scale).to(dtype=compute_dtype)
    if format_name == "nvfp4":
        q_weight, q_scale, q_scale_2 = quantize_nvfp4_tensor(
            weight,
            scale_multiplier=nvfp4_scale_multiplier,
        )
        return dequantize_nvfp4_tensor(
            q_weight,
            q_scale,
            q_scale_2,
            int(weight.shape[0]),
            int(weight.shape[1]),
            compute_dtype,
            weight.device,
        )
    max_value = w.abs().max() if w.numel() else torch.tensor(0.0, device=w.device)
    scale = (max_value / float(torch.finfo(storage_dtype).max)).clamp(min=1.0e-12)
    quantized = (w / scale).to(dtype=storage_dtype)
    return (quantized.float() * scale).to(dtype=compute_dtype)


def quant_format_sort_key(format_name: str) -> tuple[int, str]:
    # Smaller file format first. INT8 and FP8 are both roughly 1 byte/weight; FP8 usually wins quality.
    order = {
        "ternary_1_58": 0,
        "nvfp4": 1,
        "float8_e4m3fn": 2,
        "float8_e5m2": 3,
        "int8_tensorwise": 4,
    }
    return order.get(format_name, 99), format_name


def parse_mixed_formats(text: str) -> list[str]:
    seen = set()
    out = []
    for item in (text or "").split(","):
        choice = item.strip().lower()
        if not choice:
            continue
        if choice not in FP8_DTYPES:
            raise ValueError(f"Unsupported mixed format: {choice}. Use one of: {', '.join(sorted(FP8_DTYPES))}")
        format_name = quant_format_name(choice)
        if format_name not in seen:
            seen.add(format_name)
            out.append(format_name)
    if not out:
        raise ValueError("--mixed-formats did not contain any valid formats")
    return out


def parse_float_list(text: str, *, name: str = "values") -> list[float]:
    values: list[float] = []
    seen = set()
    for raw in (text or "").split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} entry {item!r}; expected a comma-separated list of numbers.") from exc
        if value <= 0.0 or not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"Invalid {name} entry {item!r}; values must be positive finite numbers.")
        key = round(value, 12)
        if key not in seen:
            seen.add(key)
            values.append(float(value))
    if not values:
        raise ValueError(f"{name} did not contain any valid values")
    return sorted(values)

def write_streaming_safetensors(
    output_path: Path,
    records: list[tuple[str, torch.Tensor, torch.dtype]],
    metadata: dict[str, str],
) -> None:
    tmp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    offset = 0
    for key, tensor, save_dtype in records:
        nbytes = tensor.numel() * dtype_element_size(save_dtype)
        header[key] = {
            "dtype": dtype_name(save_dtype),
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)

    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for _, source_tensor, save_dtype in records:
                if torch.is_floating_point(source_tensor):
                    tensor_cpu = source_tensor.detach().to(device="cpu", dtype=save_dtype).contiguous()
                else:
                    tensor_cpu = source_tensor.detach().cpu().contiguous()
                stream_tensor_to_file(f, tensor_cpu)
                del tensor_cpu
                gc.collect()
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


class LayerProfile:
    def __init__(self, name: str, weight_key: str, elements: int):
        self.name = name
        self.weight_key = weight_key
        self.elements = int(elements)
        self.diff_sq = 0.0
        self.ref_sq = 0.0
        self.quant_sq = 0.0
        self.dot = 0.0
        self.max_abs = 0.0
        self.max_ref_abs = 0.0
        self.count = 0
        self.batches = 0
        self.early_stopped = False

    def update(self, ref: torch.Tensor, quant: torch.Tensor) -> None:
        ref_f = ref.detach().float()
        quant_f = quant.detach().float()
        diff = quant_f - ref_f
        self.diff_sq += float(diff.pow(2).sum().item())
        self.ref_sq += float(ref_f.pow(2).sum().item())
        self.quant_sq += float(quant_f.pow(2).sum().item())
        self.dot += float((ref_f * quant_f).sum().item())
        self.max_abs = max(self.max_abs, float(diff.abs().amax().item()) if diff.numel() else 0.0)
        self.max_ref_abs = max(self.max_ref_abs, float(ref_f.abs().amax().item()) if ref_f.numel() else 0.0)
        self.count += int(ref_f.numel())
        self.batches += 1

    @property
    def rel_mse(self) -> float:
        return self.diff_sq / max(self.ref_sq, 1.0e-12)

    @property
    def cosine(self) -> float:
        denom = (self.ref_sq ** 0.5) * (self.quant_sq ** 0.5)
        return self.dot / max(denom, 1.0e-12)

    @property
    def max_abs_ratio(self) -> float:
        return self.max_abs / max(self.max_ref_abs, 1.0e-12)


def load_anima_config_from_json(config_path: str, dit_path: Path):
    from train import TrainingConfig
    import config as default_config

    cfg = TrainingConfig()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        flat = default_config.flatten_preset(json.load(f), default_config.MODE_ANIMA)
    for key, value in flat.items():
        setattr(cfg, key, value)
    cfg.TRAINING_MODE = default_config.TRAINING_MODE_ANIMA
    cfg.DIT_PATH = str(dit_path)
    cfg.RESUME_TRAINING = False
    cfg.ANIMA_RESUME_MODEL_PATH = ""
    cfg.ANIMA_RESUME_STATE_PATH = ""
    cfg._type_check_and_correct()
    cfg.compute_dtype = torch.bfloat16 if cfg.MIXED_PRECISION == "bfloat16" else torch.float16
    cfg.is_rectified_flow = getattr(cfg, "PREDICTION_TYPE", "epsilon") == "rectified_flow"
    return cfg


def calibrated_profile(
    args: SimpleNamespace,
    storage_dtype: torch.dtype,
    format_name: str,
    log=print,
) -> set[str]:
    from torch.utils.data import DataLoader
    from train import PrecomputedImageBatchSampler, build_image_batch_schedule, set_seed
    from train_anima import (
        AnimaCachedDataset,
        AnimaTimestepSampler,
        anima_collate_fn,
        flowmatch_noise_and_target,
        load_anima_pipe,
        normalize_anima_config,
        precompute_and_cache_anima,
        run_dit_forward,
        anima_ticket_to_sigma_timestep,
    )

    input_path = Path(args.input)
    config = load_anima_config_from_json(getattr(args, "config"), input_path)
    dataset_overrides = [
        str(Path(p))
        for p in getattr(args, "dataset_overrides", []) or []
        if str(p).strip()
    ]
    if dataset_overrides:
        config.INSTANCE_DATASETS = [{"path": path, "repeats": 1} for path in dataset_overrides]
    normalize_anima_config(config)

    if config.SEED:
        set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("\nCalibrated profiling:")
    log(f"- config: {getattr(args, 'config')}")
    if dataset_overrides:
        log("- dataset overrides:")
        for path in dataset_overrides:
            log(f"  - {path}")
    log(f"- device: {device}")
    log(f"- steps:  {getattr(args, 'calibration_steps')}")

    required_paths = {
        "DIT_PATH": config.DIT_PATH,
        "VAE_PATH": config.VAE_PATH,
        "TEXT_ENCODER_PATH": config.TEXT_ENCODER_PATH,
    }
    for label, value in required_paths.items():
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{label} is required for calibration: {value}")

    log("Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))

    log("Preparing/reusing Anima cache...")
    precompute_and_cache_anima(config, pipe, device)

    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit
    dit.to(device=device, dtype=config.compute_dtype)
    dit.eval()

    profiles: dict[str, LayerProfile] = {}
    handles = []

    def make_hook(name: str, module: torch.nn.Linear):
        weight_key = f"{name}.weight"
        profiles[name] = LayerProfile(name, weight_key, module.weight.numel())

        def hook(layer, layer_input, output):
            x = layer_input[0]
            if not torch.is_floating_point(x):
                return
            q_weight = simulated_quant_weight(layer.weight.detach(), storage_dtype, format_name, x.dtype)
            bias = layer.bias.detach().to(dtype=x.dtype) if layer.bias is not None else None
            quant_output = F.linear(x, q_weight, bias)
            profiles[name].update(output, quant_output)
            del q_weight, quant_output

        return hook

    for name, module in dit.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        # Always scan all compatible Linear weights, excluding protected names.
        if not is_full_quant_compatible_name(key):
            continue
        if format_name == "nvfp4" and not is_nvfp4_compatible_weight(key, module.weight):
            continue
        handles.append(module.register_forward_hook(make_hook(name, module)))

    if not handles:
        raise RuntimeError("No candidate Linear layers found for calibration.")
    log(f"Hooked candidate layers: {len(handles):,}")

    dataset = AnimaCachedDataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("No cached Anima dataset items found for calibration.")
    total_scheduler_timesteps = 1000
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    steps = max(1, int(getattr(args, "calibration_steps", 16) or 16))
    image_schedule = build_image_batch_schedule(
        dataset,
        steps,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        True,  # Force timestep-bin coverage during quant calibration.
    )
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, 0)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)

    completed = 0
    with torch.inference_mode():
        for batch in dataloader:
            if completed >= steps:
                break
            if not batch:
                continue
            input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
            prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
            t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)
            batch_size = input_latents.shape[0]
            timestep_indices, _ = timestep_sampler.sample(batch_size)
            timestep_indices = timestep_indices.to(device=device)
            sigmas, timesteps = anima_ticket_to_sigma_timestep(timestep_indices, config.compute_dtype)
            noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)
            noisy_latents, _ = flowmatch_noise_and_target(input_latents, noise, sigmas)
            run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
            completed += 1
            log(f"- profiled batch {completed}/{steps}")

    for handle in handles:
        handle.remove()

    max_rel_mse = float(getattr(args, "calibration_rel_mse", 0.002))
    min_cosine = float(getattr(args, "calibration_cosine", 0.999))
    max_abs_ratio = float(getattr(args, "calibration_max_abs_ratio", 0.10))
    selected = {
        p.weight_key
        for p in profiles.values()
        if p.count > 0 and p.rel_mse <= max_rel_mse and p.cosine >= min_cosine and p.max_abs_ratio <= max_abs_ratio
    }

    ranked = sorted(profiles.values(), key=lambda p: (p.weight_key not in selected, p.rel_mse))
    log("\nCalibration summary:")
    log(f"- selected: {len(selected):,} / {len(profiles):,} candidate layer weights")
    log(f"- thresholds: rel_mse <= {max_rel_mse:g}, cosine >= {min_cosine:g}, max_abs_ratio <= {max_abs_ratio:g}")
    for p in ranked[:20]:
        mark = quant_label(getattr(args, "fp8", "e4m3")) if p.weight_key in selected else "keep"
        log(f"  {mark:4s} rel_mse={p.rel_mse:.6g} cosine={p.cosine:.6f} elems={p.elements:,} {p.weight_key}")

    profile_path = Path(args.output).with_suffix(".profile.json")
    payload = {
        "source": str(input_path),
        "quant": getattr(args, "fp8", "e4m3"),
        "format": format_name,
        "steps": completed,
        "thresholds": {"rel_mse": max_rel_mse, "cosine": min_cosine, "max_abs_ratio": max_abs_ratio},
        "selected": sorted(selected),
        "layers": [
            {
                "name": p.name,
                "weight_key": p.weight_key,
                "elements": p.elements,
                "rel_mse": p.rel_mse,
                "cosine": p.cosine,
                "max_abs_ratio": p.max_abs_ratio,
                "selected": p.weight_key in selected,
            }
            for p in sorted(profiles.values(), key=lambda item: item.weight_key)
        ],
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"- wrote profile: {profile_path}")

    dit.cpu()
    pipe.dit = None
    del dit, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected



def mixed_precision_profile(
    args: SimpleNamespace,
    log=print,
) -> dict[str, dict[str, object]]:
    """Full-forward Dynamic scan.

    This replaces the older local layer-hook scan. It keeps only one DiT on GPU:
    1) Load original DiT, run real no-grad training-style forwards, and cache
       the original teacher outputs on CPU.
    2) For each layer/format candidate, temporarily replace that one layer's
       weight with its simulated quantized/dequantized weight, run the full DiT
       forward on the same cached samples, compare final DiT prediction output,
       then restore the original weight.
    3) Pick the smallest candidate that passes thresholds for each layer.
    4) Optionally apply the whole selected mixed profile, run another full-DiT
       validation pass, and promote the worst selected layers upward until the
       complete mixed profile passes or promotion budget is exhausted.
    """
    from torch.utils.data import DataLoader
    from train import PrecomputedImageBatchSampler, build_image_batch_schedule, set_seed
    from train_anima import (
        AnimaCachedDataset,
        AnimaTimestepSampler,
        anima_collate_fn,
        flowmatch_noise_and_target,
        load_anima_pipe,
        normalize_anima_config,
        precompute_and_cache_anima,
        run_dit_forward,
        anima_ticket_to_sigma_timestep,
    )

    input_path = Path(args.input)
    config = load_anima_config_from_json(getattr(args, "config"), input_path)
    dataset_overrides = [str(Path(p)) for p in getattr(args, "dataset_overrides", []) or [] if str(p).strip()]
    if dataset_overrides:
        config.INSTANCE_DATASETS = [{"path": path, "repeats": 1} for path in dataset_overrides]
    normalize_anima_config(config)

    if config.SEED:
        set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_formats = parse_mixed_formats(getattr(args, "mixed_formats", "nvfp4,e4m3,int8"))
    nvfp4_scale_multipliers = parse_float_list(
        getattr(args, "nvfp4_scale_multipliers", DEFAULT_NVFP4_SCALE_MULTIPLIERS),
        name="NVFP4 scale multipliers",
    )
    storage_by_format = {fmt: quant_storage_dtype(fmt) for fmt in mixed_formats}

    max_rel_mse = float(getattr(args, "calibration_rel_mse", 0.002))
    min_cosine = float(getattr(args, "calibration_cosine", 0.999))
    max_abs_ratio = float(getattr(args, "calibration_max_abs_ratio", 0.10))
    global_promote_steps = max(0, int(getattr(args, "dynamic_global_promote_steps", 32) or 0))
    dynamic_early_stop = bool(getattr(args, "dynamic_early_stop", True))
    early_stop_min_batches = max(1, int(getattr(args, "dynamic_early_stop_min_batches", 4) or 4))
    early_stop_margin = max(1.0, float(getattr(args, "dynamic_early_stop_margin", 8.0) or 8.0))
    progress_interval = max(0.0, float(getattr(args, "dynamic_progress_interval", 15.0) or 0.0))

    log("\nDynamic full-forward profiling:")
    log(f"- config: {getattr(args, 'config')}")
    log(f"- formats: {', '.join(mixed_formats)}")
    if "nvfp4" in mixed_formats:
        log(f"- NVFP4 scale multipliers, low -> high: {', '.join(f'{v:g}' for v in nvfp4_scale_multipliers)}")
        log("  NVFP4 selector rule: choose the lowest-error passing multiplier; scale multipliers have the same file size.")
    if dataset_overrides:
        log("- dataset overrides:")
        for path in dataset_overrides:
            log(f"  - {path}")
    log(f"- device: {device}")
    log(f"- calibration batches: {getattr(args, 'calibration_steps')}")
    log(
        f"- progress: ETA logs every {progress_interval:g}s inside long candidates; "
        f"early stop {'on' if dynamic_early_stop else 'off'}"
    )
    if dynamic_early_stop:
        log(f"  early-stop rule: after {early_stop_min_batches} batch(es), reject candidates beyond {early_stop_margin:g}x thresholds")
    log("- VRAM mode: one DiT on GPU; teacher outputs are cached on CPU, then the same DiT is temporarily patched for candidate tests.")

    required_paths = {
        "DIT_PATH": config.DIT_PATH,
        "VAE_PATH": config.VAE_PATH,
        "TEXT_ENCODER_PATH": config.TEXT_ENCODER_PATH,
    }
    for label, value in required_paths.items():
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{label} is required for dynamic profiling: {value}")

    log("Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))

    log("Preparing/reusing Anima cache...")
    precompute_and_cache_anima(config, pipe, device)

    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit
    dit.to(device=device, dtype=config.compute_dtype)
    dit.eval()

    module_by_name: dict[str, torch.nn.Linear] = {}
    layer_keys: dict[str, str] = {}
    for name, module in dit.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if not is_full_quant_compatible_name(key):
            continue
        if not any(format_compatible_with_weight(fmt, key, module.weight) for fmt in mixed_formats):
            continue
        module_by_name[name] = module
        layer_keys[name] = key

    if not module_by_name:
        raise RuntimeError("No candidate Linear layers found for dynamic full-forward profiling.")
    log(f"Candidate Linear layers: {len(module_by_name):,}")

    dataset = AnimaCachedDataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("No cached Anima dataset items found for dynamic profiling.")
    total_scheduler_timesteps = 1000
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    steps = max(1, int(getattr(args, "calibration_steps", 16) or 16))
    image_schedule = build_image_batch_schedule(
        dataset,
        steps,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        True,  # Force timestep-bin coverage during quant calibration.
    )
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, 0)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)

    def extract_output_tensor(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in ("sample", "model_pred", "prediction", "pred", "output", "x"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        raise RuntimeError(
            "run_dit_forward did not return a tensor-like output. "
            "Full-forward Dynamic scan needs the final DiT prediction tensor."
        )

    def run_one_sample(sample: dict[str, torch.Tensor]) -> torch.Tensor:
        noisy_latents = sample["noisy_latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        timesteps = sample["timesteps"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        prompt_emb = sample["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        t5xxl_ids = sample["t5xxl_ids"].to(device=device, non_blocking=True)
        output = run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
        return extract_output_tensor(output)

    log("Building full-forward calibration cache from original model...")
    samples: list[dict[str, torch.Tensor]] = []
    teacher_outputs: list[torch.Tensor] = []
    completed = 0
    with torch.inference_mode():
        for batch in dataloader:
            if completed >= steps:
                break
            if not batch:
                continue
            input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
            prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
            t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)
            batch_size = input_latents.shape[0]
            timestep_indices, _ = timestep_sampler.sample(batch_size)
            timestep_indices = timestep_indices.to(device=device)
            sigmas, timesteps = anima_ticket_to_sigma_timestep(timestep_indices, config.compute_dtype)
            noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)
            noisy_latents, _ = flowmatch_noise_and_target(input_latents, noise, sigmas)

            sample = {
                "noisy_latents": noisy_latents.detach().cpu(),
                "timesteps": timesteps.detach().cpu(),
                "prompt_emb": prompt_emb.detach().cpu(),
                "t5xxl_ids": t5xxl_ids.detach().cpu(),
            }
            teacher = run_one_sample(sample).detach().cpu().float().contiguous()
            samples.append(sample)
            teacher_outputs.append(teacher)
            completed += 1
            log(f"- cached teacher batch {completed}/{steps}")

            del input_latents, prompt_emb, t5xxl_ids, timesteps, sigmas, noise, noisy_latents, teacher
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not samples:
        raise RuntimeError("No calibration samples were cached.")

    def spec_signature(spec: dict[str, object] | None) -> tuple:
        if not spec:
            return ("keep", None)
        fmt = str(spec.get("format", "keep"))
        if fmt == "nvfp4":
            return (fmt, round(float(spec.get("nvfp4_scale_multiplier", 1.0)), 12))
        return (fmt, None)

    def spec_sort_key(spec: dict[str, object]) -> tuple:
        fmt = str(spec["format"])
        base = quant_format_sort_key(fmt)
        if fmt == "nvfp4":
            return base + (float(spec.get("nvfp4_scale_multiplier", 1.0)),)
        return base + (0.0,)

    def spec_storage_tier(spec: dict[str, object]) -> int:
        fmt = str(spec["format"])
        if fmt == "ternary_1_58":
            return 0
        if fmt == "nvfp4":
            return 1
        if fmt in {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise"}:
            return 2
        return 99

    def candidate_selection_key(row: dict[str, object]) -> tuple:
        # Lowest storage first, then best measured quality among same-size choices.
        return (spec_storage_tier(row["spec"]), float(row["score"]), spec_sort_key(row["spec"]))

    def specs_for_layer(weight_key: str, weight: torch.Tensor) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for fmt in mixed_formats:
            if not format_compatible_with_weight(fmt, weight_key, weight):
                continue
            if fmt == "nvfp4":
                for scale_multiplier in nvfp4_scale_multipliers:
                    specs.append({"format": fmt, "nvfp4_scale_multiplier": float(scale_multiplier)})
            else:
                specs.append({"format": fmt})
        return sorted(specs, key=spec_sort_key)

    def apply_quant_to_layer(module: torch.nn.Linear, weight_key: str, spec: dict[str, object]) -> None:
        fmt = str(spec["format"])
        storage_dtype = storage_by_format[fmt]
        scale_multiplier = float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0
        q_weight = simulated_quant_weight(
            module.weight.detach(),
            storage_dtype,
            fmt,
            module.weight.dtype,
            nvfp4_scale_multiplier=scale_multiplier,
        )
        module.weight.data.copy_(q_weight.to(device=module.weight.device, dtype=module.weight.dtype))
        del q_weight

    def format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:d}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes:d}m {secs:02d}s"
        return f"{secs:d}s"

    def progress_bar(done: int, total: int, width: int = 24) -> str:
        total = max(1, int(total))
        done = min(max(0, int(done)), total)
        filled = int(round(width * done / total))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def clearly_failing_profile(profile: LayerProfile) -> bool:
        if not dynamic_early_stop or profile.batches < early_stop_min_batches:
            return False
        rel_bad = profile.rel_mse > max_rel_mse * early_stop_margin
        abs_bad = profile.max_abs_ratio > max_abs_ratio * early_stop_margin
        cos_bad = profile.cosine < 1.0 - ((1.0 - min_cosine) * early_stop_margin)
        return rel_bad and (abs_bad or cos_bad)

    def score_current_model(
        label: str,
        weight_key: str = "full_model",
        *,
        allow_early_stop: bool = False,
        progress_prefix: str = "",
    ) -> LayerProfile:
        profile = LayerProfile(label, weight_key, 0)
        last_progress = time.monotonic()
        total_batches = len(samples)
        with torch.inference_mode():
            for batch_index, (sample, teacher_cpu) in enumerate(zip(samples, teacher_outputs), start=1):
                student = run_one_sample(sample).detach().cpu().float().contiguous()
                profile.update(teacher_cpu, student)
                del student
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if allow_early_stop and clearly_failing_profile(profile):
                    profile.early_stopped = True
                    break

                if progress_prefix and progress_interval > 0.0:
                    now = time.monotonic()
                    if now - last_progress >= progress_interval:
                        log(
                            f"    {progress_prefix} {progress_bar(batch_index, total_batches, 12)} "
                            f"batch {batch_index}/{total_batches}"
                        )
                        last_progress = now
        return profile

    def passes_profile(profile: LayerProfile) -> bool:
        return (
            profile.count > 0
            and profile.rel_mse <= max_rel_mse
            and profile.cosine >= min_cosine
            and profile.max_abs_ratio <= max_abs_ratio
        )

    def profile_score(profile: LayerProfile) -> float:
        # Higher means worse. Used for same-size selection and promotion priority.
        return (
            profile.rel_mse / max(max_rel_mse, 1.0e-12)
            + max(0.0, (min_cosine - profile.cosine) / max(1.0 - min_cosine, 1.0e-12))
            + profile.max_abs_ratio / max(max_abs_ratio, 1.0e-12)
        )

    def score_single_layer_spec(name: str, module: torch.nn.Linear, weight_key: str, spec: dict[str, object], progress_prefix: str = "") -> LayerProfile:
        original = module.weight.detach().clone()
        try:
            apply_quant_to_layer(module, weight_key, spec)
            label = str(spec["format"])
            if label == "nvfp4":
                label = f"nvfp4@x{float(spec.get('nvfp4_scale_multiplier', 1.0)):g}"
            profile = score_current_model(label, weight_key, allow_early_stop=True, progress_prefix=progress_prefix)
            profile.elements = int(original.numel())
            return profile
        finally:
            module.weight.data.copy_(original)
            del original
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def apply_profile_specs(selected_specs: dict[str, dict[str, object]]) -> dict[str, torch.Tensor]:
        originals: dict[str, torch.Tensor] = {}
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            spec = selected_specs.get(weight_key)
            if not spec:
                continue
            originals[weight_key] = module.weight.detach().clone()
            apply_quant_to_layer(module, weight_key, spec)
        return originals

    def restore_profile_specs(originals: dict[str, torch.Tensor]) -> None:
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            original = originals.get(weight_key)
            if original is not None:
                module.weight.data.copy_(original)
        originals.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_profile_specs(selected_specs: dict[str, dict[str, object]], label: str = "full_mixed_profile") -> LayerProfile:
        originals = apply_profile_specs(selected_specs)
        try:
            return score_current_model(label, "full_mixed_profile", progress_prefix=label)
        finally:
            restore_profile_specs(originals)

    layer_items = sorted(module_by_name.items(), key=lambda item: layer_keys[item[0]])
    specs_by_name = {name: specs_for_layer(layer_keys[name], module.weight) for name, module in layer_items}
    total_candidates = sum(len(specs) for specs in specs_by_name.values())
    scan_start = time.monotonic()
    candidate_index = 0

    log("Scanning layers with full-DiT forward tests. This is intentionally much slower than local hooks.")
    log(f"- scan candidates: {total_candidates:,} quant option(s) across {len(layer_items):,} layer(s)")
    selected: dict[str, dict[str, object]] = {}
    layer_rows = []
    candidate_profiles: dict[str, list[dict[str, object]]] = {}

    for index, (name, module) in enumerate(layer_items, start=1):
        weight_key = layer_keys[name]
        specs = specs_by_name[name]
        candidates: list[dict[str, object]] = []
        log(f"\n[{index}/{len(module_by_name)}] {weight_key}")

        for spec in specs:
            fmt = str(spec["format"])
            label = fmt
            if fmt == "nvfp4":
                label = f"nvfp4@x{float(spec.get('nvfp4_scale_multiplier', 1.0)):g}"
            candidate_index += 1
            progress_prefix = f"candidate {candidate_index}/{total_candidates} {label}"
            profile = score_single_layer_spec(name, module, weight_key, spec, progress_prefix=progress_prefix)
            passed = passes_profile(profile)
            row = {
                "spec": dict(spec),
                "format": fmt,
                "nvfp4_scale_multiplier": float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else None,
                "rel_mse": profile.rel_mse,
                "cosine": profile.cosine,
                "max_abs_ratio": profile.max_abs_ratio,
                "passes": passed,
                "selected": False,
                "score": profile_score(profile),
                "batches": profile.batches,
                "early_stopped": profile.early_stopped,
            }
            candidates.append(row)
            stop_note = f" early-stop {profile.batches}/{len(samples)}" if profile.early_stopped else ""
            log(
                f"  {label:18s} rel={profile.rel_mse:.6g} "
                f"cos={profile.cosine:.6f} maxabs={profile.max_abs_ratio:.6g} "
                f"{'PASS' if passed else 'fail'}{stop_note}"
            )
            elapsed = time.monotonic() - scan_start
            avg_candidate = elapsed / max(candidate_index, 1)
            eta = avg_candidate * max(total_candidates - candidate_index, 0)
            log(
                f"  scan {progress_bar(candidate_index, total_candidates)} "
                f"{candidate_index}/{total_candidates} elapsed {format_duration(elapsed)} ETA {format_duration(eta)}"
            )
            # Keep profiling all candidates so the JSON report can explain the
            # quality/size tradeoff and later global promotions.

        passing = [row for row in candidates if row["passes"]]
        if passing:
            chosen = sorted(passing, key=candidate_selection_key)[0]
            chosen["selected"] = True
            selected[weight_key] = dict(chosen["spec"])
            chosen_label = str(chosen["format"])
            if chosen_label == "nvfp4":
                chosen_label = f"nvfp4@x{float(chosen['nvfp4_scale_multiplier']):g}"
        else:
            chosen = None
            chosen_label = "keep"

        candidate_profiles[weight_key] = candidates
        layer_rows.append({
            "weight_key": weight_key,
            "chosen": chosen_label,
            "chosen_spec": dict(chosen["spec"]) if chosen else None,
            "formats": [
                {k: v for k, v in row.items() if k != "spec"} | {"spec": row["spec"]}
                for row in candidates
            ],
        })
        log(f"  -> chosen: {chosen_label}")

    global_history = []
    if selected:
        log("\nValidating complete mixed profile with full-DiT forwards...")
        full_profile = score_profile_specs(selected, "full_mixed_profile")
        full_passes = passes_profile(full_profile)
        global_history.append({
            "promotion_step": 0,
            "rel_mse": full_profile.rel_mse,
            "cosine": full_profile.cosine,
            "max_abs_ratio": full_profile.max_abs_ratio,
            "passes": full_passes,
            "selected_layers": len(selected),
        })
        log(
            f"- global profile rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
            f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
        )

        promotion_step = 0
        while not full_passes and promotion_step < global_promote_steps:
            promotable = []
            for weight_key, current_spec in selected.items():
                rows = candidate_profiles.get(weight_key, [])
                current_tier = spec_storage_tier(current_spec)
                passing_rows = [
                    row for row in rows
                    if row["passes"] and spec_storage_tier(row["spec"]) > current_tier
                ]
                passing_rows = sorted(passing_rows, key=candidate_selection_key)
                current_sig = spec_signature(current_spec)
                next_row = passing_rows[0] if passing_rows else None
                current_row = next((row for row in rows if spec_signature(row["spec"]) == current_sig), None)
                if next_row is not None and current_row is not None:
                    promotable.append((float(current_row["score"]), weight_key, next_row))

            if not promotable:
                log("- global profile still fails, but no passing larger storage-tier candidates remain to promote.")
                break

            promotable.sort(reverse=True, key=lambda item: item[0])
            _score, promoted_key, next_row = promotable[0]
            selected[promoted_key] = dict(next_row["spec"])
            promotion_step += 1
            promoted_label = str(next_row["format"])
            if promoted_label == "nvfp4":
                promoted_label = f"nvfp4@x{float(next_row['nvfp4_scale_multiplier']):g}"
            log(f"- promotion {promotion_step}/{global_promote_steps}: {promoted_key} -> {promoted_label}")

            full_profile = score_profile_specs(selected, "full_mixed_profile")
            full_passes = passes_profile(full_profile)
            global_history.append({
                "promotion_step": promotion_step,
                "promoted_key": promoted_key,
                "promoted_to": dict(next_row["spec"]),
                "rel_mse": full_profile.rel_mse,
                "cosine": full_profile.cosine,
                "max_abs_ratio": full_profile.max_abs_ratio,
                "passes": full_passes,
                "selected_layers": len(selected),
            })
            log(
                f"  global rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
                f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
            )
    else:
        log("\nNo layers passed Dynamic full-forward thresholds; output will keep all candidate layers in keep-dtype.")

    # Refresh selected flags after global promotions.
    selected_sigs = {key: spec_signature(spec) for key, spec in selected.items()}
    for row in layer_rows:
        weight_key = row["weight_key"]
        selected_sig = selected_sigs.get(weight_key)
        chosen_label = "keep"
        chosen_spec = None
        for item in row["formats"]:
            is_selected = selected_sig is not None and spec_signature(item["spec"]) == selected_sig
            item["selected"] = is_selected
            if is_selected:
                chosen_spec = dict(item["spec"])
                chosen_label = str(item["format"])
                if chosen_label == "nvfp4":
                    chosen_label = f"nvfp4@x{float(item['nvfp4_scale_multiplier']):g}"
        row["chosen"] = chosen_label
        row["chosen_spec"] = chosen_spec

    counts = defaultdict(int)
    nvfp4_scale_counts = defaultdict(int)
    for spec in selected.values():
        fmt = str(spec["format"])
        counts[fmt] += 1
        if fmt == "nvfp4":
            nvfp4_scale_counts[float(spec.get("nvfp4_scale_multiplier", 1.0))] += 1
    kept = len(layer_keys) - len(selected)

    log("\nDynamic full-forward summary:")
    log(f"- thresholds: rel_mse <= {max_rel_mse:g}, cosine >= {min_cosine:g}, max_abs_ratio <= {max_abs_ratio:g}")
    for fmt, count in sorted(counts.items(), key=lambda item: quant_format_sort_key(item[0])):
        log(f"- {fmt:18s}: {count:,} layer weight(s)")
    if nvfp4_scale_counts:
        log("- NVFP4 selected scale multipliers:")
        for scale_multiplier, count in sorted(nvfp4_scale_counts.items()):
            log(f"  - x{scale_multiplier:g}: {count:,} layer weight(s)")
    log(f"- {'keep':18s}: {kept:,} layer weight(s)")
    if global_history:
        last = global_history[-1]
        log(
            f"- final global drift: rel={last['rel_mse']:.6g}, cos={last['cosine']:.6f}, "
            f"maxabs={last['max_abs_ratio']:.6g}, {'PASS' if last['passes'] else 'fail'}"
        )

    profile_path = Path(args.output).with_suffix(".mixed_profile.json")
    payload = {
        "source": str(input_path),
        "mode": "dynamic_full_forward",
        "description": "Full-DiT teacher/student output matching on cached noisy latents/timesteps/prompts; one DiT is kept on GPU and weights are temporarily patched per candidate.",
        "formats": mixed_formats,
        "nvfp4_scale_multipliers": nvfp4_scale_multipliers,
        "steps": completed,
        "thresholds": {"rel_mse": max_rel_mse, "cosine": min_cosine, "max_abs_ratio": max_abs_ratio},
        "global_promote_steps": global_promote_steps,
        "early_stop": {
            "enabled": dynamic_early_stop,
            "min_batches": early_stop_min_batches,
            "margin": early_stop_margin,
        },
        "scan_candidates": total_candidates,
        "global_history": global_history,
        "selected": dict(sorted(selected.items())),
        "layers": layer_rows,
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"- wrote profile: {profile_path}")

    dit.cpu()
    pipe.dit = None
    del dit, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected


def fast_mixed_precision_profile(
    args: SimpleNamespace,
    log=print,
) -> dict[str, dict[str, object]]:
    """Fast activation-aware Dynamic scan.

    This is the practical path for large DiTs: run the original model only for the
    requested calibration batches, use Linear hooks to compare each candidate
    quantized weight against the real full-precision Linear output, then do a
    small full-model validation pass for the selected mixed profile.
    """
    from torch.utils.data import DataLoader
    from train import PrecomputedImageBatchSampler, build_image_batch_schedule, set_seed
    from train_anima import (
        AnimaCachedDataset,
        AnimaTimestepSampler,
        anima_collate_fn,
        flowmatch_noise_and_target,
        load_anima_pipe,
        normalize_anima_config,
        precompute_and_cache_anima,
        run_dit_forward,
        anima_ticket_to_sigma_timestep,
    )

    input_path = Path(args.input)
    config = load_anima_config_from_json(getattr(args, "config"), input_path)
    dataset_overrides = [str(Path(p)) for p in getattr(args, "dataset_overrides", []) or [] if str(p).strip()]
    if dataset_overrides:
        config.INSTANCE_DATASETS = [{"path": path, "repeats": 1} for path in dataset_overrides]
    normalize_anima_config(config)

    if config.SEED:
        set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_formats = parse_mixed_formats(getattr(args, "mixed_formats", "nvfp4,e4m3,int8"))
    nvfp4_scale_multipliers = parse_float_list(
        getattr(args, "nvfp4_scale_multipliers", DEFAULT_NVFP4_SCALE_MULTIPLIERS),
        name="NVFP4 scale multipliers",
    )
    storage_by_format = {fmt: quant_storage_dtype(fmt) for fmt in mixed_formats}

    max_rel_mse = float(getattr(args, "calibration_rel_mse", 0.002))
    min_cosine = float(getattr(args, "calibration_cosine", 0.999))
    max_abs_ratio = float(getattr(args, "calibration_max_abs_ratio", 0.10))
    global_promote_steps = max(0, int(getattr(args, "dynamic_global_promote_steps", 8) or 0))
    dynamic_early_stop = bool(getattr(args, "dynamic_early_stop", True))
    early_stop_min_batches = max(1, int(getattr(args, "dynamic_early_stop_min_batches", 4) or 4))
    early_stop_margin = max(1.0, float(getattr(args, "dynamic_early_stop_margin", 8.0) or 8.0))

    log("\nFast Dynamic activation-aware profiling:")
    log(f"- config: {getattr(args, 'config')}")
    log(f"- formats: {', '.join(mixed_formats)}")
    if "nvfp4" in mixed_formats:
        log(f"- NVFP4 scale multipliers, low -> high: {', '.join(f'{v:g}' for v in nvfp4_scale_multipliers)}")
    if dataset_overrides:
        log("- dataset overrides:")
        for path in dataset_overrides:
            log(f"  - {path}")
    log(f"- device: {device}")
    log(f"- calibration batches: {getattr(args, 'calibration_steps')}")
    log("- scan mode: one real DiT forward per batch; candidate errors are measured inside Linear hooks.")

    required_paths = {
        "DIT_PATH": config.DIT_PATH,
        "VAE_PATH": config.VAE_PATH,
        "TEXT_ENCODER_PATH": config.TEXT_ENCODER_PATH,
    }
    for label, value in required_paths.items():
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{label} is required for fast dynamic profiling: {value}")

    log("Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))

    log("Preparing/reusing Anima cache...")
    precompute_and_cache_anima(config, pipe, device)

    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit
    dit.to(device=device, dtype=config.compute_dtype)
    dit.eval()

    def spec_signature(spec: dict[str, object] | None) -> tuple:
        if not spec:
            return ("keep", None)
        fmt = str(spec.get("format", "keep"))
        if fmt == "nvfp4":
            return (fmt, round(float(spec.get("nvfp4_scale_multiplier", 1.0)), 12))
        return (fmt, None)

    def spec_sort_key(spec: dict[str, object]) -> tuple:
        fmt = str(spec["format"])
        base = quant_format_sort_key(fmt)
        if fmt == "nvfp4":
            return base + (float(spec.get("nvfp4_scale_multiplier", 1.0)),)
        return base + (0.0,)

    def spec_storage_tier(spec: dict[str, object]) -> int:
        fmt = str(spec["format"])
        if fmt == "ternary_1_58":
            return 0
        if fmt == "nvfp4":
            return 1
        if fmt in {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise"}:
            return 2
        return 99

    def specs_for_layer(weight_key: str, weight: torch.Tensor) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for fmt in mixed_formats:
            if not format_compatible_with_weight(fmt, weight_key, weight):
                continue
            if fmt == "nvfp4":
                for scale_multiplier in nvfp4_scale_multipliers:
                    specs.append({"format": fmt, "nvfp4_scale_multiplier": float(scale_multiplier)})
            else:
                specs.append({"format": fmt})
        return sorted(specs, key=spec_sort_key)

    def passes_profile(profile: LayerProfile) -> bool:
        return (
            profile.count > 0
            and profile.rel_mse <= max_rel_mse
            and profile.cosine >= min_cosine
            and profile.max_abs_ratio <= max_abs_ratio
        )

    def profile_score(profile: LayerProfile) -> float:
        return (
            profile.rel_mse / max(max_rel_mse, 1.0e-12)
            + max(0.0, (min_cosine - profile.cosine) / max(1.0 - min_cosine, 1.0e-12))
            + profile.max_abs_ratio / max(max_abs_ratio, 1.0e-12)
        )

    def candidate_selection_key(row: dict[str, object]) -> tuple:
        return (spec_storage_tier(row["spec"]), float(row["score"]), spec_sort_key(row["spec"]))

    def clearly_failing_profile(profile: LayerProfile) -> bool:
        if not dynamic_early_stop or profile.batches < early_stop_min_batches:
            return False
        rel_bad = profile.rel_mse > max_rel_mse * early_stop_margin
        abs_bad = profile.max_abs_ratio > max_abs_ratio * early_stop_margin
        cos_bad = profile.cosine < 1.0 - ((1.0 - min_cosine) * early_stop_margin)
        return rel_bad and (abs_bad or cos_bad)

    module_by_name: dict[str, torch.nn.Linear] = {}
    layer_keys: dict[str, str] = {}
    specs_by_name: dict[str, list[dict[str, object]]] = {}
    profiles: dict[str, dict[tuple, LayerProfile]] = {}
    activation_stats: dict[str, dict[str, float]] = {}

    for name, module in dit.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if not is_full_quant_compatible_name(key):
            continue
        specs = specs_for_layer(key, module.weight)
        if not specs:
            continue
        module_by_name[name] = module
        layer_keys[name] = key
        specs_by_name[name] = specs
        profiles[key] = {}
        activation_stats[key] = {"input_abs_max": 0.0, "weight_abs_max": float(module.weight.detach().float().abs().amax().item())}
        for spec in specs:
            label = str(spec["format"])
            if label == "nvfp4":
                label = f"nvfp4@x{float(spec.get('nvfp4_scale_multiplier', 1.0)):g}"
            profiles[key][spec_signature(spec)] = LayerProfile(label, key, module.weight.numel())

    if not module_by_name:
        raise RuntimeError("No candidate Linear layers found for fast dynamic profiling.")
    total_candidates = sum(len(specs) for specs in specs_by_name.values())
    log(f"Candidate Linear layers: {len(module_by_name):,}")
    log(f"Local candidate tests: {total_candidates:,}")

    handles = []

    def make_hook(name: str, module: torch.nn.Linear):
        weight_key = layer_keys[name]
        specs = specs_by_name[name]

        def hook(layer, layer_input, output):
            x = layer_input[0]
            if not torch.is_floating_point(x):
                return
            stats = activation_stats[weight_key]
            stats["input_abs_max"] = max(
                stats["input_abs_max"],
                float(x.detach().float().abs().amax().item()) if x.numel() else 0.0,
            )
            bias = layer.bias.detach().to(dtype=x.dtype) if layer.bias is not None else None
            for spec in specs:
                profile = profiles[weight_key][spec_signature(spec)]
                if profile.early_stopped:
                    continue
                fmt = str(spec["format"])
                scale_multiplier = float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0
                q_weight = simulated_quant_weight(
                    layer.weight.detach(),
                    storage_by_format[fmt],
                    fmt,
                    x.dtype,
                    nvfp4_scale_multiplier=scale_multiplier,
                )
                quant_output = F.linear(x, q_weight, bias)
                profile.update(output, quant_output)
                if clearly_failing_profile(profile):
                    profile.early_stopped = True
                del q_weight, quant_output

        return hook

    for name, module in module_by_name.items():
        handles.append(module.register_forward_hook(make_hook(name, module)))

    dataset = AnimaCachedDataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("No cached Anima dataset items found for fast dynamic profiling.")
    total_scheduler_timesteps = 1000
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    steps = max(1, int(getattr(args, "calibration_steps", 16) or 16))
    image_schedule = build_image_batch_schedule(
        dataset,
        steps,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        True,
    )
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, 0)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)

    def extract_output_tensor(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in ("sample", "model_pred", "prediction", "pred", "output", "x"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        raise RuntimeError("run_dit_forward did not return a tensor-like output.")

    def run_one_sample(sample: dict[str, torch.Tensor]) -> torch.Tensor:
        noisy_latents = sample["noisy_latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        timesteps = sample["timesteps"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        prompt_emb = sample["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        t5xxl_ids = sample["t5xxl_ids"].to(device=device, non_blocking=True)
        output = run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
        return extract_output_tensor(output)

    samples: list[dict[str, torch.Tensor]] = []
    teacher_outputs: list[torch.Tensor] = []
    completed = 0
    log("Running activation-aware calibration forwards...")
    with torch.inference_mode():
        for batch in dataloader:
            if completed >= steps:
                break
            if not batch:
                continue
            input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
            prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
            t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)
            batch_size = input_latents.shape[0]
            timestep_indices, _ = timestep_sampler.sample(batch_size)
            timestep_indices = timestep_indices.to(device=device)
            sigmas, timesteps = anima_ticket_to_sigma_timestep(timestep_indices, config.compute_dtype)
            noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)
            noisy_latents, _ = flowmatch_noise_and_target(input_latents, noise, sigmas)
            sample = {
                "noisy_latents": noisy_latents.detach().cpu(),
                "timesteps": timesteps.detach().cpu(),
                "prompt_emb": prompt_emb.detach().cpu(),
                "t5xxl_ids": t5xxl_ids.detach().cpu(),
            }
            teacher = run_one_sample(sample).detach().cpu().float().contiguous()
            samples.append(sample)
            teacher_outputs.append(teacher)
            completed += 1
            log(f"- calibrated batch {completed}/{steps}")
            del input_latents, prompt_emb, t5xxl_ids, timesteps, sigmas, noise, noisy_latents, teacher
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for handle in handles:
        handle.remove()

    selected: dict[str, dict[str, object]] = {}
    layer_rows = []
    candidate_profiles: dict[str, list[dict[str, object]]] = {}
    for name, module in sorted(module_by_name.items(), key=lambda item: layer_keys[item[0]]):
        weight_key = layer_keys[name]
        candidates: list[dict[str, object]] = []
        for spec in specs_by_name[name]:
            profile = profiles[weight_key][spec_signature(spec)]
            passed = passes_profile(profile)
            fmt = str(spec["format"])
            row = {
                "spec": dict(spec),
                "format": fmt,
                "nvfp4_scale_multiplier": float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else None,
                "rel_mse": profile.rel_mse,
                "cosine": profile.cosine,
                "max_abs_ratio": profile.max_abs_ratio,
                "passes": passed,
                "selected": False,
                "score": profile_score(profile),
                "batches": profile.batches,
                "early_stopped": profile.early_stopped,
            }
            candidates.append(row)
        passing = [row for row in candidates if row["passes"]]
        chosen = sorted(passing, key=candidate_selection_key)[0] if passing else None
        chosen_label = "keep"
        if chosen:
            chosen["selected"] = True
            selected[weight_key] = dict(chosen["spec"])
            chosen_label = str(chosen["format"])
            if chosen_label == "nvfp4":
                chosen_label = f"nvfp4@x{float(chosen['nvfp4_scale_multiplier']):g}"
        candidate_profiles[weight_key] = candidates
        layer_rows.append({
            "weight_key": weight_key,
            "chosen": chosen_label,
            "chosen_spec": dict(chosen["spec"]) if chosen else None,
            "activation": activation_stats.get(weight_key, {}),
            "formats": [{k: v for k, v in row.items() if k != "spec"} | {"spec": row["spec"]} for row in candidates],
        })

    def apply_quant_to_layer(module: torch.nn.Linear, spec: dict[str, object]) -> torch.Tensor:
        original = module.weight.detach().clone()
        fmt = str(spec["format"])
        q_weight = simulated_quant_weight(
            module.weight.detach(),
            storage_by_format[fmt],
            fmt,
            module.weight.dtype,
            nvfp4_scale_multiplier=float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0,
        )
        module.weight.data.copy_(q_weight.to(device=module.weight.device, dtype=module.weight.dtype))
        del q_weight
        return original

    def restore_originals(originals: dict[str, torch.Tensor]) -> None:
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            original = originals.get(weight_key)
            if original is not None:
                module.weight.data.copy_(original)
        originals.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_selected_profile(selected_specs: dict[str, dict[str, object]], label: str) -> LayerProfile:
        originals: dict[str, torch.Tensor] = {}
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            spec = selected_specs.get(weight_key)
            if spec:
                originals[weight_key] = apply_quant_to_layer(module, spec)
        profile = LayerProfile(label, "full_mixed_profile", 0)
        try:
            with torch.inference_mode():
                for sample, teacher_cpu in zip(samples, teacher_outputs):
                    student = run_one_sample(sample).detach().cpu().float().contiguous()
                    profile.update(teacher_cpu, student)
                    del student
        finally:
            restore_originals(originals)
        return profile

    global_history = []
    if selected and samples:
        log("\nValidating Fast Dynamic mixed profile with full-DiT forwards...")
        full_profile = score_selected_profile(selected, "fast_mixed_profile")
        full_passes = passes_profile(full_profile)
        global_history.append({
            "promotion_step": 0,
            "rel_mse": full_profile.rel_mse,
            "cosine": full_profile.cosine,
            "max_abs_ratio": full_profile.max_abs_ratio,
            "passes": full_passes,
            "selected_layers": len(selected),
        })
        log(
            f"- global profile rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
            f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
        )

        promotion_step = 0
        while not full_passes and promotion_step < global_promote_steps:
            promotable = []
            for weight_key, current_spec in selected.items():
                rows = candidate_profiles.get(weight_key, [])
                current_sig = spec_signature(current_spec)
                current_row = next((row for row in rows if spec_signature(row["spec"]) == current_sig), None)
                if current_row is None:
                    continue
                current_tier = spec_storage_tier(current_spec)
                larger = [row for row in rows if row["passes"] and spec_storage_tier(row["spec"]) > current_tier]
                next_row = sorted(larger, key=candidate_selection_key)[0] if larger else None
                promotable.append((float(current_row["score"]), weight_key, next_row))
            if not promotable:
                log("- global profile still fails, but no selected layer can be promoted or kept.")
                break
            promotable.sort(reverse=True, key=lambda item: item[0])
            _score, promoted_key, next_row = promotable[0]
            promotion_step += 1
            if next_row is None:
                selected.pop(promoted_key, None)
                promoted_label = "keep"
            else:
                selected[promoted_key] = dict(next_row["spec"])
                promoted_label = str(next_row["format"])
                if promoted_label == "nvfp4":
                    promoted_label = f"nvfp4@x{float(next_row['nvfp4_scale_multiplier']):g}"
            log(f"- promotion {promotion_step}/{global_promote_steps}: {promoted_key} -> {promoted_label}")
            full_profile = score_selected_profile(selected, "fast_mixed_profile")
            full_passes = passes_profile(full_profile)
            global_history.append({
                "promotion_step": promotion_step,
                "promoted_key": promoted_key,
                "promoted_to": dict(next_row["spec"]) if next_row else None,
                "rel_mse": full_profile.rel_mse,
                "cosine": full_profile.cosine,
                "max_abs_ratio": full_profile.max_abs_ratio,
                "passes": full_passes,
                "selected_layers": len(selected),
            })
            log(
                f"  global rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
                f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
            )

    selected_sigs = {key: spec_signature(spec) for key, spec in selected.items()}
    for row in layer_rows:
        weight_key = row["weight_key"]
        selected_sig = selected_sigs.get(weight_key)
        chosen_label = "keep"
        chosen_spec = None
        for item in row["formats"]:
            is_selected = selected_sig is not None and spec_signature(item["spec"]) == selected_sig
            item["selected"] = is_selected
            if is_selected:
                chosen_spec = dict(item["spec"])
                chosen_label = str(item["format"])
                if chosen_label == "nvfp4":
                    chosen_label = f"nvfp4@x{float(item['nvfp4_scale_multiplier']):g}"
        row["chosen"] = chosen_label
        row["chosen_spec"] = chosen_spec

    counts = defaultdict(int)
    nvfp4_scale_counts = defaultdict(int)
    for spec in selected.values():
        fmt = str(spec["format"])
        counts[fmt] += 1
        if fmt == "nvfp4":
            nvfp4_scale_counts[float(spec.get("nvfp4_scale_multiplier", 1.0))] += 1
    kept = len(layer_keys) - len(selected)

    log("\nFast Dynamic summary:")
    log(f"- thresholds: rel_mse <= {max_rel_mse:g}, cosine >= {min_cosine:g}, max_abs_ratio <= {max_abs_ratio:g}")
    for fmt, count in sorted(counts.items(), key=lambda item: quant_format_sort_key(item[0])):
        log(f"- {fmt:18s}: {count:,} layer weight(s)")
    if nvfp4_scale_counts:
        log("- NVFP4 selected scale multipliers:")
        for scale_multiplier, count in sorted(nvfp4_scale_counts.items()):
            log(f"  - x{scale_multiplier:g}: {count:,} layer weight(s)")
    log(f"- {'keep':18s}: {kept:,} layer weight(s)")
    if global_history:
        last = global_history[-1]
        log(
            f"- final global drift: rel={last['rel_mse']:.6g}, cos={last['cosine']:.6f}, "
            f"maxabs={last['max_abs_ratio']:.6g}, {'PASS' if last['passes'] else 'fail'}"
        )

    profile_path = Path(args.output).with_suffix(".mixed_profile.json")
    payload = {
        "source": str(input_path),
        "mode": "fast_dynamic_activation_hooks",
        "description": "Activation-aware Linear hook scan with final full-DiT validation; much faster than per-candidate full-forward Dynamic.",
        "formats": mixed_formats,
        "nvfp4_scale_multipliers": nvfp4_scale_multipliers,
        "steps": completed,
        "thresholds": {"rel_mse": max_rel_mse, "cosine": min_cosine, "max_abs_ratio": max_abs_ratio},
        "global_promote_steps": global_promote_steps,
        "early_stop": {
            "enabled": dynamic_early_stop,
            "min_batches": early_stop_min_batches,
            "margin": early_stop_margin,
        },
        "scan_candidates": total_candidates,
        "global_history": global_history,
        "selected": dict(sorted(selected.items())),
        "layers": layer_rows,
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"- wrote profile: {profile_path}")

    dit.cpu()
    pipe.dit = None
    del dit, pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected

def global_forward_mixed_precision_profile(
    args: SimpleNamespace,
    log=print,
) -> dict[str, dict[str, object]]:
    """Global candidate scan for Fast Dynamic.

    Cost is roughly one base pass plus one full-model pass per quant candidate,
    instead of one full-model pass per layer/candidate. It captures every Linear
    output from the base pass on CPU, applies one candidate format to all compatible
    layers, captures every Linear output again, and compares the two streams.
    """
    from torch.utils.data import DataLoader
    from train import PrecomputedImageBatchSampler, build_image_batch_schedule, set_seed
    from train_anima import (
        AnimaCachedDataset,
        AnimaTimestepSampler,
        anima_collate_fn,
        flowmatch_noise_and_target,
        load_anima_pipe,
        normalize_anima_config,
        precompute_and_cache_anima,
        run_dit_forward,
        anima_ticket_to_sigma_timestep,
    )

    input_path = Path(args.input)
    config = load_anima_config_from_json(getattr(args, "config"), input_path)
    dataset_overrides = [str(Path(p)) for p in getattr(args, "dataset_overrides", []) or [] if str(p).strip()]
    if dataset_overrides:
        config.INSTANCE_DATASETS = [{"path": path, "repeats": 1} for path in dataset_overrides]
    normalize_anima_config(config)
    if config.SEED:
        set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_formats = parse_mixed_formats(getattr(args, "mixed_formats", "nvfp4,e4m3,int8"))
    nvfp4_scale_multipliers = parse_float_list(
        getattr(args, "nvfp4_scale_multipliers", DEFAULT_NVFP4_SCALE_MULTIPLIERS),
        name="NVFP4 scale multipliers",
    )
    storage_by_format = {fmt: quant_storage_dtype(fmt) for fmt in mixed_formats}

    max_rel_mse = float(getattr(args, "calibration_rel_mse", 0.002))
    min_cosine = float(getattr(args, "calibration_cosine", 0.999))
    max_abs_ratio = float(getattr(args, "calibration_max_abs_ratio", 0.10))
    global_promote_steps = max(0, int(getattr(args, "dynamic_global_promote_steps", 8) or 0))

    log("\nFast Dynamic global-forward profiling:")
    log(f"- config: {getattr(args, 'config')}")
    log(f"- formats: {', '.join(mixed_formats)}")
    if "nvfp4" in mixed_formats:
        log(f"- NVFP4 scale multipliers: {', '.join(f'{v:g}' for v in nvfp4_scale_multipliers)}")
    log(f"- device: {device}")
    log(f"- calibration batches: {getattr(args, 'calibration_steps')}")
    log("- scan mode: capture all base Linear outputs once, then run one all-layer quantized pass per candidate.")

    required_paths = {
        "DIT_PATH": config.DIT_PATH,
        "VAE_PATH": config.VAE_PATH,
        "TEXT_ENCODER_PATH": config.TEXT_ENCODER_PATH,
    }
    for label, value in required_paths.items():
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{label} is required for fast dynamic profiling: {value}")

    log("Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))

    log("Preparing/reusing Anima cache...")
    precompute_and_cache_anima(config, pipe, device)
    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit
    dit.to(device=device, dtype=config.compute_dtype)
    dit.eval()

    def spec_signature(spec: dict[str, object] | None) -> tuple:
        if not spec:
            return ("keep", None)
        fmt = str(spec.get("format", "keep"))
        if fmt == "nvfp4":
            return (fmt, round(float(spec.get("nvfp4_scale_multiplier", 1.0)), 12))
        return (fmt, None)

    def spec_sort_key(spec: dict[str, object]) -> tuple:
        fmt = str(spec["format"])
        base = quant_format_sort_key(fmt)
        if fmt == "nvfp4":
            return base + (float(spec.get("nvfp4_scale_multiplier", 1.0)),)
        return base + (0.0,)

    def spec_storage_tier(spec: dict[str, object]) -> int:
        fmt = str(spec["format"])
        if fmt == "ternary_1_58":
            return 0
        if fmt == "nvfp4":
            return 1
        if fmt in {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise"}:
            return 2
        return 99

    def global_specs() -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        for fmt in mixed_formats:
            if fmt == "nvfp4":
                for scale_multiplier in nvfp4_scale_multipliers:
                    specs.append({"format": fmt, "nvfp4_scale_multiplier": float(scale_multiplier)})
            else:
                specs.append({"format": fmt})
        return sorted(specs, key=spec_sort_key)

    def passes_profile(profile: LayerProfile) -> bool:
        return (
            profile.count > 0
            and profile.rel_mse <= max_rel_mse
            and profile.cosine >= min_cosine
            and profile.max_abs_ratio <= max_abs_ratio
        )

    def profile_score(profile: LayerProfile) -> float:
        return (
            profile.rel_mse / max(max_rel_mse, 1.0e-12)
            + max(0.0, (min_cosine - profile.cosine) / max(1.0 - min_cosine, 1.0e-12))
            + profile.max_abs_ratio / max(max_abs_ratio, 1.0e-12)
        )

    def candidate_selection_key(row: dict[str, object]) -> tuple:
        return (spec_storage_tier(row["spec"]), float(row["score"]), spec_sort_key(row["spec"]))

    module_by_name: dict[str, torch.nn.Linear] = {}
    layer_keys: dict[str, str] = {}
    specs_by_key: dict[str, list[dict[str, object]]] = {}
    for name, module in dit.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if not is_full_quant_compatible_name(key):
            continue
        specs = [spec for spec in global_specs() if format_compatible_with_weight(str(spec["format"]), key, module.weight)]
        if not specs:
            continue
        module_by_name[name] = module
        layer_keys[name] = key
        specs_by_key[key] = specs

    if not module_by_name:
        raise RuntimeError("No candidate Linear layers found for fast dynamic profiling.")
    scan_specs = global_specs()
    total_layer_candidates = sum(len(specs) for specs in specs_by_key.values())
    log(f"Candidate Linear layers: {len(module_by_name):,}")
    log(f"Global candidate passes: {len(scan_specs):,}; layer decisions scored: {total_layer_candidates:,}")

    dataset = AnimaCachedDataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("No cached Anima dataset items found for fast dynamic profiling.")
    total_scheduler_timesteps = 1000
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    steps = max(1, int(getattr(args, "calibration_steps", 16) or 16))
    image_schedule = build_image_batch_schedule(
        dataset,
        steps,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        True,
    )
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, 0)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)

    capture_store: dict[str, torch.Tensor] | None = None
    capture_dtype = config.compute_dtype if config.compute_dtype in {torch.float16, torch.bfloat16} else torch.float16
    handles = []

    def make_capture_hook(name: str):
        weight_key = layer_keys[name]

        def hook(_layer, _layer_input, output):
            if capture_store is None or not torch.is_tensor(output):
                return
            capture_store[weight_key] = output.detach().to(device="cpu", dtype=capture_dtype).contiguous()

        return hook

    for name, module in module_by_name.items():
        handles.append(module.register_forward_hook(make_capture_hook(name)))

    def extract_output_tensor(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in ("sample", "model_pred", "prediction", "pred", "output", "x"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        raise RuntimeError("run_dit_forward did not return a tensor-like output.")

    def run_one_sample(sample: dict[str, torch.Tensor]) -> torch.Tensor:
        noisy_latents = sample["noisy_latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        timesteps = sample["timesteps"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        prompt_emb = sample["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        t5xxl_ids = sample["t5xxl_ids"].to(device=device, non_blocking=True)
        output = run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
        return extract_output_tensor(output)

    cache_root = Path(args.output).with_name(f".{Path(args.output).stem}.fast_dynamic_cache_{uuid.uuid4().hex[:8]}")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    completed = 0
    log("Capturing base model activations to disk...")
    log(f"- activation cache: {cache_root}")
    try:
        with torch.inference_mode():
            for batch in dataloader:
                if completed >= steps:
                    break
                if not batch:
                    continue
                input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
                prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
                t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)
                batch_size = input_latents.shape[0]
                timestep_indices, _ = timestep_sampler.sample(batch_size)
                timestep_indices = timestep_indices.to(device=device)
                sigmas, timesteps = anima_ticket_to_sigma_timestep(timestep_indices, config.compute_dtype)
                noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)
                noisy_latents, _ = flowmatch_noise_and_target(input_latents, noise, sigmas)
                sample = {
                    "noisy_latents": noisy_latents.detach().cpu(),
                    "timesteps": timesteps.detach().cpu(),
                    "prompt_emb": prompt_emb.detach().cpu(),
                    "t5xxl_ids": t5xxl_ids.detach().cpu(),
                }
                capture_store = {}
                teacher = run_one_sample(sample).detach().cpu().float().contiguous()
                captured_tensors = len(capture_store)
                cache_path = cache_root / f"batch_{completed + 1:04d}.pt"
                torch.save(
                    {
                        "sample": sample,
                        "base_layer_outputs": capture_store,
                        "teacher_output": teacher,
                    },
                    cache_path,
                )
                cache_paths.append(cache_path)
                capture_store = None
                completed += 1
                log(f"- captured base batch {completed}/{steps} ({captured_tensors:,} layer tensors, {cache_path.stat().st_size / (1024 ** 2):.1f} MiB)")
                del input_latents, prompt_emb, t5xxl_ids, timesteps, sigmas, noise, noisy_latents, teacher, sample
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        capture_store = None

    if not cache_paths:
        for handle in handles:
            handle.remove()
        raise RuntimeError("No calibration samples were cached.")

    def load_cached_batch(cache_path: Path) -> dict[str, object]:
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(cache_path, map_location="cpu")

    def apply_global_spec(spec: dict[str, object]) -> dict[str, torch.Tensor]:
        originals: dict[str, torch.Tensor] = {}
        fmt = str(spec["format"])
        storage_dtype = storage_by_format[fmt]
        scale_multiplier = float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            if not format_compatible_with_weight(fmt, weight_key, module.weight):
                continue
            originals[weight_key] = module.weight.detach().clone()
            q_weight = simulated_quant_weight(
                module.weight.detach(),
                storage_dtype,
                fmt,
                module.weight.dtype,
                nvfp4_scale_multiplier=scale_multiplier,
            )
            module.weight.data.copy_(q_weight.to(device=module.weight.device, dtype=module.weight.dtype))
            del q_weight
        return originals

    def apply_selected_specs(selected_specs: dict[str, dict[str, object]]) -> dict[str, torch.Tensor]:
        originals: dict[str, torch.Tensor] = {}
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            spec = selected_specs.get(weight_key)
            if not spec:
                continue
            fmt = str(spec["format"])
            originals[weight_key] = module.weight.detach().clone()
            q_weight = simulated_quant_weight(
                module.weight.detach(),
                storage_by_format[fmt],
                fmt,
                module.weight.dtype,
                nvfp4_scale_multiplier=float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0,
            )
            module.weight.data.copy_(q_weight.to(device=module.weight.device, dtype=module.weight.dtype))
            del q_weight
        return originals

    def restore_originals(originals: dict[str, torch.Tensor]) -> None:
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            original = originals.get(weight_key)
            if original is not None:
                module.weight.data.copy_(original)
        originals.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    candidate_profiles: dict[str, list[dict[str, object]]] = {key: [] for key in specs_by_key}
    final_candidate_profiles: list[dict[str, object]] = []
    log("Running global quantized candidate passes...")
    for spec_index, spec in enumerate(scan_specs, start=1):
        fmt = str(spec["format"])
        label = fmt
        if fmt == "nvfp4":
            label = f"nvfp4@x{float(spec.get('nvfp4_scale_multiplier', 1.0)):g}"
        layer_profiles = {
            key: LayerProfile(label, key, next(module_by_name[name].weight.numel() for name in module_by_name if layer_keys[name] == key))
            for key, specs in specs_by_key.items()
            if any(spec_signature(s) == spec_signature(spec) for s in specs)
        }
        final_profile = LayerProfile(label, "full_model", 0)
        originals = apply_global_spec(spec)
        try:
            with torch.inference_mode():
                for cache_path in cache_paths:
                    cached = load_cached_batch(cache_path)
                    sample = cached["sample"]
                    teacher_cpu = cached["teacher_output"]
                    base_outputs = cached["base_layer_outputs"]
                    capture_store = {}
                    student = run_one_sample(sample).detach().cpu().float().contiguous()
                    quant_layer_outputs = capture_store
                    capture_store = None
                    final_profile.update(teacher_cpu, student)
                    for weight_key, profile in layer_profiles.items():
                        ref = base_outputs.get(weight_key)
                        quant = quant_layer_outputs.get(weight_key)
                        if ref is None or quant is None:
                            continue
                        profile.update(ref, quant)
                    del student, quant_layer_outputs, cached, sample, teacher_cpu, base_outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            capture_store = None
            restore_originals(originals)
        final_candidate_profiles.append({
            "spec": dict(spec),
            "format": fmt,
            "label": label,
            "rel_mse": final_profile.rel_mse,
            "cosine": final_profile.cosine,
            "max_abs_ratio": final_profile.max_abs_ratio,
            "passes": passes_profile(final_profile),
            "score": profile_score(final_profile),
        })
        log(
            f"- [{spec_index}/{len(scan_specs)}] {label:18s} final rel={final_profile.rel_mse:.6g} "
            f"cos={final_profile.cosine:.6f} maxabs={final_profile.max_abs_ratio:.6g}"
        )
        for weight_key, profile in layer_profiles.items():
            passed = passes_profile(profile)
            row = {
                "spec": dict(spec),
                "format": fmt,
                "nvfp4_scale_multiplier": float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else None,
                "rel_mse": profile.rel_mse,
                "cosine": profile.cosine,
                "max_abs_ratio": profile.max_abs_ratio,
                "passes": passed,
                "selected": False,
                "score": profile_score(profile),
                "batches": profile.batches,
                "global_candidate_final_score": profile_score(final_profile),
            }
            candidate_profiles[weight_key].append(row)

    selected: dict[str, dict[str, object]] = {}
    layer_rows = []
    for weight_key in sorted(specs_by_key):
        candidates = candidate_profiles.get(weight_key, [])
        passing = [row for row in candidates if row["passes"]]
        chosen = sorted(passing, key=candidate_selection_key)[0] if passing else None
        chosen_label = "keep"
        if chosen:
            chosen["selected"] = True
            selected[weight_key] = dict(chosen["spec"])
            chosen_label = str(chosen["format"])
            if chosen_label == "nvfp4":
                chosen_label = f"nvfp4@x{float(chosen['nvfp4_scale_multiplier']):g}"
        layer_rows.append({
            "weight_key": weight_key,
            "chosen": chosen_label,
            "chosen_spec": dict(chosen["spec"]) if chosen else None,
            "formats": [{k: v for k, v in row.items() if k != "spec"} | {"spec": row["spec"]} for row in candidates],
        })

    def score_selected_profile(selected_specs: dict[str, dict[str, object]], label: str) -> LayerProfile:
        originals = apply_selected_specs(selected_specs)
        profile = LayerProfile(label, "full_mixed_profile", 0)
        try:
            with torch.inference_mode():
                for cache_path in cache_paths:
                    cached = load_cached_batch(cache_path)
                    sample = cached["sample"]
                    teacher_cpu = cached["teacher_output"]
                    capture_store = None
                    student = run_one_sample(sample).detach().cpu().float().contiguous()
                    profile.update(teacher_cpu, student)
                    del student, cached, sample, teacher_cpu
        finally:
            restore_originals(originals)
        return profile

    global_history = []
    if selected:
        log("\nValidating selected mixed profile...")
        full_profile = score_selected_profile(selected, "fast_global_mixed_profile")
        full_passes = passes_profile(full_profile)
        global_history.append({
            "promotion_step": 0,
            "rel_mse": full_profile.rel_mse,
            "cosine": full_profile.cosine,
            "max_abs_ratio": full_profile.max_abs_ratio,
            "passes": full_passes,
            "selected_layers": len(selected),
        })
        log(
            f"- mixed profile rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
            f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
        )
        promotion_step = 0
        while not full_passes and promotion_step < global_promote_steps and selected:
            worst_key = None
            worst_score = -1.0
            next_spec = None
            for weight_key, current_spec in selected.items():
                rows = candidate_profiles.get(weight_key, [])
                current_sig = spec_signature(current_spec)
                current_row = next((row for row in rows if spec_signature(row["spec"]) == current_sig), None)
                if current_row is None:
                    continue
                current_tier = spec_storage_tier(current_spec)
                larger = [row for row in rows if row["passes"] and spec_storage_tier(row["spec"]) > current_tier]
                replacement = sorted(larger, key=candidate_selection_key)[0]["spec"] if larger else None
                score = float(current_row["score"])
                if score > worst_score:
                    worst_key = weight_key
                    worst_score = score
                    next_spec = replacement
            if worst_key is None:
                break
            promotion_step += 1
            if next_spec is None:
                selected.pop(worst_key, None)
                promoted_label = "keep"
            else:
                selected[worst_key] = dict(next_spec)
                promoted_label = str(next_spec["format"])
                if promoted_label == "nvfp4":
                    promoted_label = f"nvfp4@x{float(next_spec.get('nvfp4_scale_multiplier', 1.0)):g}"
            log(f"- promotion {promotion_step}/{global_promote_steps}: {worst_key} -> {promoted_label}")
            full_profile = score_selected_profile(selected, "fast_global_mixed_profile")
            full_passes = passes_profile(full_profile)
            global_history.append({
                "promotion_step": promotion_step,
                "promoted_key": worst_key,
                "promoted_to": dict(next_spec) if next_spec else None,
                "rel_mse": full_profile.rel_mse,
                "cosine": full_profile.cosine,
                "max_abs_ratio": full_profile.max_abs_ratio,
                "passes": full_passes,
                "selected_layers": len(selected),
            })
            log(
                f"  mixed rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
                f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
            )

    selected_sigs = {key: spec_signature(spec) for key, spec in selected.items()}
    for row in layer_rows:
        weight_key = row["weight_key"]
        selected_sig = selected_sigs.get(weight_key)
        chosen_label = "keep"
        chosen_spec = None
        for item in row["formats"]:
            is_selected = selected_sig is not None and spec_signature(item["spec"]) == selected_sig
            item["selected"] = is_selected
            if is_selected:
                chosen_spec = dict(item["spec"])
                chosen_label = str(item["format"])
                if chosen_label == "nvfp4":
                    chosen_label = f"nvfp4@x{float(item['nvfp4_scale_multiplier']):g}"
        row["chosen"] = chosen_label
        row["chosen_spec"] = chosen_spec

    counts = defaultdict(int)
    nvfp4_scale_counts = defaultdict(int)
    for spec in selected.values():
        fmt = str(spec["format"])
        counts[fmt] += 1
        if fmt == "nvfp4":
            nvfp4_scale_counts[float(spec.get("nvfp4_scale_multiplier", 1.0))] += 1
    kept = len(layer_keys) - len(selected)
    log("\nFast Dynamic global-forward summary:")
    log(f"- thresholds: rel_mse <= {max_rel_mse:g}, cosine >= {min_cosine:g}, max_abs_ratio <= {max_abs_ratio:g}")
    for fmt, count in sorted(counts.items(), key=lambda item: quant_format_sort_key(item[0])):
        log(f"- {fmt:18s}: {count:,} layer weight(s)")
    if nvfp4_scale_counts:
        log("- NVFP4 selected scale multipliers:")
        for scale_multiplier, count in sorted(nvfp4_scale_counts.items()):
            log(f"  - x{scale_multiplier:g}: {count:,} layer weight(s)")
    log(f"- {'keep':18s}: {kept:,} layer weight(s)")
    if global_history:
        last = global_history[-1]
        log(
            f"- final global drift: rel={last['rel_mse']:.6g}, cos={last['cosine']:.6f}, "
            f"maxabs={last['max_abs_ratio']:.6g}, {'PASS' if last['passes'] else 'fail'}"
        )

    profile_path = Path(args.output).with_suffix(".mixed_profile.json")
    payload = {
        "source": str(input_path),
        "mode": "fast_dynamic_global_forward",
        "description": "One base activation capture plus one all-layer quantized forward per candidate; layer decisions are made by comparing captured Linear outputs.",
        "formats": mixed_formats,
        "nvfp4_scale_multipliers": nvfp4_scale_multipliers,
        "steps": completed,
        "capture_dtype": str(capture_dtype).replace("torch.", ""),
        "thresholds": {"rel_mse": max_rel_mse, "cosine": min_cosine, "max_abs_ratio": max_abs_ratio},
        "global_promote_steps": global_promote_steps,
        "scan_candidates": len(scan_specs),
        "global_candidate_profiles": final_candidate_profiles,
        "global_history": global_history,
        "selected": dict(sorted(selected.items())),
        "layers": layer_rows,
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"- wrote profile: {profile_path}")

    for handle in handles:
        handle.remove()
    for cache_path in cache_paths:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError as exc:
            log(f"WARNING: Could not delete activation cache file {cache_path}: {exc}")
    try:
        cache_root.rmdir()
    except OSError:
        pass
    dit.cpu()
    pipe.dit = None
    del dit, pipe, cache_paths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected

def compression_first_profile(
    args: SimpleNamespace,
    log=print,
) -> dict[str, dict[str, object]]:
    """Compression-first mixed profile.

    This path is built for size, not forensic layer proof. It runs a small number
    of real DiT forwards to collect per-channel Linear input energy, estimates
    quantization damage as E[x^2] * (W - Q(W))^2, then starts with a target-heavy
    4-bit allocation and promotes only the highest-risk layers.
    """
    from torch.utils.data import DataLoader
    from train import PrecomputedImageBatchSampler, build_image_batch_schedule, set_seed
    from train_anima import (
        AnimaCachedDataset,
        AnimaTimestepSampler,
        anima_collate_fn,
        flowmatch_noise_and_target,
        load_anima_pipe,
        normalize_anima_config,
        precompute_and_cache_anima,
        run_dit_forward,
        anima_ticket_to_sigma_timestep,
    )

    input_path = Path(args.input)
    config = load_anima_config_from_json(getattr(args, "config"), input_path)
    dataset_overrides = [str(Path(p)) for p in getattr(args, "dataset_overrides", []) or [] if str(p).strip()]
    if dataset_overrides:
        config.INSTANCE_DATASETS = [{"path": path, "repeats": 1} for path in dataset_overrides]
    normalize_anima_config(config)
    if config.SEED:
        set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_formats = parse_mixed_formats(getattr(args, "mixed_formats", "nvfp4,e4m3,int8"))
    requested_nvfp4_scales = parse_float_list(
        getattr(args, "nvfp4_scale_multipliers", DEFAULT_NVFP4_SCALE_MULTIPLIERS),
        name="NVFP4 scale multipliers",
    )
    # Compression-first should be quick. A few representative scales are enough
    # for ranking; the exhaustive scale list belongs to the audit-style modes.
    preferred_scales = [0.7, 1.0, 1.25]
    nvfp4_scale_multipliers = [s for s in preferred_scales if any(abs(s - r) < 1.0e-9 for r in requested_nvfp4_scales)]
    if not nvfp4_scale_multipliers:
        nvfp4_scale_multipliers = requested_nvfp4_scales[:3]
    storage_by_format = {fmt: quant_storage_dtype(fmt) for fmt in mixed_formats}

    strict_rel = float(getattr(args, "calibration_rel_mse", 0.002))
    strict_cos = float(getattr(args, "calibration_cosine", 0.999))
    strict_abs = float(getattr(args, "calibration_max_abs_ratio", 0.10))
    # Image quality does not require near-identical tensor output. Use a looser
    # default for this compression mode, while still honoring stricter values if
    # a caller explicitly provides already-loose ones.
    max_rel_mse = max(strict_rel, 0.010)
    min_cosine = min(strict_cos, 0.995)
    max_abs_ratio = max(strict_abs, 0.20)
    global_promote_steps = max(0, int(getattr(args, "dynamic_global_promote_steps", 32) or 0))
    composition_guard_enabled = bool(getattr(args, "composition_guard", False))
    target_4bit_ratio = float(getattr(args, "compression_4bit_ratio", 0.75) or 0.75)
    keep_worst_ratio = float(getattr(args, "compression_keep_worst_ratio", 0.03) or 0.03)
    target_4bit_ratio = min(max(target_4bit_ratio, 0.0), 1.0)
    keep_worst_ratio = min(max(keep_worst_ratio, 0.0), 0.25)

    log("\nCompression First profiling:")
    log(f"- config: {getattr(args, 'config')}")
    log(f"- formats: {', '.join(mixed_formats)}")
    if "nvfp4" in mixed_formats:
        log(f"- NVFP4 ranking scales: {', '.join(f'{v:g}' for v in nvfp4_scale_multipliers)}")
    log(f"- device: {device}")
    log(f"- calibration batches: {getattr(args, 'calibration_steps')}")
    log(f"- allocation target: {target_4bit_ratio:.0%} 4-bit candidates, worst {keep_worst_ratio:.0%} kept")
    log(f"- validation thresholds: rel <= {max_rel_mse:g}, cosine >= {min_cosine:g}, maxabs <= {max_abs_ratio:g}")
    log(
        "- composition guard: "
        + ("on, 4-bit disabled for cross-attn, q/k, modulation, and edge blocks" if composition_guard_enabled else "off")
    )

    required_paths = {
        "DIT_PATH": config.DIT_PATH,
        "VAE_PATH": config.VAE_PATH,
        "TEXT_ENCODER_PATH": config.TEXT_ENCODER_PATH,
    }
    for label, value in required_paths.items():
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{label} is required for compression-first profiling: {value}")

    log("Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))
    log("Preparing/reusing Anima cache...")
    precompute_and_cache_anima(config, pipe, device)
    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit
    dit.to(device=device, dtype=config.compute_dtype)
    dit.eval()

    def spec_signature(spec: dict[str, object] | None) -> tuple:
        if not spec:
            return ("keep", None)
        fmt = str(spec.get("format", "keep"))
        if fmt == "nvfp4":
            return (fmt, round(float(spec.get("nvfp4_scale_multiplier", 1.0)), 12))
        return (fmt, None)

    def spec_sort_key(spec: dict[str, object]) -> tuple:
        fmt = str(spec["format"])
        base = quant_format_sort_key(fmt)
        if fmt == "nvfp4":
            return base + (float(spec.get("nvfp4_scale_multiplier", 1.0)),)
        return base + (0.0,)

    def spec_storage_tier(spec: dict[str, object] | None) -> int:
        if not spec:
            return 3
        fmt = str(spec["format"])
        if fmt == "ternary_1_58":
            return 0
        if fmt == "nvfp4":
            return 1
        if fmt in {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise"}:
            return 2
        return 3

    def candidate_specs(weight_key: str, weight: torch.Tensor) -> list[dict[str, object]]:
        specs = []
        for fmt in mixed_formats:
            if not format_compatible_with_weight(fmt, weight_key, weight):
                continue
            if fmt == "nvfp4":
                for scale_multiplier in nvfp4_scale_multipliers:
                    specs.append({"format": fmt, "nvfp4_scale_multiplier": float(scale_multiplier)})
            else:
                specs.append({"format": fmt})
        return sorted(specs, key=spec_sort_key)

    def passes_profile(profile: LayerProfile) -> bool:
        return (
            profile.count > 0
            and profile.rel_mse <= max_rel_mse
            and profile.cosine >= min_cosine
            and profile.max_abs_ratio <= max_abs_ratio
        )

    def profile_score(profile: LayerProfile) -> float:
        return (
            profile.rel_mse / max(max_rel_mse, 1.0e-12)
            + max(0.0, (min_cosine - profile.cosine) / max(1.0 - min_cosine, 1.0e-12))
            + profile.max_abs_ratio / max(max_abs_ratio, 1.0e-12)
        )

    module_by_name: dict[str, torch.nn.Linear] = {}
    layer_keys: dict[str, str] = {}
    specs_by_key: dict[str, list[dict[str, object]]] = {}
    input_stats: dict[str, dict[str, object]] = {}
    block_indices = [idx for idx in (_block_index_from_key(name) for name, _module in dit.named_modules()) if idx is not None]
    total_blocks = (max(block_indices) + 1) if block_indices else None
    for name, module in dit.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = f"{name}.weight"
        if not is_full_quant_compatible_name(key):
            continue
        specs = candidate_specs(key, module.weight)
        if not specs:
            continue
        module_by_name[name] = module
        layer_keys[name] = key
        specs_by_key[key] = specs
        input_stats[key] = {
            "sum_sq": torch.zeros(int(module.weight.shape[1]), dtype=torch.float32),
            "count": 0,
            "input_abs_max": 0.0,
            "weight_abs_max": float(module.weight.detach().float().abs().amax().item()),
        }

    if not module_by_name:
        raise RuntimeError("No candidate Linear layers found for compression-first profiling.")
    log(f"Candidate Linear layers: {len(module_by_name):,}")

    handles = []

    def make_input_stat_hook(name: str):
        weight_key = layer_keys[name]

        def hook(_layer, layer_input, _output):
            x = layer_input[0]
            if not torch.is_tensor(x) or not torch.is_floating_point(x) or x.shape[-1] != input_stats[weight_key]["sum_sq"].numel():
                return
            xf = x.detach().float().reshape(-1, x.shape[-1])
            stats = input_stats[weight_key]
            stats["sum_sq"] += xf.pow(2).sum(dim=0).cpu()
            stats["count"] = int(stats["count"]) + int(xf.shape[0])
            stats["input_abs_max"] = max(float(stats["input_abs_max"]), float(xf.abs().amax().item()) if xf.numel() else 0.0)

        return hook

    for name, module in module_by_name.items():
        handles.append(module.register_forward_hook(make_input_stat_hook(name)))

    def extract_output_tensor(output) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item
        if isinstance(output, dict):
            for key in ("sample", "model_pred", "prediction", "pred", "output", "x"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        raise RuntimeError("run_dit_forward did not return a tensor-like output.")

    def run_one_sample(sample: dict[str, torch.Tensor]) -> torch.Tensor:
        noisy_latents = sample["noisy_latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        timesteps = sample["timesteps"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        prompt_emb = sample["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        t5xxl_ids = sample["t5xxl_ids"].to(device=device, non_blocking=True)
        output = run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
        return extract_output_tensor(output)

    dataset = AnimaCachedDataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("No cached Anima dataset items found for compression-first profiling.")
    total_scheduler_timesteps = 1000
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    steps = max(1, int(getattr(args, "calibration_steps", 8) or 8))
    image_schedule = build_image_batch_schedule(
        dataset,
        steps,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        True,
    )
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, 0)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)

    cache_root = Path(args.output).with_name(f".{Path(args.output).stem}.compression_first_cache_{uuid.uuid4().hex[:8]}")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths: list[Path] = []
    completed = 0
    log("Collecting activation statistics and teacher outputs...")
    try:
        with torch.inference_mode():
            for batch in dataloader:
                if completed >= steps:
                    break
                if not batch:
                    continue
                input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
                prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
                t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)
                batch_size = input_latents.shape[0]
                timestep_indices, _ = timestep_sampler.sample(batch_size)
                timestep_indices = timestep_indices.to(device=device)
                sigmas, timesteps = anima_ticket_to_sigma_timestep(timestep_indices, config.compute_dtype)
                noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)
                noisy_latents, _ = flowmatch_noise_and_target(input_latents, noise, sigmas)
                sample = {
                    "noisy_latents": noisy_latents.detach().cpu(),
                    "timesteps": timesteps.detach().cpu(),
                    "prompt_emb": prompt_emb.detach().cpu(),
                    "t5xxl_ids": t5xxl_ids.detach().cpu(),
                }
                teacher = run_one_sample(sample).detach().cpu().float().contiguous()
                cache_path = cache_root / f"batch_{completed + 1:04d}.pt"
                torch.save({"sample": sample, "teacher_output": teacher}, cache_path)
                cache_paths.append(cache_path)
                completed += 1
                log(f"- profiled batch {completed}/{steps} ({cache_path.stat().st_size / (1024 ** 2):.1f} MiB replay cache)")
                del input_latents, prompt_emb, t5xxl_ids, timesteps, sigmas, noise, noisy_latents, teacher, sample
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    if not cache_paths:
        raise RuntimeError("No calibration samples were cached.")

    def load_cached_batch(cache_path: Path) -> dict[str, object]:
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(cache_path, map_location="cpu")

    def estimate_candidate(weight: torch.Tensor, spec: dict[str, object], act_mean_sq: torch.Tensor) -> dict[str, object]:
        fmt = str(spec["format"])
        weight_cpu = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        q_weight = simulated_quant_weight(
            weight_cpu,
            storage_by_format[fmt],
            fmt,
            torch.float32,
            nvfp4_scale_multiplier=float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0,
        ).to(dtype=torch.float32)
        delta = q_weight - weight_cpu
        act = act_mean_sq.to(dtype=torch.float32).clamp(min=1.0e-12)
        weighted_noise = (delta.pow(2) * act.unsqueeze(0)).sum()
        weighted_signal = (weight_cpu.pow(2) * act.unsqueeze(0)).sum().clamp(min=1.0e-12)
        rel = float((weighted_noise / weighted_signal).item())
        max_ratio = float((delta.abs().amax() / weight_cpu.abs().amax().clamp(min=1.0e-12)).item()) if weight_cpu.numel() else 0.0
        del q_weight, delta, weight_cpu
        return {
            "spec": dict(spec),
            "format": fmt,
            "nvfp4_scale_multiplier": float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else None,
            "estimated_rel_noise": rel,
            "estimated_max_weight_ratio": max_ratio,
            "score": rel + 0.05 * max_ratio,
            "selected": False,
        }

    log("Scoring layer quantization risk from activation-weighted weight error...")
    layer_rows = []
    candidate_profiles: dict[str, list[dict[str, object]]] = {}
    selected: dict[str, dict[str, object]] = {}
    for index, (name, module) in enumerate(sorted(module_by_name.items(), key=lambda item: layer_keys[item[0]]), start=1):
        weight_key = layer_keys[name]
        stats = input_stats[weight_key]
        count = max(int(stats["count"]), 1)
        act_mean_sq = stats["sum_sq"] / count
        rows = [estimate_candidate(module.weight, spec, act_mean_sq) for spec in specs_by_key[weight_key]]
        candidate_profiles[weight_key] = rows
        if index % 25 == 0 or index == len(module_by_name):
            log(f"- scored {index}/{len(module_by_name)} layer(s)")

    low_bit_candidates = []
    protected_low_bit_count = 0
    for weight_key, rows in candidate_profiles.items():
        composition_sensitive = composition_guard_enabled and is_composition_sensitive_weight(weight_key, total_blocks)
        for row in rows:
            row["composition_sensitive"] = composition_sensitive
        ternary_rows = [] if composition_sensitive else [row for row in rows if row["format"] == "ternary_1_58"]
        nv_rows = [] if composition_sensitive else [row for row in rows if row["format"] == "nvfp4"]
        higher_rows = [row for row in rows if row["format"] not in {"ternary_1_58", "nvfp4"}]
        best_ternary = sorted(ternary_rows, key=lambda row: (float(row["score"]), spec_sort_key(row["spec"])))[0] if ternary_rows else None
        best_nv = sorted(nv_rows, key=lambda row: (float(row["score"]), spec_sort_key(row["spec"])))[0] if nv_rows else None
        best_higher = sorted(higher_rows, key=lambda row: (float(row["score"]), spec_sort_key(row["spec"])))[0] if higher_rows else None
        lowest = best_ternary or best_nv or best_higher
        risk = float(lowest["score"] if lowest else 1.0e9)
        low_bit_candidates.append((risk, weight_key, best_ternary, best_nv, best_higher, composition_sensitive))
        if composition_sensitive:
            protected_low_bit_count += 1

    low_bit_candidates.sort(key=lambda item: item[0])
    compressible_candidates = [item for item in low_bit_candidates if item[2] is not None or item[3] is not None]
    candidate_count = len(low_bit_candidates)
    keep_count = int(round(candidate_count * keep_worst_ratio))
    low_bit_goal = int(round(len(compressible_candidates) * target_4bit_ratio))
    keep_keys = {
        key for _risk, key, _best_ternary, _best_nv, _best_higher, _sensitive
        in low_bit_candidates[-keep_count:]
    } if keep_count > 0 else set()
    low_bit_used = 0
    for _risk, weight_key, best_ternary, best_nv, best_higher, composition_sensitive in low_bit_candidates:
        chosen = None
        if weight_key in keep_keys:
            chosen = None
        elif best_ternary is not None and low_bit_used < low_bit_goal:
            chosen = best_ternary
            low_bit_used += 1
        elif best_nv is not None:
            chosen = best_nv
        elif best_higher is not None:
            chosen = best_higher
        elif best_ternary is not None:
            chosen = best_ternary
            low_bit_used += 1
        if chosen is not None:
            selected[weight_key] = dict(chosen["spec"])
            chosen["selected"] = True
    log(
        f"- composition guard: protected {protected_low_bit_count:,} layer(s) from ternary/NVFP4"
        if composition_guard_enabled else "- composition guard: off"
    )

    def apply_selected_specs(selected_specs: dict[str, dict[str, object]]) -> dict[str, torch.Tensor]:
        originals: dict[str, torch.Tensor] = {}
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            spec = selected_specs.get(weight_key)
            if not spec:
                continue
            fmt = str(spec["format"])
            originals[weight_key] = module.weight.detach().clone()
            q_weight = simulated_quant_weight(
                module.weight.detach(),
                storage_by_format[fmt],
                fmt,
                module.weight.dtype,
                nvfp4_scale_multiplier=float(spec.get("nvfp4_scale_multiplier", 1.0)) if fmt == "nvfp4" else 1.0,
            )
            module.weight.data.copy_(q_weight.to(device=module.weight.device, dtype=module.weight.dtype))
            del q_weight
        return originals

    def restore_originals(originals: dict[str, torch.Tensor]) -> None:
        for name, module in module_by_name.items():
            weight_key = layer_keys[name]
            original = originals.get(weight_key)
            if original is not None:
                module.weight.data.copy_(original)
        originals.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_selected_profile(selected_specs: dict[str, dict[str, object]], label: str) -> LayerProfile:
        originals = apply_selected_specs(selected_specs)
        profile = LayerProfile(label, "full_mixed_profile", 0)
        try:
            with torch.inference_mode():
                for cache_path in cache_paths:
                    cached = load_cached_batch(cache_path)
                    sample = cached["sample"]
                    teacher_cpu = cached["teacher_output"]
                    student = run_one_sample(sample).detach().cpu().float().contiguous()
                    profile.update(teacher_cpu, student)
                    del student, cached, sample, teacher_cpu
        finally:
            restore_originals(originals)
        return profile

    global_history = []
    if selected:
        log("\nValidating compression-first mixed profile...")
        full_profile = score_selected_profile(selected, "compression_first_profile")
        full_passes = passes_profile(full_profile)
        global_history.append({
            "promotion_step": 0,
            "rel_mse": full_profile.rel_mse,
            "cosine": full_profile.cosine,
            "max_abs_ratio": full_profile.max_abs_ratio,
            "passes": full_passes,
            "selected_layers": len(selected),
        })
        log(
            f"- mixed profile rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
            f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
        )
        promotion_step = 0
        while not full_passes and promotion_step < global_promote_steps and selected:
            worst_key = None
            worst_score = -1.0
            next_spec = None
            for weight_key, current_spec in selected.items():
                rows = candidate_profiles.get(weight_key, [])
                current_sig = spec_signature(current_spec)
                current_row = next((row for row in rows if spec_signature(row["spec"]) == current_sig), None)
                if current_row is None:
                    continue
                current_tier = spec_storage_tier(current_spec)
                larger = [row for row in rows if spec_storage_tier(row["spec"]) > current_tier]
                replacement = sorted(larger, key=lambda row: (float(row["score"]), spec_sort_key(row["spec"])))[0]["spec"] if larger else None
                score = float(current_row["score"])
                if score > worst_score:
                    worst_key = weight_key
                    worst_score = score
                    next_spec = replacement
            if worst_key is None:
                break
            promotion_step += 1
            if next_spec is None:
                selected.pop(worst_key, None)
                promoted_label = "keep"
            else:
                selected[worst_key] = dict(next_spec)
                promoted_label = str(next_spec["format"])
                if promoted_label == "nvfp4":
                    promoted_label = f"nvfp4@x{float(next_spec.get('nvfp4_scale_multiplier', 1.0)):g}"
            log(f"- promotion {promotion_step}/{global_promote_steps}: {worst_key} -> {promoted_label}")
            full_profile = score_selected_profile(selected, "compression_first_profile")
            full_passes = passes_profile(full_profile)
            global_history.append({
                "promotion_step": promotion_step,
                "promoted_key": worst_key,
                "promoted_to": dict(next_spec) if next_spec else None,
                "rel_mse": full_profile.rel_mse,
                "cosine": full_profile.cosine,
                "max_abs_ratio": full_profile.max_abs_ratio,
                "passes": full_passes,
                "selected_layers": len(selected),
            })
            log(
                f"  mixed rel={full_profile.rel_mse:.6g} cos={full_profile.cosine:.6f} "
                f"maxabs={full_profile.max_abs_ratio:.6g} {'PASS' if full_passes else 'fail'}"
            )

    selected_sigs = {key: spec_signature(spec) for key, spec in selected.items()}
    for weight_key in sorted(specs_by_key):
        rows = candidate_profiles.get(weight_key, [])
        selected_sig = selected_sigs.get(weight_key)
        chosen_label = "keep"
        chosen_spec = None
        for row in rows:
            is_selected = selected_sig is not None and spec_signature(row["spec"]) == selected_sig
            row["selected"] = is_selected
            if is_selected:
                chosen_spec = dict(row["spec"])
                chosen_label = str(row["format"])
                if chosen_label == "nvfp4":
                    chosen_label = f"nvfp4@x{float(row['nvfp4_scale_multiplier']):g}"
        stats = input_stats[weight_key]
        layer_rows.append({
            "weight_key": weight_key,
            "chosen": chosen_label,
            "chosen_spec": chosen_spec,
            "input_abs_max": float(stats["input_abs_max"]),
            "weight_abs_max": float(stats["weight_abs_max"]),
            "formats": [{k: v for k, v in row.items() if k != "spec"} | {"spec": row["spec"]} for row in rows],
        })

    counts = defaultdict(int)
    nvfp4_scale_counts = defaultdict(int)
    for spec in selected.values():
        fmt = str(spec["format"])
        counts[fmt] += 1
        if fmt == "nvfp4":
            nvfp4_scale_counts[float(spec.get("nvfp4_scale_multiplier", 1.0))] += 1
    kept = len(layer_keys) - len(selected)
    log("\nCompression First summary:")
    for fmt, count in sorted(counts.items(), key=lambda item: quant_format_sort_key(item[0])):
        log(f"- {fmt:18s}: {count:,} layer weight(s)")
    if nvfp4_scale_counts:
        log("- NVFP4 selected scale multipliers:")
        for scale_multiplier, count in sorted(nvfp4_scale_counts.items()):
            log(f"  - x{scale_multiplier:g}: {count:,} layer weight(s)")
    log(f"- {'keep':18s}: {kept:,} layer weight(s)")
    if global_history:
        last = global_history[-1]
        log(
            f"- final global drift: rel={last['rel_mse']:.6g}, cos={last['cosine']:.6f}, "
            f"maxabs={last['max_abs_ratio']:.6g}, {'PASS' if last['passes'] else 'fail'}"
        )

    profile_path = Path(args.output).with_suffix(".mixed_profile.json")
    payload = {
        "source": str(input_path),
        "mode": "compression_first_activation_weighted",
        "description": "Activation-weighted weight-error ranking: collect Linear input energy, allocate ternary/NVFP4 where requested, promote highest-risk layers to higher precision or keep, then validate final DiT output.",
        "formats": mixed_formats,
        "nvfp4_scale_multipliers": nvfp4_scale_multipliers,
        "requested_nvfp4_scale_multipliers": requested_nvfp4_scales,
        "steps": completed,
        "thresholds": {"rel_mse": max_rel_mse, "cosine": min_cosine, "max_abs_ratio": max_abs_ratio},
        "allocation": {"target_4bit_ratio": target_4bit_ratio, "keep_worst_ratio": keep_worst_ratio},
        "composition_guard": {
            "enabled": composition_guard_enabled,
            "protected_from_low_bit": protected_low_bit_count,
            "total_blocks": total_blocks,
        },
        "global_promote_steps": global_promote_steps,
        "global_history": global_history,
        "selected": dict(sorted(selected.items())),
        "layers": layer_rows,
    }
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"- wrote profile: {profile_path}")

    for cache_path in cache_paths:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError as exc:
            log(f"WARNING: Could not delete replay cache file {cache_path}: {exc}")
    try:
        cache_root.rmdir()
    except OSError:
        pass
    dit.cpu()
    pipe.dit = None
    del dit, pipe, cache_paths
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selected

GGUF_QUANT_CHOICES = {"q8_0", "q5_1", "q5_0", "q4_0"}


def convert_checkpoint_gguf(args: SimpleNamespace, log=print) -> Path | None:
    """Write an experimental ComfyUI-GGUF Anima diffusion model."""
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("GGUF export requires the 'gguf' package: pip install gguf") from exc

    choice = str(args.fp8).strip().lower()
    quant_types = {
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q5_0": gguf.GGMLQuantizationType.Q5_0,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
    }
    file_types = {
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q5_1": gguf.LlamaFileType.MOSTLY_Q5_1,
        "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }
    target_qtype = quant_types[choice]

    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".gguf":
        output_path = output_path.with_suffix(".gguf")
        args.output = str(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output_path} already exists; enable overwrite to replace it.")

    plan: list[tuple[str, torch.Tensor, object]] = []
    counts = defaultdict(int)
    input_bytes = output_bytes = 0
    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key).cpu().contiguous()
            input_bytes += tensor.numel() * tensor.element_size()
            qtype = gguf.GGMLQuantizationType.F32
            if torch.is_floating_point(tensor):
                # GGML quants operate in fixed-size blocks. Restrict them to
                # matrix weights; norms, biases and incompatible matrices stay
                # lossless.
                quant_block = gguf.GGML_QUANT_SIZES[target_qtype][0]
                if tensor.ndim == 2 and tensor.numel() >= 1024 and tensor.shape[-1] % quant_block == 0:
                    qtype = target_qtype
                elif tensor.ndim > 1:
                    qtype = gguf.GGMLQuantizationType.BF16
            counts[qtype.name] += 1
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            output_bytes += ((tensor.numel() + block_size - 1) // block_size) * type_size
            plan.append((key, tensor, qtype))

    log(f"Input:  {input_path}")
    log(f"Output: {output_path}")
    log(f"Format: experimental ComfyUI-GGUF {choice.upper()} (architecture=cosmos/Anima key layout)")
    for qtype, count in sorted(counts.items()):
        log(f"- {qtype:8s}: {count:,} tensor(s)")
    log(f"Estimated tensor bytes: {input_bytes/(1024**2):.2f} MiB -> {output_bytes/(1024**2):.2f} MiB")
    if args.dry_run:
        log("Dry run complete; no GGUF file written.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path=None, arch="cosmos")
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(file_types[choice])
    writer.add_string("aozora.source", input_path.name)
    writer.add_string("aozora.quantization", choice)
    try:
        for index, (key, tensor, qtype) in enumerate(plan, start=1):
            data = tensor.float().numpy()
            try:
                quantized = gguf.quantize(data, qtype)
            except gguf.QuantError as exc:
                log(f"WARNING: {key} could not use {qtype.name}; keeping F32 ({exc})")
                qtype = gguf.GGMLQuantizationType.F32
                quantized = gguf.quantize(data, qtype)
            writer.add_tensor(key, quantized, raw_dtype=qtype)
            if index % 50 == 0 or index == len(plan):
                log(f"Prepared {index:,}/{len(plan):,} tensors...")
        writer.write_header_to_file(path=output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()
    log(f"Wrote: {output_path}")
    log(f"File size: {output_path.stat().st_size/(1024**2):.2f} MiB")
    return output_path


def convert_checkpoint(args: SimpleNamespace, log=print) -> Path | None:
    if str(args.fp8).strip().lower() in GGUF_QUANT_CHOICES:
        if normalized_mode(getattr(args, "mode", "simple")) != "simple":
            raise ValueError("GGUF quantization is available in Simple mode only.")
        return convert_checkpoint_gguf(args, log)
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Always use a standard checkpoint suffix so ComfyUI can discover the file.
    if output_path.suffix.lower() not in {".safetensors", ".safetensor"}:
        output_path = output_path.with_name(output_path.name + ".safetensors")
        args.output = str(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")

    mode = normalized_mode(getattr(args, "mode", "simple"))
    format_name = quant_format_name(args.fp8)
    storage_dtype = quant_storage_dtype(format_name)
    if format_name in {"int8_tensorwise", "nvfp4"} and getattr(args, "raw_fp8", False):
        raise ValueError("INT8/NVFP4 require Comfy side tensors; --raw-fp8 is FP8-only.")

    keep_dtype = KEEP_DTYPES[getattr(args, "keep_dtype", "bfloat16")]
    include_patterns = compile_patterns(getattr(args, "include_regex", []) or [])
    exclude_patterns = compile_patterns(getattr(args, "exclude_regex", []) or [])
    quantize_keys = set(getattr(args, "quantize_keys", None) or [])
    raw_quantize_specs = getattr(args, "quantize_specs", None)
    if raw_quantize_specs is None:
        raw_quantize_specs = getattr(args, "quantize_formats", None)
    quantize_specs = dict(raw_quantize_specs or {})
    quantize_formats = {
        key: normalized_quant_spec(spec)["format"]
        for key, spec in quantize_specs.items()
    }
    zero_keys = set(getattr(args, "zero_keys", None) or [])
    use_comfy_quant = not bool(getattr(args, "raw_fp8", False))

    # Dynamic calibration is always strict: only keys explicitly selected by the scan are quantized.
    dynamic_strict = bool(quantize_keys or quantize_specs)

    records = []
    reason_counts = defaultdict(int)
    dtype_counts = defaultdict(lambda: {"tensors": 0, "elements": 0, "bytes": 0})
    in_bytes = 0
    out_bytes = 0
    original_metadata = {}
    calibrated_matched_keys = set()
    comfy_quant_layers: dict[str, dict[str, object]] = {}

    log(f"Input:  {input_path}")
    log(f"Output: {output_path}")
    log(f"Mode:   {mode}, quant={format_name}, keep={getattr(args, 'keep_dtype', 'bfloat16')}")
    log("Scan:   all compatible matrix weights; protected tensors are excluded")
    log(f"Format: {'Comfy scaled quant' if use_comfy_quant else 'raw safetensors FP8'}")
    if quantize_specs:
        log(f"Dynamic strict keys: {len(quantize_specs):,}")
    elif quantize_keys:
        log(f"Dynamic strict keys: {len(quantize_keys):,}")

    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        original_metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            current_format_name = format_name
            current_storage_dtype = storage_dtype
            current_nvfp4_scale_multiplier = float(getattr(args, "nvfp4_scale_multiplier", 1.0) or 1.0)

            if key in zero_keys or should_prune_tensor(key, tensor, args):
                tensor = torch.zeros_like(tensor)
                save_dtype, reason = (keep_dtype or tensor.dtype), "pruned"
            elif quantize_specs:
                matched_spec = calibrated_spec_for_key(key, quantize_specs)
                if matched_spec is not None:
                    calibrated_matched_keys.add(key)
                    matched_format = str(matched_spec["format"])
                    current_format_name = matched_format
                    current_storage_dtype = quant_storage_dtype(matched_format)
                    current_nvfp4_scale_multiplier = float(
                        matched_spec.get("nvfp4_scale_multiplier", current_nvfp4_scale_multiplier)
                    )
                    if matched_format == "nvfp4":
                        save_dtype, reason = current_storage_dtype, f"dynamic_{matched_format}_x{current_nvfp4_scale_multiplier:g}"
                    else:
                        save_dtype, reason = current_storage_dtype, f"dynamic_{matched_format}"
                else:
                    save_dtype, reason = keep_dtype or tensor.dtype, "kept_strict"
            elif quantize_keys:
                if calibrated_key_matches(key, quantize_keys):
                    calibrated_matched_keys.add(key)
                    save_dtype, reason = storage_dtype, f"dynamic_{format_name}"
                else:
                    save_dtype, reason = keep_dtype or tensor.dtype, "kept_strict"
            else:
                if not torch.is_floating_point(tensor):
                    save_dtype, reason = tensor.dtype, "non_float"
                elif pattern_matches(exclude_patterns, key):
                    save_dtype, reason = keep_dtype or tensor.dtype, "excluded"
                elif pattern_matches(include_patterns, key):
                    if format_compatible_with_weight(format_name, key, tensor):
                        save_dtype, reason = storage_dtype, "included"
                    else:
                        save_dtype, reason = keep_dtype or tensor.dtype, f"kept_incompatible_{format_name}"
                elif is_full_quant_compatible_name(key) and format_compatible_with_weight(format_name, key, tensor):
                    save_dtype, reason = storage_dtype, f"simple_{format_name}"
                else:
                    save_dtype, reason = keep_dtype or tensor.dtype, "kept"

            if (
                torch.is_floating_point(tensor)
                and save_dtype == current_storage_dtype
                and use_comfy_quant
                and key.endswith(".weight")
                and tensor.ndim >= 2
                and not format_compatible_with_weight(current_format_name, key, tensor)
            ):
                save_dtype, reason = keep_dtype or tensor.dtype, f"kept_incompatible_{current_format_name}"

            reason_counts[reason] += 1
            in_bytes += tensor.numel() * tensor.element_size()

            if (
                torch.is_floating_point(tensor)
                and save_dtype == current_storage_dtype
                and use_comfy_quant
                and key.endswith(".weight")
                and tensor.ndim >= 2
            ):
                extra_records = comfy_quant_records_for_weight(
                    key,
                    tensor,
                    current_format_name,
                    current_storage_dtype,
                    nvfp4_scale_multiplier=current_nvfp4_scale_multiplier,
                )
                records.extend(extra_records)
                # Standard ComfyUI checkpoint-level manifest, in addition to the
                # per-layer .comfy_quant tensors used during direct loading.
                quant_conf: dict[str, object] = {"format": current_format_name}
                if current_format_name == "ternary_1_58":
                    quant_conf.update({
                        "logical_bits_per_weight": math.log2(3.0),
                        "storage_bits_per_weight": 2,
                        "packing": "ternary_2bit_four_per_byte",
                        "original_shape": [int(tensor.shape[0]), int(tensor.shape[1])],
                        "scale": "per_output_absmean",
                    })
                comfy_quant_layers[key[:-len(".weight")]] = quant_conf
                for _, rec_tensor, rec_dtype in extra_records:
                    rec_bytes = rec_tensor.numel() * dtype_element_size(rec_dtype)
                    out_bytes += rec_bytes
                    dtype_key = str(rec_dtype).replace("torch.", "")
                    dtype_counts[dtype_key]["tensors"] += 1
                    dtype_counts[dtype_key]["elements"] += rec_tensor.numel()
                    dtype_counts[dtype_key]["bytes"] += rec_bytes
            else:
                records.append((key, tensor, save_dtype))
                rec_bytes = tensor.numel() * dtype_element_size(save_dtype)
                out_bytes += rec_bytes
                dtype_key = str(save_dtype).replace("torch.", "")
                dtype_counts[dtype_key]["tensors"] += 1
                dtype_counts[dtype_key]["elements"] += tensor.numel()
                dtype_counts[dtype_key]["bytes"] += rec_bytes

    log("\nConversion plan:")
    for reason, count in sorted(reason_counts.items()):
        log(f"- {reason:24s}: {count:,} tensor(s)")
    log("\nOutput dtypes:")
    for dtype_key, info in sorted(dtype_counts.items()):
        mib = info["bytes"] / (1024 ** 2)
        log(f"- {dtype_key:18s}: {info['tensors']:,} tensor(s), {info['elements']:,} elems, {mib:.2f} MiB")
    log(f"\nEstimated tensor bytes: {in_bytes / (1024 ** 2):.2f} MiB -> {out_bytes / (1024 ** 2):.2f} MiB")
    if dynamic_strict:
        log(f"Dynamic checkpoint matches: {len(calibrated_matched_keys):,} tensor(s)")
        if not calibrated_matched_keys:
            log("WARNING: No dynamic keys matched checkpoint tensors; check key prefixes.")
    if reason_counts.get("pruned", 0):
        log("NOTE: Small-tensor pruning currently zeroes tensors but keeps their keys/shapes, so it is a quality experiment, not a file-size reduction.")

    if args.dry_run:
        log("\nDry run complete; no file written.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **original_metadata,
        "_quantization_metadata": json.dumps(
            {"version": 1, "layers": comfy_quant_layers}, separators=(",", ":")
        ),
        "aozora_quantization": "comfy_dynamic" if quantize_specs or quantize_keys else f"comfy_{args.fp8}",
        "aozora_quantization_mode": mode,
        "aozora_quantization_scope": "all_compatible_strict" if dynamic_strict else "all_compatible_simple",
        "aozora_keep_dtype": getattr(args, "keep_dtype", "bfloat16"),
        "aozora_source": input_path.name,
        "aozora_quantization_format": "dynamic" if quantize_specs else (format_name if use_comfy_quant else "raw_fp8"),
        "aozora_nvfp4_scale_multiplier": str(getattr(args, "nvfp4_scale_multiplier", 1.0)),
        "aozora_nvfp4_scale_multipliers": str(getattr(args, "nvfp4_scale_multipliers", DEFAULT_NVFP4_SCALE_MULTIPLIERS)),
    }
    write_streaming_safetensors(output_path, records, metadata)

    with safe_open(str(output_path), framework="pt", device="cpu") as handle:
        written_keys = set(handle.keys())
    expected_keys = {key for key, _, _ in records}
    if written_keys != expected_keys:
        missing = sorted(expected_keys - written_keys)
        unexpected = sorted(written_keys - expected_keys)
        log(f"WARNING: key mismatch: missing={len(missing):,}, unexpected={len(unexpected):,}")
    log(f"\nWrote: {output_path}")
    log(f"File size: {output_path.stat().st_size / (1024 ** 2):.2f} MiB")
    return output_path


def launch_gui() -> int:
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Anima Quant Converter - Simple / Dynamic")
    root.geometry("920x720")
    root.minsize(820, 620)

    log_queue: queue.Queue[str] = queue.Queue()
    running = tk.BooleanVar(value=False)
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    mode_var = tk.StringVar(value="Dynamic")
    simple_quant_var = tk.StringVar(value="nvfp4")
    keep_var = tk.StringVar(value="bfloat16")
    config_var = tk.StringVar(value="configs/Hysocs_Personal_Config.json")
    calibration_steps_var = tk.StringVar(value="8")
    rel_mse_var = tk.StringVar(value="0.002")
    cosine_var = tk.StringVar(value="0.999")
    max_abs_ratio_var = tk.StringVar(value="0.10")
    nvfp4_scale_multipliers_var = tk.StringVar(value=DEFAULT_NVFP4_SCALE_MULTIPLIERS)
    global_promote_steps_var = tk.StringVar(value="12")
    early_stop_var = tk.BooleanVar(value=True)
    early_stop_min_batches_var = tk.StringVar(value="4")
    early_stop_margin_var = tk.StringVar(value="8.0")
    progress_interval_var = tk.StringVar(value="15")
    prune_var = tk.BooleanVar(value=False)
    prune_max_var = tk.StringVar(value="0")
    overwrite_var = tk.BooleanVar(value=False)
    dataset_paths: list[str] = []

    quant_labels = {
        "nvfp4": "NVFP4 / FP4",
        "e4m3": "FP8 E4M3",
        "e5m2": "FP8 E5M2",
        "int8": "INT8 tensorwise",
        "q8_0": "GGUF Q8_0 (experimental)",
        "q5_1": "GGUF Q5_1 (best sub-8-bit)",
        "q5_0": "GGUF Q5_0 (experimental)",
        "q4_0": "GGUF Q4_0 (experimental)",
    }
    quant_choices = list(quant_labels.keys())
    dynamic_floor_var = tk.StringVar(value="4-bit NVFP4 + FP8 best-of")
    dynamic_strategy_var = tk.StringVar(value="Max compression")
    dynamic_floor_labels = {
        "nvfp4_all": "4-bit NVFP4 + FP8 + INT8",
        "nvfp4_fp8": "4-bit NVFP4 + FP8 best-of",
        "nvfp4_int8": "4-bit NVFP4 + INT8",
        "fp8": "8-bit FP8 E4M3 + E5M2",
        "int8": "8-bit INT8",
    }

    def append_log(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")

    def pump_log() -> None:
        try:
            while True:
                append_log(log_queue.get_nowait())
        except queue.Empty:
            pass
        root.after(100, pump_log)

    def current_mode() -> str:
        value = mode_var.get().lower()
        if value.startswith("dynamic"):
            return "dynamic"
        return "simple"

    def selected_dynamic_formats() -> list[str]:
        floor_label = dynamic_floor_var.get()
        floor = next((key for key, label in dynamic_floor_labels.items() if label == floor_label), floor_label)
        if floor == "nvfp4_all":
            return ["nvfp4", "e4m3", "e5m2", "int8"]
        if floor == "nvfp4_fp8":
            return ["nvfp4", "e4m3", "e5m2"]
        if floor == "nvfp4_int8":
            return ["nvfp4", "int8"]
        if floor == "fp8":
            return ["e4m3", "e5m2"]
        if floor == "int8":
            return ["int8"]
        return ["nvfp4", "e4m3", "e5m2"]

    def suggest_output(path_text: str) -> None:
        path = Path(path_text)
        if not path.name:
            return
        mode = current_mode()
        if mode == "dynamic":
            selected = "_".join(selected_dynamic_formats()) or mode
            if selected == "nvfp4_e4m3_e5m2_int8":
                selected = "nvfp4_fp8_int8"
            elif selected == "e4m3_e5m2":
                selected = "fp8_bestof_e4m3_e5m2"
            elif selected == "nvfp4_e4m3_e5m2":
                selected = "nvfp4_fp8_bestof"
            elif selected == "nvfp4_int8":
                selected = "nvfp4_int8"
            suffix = f"_{mode}_{selected}.safetensors"
        else:
            extension = ".gguf" if simple_quant_var.get() in GGUF_QUANT_CHOICES else ".safetensors"
            suffix = f"_simple_{simple_quant_var.get()}{extension}"
        output_var.set(str(path.with_name(path.stem + suffix)))

    def browse_input() -> None:
        path = filedialog.askopenfilename(
            title="Select Anima/DiT checkpoint",
            filetypes=(("Safetensors", "*.safetensors *.safetensor"), ("All files", "*.*")),
        )
        if path:
            input_var.set(path)
            if not output_var.get().strip():
                suggest_output(path)

    def browse_output() -> None:
        initial = output_var.get().strip() or None
        is_gguf = current_mode() == "simple" and simple_quant_var.get() in GGUF_QUANT_CHOICES
        path = filedialog.asksaveasfilename(
            title="Save quantized checkpoint as",
            initialfile=Path(initial).name if initial else None,
            initialdir=str(Path(initial).parent) if initial else None,
            defaultextension=".gguf" if is_gguf else ".safetensors",
            filetypes=(("GGUF", "*.gguf"), ("All files", "*.*")) if is_gguf else (("Safetensors", "*.safetensors"), ("All files", "*.*")),
        )
        if path:
            output_var.set(path)

    def browse_config() -> None:
        path = filedialog.askopenfilename(
            title="Select Anima config",
            filetypes=(("JSON config", "*.json"), ("All files", "*.*")),
        )
        if path:
            config_var.set(path)

    def refresh_dataset_list() -> None:
        dataset_listbox.delete(0, "end")
        for path in dataset_paths:
            dataset_listbox.insert("end", path)

    def add_dataset() -> None:
        path = filedialog.askdirectory(title="Add calibration image dataset")
        if path and path not in dataset_paths:
            dataset_paths.append(path)
            refresh_dataset_list()

    def remove_dataset() -> None:
        selected = list(dataset_listbox.curselection())
        for index in reversed(selected):
            del dataset_paths[index]
        refresh_dataset_list()

    def refresh_mode_state(*_args) -> None:
        mode = current_mode()
        simple_combo_state = "readonly" if mode == "simple" else "disabled"
        dynamic_combo_state = "readonly" if mode == "dynamic" else "disabled"
        dynamic_state = "normal" if mode == "dynamic" else "disabled"
        simple_quant_combo.configure(state=simple_combo_state)
        dynamic_floor_combo.configure(state=dynamic_combo_state)
        dynamic_strategy_combo.configure(state=dynamic_combo_state)
        for widget in dynamic_only_widgets:
            try:
                widget.configure(state=dynamic_state)
            except tk.TclError:
                pass

    def build_args(dry_run: bool = False) -> SimpleNamespace | None:
        input_path = input_var.get().strip().strip("\"'")
        output_path = output_var.get().strip().strip("\"'")
        if not input_path:
            messagebox.showerror("Missing input", "Select an input .safetensors model first.")
            return None
        if not output_path:
            suggest_output(input_path)
            output_path = output_var.get().strip().strip("\"'")

        mode = current_mode()
        formats = selected_dynamic_formats()
        if mode == "dynamic" and not formats:
            messagebox.showerror("Missing dynamic formats", "Select at least one quant format for Dynamic mode.")
            return None
        try:
            calibration_steps = int(calibration_steps_var.get())
            rel_mse = float(rel_mse_var.get())
            cosine = float(cosine_var.get())
            max_abs_ratio = float(max_abs_ratio_var.get())
            global_promote_steps = int(global_promote_steps_var.get())
            early_stop_min_batches = int(early_stop_min_batches_var.get())
            early_stop_margin = float(early_stop_margin_var.get())
            progress_interval = float(progress_interval_var.get())
            prune_max = int(prune_max_var.get())
            # Validate here so GUI users get an immediate readable error.
            parse_float_list(nvfp4_scale_multipliers_var.get(), name="NVFP4 scale multipliers")
        except ValueError as exc:
            messagebox.showerror("Invalid value", str(exc))
            return None

        return SimpleNamespace(
            input=input_path,
            output=output_path,
            mode=mode,
            config=config_var.get().strip().strip("\"'"),
            fp8=simple_quant_var.get(),
            mixed_formats=",".join(formats),
            composition_guard=(dynamic_strategy_var.get() == "Composition safe"),
            keep_dtype=keep_var.get(),
            include_regex=[],
            exclude_regex=[],
            dataset_overrides=list(dataset_paths),
            calibration_steps=calibration_steps,
            calibration_rel_mse=rel_mse,
            calibration_cosine=cosine,
            calibration_max_abs_ratio=max_abs_ratio,
            dynamic_global_promote_steps=global_promote_steps,
            dynamic_early_stop=early_stop_var.get(),
            dynamic_early_stop_min_batches=early_stop_min_batches,
            dynamic_early_stop_margin=early_stop_margin,
            dynamic_progress_interval=progress_interval,
            nvfp4_scale_multipliers=nvfp4_scale_multipliers_var.get(),
            nvfp4_scale_multiplier=1.0,
            prune_small=prune_var.get(),
            prune_max_elements=prune_max,
            raw_fp8=False,
            overwrite=overwrite_var.get(),
            dry_run=dry_run,
        )

    def run_conversion() -> None:
        if running.get():
            return
        args = build_args(False)
        if args is None:
            return
        running.set(True)
        convert_btn.configure(state="disabled")
        append_log("")
        append_log("Starting conversion...")

        def worker() -> None:
            try:
                mode = normalized_mode(args.mode)
                if mode == "dynamic":
                    args.quantize_specs = compression_first_profile(args, log=log_queue.put)
                convert_checkpoint(args, log=log_queue.put)
                log_queue.put("Done.")
            except Exception as exc:
                log_queue.put(f"ERROR: {exc}")
                root.after(0, lambda: messagebox.showerror("Conversion failed", str(exc)))
            finally:
                root.after(0, lambda: running.set(False))
                root.after(0, lambda: convert_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(14, weight=1)

    ttk.Label(main_frame, text="Input model").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=input_var).grid(row=0, column=1, sticky="ew", padx=8)
    ttk.Button(main_frame, text="Browse", command=browse_input).grid(row=0, column=2, sticky="ew")

    ttk.Label(main_frame, text="Output model").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=output_var).grid(row=1, column=1, sticky="ew", padx=8)
    ttk.Button(main_frame, text="Save As", command=browse_output).grid(row=1, column=2, sticky="ew")

    mode_frame = ttk.LabelFrame(main_frame, text="Mode", padding=8)
    mode_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 6))
    mode_frame.columnconfigure(1, weight=1)
    ttk.Label(mode_frame, text="Mode").grid(row=0, column=0, sticky="w")
    mode_combo = ttk.Combobox(mode_frame, textvariable=mode_var, values=("Simple", "Dynamic"), width=14, state="readonly")
    mode_combo.grid(row=0, column=1, sticky="w", padx=(8, 18))
    ttk.Label(mode_frame, text="Keep dtype").grid(row=0, column=2, sticky="w")
    ttk.Combobox(mode_frame, textvariable=keep_var, values=sorted(KEEP_DTYPES), width=12, state="readonly").grid(row=0, column=3, sticky="w", padx=(8, 0))

    simple_frame = ttk.LabelFrame(main_frame, text="Simple mode: apply one quant to every compatible layer", padding=8)
    simple_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=6)
    ttk.Label(simple_frame, text="Quant").pack(side="left")
    simple_quant_combo = ttk.Combobox(
        simple_frame,
        textvariable=simple_quant_var,
        values=quant_choices,
        width=12,
        state="readonly",
    )
    simple_quant_combo.pack(side="left", padx=(8, 18))
    ttk.Label(simple_frame, text="Protected tensors are excluded from quantization.").pack(side="left")

    dynamic_frame = ttk.LabelFrame(main_frame, text="Dynamic mode: compression-first mixed precision", padding=8)
    dynamic_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
    ttk.Label(dynamic_frame, text="Candidate floor").pack(side="left")
    dynamic_floor_combo = ttk.Combobox(
        dynamic_frame,
        textvariable=dynamic_floor_var,
        values=list(dynamic_floor_labels.values()),
        width=24,
        state="readonly",
    )
    dynamic_floor_combo.pack(side="left", padx=(8, 18))
    ttk.Label(dynamic_frame, text="Strategy").pack(side="left")
    dynamic_strategy_combo = ttk.Combobox(
        dynamic_frame,
        textvariable=dynamic_strategy_var,
        values=("Max compression", "Composition safe"),
        width=18,
        state="readonly",
    )
    dynamic_strategy_combo.pack(side="left", padx=(8, 18))
    ttk.Label(dynamic_frame, text="Risky layers can be promoted to FP8/INT8 or kept BF16.").pack(side="left")

    ttk.Label(main_frame, text="Config").grid(row=5, column=0, sticky="w", pady=4)
    config_entry = ttk.Entry(main_frame, textvariable=config_var)
    config_entry.grid(row=5, column=1, sticky="ew", padx=8)
    config_btn = ttk.Button(main_frame, text="Browse", command=browse_config)
    config_btn.grid(row=5, column=2, sticky="ew")

    calib = ttk.Frame(main_frame)
    calib.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 2))
    ttk.Label(calib, text="Cal steps").pack(side="left")
    steps_entry = ttk.Entry(calib, textvariable=calibration_steps_var, width=8)
    steps_entry.pack(side="left", padx=(6, 0))

    perf_frame = ttk.Frame(main_frame)
    perf_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 2))
    ttk.Label(perf_frame, text="Promote steps").pack(side="left")
    promote_steps_entry = ttk.Entry(perf_frame, textvariable=global_promote_steps_var, width=6)
    promote_steps_entry.pack(side="left", padx=(6, 18))
    ttk.Checkbutton(perf_frame, text="Overwrite existing output", variable=overwrite_var).pack(side="left")

    dataset_frame = ttk.Frame(main_frame)
    dataset_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(6, 4))
    dataset_frame.columnconfigure(1, weight=1)
    ttk.Label(dataset_frame, text="Calibration datasets").grid(row=0, column=0, sticky="nw", padx=(0, 8))
    dataset_listbox = tk.Listbox(dataset_frame, height=3, selectmode="extended")
    dataset_listbox.grid(row=0, column=1, sticky="ew")
    dataset_buttons = ttk.Frame(dataset_frame)
    dataset_buttons.grid(row=0, column=2, sticky="n", padx=(8, 0))
    add_dataset_btn = ttk.Button(dataset_buttons, text="+", width=3, command=add_dataset)
    add_dataset_btn.pack(side="top")
    remove_dataset_btn = ttk.Button(dataset_buttons, text="-", width=3, command=remove_dataset)
    remove_dataset_btn.pack(side="top", pady=(4, 0))

    dynamic_only_widgets = [
        config_entry,
        config_btn,
        steps_entry,
        promote_steps_entry,
        dataset_listbox,
        add_dataset_btn,
        remove_dataset_btn,
    ]

    help_text = (
        "Simple applies one selected quant to all compatible matrix weights. "
        "Dynamic is size-first: it ranks layers with activation-weighted quantization error, uses the selected quant family, and promotes risky layers within that family. Use Composition safe when NVFP4 harms layout. "
        "Use FP8 best-of for quality-oriented 8-bit, or INT8 when you want integer 8-bit only."
    )
    ttk.Label(main_frame, text=help_text, wraplength=860).grid(row=10, column=0, columnspan=3, sticky="ew", pady=(2, 8))

    buttons = ttk.Frame(main_frame)
    buttons.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    convert_btn = ttk.Button(buttons, text="Convert", command=run_conversion)
    convert_btn.pack(side="left")

    ttk.Label(main_frame, text="Log").grid(row=13, column=0, columnspan=3, sticky="w")
    log_box = scrolledtext.ScrolledText(main_frame, height=18, state="disabled", wrap="word")
    log_box.grid(row=14, column=0, columnspan=3, sticky="nsew")

    mode_var.trace_add("write", refresh_mode_state)
    append_log("Select an Anima/DiT .safetensors model, then convert.")
    append_log("Dynamic is the size-first scan. Strategy defaults to Max compression; use Composition safe if 4-bit harms layout.")
    refresh_mode_state()
    pump_log()
    root.mainloop()
    return 0


def main() -> int:
    return launch_gui()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
