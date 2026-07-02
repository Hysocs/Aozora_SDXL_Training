"""
Convert an Anima/DiT safetensors checkpoint to ComfyUI quantized weights.

This is a post-training inference checkpoint converter. It can write ComfyUI
scaled FP8 or tensorwise INT8 mixed-precision checkpoints. It does not create
MXFP8 block-scaled checkpoints.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import struct
import sys
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
}

QUANT_STORAGE_DTYPES = {
    "int8_tensorwise": torch.int8,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an Anima/DiT safetensors checkpoint to ComfyUI quantized weights."
    )
    parser.add_argument("input", nargs="?", help="Input .safetensors checkpoint.")
    parser.add_argument("output", nargs="?", help="Output .safetensors checkpoint.")
    parser.add_argument("--gui", action="store_true", help="Launch the file-picker GUI.")
    parser.add_argument(
        "--mode",
        choices=("fast", "calibrated"),
        default="fast",
        help="fast uses name/shape heuristics; calibrated profiles Anima activations from a config dataset.",
    )
    parser.add_argument(
        "--config",
        default="configs/Hysocs_Personal_Config.json",
        help="Anima GUI config JSON for calibrated mode.",
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        default=16,
        help="No-grad Anima batches to profile in calibrated mode.",
    )
    parser.add_argument(
        "--calibration-rel-mse",
        type=float,
        default=0.002,
        help="Max relative output MSE for a layer to be quantized in calibrated mode.",
    )
    parser.add_argument(
        "--calibration-cosine",
        type=float,
        default=0.999,
        help="Minimum output cosine similarity for a layer to be quantized in calibrated mode.",
    )
    parser.add_argument(
        "--strict-calibration",
        action="store_true",
        help="In calibrated mode, do not quantize preset extras that failed or were not selected by profiling.",
    )
    parser.add_argument(
        "--fp8",
        choices=sorted(FP8_DTYPES),
        default="e4m3",
        help="Quantization format to use. e4m3 is the usual FP8 inference-first choice; int8 writes Comfy tensorwise INT8.",
    )
    parser.add_argument(
        "--preset",
        choices=("conservative", "broad", "all"),
        default="conservative",
        help=(
            "conservative: only obvious attention/MLP/proj 2D weights; "
            "broad: all rank>=2 floating tensors except protected names; "
            "all: every floating tensor not excluded."
        ),
    )
    parser.add_argument(
        "--keep-dtype",
        choices=sorted(KEEP_DTYPES),
        default="bfloat16",
        help="Dtype for floating tensors that are not quantized.",
    )
    parser.add_argument(
        "--include-regex",
        action="append",
        default=[],
        help="Extra regex pattern to force matching floating tensors into the quantized format.",
    )
    parser.add_argument(
        "--exclude-regex",
        action="append",
        default=[],
        help="Regex pattern to keep matching tensors out of the quantized format.",
    )
    parser.add_argument(
        "--min-elements",
        type=int,
        default=1024,
        help="Small floating tensors below this size stay in keep-dtype unless preset=all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--prune-small",
        action="store_true",
        help="Zero eligible small floating tensors instead of saving their original values.",
    )
    parser.add_argument(
        "--prune-max-elements",
        type=int,
        default=0,
        help="Max tensor elements for --prune-small. 0 disables pruning.",
    )
    parser.add_argument(
        "--raw-fp8",
        action="store_true",
        help="Write bare FP8 tensors without Comfy weight_scale/comfy_quant companion tensors.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write output.")
    return parser.parse_args()


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
    return "INT8" if choice == "int8" else "FP8"


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


def should_prune_tensor(key: str, tensor: torch.Tensor, args: argparse.Namespace | SimpleNamespace) -> bool:
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


def comfy_quant_info_tensor(format_name: str) -> torch.Tensor:
    payload = json.dumps({"format": format_name}).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


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


def simulated_quant_weight(weight: torch.Tensor, storage_dtype: torch.dtype, format_name: str, compute_dtype: torch.dtype) -> torch.Tensor:
    w = weight.detach().float()
    if format_name == "int8_tensorwise":
        if w.ndim >= 2:
            scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1.0e-30)
        else:
            scale = (w.abs().max() / 127.0).clamp(min=1.0e-30)
        quantized = (w / scale).round().clamp(-128.0, 127.0).to(dtype=torch.int8)
    else:
        max_value = w.abs().max() if w.numel() else torch.tensor(0.0, device=w.device)
        scale = (max_value / float(torch.finfo(storage_dtype).max)).clamp(min=1.0e-12)
        quantized = (w / scale).to(dtype=storage_dtype)
    return (quantized.float() * scale).to(dtype=compute_dtype)


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
        self.count = 0

    def update(self, ref: torch.Tensor, quant: torch.Tensor) -> None:
        ref_f = ref.detach().float()
        quant_f = quant.detach().float()
        diff = quant_f - ref_f
        self.diff_sq += float(diff.pow(2).sum().item())
        self.ref_sq += float(ref_f.pow(2).sum().item())
        self.quant_sq += float(quant_f.pow(2).sum().item())
        self.dot += float((ref_f * quant_f).sum().item())
        self.count += int(ref_f.numel())

    @property
    def rel_mse(self) -> float:
        return self.diff_sq / max(self.ref_sq, 1.0e-12)

    @property
    def cosine(self) -> float:
        denom = (self.ref_sq ** 0.5) * (self.quant_sq ** 0.5)
        return self.dot / max(denom, 1.0e-12)


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
    args: argparse.Namespace | SimpleNamespace,
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
        safe_set_anima_scheduler_training,
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
    safe_set_anima_scheduler_training(pipe)

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
        if module.weight.numel() < int(getattr(args, "min_elements", 1024) or 0):
            continue
        preset = getattr(args, "preset", "conservative")
        if preset == "all":
            if not is_full_quant_compatible_name(key):
                continue
        elif is_protected_name(key):
            continue
        if preset == "conservative" and not is_target_name(key):
            continue
        handles.append(module.register_forward_hook(make_hook(name, module)))

    if not handles:
        raise RuntimeError("No candidate Linear layers found for calibration.")
    log(f"Hooked candidate layers: {len(handles):,}")

    dataset = AnimaCachedDataset(config)
    if len(dataset) <= 0:
        raise RuntimeError("No cached Anima dataset items found for calibration.")
    total_scheduler_timesteps = len(pipe.scheduler.timesteps)
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    steps = max(1, int(getattr(args, "calibration_steps", 16) or 16))
    image_schedule = build_image_batch_schedule(
        dataset,
        steps,
        config.BATCH_SIZE,
        config.SEED if config.SEED else 42,
        timestep_sampler.ticket_pool,
        timestep_sampler.bin_ranges,
        bool(getattr(config, "TIMESTEP_FORCE_IMAGE_BIN_SPREAD", False)),
    )
    batch_sampler = PrecomputedImageBatchSampler(image_schedule, config.SEED if config.SEED else 42, 0)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )

    scheduler_timesteps = pipe.scheduler.timesteps.to(device=device)
    scheduler_sigmas = pipe.scheduler.sigmas.to(device=device, dtype=config.compute_dtype)
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
            timesteps = scheduler_timesteps[timestep_indices].to(device=device, dtype=config.compute_dtype)
            sigmas = scheduler_sigmas[timestep_indices]
            noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)
            noisy_latents, _ = flowmatch_noise_and_target(input_latents, noise, sigmas)
            run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
            completed += 1
            log(f"- profiled batch {completed}/{steps}")

    for handle in handles:
        handle.remove()

    max_rel_mse = float(getattr(args, "calibration_rel_mse", 0.002))
    min_cosine = float(getattr(args, "calibration_cosine", 0.999))
    selected = {
        p.weight_key
        for p in profiles.values()
        if p.count > 0 and p.rel_mse <= max_rel_mse and p.cosine >= min_cosine
    }

    ranked = sorted(profiles.values(), key=lambda p: (p.weight_key not in selected, p.rel_mse))
    log("\nCalibration summary:")
    log(f"- selected: {len(selected):,} / {len(profiles):,} candidate layer weights")
    log(f"- thresholds: rel_mse <= {max_rel_mse:g}, cosine >= {min_cosine:g}")
    for p in ranked[:20]:
        mark = quant_label(getattr(args, "fp8", "e4m3")) if p.weight_key in selected else "keep"
        log(f"  {mark:4s} rel_mse={p.rel_mse:.6g} cosine={p.cosine:.6f} elems={p.elements:,} {p.weight_key}")

    profile_path = Path(args.output).with_suffix(".profile.json")
    payload = {
        "source": str(input_path),
        "quant": getattr(args, "fp8", "e4m3"),
        "format": format_name,
        "steps": completed,
        "thresholds": {"rel_mse": max_rel_mse, "cosine": min_cosine},
        "selected": sorted(selected),
        "layers": [
            {
                "name": p.name,
                "weight_key": p.weight_key,
                "elements": p.elements,
                "rel_mse": p.rel_mse,
                "cosine": p.cosine,
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


def convert_checkpoint(args: argparse.Namespace | SimpleNamespace, log=print) -> Path | None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")

    format_name = quant_format_name(args.fp8)
    storage_dtype = quant_storage_dtype(format_name)
    if format_name == "int8_tensorwise" and getattr(args, "raw_fp8", False):
        raise ValueError("INT8 requires Comfy weight_scale/comfy_quant metadata; --raw-fp8 is FP8-only.")

    keep_dtype = KEEP_DTYPES[args.keep_dtype]
    include_patterns = compile_patterns(args.include_regex)
    exclude_patterns = compile_patterns(args.exclude_regex)
    quantize_keys = set(getattr(args, "quantize_keys", None) or [])
    zero_keys = set(getattr(args, "zero_keys", None) or [])
    use_comfy_quant = not bool(getattr(args, "raw_fp8", False))

    records = []
    reason_counts = defaultdict(int)
    dtype_counts = defaultdict(lambda: {"tensors": 0, "elements": 0, "bytes": 0})
    in_bytes = 0
    out_bytes = 0
    original_metadata = {}
    calibrated_matched_keys = set()

    log(f"Input:  {input_path}")
    log(f"Output: {output_path}")
    log(f"Mode:   preset={args.preset}, quant={format_name}, keep={args.keep_dtype}")
    log(f"Format: {'Comfy scaled quant' if use_comfy_quant else 'raw safetensors FP8'}")
    if quantize_keys:
        log(f"Forced calibrated {quant_label(args.fp8)} keys: {len(quantize_keys):,}")
        if getattr(args, "strict_calibration", False):
            log("Strict calibration: enabled; preset extras will stay in keep-dtype.")

    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        original_metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key in zero_keys or should_prune_tensor(key, tensor, args):
                tensor = torch.zeros_like(tensor)
                save_dtype, reason = (keep_dtype or tensor.dtype), "pruned"
            elif quantize_keys:
                if calibrated_key_matches(key, quantize_keys):
                    calibrated_matched_keys.add(key)
                    save_dtype, reason = storage_dtype, "calibrated"
                elif getattr(args, "strict_calibration", False):
                    save_dtype, reason = keep_dtype or tensor.dtype, "kept"
                else:
                    save_dtype, reason = calibrated_extra_dtype(
                        key,
                        tensor,
                        storage_dtype,
                        keep_dtype,
                        args.preset,
                        int(args.min_elements),
                    )
            else:
                save_dtype, reason = choose_save_dtype(
                    key,
                    tensor,
                    storage_dtype,
                    keep_dtype,
                    args.preset,
                    include_patterns,
                    exclude_patterns,
                    args.min_elements,
                )
            reason_counts[reason] += 1
            in_bytes += tensor.numel() * tensor.element_size()

            if (
                torch.is_floating_point(tensor)
                and save_dtype == storage_dtype
                and use_comfy_quant
                and key.endswith(".weight")
                and tensor.ndim >= 2
            ):
                quant_tensor, scale_tensor = scaled_quant_tensor(tensor, storage_dtype, format_name)
                quant_info = comfy_quant_info_tensor(format_name)
                extra_records = (
                    (key, quant_tensor, storage_dtype),
                    (comfy_scale_key_for_weight(key), scale_tensor, torch.float32),
                    (comfy_quant_key_for_weight(key), quant_info, torch.uint8),
                )
                records.extend(extra_records)
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
        log(f"- {reason:10s}: {count:,} tensor(s)")
    log("\nOutput dtypes:")
    for dtype_key, info in sorted(dtype_counts.items()):
        mib = info["bytes"] / (1024 ** 2)
        log(f"- {dtype_key:18s}: {info['tensors']:,} tensor(s), {info['elements']:,} elems, {mib:.2f} MiB")
    log(f"\nEstimated tensor bytes: {in_bytes / (1024 ** 2):.2f} MiB -> {out_bytes / (1024 ** 2):.2f} MiB")
    if quantize_keys:
        log(f"Calibrated checkpoint matches: {len(calibrated_matched_keys):,} tensor(s)")
        if not calibrated_matched_keys:
            log("WARNING: No calibrated keys matched checkpoint tensors; check key prefixes.")
    if reason_counts.get("pruned", 0):
        log("NOTE: Small-tensor pruning currently zeroes tensors but keeps their keys/shapes, so it is a quality experiment, not a file-size reduction.")

    if args.dry_run:
        log("\nDry run complete; no file written.")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **original_metadata,
        "aozora_quantization": f"comfy_{args.fp8}",
        "aozora_quantization_preset": args.preset,
        "aozora_keep_dtype": args.keep_dtype,
        "aozora_source": input_path.name,
        "aozora_quantization_format": format_name if use_comfy_quant else "raw_fp8",
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
    root.title("Anima Quant Converter")
    root.geometry("860x620")
    root.minsize(760, 520)

    log_queue: queue.Queue[str] = queue.Queue()
    running = tk.BooleanVar(value=False)
    input_var = tk.StringVar()
    output_var = tk.StringVar()
    fp8_var = tk.StringVar(value="e4m3")
    mode_var = tk.StringVar(value="fast")
    config_var = tk.StringVar(value="configs/Hysocs_Personal_Config.json")
    preset_var = tk.StringVar(value="conservative")
    keep_var = tk.StringVar(value="bfloat16")
    min_elements_var = tk.StringVar(value="1024")
    calibration_steps_var = tk.StringVar(value="16")
    rel_mse_var = tk.StringVar(value="0.002")
    cosine_var = tk.StringVar(value="0.999")
    strict_calibration_var = tk.BooleanVar(value=False)
    prune_var = tk.BooleanVar(value=False)
    prune_max_var = tk.StringVar(value="0")
    overwrite_var = tk.BooleanVar(value=False)
    dataset_paths: list[str] = []

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

    def suggest_output(path_text: str) -> None:
        path = Path(path_text)
        if not path.name:
            return
        suffix = f"_quant_{fp8_var.get()}_{preset_var.get()}.safetensors"
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
        path = filedialog.asksaveasfilename(
            title="Save quantized checkpoint as",
            initialfile=Path(initial).name if initial else None,
            initialdir=str(Path(initial).parent) if initial else None,
            defaultextension=".safetensors",
            filetypes=(("Safetensors", "*.safetensors"), ("All files", "*.*")),
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

    def build_args(dry_run: bool = False) -> SimpleNamespace | None:
        input_path = input_var.get().strip().strip("\"'")
        output_path = output_var.get().strip().strip("\"'")
        if not input_path:
            messagebox.showerror("Missing input", "Select an input .safetensors model first.")
            return None
        if not output_path:
            suggest_output(input_path)
            output_path = output_var.get().strip().strip("\"'")
        try:
            min_elements = int(min_elements_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Min elements must be an integer.")
            return None
        try:
            calibration_steps = int(calibration_steps_var.get())
            rel_mse = float(rel_mse_var.get())
            cosine = float(cosine_var.get())
            prune_max = int(prune_max_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Calibration and pruning values must be numeric.")
            return None
        return SimpleNamespace(
            input=input_path,
            output=output_path,
            mode=mode_var.get(),
            config=config_var.get().strip().strip("\"'"),
            fp8=fp8_var.get(),
            preset=preset_var.get(),
            keep_dtype=keep_var.get(),
            include_regex=[],
            exclude_regex=[],
            min_elements=min_elements,
            dataset_overrides=list(dataset_paths),
            calibration_steps=calibration_steps,
            calibration_rel_mse=rel_mse,
            calibration_cosine=cosine,
            strict_calibration=strict_calibration_var.get(),
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
                if args.mode == "calibrated":
                    format_name = quant_format_name(args.fp8)
                    storage_dtype = quant_storage_dtype(format_name)
                    args.quantize_keys = calibrated_profile(args, storage_dtype, format_name, log=log_queue.put)
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
    main_frame.rowconfigure(10, weight=1)

    ttk.Label(main_frame, text="Input model").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=input_var).grid(row=0, column=1, sticky="ew", padx=8)
    ttk.Button(main_frame, text="Browse", command=browse_input).grid(row=0, column=2, sticky="ew")

    ttk.Label(main_frame, text="Output model").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=output_var).grid(row=1, column=1, sticky="ew", padx=8)
    ttk.Button(main_frame, text="Save As", command=browse_output).grid(row=1, column=2, sticky="ew")

    ttk.Label(main_frame, text="Config").grid(row=2, column=0, sticky="w", pady=4)
    ttk.Entry(main_frame, textvariable=config_var).grid(row=2, column=1, sticky="ew", padx=8)
    ttk.Button(main_frame, text="Browse", command=browse_config).grid(row=2, column=2, sticky="ew")

    options = ttk.Frame(main_frame)
    options.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 2))
    for col in range(8):
        options.columnconfigure(col, weight=1 if col in (1, 3, 5, 7) else 0)

    ttk.Label(options, text="Mode").grid(row=0, column=0, sticky="w")
    ttk.Combobox(options, textvariable=mode_var, values=("fast", "calibrated"), width=12, state="readonly").grid(row=0, column=1, sticky="w", padx=(6, 18))
    ttk.Label(options, text="Quant").grid(row=0, column=2, sticky="w")
    ttk.Combobox(options, textvariable=fp8_var, values=sorted(FP8_DTYPES), width=12, state="readonly").grid(row=0, column=3, sticky="w", padx=(6, 18))
    ttk.Label(options, text="Preset").grid(row=0, column=4, sticky="w")
    ttk.Combobox(options, textvariable=preset_var, values=("conservative", "broad", "all"), width=16, state="readonly").grid(row=0, column=5, sticky="w", padx=(6, 18))
    ttk.Label(options, text="Keep dtype").grid(row=0, column=6, sticky="w")
    ttk.Combobox(options, textvariable=keep_var, values=sorted(KEEP_DTYPES), width=12, state="readonly").grid(row=0, column=7, sticky="w", padx=(6, 0))

    calib = ttk.Frame(main_frame)
    calib.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 2))
    ttk.Label(calib, text="Min elements").pack(side="left")
    ttk.Entry(calib, textvariable=min_elements_var, width=10).pack(side="left", padx=(6, 18))
    ttk.Label(calib, text="Cal steps").pack(side="left")
    ttk.Entry(calib, textvariable=calibration_steps_var, width=8).pack(side="left", padx=(6, 18))
    ttk.Label(calib, text="Max rel MSE").pack(side="left")
    ttk.Entry(calib, textvariable=rel_mse_var, width=10).pack(side="left", padx=(6, 18))
    ttk.Label(calib, text="Min cosine").pack(side="left")
    ttk.Entry(calib, textvariable=cosine_var, width=10).pack(side="left", padx=(6, 0))

    prune_frame = ttk.Frame(main_frame)
    prune_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 2))
    ttk.Checkbutton(prune_frame, text="Strict calibration", variable=strict_calibration_var).pack(side="left")
    ttk.Checkbutton(prune_frame, text="Zero eligible small tensors (no size gain)", variable=prune_var).pack(side="left")
    ttk.Label(prune_frame, text="Max elements").pack(side="left", padx=(18, 6))
    ttk.Entry(prune_frame, textvariable=prune_max_var, width=10).pack(side="left")
    ttk.Checkbutton(prune_frame, text="Overwrite existing output", variable=overwrite_var).pack(side="left", padx=(18, 0))

    dataset_frame = ttk.Frame(main_frame)
    dataset_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(6, 4))
    dataset_frame.columnconfigure(1, weight=1)
    ttk.Label(dataset_frame, text="Calibration datasets").grid(row=0, column=0, sticky="nw", padx=(0, 8))
    dataset_listbox = tk.Listbox(dataset_frame, height=3, selectmode="extended")
    dataset_listbox.grid(row=0, column=1, sticky="ew")
    dataset_buttons = ttk.Frame(dataset_frame)
    dataset_buttons.grid(row=0, column=2, sticky="n", padx=(8, 0))
    ttk.Button(dataset_buttons, text="+", width=3, command=add_dataset).pack(side="top")
    ttk.Button(dataset_buttons, text="-", width=3, command=remove_dataset).pack(side="top", pady=(4, 0))

    help_text = (
        "conservative is the safest first test. broad quantizes more large matrix weights. "
        "In calibrated mode, broad/all add extra quantized tensors beyond the profile. "
        "Zeroing small tensors is experimental and does not shrink safetensors files. "
        "This creates Comfy scaled FP8 or tensorwise INT8, not MXFP8."
    )
    ttk.Label(main_frame, text=help_text, wraplength=780).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(2, 8))

    buttons = ttk.Frame(main_frame)
    buttons.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    convert_btn = ttk.Button(buttons, text="Convert", command=run_conversion)
    convert_btn.pack(side="left")

    ttk.Label(main_frame, text="Log").grid(row=9, column=0, columnspan=3, sticky="w")
    log_box = scrolledtext.ScrolledText(main_frame, height=18, state="disabled", wrap="word")
    log_box.grid(row=10, column=0, columnspan=3, sticky="nsew")

    append_log("Select an Anima/DiT .safetensors model, then convert.")
    pump_log()
    root.mainloop()
    return 0


def main() -> int:
    args = parse_args()
    if args.gui or (args.input is None and args.output is None):
        return launch_gui()
    if not args.input or not args.output:
        raise ValueError("Input and output paths are required unless --gui is used.")
    if args.mode == "calibrated":
        format_name = quant_format_name(args.fp8)
        storage_dtype = quant_storage_dtype(format_name)
        args.quantize_keys = calibrated_profile(args, storage_dtype, format_name)
    convert_checkpoint(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
