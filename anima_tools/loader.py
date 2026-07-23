# Copyright 2026 Aozora SDXL Training contributors
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

from .models import AnimaDiT, WanVideoVAE, ZImageTextEncoder


def _paths(path) -> list[Path]:
    values = path if isinstance(path, (list, tuple)) else [path]
    return [Path(value) for value in values]


def _load_tensor_files(paths: Iterable[Path], dtype: torch.dtype, key_map=None) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for path in paths:
        if path.suffix.lower() == ".safetensors":
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for source_key in handle.keys():
                    target_key = key_map(source_key) if key_map else source_key
                    if target_key is None:
                        continue
                    tensor = handle.get_tensor(source_key)
                    state_dict[target_key] = tensor.to(dtype) if tensor.is_floating_point() else tensor
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for wrapper in ("state_dict", "module", "model_state"):
                if isinstance(payload, dict) and wrapper in payload and len(payload) == 1:
                    payload = payload[wrapper]
            for source_key, tensor in payload.items():
                target_key = key_map(source_key) if key_map else source_key
                if target_key is None:
                    continue
                state_dict[target_key] = tensor.to(dtype) if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() else tensor
    return state_dict


def _strip_prefix(state_dict, prefixes):
    keys = tuple(state_dict)
    for prefix in prefixes:
        if keys and sum(key.startswith(prefix) for key in keys) / len(keys) >= 0.8:
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _load_model(model, state_dict, dtype, device):
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match {type(model).__name__}: "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys."
        )
    return model.to(device=device, dtype=dtype).eval()


def load_anima_dit(path, dtype=torch.bfloat16, device="cpu"):
    with init_empty_weights(include_buffers=False):
        model = AnimaDiT()
    state_dict = _load_tensor_files(_paths(path), dtype)
    state_dict = _strip_prefix(
        state_dict,
        ("pipe.dit.", "model.diffusion_model.", "diffusion_model.", "dit.", "net."),
    )
    return _load_model(model, state_dict, dtype, device)


def load_anima_text_encoder(path, dtype=torch.bfloat16, device="cpu"):
    with init_empty_weights(include_buffers=False):
        model = ZImageTextEncoder(model_size="0.6B")
    state_dict = _load_tensor_files(_paths(path), dtype)
    state_dict = _strip_prefix(state_dict, ("text_encoder.", "model.text_encoder."))
    state_dict.pop("lm_head.weight", None)
    return _load_model(model, state_dict, dtype, device)


def load_anima_vae(path, dtype=torch.bfloat16, device="cpu"):
    with init_empty_weights(include_buffers=False):
        model = WanVideoVAE()
    model.reset_scale()
    encoder_keys = set(model.state_dict())

    def map_encoder_key(key):
        for prefix in ("pipe.vae.", "vae."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        target_key = key if key.startswith("model.") else f"model.{key}"
        return target_key if target_key in encoder_keys else None

    state_dict = _load_tensor_files(_paths(path), dtype, key_map=map_encoder_key)
    return _load_model(model, state_dict, dtype, device)
