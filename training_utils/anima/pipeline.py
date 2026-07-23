# Copyright 2026 Aozora SDXL Training contributors
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from transformers import AutoTokenizer

from .loader import load_anima_dit, load_anima_text_encoder, load_anima_vae


@dataclass
class ModelConfig:
    path: object = None
    skip_download: bool = True


class AnimaTrainingComponents:
    """Minimal training-only pipeline containing the components Aozora uses."""

    def __init__(self, device="cpu", torch_dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.dit = None
        self.text_encoder = None
        self.vae = None
        self.tokenizer = None
        self.tokenizer_t5xxl = None

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=(),
        tokenizer_config=None,
        tokenizer_t5xxl_config=None,
        **_unused,
    ):
        if len(model_configs) != 3:
            raise ValueError("Anima training expects DiT, text encoder, and VAE configs in that order.")
        pipe = cls(device=device, torch_dtype=torch_dtype)
        pipe.dit = load_anima_dit(model_configs[0].path, torch_dtype, device)
        pipe.text_encoder = load_anima_text_encoder(model_configs[1].path, torch_dtype, device)
        pipe.vae = load_anima_vae(model_configs[2].path, torch_dtype, device)
        if tokenizer_config is not None:
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        if tokenizer_t5xxl_config is not None:
            pipe.tokenizer_t5xxl = AutoTokenizer.from_pretrained(tokenizer_t5xxl_config.path)
        return pipe

    def preprocess_image(self, image, torch_dtype=None, device=None):
        image = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * (2.0 / 255.0) - 1.0
        return rearrange(image, "h w c -> 1 c h w")


AnimaImagePipeline = AnimaTrainingComponents
