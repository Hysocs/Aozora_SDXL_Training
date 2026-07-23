# Copyright 2026 Aozora Trainer contributors
# Portions derived from ModelScope DiffSynth-Studio 2.0.16 and modified for Aozora.
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import Qwen3Config, Qwen3Model


class ZImageTextEncoder(torch.nn.Module):
    """The Qwen3-0.6B encoder configuration used by Anima training."""

    def __init__(self, model_size="0.6B"):
        super().__init__()
        if model_size != "0.6B":
            raise ValueError("Aozora Anima training only supports the Qwen3-0.6B encoder.")
        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            head_dim=128,
            hidden_act="silu",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=3072,
            max_position_embeddings=40960,
            max_window_layers=28,
            num_attention_heads=16,
            num_hidden_layers=28,
            num_key_value_heads=8,
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            sliding_window=None,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_sliding_window=False,
            vocab_size=151936,
        )
        self.model = Qwen3Model(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
