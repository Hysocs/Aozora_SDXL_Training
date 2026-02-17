""" Anima DiT Trainer - VRAM Optimized with RavenAdamW """
import os
import gc
import json
import random
import math
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
import numpy as np
from copy import deepcopy

# Import Transformers components needed for the text encoder hack
from transformers import Qwen2Config, Qwen2Model, AutoTokenizer, Qwen3Config, Qwen3Model

# Import RavenAdamW from optimizer folder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'optimizer'))
from optimizer.raven import RavenAdamW

# Enable memory efficient attention backends
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Force aggressive memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
ANIMA_DIT_PATH = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\diffusion_models\anima-preview.safetensors"
QWEN3_PATH = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\TextEncoders\qwen_3_06b_base.safetensors"
VAE_PATH = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\VAE\qwen_image_vae.safetensors"
DATA_DIR = r"C:\Users\Administrator\Pictures\Artists\by quilm"
OUTPUT_DIR = "./anima_output"

LEARNING_RATE = 2e-7
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
MAX_STEPS = 2500
SAVE_EVERY = 500
RF_SHIFT = 3.0
MIXED_PRECISION = "bf16"

# Qwen3 vocab size - CRITICAL FIX
QWEN_VOCAB_SIZE = 151936

# Target sequence length (64x64 patches for 1024px image with 8x VAE + 2x patch)
TARGET_SEQ_LEN = 4096  # 64*64

# ==============================================================================
# MINIMAL UTILITIES
# ==============================================================================
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[VRAM {tag}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

# ==============================================================================
# EXACT WanVAE REFERENCE IMPLEMENTATION
# ==============================================================================
CACHE_T = 2

class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)

class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias

class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity

class Encoder3d(nn.Module):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

class Decoder3d(nn.Module):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in [1, 2, 3]:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count

class WanVAE_(nn.Module):
    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

        self.clear_cache()

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

# ==============================================================================
# DiT ARCHITECTURE - OPTIMIZED
# ==============================================================================
class RMSNorm2(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=8, head_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = head_dim * n_heads
        context_dim = query_dim if context_dim is None else context_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm2(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm2(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.gradient_checkpointing = False
        
    def forward(self, x, context=None):
        # Memory-efficient attention with optional gradient checkpointing
        def _attn_forward(x, context):
            context = x if context is None else context
            q = self.q_proj(x)
            k = self.k_proj(context)
            v = self.v_proj(context)
            q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim), (q, k, v))
            q, k = self.q_norm(q), self.k_norm(k)
            
            # Use memory efficient attention
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).reshape(out.shape[0], -1, self.n_heads * self.head_dim)
            return self.output_proj(out)
        
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(_attn_forward, x, context, use_reentrant=False)
        else:
            return _attn_forward(x, context)

class AnimaTimestepEmbedder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear_1 = nn.Linear(channels, channels, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(channels, 3 * channels, bias=False)
    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        half_dim = self.linear_1.in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        sample = self.linear_1(emb)
        sample = self.activation(sample)
        return sample, self.linear_2(sample)

class Block(nn.Module):
    def __init__(self, x_dim, context_dim, num_heads, mlp_ratio=4.0, adaln_lora_dim=256):
        super().__init__()
        self.layer_norm_self = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, None, num_heads, x_dim // num_heads)
        self.layer_norm_cross = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(x_dim, int(x_dim * mlp_ratio), bias=False), nn.GELU(), nn.Linear(int(x_dim * mlp_ratio), x_dim, bias=False))
        self.adaln_self = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
        self.adaln_cross = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
        self.adaln_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
        self.gradient_checkpointing = False
        
    def forward(self, x, t_emb, context, global_lora):
        # Ensure shapes are correct to prevent accidental broadcasting explosions
        assert x.dim() == 3, f"Expected x to be 3D (B,N,D), got {x.shape}"
        assert t_emb.dim() == 3, f"Expected t_emb to be 3D (B,1,D), got {t_emb.shape}"
        
        # Self Attention
        shift, scale, gate = (self.adaln_self(t_emb) + global_lora).chunk(3, dim=-1)
        x = x + gate.unsqueeze(1) * self.self_attn(self.layer_norm_self(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        
        # Cross Attention
        shift, scale, gate = (self.adaln_cross(t_emb) + global_lora).chunk(3, dim=-1)
        x = x + gate.unsqueeze(1) * self.cross_attn(self.layer_norm_cross(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), context)
        
        # MLP
        shift, scale, gate = (self.adaln_mlp(t_emb) + global_lora).chunk(3, dim=-1)
        x = x + gate.unsqueeze(1) * self.mlp(self.layer_norm_mlp(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_channels, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, out_channels, bias=False)
    def forward(self, x):
        B, C, T, H, W = x.shape
        # CRITICAL: Verify patching is happening
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image size ({H},{W}) must be divisible by patch_size {self.patch_size}")
        x = rearrange(x, 'b c t (h p1) (w p2) -> b t h w (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        return self.proj(x)

class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels, patch_size, adaln_lora_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, patch_size * patch_size * out_channels, bias=False)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 2 * model_channels, bias=False))
        self.model_channels = model_channels
    def forward(self, x, t_emb, global_lora):
        shift, scale = (self.adaln_modulation(t_emb) + global_lora[:, :, :2 * self.model_channels]).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))

class LLMAdapterRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.to(torch.float32)
        variance = hidden_states_fp32.pow(2).mean(-1, keepdim=True)
        hidden_states_fp32 = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states_fp32.to(input_dtype)
        return self.weight * hidden_states

class LLMAdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        target_dtype = self.q_proj.weight.dtype
        x = x.to(target_dtype)
        if context is not None and context is not x:
            context = context.to(target_dtype)
            
        context = x if context is None else context
        batch_size = x.shape[0]
        seq_len_q = x.shape[1]
        seq_len_kv = context.shape[1]
        
        q_shape = (batch_size, seq_len_q, self.n_heads, self.head_dim)
        kv_shape = (batch_size, seq_len_kv, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(context).view(kv_shape)).transpose(1, 2)
        value_states = self.v_proj(context).view(kv_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states = _adapter_apply_rotary_pos_emb(query_states, cos, sin)
        if position_embeddings_context is not None:
            cos, sin = position_embeddings_context
            key_states = _adapter_apply_rotary_pos_emb(key_states, cos, sin)

        attn_mask = None
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = mask.to(query_states.device)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(1).unsqueeze(1)
                elif mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.shape[-2] == 1 and seq_len_q > 1:
                    mask = mask.expand(batch_size, 1, seq_len_q, seq_len_kv)
                elif mask.shape[-2] != seq_len_q:
                    mask = mask.expand(batch_size, 1, seq_len_q, seq_len_kv)
                attn_mask = torch.where(mask, 0.0, float('-inf')).to(query_states.dtype)
            else:
                attn_mask = mask.to(query_states.dtype) if mask.dtype != query_states.dtype else mask

        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, 
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len_q, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, self_attn=False, layer_norm=False):
        super().__init__()
        self.has_self_attn = self_attn
        self.gradient_checkpointing = False
        norm_cls = nn.LayerNorm if layer_norm else LLMAdapterRMSNorm

        if self.has_self_attn:
            self.norm_self_attn = norm_cls(model_dim)
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim // num_heads,
            )

        self.norm_cross_attn = norm_cls(model_dim)
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )

        self.norm_mlp = norm_cls(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim)
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None,
                position_embeddings=None, position_embeddings_context=None):
        if self.has_self_attn:
            normed = self.norm_self_attn(x)
            
            def self_attn_forward(hidden):
                return self.self_attn(
                    hidden, 
                    mask=target_attention_mask,
                    context=None,
                    position_embeddings=position_embeddings,
                    position_embeddings_context=position_embeddings
                )
            
            if self.training and self.gradient_checkpointing:
                attn_out = torch.utils.checkpoint.checkpoint(self_attn_forward, normed, use_reentrant=False)
            else:
                attn_out = self_attn_forward(normed)
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        
        def cross_attn_forward(hidden):
            return self.cross_attn(
                hidden, 
                mask=source_attention_mask,
                context=context,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context
            )
        
        if self.training and self.gradient_checkpointing:
            attn_out = torch.utils.checkpoint.checkpoint(cross_attn_forward, normed, use_reentrant=False)
        else:
            attn_out = cross_attn_forward(normed)
        x = x + attn_out

        def mlp_forward(hidden):
            return self.mlp(self.norm_mlp(hidden))
        
        if self.training and self.gradient_checkpointing:
            x = x + torch.utils.checkpoint.checkpoint(mlp_forward, x, use_reentrant=False)
        else:
            x = x + mlp_forward(x)
            
        return x
    
def _adapter_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (_adapter_rotate_half(x) * sin)
    return x_embed

class AdapterRotaryEmbedding(nn.Module):
    def __init__(self, head_dim, rope_theta=10000.0):
        super().__init__()
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class LLMAdapter(nn.Module):
    def __init__(self, source_dim, target_dim, model_dim, num_layers=6, num_heads=16,
                 embed=None, self_attn=False, layer_norm=False, vocab_size=QWEN_VOCAB_SIZE):
        super().__init__()
        if embed is not None:
            self.embed = nn.Embedding.from_pretrained(embed.weight)
        else:
            self.embed = nn.Embedding(vocab_size, target_dim)
            
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            LLMAdapterTransformerBlock(source_dim, model_dim, num_heads=num_heads,
                                       self_attn=self_attn, layer_norm=layer_norm)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = LLMAdapterRMSNorm(target_dim)
        self.vocab_size = vocab_size

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        dtype = self.embed.weight.dtype
        device = self.embed.weight.device
        
        source_hidden_states = source_hidden_states.to(dtype=dtype, device=device)
        target_input_ids = target_input_ids.to(device=device)
        
        if target_input_ids.max() >= self.vocab_size:
            target_input_ids = torch.clamp(target_input_ids, 0, self.vocab_size - 1)
        
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(device=device)
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 2:
                target_attention_mask = target_attention_mask.unsqueeze(1).unsqueeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(device=device)
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 2:
                source_attention_mask = source_attention_mask.unsqueeze(1).unsqueeze(1)

        x = self.in_proj(self.embed(target_input_ids))
        context = source_hidden_states
        
        position_ids = torch.arange(x.shape[1], device=device).unsqueeze(0).expand(x.shape[0], -1)
        position_ids_context = torch.arange(context.shape[1], device=device).unsqueeze(0).expand(context.shape[0], -1)
        
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        
        for i, block in enumerate(self.blocks):
            x = block(x, context, target_attention_mask=target_attention_mask,
                      source_attention_mask=source_attention_mask,
                      position_embeddings=position_embeddings,
                      position_embeddings_context=position_embeddings_context)
                
        return self.norm(self.out_proj(x))
    
class MiniTrainDIT(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, patch_spatial=2, model_channels=2048, num_blocks=28, num_heads=16, crossattn_emb_channels=1024, use_llm_adapter=True, vocab_size=QWEN_VOCAB_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.use_llm_adapter = use_llm_adapter
        self.x_embedder = PatchEmbed(patch_spatial, in_channels, model_channels)
        self.t_embedder = AnimaTimestepEmbedder(model_channels)
        if use_llm_adapter:
            self.llm_adapter = LLMAdapter(source_dim=1024, target_dim=1024, model_dim=1024, vocab_size=vocab_size, self_attn=True)
        self.blocks = nn.ModuleList([Block(model_channels, crossattn_emb_channels, num_heads, adaln_lora_dim=256) for _ in range(num_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels, patch_spatial, adaln_lora_dim=256)
        self.t_embedding_norm = RMSNorm2(model_channels, eps=1e-6)

    def forward(self, x, timesteps, crossattn_emb, **kwargs):
        # Ensure input is 5D (B,C,T,H,W)
        if x.dim() == 4:
            x = x.unsqueeze(2)
            
        if self.in_channels == 17 and x.shape[1] == 16:
            mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4], device=x.device, dtype=x.dtype)
            x = torch.cat([x, mask], dim=1)
            
        x = self.x_embedder(x)
        B, T, H, W, D = x.shape
        
        # CRITICAL CHECK: Ensure sequence length is manageable
        seq_len = T * H * W
        if seq_len > TARGET_SEQ_LEN * 2:  # Allow some tolerance but catch 16384
            print(f"[WARNING] Sequence length {seq_len} is too large! Forcing downsample.")
            # Emergency downsample by averaging patches
            x = rearrange(x, 'b t h w d -> b d (t h w)')
            x = F.adaptive_avg_pool1d(x, TARGET_SEQ_LEN)
            x = rearrange(x, 'b d n -> b 1 (n // 64) 64 d')  # Rough reshaping, may need adjustment
            B, T, H, W, D = x.shape
            seq_len = T * H * W
            
        assert seq_len == TARGET_SEQ_LEN, f"Sequence length must be {TARGET_SEQ_LEN}, got {seq_len} (T={T}, H={H}, W={W})"
        
        t_local, t_global = self.t_embedder(timesteps.flatten().float())
        if t_local.shape[1] != T:
            t_local = t_local.unsqueeze(1).expand(-1, T, -1)
            t_global = t_global.unsqueeze(1).expand(-1, T, -1)
        t_local = self.t_embedding_norm(t_local)
        
        for block in self.blocks:
            x = rearrange(x, 'b t h w d -> b (t h w) d')
            
            if self.training and block.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, t_local, crossattn_emb, t_global, 
                    use_reentrant=False
                )
            else:
                x = block(x, t_local, crossattn_emb, t_global)
                
            x = rearrange(x, 'b (t h w) d -> b t h w d', t=T, h=H, w=W)
            
        x = rearrange(x, 'b t h w d -> b (t h w) d')
        x = self.final_layer(x, t_local, t_global)
        return rearrange(x, 'b (t h w) (p1 p2 c) -> b c t (h p1) (w p2)', t=T, h=H, w=W, p1=self.patch_spatial, p2=self.patch_spatial)

# ==============================================================================
# LOADING UTILITIES
# ==============================================================================
def load_anima_dit(dit_path, dtype, device="cpu", vocab_size=QWEN_VOCAB_SIZE):
    print(f"Loading DiT from {dit_path}...")
    state_dict = load_file(dit_path, device="cpu")
    new_sd = {k[4:] if k.startswith('net.') else k: v for k, v in state_dict.items()}
    state_dict = new_sd
    try:
        q_key = next(k for k in state_dict.keys() if 'self_attn.q_proj.weight' in k)
        hidden_size = state_dict[q_key].shape[0]
    except StopIteration:
        hidden_size = 2048
    max_block = 0
    for k in state_dict.keys():
        if k.startswith('blocks.'):
            try:
                block_num = int(k.split('.')[1])
                max_block = max(max_block, block_num)
            except:
                pass
    num_blocks = max_block + 1
    print(f" Config: {num_blocks} blocks, {hidden_size} channels, vocab_size={vocab_size}")
    model = MiniTrainDIT(in_channels=17, out_channels=16, patch_spatial=2, model_channels=hidden_size,
                         num_blocks=num_blocks, num_heads=16 if hidden_size == 2048 else 20,
                         crossattn_emb_channels=1024, use_llm_adapter=True, vocab_size=vocab_size)
    
    mapped_sd = {}
    
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('pos_embed.proj.'):
            new_k = new_k.replace('pos_embed.proj.', 'x_embedder.proj.')
        if 'x_embedder.proj.1.weight' in new_k:
            new_k = new_k.replace('x_embedder.proj.1.weight', 'x_embedder.proj.weight')
        if new_k.startswith('timestep_embedder.'):
            new_k = new_k.replace('timestep_embedder.', 't_embedder.')
        if 't_embedder.1.linear_1' in new_k:
            new_k = new_k.replace('t_embedder.1.linear_1', 't_embedder.linear_1')
        if 't_embedder.1.linear_2' in new_k:
            new_k = new_k.replace('t_embedder.1.linear_2', 't_embedder.linear_2')
        if '.adaln_modulation_self_attn.' in new_k:
            new_k = new_k.replace('.adaln_modulation_self_attn.', '.adaln_self.')
        if '.adaln_modulation_cross_attn.' in new_k:
            new_k = new_k.replace('.adaln_modulation_cross_attn.', '.adaln_cross.')
        if '.adaln_modulation_mlp.' in new_k:
            new_k = new_k.replace('.adaln_modulation_mlp.', '.adaln_mlp.')
        if 'mlp.layer1.weight' in new_k:
            new_k = new_k.replace('mlp.layer1.weight', 'mlp.0.weight')
        if 'mlp.layer2.weight' in new_k:
            new_k = new_k.replace('mlp.layer2.weight', 'mlp.2.weight')
        
        if 'llm_adapter.embed.weight' in new_k:
            if v.shape[0] != vocab_size:
                print(f"  Resizing embedding layer from {v.shape[0]} to {vocab_size}")
                new_embed = torch.zeros(vocab_size, v.shape[1], dtype=v.dtype, device=v.device)
                min_vocab = min(v.shape[0], vocab_size)
                new_embed[:min_vocab] = v[:min_vocab]
                v = new_embed
                print(f"  Copied {min_vocab} token embeddings, initialized rest randomly")
        
        mapped_sd[new_k] = v
    
    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    if missing:
        real_missing = [k for k in missing if not any(buf in k for buf in ('seq', 'dim_spatial', 'inv_freq', 'pos_embedder'))]
        if real_missing:
            print(f" Warning: Missing {len(real_missing)} keys. Sample: {real_missing[:3]}")
    if unexpected:
        print(f" Warning: Unexpected {len(unexpected)} keys. Sample: {unexpected[:3]}")
        self_attn_unexpected = [k for k in unexpected if 'self_attn' in k]
        if self_attn_unexpected:
            print(f"  Note: {len(self_attn_unexpected)} self_attn keys in checkpoint")
    
    if hasattr(model, 'llm_adapter') and model.llm_adapter is not None:
        if isinstance(model.llm_adapter.in_proj, nn.Linear):
            nn.init.normal_(model.llm_adapter.in_proj.weight, std=0.02)
            if model.llm_adapter.in_proj.bias is not None:
                nn.init.zeros_(model.llm_adapter.in_proj.bias)
    
    for name, p in model.named_parameters():
        if any(k in name for k in ['x_embedder', 't_embedder', 'final_layer']):
            p.data = p.data.to(torch.float32)
        else:
            p.data = p.data.to(dtype)
            
    print(f" Model device: {device}, dtype: {dtype}")
    print_gpu_memory("After model load")
    return model.to(device)

def load_qwen_vae(vae_path, device):
    print(f"Loading Qwen VAE from {vae_path}...")
    checkpoint_sd = load_file(vae_path, device="cpu")
    
    if 'conv1.weight' in checkpoint_sd:
        conv1_shape = checkpoint_sd['conv1.weight'].shape
        z_dim = conv1_shape[0] // 2
    else:
        z_dim = 16
    
    if 'encoder.conv1.weight' in checkpoint_sd:
        enc_conv1_shape = checkpoint_sd['encoder.conv1.weight'].shape
        dim = enc_conv1_shape[0]
    else:
        dim = 96
    
    vae = WanVAE_(
        dim=dim, 
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4], 
        num_res_blocks=2,
        temperal_downsample=[False, True, True], 
        dropout=0.0
    )
    
    missing, unexpected = vae.load_state_dict(checkpoint_sd, strict=False)
    vae = vae.eval().requires_grad_(False).to(device, dtype=torch.float32)
    
    if z_dim == 16:
        mean = torch.tensor([0.0] * 16, device=device)
        std = torch.tensor([10.0] * 16, device=device)
    else:
        mean = 0.0
        std = 10.0
    
    print(f"[OK] Loaded WanVAE with dim={dim}, z_dim={z_dim}")
    return vae, [mean, std]

class CustomAnimaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def load_qwen3_encoder(path, device="cpu", dtype=torch.bfloat16):
    print(f"\n--- Loading Qwen Text Encoder from {path} ---")
    
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=32768,
    )
    
    model = Qwen3Model(config)
    
    head_dim = 128
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    
    for i, layer in enumerate(model.layers):
        layer.self_attn.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True).to(device)
        layer.self_attn.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True).to(device)
        layer.self_attn.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True).to(device)
        layer.self_attn.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(device)
        layer.self_attn.head_dim = head_dim
        layer.self_attn.rotary_emb = CustomAnimaRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=1000000.0,
            device=device
        )
    
    if os.path.isfile(path):
        state_dict_path = path
    else:
        files = list(Path(path).glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors found in {path}")
        state_dict_path = str(files[0])
        
    state_dict = load_file(state_dict_path)
    new_sd = {k[6:] if k.startswith('model.') else k: v for k, v in state_dict.items()}
    
    skip_patterns = ['rotary_emb', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
    filtered_sd = {k: v for k, v in new_sd.items() if not any(p in k for p in skip_patterns)}
    
    model.load_state_dict(filtered_sd, strict=False)
    model = model.to(device).eval().to(dtype=dtype)
    
    try:
        if os.path.isdir(path):
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

class AnimaDataset(Dataset):
    def __init__(self, data_dir, tokenizer, text_encoder, vae, vae_scale, device):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / ".anima_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.image_files = list(self.data_dir.rglob("*.jpg")) + list(self.data_dir.rglob("*.png")) + list(self.data_dir.rglob("*.webp"))
        self.vae_scale = vae_scale
        self.device = device
        
        # Check if cache exists; if you changed VAE/Resolution, DELETE the cache folder manually first!
        if not list(self.cache_dir.glob("*.pt")):
            self._cache_data(tokenizer, text_encoder, vae)
        else:
            print(f"Found existing cache in {self.cache_dir}. Note: Delete this folder if you change VAE/Resolution.")

    def _cache_data(self, tokenizer, text_encoder, vae):
        print("Pre-caching latents and embeddings...")
        transform = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        text_encoder.eval()
        vae.eval()
        
        for img_path in tqdm(self.image_files):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)
                elif img_tensor.shape[0] == 4:
                    img_tensor = img_tensor[:3, :, :]
                
                # Format for WanVAE: (1, 3, 1, H, W)
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(2).to(self.device)
                
                with torch.no_grad():
                    latent = vae.encode(img_tensor, self.vae_scale)
                    # Ensure latent is reasonable size
                    if latent.shape[-1] > 128:
                        print(f"[WARNING] Latent for {img_path.name} is {latent.shape[-1]}x{latent.shape[-1]}, resizing to 128x128")
                        latent = F.interpolate(latent, size=(1, 128, 128), mode='nearest')
                    
                caption = img_path.stem.replace("_", " ")
                txt_path = img_path.with_suffix(".txt")
                if txt_path.exists():
                    caption = open(txt_path, "r", encoding="utf-8").read().strip()
                    
                inputs = tokenizer(caption, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(self.device)
                
                with torch.no_grad():
                    text_emb = text_encoder(**inputs, output_hidden_states=True).hidden_states[-1]
                    
                safe_name = str(img_path.relative_to(self.data_dir)).replace(os.sep, "_").replace(":", "_")
                
                # Save to CPU to save VRAM
                torch.save({
                    "latent": latent.cpu(), # Shape might be (1, 16, 1, H, W)
                    "text_emb": text_emb.cpu().squeeze(0),
                    "t5_ids": inputs.input_ids.cpu().squeeze(0),
                    "caption": caption
                }, self.cache_dir / f"{safe_name}.pt")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            clean_memory()

    def __len__(self):
        return len(list(self.cache_dir.glob("*.pt")))

    def __getitem__(self, idx):
        files = sorted(list(self.cache_dir.glob("*.pt")))
        # Use weights_only=True for security, map to CPU initially
        data = torch.load(files[idx], map_location="cpu", weights_only=True)
        
        # Ensure latent is squeezed correctly from (1, C, T, H, W) to (C, T, H, W)
        latent = data["latent"]
        if latent.shape[0] == 1: 
            latent = latent.squeeze(0)
            
        return latent, data["text_emb"], data["t5_ids"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16

    print("Loading tokenizer...")
    try:
        if os.path.isdir(QWEN3_PATH):
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    
    vocab_size = len(tokenizer)
    
    # 1. Load DiT
    dit = load_anima_dit(ANIMA_DIT_PATH, dtype, device, vocab_size=vocab_size)
    
    # 2. Load Encoders for Caching
    print("Loading Text Encoder and VAE for caching...")
    text_encoder = load_qwen3_encoder(QWEN3_PATH, device, dtype)[1]
    vae, vae_scale = load_qwen_vae(VAE_PATH, device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 3. Create Dataset (Caches if needed)
    dataset = AnimaDataset(DATA_DIR, tokenizer, text_encoder, vae, vae_scale, device)
    
    # 4. Aggressive Cleanup
    print("Freeing Text Encoder and VAE from GPU memory...")
    del text_encoder
    del vae
    del tokenizer
    clean_memory()
    print_gpu_memory("After cleanup")
    
    # 5. Setup Training
    dit.train()
    # Enable Gradient Checkpointing on everything including attention internals
    for block in dit.blocks:
        block.gradient_checkpointing = True
        block.self_attn.gradient_checkpointing = True  # Enable attention-level checkpointing
        block.cross_attn.gradient_checkpointing = True
        
    if hasattr(dit, 'llm_adapter') and dit.llm_adapter is not None:
        for block in dit.llm_adapter.blocks:
            block.gradient_checkpointing = True
            
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    
    optimizer = RavenAdamW(
        dit.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.9, 0.98),
        weight_decay=0.06,
        eps=1e-08,
        debias_strength=0.9,
        use_grad_centralization=False,
        gc_alpha=0.9
    )

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        dit.load_state_dict(torch.load(args.resume, map_location="cpu"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pbar = tqdm(total=MAX_STEPS, initial=start_step, desc="Training")
    step = start_step
    loss_accum = 0.0

    # DEBUG FLAG
    printed_shape = False

    while step < MAX_STEPS:
        for latents, text_embs, t5_ids in dataloader:
            latents = latents.to(device, dtype=dtype)
            text_embs = text_embs.to(device, dtype=dtype) 
            t5_ids = t5_ids.to(device, dtype=torch.long)
            
            # --- CRITICAL MEMORY FIX START ---
            # 1. Normalize Dimensions: Ensure (B, C, T, H, W)
            if latents.dim() == 4:
                latents = latents.unsqueeze(2) # Add T dim if missing
            
            # 2. DEBUG PRINT: Check what the VAE actually gave us
            if not printed_shape:
                print(f"\n[DEBUG] Latent Shape Input: {latents.shape}")
            
            # 3. FORCE RESIZE: If Latents are huge (e.g., 1024x1024), downsample to 128x128
            # Standard Anima/Wan is 128x128 latent for 1024px image
            TARGET_SIZE = 128
            if latents.shape[-1] > TARGET_SIZE:
                if not printed_shape:
                    print(f"[DEBUG] Resizing latents from {latents.shape[-1]} to {TARGET_SIZE} to save VRAM")
                
                # F.interpolate expects (B, C, D, H, W)
                latents = F.interpolate(
                    latents, 
                    size=(1, TARGET_SIZE, TARGET_SIZE), 
                    mode='nearest'
                )
            
            if not printed_shape:
                print(f"[DEBUG] Latent Shape Final: {latents.shape}")
                printed_shape = True
            # --- CRITICAL MEMORY FIX END ---

            t = torch.rand(latents.shape[0], device=device)
            t_shift = (RF_SHIFT * t) / (1.0 + (RF_SHIFT - 1.0) * t)
            noise = torch.randn_like(latents)
            noisy = (1 - t_shift.view(-1, 1, 1, 1, 1)) * latents + t_shift.view(-1, 1, 1, 1, 1) * noise
            target = noise - latents
            
            # Forward pass with Autocast
            with torch.autocast(device_type="cuda", dtype=dtype):
                if hasattr(dit, 'llm_adapter'):
                    crossattn_emb = dit.llm_adapter(text_embs, t5_ids)
                else:
                    crossattn_emb = text_embs
                    
                pred = dit(noisy, t_shift.unsqueeze(1) * 1000, crossattn_emb)
                loss = F.mse_loss(pred.float(), target.float())
            
            (loss / GRADIENT_ACCUMULATION).backward()
            
            loss_accum += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                optimizer.step()
                # set_to_none=True saves memory!
                optimizer.zero_grad(set_to_none=True)
                
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss_accum / GRADIENT_ACCUMULATION})
            loss_accum = 0.0
            
            # Cleanup less frequently to save CPU overhead
            if step % 100 == 0:
                clean_memory()
                
            if step % SAVE_EVERY == 0:
                save_path = os.path.join(OUTPUT_DIR, f"anima_step_{step}.pt")
                torch.save(dit.state_dict(), save_path)
            if step >= MAX_STEPS:
                break
    
    pbar.close()
    print("Training complete!")

if __name__ == "__main__":
    main()