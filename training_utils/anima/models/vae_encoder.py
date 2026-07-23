# Copyright 2026 Aozora Trainer contributors
# Portions derived from ModelScope DiffSynth-Studio 2.0.16 and modified for Aozora.
# Licensed under the Apache License, Version 2.0. See the repository LICENSE.
# SPDX-License-Identifier: Apache-2.0

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

CACHE_T = 2


def check_is_instance(model, module_class):
    return isinstance(model, module_class) or (
        hasattr(model, "module") and isinstance(model.module, module_class)
    )


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

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


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        if mode not in ("downsample2d", "downsample3d"):
            raise ValueError(f"Encode-only VAE cannot use {mode}.")
        self.mode = mode
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(dim, dim, 3, stride=(2, 2)),
        )
        if mode == "downsample3d":
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        frames = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=frames)
        if self.mode == "downsample3d" and feat_cache is not None:
            index = feat_idx[0]
            if feat_cache[index] is None:
                feat_cache[index] = x.clone()
            else:
                cached = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([feat_cache[index][:, :, -1:], x], dim=2))
                feat_cache[index] = cached
            feat_idx[0] += 1
        return x, feat_cache, feat_idx

class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
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
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
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
        return x + h, feat_cache, feat_idx


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
            0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            #attn_mask=block_causal_mask(q, block_size=h * w)
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
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

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout),
                                    AttentionBlock(out_dim),
                                    ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(),
                                  CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
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

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
                x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
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
        return x, feat_cache, feat_idx


def count_conv3d(model):
    return sum(isinstance(module, CausalConv3d) for module in model.modules())


class _EncoderCore(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(
            dim=96,
            z_dim=z_dim * 2,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    def encode(self, video, scale):
        cache = [None] * count_conv3d(self.encoder)
        chunks = 1 + (video.shape[2] - 1) // 4
        outputs = []
        for index in range(chunks):
            feature_index = [0]
            frames = video[:, :, :1] if index == 0 else video[:, :, 1 + 4 * (index - 1):1 + 4 * index]
            output, cache, feature_index = self.encoder(frames, cache, feature_index)
            outputs.append(output)
        mu, _log_variance = self.conv1(torch.cat(outputs, dim=2)).chunk(2, dim=1)
        mean, inverse_std = [value.to(dtype=mu.dtype, device=mu.device) for value in scale]
        return (mu - mean.view(1, self.z_dim, 1, 1, 1)) * inverse_std.view(1, self.z_dim, 1, 1, 1)


class WanVideoVAE(nn.Module):
    """Aozora's encode-only Anima VAE with optional spatial tiling."""

    def __init__(self, z_dim=16):
        super().__init__()
        self.reset_scale()
        self.model = _EncoderCore(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim

    def reset_scale(self):
        mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
        std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
        self.scale = [torch.tensor(mean), 1.0 / torch.tensor(std)]

    @staticmethod
    def _axis_mask(length, lower_bound, upper_bound, border):
        mask = torch.ones(length)
        ramp = (torch.arange(border) + 1) / border
        if not lower_bound:
            mask[:border] = ramp
        if not upper_bound:
            mask[-border:] = ramp.flip(0)
        return mask

    def _tile_mask(self, data, bounds, border):
        height, width = data.shape[-2:]
        h = self._axis_mask(height, bounds[0], bounds[1], border[0])[:, None]
        w = self._axis_mask(width, bounds[2], bounds[3], border[1])[None, :]
        return torch.minimum(h, w).reshape(1, 1, 1, height, width)

    def _single_encode(self, video, device):
        return self.model.encode(video.to(device), self.scale)

    def _tiled_encode(self, video, device, tile_size, tile_stride):
        _, _, frames, height, width = video.shape
        tile_h, tile_w = tile_size
        stride_h, stride_w = tile_stride
        tasks = []
        for top in range(0, height, stride_h):
            if top >= stride_h and top - stride_h + tile_h >= height:
                continue
            for left in range(0, width, stride_w):
                if left >= stride_w and left - stride_w + tile_w >= width:
                    continue
                tasks.append((top, top + tile_h, left, left + tile_w))

        output_shape = (1, self.z_dim, (frames + 3) // 4, height // 8, width // 8)
        weights = torch.zeros((1, 1, output_shape[2], output_shape[3], output_shape[4]), dtype=video.dtype)
        values = torch.zeros(output_shape, dtype=video.dtype)
        for top, bottom, left, right in tqdm(tasks, desc="VAE encoding"):
            encoded = self.model.encode(video[:, :, :, top:bottom, left:right].to(device), self.scale).cpu()
            mask = self._tile_mask(
                encoded,
                (top == 0, bottom >= height, left == 0, right >= width),
                ((tile_h - stride_h) // 8, (tile_w - stride_w) // 8),
            ).to(dtype=video.dtype)
            out_top, out_left = top // 8, left // 8
            out_bottom, out_right = out_top + encoded.shape[-2], out_left + encoded.shape[-1]
            values[:, :, :, out_top:out_bottom, out_left:out_right] += encoded * mask
            weights[:, :, :, out_top:out_bottom, out_left:out_right] += mask
        return values / weights

    def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        outputs = []
        for video in videos.cpu():
            video = video.unsqueeze(0)
            if tiled:
                scaled_size = tuple(value * 8 for value in tile_size)
                scaled_stride = tuple(value * 8 for value in tile_stride)
                encoded = self._tiled_encode(video, device, scaled_size, scaled_stride)
            else:
                encoded = self._single_encode(video, device)
            outputs.append(encoded.squeeze(0))
        return torch.stack(outputs)


