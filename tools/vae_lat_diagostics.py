"""
VAE Latent Inspector
====================
Diagnoses what your training pipeline actually feeds the UNet by replicating
the exact resize/encode/normalize path and visualizing every stage.

Loads:
- A model checkpoint (extracts its VAE) OR a standalone VAE file
- A test image
- Optionally, a cached _lat.pt + _te.pt pair from your training cache

Shows:
- Original image
- Resized input (what the VAE actually sees)
- Per-channel latent visualization (with stats)
- Round-trip decoded image (what the model is being asked to reproduce)
- If a cached latent is loaded: a fresh-encode vs cached-encode diff,
  so you can confirm the cache matches current encoding.
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image, ImageOps
from PyQt6 import QtCore, QtGui, QtWidgets
from safetensors import safe_open
from torchvision import transforms


# =============================================================================
# Flux2 BN-stats extraction + per-channel normalization
# =============================================================================
# The Flux.2 VAE normalizes latents using BatchNorm running statistics with
# affine=False, eps=1e-4. Encoder output is (B, 32, H/32, W/32), then patch
# rearranged to (B, 128, H/64, W/64), then the 128-vector BN stats are applied.
#
# This file works with EITHER the post-rearrange 128ch latent (preferred,
# matches BFL exactly) OR the pre-rearrange 32ch latent (what older pipelines
# extending SDXL UNet use). For the 32ch case we reduce the 128-vector to 32
# by averaging the 4 spatial-patch positions per original channel.

BN_EPS = 1e-4
BN_MEAN_KEYS = [
    "bn.running_mean",
    "normalize.bn.running_mean",
    "normalize.running_mean",
    "first_stage_model.normalize.bn.running_mean",
]
BN_VAR_KEYS = [
    "bn.running_var",
    "normalize.bn.running_var",
    "normalize.running_var",
    "first_stage_model.normalize.bn.running_var",
]


def extract_bn_stats(safetensors_path):
    """Pull (running_mean, running_var) 128-vectors out of a Flux.2 VAE file.
    Returns (mean_128, var_128) as torch.float32 tensors, or (None, None) if
    not found."""
    try:
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            mean_t = var_t = None
            for k in keys:
                if k in BN_MEAN_KEYS or k.endswith("bn.running_mean"):
                    mean_t = f.get_tensor(k).float()
                if k in BN_VAR_KEYS or k.endswith("bn.running_var"):
                    var_t = f.get_tensor(k).float()
            if mean_t is None or var_t is None:
                return None, None
            return mean_t, var_t
    except Exception as e:
        print(f"WARNING: BN stats extraction failed: {e}")
        return None, None


def reduce_128_to_32(vec_128, agg="mean"):
    """The 128-vector packs (32 channels x 4 spatial-patch positions). For
    32ch latents (pre-patch-rearrange), average across the 4 patch positions
    to get a 32-vector. For variance, averaging is correct because the 4
    positions are i.i.d. samples of the same channel distribution."""
    assert vec_128.numel() == 128, f"expected 128 elements, got {vec_128.numel()}"
    return vec_128.view(32, 4).mean(dim=1)


def get_normalization_vectors(bn_mean_128, bn_var_128, latent_channels):
    """Return (mu, sigma) appropriate for the given channel count.
    For 128ch: use the raw vectors.
    For 32ch: reduce by averaging across the 4 patch positions."""
    if latent_channels == 128:
        mu = bn_mean_128.clone()
        sigma = torch.sqrt(bn_var_128 + BN_EPS)
    elif latent_channels == 32:
        mu = reduce_128_to_32(bn_mean_128)
        # Variance reduction first, then sqrt — averaging stds is wrong
        var_32 = bn_var_128.view(32, 4).mean(dim=1)
        sigma = torch.sqrt(var_32 + BN_EPS)
    else:
        raise ValueError(f"unsupported latent channel count {latent_channels}")
    return mu, sigma


def apply_per_channel_norm(latent, mu, sigma):
    """Apply (latent - mu) / sigma along channel dim. mu/sigma are 1D
    tensors of length C; latent is (B, C, H, W)."""
    mu = mu.to(latent.device, latent.dtype).view(1, -1, 1, 1)
    sigma = sigma.to(latent.device, latent.dtype).view(1, -1, 1, 1)
    return (latent - mu) / sigma


def invert_per_channel_norm(latent, mu, sigma):
    """Inverse of apply_per_channel_norm: latent * sigma + mu."""
    mu = mu.to(latent.device, latent.dtype).view(1, -1, 1, 1)
    sigma = sigma.to(latent.device, latent.dtype).view(1, -1, 1, 1)
    return latent * sigma + mu


def patchify_32_to_128(latent_32):
    """Flux.2 2x2 latent packing: (B,32,H,W) -> (B,128,H/2,W/2)."""
    b, c, h, w = latent_32.shape
    if c != 32:
        raise ValueError(f"expected 32 channels before Flux packing, got {c}")
    if h % 2 or w % 2:
        raise ValueError(f"latent H/W must be even for Flux packing, got {h}x{w}")
    return latent_32.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify_128_to_32(latent_128):
    """Inverse Flux.2 2x2 latent packing: (B,128,H,W) -> (B,32,H*2,W*2)."""
    b, c4, h, w = latent_128.shape
    if c4 != 128:
        raise ValueError(f"expected 128 channels before Flux unpacking, got {c4}")
    c = c4 // 4
    return latent_128.reshape(b, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c, h * 2, w * 2)


STYLE = """
QWidget { background: #15171b; color: #e7e9ee; font-family: Segoe UI, sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #353b45; border-radius: 6px; margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #78c2ad; font-weight: 600; }
QPushButton { background: #242a31; border: 1px solid #3d4651; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { border-color: #78c2ad; color: #78c2ad; }
QPushButton:disabled { color: #69717d; border-color: #2b3037; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #101216; border: 1px solid #353b45; border-radius: 5px; padding: 6px; }
QPlainTextEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; font-family: Consolas, monospace; }
QLabel#ImagePanel { background: #050607; border: 1px solid #353b45; border-radius: 4px; }
QLabel#TitleLabel { color: #aeb7c2; font-weight: 600; }
"""


# Mirror of the training script's bucket list and resize logic
STANDARD_SDXL_BUCKETS = [
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1440, 720), (720, 1440),
    (1536, 640), (640, 1536),
    (1600, 512), (512, 1600),
    (896, 896), (768, 768),
]


def get_optimal_bucket(orig_w, orig_h, target_area=None):
    orig_ar = orig_w / max(orig_h, 1)
    if target_area is None:
        target_area = 1024 * 1024

    def score(bw, bh):
        bucket_ar = bw / max(bh, 1)
        bucket_area = bw * bh
        ar_error = abs(bucket_ar - orig_ar) / max(orig_ar, 0.01)
        if bucket_area > 0 and target_area > 0:
            area_error = abs(math.log(bucket_area / target_area))
        else:
            area_error = 100.0
        return ar_error * 10.0 + area_error

    best = min(STANDARD_SDXL_BUCKETS, key=lambda b: score(*b))
    bw, bh = best
    if bw > orig_w and bh > orig_h:
        fitting = [(w, h) for w, h in STANDARD_SDXL_BUCKETS if w <= orig_w and h <= orig_h]
        if fitting:
            best = max(fitting, key=lambda b: b[0] * b[1])
        else:
            best = min(STANDARD_SDXL_BUCKETS, key=lambda b: b[0] * b[1])
    return best


def smart_resize(image, target_w, target_h):
    orig_w, orig_h = image.size
    scale = max(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    new_w = max(int(round(orig_w * scale)), target_w)
    new_h = max(int(round(orig_h * scale)), target_h)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    crop_left = (new_w - target_w) // 2
    crop_top = (new_h - target_h) // 2
    return resized.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))


def fix_alpha(image):
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "PA", "LA"):
        return image.convert("RGB")
    return image.convert("RGB")


def tensor_to_pil(tensor):
    arr = (tensor / 2 + 0.5).clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((arr * 255).round().astype(np.uint8))


def pil_to_pixmap(image, max_size):
    image = ImageOps.contain(image.convert("RGB"), (max_size, max_size), Image.Resampling.LANCZOS)
    data = image.convert("RGBA").tobytes("raw", "RGBA")
    qimage = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimage.copy())


def latent_to_channel_grid(latent, scale_each=True):
    """Tile each latent channel as a grayscale panel for inspection."""
    arr = latent.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    c, h, w = arr.shape
    # Pick a column count that keeps the grid roughly square
    cols = min(c, 8) if c > 8 else min(c, 4)
    rows = math.ceil(c / cols)
    pad = 4
    grid = np.zeros((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad), dtype=np.float32)
    for i in range(c):
        r, col = i // cols, i % cols
        ch = arr[i]
        if scale_each:
            lo, hi = float(ch.min()), float(ch.max())
            if hi - lo > 1e-8:
                ch = (ch - lo) / (hi - lo)
            else:
                ch = ch * 0.5
        else:
            ch = (ch - arr.min()) / max(arr.max() - arr.min(), 1e-8)
        y0 = pad + r * (h + pad)
        x0 = pad + col * (w + pad)
        grid[y0:y0 + h, x0:x0 + w] = np.clip(ch, 0, 1)
    grid_rgb = (np.stack([grid] * 3, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(grid_rgb)


def parse_optional_float(text):
    text = (text or "").strip()
    if not text or text.lower() in {"auto", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def detect_vae_latent_channels(path):
    """Peek at the safetensors file to figure out the latent channel count.
    Looks at quant_conv.weight (out_channels = 2 * latent_channels for KL) or
    decoder conv_in.weight (in_channels = latent_channels)."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Try the standard KL VAE quant_conv path (covers diffusers and original SD layouts)
            for k in [
                "quant_conv.weight",
                "first_stage_model.quant_conv.weight",
                "vae.quant_conv.weight",
            ]:
                if k in keys:
                    shape = f.get_slice(k).get_shape()
                    # quant_conv outputs 2 * latent_channels (mean + logvar)
                    return shape[0] // 2
            # Fall back to decoder conv_in (takes latent_channels as input)
            for k in [
                "decoder.conv_in.weight",
                "first_stage_model.decoder.conv_in.weight",
                "vae.decoder.conv_in.weight",
            ]:
                if k in keys:
                    shape = f.get_slice(k).get_shape()
                    return shape[1]  # in_channels
    except Exception as e:
        print(f"WARNING: channel detection failed: {e}")
    return None


def load_vae_from_path(path, latent_channels=0):
    """Load VAE from a checkpoint file (full SDXL or standalone VAE).
    Auto-detects latent channel count from the file when latent_channels=0."""
    path = str(path)
    kwargs = {"torch_dtype": torch.float32, "low_cpu_mem_usage": False}

    # Resolve channel count: explicit override > auto-detect > default 4
    if latent_channels and latent_channels > 0:
        target_channels = latent_channels
    else:
        detected = detect_vae_latent_channels(path)
        target_channels = detected if detected else 4
        if detected:
            print(f"INFO: auto-detected {detected}-channel VAE from {Path(path).name}")

    # Try AutoencoderKL load with the correct channel count
    try:
        if target_channels != 4:
            return AutoencoderKL.from_single_file(
                path,
                latent_channels=target_channels,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
        return AutoencoderKL.from_single_file(path, **kwargs)
    except Exception as e:
        print(f"WARNING: AutoencoderKL.from_single_file failed ({e}), trying SDXL pipeline path...")

    # Fall back to extracting from a full SDXL pipeline (only works for 4ch)
    if target_channels == 4:
        pipe = StableDiffusionXLPipeline.from_single_file(
            path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        return pipe.vae

    # Last resort: try once more with explicit channel count and see what error surfaces
    return AutoencoderKL.from_single_file(
        path, latent_channels=target_channels, ignore_mismatched_sizes=True, **kwargs
    )


class ImageSlot(QtWidgets.QWidget):
    def __init__(self, title, min_size=240):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(title)
        label.setObjectName("TitleLabel")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image = QtWidgets.QLabel("—")
        self.image.setObjectName("ImagePanel")
        self.image.setMinimumSize(min_size, min_size)
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(self.image, 1)
        self._title = title

    def set_image(self, image, max_size=320):
        if image is None:
            self.image.setText("—")
            self.image.setPixmap(QtGui.QPixmap())
        else:
            self.image.setPixmap(pil_to_pixmap(image, max_size))


class InspectionWorker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, model_path, image_path, latent_channels, shift_text, scale_text,
                 cached_lat_path, cached_te_path, force_resolution,
                 norm_mode, bn_stats_path):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.latent_channels = latent_channels
        self.shift_text = shift_text
        self.scale_text = scale_text
        self.cached_lat_path = cached_lat_path
        self.cached_te_path = cached_te_path
        self.force_resolution = force_resolution
        self.norm_mode = norm_mode  # "raw" | "scalar" | "bn32" | "bn128"
        self.bn_stats_path = bn_stats_path

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.status.emit(f"Loading VAE on {device}...")
            vae = load_vae_from_path(self.model_path, self.latent_channels).to(device, dtype=torch.float32)
            vae.eval()
            try:
                vae.enable_slicing()
                vae.enable_tiling()
            except Exception:
                pass

            cfg_shift = getattr(vae.config, "shift_factor", None)
            cfg_scale = getattr(vae.config, "scaling_factor", None)
            override_shift = parse_optional_float(self.shift_text)
            override_scale = parse_optional_float(self.scale_text)
            shift = override_shift if override_shift is not None else cfg_shift
            scale = override_scale if override_scale is not None else (cfg_scale if cfg_scale is not None else 1.0)

            # Load BN stats if per-channel mode is selected
            bn_mu = bn_sigma = None
            bn_info = ""
            if self.norm_mode in {"bn32", "bn128", "per_channel_bn"}:
                source = self.bn_stats_path or self.model_path
                bn_mean_128, bn_var_128 = extract_bn_stats(source)
                if bn_mean_128 is None:
                    raise RuntimeError(
                        f"BN stats not found in {source}. "
                        "Provide a Flux.2 raw VAE file via the BN stats source field."
                    )
                bn_info = (
                    f"Loaded BN stats from {Path(source).name}: "
                    f"mean_128 range=[{bn_mean_128.min():.4f}, {bn_mean_128.max():.4f}], "
                    f"var_128 mean={bn_var_128.mean():.4f}\n"
                )

            # Load image
            original = fix_alpha(ImageOps.exif_transpose(Image.open(self.image_path)))
            ow, oh = original.size

            # Decide target resolution: use forced if given, else compute via training bucket logic
            if self.force_resolution and self.force_resolution[0] > 0 and self.force_resolution[1] > 0:
                target_w, target_h = self.force_resolution
            else:
                target_w, target_h = get_optimal_bucket(ow, oh, ow * oh)

            resized = smart_resize(original, target_w, target_h)

            # Match training transform exactly: ToTensor + Normalize([0.5],[0.5])
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            pixels = transform(resized).unsqueeze(0).to(device, dtype=torch.float32)

            # ENCODE → mean (matches training, which uses .latent_dist.mean)
            with torch.no_grad():
                encoded = vae.encode(pixels)
                raw_latent = encoded.latent_dist.mean
                raw_min, raw_max = raw_latent.min().item(), raw_latent.max().item()
                raw_mean, raw_std = raw_latent.mean().item(), raw_latent.std().item()

                # Apply chosen normalization.
                # raw/scalar keep native latent shape.
                # bn32 applies Flux BN reduced from 128 -> 32 channels.
                # bn128 packs native 32ch -> exact Flux 128ch before applying BN.
                latent_c = raw_latent.shape[1]
                if self.norm_mode == "raw":
                    norm_latent = raw_latent
                elif self.norm_mode == "scalar":
                    if shift is not None:
                        norm_latent = (raw_latent - shift) * scale
                    else:
                        norm_latent = raw_latent * scale
                elif self.norm_mode in {"bn32", "per_channel_bn"}:
                    bn_mu, bn_sigma = get_normalization_vectors(bn_mean_128, bn_var_128, 32)
                    norm_latent = apply_per_channel_norm(raw_latent, bn_mu, bn_sigma)
                    bn_info += (
                        f"BN mode: Flux BN reduced 128 -> 32ch\n"
                        f"  mu_32 range=[{bn_mu.min():.4f}, {bn_mu.max():.4f}]  "
                        f"sigma_32 range=[{bn_sigma.min():.4f}, {bn_sigma.max():.4f}]\n"
                    )
                elif self.norm_mode == "bn128":
                    latent_128 = patchify_32_to_128(raw_latent) if latent_c == 32 else raw_latent
                    bn_mu, bn_sigma = get_normalization_vectors(bn_mean_128, bn_var_128, 128)
                    norm_latent = apply_per_channel_norm(latent_128, bn_mu, bn_sigma)
                    bn_info += (
                        f"BN mode: Flux BN exact 128ch post-rearrange\n"
                        f"  packed latent shape={tuple(norm_latent.shape)}\n"
                        f"  mu_128 range=[{bn_mu.min():.4f}, {bn_mu.max():.4f}]  "
                        f"sigma_128 range=[{bn_sigma.min():.4f}, {bn_sigma.max():.4f}]\n"
                    )
                else:
                    raise ValueError(f"unknown normalization mode: {self.norm_mode}")

                norm_mean, norm_std = norm_latent.mean().item(), norm_latent.std().item()
                norm_min, norm_max = norm_latent.min().item(), norm_latent.max().item()

                # Per-channel stats on the normalized latent
                per_ch = []
                nl = norm_latent[0]
                for ci in range(nl.shape[0]):
                    ch = nl[ci]
                    per_ch.append((ch.mean().item(), ch.std().item(), ch.min().item(), ch.max().item()))

                # Round-trip: denormalize then decode (using inverse of selected mode)
                if self.norm_mode == "raw":
                    restored = norm_latent
                elif self.norm_mode == "scalar":
                    if shift is not None:
                        restored = norm_latent / scale + shift
                    else:
                        restored = norm_latent / scale
                elif self.norm_mode in {"bn32", "per_channel_bn"}:
                    restored = invert_per_channel_norm(norm_latent, bn_mu, bn_sigma)
                elif self.norm_mode == "bn128":
                    restored_128 = invert_per_channel_norm(norm_latent, bn_mu, bn_sigma)
                    restored = unpatchify_128_to_32(restored_128) if raw_latent.shape[1] == 32 else restored_128
                else:
                    raise ValueError(f"unknown normalization mode: {self.norm_mode}")
                decoded = vae.decode(restored)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample

            roundtrip_pil = tensor_to_pil(decoded)
            latent_grid = latent_to_channel_grid(norm_latent)

            # Compare to cached latent if provided
            cache_compare_text = ""
            cached_decoded_pil = None
            cached_grid = None
            if self.cached_lat_path and Path(self.cached_lat_path).exists():
                self.status.emit("Loading cached latent...")
                cached_payload = torch.load(self.cached_lat_path, map_location="cpu", weights_only=True)
                if isinstance(cached_payload, dict):
                    cached_latent = cached_payload.get("latents", cached_payload)
                else:
                    cached_latent = cached_payload
                cached_latent = cached_latent.float()
                if cached_latent.dim() == 3:
                    cached_latent = cached_latent.unsqueeze(0)

                # Pull shift/scale from te file if available
                cached_shift, cached_scale = shift, scale
                cached_te_caption = "(no te file loaded)"
                if self.cached_te_path and Path(self.cached_te_path).exists():
                    te_payload = torch.load(self.cached_te_path, map_location="cpu", weights_only=True)
                    cached_shift = te_payload.get("vae_shift", shift)
                    cached_scale = te_payload.get("vae_scale", scale)
                    variants = te_payload.get("caption_variants", {})
                    if "original" in variants:
                        cached_te_caption = variants["original"].get("caption", "")[:120]

                cached_grid = latent_to_channel_grid(cached_latent)

                # Decode the cached latent through the SAME vae using the cache's own shift/scale
                with torch.no_grad():
                    if cached_shift is not None:
                        cached_restored = cached_latent.to(device) / cached_scale + cached_shift
                    else:
                        cached_restored = cached_latent.to(device) / cached_scale
                    cached_dec = vae.decode(cached_restored)
                    if hasattr(cached_dec, "sample"):
                        cached_dec = cached_dec.sample
                cached_decoded_pil = tensor_to_pil(cached_dec)

                # Diff stats
                if cached_latent.shape == norm_latent.shape:
                    diff = (cached_latent.to(device) - norm_latent).abs()
                    cache_compare_text = (
                        f"Cache compare:\n"
                        f"  cached shape:    {tuple(cached_latent.shape)}  (matches fresh)\n"
                        f"  cache shift/scale: {cached_shift} / {cached_scale}\n"
                        f"  fresh shift/scale: {shift} / {scale}\n"
                        f"  cached caption:  {cached_te_caption}\n"
                        f"  |cached - fresh| mean: {diff.mean().item():.6f}\n"
                        f"  |cached - fresh| max:  {diff.max().item():.6f}\n"
                        f"  cached mean / std:     {cached_latent.mean().item():.4f} / {cached_latent.std().item():.4f}\n"
                        f"  fresh  mean / std:     {norm_mean:.4f} / {norm_std:.4f}\n"
                    )
                else:
                    cache_compare_text = (
                        f"Cache compare:\n"
                        f"  cached shape:    {tuple(cached_latent.shape)} (DIFFERENT FROM FRESH {tuple(norm_latent.shape)})\n"
                        f"  This means resolution / bucket changed between caching and now.\n"
                    )

            per_ch_text = "\n".join(
                f"  ch{ci}: mean={m:+.4f}  std={s:.4f}  min={mn:+.3f}  max={mx:+.3f}"
                for ci, (m, s, mn, mx) in enumerate(per_ch)
            )

            stats = (
                f"VAE config: class={getattr(vae.config, '_class_name', type(vae).__name__)}, "
                f"channels={getattr(vae.config, 'latent_channels', '?')}\n"
                f"VAE config shift / scale: {cfg_shift} / {cfg_scale}\n"
                f"Override shift / scale:   {override_shift} / {override_scale}\n"
                f"Effective shift / scale:  {shift} / {scale}\n"
                f"Normalization mode:       {self.norm_mode}\n"
                f"{bn_info}\n"
                f"Original size: {ow}x{oh}\n"
                f"Resized to bucket: {target_w}x{target_h}\n"
                f"Latent shape: {tuple(norm_latent.shape)}\n\n"
                f"RAW latent (post-encode, pre-normalize):\n"
                f"  mean={raw_mean:+.4f}  std={raw_std:.4f}  min={raw_min:+.3f}  max={raw_max:+.3f}\n\n"
                f"NORMALIZED latent (what training feeds to UNet):\n"
                f"  mean={norm_mean:+.4f}  std={norm_std:.4f}  min={norm_min:+.3f}  max={norm_max:+.3f}\n"
                f"  IDEAL: mean near 0, std near 1\n"
                f"  Per-channel:\n{per_ch_text}\n\n"
                f"{cache_compare_text}"
            )

            self.result.emit({
                "original": original,
                "resized": resized,
                "latent_grid": latent_grid,
                "roundtrip": roundtrip_pil,
                "cached_decoded": cached_decoded_pil,
                "cached_grid": cached_grid,
                "stats": stats,
                "target_size": (target_w, target_h),
            })

        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAE Latent Inspector")
        self.resize(1500, 1000)
        self.worker = None
        self._build()

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)

        # Inputs group
        inputs = QtWidgets.QGroupBox("Inputs")
        grid = QtWidgets.QGridLayout(inputs)

        self.model_path = QtWidgets.QLineEdit()
        model_btn = QtWidgets.QPushButton("Browse Model/VAE")
        model_btn.clicked.connect(lambda: self._pick_file(self.model_path,
            "Model files (*.safetensors *.ckpt *.pt);;All files (*)"))

        self.image_path = QtWidgets.QLineEdit()
        image_btn = QtWidgets.QPushButton("Browse Image")
        image_btn.clicked.connect(lambda: self._pick_file(self.image_path,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"))

        self.cache_lat = QtWidgets.QLineEdit()
        cache_lat_btn = QtWidgets.QPushButton("Browse _lat.pt")
        cache_lat_btn.clicked.connect(lambda: self._pick_file(self.cache_lat,
            "Cached latent (*_lat.pt);;All files (*)"))

        self.cache_te = QtWidgets.QLineEdit()
        cache_te_btn = QtWidgets.QPushButton("Browse _te.pt")
        cache_te_btn.clicked.connect(lambda: self._pick_file(self.cache_te,
            "Cached te (*_te.pt);;All files (*)"))

        self.bn_stats_path = QtWidgets.QLineEdit()
        self.bn_stats_path.setPlaceholderText("optional; defaults to model/VAE file")
        bn_stats_btn = QtWidgets.QPushButton("Browse BN Stats VAE")
        bn_stats_btn.clicked.connect(lambda: self._pick_file(self.bn_stats_path,
            "BN stats source (*.safetensors *.ckpt *.pt);;All files (*)"))

        grid.addWidget(QtWidgets.QLabel("Checkpoint or VAE"), 0, 0)
        grid.addWidget(self.model_path, 0, 1)
        grid.addWidget(model_btn, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Test image"), 1, 0)
        grid.addWidget(self.image_path, 1, 1)
        grid.addWidget(image_btn, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Cached _lat.pt (optional)"), 2, 0)
        grid.addWidget(self.cache_lat, 2, 1)
        grid.addWidget(cache_lat_btn, 2, 2)
        grid.addWidget(QtWidgets.QLabel("Cached _te.pt  (optional)"), 3, 0)
        grid.addWidget(self.cache_te, 3, 1)
        grid.addWidget(cache_te_btn, 3, 2)
        grid.addWidget(QtWidgets.QLabel("Flux BN stats source (optional)"), 4, 0)
        grid.addWidget(self.bn_stats_path, 4, 1)
        grid.addWidget(bn_stats_btn, 4, 2)

        root.addWidget(inputs)

        # Settings row
        settings = QtWidgets.QHBoxLayout()
        self.latent_channels = QtWidgets.QSpinBox()
        self.latent_channels.setRange(0, 128)
        self.latent_channels.setValue(0)
        self.latent_channels.setToolTip("0 = auto-detect from VAE file (recommended)")
        self.shift_override = QtWidgets.QLineEdit()
        self.shift_override.setPlaceholderText("auto (use VAE config)")
        self.scale_override = QtWidgets.QLineEdit()
        self.scale_override.setPlaceholderText("auto (use VAE config)")
        self.force_w = QtWidgets.QSpinBox()
        self.force_w.setRange(0, 4096)
        self.force_w.setValue(0)
        self.force_w.setToolTip("0 = auto via bucket logic")
        self.force_h = QtWidgets.QSpinBox()
        self.force_h.setRange(0, 4096)
        self.force_h.setValue(0)
        self.force_h.setToolTip("0 = auto via bucket logic")
        self.norm_mode = QtWidgets.QComboBox()
        self.norm_mode.addItem("Raw / no normalization", "raw")
        self.norm_mode.addItem("Scalar shift/scale", "scalar")
        self.norm_mode.addItem("Flux BN reduced 32ch", "bn32")
        self.norm_mode.addItem("Flux BN exact 128ch packed", "bn128")
        self.norm_mode.setCurrentIndex(2)
        self.norm_mode.setToolTip("For FLUX.2 VAE diagnostics, compare 32ch reduced vs exact 128ch packed.")
        self.run_btn = QtWidgets.QPushButton("Run Inspection")
        self.run_btn.clicked.connect(self._run)

        for label, w in [
            ("Channels", self.latent_channels),
            ("Shift override", self.shift_override),
            ("Scale override", self.scale_override),
            ("Force W", self.force_w),
            ("Force H", self.force_h),
            ("Norm", self.norm_mode),
        ]:
            settings.addWidget(QtWidgets.QLabel(label))
            settings.addWidget(w)
        settings.addStretch(1)
        settings.addWidget(self.run_btn)
        root.addLayout(settings)

        # Image rows
        row1 = QtWidgets.QHBoxLayout()
        self.original_slot = ImageSlot("Original")
        self.resized_slot = ImageSlot("Resized (what VAE encodes)")
        self.roundtrip_slot = ImageSlot("Round-trip decode (fresh)")
        self.cached_decoded_slot = ImageSlot("Round-trip decode (cached)")
        for s in [self.original_slot, self.resized_slot, self.roundtrip_slot, self.cached_decoded_slot]:
            row1.addWidget(s, 1)
        root.addLayout(row1, 1)

        row2 = QtWidgets.QHBoxLayout()
        self.latent_grid_slot = ImageSlot("Selected normalized latent (fresh)")
        self.cached_grid_slot = ImageSlot("Cached latent")
        for s in [self.latent_grid_slot, self.cached_grid_slot]:
            row2.addWidget(s, 1)
        root.addLayout(row2, 1)

        self.status = QtWidgets.QLabel("Pick a model and image, then Run Inspection.")
        root.addWidget(self.status)

        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(280)
        root.addWidget(self.stats)

    def _pick_file(self, target, filter_text):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", filter_text)
        if path:
            target.setText(path)

    def _run(self):
        if not Path(self.model_path.text()).exists():
            QtWidgets.QMessageBox.warning(self, "Missing input", "Choose a valid model/VAE file.")
            return
        if not Path(self.image_path.text()).exists():
            QtWidgets.QMessageBox.warning(self, "Missing input", "Choose a valid image file.")
            return

        force_res = (self.force_w.value(), self.force_h.value()) if self.force_w.value() > 0 and self.force_h.value() > 0 else None
        self.run_btn.setEnabled(False)
        self.worker = InspectionWorker(
            model_path=self.model_path.text(),
            image_path=self.image_path.text(),
            latent_channels=self.latent_channels.value(),
            shift_text=self.shift_override.text(),
            scale_text=self.scale_override.text(),
            cached_lat_path=self.cache_lat.text() or None,
            cached_te_path=self.cache_te.text() or None,
            force_resolution=force_res,
            norm_mode=self.norm_mode.currentData(),
            bn_stats_path=self.bn_stats_path.text() or None,
        )
        self.worker.status.connect(self.status.setText)
        self.worker.result.connect(self._show_result)
        self.worker.error.connect(self._show_error)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def _show_result(self, data):
        size = max(220, min(360, self.width() // 5))
        self.original_slot.set_image(data["original"], size)
        self.resized_slot.set_image(data["resized"], size)
        self.roundtrip_slot.set_image(data["roundtrip"], size)
        self.cached_decoded_slot.set_image(data.get("cached_decoded"), size)
        self.latent_grid_slot.set_image(data["latent_grid"], size)
        self.cached_grid_slot.set_image(data.get("cached_grid"), size)
        self.stats.setPlainText(data["stats"])
        self.status.setText(f"Done. Target bucket: {data['target_size']}")

    def _show_error(self, msg):
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Inspection failed", msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()