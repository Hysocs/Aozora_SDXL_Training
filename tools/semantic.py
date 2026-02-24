"""
semantic.py  —  Semantic loss weight map generator for SDXL trainer
Latent-Space Edition: Maps are computed directly from the encoded VAE latents.

All maps are computed at native latent resolution (image_size / 8).
Sub-8px pixel detail is invisible to the VAE, so there is no loss of useful
information by working at this scale — and it is far cheaper.
"""

import cv2
import numpy as np
import torch
from PIL import Image


def generate_latent_seperation_map(latent_np: np.ndarray) -> np.ndarray:
    """Derives saliency by measuring per-pixel deviation from the mean across latent channels."""
    deviation = np.abs(latent_np - latent_np.mean(axis=(0, 1), keepdims=True))
    sal = deviation.mean(axis=2)
    mx = sal.max()
    if mx > 1e-6:
        sal /= mx
    return cv2.GaussianBlur(sal.astype(np.float32), (3, 3), 0)


def generate_latent_detail_map(latent_np: np.ndarray) -> np.ndarray:
    """Measures edge magnitude directly across all latent channels."""
    num_channels = latent_np.shape[2]
    mag_total = np.zeros(latent_np.shape[:2], dtype=np.float32)
    for c in range(num_channels):
        ch = latent_np[:, :, c].astype(np.float64)
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag_total += np.sqrt(sx**2 + sy**2).astype(np.float32)
    mag_total /= num_channels
    mx = mag_total.max()
    if mx > 1e-6:
        mag_total /= mx
    return cv2.GaussianBlur(mag_total, (3, 3), 0)


def _pick_disk_radius(dim: int) -> int:
    """
    Pick a disk radius appropriate for the given image dimension.
    Keeps the neighbourhood proportional regardless of latent size.
      < 48px  -> radius 1
      < 80px  -> radius 2
      < 128px -> radius 3
      >= 128px -> radius 4
    """
    if dim < 48:
        return 1
    elif dim < 80:
        return 2
    elif dim < 128:
        return 3
    return 4


def generate_entropy_map(pil_image: Image.Image, target_w: int, target_h: int) -> np.ndarray:
    """Texture complexity via local entropy, computed at native latent resolution.

    The image is resized to (target_w, target_h) — exactly the latent spatial
    dims — before any processing.  This means:
      - One resize, to the exact size needed, done once.
      - Entropy is computed only on pixels that the VAE can actually represent.
      - Disk radii are chosen dynamically based on the shorter latent dimension
        so they stay proportional for non-square images.
      - The caller does NOT need to resize the output.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    pil_image = pil_image.resize((target_w, target_h), Image.Resampling.BILINEAR)

    from skimage.filters.rank import entropy as sk_entropy
    from skimage.morphology import disk

    np_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    lab    = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2LAB)
    l_ch   = lab[:, :, 0].astype(np.float32)
    a_ch   = lab[:, :, 1].astype(np.float32)

    l_u8 = cv2.normalize(l_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    a_u8 = cv2.normalize(a_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Dynamic radii based on the shorter latent dimension
    short_dim   = min(target_w, target_h)
    r_fine      = _pick_disk_radius(short_dim)
    r_coarse    = r_fine * 2 + 1   # e.g. 3->7, 2->5, 1->3

    ent_fine_l   = sk_entropy(l_u8, disk(r_fine)).astype(np.float32)
    ent_fine_a   = sk_entropy(a_u8, disk(r_fine)).astype(np.float32)
    ent_coarse_l = sk_entropy(l_u8, disk(r_coarse)).astype(np.float32)

    ent_combined = (ent_fine_l * 0.45 + ent_fine_a * 0.20 + ent_coarse_l * 0.35)

    mx = ent_combined.max()
    if mx > 1e-6:
        ent_combined /= mx

    sx       = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    sy       = cv2.Sobel(l_ch, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx**2 + sy**2).astype(np.float32)
    mx = edge_mag.max()
    if mx > 1e-6:
        edge_mag /= mx
    edge_bin = (edge_mag > 0.3).astype(np.uint8)
    dil_r    = max(3, r_fine * 2 - 1)   # also scale the dilation kernel
    edge_dil = cv2.dilate(edge_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_r, dil_r)))
    not_edge = 1.0 - edge_dil.astype(np.float32)

    result = ent_combined * not_edge
    mx = result.max()
    if mx > 1e-6:
        result /= mx

    blur_k = max(3, r_fine * 2 + 1)   # must be odd
    if blur_k % 2 == 0:
        blur_k += 1
    result = cv2.GaussianBlur(result, (blur_k, blur_k), 0)
    mx = result.max()
    if mx > 1e-6:
        result /= mx

    return result.astype(np.float32)


# Module-level flag to suppress repeated skimage import warnings
_entropy_import_ok = None


def generate_latent_semantic_map_batch(latents, images, sep_weight, detail_weight, entropy_weight, device, dtype):
    """
    Computes semantic maps in PIXEL space, then downsamples to latent resolution.
    This preserves fine detail that is lost when computing directly from latents.
    """
    batch_maps = []
    B, C, H, W = latents.shape  # latent dims

    for i in range(B):
        pil_img = images[i]
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Work at a higher intermediate resolution — 4x latent size
        # This is still cheaper than full res but preserves far more signal
        proc_w = W * 4
        proc_h = H * 4
        img_resized = pil_img.resize((proc_w, proc_h), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        combined = np.zeros((proc_h, proc_w), dtype=np.float32)

        # Detail map: Sobel on luminance in pixel space
        if detail_weight > 0.0:
            gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            detail_map = np.sqrt(sx**2 + sy**2).astype(np.float32)
            mx = detail_map.max()
            if mx > 1e-6:
                detail_map /= mx
            detail_map = cv2.GaussianBlur(detail_map, (3, 3), 0)
            combined += detail_map * detail_weight

        # Separation map: colour variance / saturation as proxy for subject vs background
        if sep_weight > 0.0:
            # Use HSV saturation — subjects tend to have more saturated colours
            # than plain backgrounds; works well for anime style
            img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            sat = hsv[:, :, 1] / 255.0
            # Also add local variance as a texture signal
            gray_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            local_var = cv2.blur(gray_f**2, (5, 5)) - cv2.blur(gray_f, (5, 5))**2
            local_var = np.clip(local_var * 10.0, 0.0, 1.0)
            sep_map = np.clip(sat * 0.6 + local_var * 0.4, 0.0, 1.0)
            sep_map = cv2.GaussianBlur(sep_map, (5, 5), 0)
            mx = sep_map.max()
            if mx > 1e-6:
                sep_map /= mx
            combined += sep_map * sep_weight

        # Entropy map: still valid in pixel space
        if entropy_weight > 0.0:
            try:
                entropy_np = generate_entropy_map(pil_img, target_w=proc_w, target_h=proc_h)
                combined += entropy_np * entropy_weight
            except Exception as e:
                print(f"WARNING: entropy map failed: {e}")

        # Normalize combined map
        mx = combined.max()
        if mx > 1e-6:
            combined /= mx
        combined = np.clip(combined, 0.0, 10.0)

        # Downsample from pixel space to latent space using area averaging
        # INTER_AREA is the correct interpolation for downsampling — it averages
        # rather than sampling, preserving the weight distribution properly
        combined_latent = cv2.resize(combined, (W, H), interpolation=cv2.INTER_AREA)

        batch_maps.append(torch.from_numpy(combined_latent).to(dtype))

    final_map_batch = torch.stack(batch_maps).to(device)
    final_map_batch = final_map_batch.unsqueeze(1).expand(-1, C, -1, -1)

    return final_map_batch