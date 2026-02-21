"""
Semantic Loss Visualizer v2
- True latent-space map computation (maps computed on decoded latents, not pixel image)
- Per-channel latent heatmaps
- Live weight sliders with instant feedback
- MSE vs Semantic loss comparison overlay
- Export educational grid
"""

import sys
import numpy as np
import os
import cv2
import gc
from PIL import Image
import torch
import torch.nn.functional as F

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QDoubleSpinBox, QFormLayout, QGroupBox, QCheckBox,
    QProgressBar, QMessageBox, QTabWidget, QComboBox, QSplitter, QFrame,
    QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────
#  COLOUR PALETTE
# ──────────────────────────────────────────────
DARK_BG   = "#0d0f14"
PANEL_BG  = "#141720"
BORDER    = "#252a38"
ACCENT    = "#7c6af7"
ACCENT2   = "#3ecfcf"
TEXT_PRI  = "#e8eaf0"
TEXT_SEC  = "#7a7f9a"
DANGER    = "#f74b6a"
SUCCESS   = "#3ef78e"

HEATMAP_COLORS = [
    (0.04, 0.04, 0.18),
    (0.00, 0.20, 0.60),
    (0.00, 0.70, 0.70),
    (0.90, 0.80, 0.00),
    (0.95, 0.15, 0.10),
]

# ──────────────────────────────────────────────
#  IMAGE UTILS
# ──────────────────────────────────────────────
def fix_alpha(img: Image.Image) -> Image.Image:
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


# ──────────────────────────────────────────────
#  PIXEL-SPACE MAP GENERATORS (unchanged logic)
# ──────────────────────────────────────────────
def generate_character_map(pil_image: Image.Image) -> np.ndarray:
    """
    Detects filled texture regions - hair masses, clothing areas, shaded skin.
    NOT lineart. Uses medium/large variance windows then suppresses pure edge pixels,
    so it lights up the interior of complex regions rather than their outlines.
    Complements the detail map which handles lines/edges.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    np_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    lab    = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2LAB)
    l_ch   = lab[:, :, 0].astype(np.float32)
    a_ch   = lab[:, :, 1].astype(np.float32)
    b_ch   = lab[:, :, 2].astype(np.float32)

    def local_variance(ch, ksize):
        mean  = cv2.blur(ch, (ksize, ksize))
        mean2 = cv2.blur(ch * ch, (ksize, ksize))
        return np.sqrt(np.maximum(mean2 - mean * mean, 0))

    # Medium windows catch filled texture regions (hair, clothing, shading)
    var_med_l   = local_variance(l_ch, 21)
    var_med_a   = local_variance(a_ch, 21)
    var_med_b   = local_variance(b_ch, 21)
    # Large window catches broad shaded/gradient regions
    var_large_l = local_variance(l_ch, 45)

    region_map = (var_med_l   * 0.45 +
                  var_med_a   * 0.20 +
                  var_med_b   * 0.20 +
                  var_large_l * 0.15)

    # Build edge mask from Sobel, then invert so lineart pixels get down-weighted
    # This keeps the character map focused on region interiors, not outlines
    sx       = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    sy       = cv2.Sobel(l_ch, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx**2 + sy**2).astype(np.float32)
    mx = edge_mag.max()
    if mx > 1e-6:
        edge_mag /= mx
    not_edge = 1.0 - np.clip(edge_mag * 2.0, 0, 1)

    combined = region_map * not_edge

    mx = combined.max()
    if mx > 1e-6:
        combined /= mx

    # Larger blur smooths into filled blobs rather than tight edges
    result = cv2.GaussianBlur(combined, (15, 15), 0)
    mx = result.max()
    if mx > 1e-6:
        result /= mx

    return result.astype(np.float32)


def generate_detail_map(pil_image: Image.Image) -> np.ndarray:
    """
    Detects lineart, edges, fine patterns, eyes - anything with rapid local change.
    Local variance (fine window) + Laplacian energy catches both texture and lines.
    Proven better than plain Sobel for fine detail in anime/illustrious style.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    np_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    lab    = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2LAB)
    l_ch   = lab[:, :, 0].astype(np.float32)
    a_ch   = lab[:, :, 1].astype(np.float32)
    b_ch   = lab[:, :, 2].astype(np.float32)

    def local_variance(ch, ksize):
        mean  = cv2.blur(ch, (ksize, ksize))
        mean2 = cv2.blur(ch * ch, (ksize, ksize))
        return np.sqrt(np.maximum(mean2 - mean * mean, 0))

    # Fine window - catches eyes, small patterns, fine detail
    var_fine_l = local_variance(l_ch, 7)
    var_fine_a = local_variance(a_ch, 7)
    var_fine_b = local_variance(b_ch, 7)
    var_med_l  = local_variance(l_ch, 21)

    var_combined = (var_fine_l * 0.5 +
                    var_fine_a * 0.2 +
                    var_fine_b * 0.2 +
                    var_med_l  * 0.1)

    # Laplacian fires on both sides of lines - more sensitive than Sobel for fine detail
    # Cast to float64 - required for CV_64F dest on AVX2 builds
    lap        = cv2.Laplacian(l_ch.astype(np.float64), cv2.CV_64F, ksize=3)
    lap_energy = np.abs(lap).astype(np.float32)

    l_blur      = cv2.GaussianBlur(l_ch, (3, 3), 0)
    lap2        = cv2.Laplacian(l_blur.astype(np.float64), cv2.CV_64F, ksize=5)
    lap_energy2 = np.abs(lap2).astype(np.float32)

    combined = var_combined * 0.6 + lap_energy * 0.3 + lap_energy2 * 0.1

    mx = combined.max()
    if mx > 1e-6:
        combined /= mx

    # Tight blur - kills noise but preserves fine structure
    result = cv2.GaussianBlur(combined, (5, 5), 0)
    mx = result.max()
    if mx > 1e-6:
        result /= mx

    return result.astype(np.float32)


def generate_entropy_map(pil_image: Image.Image) -> np.ndarray:
    """
    Texture detection using local entropy + dilated-edge suppression.
    Entropy measures information complexity per neighborhood:
    high in textured regions (hair, clothing patterns, eye irises),
    low in flat fills AND suppressed on thin lineart via edge mask.
    Requires scikit-image: pip install scikit-image
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    from skimage.filters.rank import entropy as sk_entropy
    from skimage.morphology import disk

    np_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    lab    = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2LAB)
    l_ch   = lab[:, :, 0].astype(np.float32)
    a_ch   = lab[:, :, 1].astype(np.float32)

    l_u8 = cv2.normalize(l_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    a_u8 = cv2.normalize(a_ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Small disk = fine texture (eye iris, small patterns)
    # Large disk = coarse texture (hair masses, cloth folds)
    ent_fine_l   = sk_entropy(l_u8, disk(5)).astype(np.float32)
    ent_fine_a   = sk_entropy(a_u8, disk(5)).astype(np.float32)
    ent_coarse_l = sk_entropy(l_u8, disk(15)).astype(np.float32)

    ent_combined = (ent_fine_l   * 0.45 +
                    ent_fine_a   * 0.20 +
                    ent_coarse_l * 0.35)

    mx = ent_combined.max()
    if mx > 1e-6:
        ent_combined /= mx

    # Suppress pure lineart: dilate Sobel edges into fat masks then invert
    sx       = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    sy       = cv2.Sobel(l_ch, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx**2 + sy**2).astype(np.float32)
    mx = edge_mag.max()
    if mx > 1e-6:
        edge_mag /= mx
    edge_bin = (edge_mag > 0.3).astype(np.uint8)
    edge_dil = cv2.dilate(edge_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    not_edge = 1.0 - edge_dil.astype(np.float32)

    result = ent_combined * not_edge
    mx = result.max()
    if mx > 1e-6:
        result /= mx

    result = cv2.GaussianBlur(result, (11, 11), 0)
    mx = result.max()
    if mx > 1e-6:
        result /= mx

    return result.astype(np.float32)


# ──────────────────────────────────────────────
#  NEW: LATENT-SPACE MAP GENERATORS
#  Operates on the decoded-then-re-encoded latent
#  so the map reflects what the model actually sees
# ──────────────────────────────────────────────
def generate_latent_character_map(latent_np: np.ndarray) -> np.ndarray:
    """
    latent_np: (H, W, C) float32, values roughly in [-4, 4]
    Returns a (H, W) saliency map in latent space.
    Strategy: per-pixel deviation from mean in each channel, summed.
    """
    deviation = np.abs(latent_np - latent_np.mean(axis=(0, 1), keepdims=True))
    sal = deviation.mean(axis=2)          # (H, W)
    mx = sal.max()
    if mx > 1e-6:
        sal /= mx
    return cv2.GaussianBlur(sal.astype(np.float32), (3, 3), 0)


def generate_latent_detail_map(latent_np: np.ndarray) -> np.ndarray:
    """
    Edge magnitude across all latent channels.
    """
    mag_total = np.zeros(latent_np.shape[:2], dtype=np.float32)
    for c in range(latent_np.shape[2]):
        ch = latent_np[:, :, c].astype(np.float64)
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag_total += np.sqrt(sx**2 + sy**2).astype(np.float32)
    mag_total /= latent_np.shape[2]
    mx = mag_total.max()
    if mx > 1e-6:
        mag_total /= mx
    return cv2.GaussianBlur(mag_total, (3, 3), 0)


def encode_image_to_latent(pil_image: Image.Image, vae, device, dtype):
    """
    Encode a PIL image to latent space using the VAE.
    Returns (latent_tensor [1,C,H,W], latent_np [H,W,C])
    """
    w, h = pil_image.size
    w = (w // 64) * 64
    h = (h // 64) * 64
    img = pil_image.resize((w, h), Image.Resampling.LANCZOS)
    t = torch.from_numpy(np.array(img)).to(dtype=dtype, device=device)
    t = (t / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(t).latent_dist.sample()
    latent_np = latent[0].permute(1, 2, 0).float().cpu().numpy()  # (H, W, C)
    return latent, latent_np


def decode_latent(latent_tensor, vae, device, dtype):
    """Decode latent back to pixel image."""
    with torch.no_grad():
        decoded = vae.decode(latent_tensor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    np_img  = decoded[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((np_img * 255).astype(np.uint8))


# ──────────────────────────────────────────────
#  VAE LOADER THREAD
# ──────────────────────────────────────────────
class VAELoaderThread(QThread):
    finished = pyqtSignal(object, str)
    error    = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            from diffusers import AutoencoderKL
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            try:
                vae = AutoencoderKL.from_single_file(self.path, torch_dtype=dtype)
            except Exception as e:
                if "latent_channels" in str(e) or "size mismatch" in str(e):
                    vae = AutoencoderKL.from_single_file(
                        self.path, torch_dtype=dtype,
                        latent_channels=32, ignore_mismatched_sizes=True)
                else:
                    raise
            vae.enable_tiling()
            vae.enable_slicing()
            vae.eval()
            if torch.cuda.is_available():
                vae = vae.to("cuda")
            n_ch = vae.config.latent_channels
            self.finished.emit(vae, f"VAE loaded – {n_ch}ch ({dtype})")
        except Exception as e:
            self.error.emit(str(e))


# ──────────────────────────────────────────────
#  COMPUTE THREAD
# ──────────────────────────────────────────────
class ComputeThread(QThread):
    """
    Computes everything needed for display:
      - pixel-space maps
      - latent encode/decode (if VAE loaded)
      - latent-space maps
      - per-channel latent arrays for channel inspector
    """
    result_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.params = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    def set_params(self, p: dict):
        self.params = p

    def run(self):
        p = self.params
        if p is None or p.get("image") is None:
            return

        img: Image.Image = p["image"]
        vae              = p.get("vae")
        char_w: float    = p["char_w"]
        detail_w: float  = p["detail_w"]
        entropy_w: float = p.get("entropy_w", 0.0)
        map_source: str  = p["map_source"]   # "pixel" | "latent"

        out = {}

        # ── pixel maps (always computed, cheap)
        char_px   = generate_character_map(img)
        detail_px = generate_detail_map(img)
        out["char_px"]    = char_px
        out["detail_px"]  = detail_px
        out["pixel_image"] = img

        # ── entropy map (only if weight > 0 to avoid skimage import cost every frame)
        entropy_px = None
        if entropy_w > 0.0:
            try:
                entropy_px = generate_entropy_map(img)
                out["entropy_px"] = entropy_px
            except ImportError:
                out["entropy_error"] = "scikit-image not installed (pip install scikit-image)"
            except Exception as e:
                out["entropy_error"] = str(e)

        # ── VAE encode/decode
        if vae is not None:
            try:
                latent_tensor, latent_np = encode_image_to_latent(
                    img, vae, self.device, self.dtype)
                decoded_img = decode_latent(latent_tensor, vae, self.device, self.dtype)

                out["latent_np"]    = latent_np      # (H, W, C)
                out["decoded_img"]  = decoded_img
                out["latent_shape"] = latent_np.shape

                # latent maps
                char_lat   = generate_latent_character_map(latent_np)
                detail_lat = generate_latent_detail_map(latent_np)
                out["char_lat"]   = char_lat
                out["detail_lat"] = detail_lat

                # per-channel maps for inspector (normalised individually)
                ch_maps = []
                for c in range(latent_np.shape[2]):
                    ch = latent_np[:, :, c]
                    norm = (ch - ch.min()) / (ch.ptp() + 1e-8)
                    ch_maps.append(norm.astype(np.float32))
                out["channel_maps"] = ch_maps

                del latent_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                out["vae_error"] = str(e)

        # ── choose which maps to use for the semantic weight
        if map_source == "latent" and "char_lat" in out:
            char_map   = out["char_lat"]
            detail_map = out["detail_lat"]
            bg_for_map = out.get("decoded_img", img)
            bg_w, bg_h = bg_for_map.size
        else:
            char_map   = char_px
            detail_map = detail_px
            bg_for_map = img
            bg_w, bg_h = img.size

        # final combined map - entropy added if computed
        combined = char_map * char_w + detail_map * detail_w
        if entropy_px is not None and entropy_w > 0.0:
            # resize entropy map to match char/detail map if needed
            if entropy_px.shape != char_map.shape:
                ep = Image.fromarray(entropy_px, mode='F').resize(
                    (char_map.shape[1], char_map.shape[0]), Image.Resampling.BILINEAR)
                entropy_px = np.array(ep)
            combined = combined + entropy_px * entropy_w
        combined = np.clip(combined, 0.0, 10.0)

        # resize map to bg resolution for display
        map_pil = Image.fromarray(combined, mode='F')
        if map_pil.size != (bg_w, bg_h):
            map_pil = map_pil.resize((bg_w, bg_h), Image.Resampling.BILINEAR)
        combined_display = np.array(map_pil)

        out["combined_map"] = combined_display
        out["bg_image"]     = bg_for_map
        out["map_source"]   = map_source
        out["char_w"]       = char_w
        out["detail_w"]     = detail_w
        out["entropy_w"]    = entropy_w

        gc.collect()
        self.result_ready.emit(out)


# ──────────────────────────────────────────────
#  MAIN WINDOW
# ──────────────────────────────────────────────
class SemanticVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantic Loss Visualizer  v2  —  Latent-Space Edition")
        self.setGeometry(80, 80, 1600, 960)
        self._apply_dark_theme()

        # state
        self.original_image  = None
        self.vae_model       = None
        self.last_result     = {}
        self._pending        = False
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._dispatch_compute)

        self.compute_thread = ComputeThread()
        self.compute_thread.result_ready.connect(self._on_result)
        self.vae_thread = None

        self._build_ui()
        self._build_cmap()

    # ── theme ──────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QWidget {{ background: {DARK_BG}; color: {TEXT_PRI}; font-family: 'Consolas', monospace; font-size: 12px; }}
            QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 8px; padding-top: 8px;
                         color: {TEXT_SEC}; font-size: 11px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {ACCENT}; }}
            QPushButton {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px;
                           color: {TEXT_PRI}; padding: 6px 12px; }}
            QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
            QPushButton:pressed {{ background: {ACCENT}; color: white; }}
            QPushButton:disabled {{ color: {TEXT_SEC}; }}
            QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px;
                                          margin: -5px 0; border-radius: 7px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
            QDoubleSpinBox, QSpinBox, QComboBox {{ background: {PANEL_BG}; border: 1px solid {BORDER};
                border-radius: 4px; color: {TEXT_PRI}; padding: 2px 6px; }}
            QLabel {{ color: {TEXT_PRI}; }}
            QCheckBox {{ color: {TEXT_PRI}; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER}; border-radius: 3px; }}
            QCheckBox::indicator:checked {{ background: {ACCENT}; }}
            QTabWidget::pane {{ border: 1px solid {BORDER}; }}
            QTabBar::tab {{ background: {PANEL_BG}; border: 1px solid {BORDER}; color: {TEXT_SEC};
                            padding: 6px 14px; border-radius: 4px 4px 0 0; }}
            QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
            QSplitter::handle {{ background: {BORDER}; }}
        """)

    def _build_cmap(self):
        self.heatmap_cmap = LinearSegmentedColormap.from_list("loss_heat", HEATMAP_COLORS)

    # ── UI layout ──────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # LEFT panel
        left = QWidget()
        left.setFixedWidth(300)
        left.setStyleSheet(f"background:{PANEL_BG}; border-radius:8px;")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

        # VAE
        vae_grp = QGroupBox("VAE / Model")
        vg = QVBoxLayout(vae_grp)
        self.btn_vae = QPushButton("⬆  Load VAE (.safetensors / .pt)")
        self.btn_vae.setStyleSheet(f"background:{ACCENT}; color:white; font-weight:bold; padding:8px;")
        self.btn_vae.clicked.connect(self._load_vae)
        self.lbl_vae = QLabel("No VAE loaded")
        self.lbl_vae.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.lbl_vae.setWordWrap(True)
        vg.addWidget(self.btn_vae)
        vg.addWidget(self.lbl_vae)
        lv.addWidget(vae_grp)

        # Image
        img_grp = QGroupBox("Image")
        ig = QVBoxLayout(img_grp)
        self.btn_img = QPushButton("🖼  Load Image")
        self.btn_img.clicked.connect(self._load_image)
        self.lbl_img = QLabel("No image loaded")
        self.lbl_img.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.lbl_img.setWordWrap(True)
        ig.addWidget(self.btn_img)
        ig.addWidget(self.lbl_img)
        lv.addWidget(img_grp)

        # Map source
        src_grp = QGroupBox("Map Source")
        sg = QVBoxLayout(src_grp)
        self.combo_src = QComboBox()
        self.combo_src.addItems(["Pixel Space (fast)", "Latent Space (requires VAE)"])
        self.combo_src.currentIndexChanged.connect(self._schedule)
        self.lbl_src_info = QLabel("Pixel: maps computed on original image\nLatent: maps computed on VAE latents")
        self.lbl_src_info.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.lbl_src_info.setWordWrap(True)
        sg.addWidget(self.combo_src)
        sg.addWidget(self.lbl_src_info)
        lv.addWidget(src_grp)

        # Weights
        w_grp = QGroupBox("Loss Weights")
        wf = QFormLayout(w_grp)
        self.slider_char, self.spin_char = self._make_slider(0, 5, 1.0)
        self.slider_det,  self.spin_det  = self._make_slider(0, 5, 1.0)
        self.slider_ent,  self.spin_ent  = self._make_slider(0, 5, 0.0)
        wf.addRow("Character:", self.spin_char)
        wf.addRow(self.slider_char)
        wf.addRow("Detail:", self.spin_det)
        wf.addRow(self.slider_det)
        wf.addRow("Entropy:", self.spin_ent)
        wf.addRow(self.slider_ent)
        self.lbl_entropy_note = QLabel("Entropy: texture complexity map\n(requires scikit-image)")
        self.lbl_entropy_note.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        wf.addRow(self.lbl_entropy_note)
        lv.addWidget(w_grp)

        # Display options
        disp_grp = QGroupBox("Display")
        df = QVBoxLayout(disp_grp)
        self.chk_bg     = QCheckBox("Show background image")
        self.chk_bg.setChecked(True)
        self.chk_auto   = QCheckBox("Auto-scale intensity")
        self.chk_smooth = QCheckBox("Smooth heatmap")
        for w in (self.chk_bg, self.chk_auto, self.chk_smooth):
            w.toggled.connect(self._rerender)
            df.addWidget(w)
        lv.addWidget(disp_grp)

        # Channel inspector (latent)
        ch_grp = QGroupBox("Latent Channel Inspector")
        chv = QVBoxLayout(ch_grp)
        self.lbl_ch = QLabel("Load VAE + image to inspect channels")
        self.lbl_ch.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.lbl_ch.setWordWrap(True)
        self.spin_ch = QSpinBox()
        self.spin_ch.setRange(0, 3)
        self.spin_ch.setPrefix("Channel: ")
        self.spin_ch.valueChanged.connect(self._rerender)
        self.chk_show_ch = QCheckBox("Show individual channel map")
        self.chk_show_ch.toggled.connect(self._rerender)
        chv.addWidget(self.lbl_ch)
        chv.addWidget(self.spin_ch)
        chv.addWidget(self.chk_show_ch)
        lv.addWidget(ch_grp)

        # Stats
        self.lbl_stats = QLabel("—")
        self.lbl_stats.setStyleSheet(f"color:{ACCENT2}; font-size:10px; font-family:monospace;")
        self.lbl_stats.setWordWrap(True)
        lv.addWidget(self.lbl_stats)

        # Export
        self.btn_export = QPushButton("📊  Export Educational Grid")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export)
        lv.addWidget(self.btn_export)

        self.btn_vram = QPushButton("🗑  Clear VRAM")
        self.btn_vram.clicked.connect(self._clear_vram)
        lv.addWidget(self.btn_vram)

        lv.addStretch()
        root.addWidget(left)

        # RIGHT: tab display
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"background:{PANEL_BG};")

        # Tab 1: Main heatmap
        self.fig_main = Figure(figsize=(9, 7), dpi=100, facecolor=DARK_BG)
        self.canvas_main = FigureCanvas(self.fig_main)
        self.canvas_main.setStyleSheet(f"background:{DARK_BG};")
        self.ax_main = self.fig_main.add_subplot(111)
        self.ax_main.set_facecolor(DARK_BG)
        self.cbar_main = None
        self.tabs.addTab(self.canvas_main, "Loss Heatmap")

        # Tab 2: Component maps
        self.fig_comp = Figure(figsize=(9, 7), dpi=100, facecolor=DARK_BG)
        self.canvas_comp = FigureCanvas(self.fig_comp)
        self.canvas_comp.setStyleSheet(f"background:{DARK_BG};")
        self.tabs.addTab(self.canvas_comp, "Component Maps")

        # Tab 3: Latent channels
        self.fig_ch = Figure(figsize=(9, 7), dpi=100, facecolor=DARK_BG)
        self.canvas_ch = FigureCanvas(self.fig_ch)
        self.canvas_ch.setStyleSheet(f"background:{DARK_BG};")
        self.tabs.addTab(self.canvas_ch, "Latent Channels")

        # Tab 4: Side-by-side comparison
        self.fig_cmp = Figure(figsize=(9, 7), dpi=100, facecolor=DARK_BG)
        self.canvas_cmp = FigureCanvas(self.fig_cmp)
        self.canvas_cmp.setStyleSheet(f"background:{DARK_BG};")
        self.tabs.addTab(self.canvas_cmp, "MSE vs Semantic")

        root.addWidget(self.tabs, 1)

    # ── helpers ───────────────────────────────
    def _make_slider(self, lo, hi, default):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(int(lo * 100), int(hi * 100))
        sl.setValue(int(default * 100))
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(0.05)
        sp.setValue(default)
        sp.setDecimals(2)
        sl.valueChanged.connect(lambda v: sp.setValue(v / 100))
        sp.valueChanged.connect(lambda v: sl.setValue(int(v * 100)))
        sp.valueChanged.connect(self._schedule)
        return sl, sp

    def _map_source(self):
        return "latent" if self.combo_src.currentIndex() == 1 else "pixel"

    # ── events ────────────────────────────────
    def _load_vae(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select VAE / Model", "", "Model files (*.safetensors *.pt *.ckpt)")
        if not path:
            return
        self.btn_vae.setEnabled(False)
        self.lbl_vae.setText("Loading…")
        self.vae_thread = VAELoaderThread(path)
        self.vae_thread.finished.connect(self._on_vae_ok)
        self.vae_thread.error.connect(self._on_vae_err)
        self.vae_thread.start()

    def _on_vae_ok(self, vae, msg):
        self.vae_model = vae
        self.btn_vae.setEnabled(True)
        self.lbl_vae.setText(f"✔ {msg}")
        self.lbl_vae.setStyleSheet(f"color:{SUCCESS}; font-size:10px;")
        self._schedule()

    def _on_vae_err(self, err):
        self.btn_vae.setEnabled(True)
        self.lbl_vae.setText(f"✘ {err}")
        self.lbl_vae.setStyleSheet(f"color:{DANGER}; font-size:10px;")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        try:
            img = fix_alpha(Image.open(path))
            self.original_image = img
            fname = os.path.basename(path)
            self.lbl_img.setText(f"{fname}  ({img.width}×{img.height})")
            self.btn_export.setEnabled(True)
            self._schedule()
        except Exception as e:
            self.lbl_img.setText(f"Error: {e}")

    def _schedule(self):
        """Debounce slider spam."""
        self._debounce_timer.start(120)

    def _dispatch_compute(self):
        if self.compute_thread.isRunning():
            self._pending = True
            return
        self._pending = False
        if self.original_image is None:
            return
        params = {
            "image":    self.original_image,
            "vae":      self.vae_model,
            "char_w":   self.spin_char.value(),
            "detail_w": self.spin_det.value(),
            "entropy_w": self.spin_ent.value(),
            "map_source": self._map_source(),
        }
        self.compute_thread.set_params(params)
        self.compute_thread.start()

    def _on_result(self, result: dict):
        self.last_result = result
        if result.get("vae_error"):
            self.lbl_vae.setText(f"VAE error: {result['vae_error']}")
        if result.get("entropy_error"):
            self.lbl_entropy_note.setText(f"⚠ {result['entropy_error']}")
            self.lbl_entropy_note.setStyleSheet(f"color:{DANGER}; font-size:10px;")
        elif result.get("entropy_px") is not None:
            self.lbl_entropy_note.setText("Entropy map: ✔ computed")
            self.lbl_entropy_note.setStyleSheet(f"color:{SUCCESS}; font-size:10px;")
        if "latent_shape" in result:
            sh = result["latent_shape"]
            self.lbl_ch.setText(f"Latent: {sh[0]}×{sh[1]}  {sh[2]}ch")
            self.spin_ch.setRange(0, sh[2] - 1)
        self._rerender()
        if self._pending:
            self._dispatch_compute()

    def _clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── rendering ─────────────────────────────
    def _rerender(self):
        r = self.last_result
        if not r:
            return
        self._render_main(r)
        self._render_components(r)
        self._render_channels(r)
        self._render_comparison(r)

    def _cmap_norm(self, data, v_max=None):
        if v_max is None:
            v_max = max(0.1, data.max()) if self.chk_auto.isChecked() else 3.0
        return data, v_max

    def _ax_style(self, ax, title=""):
        ax.axis("off")
        ax.set_facecolor(DARK_BG)
        if title:
            ax.set_title(title, color=TEXT_SEC, fontsize=9, pad=4)

    def _render_main(self, r):
        fig = self.fig_main
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_BG)

        bg  = r.get("bg_image", r.get("pixel_image"))
        wm  = r.get("combined_map")
        if bg is None or wm is None:
            self.canvas_main.draw()
            return

        # override with single channel if requested
        if self.chk_show_ch.isChecked() and "channel_maps" in r:
            ch_idx = self.spin_ch.value()
            ch_idx = min(ch_idx, len(r["channel_maps"]) - 1)
            wm_raw = r["channel_maps"][ch_idx]
            # resize to bg size
            bg_w, bg_h = bg.size
            wm_pil = Image.fromarray(wm_raw, mode='F').resize(
                (bg_w, bg_h), Image.Resampling.BILINEAR if self.chk_smooth.isChecked()
                              else Image.Resampling.NEAREST)
            wm = np.array(wm_pil)
            title_suffix = f"  [Ch {ch_idx} raw latent]"
        else:
            title_suffix = f"  [{r['map_source']} space]"

        if self.chk_bg.isChecked():
            ax.imshow(np.array(bg))
            alpha = 0.60
        else:
            ax.imshow(np.zeros_like(np.array(bg)))
            alpha = 1.0

        v_max = max(0.1, wm.max()) if self.chk_auto.isChecked() else 3.0
        hm = ax.imshow(wm, cmap=self.heatmap_cmap, alpha=alpha, vmin=0.0, vmax=v_max)

        cbar = fig.colorbar(hm, ax=ax, fraction=0.035, pad=0.01)
        cbar.set_label("Loss weight  (1.0 + map)", color=TEXT_SEC, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=TEXT_SEC)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_SEC)

        avg = wm.mean()
        mx  = wm.max()
        ax.set_title(
            f"Semantic loss weight map{title_suffix}   avg={avg:.3f}  max={mx:.3f}",
            color=TEXT_PRI, fontsize=10, pad=6)
        ax.axis("off")

        self.lbl_stats.setText(
            f"avg: {avg:.4f}\nmax: {mx:.4f}\ncov>1.0: {(wm > 1.0).mean()*100:.1f}%")

        fig.tight_layout()
        self.canvas_main.draw()

    def _render_components(self, r):
        fig = self.fig_comp
        fig.clear()
        fig.patch.set_facecolor(DARK_BG)

        src  = r.get("map_source", "pixel")
        if src == "latent" and "char_lat" in r:
            char_m   = r["char_lat"]
            detail_m = r["detail_lat"]
            label_c  = "Character (latent)"
            label_d  = "Detail (latent)"
        else:
            char_m   = r.get("char_px")
            detail_m = r.get("detail_px")
            label_c  = "Character (pixel)"
            label_d  = "Detail (pixel)"

        if char_m is None:
            self.canvas_comp.draw()
            return

        bg  = r.get("bg_image", r.get("pixel_image"))
        cw  = r["char_w"]
        dw  = r["detail_w"]
        ew  = r.get("entropy_w", 0.0)
        ent = r.get("entropy_px")

        n_panels = 4 if (ent is not None and ew > 0.0) else 3
        axes = fig.subplots(1, n_panels)
        if n_panels == 1:
            axes = [axes]
        for ax in axes:
            ax.set_facecolor(DARK_BG)
            ax.axis("off")

        axes[0].imshow(np.array(bg))
        axes[0].set_title("Source Image", color=TEXT_PRI, fontsize=9)

        axes[1].imshow(char_m, cmap=self.heatmap_cmap, vmin=0, vmax=1)
        axes[1].set_title(f"{label_c}  ×{cw:.2f}", color=TEXT_PRI, fontsize=9)

        axes[2].imshow(detail_m, cmap=self.heatmap_cmap, vmin=0, vmax=1)
        axes[2].set_title(f"{label_d}  ×{dw:.2f}", color=TEXT_PRI, fontsize=9)

        if ent is not None and ew > 0.0:
            axes[3].imshow(ent, cmap=self.heatmap_cmap, vmin=0, vmax=1)
            axes[3].set_title(f"Entropy (texture)  ×{ew:.2f}", color=TEXT_PRI, fontsize=9)

        fig.tight_layout(pad=0.5)
        self.canvas_comp.draw()

    def _render_channels(self, r):
        fig = self.fig_ch
        fig.clear()
        fig.patch.set_facecolor(DARK_BG)

        ch_maps = r.get("channel_maps")
        if not ch_maps:
            ax = fig.add_subplot(111)
            ax.set_facecolor(DARK_BG)
            ax.text(0.5, 0.5, "Load a VAE to inspect latent channels",
                    ha='center', va='center', color=TEXT_SEC, fontsize=12,
                    transform=ax.transAxes)
            ax.axis("off")
            self.canvas_ch.draw()
            return

        n = len(ch_maps)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        axes = fig.subplots(rows, cols, squeeze=False)

        for i, ch_map in enumerate(ch_maps):
            row, col = divmod(i, cols)
            ax = axes[row][col]
            ax.set_facecolor(DARK_BG)
            ax.imshow(ch_map, cmap='magma', vmin=0, vmax=1)
            ax.set_title(f"Ch {i}", color=TEXT_SEC, fontsize=8)
            ax.axis("off")

        # hide unused axes
        for i in range(n, rows * cols):
            row, col = divmod(i, cols)
            axes[row][col].set_visible(False)

        sh = r.get("latent_shape", (0, 0, n))
        fig.suptitle(
            f"Latent channels  {sh[0]}×{sh[1]}  ({n} channels)",
            color=TEXT_PRI, fontsize=10)
        fig.tight_layout(pad=0.5)
        self.canvas_ch.draw()

    def _render_comparison(self, r):
        """Side-by-side: plain MSE weight (uniform 1.0) vs semantic weight."""
        fig = self.fig_cmp
        fig.clear()
        fig.patch.set_facecolor(DARK_BG)

        wm  = r.get("combined_map")
        bg  = r.get("bg_image", r.get("pixel_image"))
        if wm is None or bg is None:
            self.canvas_cmp.draw()
            return

        uniform = np.ones_like(wm) * 1.0
        semantic = 1.0 + wm            # actual multiplier applied in training

        v_max = max(0.1, semantic.max()) if self.chk_auto.isChecked() else 3.5

        axes = fig.subplots(1, 3)
        titles = ["MSE (uniform 1.0×)", "Semantic weight map", "Difference (gain)"]
        data   = [uniform, semantic, semantic - uniform]
        vmins  = [0.9, 0.9, 0.0]
        vmaxs  = [1.1, v_max, v_max - 1.0]
        cmaps  = ['gray', self.heatmap_cmap, 'plasma']

        for ax, title, d, vmin, vmax, cmap in zip(axes, titles, data, vmins, vmaxs, cmaps):
            ax.set_facecolor(DARK_BG)
            if self.chk_bg.isChecked():
                ax.imshow(np.array(bg))
                hm = ax.imshow(d, cmap=cmap, alpha=0.65, vmin=vmin, vmax=vmax)
            else:
                hm = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = fig.colorbar(hm, ax=ax, fraction=0.04, pad=0.01)
            cb.ax.yaxis.set_tick_params(color=TEXT_SEC, labelsize=7)
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC)
            ax.set_title(title, color=TEXT_PRI, fontsize=9)
            ax.axis("off")

        fig.suptitle("MSE vs. Semantic Loss — weight comparison", color=TEXT_PRI, fontsize=10)
        fig.tight_layout(pad=0.5)
        self.canvas_cmp.draw()

    # ── export ────────────────────────────────
    def _export(self):
        r = self.last_result
        if not r:
            QMessageBox.warning(self, "Export", "Nothing to export yet.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Educational Grid", "semantic_loss_grid.png", "PNG (*.png)")
        if not save_path:
            return

        bg  = r.get("bg_image", r.get("pixel_image"))
        wm  = r.get("combined_map")
        if bg is None or wm is None:
            return

        src   = r.get("map_source", "pixel")
        char  = r["char_lat"] if (src == "latent" and "char_lat" in r) else r.get("char_px")
        detail = r["detail_lat"] if (src == "latent" and "detail_lat" in r) else r.get("detail_px")

        fig = plt.figure(figsize=(18, 10), facecolor='white')
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.05)

        def show(ax, data, title, cmap='viridis', vmin=None, vmax=None):
            if isinstance(data, Image.Image):
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10, pad=4)
            ax.axis("off")

        show(fig.add_subplot(gs[0, 0]), bg,     "Source / Decoded")
        show(fig.add_subplot(gs[0, 1]), char,   f"Character map ({src})", 'plasma', 0, 1)
        show(fig.add_subplot(gs[0, 2]), detail, f"Detail map ({src})",    'plasma', 0, 1)
        show(fig.add_subplot(gs[0, 3]), wm,     "Combined map",          'inferno', 0, wm.max())

        sem = 1.0 + wm
        show(fig.add_subplot(gs[1, 0]), np.ones_like(wm), "MSE (1.0× uniform)", 'gray', 0.8, 1.2)
        show(fig.add_subplot(gs[1, 1]), sem,              f"Semantic weight", self.heatmap_cmap, 1.0, sem.max())
        show(fig.add_subplot(gs[1, 2]), sem - 1.0,        "Gain over MSE",   'plasma', 0, (sem-1).max())

        # channel grid (up to 4)
        if "channel_maps" in r:
            ax_ch = fig.add_subplot(gs[1, 3])
            n = min(4, len(r["channel_maps"]))
            ch_stack = np.hstack([r["channel_maps"][i] for i in range(n)])
            ax_ch.imshow(ch_stack, cmap='magma', vmin=0, vmax=1)
            ax_ch.set_title("Latent channels 0-3", fontsize=10)
            ax_ch.axis("off")

        fig.suptitle(
            f"Semantic Loss Analysis  |  source: {src}  |  "
            f"char×{r['char_w']:.2f}  detail×{r['detail_w']:.2f}",
            fontsize=13, fontweight='bold')

        plt.savefig(save_path, dpi=180, facecolor='white', bbox_inches='tight')
        plt.close()
        QMessageBox.information(self, "Saved", f"Grid saved to:\n{save_path}")


# ──────────────────────────────────────────────
#  ENTRY
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas", 10))
    w = SemanticVisualizer()
    w.show()
    sys.exit(app.exec())