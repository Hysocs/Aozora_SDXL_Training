import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps
from PyQt6 import QtCore, QtGui, QtWidgets
from torchvision import transforms


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

STYLE = """
QWidget { background: #15171b; color: #e7e9ee; font-family: Segoe UI, sans-serif; font-size: 13px; }
QTabWidget::pane, QGroupBox { border: 1px solid #353b45; border-radius: 6px; }
QGroupBox { margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #78c2ad; font-weight: 600; }
QPushButton { background: #242a31; border: 1px solid #3d4651; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { border-color: #78c2ad; color: #78c2ad; }
QPushButton:disabled { color: #69717d; border-color: #2b3037; }
QLineEdit, QSpinBox, QDoubleSpinBox { background: #101216; border: 1px solid #353b45; border-radius: 5px; padding: 6px; }
QPlainTextEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; font-family: Consolas, monospace; }
QProgressBar { border: 1px solid #353b45; border-radius: 5px; text-align: center; background: #101216; }
QProgressBar::chunk { background: #78c2ad; }
QLabel#ImagePanel { background: #050607; border: 1px solid #353b45; border-radius: 4px; }
QLabel#TitleLabel { color: #aeb7c2; font-weight: 600; }
"""


def load_config(path):
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def config_model_path(config):
    vae_path = str(config.get("VAE_PATH", "") or "").strip()
    if vae_path and Path(vae_path).exists():
        return vae_path
    return str(config.get("SINGLE_FILE_CHECKPOINT_PATH", "") or "")


def config_dataset_path(config):
    datasets = config.get("INSTANCE_DATASETS") or []
    if datasets:
        return str(datasets[0].get("path", "") or "")
    return ""


def config_text_value(value):
    if value is None:
        return ""
    return str(value)


def fix_alpha(image):
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "PA", "LA"):
        return image.convert("RGB")
    return image.convert("RGB")


def resize_for_vae(image, max_side, multiple=64):
    image = fix_alpha(ImageOps.exif_transpose(image))
    width, height = image.size
    scale = min(float(max_side) / max(width, height), 1.0)
    width = max(multiple, int(round(width * scale)))
    height = max(multiple, int(round(height * scale)))
    width = max(multiple, (width // multiple) * multiple)
    height = max(multiple, (height // multiple) * multiple)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def tensor_to_numpy(tensor):
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()


def tensor_to_pil(tensor):
    return numpy_to_pil(tensor_to_numpy(tensor)[0])


def numpy_to_pil(array):
    return Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8))


def pil_to_pixmap(image, max_size):
    image = ImageOps.contain(image.convert("RGB"), (max_size, max_size), Image.Resampling.LANCZOS)
    data = image.convert("RGBA").tobytes("raw", "RGBA")
    qimage = QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimage.copy())


def pil_to_qimage(image):
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    return QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format.Format_RGBA8888).copy()


def load_vae(path, latent_channels):
    kwargs = {"torch_dtype": torch.float32, "low_cpu_mem_usage": False}
    if latent_channels <= 0:
        try:
            return AutoencoderKL.from_single_file(path, **kwargs)
        except Exception as exc:
            message = str(exc)
            if "latent_channels" not in message and "torch.Size([32" not in message and "torch.Size([64" not in message:
                raise
            kwargs.update({"latent_channels": 32, "ignore_mismatched_sizes": True})
            return AutoencoderKL.from_single_file(path, **kwargs)
    if latent_channels != 4:
        kwargs.update({"latent_channels": latent_channels, "ignore_mismatched_sizes": True})
    return AutoencoderKL.from_single_file(path, **kwargs)


def parse_optional_float(text):
    text = text.strip()
    if not text or text.lower() in {"auto", "none", "null"}:
        return None
    return float(text)


def rgb_stats(array):
    array = np.clip(array.astype(np.float32), 0, 1)
    maxc = array.max(axis=2)
    minc = array.min(axis=2)
    luma = 0.2126 * array[:, :, 0] + 0.7152 * array[:, :, 1] + 0.0722 * array[:, :, 2]
    sat = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    return float(luma.mean()), float(sat.mean()), float((luma < 0.04).mean()), float((luma > 0.96).mean())


def compare_score(reference, candidate):
    ref = np.clip(reference.astype(np.float32), 0, 1)
    cand = np.clip(candidate.astype(np.float32), 0, 1)
    rmse = float(np.sqrt(np.mean((cand - ref) ** 2)))
    ref_luma, ref_sat, ref_black, ref_white = rgb_stats(ref)
    cand_luma, cand_sat, cand_black, cand_white = rgb_stats(cand)
    luma_err = abs(cand_luma - ref_luma)
    sat_err = abs(cand_sat - ref_sat)
    black_err = abs(cand_black - ref_black)
    white_err = abs(cand_white - ref_white)
    score = rmse + 1.5 * luma_err + 1.5 * sat_err + 0.75 * black_err + 0.75 * white_err
    return score, rmse, luma_err, sat_err, black_err, white_err


def build_values(start, stop, step):
    if step <= 0:
        raise ValueError("Step must be greater than zero.")
    count = int(round((stop - start) / step)) + 1
    values = [start + i * step for i in range(max(0, count))]
    return [round(v, 6) for v in values if start - 1e-9 <= v <= stop + 1e-9]


def choose_balanced_random_images(dataset, max_images, seed):
    grouped = {}
    for path in dataset.rglob("*"):
        if path.suffix.lower() in VALID_EXTENSIONS:
            grouped.setdefault(str(path.parent.relative_to(dataset)), []).append(path)
    rng = random.Random(seed)
    groups = []
    for folder, paths in grouped.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        groups.append((folder, paths))
    groups.sort(key=lambda item: item[0])
    rng.shuffle(groups)
    selected, cursors = [], {folder: 0 for folder, _ in groups}
    while len(selected) < max_images:
        added = False
        for folder, paths in groups:
            cursor = cursors[folder]
            if cursor < len(paths):
                selected.append(paths[cursor])
                cursors[folder] = cursor + 1
                added = True
                if len(selected) >= max_images:
                    break
        if not added:
            break
    rng.shuffle(selected)
    return selected


class ImageSlot(QtWidgets.QWidget):
    def __init__(self, title, min_size=240):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QtWidgets.QLabel(title)
        label.setObjectName("TitleLabel")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image = QtWidgets.QLabel("No image")
        self.image.setObjectName("ImagePanel")
        self.image.setMinimumSize(min_size, min_size)
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(self.image, 1)

    def set_image(self, image, max_size=320):
        if image is not None:
            self.image.setPixmap(pil_to_pixmap(image, max_size))


class ReconstructionWorker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, model_path, image_path, max_side, latent_channels, shift_text, scale_text):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.max_side = max_side
        self.latent_channels = latent_channels
        self.shift_text = shift_text
        self.scale_text = scale_text

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.status.emit(f"Loading VAE on {device}...")
            vae = load_vae(self.model_path, self.latent_channels).to(device, dtype=torch.float32)
            vae.eval()
            vae.enable_slicing()
            try:
                vae.enable_tiling()
            except Exception:
                pass

            config_shift = getattr(vae.config, "shift_factor", None)
            config_scale = getattr(vae.config, "scaling_factor", None)
            shift = parse_optional_float(self.shift_text)
            scale = parse_optional_float(self.scale_text)
            if shift is None:
                shift = config_shift
            if scale is None:
                scale = config_scale if config_scale is not None else 1.0

            image = Image.open(self.image_path)
            original = fix_alpha(ImageOps.exif_transpose(image))
            resized = resize_for_vae(original, self.max_side)
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            pixels = transform(resized).unsqueeze(0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                encoded = vae.encode(pixels)
                raw_latents = encoded.latent_dist.mean if hasattr(encoded, "latent_dist") else encoded
                raw_decoded = vae.decode(raw_latents)
                if hasattr(raw_decoded, "sample"):
                    raw_decoded = raw_decoded.sample
                if shift is not None:
                    normalized = (raw_latents - shift) * scale
                    restored = normalized / scale + shift
                else:
                    normalized = raw_latents * scale
                    restored = normalized / scale
                training_decoded = vae.decode(restored)
                if hasattr(training_decoded, "sample"):
                    training_decoded = training_decoded.sample

            raw_pil = tensor_to_pil(raw_decoded)
            training_pil = tensor_to_pil(training_decoded)
            mae = float(np.abs(np.asarray(resized, dtype=np.float32) - np.asarray(raw_pil, dtype=np.float32)).mean())
            self.result.emit(
                {
                    "original": original,
                    "resized": resized,
                    "raw": raw_pil,
                    "roundtrip": training_pil,
                    "stats": (
                        f"VAE config: class={getattr(vae.config, '_class_name', type(vae).__name__)}, "
                        f"channels={getattr(vae.config, 'latent_channels', '?')}, shift={config_shift}, scale={config_scale}\n"
                        f"Roundtrip used: shift={shift}, scale={scale}\n"
                        f"Input size: {resized.width}x{resized.height} | "
                        f"Raw latent mean={raw_latents.mean().item():.5f}, std={raw_latents.std().item():.5f}, "
                        f"min={raw_latents.min().item():.3f}, max={raw_latents.max().item():.3f}\n"
                        f"Mean absolute pixel difference, resized input vs raw reconstruction: {mae:.2f} / 255"
                    ),
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))


class ScannerWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, int)
    status = QtCore.pyqtSignal(str)
    result = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, options):
        super().__init__()
        self.options = options
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def run(self):
        try:
            opts = self.options
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.status.emit(f"Loading VAE on {device}...")
            vae = load_vae(opts["model_path"], opts["latent_channels"]).to(device, dtype=torch.float32)
            vae.eval()
            vae.enable_slicing()
            try:
                vae.enable_tiling()
            except Exception:
                pass

            paths = choose_balanced_random_images(Path(opts["dataset_path"]), opts["max_images"], opts["seed"])
            if not paths:
                raise ValueError("No images found in dataset folder.")
            candidates = [(s, c) for s in build_values(opts["shift_min"], opts["shift_max"], opts["shift_step"])
                           for c in build_values(opts["scale_min"], opts["scale_max"], opts["scale_step"])]
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            totals = {candidate: np.zeros(6, dtype=np.float64) for candidate in candidates}
            raw_totals = np.zeros(6, dtype=np.float64)
            latent_means, latent_stds, previews, done = [], [], {}, 0

            for image_idx, path in enumerate(paths):
                if self.stop_requested:
                    break
                self.status.emit(f"Scanning {path.name} ({image_idx + 1}/{len(paths)})...")
                resized = resize_for_vae(Image.open(path), opts["max_side"])
                reference = np.asarray(resized, dtype=np.float32) / 255.0
                pixels = transform(resized).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    encoded = vae.encode(pixels)
                    raw_latents = encoded.latent_dist.mean if hasattr(encoded, "latent_dist") else encoded
                    latent_means.append(raw_latents.mean().item())
                    latent_stds.append(raw_latents.std().item())
                    raw_decoded = vae.decode(raw_latents)
                    if hasattr(raw_decoded, "sample"):
                        raw_decoded = raw_decoded.sample
                    raw_np = tensor_to_numpy(raw_decoded)[0]
                    raw_totals += np.asarray(compare_score(reference, raw_np), dtype=np.float64)
                    for start in range(0, len(candidates), opts["decode_batch"]):
                        if self.stop_requested:
                            break
                        chunk = candidates[start:start + opts["decode_batch"]]
                        batch = torch.cat([(raw_latents - shift) * scale for shift, scale in chunk], dim=0)
                        decoded = vae.decode(batch)
                        if hasattr(decoded, "sample"):
                            decoded = decoded.sample
                        decoded_np = tensor_to_numpy(decoded)
                        for idx, candidate in enumerate(chunk):
                            totals[candidate] += np.asarray(compare_score(reference, decoded_np[idx]), dtype=np.float64)
                done += 1
                self.progress.emit(done, len(paths))
                if image_idx == 0:
                    previews["reference"] = resized
                    previews["raw"] = numpy_to_pil(raw_np)

            if done == 0:
                raise RuntimeError("Scan was stopped before any images completed.")
            averaged = []
            for candidate, metric_sum in totals.items():
                metric = metric_sum / done
                averaged.append((float(metric[0]), candidate, metric))
            averaged.sort(key=lambda item: item[0])
            best_score, best_candidate, best_metric = averaged[0]

            if "reference" in previews:
                resized = resize_for_vae(Image.open(paths[0]), opts["max_side"])
                pixels = transform(resized).unsqueeze(0).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    encoded = vae.encode(pixels)
                    raw_latents = encoded.latent_dist.mean if hasattr(encoded, "latent_dist") else encoded
                    best_decoded = vae.decode((raw_latents - best_candidate[0]) * best_candidate[1])
                    if hasattr(best_decoded, "sample"):
                        best_decoded = best_decoded.sample
                    previews["best"] = tensor_to_pil(best_decoded)

            raw_avg = raw_totals / done
            top_lines = [
                f"{rank:02d}. shift={shift:.6f}, scale={scale:.6f} | score={score:.6f}, "
                f"rmse={metric[1]:.6f}, luma_err={metric[2]:.6f}, sat_err={metric[3]:.6f}, "
                f"black_err={metric[4]:.6f}, white_err={metric[5]:.6f}"
                for rank, (score, (shift, scale), metric) in enumerate(averaged[:10], start=1)
            ]
            config_shift = getattr(vae.config, "shift_factor", None)
            config_scale = getattr(vae.config, "scaling_factor", None)
            summary = (
                f"Images scanned: {done}\n"
                f"Sampling seed: {opts['seed']} | balanced random sample across subfolders when possible\n"
                f"VAE config: class={getattr(vae.config, '_class_name', type(vae).__name__)}, "
                f"channels={getattr(vae.config, 'latent_channels', '?')}, shift={config_shift}, scale={config_scale}\n"
                f"Raw latent mean={np.mean(latent_means):.6f}, std={np.mean(latent_stds):.6f}\n\n"
                f"Raw VAE reconstruction baseline:\n"
                f"score={raw_avg[0]:.6f}, rmse={raw_avg[1]:.6f}, luma_err={raw_avg[2]:.6f}, "
                f"sat_err={raw_avg[3]:.6f}, black_err={raw_avg[4]:.6f}, white_err={raw_avg[5]:.6f}\n\n"
                f"Best direct-latent decode candidate:\n"
                f"shift={best_candidate[0]:.6f}, scale={best_candidate[1]:.6f}\n"
                f"score={best_score:.6f}, rmse={best_metric[1]:.6f}, luma_err={best_metric[2]:.6f}, "
                f"sat_err={best_metric[3]:.6f}, black_err={best_metric[4]:.6f}, white_err={best_metric[5]:.6f}\n\n"
                "Top candidates:\n" + "\n".join(top_lines)
            )
            self.result.emit({"summary": summary, "reference": previews.get("reference"), "raw": previews.get("raw"), "best": previews.get("best")})
        except Exception as exc:
            self.error.emit(str(exc))


class SplitCompareCanvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.left_image = None
        self.right_image = None
        self.left_label = "Left"
        self.right_label = "Right"
        self.split = 0.5
        self.zoom = 1.0
        self.dragging = False
        self.setMouseTracking(True)
        self.setMinimumSize(640, 480)

    def set_images(self, left_image, right_image, left_label, right_label):
        self.left_image = pil_to_qimage(left_image)
        self.right_image = pil_to_qimage(right_image)
        self.left_label = left_label
        self.right_label = right_label
        self._update_size()
        self.update()

    def set_zoom(self, zoom_percent):
        self.zoom = max(0.1, float(zoom_percent) / 100.0)
        self._update_size()
        self.update()

    def _update_size(self):
        if self.left_image is None:
            return
        size = self.left_image.size()
        scaled = QtCore.QSize(max(1, int(size.width() * self.zoom)), max(1, int(size.height() * self.zoom)))
        self.setMinimumSize(scaled)
        self.resize(scaled)
        self.updateGeometry()

    def _set_split_from_x(self, x):
        self.split = min(0.99, max(0.01, x / max(1, self.width())))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dragging = True
            self._set_split_from_x(event.position().x())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self._set_split_from_x(event.position().x())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dragging = False
            self._set_split_from_x(event.position().x())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#050607"))
        if self.left_image is None or self.right_image is None:
            painter.setPen(QtGui.QColor("#aeb7c2"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Run reconstruction first")
            return
        target = QtCore.QRect(0, 0, self.width(), self.height())
        painter.drawImage(target, self.right_image)
        split_x = int(self.width() * self.split)
        painter.save()
        painter.setClipRect(0, 0, split_x, self.height())
        painter.drawImage(target, self.left_image)
        painter.restore()
        painter.setPen(QtGui.QPen(QtGui.QColor("#78c2ad"), 2))
        painter.drawLine(split_x, 0, split_x, self.height())
        painter.fillRect(split_x - 1, 0, 2, self.height(), QtGui.QColor("#78c2ad"))
        painter.setPen(QtGui.QColor("#e7e9ee"))
        painter.fillRect(12, 12, 220, 28, QtGui.QColor(0, 0, 0, 150))
        painter.drawText(22, 31, self.left_label)
        painter.fillRect(self.width() - 232, 12, 220, 28, QtGui.QColor(0, 0, 0, 150))
        painter.drawText(self.width() - 222, 31, self.right_label)


class CompareWindow(QtWidgets.QWidget):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.setWindowTitle("Large VAE Compare")
        self.resize(1400, 950)
        root = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.left_combo = QtWidgets.QComboBox()
        self.right_combo = QtWidgets.QComboBox()
        for label in images:
            self.left_combo.addItem(label)
            self.right_combo.addItem(label)
        self.left_combo.setCurrentText("Resized Input To VAE")
        self.right_combo.setCurrentText("Raw VAE Reconstruction")
        self.zoom = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom.setRange(25, 400)
        self.zoom.setValue(100)
        self.zoom_label = QtWidgets.QLabel("100%")
        controls.addWidget(QtWidgets.QLabel("Left"))
        controls.addWidget(self.left_combo)
        controls.addWidget(QtWidgets.QLabel("Right"))
        controls.addWidget(self.right_combo)
        controls.addStretch(1)
        controls.addWidget(QtWidgets.QLabel("Zoom"))
        controls.addWidget(self.zoom)
        controls.addWidget(self.zoom_label)
        fit_btn = QtWidgets.QPushButton("Fit")
        actual_btn = QtWidgets.QPushButton("100%")
        controls.addWidget(fit_btn)
        controls.addWidget(actual_btn)
        root.addLayout(controls)
        self.canvas = SplitCompareCanvas()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidgetResizable(False)
        root.addWidget(self.scroll, 1)
        self.left_combo.currentTextChanged.connect(self.update_compare)
        self.right_combo.currentTextChanged.connect(self.update_compare)
        self.zoom.valueChanged.connect(self.set_zoom)
        fit_btn.clicked.connect(self.fit_to_window)
        actual_btn.clicked.connect(lambda: self.zoom.setValue(100))
        self.update_compare()

    def update_compare(self):
        left = self.left_combo.currentText()
        right = self.right_combo.currentText()
        self.canvas.set_images(self.images[left], self.images[right], left, right)

    def set_zoom(self, value):
        self.zoom_label.setText(f"{value}%")
        self.canvas.set_zoom(value)

    def fit_to_window(self):
        image = self.images[self.left_combo.currentText()]
        viewport = self.scroll.viewport().size()
        percent = int(max(25, min(400, min(viewport.width() / max(1, image.width), viewport.height() / max(1, image.height)) * 100)))
        self.zoom.setValue(percent)


class ReconstructionTab(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        self.worker = None
        self.current_images = None
        self.compare_window = None
        self.build_ui(config)

    def build_ui(self, config):
        root = QtWidgets.QVBoxLayout(self)
        paths_group = QtWidgets.QGroupBox("Inputs")
        paths = QtWidgets.QGridLayout(paths_group)
        self.model_path = QtWidgets.QLineEdit(config_model_path(config))
        self.image_path = QtWidgets.QLineEdit()
        model_btn = QtWidgets.QPushButton("Browse Model/VAE")
        image_btn = QtWidgets.QPushButton("Browse Image")
        model_btn.clicked.connect(lambda: self.pick_file(self.model_path, "Model files (*.safetensors *.ckpt *.pt);;All files (*)"))
        image_btn.clicked.connect(lambda: self.pick_file(self.image_path, "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"))
        paths.addWidget(QtWidgets.QLabel("Checkpoint or VAE"), 0, 0)
        paths.addWidget(self.model_path, 0, 1)
        paths.addWidget(model_btn, 0, 2)
        paths.addWidget(QtWidgets.QLabel("Test image"), 1, 0)
        paths.addWidget(self.image_path, 1, 1)
        paths.addWidget(image_btn, 1, 2)
        root.addWidget(paths_group)

        settings = QtWidgets.QHBoxLayout()
        self.max_side = QtWidgets.QSpinBox()
        self.max_side.setRange(256, 4096)
        self.max_side.setSingleStep(64)
        self.max_side.setValue(1024)
        self.latent_channels = QtWidgets.QSpinBox()
        self.latent_channels.setRange(0, 128)
        self.latent_channels.setValue(int(config.get("VAE_LATENT_CHANNELS", 0) or 0))
        self.shift_override = QtWidgets.QLineEdit(config_text_value(config.get("VAE_SHIFT_FACTOR")))
        self.scale_override = QtWidgets.QLineEdit(config_text_value(config.get("VAE_SCALING_FACTOR")))
        self.run_btn = QtWidgets.QPushButton("Run Reconstruction")
        self.compare_btn = QtWidgets.QPushButton("Open Large Compare")
        self.compare_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_reconstruction)
        self.compare_btn.clicked.connect(self.open_compare)
        for label, widget in [("Max side", self.max_side), ("Channels", self.latent_channels), ("Shift", self.shift_override), ("Scale", self.scale_override)]:
            settings.addWidget(QtWidgets.QLabel(label))
            settings.addWidget(widget)
        settings.addWidget(self.run_btn)
        settings.addWidget(self.compare_btn)
        root.addLayout(settings)

        image_row = QtWidgets.QHBoxLayout()
        self.original_slot = ImageSlot("Original")
        self.resized_slot = ImageSlot("Resized Input To VAE")
        self.raw_slot = ImageSlot("Raw VAE Reconstruction")
        self.roundtrip_slot = ImageSlot("Training Scale/Shift Roundtrip")
        for slot in [self.original_slot, self.resized_slot, self.raw_slot, self.roundtrip_slot]:
            image_row.addWidget(slot, 1)
        root.addLayout(image_row, 1)
        self.status = QtWidgets.QLabel("Load an image, then run reconstruction.")
        root.addWidget(self.status)
        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(120)
        root.addWidget(self.stats)

    def pick_file(self, target, filter_text):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", filter_text)
        if path:
            target.setText(path)

    def run_reconstruction(self):
        if not Path(self.model_path.text()).exists() or not Path(self.image_path.text()).exists():
            QtWidgets.QMessageBox.warning(self, "Missing input", "Choose a valid model/VAE and image.")
            return
        self.run_btn.setEnabled(False)
        self.compare_btn.setEnabled(False)
        self.worker = ReconstructionWorker(self.model_path.text(), self.image_path.text(), self.max_side.value(), self.latent_channels.value(), self.shift_override.text(), self.scale_override.text())
        self.worker.status.connect(self.status.setText)
        self.worker.result.connect(self.show_result)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()

    def show_result(self, data):
        self.current_images = {"Original": data["original"], "Resized Input To VAE": data["resized"], "Raw VAE Reconstruction": data["raw"], "Training Scale/Shift Roundtrip": data["roundtrip"]}
        max_size = max(220, min(320, self.width() // 4 - 24))
        self.original_slot.set_image(data["original"], max_size)
        self.resized_slot.set_image(data["resized"], max_size)
        self.raw_slot.set_image(data["raw"], max_size)
        self.roundtrip_slot.set_image(data["roundtrip"], max_size)
        self.stats.setPlainText(data["stats"])
        self.compare_btn.setEnabled(True)
        self.status.setText("Done.")

    def show_error(self, error):
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Reconstruction failed", error)

    def open_compare(self):
        if self.current_images:
            self.compare_window = CompareWindow(self.current_images)
            self.compare_window.show()


class ScannerTab(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        self.worker = None
        self.build_ui(config)

    def build_ui(self, config):
        root = QtWidgets.QVBoxLayout(self)
        paths_group = QtWidgets.QGroupBox("Inputs")
        paths = QtWidgets.QGridLayout(paths_group)
        self.model_path = QtWidgets.QLineEdit(config_model_path(config))
        self.dataset_path = QtWidgets.QLineEdit(config_dataset_path(config))
        model_btn = QtWidgets.QPushButton("Browse Model/VAE")
        dataset_btn = QtWidgets.QPushButton("Browse Dataset")
        model_btn.clicked.connect(lambda: self.pick_file(self.model_path))
        dataset_btn.clicked.connect(self.pick_dataset)
        paths.addWidget(QtWidgets.QLabel("Checkpoint or VAE"), 0, 0)
        paths.addWidget(self.model_path, 0, 1)
        paths.addWidget(model_btn, 0, 2)
        paths.addWidget(QtWidgets.QLabel("Dataset folder"), 1, 0)
        paths.addWidget(self.dataset_path, 1, 1)
        paths.addWidget(dataset_btn, 1, 2)
        root.addWidget(paths_group)

        grid = QtWidgets.QGridLayout()
        self.latent_channels = self.spin(0, 128, int(config.get("VAE_LATENT_CHANNELS", 32) or 32))
        self.max_images = self.spin(1, 500, 12)
        self.max_side = self.spin(128, 2048, 256)
        self.decode_batch = self.spin(1, 64, 8)
        self.seed = self.spin(0, 2147483647, 42)
        self.shift_min = self.dspin(-5, 5, -0.10, 0.01, 4)
        self.shift_max = self.dspin(-5, 5, 0.10, 0.01, 4)
        self.shift_step = self.dspin(0.0001, 5, 0.02, 0.005, 4)
        self.scale_min = self.dspin(0.01, 10, 0.50, 0.05, 4)
        self.scale_max = self.dspin(0.01, 10, 1.20, 0.05, 4)
        self.scale_step = self.dspin(0.0001, 5, 0.05, 0.01, 4)
        fields = [("Channels", self.latent_channels), ("Max images", self.max_images), ("Max side", self.max_side), ("Decode batch", self.decode_batch), ("Seed", self.seed), ("Shift min", self.shift_min), ("Shift max", self.shift_max), ("Shift step", self.shift_step), ("Scale min", self.scale_min), ("Scale max", self.scale_max), ("Scale step", self.scale_step)]
        for idx, (label, widget) in enumerate(fields):
            row, col = idx // 5, (idx % 5) * 2
            grid.addWidget(QtWidgets.QLabel(label), row, col)
            grid.addWidget(widget, row, col + 1)
        self.run_btn = QtWidgets.QPushButton("Run Scan")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_scan)
        self.stop_btn.clicked.connect(self.stop_scan)
        grid.addWidget(self.run_btn, 2, 8)
        grid.addWidget(self.stop_btn, 2, 9)
        root.addLayout(grid)

        preview_row = QtWidgets.QHBoxLayout()
        self.reference_slot = ImageSlot("Reference", 260)
        self.raw_slot = ImageSlot("Raw VAE Reconstruction", 260)
        self.best_slot = ImageSlot("Best Direct-Latent Candidate", 260)
        for slot in [self.reference_slot, self.raw_slot, self.best_slot]:
            preview_row.addWidget(slot, 1)
        root.addLayout(preview_row, 1)
        self.progress = QtWidgets.QProgressBar()
        root.addWidget(self.progress)
        self.status = QtWidgets.QLabel("Choose a dataset and run scan.")
        root.addWidget(self.status)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        root.addWidget(self.output, 1)

    def spin(self, minimum, maximum, value):
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        return widget

    def dspin(self, minimum, maximum, value, step, decimals):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def pick_file(self, target):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select checkpoint or VAE", "", "Model files (*.safetensors *.ckpt *.pt);;All files (*)")
        if path:
            target.setText(path)

    def pick_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if path:
            self.dataset_path.setText(path)

    def run_scan(self):
        if not Path(self.model_path.text()).exists() or not Path(self.dataset_path.text()).exists():
            QtWidgets.QMessageBox.warning(self, "Missing input", "Choose a valid model/VAE and dataset.")
            return
        options = {
            "model_path": self.model_path.text(),
            "dataset_path": self.dataset_path.text(),
            "latent_channels": self.latent_channels.value(),
            "max_images": self.max_images.value(),
            "max_side": self.max_side.value(),
            "decode_batch": self.decode_batch.value(),
            "seed": self.seed.value(),
            "shift_min": self.shift_min.value(),
            "shift_max": self.shift_max.value(),
            "shift_step": self.shift_step.value(),
            "scale_min": self.scale_min.value(),
            "scale_max": self.scale_max.value(),
            "scale_step": self.scale_step.value(),
        }
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.output.clear()
        self.progress.setValue(0)
        self.worker = ScannerWorker(options)
        self.worker.status.connect(self.status.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.show_result)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.scan_finished)
        self.worker.start()

    def stop_scan(self):
        if self.worker:
            self.worker.stop()
            self.status.setText("Stopping after current image/chunk...")

    def update_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def show_result(self, data):
        self.output.setPlainText(data["summary"])
        self.reference_slot.set_image(data.get("reference"), 360)
        self.raw_slot.set_image(data.get("raw"), 360)
        self.best_slot.set_image(data.get("best"), 360)
        self.status.setText("Scan complete.")

    def show_error(self, error):
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Scan failed", error)

    def scan_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)


class VAEDiagnosticsWindow(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("VAE Diagnostics")
        self.resize(1260, 860)
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(ReconstructionTab(config), "Reconstruction Compare")
        tabs.addTab(ScannerTab(config), "Scale/Shift Scanner")
        layout.addWidget(tabs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = VAEDiagnosticsWindow(load_config(args.config))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
