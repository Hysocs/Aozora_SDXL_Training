import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps
from PyQt6 import QtCore, QtGui, QtWidgets
from safetensors import safe_open
from torchvision import transforms


STYLE = """
QWidget { background: #15171b; color: #e7e9ee; font-family: Segoe UI, sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #353b45; border-radius: 6px; margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #78c2ad; font-weight: 600; }
QPushButton { background: #242a31; border: 1px solid #3d4651; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { border-color: #78c2ad; color: #78c2ad; }
QPushButton:disabled { color: #69717d; border-color: #2b3037; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #101216; border: 1px solid #353b45; border-radius: 5px; padding: 6px; }
QPlainTextEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; font-family: Consolas, monospace; }
QProgressBar { border: 1px solid #353b45; border-radius: 5px; text-align: center; background: #101216; }
QProgressBar::chunk { background: #78c2ad; }
"""

FLUX_BN_EPS = 1e-4
BN_MEAN_SUFFIXES = [
    "bn.running_mean",
    "normalize.bn.running_mean",
    "normalize.running_mean",
]
BN_VAR_SUFFIXES = [
    "bn.running_var",
    "normalize.bn.running_var",
    "normalize.running_var",
]


def fix_alpha(image):
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "PA", "LA"):
        return image.convert("RGB")
    return image.convert("RGB")


def resize_for_vae(image, max_side, multiple=64):
    image = fix_alpha(image)
    width, height = image.size
    if max_side > 0:
        scale = min(float(max_side) / max(width, height), 1.0)
        width = max(multiple, int(round(width * scale)))
        height = max(multiple, int(round(height * scale)))
    width = max(multiple, (width // multiple) * multiple)
    height = max(multiple, (height // multiple) * multiple)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def pil_to_qimage(image):
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    return QtGui.QImage(data, image.width, image.height, QtGui.QImage.Format.Format_RGBA8888).copy()


def tensor_to_pil(tensor):
    tensor = (tensor.detach().cpu().float() / 2 + 0.5).clamp(0, 1)
    array = tensor[0].permute(1, 2, 0).numpy()
    array = (array * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(array)


def image_stats(reference, candidate):
    ref = np.asarray(reference, dtype=np.float32) / 255.0
    cand = np.asarray(candidate, dtype=np.float32) / 255.0
    mae = float(np.abs(ref - cand).mean() * 255.0)
    rmse = float(np.sqrt(((ref - cand) ** 2).mean()) * 255.0)
    ref_max = ref.max(axis=2)
    ref_min = ref.min(axis=2)
    cand_max = cand.max(axis=2)
    cand_min = cand.min(axis=2)
    ref_sat = np.where(ref_max > 1e-6, (ref_max - ref_min) / np.maximum(ref_max, 1e-6), 0.0)
    cand_sat = np.where(cand_max > 1e-6, (cand_max - cand_min) / np.maximum(cand_max, 1e-6), 0.0)
    sat_delta = float((cand_sat.mean() - ref_sat.mean()) * 100.0)
    return mae, rmse, sat_delta


def load_vae(path, latent_channels):
    kwargs = {"torch_dtype": torch.float32, "low_cpu_mem_usage": False}
    if latent_channels and latent_channels != 4:
        kwargs.update({"latent_channels": latent_channels, "ignore_mismatched_sizes": True})
    return AutoencoderKL.from_single_file(path, **kwargs)


def find_tensor_by_suffix(safetensors_path, suffixes):
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for suffix in suffixes:
            matches = [key for key in keys if key == suffix or key.endswith("." + suffix)]
            if matches:
                key = sorted(matches, key=len)[0]
                return handle.get_tensor(key).float(), key
    return None, None


def extract_flux_bn_stats(safetensors_path):
    mean, mean_key = find_tensor_by_suffix(safetensors_path, BN_MEAN_SUFFIXES)
    var, var_key = find_tensor_by_suffix(safetensors_path, BN_VAR_SUFFIXES)
    if mean is None or var is None:
        raise RuntimeError("Could not find Flux BN stats ending with bn.running_mean and bn.running_var.")
    if mean.numel() != 128 or var.numel() != 128:
        raise RuntimeError(f"Flux BN stats have wrong shape: mean={tuple(mean.shape)}, var={tuple(var.shape)}.")
    return mean, var, mean_key, var_key


def flux_bn128_to_bn32(mean_128, var_128):
    mean_groups = mean_128.view(32, 4)
    var_groups = var_128.view(32, 4)
    mean_32 = mean_groups.mean(dim=1)
    var_32 = (var_groups + mean_groups.pow(2)).mean(dim=1) - mean_32.pow(2)
    sigma_32 = torch.sqrt(var_32 + FLUX_BN_EPS)
    return mean_32, sigma_32


def apply_flux_bn32_norm(latents, mean_32, sigma_32):
    mean_32 = mean_32.to(device=latents.device, dtype=latents.dtype).view(1, 32, 1, 1)
    sigma_32 = sigma_32.to(device=latents.device, dtype=latents.dtype).view(1, 32, 1, 1)
    return (latents - mean_32) / sigma_32


def invert_flux_bn32_norm(latents, mean_32, sigma_32):
    mean_32 = mean_32.to(device=latents.device, dtype=latents.dtype).view(1, 32, 1, 1)
    sigma_32 = sigma_32.to(device=latents.device, dtype=latents.dtype).view(1, 32, 1, 1)
    return latents * sigma_32 + mean_32


def parse_optional_float(text):
    text = text.strip()
    if not text or text.lower() in {"none", "null", "auto"}:
        return None
    return float(text)


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
        self.setMinimumSize(900, 600)

    def set_images(self, left_image, right_image, left_label, right_label):
        self.left_image = pil_to_qimage(left_image)
        self.right_image = pil_to_qimage(right_image)
        self.left_label = left_label
        self.right_label = right_label
        self.update_size()
        self.update()

    def set_zoom(self, percent):
        self.zoom = max(0.05, float(percent) / 100.0)
        self.update_size()
        self.update()

    def update_size(self):
        if self.left_image is None:
            return
        width = max(1, int(self.left_image.width() * self.zoom))
        height = max(1, int(self.left_image.height() * self.zoom))
        self.setMinimumSize(width, height)
        self.resize(width, height)
        self.updateGeometry()

    def set_split_from_x(self, x):
        self.split = min(0.995, max(0.005, float(x) / max(1, self.width())))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dragging = True
            self.set_split_from_x(event.position().x())

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.set_split_from_x(event.position().x())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.dragging = False
            self.set_split_from_x(event.position().x())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#050607"))
        if self.left_image is None or self.right_image is None:
            painter.setPen(QtGui.QColor("#aeb7c2"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Load an image and run reconstruction")
            return

        target = QtCore.QRect(0, 0, self.width(), self.height())
        painter.drawImage(target, self.right_image)
        split_x = int(self.width() * self.split)
        painter.save()
        painter.setClipRect(0, 0, split_x, self.height())
        painter.drawImage(target, self.left_image)
        painter.restore()

        painter.fillRect(split_x - 2, 0, 4, self.height(), QtGui.QColor("#78c2ad"))
        painter.setPen(QtGui.QColor("#e7e9ee"))
        painter.fillRect(12, 12, 260, 28, QtGui.QColor(0, 0, 0, 155))
        painter.drawText(22, 31, self.left_label)
        painter.fillRect(max(12, self.width() - 272), 12, 260, 28, QtGui.QColor(0, 0, 0, 155))
        painter.drawText(max(22, self.width() - 262), 31, self.right_label)


class ReconstructionWorker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, options):
        super().__init__()
        self.options = options

    def run(self):
        try:
            opts = self.options
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae_path = opts["vae_path"] or opts["model_path"]
            bn_source = opts["bn_source"] or vae_path

            self.status.emit(f"Loading VAE on {device}...")
            vae = load_vae(vae_path, opts["latent_channels"]).to(device, dtype=torch.float32)
            vae.eval()
            vae.enable_slicing()
            try:
                vae.enable_tiling()
            except Exception:
                pass

            self.status.emit("Loading image...")
            original = fix_alpha(Image.open(opts["image_path"]))
            resized = resize_for_vae(original, opts["max_side"], opts["multiple"])
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            pixels = transform(resized).unsqueeze(0).to(device=device, dtype=torch.float32)

            mean_32 = sigma_32 = None
            bn_info = ""
            if opts["norm_mode"] == "flux_bn32":
                self.status.emit("Loading Flux BN stats...")
                mean_128, var_128, mean_key, var_key = extract_flux_bn_stats(bn_source)
                mean_32, sigma_32 = flux_bn128_to_bn32(mean_128, var_128)
                bn_info = (
                    f"BN stats source: {bn_source}\n"
                    f"mean key: {mean_key}\n"
                    f"var key:  {var_key}\n"
                    f"mu_32 range:    [{mean_32.min().item():+.5f}, {mean_32.max().item():+.5f}]\n"
                    f"sigma_32 range: [{sigma_32.min().item():+.5f}, {sigma_32.max().item():+.5f}]\n"
                )

            self.status.emit("Encoding, caching latent, and decoding...")
            with torch.no_grad():
                encoded = vae.encode(pixels)
                raw_latents = encoded.latent_dist.mean if hasattr(encoded, "latent_dist") else encoded
                raw_decoded = vae.decode(raw_latents)
                if hasattr(raw_decoded, "sample"):
                    raw_decoded = raw_decoded.sample

                if opts["norm_mode"] == "flux_bn32":
                    if raw_latents.shape[1] != 32:
                        raise RuntimeError(f"flux_bn32 expects 32-channel latents, got {tuple(raw_latents.shape)}")
                    cached_latents = apply_flux_bn32_norm(raw_latents, mean_32, sigma_32)
                    restored = invert_flux_bn32_norm(cached_latents, mean_32, sigma_32)
                else:
                    shift = opts["shift"]
                    scale = opts["scale"]
                    if scale is None:
                        scale = getattr(vae.config, "scaling_factor", 1.0) or 1.0
                    if shift is None:
                        shift = getattr(vae.config, "shift_factor", None)
                    if shift is not None:
                        cached_latents = (raw_latents - shift) * scale
                        restored = cached_latents / scale + shift
                    else:
                        cached_latents = raw_latents * scale
                        restored = cached_latents / scale

                cached_decoded = vae.decode(restored)
                if hasattr(cached_decoded, "sample"):
                    cached_decoded = cached_decoded.sample

            raw_pil = tensor_to_pil(raw_decoded)
            cached_pil = tensor_to_pil(cached_decoded)
            mae, rmse, sat_delta = image_stats(resized, cached_pil)

            cache_dir = Path(__file__).resolve().with_name("_vae_recon_cache")
            cache_dir.mkdir(exist_ok=True)
            stamp = int(time.time())
            stem = Path(opts["image_path"]).stem.replace(" ", "_")
            lat_path = cache_dir / f"{stem}_{stamp}_lat.pt"
            meta_path = cache_dir / f"{stem}_{stamp}_meta.json"
            torch.save(cached_latents[0].to(dtype=torch.float32).cpu(), lat_path)
            metadata = {
                "image_path": opts["image_path"],
                "model_path": opts["model_path"],
                "vae_path": vae_path,
                "bn_source": bn_source if opts["norm_mode"] == "flux_bn32" else None,
                "normalization_mode": opts["norm_mode"],
                "latent_shape": list(cached_latents.shape),
                "raw_mean": float(raw_latents.mean().item()),
                "raw_std": float(raw_latents.std().item()),
                "cached_mean": float(cached_latents.mean().item()),
                "cached_std": float(cached_latents.std().item()),
            }
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            stats = (
                f"Image: {resized.width}x{resized.height}\n"
                f"VAE: {vae_path}\n"
                f"VAE config: channels={getattr(vae.config, 'latent_channels', '?')}, "
                f"shift={getattr(vae.config, 'shift_factor', None)}, "
                f"scale={getattr(vae.config, 'scaling_factor', None)}\n"
                f"Norm mode: {opts['norm_mode']}\n"
                f"{bn_info}"
                f"Raw latent mean/std/range: {raw_latents.mean().item():+.5f} / {raw_latents.std().item():.5f} / "
                f"[{raw_latents.min().item():+.4f}, {raw_latents.max().item():+.4f}]\n"
                f"Cached latent mean/std/range: {cached_latents.mean().item():+.5f} / {cached_latents.std().item():.5f} / "
                f"[{cached_latents.min().item():+.4f}, {cached_latents.max().item():+.4f}]\n"
                f"Cached reconstruction MAE/RMSE: {mae:.3f} / {rmse:.3f} px\n"
                f"Cached reconstruction saturation delta: {sat_delta:+.3f} percentage points\n"
                f"Saved diagnostic latent: {lat_path}\n"
                f"Saved diagnostic metadata: {meta_path}"
            )

            self.finished_ok.emit(
                {
                    "images": {
                        "Input": resized,
                        "Raw VAE Reconstruction": raw_pil,
                        "Cached Latent Reconstruction": cached_pil,
                    },
                    "stats": stats,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAE Latent Reconstruction Viewer")
        self.resize(1500, 1000)
        self.images = {}
        self.worker = None

        root = QtWidgets.QVBoxLayout(self)
        inputs = QtWidgets.QGroupBox("Inputs")
        grid = QtWidgets.QGridLayout(inputs)
        self.model_path = QtWidgets.QLineEdit()
        self.vae_path = QtWidgets.QLineEdit()
        self.image_path = QtWidgets.QLineEdit()
        self.bn_source = QtWidgets.QLineEdit()
        grid.addWidget(QtWidgets.QLabel("Checkpoint or VAE"), 0, 0)
        grid.addWidget(self.model_path, 0, 1)
        grid.addWidget(self.browse_button(self.model_path, "Model files (*.safetensors *.ckpt *.pt);;All files (*)"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("Separate VAE optional"), 1, 0)
        grid.addWidget(self.vae_path, 1, 1)
        grid.addWidget(self.browse_button(self.vae_path, "Model files (*.safetensors *.ckpt *.pt);;All files (*)"), 1, 2)
        grid.addWidget(QtWidgets.QLabel("Image"), 2, 0)
        grid.addWidget(self.image_path, 2, 1)
        grid.addWidget(self.browse_button(self.image_path, "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)"), 2, 2)
        grid.addWidget(QtWidgets.QLabel("BN stats source optional"), 3, 0)
        grid.addWidget(self.bn_source, 3, 1)
        grid.addWidget(self.browse_button(self.bn_source, "Model files (*.safetensors *.ckpt *.pt);;All files (*)"), 3, 2)
        root.addWidget(inputs)

        settings = QtWidgets.QHBoxLayout()
        self.norm_mode = QtWidgets.QComboBox()
        self.norm_mode.addItems(["flux_bn32", "scalar"])
        self.latent_channels = QtWidgets.QSpinBox()
        self.latent_channels.setRange(4, 128)
        self.latent_channels.setValue(32)
        self.max_side = QtWidgets.QSpinBox()
        self.max_side.setRange(64, 4096)
        self.max_side.setValue(1024)
        self.multiple = QtWidgets.QSpinBox()
        self.multiple.setRange(8, 128)
        self.multiple.setValue(64)
        self.shift = QtWidgets.QLineEdit("auto")
        self.scale = QtWidgets.QLineEdit("auto")
        for label, widget in [
            ("Norm", self.norm_mode),
            ("Latent ch", self.latent_channels),
            ("Max side", self.max_side),
            ("Multiple", self.multiple),
            ("Scalar shift", self.shift),
            ("Scalar scale", self.scale),
        ]:
            settings.addWidget(QtWidgets.QLabel(label))
            settings.addWidget(widget)
        self.run_btn = QtWidgets.QPushButton("Run Reconstruction")
        self.run_btn.clicked.connect(self.run_reconstruction)
        settings.addWidget(self.run_btn)
        root.addLayout(settings)

        controls = QtWidgets.QHBoxLayout()
        self.left_combo = QtWidgets.QComboBox()
        self.right_combo = QtWidgets.QComboBox()
        self.zoom = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.zoom.setRange(25, 800)
        self.zoom.setValue(100)
        self.zoom_label = QtWidgets.QLabel("100%")
        fit_btn = QtWidgets.QPushButton("Fit")
        actual_btn = QtWidgets.QPushButton("100%")
        fit_btn.clicked.connect(self.fit_to_window)
        actual_btn.clicked.connect(lambda: self.zoom.setValue(100))
        self.left_combo.currentTextChanged.connect(self.update_compare)
        self.right_combo.currentTextChanged.connect(self.update_compare)
        self.zoom.valueChanged.connect(self.set_zoom)
        for label, widget in [("Left", self.left_combo), ("Right", self.right_combo), ("Zoom", self.zoom)]:
            controls.addWidget(QtWidgets.QLabel(label))
            controls.addWidget(widget)
        controls.addWidget(self.zoom_label)
        controls.addWidget(fit_btn)
        controls.addWidget(actual_btn)
        root.addLayout(controls)

        self.canvas = SplitCompareCanvas()
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.scroll, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("Choose a model/VAE and image, then run reconstruction.")
        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setFixedHeight(160)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.stats, 3)
        root.addLayout(bottom)

    def browse_button(self, target, filter_text):
        button = QtWidgets.QPushButton("Browse")
        button.clicked.connect(lambda: self.browse_file(target, filter_text))
        return button

    def browse_file(self, target, filter_text):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select file", "", filter_text)
        if path:
            target.setText(path.replace("\\", "/"))

    def run_reconstruction(self):
        model = self.model_path.text().strip()
        vae = self.vae_path.text().strip()
        image = self.image_path.text().strip()
        if not Path(vae or model).exists() or not Path(image).exists():
            QtWidgets.QMessageBox.warning(self, "Missing input", "Choose a valid model/VAE and image.")
            return
        try:
            shift = parse_optional_float(self.shift.text())
            scale = parse_optional_float(self.scale.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid scalar setting", str(exc))
            return
        options = {
            "model_path": model,
            "vae_path": vae,
            "image_path": image,
            "bn_source": self.bn_source.text().strip(),
            "norm_mode": self.norm_mode.currentText(),
            "latent_channels": self.latent_channels.value(),
            "max_side": self.max_side.value(),
            "multiple": self.multiple.value(),
            "shift": shift,
            "scale": scale,
        }
        self.run_btn.setEnabled(False)
        self.status.setText("Starting...")
        self.worker = ReconstructionWorker(options)
        self.worker.status.connect(self.status.setText)
        self.worker.finished_ok.connect(self.on_result)
        self.worker.failed.connect(self.on_error)
        self.worker.start()

    def on_result(self, result):
        self.run_btn.setEnabled(True)
        self.images = result["images"]
        self.stats.setPlainText(result["stats"])
        self.status.setText("Reconstruction complete.")
        self.left_combo.blockSignals(True)
        self.right_combo.blockSignals(True)
        self.left_combo.clear()
        self.right_combo.clear()
        for label in self.images:
            self.left_combo.addItem(label)
            self.right_combo.addItem(label)
        self.left_combo.setCurrentText("Input")
        self.right_combo.setCurrentText("Cached Latent Reconstruction")
        self.left_combo.blockSignals(False)
        self.right_combo.blockSignals(False)
        self.update_compare()

    def on_error(self, message):
        self.run_btn.setEnabled(True)
        self.status.setText("Reconstruction failed.")
        QtWidgets.QMessageBox.critical(self, "Reconstruction failed", message)

    def update_compare(self):
        left = self.left_combo.currentText()
        right = self.right_combo.currentText()
        if left in self.images and right in self.images:
            self.canvas.set_images(self.images[left], self.images[right], left, right)

    def set_zoom(self, value):
        self.zoom_label.setText(f"{value}%")
        self.canvas.set_zoom(value)

    def fit_to_window(self):
        if not self.images:
            return
        image = self.images[self.left_combo.currentText()]
        viewport = self.scroll.viewport().size()
        percent = int(max(25, min(800, min(viewport.width() / max(1, image.width), viewport.height() / max(1, image.height)) * 100)))
        self.zoom.setValue(percent)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
