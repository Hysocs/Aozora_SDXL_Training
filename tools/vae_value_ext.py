"""
Flux2 VAE Stats Inspector
=========================
Loads a checkpoint (full model or standalone VAE) and dumps every key that
looks like a normalization statistic — running_mean, running_var, BatchNorm
weights/biases, GroupNorm, etc. — so you can find the per-channel
shift/scale that Flux.2's autoencoder bakes in via its `normalize` submodule.

Two file inputs:
- Model checkpoint (the full unified .safetensors that ships with VAE inside)
- Raw VAE file (the standalone Flux.2 VAE)

Outputs for each file:
- Every key matching normalization patterns, with shape and dtype
- For 1D tensors of plausible channel sizes (32, 64, 128), the actual values
- Filterable text view + JSON export
"""

import json
import sys
from pathlib import Path

import torch
from PyQt6 import QtCore, QtGui, QtWidgets
from safetensors import safe_open


STYLE = """
QWidget { background: #15171b; color: #e7e9ee; font-family: Segoe UI, sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #353b45; border-radius: 6px; margin-top: 10px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: #78c2ad; font-weight: 600; }
QPushButton { background: #242a31; border: 1px solid #3d4651; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { border-color: #78c2ad; color: #78c2ad; }
QPushButton:disabled { color: #69717d; border-color: #2b3037; }
QLineEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; padding: 6px; }
QPlainTextEdit { background: #101216; border: 1px solid #353b45; border-radius: 5px; font-family: Consolas, monospace; font-size: 12px; }
QLabel#TitleLabel { color: #aeb7c2; font-weight: 600; }
QCheckBox { spacing: 6px; }
"""

# Patterns that indicate the latent normalization stats specifically.
# We do NOT match generic GroupNorm/LayerNorm in the encoder/decoder body —
# those are internal architecture and not what we need.
NORM_PATTERNS = [
    "running_mean",
    "running_var",
    "shift_factor",
    "scaling_factor",
    "scale_factor",
    "latent_mean",
    "latent_std",
    "latents_mean",
    "latents_std",
]

# Plausible channel counts for a Flux.2 32ch VAE
PLAUSIBLE_CHANNEL_COUNTS = {32, 64, 128}


def matches_norm_pattern(key):
    k = key.lower()
    return any(p in k for p in NORM_PATTERNS)


def is_plausible_channel_vector(tensor):
    """A 1D float tensor of length 32, 64, or 128 — what BatchNorm running stats look like."""
    return (
        tensor.ndim == 1
        and tensor.is_floating_point()
        and tensor.shape[0] in PLAUSIBLE_CHANNEL_COUNTS
    )


def format_tensor_summary(tensor, key, max_values_inline=8):
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype).replace("torch.", "")
    numel = tensor.numel()

    lines = [f"  shape={shape}  dtype={dtype}  numel={numel}"]

    # Quick stats on float tensors
    if tensor.is_floating_point():
        try:
            t = tensor.float()
            mean = t.mean().item()
            std = t.std().item() if numel > 1 else 0.0
            tmin = t.min().item()
            tmax = t.max().item()
            lines.append(f"  mean={mean:+.6f}  std={std:.6f}  min={tmin:+.6f}  max={tmax:+.6f}")
        except Exception:
            pass

    # If 1D and a plausible channel size, show full values
    if tensor.ndim == 1 and shape[0] in PLAUSIBLE_CHANNEL_COUNTS:
        try:
            vals = tensor.float().tolist()
            lines.append(f"  values ({shape[0]}):")
            # Format 4 per row
            for i in range(0, len(vals), 4):
                row = vals[i:i + 4]
                row_str = "  ".join(f"[{i + j:3d}]={v:+.6f}" for j, v in enumerate(row))
                lines.append(f"    {row_str}")
        except Exception as e:
            lines.append(f"  (could not extract values: {e})")
    elif tensor.ndim == 1 and numel <= max_values_inline:
        try:
            vals = tensor.float().tolist()
            lines.append(f"  values: {vals}")
        except Exception:
            pass

    return "\n".join(lines)


def scan_checkpoint(path, mode="norm"):
    """Walk every key in a safetensors file.
    mode: 'norm' = only normalization-pattern keys
          'channel_vectors' = only 1D float tensors of plausible channel sizes
          'all' = everything (likely huge)
    """
    path = Path(path)
    if not path.exists():
        return [], {"error": f"file not found: {path}"}

    matches = []
    structured = {"path": str(path), "all_keys_count": 0, "matches": {}}

    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            structured["all_keys_count"] = len(keys)

            for k in keys:
                if mode == "norm" and not matches_norm_pattern(k):
                    continue
                if mode == "channel_vectors":
                    # Need to peek at shape without loading full tensor
                    slice_ = f.get_slice(k)
                    shape = slice_.get_shape()
                    if not (len(shape) == 1 and shape[0] in PLAUSIBLE_CHANNEL_COUNTS):
                        continue
                tensor = f.get_tensor(k)
                if mode == "channel_vectors" and not tensor.is_floating_point():
                    continue
                summary = format_tensor_summary(tensor, k)
                matches.append((k, summary))

                entry = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": tensor.numel(),
                }
                if tensor.is_floating_point():
                    try:
                        t = tensor.float()
                        entry["mean"] = float(t.mean())
                        entry["std"] = float(t.std()) if tensor.numel() > 1 else 0.0
                        entry["min"] = float(t.min())
                        entry["max"] = float(t.max())
                    except Exception:
                        pass
                if tensor.ndim == 1 and tensor.shape[0] in PLAUSIBLE_CHANNEL_COUNTS:
                    try:
                        entry["values"] = tensor.float().tolist()
                    except Exception:
                        pass
                structured["matches"][k] = entry
    except Exception as e:
        structured["error"] = str(e)
        return [], structured

    return matches, structured


def scan_pytorch_file(path, mode="norm"):
    """Same as scan_checkpoint but for .pt / .ckpt files."""
    path = Path(path)
    if not path.exists():
        return [], {"error": f"file not found: {path}"}

    structured = {"path": str(path), "all_keys_count": 0, "matches": {}}
    matches = []

    try:
        sd = torch.load(str(path), map_location="cpu", weights_only=False)
        candidates = [sd]
        if isinstance(sd, dict):
            for k in ("state_dict", "model", "vae", "first_stage_model", "weights"):
                if k in sd and isinstance(sd[k], dict):
                    candidates.append(sd[k])

        def tensor_count(d):
            return sum(1 for v in d.values() if isinstance(v, torch.Tensor)) if isinstance(d, dict) else 0

        best = max(candidates, key=tensor_count)
        keys = [k for k, v in best.items() if isinstance(v, torch.Tensor)]
        structured["all_keys_count"] = len(keys)

        for k in keys:
            if mode == "norm" and not matches_norm_pattern(k):
                continue
            tensor = best[k]
            if mode == "channel_vectors" and not is_plausible_channel_vector(tensor):
                continue
            summary = format_tensor_summary(tensor, k)
            matches.append((k, summary))

            entry = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": tensor.numel(),
            }
            if tensor.is_floating_point():
                try:
                    t = tensor.float()
                    entry["mean"] = float(t.mean())
                    entry["std"] = float(t.std()) if tensor.numel() > 1 else 0.0
                    entry["min"] = float(t.min())
                    entry["max"] = float(t.max())
                except Exception:
                    pass
            if tensor.ndim == 1 and tensor.shape[0] in PLAUSIBLE_CHANNEL_COUNTS:
                try:
                    entry["values"] = tensor.float().tolist()
                except Exception:
                    pass
            structured["matches"][k] = entry

    except Exception as e:
        structured["error"] = str(e)
        return [], structured

    return matches, structured


def scan_any(path, mode="norm"):
    """Dispatch to the right scanner based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".safetensors":
        return scan_checkpoint(path, mode)
    return scan_pytorch_file(path, mode)


def render_matches(label, matches, total_keys):
    if not matches:
        return f"=== {label} ===\nNo matching keys among {total_keys} total keys.\n"

    lines = [f"=== {label} ==="]
    lines.append(f"Total keys in file: {total_keys}")
    lines.append(f"Matching keys: {len(matches)}")
    lines.append("")
    for k, summary in matches:
        lines.append(k)
        lines.append(summary)
        lines.append("")
    return "\n".join(lines)


class ScanWorker(QtCore.QThread):
    finished_with_data = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, model_path, vae_path, mode, custom_filter):
        super().__init__()
        self.model_path = model_path
        self.vae_path = vae_path
        self.mode = mode
        self.custom_filter = (custom_filter or "").strip().lower()

    def _filter_matches(self, matches):
        if not self.custom_filter:
            return matches
        return [(k, s) for k, s in matches if self.custom_filter in k.lower()]

    def run(self):
        try:
            results = {"model": None, "vae": None}
            if self.model_path and Path(self.model_path).exists():
                matches, structured = scan_any(self.model_path, self.mode)
                matches = self._filter_matches(matches)
                results["model"] = {
                    "label": f"MODEL: {Path(self.model_path).name}",
                    "matches": matches,
                    "total_keys": structured.get("all_keys_count", 0),
                    "structured": structured,
                }
            if self.vae_path and Path(self.vae_path).exists():
                matches, structured = scan_any(self.vae_path, self.mode)
                matches = self._filter_matches(matches)
                results["vae"] = {
                    "label": f"VAE: {Path(self.vae_path).name}",
                    "matches": matches,
                    "total_keys": structured.get("all_keys_count", 0),
                    "structured": structured,
                }
            self.finished_with_data.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux2 VAE Stats Inspector")
        self.resize(1200, 900)
        self.worker = None
        self.last_results = None
        self._build()

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)

        inputs = QtWidgets.QGroupBox("Inputs (you can fill one or both)")
        grid = QtWidgets.QGridLayout(inputs)

        self.model_path = QtWidgets.QLineEdit()
        model_btn = QtWidgets.QPushButton("Browse Model")
        model_btn.clicked.connect(lambda: self._pick(self.model_path))

        self.vae_path = QtWidgets.QLineEdit()
        vae_btn = QtWidgets.QPushButton("Browse Raw VAE")
        vae_btn.clicked.connect(lambda: self._pick(self.vae_path))

        grid.addWidget(QtWidgets.QLabel("Model checkpoint"), 0, 0)
        grid.addWidget(self.model_path, 0, 1)
        grid.addWidget(model_btn, 0, 2)
        grid.addWidget(QtWidgets.QLabel("Raw VAE file"), 1, 0)
        grid.addWidget(self.vae_path, 1, 1)
        grid.addWidget(vae_btn, 1, 2)

        root.addWidget(inputs)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Latent stats only (running_mean, running_var, etc.)", "norm")
        self.mode_combo.addItem("All 1D channel vectors (length 32/64/128)", "channel_vectors")
        self.mode_combo.addItem("Everything (huge output)", "all")
        self.mode_combo.setToolTip(
            "Latent stats: only keys named like running_mean / running_var / shift_factor.\n"
            "Channel vectors: any 1D float tensor of length 32/64/128 — the actual μ/σ "
            "shape for a Flux.2 VAE, regardless of what it's named.\n"
            "Everything: dumps every key. Only use if both other modes find nothing."
        )
        controls.addWidget(self.mode_combo)

        controls.addWidget(QtWidgets.QLabel("Extra filter:"))
        self.custom_filter = QtWidgets.QLineEdit()
        self.custom_filter.setPlaceholderText("substring filter (optional)")
        controls.addWidget(self.custom_filter, 1)

        self.scan_btn = QtWidgets.QPushButton("Scan")
        self.scan_btn.clicked.connect(self._scan)
        controls.addWidget(self.scan_btn)

        self.export_btn = QtWidgets.QPushButton("Export JSON…")
        self.export_btn.clicked.connect(self._export_json)
        self.export_btn.setEnabled(False)
        controls.addWidget(self.export_btn)

        root.addLayout(controls)

        self.status = QtWidgets.QLabel("Pick a checkpoint and/or raw VAE, then Scan.")
        root.addWidget(self.status)

        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        root.addWidget(self.output, 1)

        hint = QtWidgets.QLabel(
            "Hint for Flux.2 VAE: look for keys containing 'normalize', 'running_mean', 'running_var'. "
            "Per-channel vectors (length 32 or 128) are the actual μ/σ used to normalize latents."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #78c2ad; padding: 6px;")
        root.addWidget(hint)

    def _pick(self, target):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", "", "Model files (*.safetensors *.ckpt *.pt *.bin);;All files (*)"
        )
        if path:
            target.setText(path)

    def _scan(self):
        model_path = self.model_path.text().strip()
        vae_path = self.vae_path.text().strip()
        if not model_path and not vae_path:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Provide at least one file path.")
            return

        self.scan_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.status.setText("Scanning...")
        self.output.clear()

        mode = self.mode_combo.currentData()
        self.worker = ScanWorker(
            model_path=model_path,
            vae_path=vae_path,
            mode=mode,
            custom_filter=self.custom_filter.text(),
        )
        self.worker.finished_with_data.connect(self._show_results)
        self.worker.error.connect(self._show_error)
        self.worker.finished.connect(lambda: self.scan_btn.setEnabled(True))
        self.worker.start()

    def _show_results(self, results):
        self.last_results = results
        chunks = []
        for key in ("model", "vae"):
            r = results.get(key)
            if r is None:
                continue
            chunks.append(render_matches(r["label"], r["matches"], r["total_keys"]))
        self.output.setPlainText("\n\n".join(chunks))
        total_matches = sum(len(r["matches"]) for r in results.values() if r)
        self.status.setText(f"Done. {total_matches} matching keys across files.")
        self.export_btn.setEnabled(True)

    def _show_error(self, msg):
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Scan failed", msg)

    def _export_json(self):
        if not self.last_results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export JSON", "vae_stats.json", "JSON (*.json)")
        if not path:
            return
        export = {}
        for key, r in self.last_results.items():
            if r is not None:
                export[key] = r["structured"]
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export, f, indent=2, default=str)
            self.status.setText(f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()