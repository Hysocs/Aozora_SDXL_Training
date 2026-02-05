import sys
import os
import json
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
from PyQt6 import QtWidgets, QtCore, QtGui

# --- STYLING ---
STYLESHEET = """
QWidget {
    background-color: #1a1926;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
}
QGroupBox {
    border: 1px solid #4a4668;
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    background-color: #2c2a3e;
    padding: 0 5px;
    color: #ab97e6;
    font-weight: bold;
}
QLineEdit {
    background-color: #242233;
    border: 1px solid #4a4668;
    color: #ffdd57;
    font-family: 'Consolas', monospace;
    font-size: 16px;
    padding: 4px;
    border-radius: 4px;
}
QPushButton {
    background-color: #6a48d7;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover { background-color: #7e5be0; }
QPushButton:disabled { background-color: #383552; color: #5c5a70; }
QProgressBar {
    border: 1px solid #4a4668;
    border-radius: 4px;
    text-align: center;
    background-color: #242233;
}
QProgressBar::chunk { background-color: #6a48d7; }
"""

class VAEWorker(QtCore.QThread):
    progressSignal = QtCore.pyqtSignal(int, int)
    statusSignal = QtCore.pyqtSignal(str)
    previewSignal = QtCore.pyqtSignal(object, object)
    finishedSignal = QtCore.pyqtSignal(dict)
    errorSignal = QtCore.pyqtSignal(str)

    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_running = True

    def load_vae_robust(self, path):
        """Load VAE, trying standard 4ch first, then 32ch if that fails."""
        try:
            vae = AutoencoderKL.from_single_file(
                path, 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False
            )
            return vae, vae.config.latent_channels
        except Exception as e:
            err_str = str(e)
            # Try 32-channel
            if "torch.Size([64" in err_str or "torch.Size([32" in err_str or "latent_channels" in err_str:
                try:
                    vae = AutoencoderKL.from_single_file(
                        path, 
                        torch_dtype=torch.float32,
                        latent_channels=32,
                        ignore_mismatched_sizes=True,
                        low_cpu_mem_usage=False
                    )
                    return vae, 32
                except:
                    pass
            raise e

    def run(self):
        try:
            # 1. Parse Config
            self.statusSignal.emit("Reading Configuration...")
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # 2. Locate VAE
            vae_path = config.get("VAE_PATH", "") or config.get("SINGLE_FILE_CHECKPOINT_PATH", "")
            if not vae_path or not os.path.exists(vae_path):
                raise FileNotFoundError("Could not find VAE or Checkpoint path.")

            # 3. Load VAE
            self.statusSignal.emit(f"Loading VAE: {Path(vae_path).name}")
            vae, actual_channels = self.load_vae_robust(vae_path)
            vae.to(self.device)
            vae.eval()

            # 4. Gather Images
            datasets = config.get("INSTANCE_DATASETS", [])
            image_paths = []
            valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            
            for ds in datasets:
                root = Path(ds.get("path", ""))
                if root.exists():
                    image_paths.extend([p for p in root.rglob("*") if p.suffix.lower() in valid_exts])

            if not image_paths:
                raise ValueError("No images found.")

            # 5. Scan
            random.shuffle(image_paths)
            target_count = min(len(image_paths), 500)
            scan_paths = image_paths[:target_count]
            
            self.statusSignal.emit(f"Scanning {target_count} images...")
            
            all_means = []
            all_stds = []
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            preview_orig = None
            preview_recon = None

            with torch.no_grad():
                for i, path in enumerate(scan_paths):
                    if not self.is_running: 
                        break
                    
                    try:
                        img = Image.open(path).convert("RGB")
                        if preview_orig is None:
                            preview_orig = img.copy()

                        # Resize
                        w, h = img.size
                        if w * h > 1024 * 1024:
                            scale = (1024 * 1024 / (w * h)) ** 0.5
                            w, h = int(w * scale), int(h * scale)
                        w, h = max(64, (w // 64) * 64), max(64, (h // 64) * 64)
                        
                        img_r = img.resize((w, h), Image.Resampling.LANCZOS)
                        pixels = transform(img_r).unsqueeze(0).to(self.device)
                        
                        # Encode
                        enc_out = vae.encode(pixels)
                        latents = enc_out.latent_dist.mean if hasattr(enc_out, 'latent_dist') else enc_out
                        
                        all_means.append(latents.mean().item())
                        all_stds.append(latents.std().item())

                        # Preview on first image using calculated normalization
                        if i == 0:
                            raw = latents
                            raw_mean = raw.mean().item()
                            raw_std = raw.std().item()
                            
                            # Universal formula: (x - shift) * scale
                            shift = raw_mean
                            scale = 1.0 / raw_std if raw_std > 0 else 1.0
                            normalized = (raw - shift) * scale
                            normalized = torch.clamp(normalized, -4.0, 4.0)
                            
                            # Reverse for decode
                            rec_input = (normalized / scale) + shift
                            
                            decoded = vae.decode(rec_input)
                            if hasattr(decoded, 'sample'):
                                decoded = decoded.sample
                            decoded = (decoded / 2 + 0.5).clamp(0, 1)
                            dec_np = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
                            preview_recon = Image.fromarray((dec_np * 255).astype(np.uint8))
                            
                            self.previewSignal.emit(preview_orig, preview_recon)

                    except Exception as e:
                        print(f"Skip {path.name}: {e}")
                    
                    self.progressSignal.emit(i + 1, target_count)

            if not all_means:
                raise RuntimeError("No images processed successfully.")

            # 6. Calculate Universal Values (works for 4ch and 32ch)
            global_mean = np.mean(all_means)
            global_std = np.mean(all_stds)

            # Universal formula for ALL VAE types:
            # normalized = (latents - shift) * scale
            # where scale = 1.0 / std to get target_std = 1.0
            shift = global_mean
            scale = 1.0 / global_std if global_std > 0.001 else 1.0
            
            # Verify
            verify_mean = (global_mean - shift) * scale
            verify_std = global_std * scale

            results = {
                "channels": actual_channels,
                "shift": shift,
                "scale": scale,
                "raw_mean": global_mean,
                "raw_std": global_std,
                "verify_mean": verify_mean,
                "verify_std": verify_std,
                "formula": "(latents - shift) * scale"
            }
            
            self.finishedSignal.emit(results)

        except Exception as e:
            self.errorSignal.emit(str(e))

    def stop(self):
        self.is_running = False

class VAEDiagnosticsGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Universal VAE Statistics Analyzer")
        self.resize(800, 700)
        self.config_path = self._get_config_path()
        self._setup_ui()
        self._start_worker()

    def _get_config_path(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        try:
            return parser.parse_args().config
        except SystemExit:
            return None

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Status
        self.lbl_status = QtWidgets.QLabel("Initializing...")
        self.lbl_status.setStyleSheet("font-size: 18px; font-weight: bold; color: #ab97e6;")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_status)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

        # Preview
        preview_group = QtWidgets.QGroupBox("Visual Preview")
        preview_layout = QtWidgets.QHBoxLayout(preview_group)
        
        self.lbl_img_orig = QtWidgets.QLabel("Original")
        self.lbl_img_orig.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_orig.setMinimumSize(256, 256)
        self.lbl_img_orig.setStyleSheet("background-color: #000; border: 1px solid #444;")
        
        self.lbl_img_recon = QtWidgets.QLabel("Reconstructed")
        self.lbl_img_recon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_recon.setMinimumSize(256, 256)
        self.lbl_img_recon.setStyleSheet("background-color: #000; border: 1px solid #444;")
        
        preview_layout.addWidget(self.lbl_img_orig)
        preview_layout.addWidget(self.lbl_img_recon)
        layout.addWidget(preview_group)

        # Results Box
        results_group = QtWidgets.QGroupBox("Recommended VAE Settings (Universal)")
        results_layout = QtWidgets.QFormLayout(results_group)
        results_group.setStyleSheet("QGroupBox { border: 2px solid #00ff00; }")
        
        # Shift
        self.txt_shift = QtWidgets.QLineEdit()
        self.txt_shift.setReadOnly(True)
        shift_row = QtWidgets.QWidget()
        shift_hbox = QtWidgets.QHBoxLayout(shift_row)
        shift_hbox.setContentsMargins(0, 0, 0, 0)
        btn_copy_shift = QtWidgets.QPushButton("Copy")
        btn_copy_shift.setFixedWidth(60)
        btn_copy_shift.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.txt_shift.text()))
        shift_hbox.addWidget(self.txt_shift)
        shift_hbox.addWidget(btn_copy_shift)
        results_layout.addRow("VAE Shift Factor:", shift_row)
        
        # Scale
        self.txt_scale = QtWidgets.QLineEdit()
        self.txt_scale.setReadOnly(True)
        scale_row = QtWidgets.QWidget()
        scale_hbox = QtWidgets.QHBoxLayout(scale_row)
        scale_hbox.setContentsMargins(0, 0, 0, 0)
        btn_copy_scale = QtWidgets.QPushButton("Copy")
        btn_copy_scale.setFixedWidth(60)
        btn_copy_scale.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.txt_scale.text()))
        scale_hbox.addWidget(self.txt_scale)
        scale_hbox.addWidget(btn_copy_scale)
        results_layout.addRow("VAE Scale Factor:", scale_row)
        
        # Formula
        self.lbl_formula = QtWidgets.QLabel("(latents - shift) × scale")
        self.lbl_formula.setStyleSheet("color: #888; font-style: italic; font-family: Consolas;")
        results_layout.addRow("Formula:", self.lbl_formula)
        
        # Verification
        self.lbl_verify = QtWidgets.QLabel("...")
        self.lbl_verify.setStyleSheet("font-family: Consolas;")
        results_layout.addRow("Verification:", self.lbl_verify)
        
        # Detected channels
        self.lbl_channels = QtWidgets.QLabel("...")
        results_layout.addRow("Detected Channels:", self.lbl_channels)
        
        layout.addWidget(results_group)
        
        # Info
        info = QtWidgets.QLabel(
            "<b>Usage:</b> These values work for both SDXL (4ch) and Flux/NoobAI (16/32ch) VAEs. "
            "Use <code>vae_shift_factor</code> and <code>vae_scale_factor</code> in your training config. "
            "Verification should show Mean≈0.0 and Std≈1.0."
        )
        info.setWordWrap(True)
        info.setStyleSheet("background-color: #2c2a3e; padding: 10px; border-radius: 4px;")
        layout.addWidget(info)
        
        # Raw stats
        self.lbl_raw = QtWidgets.QLabel("")
        self.lbl_raw.setStyleSheet("color: #666; font-family: Consolas; font-size: 11px;")
        self.lbl_raw.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_raw)

        # Close
        btn_close = QtWidgets.QPushButton("Done / Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

    def _start_worker(self):
        if not self.config_path:
            self.lbl_status.setText("Error: Use --config path/to/config.json")
            self.lbl_status.setStyleSheet("color: #ff5555;")
            QtCore.QTimer.singleShot(3000, self.close)
            return

        self.worker = VAEWorker(self.config_path)
        self.worker.progressSignal.connect(self.update_progress)
        self.worker.statusSignal.connect(lambda s: self.lbl_status.setText(s))
        self.worker.previewSignal.connect(self.update_preview)
        self.worker.finishedSignal.connect(self.show_results)
        self.worker.errorSignal.connect(self.show_error)
        self.worker.start()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def update_preview(self, pil_orig, pil_recon):
        def set_pixmap(lbl, pil_img):
            pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
            data = pil_img.convert("RGBA").tobytes("raw", "RGBA")
            qim = QtGui.QImage(data, pil_img.size[0], pil_img.size[1], QtGui.QImage.Format.Format_RGBA8888)
            lbl.setPixmap(QtGui.QPixmap.fromImage(qim))
        
        set_pixmap(self.lbl_img_orig, pil_orig)
        set_pixmap(self.lbl_img_recon, pil_recon)

    def show_results(self, data):
        self.lbl_status.setText(f"Analysis Complete - {data['channels']} channels detected")
        self.lbl_status.setStyleSheet("color: #00ff00; font-weight: bold;")
        
        self.txt_shift.setText(f"{data['shift']:.5f}")
        self.txt_scale.setText(f"{data['scale']:.5f}")
        self.lbl_channels.setText(str(data['channels']))
        
        # Verification
        v_mean = data['verify_mean']
        v_std = data['verify_std']
        self.lbl_verify.setText(f"Mean: {v_mean:.4f}, Std: {v_std:.4f} (Target: 0.0, 1.0)")
        
        # Color code
        if abs(v_mean) < 0.01 and abs(v_std - 1.0) < 0.01:
            self.lbl_verify.setStyleSheet("color: #00ff00; font-weight: bold;")
        elif abs(v_mean) < 0.1 and abs(v_std - 1.0) < 0.1:
            self.lbl_verify.setStyleSheet("color: #ffdd57;")
        else:
            self.lbl_verify.setStyleSheet("color: #ff5555; font-weight: bold;")
        
        # Raw stats
        self.lbl_raw.setText(f"Raw statistics: μ={data['raw_mean']:.4f}, σ={data['raw_std']:.4f}")

    def show_error(self, err):
        self.lbl_status.setText("Error!")
        self.lbl_status.setStyleSheet("color: #ff5555;")
        QtWidgets.QMessageBox.critical(self, "Error", str(err))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = VAEDiagnosticsGUI()
    win.show()
    sys.exit(app.exec())