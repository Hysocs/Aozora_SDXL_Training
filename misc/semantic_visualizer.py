import sys
import numpy as np
import os
import cv2
import gc
from PIL import Image
import torch
from diffusers import AutoencoderKL

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QDoubleSpinBox, QFormLayout, QGroupBox, QCheckBox,
    QProgressBar, QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# --- IMAGE UTILS ---
def fix_alpha_channel(img):
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

# --- SEMANTIC LOGIC (Tighter Kernels) ---
def generate_character_map(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    np_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    bilateral = cv2.bilateralFilter(np_image_bgr, d=5, sigmaColor=50, sigmaSpace=50)
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    mean_a, mean_b = np.mean(a_channel), np.mean(b_channel)
    saliency_a = np.abs(a_channel.astype(np.float32) - mean_a)
    saliency_b = np.abs(b_channel.astype(np.float32) - mean_b)
    color_saliency = saliency_a + saliency_b
   
    if color_saliency.max() > 1e-6:
        color_saliency_norm = color_saliency / color_saliency.max()
    else:
        color_saliency_norm = np.zeros_like(color_saliency, dtype=np.float32)
   
    final_map = cv2.GaussianBlur(color_saliency_norm, (9, 9), 0)
    return final_map

def generate_detail_map(pil_image):
    np_image_gray = np.array(pil_image.convert("L"))
    
    sobelx = cv2.Sobel(np_image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(np_image_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
   
    if magnitude.max() > 1e-6:
        magnitude_norm = magnitude / magnitude.max()
    else:
        magnitude_norm = np.zeros_like(magnitude, dtype=np.float32)
   
    final_map = cv2.GaussianBlur(magnitude_norm.astype(np.float32), (3, 3), 0)
    return final_map

# --- THREADS ---

class VAELoaderThread(QThread):
    finished = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            try:
                vae = AutoencoderKL.from_single_file(
                    self.path, torch_dtype=dtype
                )
            except Exception as e:
                if "latent_channels" in str(e) or "size mismatch" in str(e):
                    vae = AutoencoderKL.from_single_file(
                        self.path, torch_dtype=dtype, 
                        latent_channels=32, ignore_mismatched_sizes=True
                    )
                else:
                    raise e
            
            vae.enable_tiling()
            vae.enable_slicing()
            vae.eval()
            
            if torch.cuda.is_available():
                vae = vae.to("cuda")
                
            msg = f"Loaded VAE: {vae.config.latent_channels} Ch ({dtype})"
            self.finished.emit(vae, msg)
        except Exception as e:
            self.error.emit(str(e))

class ComputeThread(QThread):
    result_ready = pyqtSignal(object, object) 

    def __init__(self):
        super().__init__()
        self.original_pil = None
        self.char_map_raw = None
        self.detail_map_raw = None
        self.char_weight = 1.0
        self.detail_weight = 1.0
        self.vae = None
        self.use_latent_view = False
        self.smooth_view = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def set_data(self, orig_pil, char_map, detail_map, cw, dw, vae, use_latent_view, smooth_view):
        self.original_pil = orig_pil
        self.char_map_raw = char_map
        self.detail_map_raw = detail_map
        self.char_weight = cw
        self.detail_weight = dw
        self.vae = vae
        self.use_latent_view = use_latent_view
        self.smooth_view = smooth_view

    def run(self):
        if self.original_pil is None: return

        combined = (self.char_map_raw * self.char_weight) + (self.detail_map_raw * self.detail_weight)
        combined = np.clip(combined, 0.0, 10.0)

        final_bg = self.original_pil
        final_map = combined
        
        MAX_RES = 2048

        if self.use_latent_view:
            w, h = self.original_pil.size
            if w > MAX_RES or h > MAX_RES:
                scale = min(MAX_RES/w, MAX_RES/h)
                w = int(w * scale)
                h = int(h * scale)

            w = (w // 64) * 64
            h = (h // 64) * 64
            
            latent_w = max(1, w // 8)
            latent_h = max(1, h // 8)
            
            if self.vae:
                try:
                    img_resized = self.original_pil.resize((w, h), Image.Resampling.LANCZOS)
                    tensor = torch.from_numpy(np.array(img_resized)).to(dtype=self.dtype)
                    tensor = tensor / 127.5 - 1.0
                    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        latents = self.vae.encode(tensor).latent_dist.sample()
                        decoded = self.vae.decode(latents).sample
                        
                    decoded_cpu = decoded.cpu()
                    del tensor, latents, decoded
                    torch.cuda.empty_cache()
                        
                    decoded_cpu = (decoded_cpu / 2 + 0.5).clamp(0, 1)
                    decoded_np = decoded_cpu.permute(0, 2, 3, 1).numpy()[0]
                    final_bg = Image.fromarray((decoded_np * 255).astype(np.uint8))
                except Exception as e:
                    final_bg = self.original_pil.resize((w, h))
                    torch.cuda.empty_cache()
            else:
                temp = self.original_pil.resize((latent_w, latent_h), Image.Resampling.BILINEAR)
                final_bg = temp.resize((w, h), Image.Resampling.NEAREST)

            map_pil = Image.fromarray(combined, mode='F')
            map_latent = map_pil.resize((latent_w, latent_h), Image.Resampling.BILINEAR)
            
            if self.smooth_view:
                map_vis = map_latent.resize((w, h), Image.Resampling.BICUBIC)
            else:
                map_vis = map_latent.resize((w, h), Image.Resampling.NEAREST)
                
            final_map = np.array(map_vis)
            
        else:
            if combined.shape[:2][::-1] != self.original_pil.size:
                map_pil = Image.fromarray(combined, mode='F')
                map_vis = map_pil.resize(self.original_pil.size, Image.Resampling.BILINEAR)
                final_map = np.array(map_vis)
        
        gc.collect()
        self.result_ready.emit(final_bg, final_map)


class SemanticVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Semantic Visualizer - Educational Edition')
        self.setGeometry(100, 100, 1400, 900)

        # State
        self.original_image = None
        self.vae_model = None
        self.char_map_raw = None
        self.detail_map_raw = None
        self.vae_reconstructed = None  # Store for MSE calc
        self.bg_image = None
        self.weight_map = None
        self.cbar = None
        
        self._needs_recompute = False

        self.compute_thread = ComputeThread()
        self.compute_thread.result_ready.connect(self._on_compute_finished)
        self.vae_thread = None

        self._setup_ui()
        self._create_colormap()

    def _create_colormap(self):
        colors = [
            (0.0, 0.0, 0.5),    # Dark Blue (0.0)
            (0.0, 0.0, 1.0),    # Blue
            (0.0, 1.0, 1.0),    # Cyan
            (1.0, 1.0, 0.0),    # Yellow
            (1.0, 0.0, 0.0),    # Red (High)
        ]
        self.heatmap_cmap = LinearSegmentedColormap.from_list("custom_heat", colors)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # --- CONTROLS ---
        controls_panel = QGroupBox("Configuration")
        controls_panel.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls_panel)

        # Model
        model_group = QGroupBox("VAE / Model")
        model_layout = QVBoxLayout(model_group)
        self.btn_load_vae = QPushButton("Load Model/VAE...")
        self.btn_load_vae.clicked.connect(self._load_vae_dialog)
        self.lbl_vae_status = QLabel("No Model Loaded")
        self.lbl_vae_status.setStyleSheet("color: gray;")
        model_layout.addWidget(self.btn_load_vae)
        model_layout.addWidget(self.lbl_vae_status)
        controls_layout.addWidget(model_group)

        # Image
        self.btn_load_img = QPushButton("Load Target Image...")
        self.btn_load_img.setStyleSheet("background-color: #6a48d7; color: white; font-weight: bold; padding: 10px;")
        self.btn_load_img.clicked.connect(self._load_image)
        controls_layout.addWidget(self.btn_load_img)
        self.lbl_filename = QLabel("No image loaded.")
        controls_layout.addWidget(self.lbl_filename)

        # View Mode
        view_group = QGroupBox("View Settings")
        view_layout = QVBoxLayout(view_group)
        
        self.chk_latent_view = QCheckBox("Simulate Latent Size (1/8th)")
        self.chk_latent_view.setChecked(True)
        self.chk_latent_view.toggled.connect(self._trigger_update)
        view_layout.addWidget(self.chk_latent_view)
        
        self.chk_smooth_vis = QCheckBox("Smooth Heatmap (No Grid)")
        self.chk_smooth_vis.setChecked(False)
        self.chk_smooth_vis.setToolTip("Interpolates the grid to look like a smooth heatmap instead of blocks")
        self.chk_smooth_vis.toggled.connect(self._trigger_update)
        view_layout.addWidget(self.chk_smooth_vis)
        
        self.chk_show_bg = QCheckBox("Show Background Image")
        self.chk_show_bg.setChecked(True)
        self.chk_show_bg.toggled.connect(self._render)
        view_layout.addWidget(self.chk_show_bg)

        self.chk_autoscale = QCheckBox("Auto-Scale Intensity")
        self.chk_autoscale.setChecked(False)
        self.chk_autoscale.toggled.connect(self._render)
        view_layout.addWidget(self.chk_autoscale)
        
        controls_layout.addWidget(view_group)

        # Weights
        form_layout = QFormLayout()
        self.slider_char, self.spin_char = self._create_control(0.0, 5.0, 1.0, 100)
        form_layout.addRow("Char Weight:", self.spin_char)
        form_layout.addRow(self.slider_char)

        self.slider_detail, self.spin_detail = self._create_control(0.0, 5.0, 1.0, 100)
        form_layout.addRow("Detail Weight:", self.spin_detail)
        form_layout.addRow(self.slider_detail)
        controls_layout.addLayout(form_layout)
        
        # Export Button (New)
        self.btn_export = QPushButton("📊 Export Educational Grid")
        self.btn_export.setStyleSheet("""
            background-color: #2E86AB; 
            color: white; 
            font-weight: bold; 
            padding: 12px;
            border-radius: 5px;
        """)
        self.btn_export.clicked.connect(self._export_educational_grid)
        self.btn_export.setEnabled(False)
        controls_layout.addWidget(self.btn_export)
        
        btn_clear = QPushButton("Clear VRAM")
        btn_clear.clicked.connect(self._force_clear_vram)
        controls_layout.addWidget(btn_clear)

        controls_layout.addStretch()
        main_layout.addWidget(controls_panel)

        # --- DISPLAY ---
        display_panel = QVBoxLayout()
        self.figure = Figure(figsize=(10, 10), dpi=100)
        self.figure.patch.set_facecolor('#f0f0f0')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.figure.tight_layout()
        display_panel.addWidget(self.canvas)
        main_layout.addLayout(display_panel, 1)

    def _create_control(self, min_val, max_val, default, divider):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(min_val * divider), int(max_val * divider))
        slider.setValue(int(default * divider))
        
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(0.1)
        spin.setValue(default)
        spin.setDecimals(2)
        
        slider.valueChanged.connect(lambda v: spin.setValue(v / divider))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * divider)))
        spin.valueChanged.connect(self._trigger_update)
        return slider, spin

    def _force_clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_vae_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Files (*.safetensors *.pt)")
        if not path: return
        self.btn_load_vae.setEnabled(False)
        self.lbl_vae_status.setText("Loading...")
        self.vae_thread = VAELoaderThread(path)
        self.vae_thread.finished.connect(self._on_vae_loaded)
        self.vae_thread.error.connect(self._on_vae_error)
        self.vae_thread.start()

    def _on_vae_loaded(self, vae, msg):
        self.vae_model = vae
        self.btn_load_vae.setEnabled(True)
        self.lbl_vae_status.setText(f"✔ {msg}")
        self._trigger_update()

    def _on_vae_error(self, err):
        self.btn_load_vae.setEnabled(True)
        self.lbl_vae_status.setText(f"Error")
        QMessageBox.critical(self, "Error", err)

    def _load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.webp)")
        if not filepath: return
        try:
            pil = Image.open(filepath)
            self.original_image = fix_alpha_channel(pil)
            self.lbl_filename.setText(f"File: {filepath.split('/')[-1]}")
            self.char_map_raw = generate_character_map(self.original_image)
            self.detail_map_raw = generate_detail_map(self.original_image)
            self.btn_export.setEnabled(True)
            self._trigger_update()
        except Exception as e:
            self.lbl_filename.setText(f"Error: {e}")

    def _trigger_update(self):
        self._needs_recompute = True
        self._try_start_thread()

    def _try_start_thread(self):
        if self.compute_thread.isRunning():
            return
        
        if self._needs_recompute and self.original_image is not None:
            self._needs_recompute = False
            self.compute_thread.set_data(
                self.original_image,
                self.char_map_raw,
                self.detail_map_raw,
                self.spin_char.value(),
                self.spin_detail.value(),
                self.vae_model,
                self.chk_latent_view.isChecked(),
                self.chk_smooth_vis.isChecked()
            )
            self.compute_thread.start()

    def _on_compute_finished(self, bg_img, weight_map):
        self.bg_image = bg_img
        self.weight_map = weight_map
        self._render()
        if self._needs_recompute:
            self._try_start_thread()

    def _render(self):
        if self.bg_image is None or self.weight_map is None: return
        
        self.ax.clear()
        self.ax.axis('off')
        
        if self.chk_show_bg.isChecked():
            self.ax.imshow(self.bg_image)
            bg_alpha = 0.55
        else:
            black_bg = np.zeros_like(np.array(self.bg_image))
            self.ax.imshow(black_bg)
            bg_alpha = 1.0
        
        mx = np.max(self.weight_map)
        avg = np.mean(self.weight_map)
        
        v_max = max(0.1, mx) if self.chk_autoscale.isChecked() else 3.0
            
        heatmap = self.ax.imshow(
            self.weight_map,
            cmap=self.heatmap_cmap,
            alpha=bg_alpha,
            vmin=0.0, vmax=v_max
        )
        
        if self.cbar:
            self.cbar.update_normal(heatmap)
        else:
            self.cbar = self.figure.colorbar(heatmap, ax=self.ax, fraction=0.046, pad=0.04)
        
        self.cbar.set_label("Semantic Score")

        title = f"Avg: {avg:.2f} | Max: {mx:.2f}"
        if self.chk_smooth_vis.isChecked():
            title += " (Smooth Heatmap)"
        else:
            title += " (Exact Latent Grid)"
            
        self.ax.set_title(title)
        self.canvas.draw()

    def _export_educational_grid(self):
        """Unified colormap starting at 1.0× (blue) - shows base + semantic boost"""
        if self.original_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            orig_np = np.array(self.original_image)
            h, w = orig_np.shape[:2]
            target_h = 512
            scale = target_h / h
            target_w = int(w * scale)
            
            def resize_arr(img):
                if isinstance(img, Image.Image):
                    return np.array(img.resize((target_w, target_h), Image.Resampling.LANCZOS))
                return cv2.resize(img, (target_w, target_h))
            
            orig_arr = resize_arr(self.original_image)
            
            if self.vae_model and self.bg_image is not None:
                vae_arr = resize_arr(self.bg_image)
            else:
                vae_arr = orig_arr.copy()
            
            char_r = resize_arr(self.char_map_raw)
            detail_r = resize_arr(self.detail_map_raw)
            
            # Semantic weights are ADDED to base 1.0×
            def get_total_weight(cw, dw):
                semantic_boost = char_r * cw + detail_r * dw
                return 1.0 + semantic_boost
            
            # MSE is uniform 1.0× (baseline, no boost)
            mse_map = np.ones_like(char_r) * 1.0
            
            # Scale: 1.0× (blue) to 3.5× (red)
            vmin, vmax = 1.0, 3.5
            
            unified_cmap = self.heatmap_cmap
            
            def apply_cmap(data):
                normalized = (data - vmin) / (vmax - vmin)
                rgba = unified_cmap(np.clip(normalized, 0, 1))
                return (rgba[:, :, :3] * 255).astype(np.uint8)
            
            def blend(base, weight_map):
                heat = apply_cmap(weight_map)
                return (base * 0.3 + heat * 0.7).astype(np.uint8)
            
            # Build rows
            row0 = np.concatenate([
                orig_arr,
                blend(orig_arr, mse_map),
                apply_cmap(mse_map)
            ], axis=1)
            
            sem_std = get_total_weight(1.0, 1.0)
            row1 = np.concatenate([
                vae_arr,
                blend(vae_arr, sem_std),
                apply_cmap(sem_std)
            ], axis=1)
            
            sem_high = get_total_weight(2.5, 2.5)
            row2 = np.concatenate([
                vae_arr,
                blend(vae_arr, sem_high),
                apply_cmap(sem_high)
            ], axis=1)
            
            full_grid = np.concatenate([row0, row1, row2], axis=0)
            grid_h, grid_w = full_grid.shape[:2]
            aspect = grid_w / grid_h
            
            fig_height = 10
            fig_width = fig_height * aspect * 1.12
            
            fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
            
            img_left = 0.12
            img_width = 0.78
            img_bottom = 0.08
            img_height = 0.82
            
            ax_img = fig.add_axes([img_left, img_bottom, img_width, img_height])
            ax_img.imshow(full_grid)
            ax_img.axis('off')
            
            # Column titles
            col_width = img_width / 3
            titles = ['Base Image', 'Loss Weight (1.0× + Semantic)', 'Weight Map Only']
            
            for i, title in enumerate(titles):
                x_pos = img_left + (i + 0.5) * col_width
                fig.text(x_pos, 0.92, title, 
                        ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color='#2E86AB',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                 edgecolor='#2E86AB', linewidth=1.5))
            
            # Row labels
            row_height = img_height / 3
            row_labels = ['MSE Loss\n(1.0× Base)', 
                         'Semantic +1.0×', 
                         'Semantic +2.5×']
            
            for i, label in enumerate(row_labels):
                y_pos = img_bottom + (2.5 - i) * row_height
                fig.text(img_left - 0.008, y_pos, label, 
                        ha='right', va='center', fontsize=10, fontweight='bold', color='#222',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor='gray', linewidth=0.5))
            
            # Colorbars - scale 1.0× to 3.5×
            cbar_width = 0.025
            cbar_left = img_left + img_width + 0.005
            
            for i in range(3):
                y_bottom = img_bottom + (2 - i) * row_height + 0.02
                y_height = row_height - 0.04
                
                ax_cbar = fig.add_axes([cbar_left, y_bottom, cbar_width, y_height])
                
                # FIX: Create gradient image properly (H x W x 3)
                gradient = np.linspace(0, 1, 256)
                # Get RGB values from colormap (256, 4) -> take only RGB
                colors = unified_cmap(gradient)[:, :3]  # (256, 3)
                # Reshape to (256, 1, 3) for vertical bar
                gradient_img = colors.reshape(256, 1, 3)
                
                ax_cbar.imshow(gradient_img, aspect='auto', origin='lower')
                
                ax_cbar.set_xticks([])
                # Ticks at 1.0×, 2.0×, 3.0×, 3.5×
                tick_positions = [0, (2.0-vmin)/(vmax-vmin), (3.0-vmin)/(vmax-vmin), 1.0]
                tick_labels = ['1.0×', '2.0×', '3.0×', '3.5×']
                tick_indices = [int(p*255) for p in tick_positions]
                ax_cbar.set_yticks(tick_indices)
                ax_cbar.set_yticklabels(tick_labels, fontsize=8)
                ax_cbar.tick_params(axis='y', which='both', pad=3)
                ax_cbar.yaxis.tick_right()
                
                if i == 1:
                    label_x = cbar_left + cbar_width + 0.01
                    label_y = y_bottom + y_height / 2
                    fig.text(label_x, label_y, 'Total\nLoss\nWeight', 
                            ha='left', va='center', fontsize=9, fontweight='bold',
                            linespacing=0.9)
            
            fig.suptitle('Loss Comparison: Base 1.0× + Semantic (Blue=1.0, Red=3.0)', 
                        fontsize=12, fontweight='bold', y=0.98)
            
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Grid", "loss_comparison.png", "PNG (*.png)")
            if save_path:
                plt.savefig(save_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.05)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            import traceback
            traceback.print_exc()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SemanticVisualizer()
    window.show()
    sys.exit(app.exec())