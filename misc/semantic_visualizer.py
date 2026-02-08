import sys
import numpy as np
import os
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the one containing misc and tools)
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.insert(0, parent_dir)

import torch

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSlider, QDoubleSpinBox, QFormLayout, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

try:
    # This import will now work correctly
    from tools.semantic import generate_character_map, generate_detail_map
except ImportError:
    print("CRITICAL ERROR: Could not find 'tools/semantic.py'.")
    sys.exit(1)

def fix_alpha_channel(img):
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


class ComputeThread(QThread):
    result_ready = pyqtSignal(object, object)
    
    def __init__(self):
        super().__init__()
        self.char_map = None
        self.detail_map = None
        self.char_weight = 1.0
        self.detail_weight = 1.0
        self.original_image = None
        
    def set_data(self, char_map, detail_map, char_weight, detail_weight, original_image):
        self.char_map = char_map
        self.detail_map = detail_map
        self.char_weight = char_weight
        self.detail_weight = detail_weight
        self.original_image = original_image
        
# GUI ComputeThread
    def run(self):
        char_np = np.array(self.char_map).astype(np.float32)
        detail_np = np.array(self.detail_map).astype(np.float32)
        
        # Asymmetric gamma: boost character mids, keep detail sharp
        char_boosted = np.power(char_np, 0.7) * self.char_weight
        detail_scaled = detail_np * self.detail_weight  # gamma 1.0 = linear
        
        # Soft max with balanced sharpness
        combined = np.logaddexp(char_boosted * 4, detail_scaled * 4) / 4
        combined = np.clip(combined, 0, 1)
        
        # Final gamma for smooth falloff
        weight_map = np.power(combined, 0.8)
        modulation_map = 1.0 + weight_map
        
        self.result_ready.emit(modulation_map, weight_map)


class SemanticVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Semantic Loss Visualizer (Soft Max)')
        self.setGeometry(100, 100, 1200, 800)

        self.original_image = None
        self.char_map_pil = None
        self.detail_map_pil = None
        self.cbar = None
        self.current_modulation_map = None
        self.current_weight_map = None
        
        self.compute_thread = ComputeThread()
        self.compute_thread.result_ready.connect(self._on_compute_finished)
        self._pending_update = False

        self._setup_ui()
        self._create_colormap()

    def _create_colormap(self):
        colors = [
            (0.0, 0.0, 1.0),  # Blue (low)
            (0.85, 0.85, 1.0),  # Light blue
            (1.0, 1.0, 1.0),  # White
            (1.0, 0.85, 0.85),  # Light red
            (1.0, 0.0, 0.0),  # Red (high)
        ]
        self.red_blue_cmap = LinearSegmentedColormap.from_list("red_blue", colors)
        
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        controls_panel = QGroupBox("Soft Max Semantic Controls")
        controls_panel.setFixedWidth(320)
        controls_layout = QVBoxLayout(controls_panel)

        self.btn_load = QPushButton("Load Image...")
        self.btn_load.clicked.connect(self._load_image)
        controls_layout.addWidget(self.btn_load)
        
        self.lbl_filename = QLabel("No image loaded.")
        self.lbl_filename.setWordWrap(True)
        controls_layout.addWidget(self.lbl_filename)

        form_layout = QFormLayout()
        form_layout.setContentsMargins(10, 15, 10, 10)
        form_layout.setSpacing(12)

        # Character Weight (0-2.0)
        self.slider_char, self.spin_char = self._create_control(
            0.0, 2.0, 1.0, 100
        )
        form_layout.addRow("Character Weight:", self.spin_char)
        form_layout.addRow(self.slider_char)

        # Detail Weight (0-2.0)
        self.slider_detail, self.spin_detail = self._create_control(
            0.0, 2.0, 1.0, 100
        )
        form_layout.addRow("Detail Weight:", self.spin_detail)
        form_layout.addRow(self.slider_detail)

        controls_layout.addLayout(form_layout)
        
        # Info text
        info = QLabel(
            "<small><b>Soft Max Formula:</b><br>"
            "combined = log(exp(char×4) + exp(detail×4)) / 4<br><br>"
            "<b>Character Weight:</b> Blob/region detection strength<br>"
            "<b>Detail Weight:</b> Lineart/edge detection strength<br><br>"
            "Red = High loss multiplier | Blue = Base loss (1.0x)</small>"
        )
        info.setWordWrap(True)
        controls_layout.addWidget(info)
        
        # Presets
        preset_layout = QHBoxLayout()
        for name, cw, dw in [("Balanced", 1.0, 1.0), ("Lines", 0.3, 1.5), ("Char", 1.5, 0.3)]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, c=cw, d=dw: self._set_preset(c, d))
            preset_layout.addWidget(btn)
        controls_layout.addLayout(preset_layout)
        
        controls_layout.addStretch()
        main_layout.addWidget(controls_panel)

        # Display
        display_panel = QVBoxLayout()
        self.figure = Figure(figsize=(9, 9), dpi=100)
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
        spin.setSingleStep(0.05)
        spin.setValue(default)
        spin.setDecimals(2)
        
        slider.valueChanged.connect(lambda v: spin.setValue(v / divider))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * divider)))
        spin.valueChanged.connect(self._trigger_update)
        
        return slider, spin

    def _set_preset(self, cw, dw):
        self.spin_char.setValue(cw)
        self.spin_detail.setValue(dw)

    def _trigger_update(self):
        if not self._pending_update:
            self._pending_update = True
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(30, self._debounced_compute)

    def _debounced_compute(self):
        self._pending_update = False
        self._update_async()

    def _load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not filepath:
            return
        try:
            pil_image = Image.open(filepath)
            self.original_image = fix_alpha_channel(pil_image)
            self.lbl_filename.setText(f"<b>File:</b> {filepath.split('/')[-1]}")
            self.char_map_pil = generate_character_map(self.original_image)
            self.detail_map_pil = generate_detail_map(self.original_image)
            self._update_async()
        except Exception as e:
            self.lbl_filename.setText(f"<font color='red'>Error: {e}</font>")

    def _update_async(self):
        if self.original_image is None:
            return
        self.compute_thread.set_data(
            self.char_map_pil, 
            self.detail_map_pil,
            self.spin_char.value(),
            self.spin_detail.value(),
            self.original_image
        )
        if not self.compute_thread.isRunning():
            self.compute_thread.start()

    def _on_compute_finished(self, modulation_map, weight_map):
        self.current_modulation_map = modulation_map
        self.current_weight_map = weight_map
        self._render()

    def _render(self):
        if self.current_modulation_map is None:
            return
            
        self.ax.clear()
        self.ax.imshow(self.original_image)
        
        # Fixed 60% opacity for clean visualization
        heatmap = self.ax.imshow(
            self.current_weight_map,
            cmap=self.red_blue_cmap,
            alpha=1.0,
            interpolation='bilinear',
            vmin=0.0, vmax=1.0
        )
        
        if self.cbar:
            self.cbar.update_normal(heatmap)
        else:
            self.cbar = self.figure.colorbar(heatmap, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.set_label("Loss Weight")
        
        min_m = self.current_modulation_map.min()
        max_m = self.current_modulation_map.max()
        self.ax.set_title(f"Soft Max | Multiplier: {min_m:.2f}x - {max_m:.2f}x")
        self.ax.axis('off')
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SemanticVisualizer()
    window.show()
    sys.exit(app.exec())