import json
import os
import re
from PyQt6 import QtWidgets, QtCore, QtGui
import subprocess
import threading
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
import config as default_config
import copy
import sys
from pathlib import Path
import random
import shutil
import math # <-- Required for the new logarithmic calculations

STYLESHEET = """
QWidget {
    background-color: #2c2a3e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Calibri', 'Helvetica Neue', sans-serif;
    font-size: 15px;
}
TrainingGUI {
    border: 2px solid #1a1926;
    border-radius: 8px;
}
#TitleLabel {
    color: #ab97e6;
    font-size: 28px;
    font-weight: bold;
    padding: 15px;
    border-bottom: 2px solid #4a4668;
}
QGroupBox {
    border: 1px solid #4a4668;
    border-radius: 8px;
    margin-top: 20px;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 10px;
    background-color: #383552;
    color: #e0e0e0;
    font-weight: bold;
    border-radius: 4px;
}
QPushButton {
    background-color: transparent;
    border: 2px solid #ab97e6;
    color: #ab97e6;
    padding: 10px 15px;
    min-height: 32px;
    max-height: 32px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover { background-color: #383552; color: #ffffff; }
QPushButton:pressed { background-color: #4a4668; }
QPushButton:disabled { color: #5c5a70; border-color: #5c5a70; background-color: transparent; }
#StartButton { background-color: #6a48d7; border-color: #6a48d7; color: #ffffff; }
#StartButton:hover { background-color: #7e5be0; border-color: #7e5be0; }
#StopButton { background-color: #e53935; border-color: #e53935; color: #ffffff; }
#StopButton:hover { background-color: #f44336; border-color: #f44336; }
QLineEdit {
    background-color: #1a1926;
    border: 1px solid #4a4668;
    padding: 6px;
    color: #e0e0e0;
    border-radius: 4px;
}
QLineEdit:focus { border: 1px solid #ab97e6; }
QLineEdit:disabled {
    background-color: #242233;
    color: #7a788c;
    border: 1px solid #383552;
}
#ParamInfoLabel {
    background-color: #1a1926;
    color: #ab97e6;
    font-weight: bold;
    font-size: 14px;
    border: 1px solid #4a4668;
    border-radius: 4px;
    padding: 6px;
}
QTextEdit {
    background-color: #1a1926;
    border: 1px solid #4a4668;
    color: #e0e0e0;
    font-family: 'Consolas', 'Courier New', monospace;
    border-radius: 4px;
    padding: 5px;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #ab97e6;
    background-color: #2c2a3e;
    border-radius: 4px;
}
QCheckBox::indicator:checked {
    background-color: #ab97e6;
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSIjMmMyYTNlIiBkPSJNOSAxNi4xN0w0LjgzIDEybC0xLjQyIDEuNDFNOSAxOUwyMSA3bC0xLjQxLTEuNDF6Ii8+PC9zdmc+");
}
QCheckBox::indicator:disabled {
    border: 1px solid #5c5a70;
    background-color: #383552;
}
QComboBox { background-color: #1a1926; border: 1px solid #4a4668; padding: 6px; border-radius: 4px; min-height: 32px; max-height: 32px; }
QComboBox:on { border: 1px solid #ab97e6; }
QComboBox::drop-down { border-left: 1px solid #4a4668; }
QComboBox QAbstractItemView { background-color: #383552; border: 1px solid #ab97e6; selection-background-color: #ab97e6; selection-color: #1a1926; }
QTabWidget::pane { border: 1px solid #4a4668; border-top: none; }
QTabBar::tab { background: #2c2a3e; border: 1px solid #4a4668; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 10px 20px; color: #e0e0e0; font-weight: bold; min-height: 40px; }
QTabBar::tab:selected { background: #383552; color: #ffffff; border-bottom: 3px solid #ab97e6; }
QTabBar::tab:!selected:hover { background: #4a4668; }
QScrollArea { border: none; }
"""

class SubOptionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.form_layout = QtWidgets.QFormLayout(self)
        self.form_layout.setContentsMargins(25, 10, 5, 10)
        self.form_layout.setSpacing(10)
        self.form_layout.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.line_color = QtGui.QColor("#6a48d7")
    def addRow(self, label_or_widget, widget=None):
        self.form_layout.addRow(label_or_widget, widget)
    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(self.line_color)
        pen.setWidth(2)
        painter.setPen(pen)
        line_x = 12
        painter.drawLine(line_x, 5, line_x, self.height() - 5)

class LRCurveWidget(QtWidgets.QWidget):
    pointsChanged = QtCore.pyqtSignal(list)
    selectionChanged = QtCore.pyqtSignal(int)

    # This is used ONLY when min_lr_bound is 0 to prevent log(0) errors.
    LOG_FLOOR_DIVISOR = 10000.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self._points = []
        self._visual_points = []
        self.max_steps = 10000
        self.min_lr_bound = 0.0
        self.max_lr_bound = 1.0e-6
        self.epoch_data = []
        
        self.padding = {'top': 40, 'bottom': 60, 'left': 80, 'right': 20}
        self.point_radius = 8
        self._dragging_point_index = -1
        self._selected_point_index = -1
        
        self.bg_color = QtGui.QColor("#1a1926")
        self.grid_color = QtGui.QColor("#4a4668")
        self.epoch_grid_color = QtGui.QColor("#5c5a70")
        self.line_color = QtGui.QColor("#ab97e6")
        self.point_color = QtGui.QColor("#ffffff")
        self.point_fill_color = QtGui.QColor("#6a48d7")
        self.selected_point_color = QtGui.QColor("#ffdd57")
        self.text_color = QtGui.QColor("#e0e0e0")

        self.setMouseTracking(True)
        self.setToolTip("Click to select a point. Drag points to shape the curve.")

    def set_epoch_data(self, epoch_data):
        self.epoch_data = epoch_data
        self.update()

    def set_bounds(self, max_steps, min_lr, max_lr):
        self.max_steps = max_steps if max_steps > 0 else 1
        self.min_lr_bound = min_lr
        self.max_lr_bound = max_lr if max_lr > min_lr else min_lr + 1e-9
        self._update_visual_points()
        self.update()

    def set_points(self, points):
        self._points = sorted(points, key=lambda p: p[0])
        self._update_visual_points()
        self.update()

    def _update_visual_points(self):
        self._visual_points = [
            [p[0], max(self.min_lr_bound, min(self.max_lr_bound, p[1]))] 
            for p in self._points
        ]
        
    def get_points(self):
        return self._points

    def _get_log_range(self):
        """Calculates the log range, correctly handling a min_lr_bound of 0."""
        safe_max_lr = max(self.max_lr_bound, 1e-12)
        log_max = math.log(safe_max_lr)
        
        # Determine the effective minimum for the log calculation
        if self.min_lr_bound > 0:
            effective_min_lr = self.min_lr_bound
        else:
            # If user wants 0, we need a small internal floor for the math
            effective_min_lr = safe_max_lr / self.LOG_FLOOR_DIVISOR
        
        # Clamp to a very small positive number to be safe
        effective_min_lr = max(effective_min_lr, 1e-12)
        log_min = math.log(effective_min_lr)
        
        return log_max, log_min

    def _to_pixel_coords(self, norm_x, abs_lr):
        graph_width = self.width() - self.padding['left'] - self.padding['right']
        graph_height = self.height() - self.padding['top'] - self.padding['bottom']
        
        px = self.padding['left'] + norm_x * graph_width
        
        # If the value is at or below the defined minimum, pin it to the bottom.
        if abs_lr <= self.min_lr_bound:
            py = self.padding['top'] + graph_height
            return QtCore.QPointF(px, py)

        log_max, log_min = self._get_log_range()
        log_range = log_max - log_min

        if log_range <= 0:
            py = self.padding['top'] # Should be at max value
            return QtCore.QPointF(px, py)

        normalized_y = (math.log(abs_lr) - log_min) / log_range
        py = self.padding['top'] + (1 - normalized_y) * graph_height
        return QtCore.QPointF(px, py)

    def _to_data_coords(self, px, py):
        graph_width = self.width() - self.padding['left'] - self.padding['right']
        graph_height = self.height() - self.padding['top'] - self.padding['bottom']
        
        norm_x = (px - self.padding['left']) / graph_width
        # Clamp normalized_y to [0, 1] to prevent values outside the bounds
        clamped_py = max(self.padding['top'], min(py, self.padding['top'] + graph_height))
        normalized_y = 1 - ((clamped_py - self.padding['top']) / graph_height)
        
        log_max, log_min = self._get_log_range()
        log_range = log_max - log_min

        if log_range <= 0:
            abs_lr = self.min_lr_bound
        else:
            log_val = log_min + (normalized_y * log_range)
            abs_lr = math.exp(log_val)
            
        # The final value must always respect the user-defined bounds.
        clamped_lr = max(self.min_lr_bound, min(self.max_lr_bound, abs_lr))
        return max(0.0, min(1.0, norm_x)), clamped_lr

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        graph_rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                                  self.width() - self.padding['left'] - self.padding['right'],
                                  self.height() - self.padding['top'] - self.padding['bottom'])
        self.draw_grid_and_labels(painter, graph_rect)
        self.draw_curve(painter)
        self.draw_points_and_labels(painter)

    def draw_grid_and_labels(self, painter, rect):
        painter.setPen(self.grid_color)
        for i in range(5): # 5 lines for 4 sections
            y = rect.top() + (i / 4.0) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            x = rect.left() + (i / 4.0) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        epoch_pen = QtGui.QPen(self.epoch_grid_color)
        epoch_pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        painter.setPen(epoch_pen)
        for norm_x, step_count in self.epoch_data:
            x = rect.left() + norm_x * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        original_font = self.font()
        font = self.font(); font.setPointSize(10); painter.setFont(font)
        painter.setPen(self.text_color)

        log_max, log_min = self._get_log_range()
        log_range = log_max - log_min
        
        for i in range(5):
            normalized_y = 1.0 - (i / 4.0) # 1.0 for top, 0.0 for bottom
            
            # Anchor top and bottom labels to the exact bounds
            if i == 0:
                lr_val = self.max_lr_bound
            elif i == 4:
                lr_val = self.min_lr_bound
            else:
                if log_range > 0:
                    lr_val = math.exp(log_min + (normalized_y * log_range))
                else: # Fallback for flat range
                    lr_val = self.max_lr_bound

            label = f"{lr_val:.1e}"
            y = rect.top() + (i / 4.0) * rect.height()
            painter.drawText(QtCore.QRect(0, int(y - 10), self.padding['left'] - 5, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, label)
            
            step_val = int(self.max_steps * (i / 4.0))
            label_x = str(step_val)
            x = rect.left() + (i / 4.0) * rect.width()
            painter.drawText(QtCore.QRect(int(x - 50), rect.bottom() + 5, 100, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, label_x)
        
        small_font = self.font()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        for norm_x, step_count in self.epoch_data:
            x = rect.left() + norm_x * rect.width()
            label_rect = QtCore.QRect(int(x - 40), rect.bottom() + 25, 80, 15)
            painter.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(step_count))

        painter.setFont(original_font)
        font.setBold(True); painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 5, 0, 0), QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop, "Learning Rate Schedule")

    def draw_curve(self, painter):
        if not self._visual_points: return
        painter.setPen(QtGui.QPen(self.line_color, 2))
        poly = QtGui.QPolygonF([self._to_pixel_coords(p[0], p[1]) for p in self._visual_points])
        painter.drawPolyline(poly)

    def draw_points_and_labels(self, painter):
        for i, p in enumerate(self._visual_points):
            pixel_pos = self._to_pixel_coords(p[0], p[1])
            is_selected = (i == self._selected_point_index)
            painter.setBrush(self.selected_point_color if is_selected else self.point_fill_color)
            painter.setPen(self.point_color)
            painter.drawEllipse(pixel_pos, self.point_radius, self.point_radius)
            
            original_point = self._points[i]
            step_val = int(original_point[0] * self.max_steps)
            lr_val = original_point[1]
            label = f"({step_val}, {lr_val:.1e})"
            painter.drawText(QtCore.QRectF(pixel_pos.x() - 50, pixel_pos.y() - 30, 100, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, label)
            
    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton: return
        
        new_selection = -1
        for i, p in enumerate(self._visual_points):
            pixel_pos = self._to_pixel_coords(p[0], p[1])
            if (QtCore.QPointF(event.pos()) - pixel_pos).manhattanLength() < self.point_radius * 1.5:
                self._dragging_point_index = i
                new_selection = i
                break
        
        if self._selected_point_index != new_selection:
            self._selected_point_index = new_selection
            self.selectionChanged.emit(self._selected_point_index)
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging_point_index != -1:
            norm_x, abs_lr = self._to_data_coords(event.pos().x(), event.pos().y())
            
            is_endpoint = self._dragging_point_index == 0 or self._dragging_point_index == len(self._points) - 1
            if is_endpoint:
                norm_x = 0.0 if self._dragging_point_index == 0 else 1.0
            else:
                min_x = self._points[self._dragging_point_index - 1][0]
                max_x = self._points[self._dragging_point_index + 1][0]
                norm_x = max(min_x, min(max_x, norm_x))

            self._points[self._dragging_point_index] = [norm_x, abs_lr]
            self._update_visual_points()
            self.pointsChanged.emit(self._points)
            self.update()
        else:
            on_point = any((QtCore.QPointF(event.pos()) - self._to_pixel_coords(p[0], p[1])).manhattanLength() < self.point_radius * 1.5 for p in self._visual_points)
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor if on_point else QtCore.Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton: return
        self._dragging_point_index = -1
        self.set_points(self._points) # Re-sorts and updates visuals
        self.pointsChanged.emit(self._points)

    def add_point(self):
        if len(self._points) < 2: return
        max_gap = 0
        insert_idx = -1
        for i in range(len(self._points) - 1):
            gap = self._points[i+1][0] - self._points[i][0]
            if gap > max_gap:
                max_gap = gap
                insert_idx = i + 1
        
        if insert_idx != -1:
            prev_p = self._points[insert_idx - 1]
            next_p = self._points[insert_idx]
            
            # When inserting a point, do it in log space to make it visually centered
            _, log_min = self._get_log_range()
            # Use max with a small number to avoid log(0) on existing points
            log_prev = math.log(max(prev_p[1], 1e-12))
            log_next = math.log(max(next_p[1], 1e-12))
            
            # Ensure the interpolated value doesn't go below the graph's floor
            new_lr = math.exp(max(log_min, (log_prev + log_next) / 2))
            
            new_p = [(prev_p[0] + next_p[0]) / 2, new_lr]
            self._points.insert(insert_idx, new_p)
            self.set_points(self._points)
            self.pointsChanged.emit(self._points)

    def remove_selected_point(self):
        if self._selected_point_index > 0 and self._selected_point_index < len(self._points) - 1:
            self._points.pop(self._selected_point_index)
            self._selected_point_index = -1
            self.selectionChanged.emit(self._selected_point_index)
            self.set_points(self._points)
            self.pointsChanged.emit(self._points)

class ProcessRunner(QThread):
    logSignal = pyqtSignal(str)
    paramInfoSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(str, bool)
    finishedSignal = pyqtSignal(int)
    errorSignal = pyqtSignal(str)

    def __init__(self, executable, args, working_dir, env=None, creation_flags=0):
        super().__init__()
        self.executable = executable
        self.args = args
        self.working_dir = working_dir
        self.env = env
        self.creation_flags = creation_flags
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                [self.executable] + self.args,
                cwd=self.working_dir,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                creationflags=self.creation_flags if os.name == 'nt' else 0
            )
            self.logSignal.emit(f"INFO: Started subprocess (PID: {self.process.pid})")

            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue
                if "NOTE: Redirects are currently not supported" in line:
                    continue

                if line.startswith("GUI_PARAM_INFO::"):
                    self.paramInfoSignal.emit(line.replace('GUI_PARAM_INFO::', '').strip())
                else:
                    is_progress = '\r' in line or bool(re.match(r'^\s*\d+%\|\S*\|', line))
                    if any(keyword in line.lower() for keyword in ["memory inaccessible", "cuda out of memory", "access violation", "nan/inf"]):
                        self.logSignal.emit(f"*** ERROR DETECTED: {line} ***")
                    else:
                        self.progressSignal.emit(line.split('\r')[-1], is_progress)

            exit_code = self.process.wait()
            self.finishedSignal.emit(exit_code)
        except Exception as e:
            self.errorSignal.emit(f"Subprocess error: {str(e)}")
            self.finishedSignal.emit(-1)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.logSignal.emit("Process stopped.")

class TrainingGUI(QtWidgets.QWidget):
    UI_DEFINITIONS = {
        "SINGLE_FILE_CHECKPOINT_PATH": {"label": "Base Model (.safetensors)", "tooltip": "Path to the base SDXL model.", "widget": "Path", "file_type": "file_safetensors"},
        "OUTPUT_DIR": {"label": "Output Directory", "tooltip": "Folder where checkpoints will be saved.", "widget": "Path", "file_type": "folder"},
        "CACHING_BATCH_SIZE": {"label": "Caching Batch Size", "tooltip": "Adjust based on VRAM (e.g., 2-4).", "widget": "QLineEdit"},
        "NUM_WORKERS": {"label": "Dataloader Workers", "tooltip": "Set to 0 on Windows if you have issues.", "widget": "QLineEdit"},
        "TARGET_PIXEL_AREA": {"label": "Target Pixel Area", "tooltip": "e.g., 1024*1024=1048576. Buckets are resolutions near this total area.", "widget": "QLineEdit"},
        "BUCKET_ASPECT_RATIOS": {"label": "Aspect Ratios (comma-sep)", "tooltip": "e.g., 1.0, 1.5, 0.66. Defines bucket shapes.", "widget": "QLineEdit"},
        "MAX_TRAIN_STEPS": {"label": "Max Training Steps:", "tooltip": "Total number of training steps.", "widget": "QLineEdit"},
        "SAVE_EVERY_N_STEPS": {"label": "Save Every N Steps:", "tooltip": "How often to save a checkpoint.", "widget": "QLineEdit"},
        "GRADIENT_ACCUMULATION_STEPS": {"label": "Gradient Accumulation:", "tooltip": "Simulates a larger batch size.", "widget": "QLineEdit"},
        "MIXED_PRECISION": {"label": "Mixed Precision:", "tooltip": "bfloat16 for modern GPUs, fp16 for older.", "widget": "QComboBox", "options": ["bfloat16", "fp16(broken)"]},
        "SEED": {"label": "Seed:", "tooltip": "Ensures reproducible training.", "widget": "QLineEdit"},
        "RESUME_MODEL_PATH": {"label": "Resume Model:", "tooltip": "The .safetensors checkpoint file.", "widget": "Path", "file_type": "file_safetensors"},
        "RESUME_STATE_PATH": {"label": "Resume State:", "tooltip": "The .pt optimizer state file.", "widget": "Path", "file_type": "file_pt"},
        "CLIP_GRAD_NORM": {"label": "Clip Gradient Norm:", "tooltip": "Helps prevent unstable training. 1.0 is safe.", "widget": "QLineEdit"},
        "LR_GRAPH_MIN": {"label": "Graph Min LR:", "tooltip": "The minimum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "LR_GRAPH_MAX": {"label": "Graph Max LR:", "tooltip": "The maximum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "MIN_SNR_GAMMA": {"label": "Gamma Value:", "tooltip": "Common range is 5.0 to 20.0.", "widget": "QLineEdit"},
        "MIN_SNR_VARIANT": {"label": "Variant:", "tooltip": "Select the Min-SNR weighting variant.", "widget": "QComboBox", "options": ["standard", "corrected", "debiased"]},
        "IP_NOISE_GAMMA": {"label": "Gamma Value:", "tooltip": "Common range is 0.05 to 0.25.", "widget": "QLineEdit"},
        "COND_DROPOUT_PROB": {"label": "Dropout Probability:", "tooltip": "e.g., 0.1 (5-15% common).", "widget": "QLineEdit"},
    }
    
    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA SDXL Trainer (UNet-Only)")
        self.setMinimumSize(QtCore.QSize(1000, 800))
        self.resize(1200, 950)
        self.config_dir = "configs"
        self.widgets = {}
        self.unet_layer_checkboxes = {}
        self.process_runner = None
        self.tabs_loaded = set()
        self.current_config = {}
        self.last_line_is_progress = False
        self.default_config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        self.presets = {}
        self.datasets = []

        self._initialize_configs()
        self._setup_ui()
        self._on_tab_change(0)
        
        if self.config_dropdown.count() > 0:
            self.config_dropdown.setCurrentIndex(0)
            self.load_selected_config(0)
        else:
            self.log("CRITICAL: No configs found or created. Using temporary defaults.")
            self.current_config = copy.deepcopy(self.default_config)
            self._apply_config_to_widgets()

    def paintEvent(self, event: QtGui.QPaintEvent):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)

    def _initialize_configs(self):
        os.makedirs(self.config_dir, exist_ok=True)
        json_files = [f for f in os.listdir(self.config_dir) if f.endswith(".json")]
        if not json_files:
            default_save_path = os.path.join(self.config_dir, "default.json")
            with open(default_save_path, 'w') as f:
                json.dump(self.default_config, f, indent=4)
            self.log("No configs found. Created 'default.json'.")
        self.presets = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.config_dir, filename)
                try:
                    with open(path, 'r') as f:
                        self.presets[name] = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    self.log(f"Warning: Could not load config '{filename}'. Error: {e}")

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        title_label = QtWidgets.QLabel("AOZORA SDXL Trainer")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)
        self.tab_view = QtWidgets.QTabWidget()
        self.tab_view.currentChanged.connect(self._on_tab_change)
        self.main_layout.addWidget(self.tab_view)
        self.tab_view.addTab(QtWidgets.QWidget(), "Dataset")
        self.tab_view.addTab(QtWidgets.QWidget(), "Model && Training Parameters")
        self.tab_view.addTab(QtWidgets.QWidget(), "Advanced")
        self.tab_view.addTab(QtWidgets.QWidget(), "Training Console")
        corner_hbox = QtWidgets.QHBoxLayout()
        corner_hbox.setContentsMargins(10, 5, 10, 5)
        corner_hbox.setSpacing(10)
        self.config_dropdown = QtWidgets.QComboBox()
        if not self.presets:
            self.log("CRITICAL: No presets loaded to populate dropdown.")
        else:
            for name in sorted(self.presets.keys()):
                display = name.replace("_", " ").title()
                self.config_dropdown.addItem(display, name)
        self.config_dropdown.currentIndexChanged.connect(self.load_selected_config)
        corner_hbox.addWidget(self.config_dropdown)
        self.save_button = QtWidgets.QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)
        corner_hbox.addWidget(self.save_button)
        self.save_as_button = QtWidgets.QPushButton("Save As...")
        self.save_as_button.clicked.connect(self.save_as_config)
        corner_hbox.addWidget(self.save_as_button)
        self.restore_button = QtWidgets.QPushButton("↺")
        self.restore_button.setToolTip("Restore Selected Config to Defaults")
        self.restore_button.setFixedHeight(32)
        self.restore_button.clicked.connect(self.restore_defaults)
        corner_hbox.addWidget(self.restore_button)
        corner_widget = QtWidgets.QWidget()
        corner_widget.setLayout(corner_hbox)
        self.tab_view.setCornerWidget(corner_widget, QtCore.Qt.Corner.TopRightCorner)
        self.param_info_label = QtWidgets.QLabel("Parameters: (awaiting training start)")
        self.param_info_label.setObjectName("ParamInfoLabel")
        self.param_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.param_info_label.setContentsMargins(0, 5, 0, 0)
        self.log_textbox = QtWidgets.QTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setMinimumHeight(200)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QtWidgets.QPushButton("Stop Training")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(button_layout)

    def load_selected_config(self, index):
        selected_key = self.config_dropdown.itemData(index) if self.config_dropdown.itemData(index) else self.config_dropdown.itemText(index).replace(" ", "_").lower()
        if selected_key and selected_key in self.presets:
            config = copy.deepcopy(self.default_config)
            config.update(self.presets[selected_key])
            self.current_config = config
            self.log(f"Loaded config: '{selected_key}.json'")
        else:
            self.log(f"Warning: Could not find selected preset '{selected_key}'. Loading hardcoded defaults.")
            self.current_config = copy.deepcopy(self.default_config)
        self._apply_config_to_widgets()

    def _prepare_config_to_save(self):
        config_to_save = {}
        default_map = self.default_config
        for key, live_val in self.current_config.items():
            if live_val is None:
                continue
            default_val = default_map.get(key)
            try:
                if key == "LR_CUSTOM_CURVE":
                    rounded_curve = [[round(p[0], 8), round(p[1], 10)] for p in live_val]
                    config_to_save[key] = rounded_curve
                    continue
                
                if isinstance(live_val, (bool, list)):
                    converted_val = live_val
                elif default_val is not None:
                    if isinstance(default_val, bool):
                        converted_val = str(live_val).strip().lower() in ('true', '1', 't', 'y', 'yes')
                    elif isinstance(default_val, int):
                        converted_val = int(str(live_val).strip()) if str(live_val).strip() else 0
                    elif isinstance(default_val, float):
                        converted_val = float(str(live_val).strip()) if str(live_val).strip() else 0.0
                    elif isinstance(default_val, list):
                        raw_list_str = str(live_val).strip().replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                        converted_val = [float(p.strip()) for p in raw_list_str.split(',') if p.strip()]
                    else:
                        converted_val = str(live_val)
                else:
                    converted_val = str(live_val)
                config_to_save[key] = converted_val
            except (ValueError, TypeError) as e:
                self.log(f"Warning: Could not convert value for '{key}'. Not saved. Error: {e}")
        if hasattr(self, "datasets"):
            config_to_save["INSTANCE_DATASETS"] = [{"path": ds["path"], "repeats": ds["repeats"]} for ds in self.datasets]
            config_to_save.pop("INSTANCE_DATA_DIR", None)
        return config_to_save
    
    def save_config(self):
        config_to_save = self._prepare_config_to_save()
        index = self.config_dropdown.currentIndex()
        if index < 0:
            self.log("Error: No configuration selected to save.")
            return
        selected_key = self.config_dropdown.itemData(index) if self.config_dropdown.itemData(index) else self.config_dropdown.itemText(index).replace(" ", "_").lower()
        save_path = os.path.join(self.config_dir, f"{selected_key}.json")
        try:
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            self.log(f"Successfully saved settings to {os.path.basename(save_path)}")
            self.presets[selected_key] = config_to_save
        except Exception as e:
            self.log(f"CRITICAL ERROR: Could not write to {save_path}. Error: {e}")

    def save_as_config(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset As", "Enter preset name (alphanumeric, underscores):")
        if ok and name:
            if not re.match(r'^[a-zA-Z0-9_]+$', name):
                self.log("Error: Preset name must be alphanumeric with underscores only.")
                return
            save_path = os.path.join(self.config_dir, f"{name}.json")
            if os.path.exists(save_path):
                reply = QtWidgets.QMessageBox.question(self, "Overwrite?", f"Preset '{name}' exists. Overwrite?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
                if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            config_to_save = self._prepare_config_to_save()
            try:
                with open(save_path, 'w') as f:
                    json.dump(config_to_save, f, indent=4)
                self.log(f"Successfully saved preset to {os.path.basename(save_path)}")
                self.presets[name] = config_to_save
                
                self.config_dropdown.blockSignals(True)
                current_text = self.config_dropdown.currentText()
                self.config_dropdown.clear()
                for preset_name in sorted(self.presets.keys()):
                    self.config_dropdown.addItem(preset_name.replace("_", " ").title(), preset_name)
                
                index = self.config_dropdown.findData(name)
                if index != -1: self.config_dropdown.setCurrentIndex(index)
                else: self.config_dropdown.setCurrentText(current_text)
                self.config_dropdown.blockSignals(False)
            except Exception as e:
                self.log(f"Error saving preset: {e}")

    def _on_tab_change(self, index):
        if self.tab_view.widget(index).layout() is not None: return
        tab_widget = self.tab_view.widget(index)
        tab_layout = QtWidgets.QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(0,0,0,0)
        tab_name = self.tab_view.tabText(index)
        
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QtWidgets.QWidget()
        scroll_area.setWidget(content_widget)
        
        if tab_name == "Dataset": 
            self._populate_dataset_tab(content_widget)
            tab_layout.addWidget(scroll_area)
        elif tab_name == "Model && Training Parameters": 
            self._populate_data_model_tab(content_widget)
            tab_layout.addWidget(scroll_area)
        elif tab_name == "Advanced": 
            self._populate_advanced_tab(content_widget)
            tab_layout.addWidget(scroll_area)
        elif tab_name == "Training Console":
            self._populate_console_tab(tab_layout)
        
        self._apply_config_to_widgets()

    def _create_widget(self, key):
        if key not in self.UI_DEFINITIONS: return None, None
        
        definition = self.UI_DEFINITIONS[key]
        label = QtWidgets.QLabel(definition["label"])
        label.setToolTip(definition["tooltip"])
        
        widget_type = definition["widget"]
        widget = None

        if widget_type == "QLineEdit":
            widget = QtWidgets.QLineEdit()
            widget.textChanged.connect(lambda text, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "QComboBox":
            widget = QtWidgets.QComboBox()
            widget.addItems(definition["options"])
            widget.currentTextChanged.connect(lambda text, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "Path":
            container = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout(container); hbox.setContentsMargins(0,0,0,0)
            widget = QtWidgets.QLineEdit()
            browse_btn = QtWidgets.QPushButton("Browse...")
            browse_btn.clicked.connect(lambda: self._browse_path(widget, definition["file_type"]))
            hbox.addWidget(widget, 1); hbox.addWidget(browse_btn)
            widget.textChanged.connect(lambda text, k=key: self._update_config_from_widget(k, widget))
            self.widgets[key] = widget
            return label, container
        
        self.widgets[key] = widget
        return label, widget

    def _populate_dataset_tab(self, parent_widget):
        layout = QtWidgets.QVBoxLayout(parent_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        top_hbox = QtWidgets.QHBoxLayout()
        top_hbox.setSpacing(20)
        
        groups = {
            "Batching & DataLoaders": ["CACHING_BATCH_SIZE", "NUM_WORKERS"],
            "Aspect Ratio Bucketing": ["TARGET_PIXEL_AREA", "BUCKET_ASPECT_RATIOS"]
        }
        for title, keys in groups.items():
            group = QtWidgets.QGroupBox(title)
            form_layout = QtWidgets.QFormLayout(group)
            for key in keys:
                label, widget = self._create_widget(key)
                form_layout.addRow(label, widget)
            top_hbox.addWidget(group)
        layout.addLayout(top_hbox)

        add_button = QtWidgets.QPushButton("Add Dataset Folder")
        add_button.clicked.connect(self.add_dataset_folder)
        layout.addWidget(add_button)
        self.dataset_grid = QtWidgets.QGridLayout()
        layout.addLayout(self.dataset_grid)
        bottom_hbox = QtWidgets.QHBoxLayout()
        self._create_bool_option(
            bottom_hbox, "MIRROR_REPEATS", "Mirror Repeats", "Horizontally mirrors repeated images.", None
        )
        self._create_bool_option(bottom_hbox, "DARKEN_REPEATS", "Increase Contrast on Repeats", "Applies a contrast-enhancing S-curve to repeated images.", None)
        bottom_hbox.addStretch(1)
        bottom_hbox.addWidget(QtWidgets.QLabel("Total Images:"))
        self.total_label = QtWidgets.QLabel("0")
        bottom_hbox.addWidget(self.total_label)
        bottom_hbox.addWidget(QtWidgets.QLabel("With Repeats:"))
        self.total_repeats_label = QtWidgets.QLabel("0")
        bottom_hbox.addWidget(self.total_repeats_label)
        bottom_hbox.addStretch()
        layout.addLayout(bottom_hbox)
        layout.addStretch()

    def _populate_data_model_tab(self, parent_widget):
        layout = QtWidgets.QHBoxLayout(parent_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 5, 15, 15)
        
        left_vbox = QtWidgets.QVBoxLayout()
        right_vbox = QtWidgets.QVBoxLayout()

        path_group = QtWidgets.QGroupBox("File & Directory Paths")
        path_layout = QtWidgets.QFormLayout(path_group)
        for key in ["SINGLE_FILE_CHECKPOINT_PATH", "OUTPUT_DIR"]:
            label, widget = self._create_widget(key); path_layout.addRow(label, widget)
        left_vbox.addWidget(path_group)
        
        core_group = QtWidgets.QGroupBox("Core Training")
        core_layout = QtWidgets.QFormLayout(core_group)
        for key in ["MAX_TRAIN_STEPS", "SAVE_EVERY_N_STEPS", "GRADIENT_ACCUMULATION_STEPS", "MIXED_PRECISION", "SEED"]:
            label, widget = self._create_widget(key); core_layout.addRow(label, widget)
        left_vbox.addWidget(core_group)
        self.widgets["MAX_TRAIN_STEPS"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["GRADIENT_ACCUMULATION_STEPS"].textChanged.connect(self._update_epoch_markers_on_graph)
        
        resume_group = QtWidgets.QGroupBox("Resume From Checkpoint")
        resume_layout = QtWidgets.QVBoxLayout(resume_group)
        self._create_bool_option(resume_layout, "RESUME_TRAINING", "Enable Resuming", "Allows resuming from a checkpoint.", self.toggle_resume_widgets)
        self.resume_sub_widget = SubOptionWidget()
        for key in ["RESUME_MODEL_PATH", "RESUME_STATE_PATH"]:
            label, widget = self._create_widget(key); self.resume_sub_widget.addRow(label, widget)
        resume_layout.addWidget(self.resume_sub_widget)
        left_vbox.addWidget(resume_group)
        left_vbox.addStretch()

        optimizer_group = QtWidgets.QGroupBox("Optimizer")
        optimizer_layout = QtWidgets.QFormLayout(optimizer_group)
        label, widget = self._create_widget("CLIP_GRAD_NORM"); optimizer_layout.addRow(label, widget)
        right_vbox.addWidget(optimizer_group)
        
        lr_group = QtWidgets.QGroupBox("Learning Rate Scheduler")
        lr_layout = QtWidgets.QVBoxLayout(lr_group)
        
        self.lr_curve_widget = LRCurveWidget()
        self.widgets['LR_CUSTOM_CURVE'] = self.lr_curve_widget
        self.lr_curve_widget.pointsChanged.connect(lambda pts: self._update_config_from_widget("LR_CUSTOM_CURVE", self.lr_curve_widget))
        self.lr_curve_widget.selectionChanged.connect(self._update_lr_button_states)
        lr_layout.addWidget(self.lr_curve_widget)
        
        lr_controls_layout = QtWidgets.QHBoxLayout()
        self.add_point_btn = QtWidgets.QPushButton("Add Point"); self.add_point_btn.clicked.connect(self.lr_curve_widget.add_point)
        self.remove_point_btn = QtWidgets.QPushButton("Remove Selected"); self.remove_point_btn.clicked.connect(self.lr_curve_widget.remove_selected_point)
        lr_controls_layout.addWidget(self.add_point_btn)
        lr_controls_layout.addWidget(self.remove_point_btn)
        lr_controls_layout.addStretch()
        lr_layout.addLayout(lr_controls_layout)

        preset_layout = QtWidgets.QHBoxLayout()
        preset_label = QtWidgets.QLabel("<b>Presets:</b>")
        preset_layout.addWidget(preset_label)
        cosine_btn = QtWidgets.QPushButton("Cosine")
        cosine_btn.clicked.connect(self._set_cosine_preset)
        linear_btn = QtWidgets.QPushButton("Linear")
        linear_btn.clicked.connect(self._set_linear_preset)
        constant_btn = QtWidgets.QPushButton("Constant")
        constant_btn.clicked.connect(self._set_constant_preset)
        step_btn = QtWidgets.QPushButton("Step")
        step_btn.clicked.connect(self._set_step_preset)
        preset_layout.addWidget(cosine_btn)
        preset_layout.addWidget(linear_btn)
        preset_layout.addWidget(constant_btn)
        preset_layout.addWidget(step_btn)
        preset_layout.addStretch()
        lr_layout.addLayout(preset_layout)

        graph_bounds_layout = QtWidgets.QFormLayout()
        min_label, min_widget = self._create_widget("LR_GRAPH_MIN")
        max_label, max_widget = self._create_widget("LR_GRAPH_MAX")
        graph_bounds_layout.addRow(min_label, min_widget)
        graph_bounds_layout.addRow(max_label, max_widget)
        lr_layout.addLayout(graph_bounds_layout)
        
        self.widgets["LR_GRAPH_MIN"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["LR_GRAPH_MAX"].textChanged.connect(self._update_and_clamp_lr_graph)
        
        right_vbox.addWidget(lr_group)
        right_vbox.addStretch()
        
        layout.addLayout(left_vbox, 1); layout.addLayout(right_vbox, 1)
        self._update_lr_button_states(-1)

    def _set_cosine_preset(self):
        try:
            max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
            min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except (ValueError, KeyError):
            self.log("ERROR: Min/Max LR must be set to use this preset.")
            return

        # --- Behavior Definitions ---
        warmup_portion = 0.05  # Use the first 5% of steps to warm up to max_lr
        num_curve_points = 15  # Generate 30 points to create a smooth curve

        # --- Point Generation ---
        points = []
        # 1. Start at min_lr and add the warmup point
        points.append([0.0, min_lr])
        points.append([warmup_portion, max_lr])
        
        # 2. Generate the cosine curve points after the warmup
        # The cosine decay happens in the range from warmup_portion to 1.0
        decay_duration = 1.0 - warmup_portion
        for i in range(num_curve_points):
            # 'progress' goes from 0.0 to 1.0, representing how far along the cosine decay we are
            progress = i / (num_curve_points - 1)
            
            # Calculate the X-axis value (the overall training progress)
            norm_x = warmup_portion + progress * decay_duration
            
            # Calculate the LR using the cosine annealing formula
            cosine_val = 0.5 * (1 + math.cos(math.pi * progress))
            lr_val = min_lr + (max_lr - min_lr) * cosine_val
            
            points.append([norm_x, lr_val])

        self.lr_curve_widget.set_points(points)
        self.lr_curve_widget.pointsChanged.emit(points)
        self.log("Applied mathematical Cosine preset with warmup.")

    def _set_linear_preset(self):
        try:
            max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
            min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except (ValueError, KeyError):
            self.log("ERROR: Min/Max LR must be set to use this preset.")
            return

        # A linear schedule is a simple triangle: warmup, then linear decay.
        warmup_portion = 0.05  # Use 5% of steps for warmup

        points = [
            [0.0, min_lr],              # Start at min
            [warmup_portion, max_lr],   # Peak at the end of warmup
            [1.0, min_lr]               # Linearly decay to min
        ]
        
        self.lr_curve_widget.set_points(points)
        self.lr_curve_widget.pointsChanged.emit(points)
        self.log("Applied Linear preset with warmup and decay.")
        
    def _set_constant_preset(self):
        try:
            max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
            min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except (ValueError, KeyError):
            self.log("ERROR: Min/Max LR must be set to use this preset.")
            return

        # Warmup, hold constant at max_lr, then cooldown.
        warmup_portion = 0.05   # 5% warmup
        cooldown_portion = 0.10   # 10% cooldown at the end
        cooldown_start = 1.0 - cooldown_portion

        points = [
            [0.0, min_lr],                  # Start at min
            [warmup_portion, max_lr],       # End of warmup
            [cooldown_start, max_lr],       # Start of cooldown
            [1.0, min_lr]                   # End at min
        ]
        
        self.lr_curve_widget.set_points(points)
        self.lr_curve_widget.pointsChanged.emit(points)
        self.log("Applied Constant preset with warmup and cooldown.")

    def _set_step_preset(self):
        try:
            max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
            min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
            max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except (ValueError, KeyError):
            self.log("ERROR: Min/Max LR and Max Steps must be set to use this preset.")
            return

        if max_steps <= 0:
            self.log("ERROR: Cannot generate preset with 0 or fewer steps.")
            return
        
        # --- BEHAVIOR DEFINITIONS ---
        num_stairs = 4                      # The number of distinct learning rate steps
        stairs_end_portion = 0.5            # Fit all stairs into the first 50% of training
        cooldown_start_portion = 0.9        # Start the final LR decay at 90% of training

        # --- LOGARITHMIC CALCULATIONS ---
        # To make steps visually even, we must work in log space.
        
        # 1. Define the safe logarithmic range, handling min_lr = 0.
        # A very small number is used as a floor to prevent math.log(0) errors.
        safe_min_lr = max(min_lr, 1e-12)
        log_min = math.log(safe_min_lr)
        log_max = math.log(max_lr)

        # 2. The first stair will be at the logarithmic midpoint, which is visually halfway.
        log_start_lr = (log_min + log_max) / 2.0
        
        points = []
        steps_per_stair = (max_steps * stairs_end_portion) / num_stairs
        
        # --- POINT GENERATION ---
        for i in range(num_stairs):
            # 3. Interpolate evenly in LOG space from our start point to the max.
            # The interpolation factor moves from 0 (at stair 0) to 1 (at the last stair).
            interpolation_factor = i / (num_stairs - 1) if num_stairs > 1 else 1.0
            
            # This formula finds the evenly spaced log value for the current stair.
            current_log_lr = log_start_lr + (log_max - log_start_lr) * interpolation_factor
            
            # 4. Convert the calculated log value back to a real LR value.
            current_lr = math.exp(current_log_lr)
            
            start_step = i * steps_per_stair
            end_step = (i + 1) * steps_per_stair
            
            # Add the two points that define the flat top of the stair
            points.append([start_step / max_steps, current_lr])
            points.append([(end_step - 1) / max_steps, current_lr])

        # Anchor the end of the stairs and create the flat top before cooldown
        points.append([stairs_end_portion, max_lr])
        points.append([cooldown_start_portion, max_lr])
        
        # Final decay to the user-defined minimum learning rate
        points.append([1.0, min_lr])

        # --- UPDATE WIDGET ---
        self.lr_curve_widget.set_points(points)
        self.lr_curve_widget.pointsChanged.emit(points)
        self.log(f"Applied mathematical Step preset. Start LR (visual midpoint): {math.exp(log_start_lr):.2e}")

    def _populate_advanced_tab(self, parent_widget):
        main_layout = QtWidgets.QHBoxLayout(parent_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        left_layout = QtWidgets.QVBoxLayout()
        adv_group = QtWidgets.QGroupBox("Advanced (v-prediction)")
        adv_layout = QtWidgets.QVBoxLayout(adv_group)
        self._create_bool_option(adv_layout, "USE_PER_CHANNEL_NOISE", "Use Per-Channel (Color) Noise", "...")
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_RESIDUAL_SHIFTING", "Use Residual Shifting Schedulers", "...")
        self._create_bool_option(adv_layout, "USE_ZERO_TERMINAL_SNR", "Use Zero-Terminal SNR Rescaling", "...")
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_MIN_SNR_GAMMA", "Use Min-SNR Gamma", "...", self.toggle_min_snr_gamma_widget)
        self.min_snr_sub_widget = SubOptionWidget()
        for key in ["MIN_SNR_GAMMA", "MIN_SNR_VARIANT"]:
            label, widget = self._create_widget(key); self.min_snr_sub_widget.addRow(label, widget)
        adv_layout.addWidget(self.min_snr_sub_widget)
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_IP_NOISE_GAMMA", "Use Input Perturbation Noise", "...", self.toggle_ip_noise_gamma_widget)
        self.ip_sub_widget = SubOptionWidget()
        label, widget = self._create_widget("IP_NOISE_GAMMA"); self.ip_sub_widget.addRow(label, widget)
        adv_layout.addWidget(self.ip_sub_widget)
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_COND_DROPOUT", "Use Text Conditioning Dropout", "...", self.toggle_cond_dropout_widget)
        self.cond_sub_widget = SubOptionWidget()
        label, widget = self._create_widget("COND_DROPOUT_PROB"); self.cond_sub_widget.addRow(label, widget)
        adv_layout.addWidget(self.cond_sub_widget)
        left_layout.addWidget(adv_group)
        left_layout.addStretch()
        right_layout = QtWidgets.QVBoxLayout()
        layers_group = QtWidgets.QGroupBox("UNet Layer Targeting")
        layers_layout = QtWidgets.QVBoxLayout(layers_group)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addStretch()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.toggle_all_unet_checkboxes(True))
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.toggle_all_unet_checkboxes(False))
        top_layout.addWidget(select_all_btn)
        top_layout.addWidget(deselect_all_btn)
        layers_layout.addLayout(top_layout)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        layers_layout.addWidget(scroll_area)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        all_unet_targets = {
            "Attention Blocks": {"attn1": "Self-Attention", "attn2": "Cross-Attention", "mid_block.attentions": "Mid-Block Attention", "ff": "Feed-Forward Networks"},
            "Attention Sub-Layers (Advanced)": {"to_q": "Query Projection", "to_k": "Key Projection", "to_v": "Value Projection", "to_out.0": "Output Projection", "proj_in": "Transformer Input Projection", "proj_out": "Transformer Output Projection"},
            "UNet Embeddings (Non-Text)": {"time_embedding": "Time Embedding", "time_emb_proj": "Time Embedding Projection", "add_embedding": "Added Conditional Embedding"},
            "Convolutional & ResNet Layers": {"conv_in": "Input Conv", "conv1": "ResNet Conv1", "conv2": "ResNet Conv2", "conv_shortcut": "ResNet Skip Conv", "downsamplers": "Downsampler Convs", "upsamplers": "Upsampler Convs", "conv_out": "Output Conv"},
            "Normalization Layers (Experimental)": {
                "norm": "Attention GroupNorm", "norm1": "ResNet GroupNorm1", "norm2": "ResNet GroupNorm2",
                "norm3": "Transformer Norm3", "conv_norm_out": "Output GroupNorm"
            },
        }
        for group_name, targets in all_unet_targets.items():
            label = QtWidgets.QLabel(f"<b>{group_name}</b>")
            label.setStyleSheet("margin-top: 15px; border: none; font-size: 16px;")
            scroll_layout.addWidget(label)
            for key, text in targets.items():
                cb = QtWidgets.QCheckBox(f"{text} (keyword: '{key}')")
                cb.setStyleSheet("margin-left: 20px;")
                scroll_layout.addWidget(cb)
                self.unet_layer_checkboxes[key] = cb
                cb.stateChanged.connect(self._update_unet_targets_config)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(layers_group)
        right_layout.addStretch()
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)

    def _populate_console_tab(self, layout):
        layout.setContentsMargins(15, 15, 15, 15)
        param_group = QtWidgets.QGroupBox("Parameter Info")
        param_group_layout = QtWidgets.QVBoxLayout(param_group)
        param_group_layout.setContentsMargins(5, 5, 5, 5)
        self.param_info_label.setWordWrap(True)
        self.param_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        param_group_layout.addWidget(self.param_info_label)
        layout.addWidget(param_group, stretch=0)
        layout.addWidget(self.log_textbox, stretch=1)

    def add_dataset_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not path: return
        exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        images = [p for ext in exts for p in Path(path).rglob(f"*{ext}") if "_truncated" not in p.stem]
        if not images:
            QtWidgets.QMessageBox.warning(self, "No Images", "No valid images found in the folder.")
            return
        image_count = len(images)
        preview_path = random.choice(images)
        self.datasets.append({
            "path": path,
            "image_count": image_count,
            "preview_path": str(preview_path),
            "repeats": 1
        })
        self.current_config["INSTANCE_DATASETS"] = [{"path": ds["path"], "repeats": ds["repeats"]} for ds in self.datasets]
        self.repopulate_dataset_grid()
        self.update_dataset_totals()

    def repopulate_dataset_grid(self):
        for i in reversed(range(self.dataset_grid.count())):
            widget = self.dataset_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        for idx, ds in enumerate(self.datasets):
            preview_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(ds["preview_path"]).scaled(128, 128, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            preview_label.setPixmap(pixmap)
            self.dataset_grid.addWidget(preview_label, idx, 0)
            path_label = QtWidgets.QLabel(ds["path"])
            path_label.setWordWrap(True)
            self.dataset_grid.addWidget(path_label, idx, 1)
            repeats_spin = QtWidgets.QSpinBox()
            repeats_spin.setMinimum(1)
            repeats_spin.setMaximum(10000)
            repeats_spin.setValue(ds["repeats"])
            repeats_spin.valueChanged.connect(lambda v, i=idx: self.update_repeats(i, v))
            self.dataset_grid.addWidget(repeats_spin, idx, 2)
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.clicked.connect(lambda _, i=idx: self.remove_dataset(i))
            self.dataset_grid.addWidget(remove_btn, idx, 3)
            clear_btn = QtWidgets.QPushButton("Clear Cached Latents")
            clear_btn.clicked.connect(lambda _, p=ds["path"]: self.confirm_clear_cache(p))
            self.dataset_grid.addWidget(clear_btn, idx, 4)

    def update_repeats(self, idx, val):
        self.datasets[idx]["repeats"] = val
        self.current_config["INSTANCE_DATASETS"] = [{"path": ds["path"], "repeats": ds["repeats"]} for ds in self.datasets]
        self.update_dataset_totals()

    def remove_dataset(self, idx):
        del self.datasets[idx]
        self.current_config["INSTANCE_DATASETS"] = [{"path": ds["path"], "repeats": ds["repeats"]} for ds in self.datasets]
        self.repopulate_dataset_grid()
        self.update_dataset_totals()

    def update_dataset_totals(self):
        total = sum(ds["image_count"] for ds in self.datasets)
        total_rep = sum(ds["image_count"] * ds["repeats"] for ds in self.datasets)
        self.total_label.setText(str(total))
        self.total_repeats_label.setText(str(total_rep))
        self._update_epoch_markers_on_graph()

    def confirm_clear_cache(self, path):
        reply = QtWidgets.QMessageBox.question(self, "Confirm", "Are you sure you want to delete all cached latents in this dataset? They will be deleted.", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.clear_cached_latents(path)

    def clear_cached_latents(self, path):
        root = Path(path)
        deleted = False
        for cache_dir in list(root.rglob(".precomputed_embeddings_cache")):
            if cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir)
                    self.log(f"Deleted cache directory: {cache_dir}")
                    deleted = True
                except Exception as e:
                    self.log(f"Error deleting {cache_dir}: {e}")
        if not deleted:
            self.log("No cached latent directories found to delete.")

    def _browse_path(self, entry_widget, file_type):
        path = ""
        current_path = os.path.dirname(entry_widget.text()) if entry_widget.text() else ""
        if file_type == "folder": path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", current_path)
        elif file_type == "file_safetensors": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", current_path, "Safetensors Files (*.safetensors)")
        elif file_type == "file_pt": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select State", current_path, "PyTorch State Files (*.pt)")
        if path: entry_widget.setText(path.replace('\\', '/'))

    def _create_bool_option(self, layout, key, label, tooltip_text, command=None):
        checkbox = QtWidgets.QCheckBox(label)
        checkbox.setToolTip(tooltip_text)
        if isinstance(layout, QtWidgets.QVBoxLayout) or isinstance(layout, QtWidgets.QHBoxLayout):
            layout.addWidget(checkbox)
        else:
            layout.addRow(checkbox)
        self.widgets[key] = checkbox
        checkbox.stateChanged.connect(lambda state, k=key: self._update_config_from_widget(k, checkbox))
        if command: checkbox.stateChanged.connect(command)

    def _update_unet_targets_config(self):
        if not self.unet_layer_checkboxes: return
        self.current_config["UNET_TRAIN_TARGETS"] = [k for k, cb in self.unet_layer_checkboxes.items() if cb.isChecked()]
    
    def _update_config_from_widget(self, key, widget):
        if isinstance(widget, QtWidgets.QLineEdit):
            self.current_config[key] = widget.text().strip()
        elif isinstance(widget, QtWidgets.QCheckBox):
            self.current_config[key] = widget.isChecked()
        elif isinstance(widget, QtWidgets.QComboBox):
            self.current_config[key] = widget.currentText()
        elif isinstance(widget, LRCurveWidget):
            self.current_config[key] = widget.get_points()

    def _apply_config_to_widgets(self):
        if self.unet_layer_checkboxes:
            unet_targets = self.current_config.get("UNET_TRAIN_TARGETS", [])
            for key, checkbox in self.unet_layer_checkboxes.items():
                checkbox.setChecked(key in unet_targets)

        for key, widget in self.widgets.items():
            if key in self.current_config:
                if key == "LR_CUSTOM_CURVE": continue
                value = self.current_config.get(key)
                if value is None: continue
                widget.blockSignals(True)
                try:
                    if isinstance(widget, QtWidgets.QLineEdit): widget.setText(", ".join(map(str, value)) if isinstance(value, list) else str(value))
                    elif isinstance(widget, QtWidgets.QCheckBox): widget.setChecked(bool(value))
                    elif isinstance(widget, QtWidgets.QComboBox): widget.setCurrentText(str(value))
                finally:
                    widget.blockSignals(False)
        
        if hasattr(self, 'lr_curve_widget'):
            self._update_and_clamp_lr_graph()
        
        for toggle_func in [self.toggle_min_snr_gamma_widget, self.toggle_resume_widgets, 
                            self.toggle_ip_noise_gamma_widget, self.toggle_cond_dropout_widget]:
            toggle_func()

        if hasattr(self, "dataset_grid"):
            self.datasets = []
            datasets_config = self.current_config.get("INSTANCE_DATASETS", [])
            for d in datasets_config:
                path = d.get("path")
                if path and os.path.exists(path):
                    exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
                    images = [p for ext in exts for p in Path(path).rglob(f"*{ext}") if "_truncated" not in p.stem]
                    if images:
                        self.datasets.append({
                            "path": path, "image_count": len(images), 
                            "preview_path": str(random.choice(images)), "repeats": d.get("repeats", 1)
                        })
            self.repopulate_dataset_grid()
            self.update_dataset_totals()

    def restore_defaults(self):
        index = self.config_dropdown.currentIndex()
        if index < 0: return
        selected_key = self.config_dropdown.itemData(index) if self.config_dropdown.itemData(index) else self.config_dropdown.itemText(index).replace(" ", "_").lower()
        reply = QtWidgets.QMessageBox.question(self, "Restore Defaults", 
            f"This will overwrite '{selected_key}.json' with hardcoded defaults. Are you sure?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.presets[selected_key] = copy.deepcopy(self.default_config)
            self.save_config()
            self.load_selected_config(index)
            self.log(f"Restored '{selected_key}.json' to defaults.")

    def toggle_all_unet_checkboxes(self, state):
        for cb in self.unet_layer_checkboxes.values(): cb.setChecked(state)
        self._update_unet_targets_config()
    def toggle_min_snr_gamma_widget(self):
        if hasattr(self, 'min_snr_sub_widget'): self.min_snr_sub_widget.setVisible(self.widgets.get("USE_MIN_SNR_GAMMA", QtWidgets.QCheckBox()).isChecked())
    def toggle_ip_noise_gamma_widget(self):
        if hasattr(self, 'ip_sub_widget'): self.ip_sub_widget.setVisible(self.widgets.get("USE_IP_NOISE_GAMMA", QtWidgets.QCheckBox()).isChecked())
    def toggle_resume_widgets(self):
        if hasattr(self, 'resume_sub_widget'): self.resume_sub_widget.setVisible(self.widgets.get("RESUME_TRAINING", QtWidgets.QCheckBox()).isChecked())
    def toggle_cond_dropout_widget(self):
        if hasattr(self, 'cond_sub_widget'): self.cond_sub_widget.setVisible(self.widgets.get("USE_COND_DROPOUT", QtWidgets.QCheckBox()).isChecked())

    def log(self, message):
        self.append_log(message.strip(), replace=False)

    def append_log(self, text, replace=False):
        scrollbar = self.log_textbox.verticalScrollBar()
        scroll_at_bottom = (scrollbar.value() >= scrollbar.maximum() - 4)
        cursor = self.log_textbox.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if replace:
            cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.log_textbox.setTextCursor(cursor)
        self.log_textbox.insertPlainText(text.rstrip() + '\n')
        if scroll_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def handle_process_output(self, text, is_progress):
        if text:
            self.append_log(text, replace=is_progress and self.last_line_is_progress)
            self.last_line_is_progress = is_progress

    def start_training(self):
        self.save_config()
        self.log("\n" + "="*50 + "\nStarting training process...\n" + "="*50)
        selected_key = self.config_dropdown.itemData(self.config_dropdown.currentIndex()) if self.config_dropdown.itemData(self.config_dropdown.currentIndex()) else self.config_dropdown.itemText(self.config_dropdown.currentIndex()).replace(" ", "_").lower()
        config_path = os.path.join(self.config_dir, f"{selected_key}.json")
        if not os.path.exists(config_path):
            self.log(f"CRITICAL ERROR: Config file not found: {config_path}. Aborting.")
            return

        # Make paths absolute to avoid working directory issues
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Assumes GUI and train.py are in the same dir
        config_path = os.path.abspath(config_path)
        train_py_path = os.path.abspath("train.py")  # Explicit absolute path to train.py

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Enhanced environment
        env_dict = os.environ.copy()
        python_dir = os.path.dirname(sys.executable)
        env_dict["PATH"] = f"{python_dir};{os.path.join(python_dir, 'Scripts')};{env_dict.get('PATH', '')}"
        env_dict["PYTHONPATH"] = f"{script_dir};{env_dict.get('PYTHONPATH', '')}"
        # Optional: Set CUDA_VISIBLE_DEVICES=0 if multi-GPU issues; adjust as needed
        # env_dict["CUDA_VISIBLE_DEVICES"] = "0"

        # Windows creation flags for isolation (new process group ONLY - no console window to avoid blank CMD)
        creation_flags = 0
        if os.name == 'nt':  # Windows
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP  # Removed | subprocess.CREATE_NEW_CONSOLE to hide window

        # Create and start the runner thread
        self.process_runner = ProcessRunner(
            executable=sys.executable,
            args=["-u", train_py_path, "--config", config_path],
            working_dir=script_dir,
            env=env_dict,
            creation_flags=creation_flags
        )
        self.process_runner.logSignal.connect(self.log)
        self.process_runner.paramInfoSignal.connect(lambda info: self.param_info_label.setText(f"Trainable Parameters: {info}"))
        self.process_runner.progressSignal.connect(self.handle_process_output)
        self.process_runner.finishedSignal.connect(self.training_finished)
        self.process_runner.errorSignal.connect(self.log)
        self.process_runner.start()

        self.log(f"INFO: Starting train.py (abs path: {train_py_path}) with config: {config_path} (working dir: {script_dir})")
        self.log("This is the training process. All output/prints are shown here in the GUI. Use the 'Stop Training' button to end it (or close the GUI).")

    def stop_training(self):
        if self.process_runner and self.process_runner.isRunning():
            self.process_runner.stop()
        else:
            self.log("No active training process to stop.")

    def training_finished(self, exit_code=0):
        if self.process_runner:
            self.process_runner.quit()
            self.process_runner.wait()
            self.process_runner = None
        status = "successfully" if exit_code == 0 else f"with an error (Code: {exit_code})"
        self.log(f"\n" + "="*50 + f"\nTraining finished {status}.\n" + "="*50)
        self.param_info_label.setText("Parameters: (training complete)" if exit_code == 0 else "Parameters: (training failed or stopped)")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _update_and_clamp_lr_graph(self):
        if not hasattr(self, 'lr_curve_widget'): return
        
        try: steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except (ValueError, KeyError): steps = 1
        try: min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except (ValueError, KeyError): min_lr = 0.0
        try: max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
        except (ValueError, KeyError): max_lr = 1e-6
        
        self.lr_curve_widget.set_bounds(steps, min_lr, max_lr)
        self.lr_curve_widget.set_points(self.current_config.get("LR_CUSTOM_CURVE", []))
        self._update_epoch_markers_on_graph()

    def _update_epoch_markers_on_graph(self):
        if not hasattr(self, 'lr_curve_widget'):
            return
        try:
            total_images = int(self.total_repeats_label.text())
            max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except (ValueError, KeyError):
            self.lr_curve_widget.set_epoch_data([])
            return
        
        epoch_data = []
        if total_images > 0 and max_steps > 0:
            steps_per_epoch = total_images
            current_epoch_step = steps_per_epoch
            while current_epoch_step < max_steps:
                normalized_x = current_epoch_step / max_steps
                epoch_data.append((normalized_x, int(current_epoch_step)))
                current_epoch_step += steps_per_epoch
        
        self.lr_curve_widget.set_epoch_data(epoch_data)
            
    def _update_lr_button_states(self, selected_index):
        if hasattr(self, 'remove_point_btn'):
            is_removable = selected_index > 0 and selected_index < len(self.lr_curve_widget.get_points()) - 1
            self.remove_point_btn.setEnabled(is_removable)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = TrainingGUI()
    main_win.show()
    sys.exit(app.exec())