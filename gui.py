import json
import os
import re
from PyQt6 import QtWidgets, QtCore, QtGui
import subprocess
from PyQt6.QtCore import QThread, pyqtSignal
import copy
import sys
from pathlib import Path
import random
import shutil
import math
import ctypes
import config as default_config

def prevent_sleep(enable=True):
    """Prevent/resume Windows sleep. Use as a context manager or toggle."""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001  # Prevent sleep
    ES_DISPLAY_REQUIRED = 0x00000002  # Prevent screen off

    kernel32 = ctypes.windll.kernel32
    if enable:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        print("Sleep prevention enabled.")  # Or emit to your log
    else:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("Sleep prevention disabled.")

        
# --- STYLESHEET CONSTANT ---
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

class InfoDialog(QtWidgets.QDialog):
    def __init__(self, title, text_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(650, 550)
        self.setStyleSheet("""
            QDialog { background-color: #2c2a3e; }
            QTextEdit {
                background-color: #1a1926;
                border: 1px solid #4a4668;
                color: #e0e0e0;
                font-size: 14px;
            }
            QPushButton { /* Inherits from main stylesheet */ }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(text_content)
        layout.addWidget(text_edit)
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

class SubOptionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.setContentsMargins(25, 10, 5, 10)
        self.main_layout.setSpacing(10)

        self.line_color = QtGui.QColor("#6a48d7")

    def get_layout(self):
        return self.main_layout

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
        safe_max_lr = max(self.max_lr_bound, 1e-12)
        log_max = math.log(safe_max_lr)

        if self.min_lr_bound > 0:
            effective_min_lr = self.min_lr_bound
        else:
            effective_min_lr = safe_max_lr / self.LOG_FLOOR_DIVISOR

        effective_min_lr = max(effective_min_lr, 1e-12)
        log_min = math.log(effective_min_lr)

        return log_max, log_min

    def _to_pixel_coords(self, norm_x, abs_lr):
        graph_width = self.width() - self.padding['left'] - self.padding['right']
        graph_height = self.height() - self.padding['top'] - self.padding['bottom']

        px = self.padding['left'] + norm_x * graph_width

        if abs_lr <= self.min_lr_bound:
            py = self.padding['top'] + graph_height
            return QtCore.QPointF(px, py)

        log_max, log_min = self._get_log_range()
        log_range = log_max - log_min

        if log_range <= 0:
            py = self.padding['top']
            return QtCore.QPointF(px, py)

        normalized_y = (math.log(abs_lr) - log_min) / log_range
        py = self.padding['top'] + (1 - normalized_y) * graph_height
        return QtCore.QPointF(px, py)

    def _to_data_coords(self, px, py):
        graph_width = self.width() - self.padding['left'] - self.padding['right']
        graph_height = self.height() - self.padding['top'] - self.padding['bottom']

        norm_x = (px - self.padding['left']) / graph_width
        clamped_py = max(self.padding['top'], min(py, self.padding['top'] + graph_height))
        normalized_y = 1 - ((clamped_py - self.padding['top']) / graph_height)

        log_max, log_min = self._get_log_range()
        log_range = log_max - log_min

        if log_range <= 0:
            abs_lr = self.min_lr_bound
        else:
            log_val = log_min + (normalized_y * log_range)
            abs_lr = math.exp(log_val)

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
        for i in range(5):
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
            normalized_y = 1.0 - (i / 4.0)

            if i == 0:
                lr_val = self.max_lr_bound
            elif i == 4:
                lr_val = self.min_lr_bound
            else:
                if log_range > 0:
                    lr_val = math.exp(log_min + (normalized_y * log_range))
                else:
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
        self.set_points(self._points)
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

            _, log_min = self._get_log_range()
            log_prev = math.log(max(prev_p[1], 1e-12))
            log_next = math.log(max(next_p[1], 1e-12))

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

    def apply_preset(self, points):
        """Applies a list of points and emits the change signal."""
        self.set_points(points)
        self.pointsChanged.emit(points)

    def set_cosine_preset(self):
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        warmup_portion = 0.10
        num_decay_points = 10
        points = [[0.0, min_lr], [warmup_portion, max_lr]]
        decay_duration = 1.0 - warmup_portion

        for i in range(num_decay_points + 1):
            decay_progress = i / num_decay_points
            norm_x = warmup_portion + decay_progress * decay_duration
            cosine_val = 0.5 * (1 + math.cos(math.pi * decay_progress))
            lr_val = min_lr + (max_lr - min_lr) * cosine_val
            points.append([norm_x, lr_val])

        unique_points = sorted(list(set(tuple(p) for p in points)), key=lambda x: x[0])
        self.apply_preset(unique_points)

    def set_linear_preset(self):
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        warmup_portion = 0.05
        points = [[0.0, min_lr], [warmup_portion, max_lr], [1.0, min_lr]]
        self.apply_preset(points)

    def set_constant_preset(self):
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        warmup_portion = 0.05
        cooldown_start = 1.0 - 0.10
        points = [[0.0, min_lr], [warmup_portion, max_lr], [cooldown_start, max_lr], [1.0, min_lr]]
        self.apply_preset(points)

    def set_step_preset(self):
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        points = [[0.0, min_lr], [0.05, max_lr], [0.60, max_lr], [0.65, max_lr / 40], [1.0, min_lr]]
        self.apply_preset(points)

    def set_cyclical_dip_preset(self):
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        warmup_end_portion = 0.05
        cooldown_start_portion = 0.60
        num_dips = 2
        dip_factor = 0.25
        dip_bottom_duration = 0.035
        dip_transition_duration = 0.025

        points = [[0.0, min_lr], [warmup_end_portion, max_lr]]
        plateau_duration = cooldown_start_portion - warmup_end_portion
        single_dip_total_duration = (dip_transition_duration * 2) + dip_bottom_duration
        flat_part_duration = (plateau_duration - (num_dips * single_dip_total_duration)) / (num_dips + 1)

        if flat_part_duration < 0: return # Or log an error

        current_pos = warmup_end_portion
        dip_lr = max(max_lr * dip_factor, min_lr)
        for _ in range(num_dips):
            current_pos += flat_part_duration
            points.append([current_pos, max_lr])
            current_pos += dip_transition_duration
            points.append([current_pos, dip_lr])
            current_pos += dip_bottom_duration
            points.append([current_pos, dip_lr])
            current_pos += dip_transition_duration
            points.append([current_pos, max_lr])

        points.extend([[cooldown_start_portion, max_lr], [0.65, max_lr / 40], [1.0, min_lr]])
        self.apply_preset(points)

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
            flags = self.creation_flags
            if os.name == 'nt':  # Windows-specific
                flags |= (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.HIGH_PRIORITY_CLASS)
                # Optional: For even higher (realtime-like), use subprocess.REALTIME_PRIORITY_CLASS
                # but this requires running your GUI as admin and can be risky.

            self.process = subprocess.Popen(
                [self.executable] + self.args,
                cwd=self.working_dir,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                creationflags=flags  # Use the modified flags here
            )
            self.logSignal.emit(f"INFO: Started subprocess (PID: {self.process.pid})")

            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                if not line or "NOTE: Redirects are currently not supported" in line:
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
        "WEIGHT_DECAY": {"label": "Optimizer Weight Decay:", "tooltip": "L2 regularization to prevent overfitting. Common range: 0.0 to 0.1.", "widget": "QDoubleSpinBox", "range": [0.0, 0.1], "step": 0.001, "decimals": 3},
        "LR_GRAPH_MIN": {"label": "Graph Min LR:", "tooltip": "The minimum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "LR_GRAPH_MAX": {"label": "Graph Max LR:", "tooltip": "The maximum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "SNR_STRATEGY": {"label": "Strategy:", "tooltip": "Min-SNR focuses on noisy steps, Max-SNR focuses on cleaner steps.", "widget": "QComboBox", "options": ["Min-SNR", "Max-SNR"]},
        "SNR_GAMMA": {"label": "Gamma Value:", "tooltip": "For Min-SNR, common range is 5-20. For Max-SNR, this acts as a clamp.", "widget": "QLineEdit"},
        "MIN_SNR_VARIANT": {"label": "Variant (Min-SNR only):", "tooltip": "Select the Min-SNR weighting variant.", "widget": "QComboBox", "options": ["debiased", "corrected", "standard"]},
        "IP_NOISE_GAMMA": {"label": "Gamma Value:", "tooltip": "Common range is 0.05 to 0.25.", "widget": "QLineEdit"},
        "COND_DROPOUT_PROB": {"label": "Dropout Probability:", "tooltip": "e.g., 0.1 (5-15% common).", "widget": "QLineEdit"},
        "NOISE_SCHEDULE_VARIANT": {
            "label": "Timestep Sampling:",
            "tooltip": "Changes how timesteps are sampled.\n- uniform: Standard random sampling.\n- logsnr_laplace: Focuses on more informative mid-range noise levels.\n- residual_shifting: Hard clamp to only the second half of timesteps.",
            "widget": "QComboBox",
            "options": ["uniform", "logsnr_laplace", "residual_shifting"]
        },
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
        self.current_config = {}
        self.last_line_is_progress = False
        self.default_config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        self.presets = {}
        
        self._initialize_configs()
        self._setup_ui()

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

        # --- FIX: Initialize these widgets BEFORE they are used in tab population ---
        self.param_info_label = QtWidgets.QLabel("Parameters: (awaiting training start)")
        self.param_info_label.setObjectName("ParamInfoLabel")
        self.param_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.param_info_label.setContentsMargins(0, 5, 0, 0)

        self.log_textbox = QtWidgets.QTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setMinimumHeight(200)
        # --- END FIX ---

        self.tab_view = QtWidgets.QTabWidget()

        # --- Create and populate all tabs at startup ---

        # Tab 1: Dataset
        dataset_content_widget = QtWidgets.QWidget()
        self._populate_dataset_tab(dataset_content_widget)
        dataset_scroll = QtWidgets.QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setWidget(dataset_content_widget)
        self.tab_view.addTab(dataset_scroll, "Dataset")

        # Tab 2: Model & Training Parameters
        model_content_widget = QtWidgets.QWidget()
        self._populate_data_model_tab(model_content_widget)
        model_scroll = QtWidgets.QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setWidget(model_content_widget)
        self.tab_view.addTab(model_scroll, "Model && Training Parameters")

        # Tab 3: Advanced
        advanced_content_widget = QtWidgets.QWidget()
        self._populate_advanced_tab(advanced_content_widget)
        advanced_scroll = QtWidgets.QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_scroll.setWidget(advanced_content_widget)
        self.tab_view.addTab(advanced_scroll, "Advanced")

        # Tab 4: Training Console
        console_tab_widget = QtWidgets.QWidget()
        console_layout = QtWidgets.QVBoxLayout(console_tab_widget)
        self._populate_console_tab(console_layout) # Now self.param_info_label exists
        self.tab_view.addTab(console_tab_widget, "Training Console")

        self.main_layout.addWidget(self.tab_view)

        self._setup_corner_widget()
        self._setup_action_buttons()

    def _setup_corner_widget(self):
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

    def _setup_action_buttons(self):
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
        """
        Builds a clean configuration dictionary for saving, using default_config as the
        source of truth for which keys are valid. This prevents saving obsolete keys
        that may exist in memory from older config files.
        """
        config_to_save = {}
        default_map = self.default_config

        # --- Handle special cases not in the main loop ---
        if hasattr(self, 'model_load_strategy_combo'):
            is_resuming = self.model_load_strategy_combo.currentIndex() == 1
            config_to_save["RESUME_TRAINING"] = is_resuming

        if hasattr(self, 'dataset_manager'):
            config_to_save["INSTANCE_DATASETS"] = self.dataset_manager.get_datasets_config()

        # --- Iterate over the canonical keys from default_config ---
        # This is the key change: we only iterate over keys that are known to be valid.
        for key in self.default_config.keys():
            # Skip keys that are handled specially or are not meant to be saved.
            if key in ["RESUME_TRAINING", "INSTANCE_DATASETS"]:
                continue

            # Get the current live value for this valid key from our in-memory config.
            live_val = self.current_config.get(key)
            if live_val is None:
                continue  # Don't save if it's not set for some reason

            default_val = default_map.get(key)

            try:
                # Handle special formatting for the learning rate curve
                if key == "LR_CUSTOM_CURVE":
                    rounded_curve = [[round(p[0], 8), round(p[1], 10)] for p in live_val]
                    config_to_save[key] = rounded_curve
                    continue

                # --- Use robust type conversion based on the default value's type ---
                converted_val = None
                if isinstance(live_val, (bool, list)):
                    converted_val = live_val
                elif default_val is not None:
                    default_type = type(default_val)
                    if default_type == bool:
                        converted_val = str(live_val).strip().lower() in ('true', '1', 't', 'y', 'yes')
                    elif default_type == int:
                        converted_val = int(str(live_val).strip()) if str(live_val).strip() else 0
                    elif default_type == float:
                        converted_val = float(str(live_val).strip()) if str(live_val).strip() else 0.0
                    elif default_type == list:
                        # Handle comma-separated strings for list values like BUCKET_ASPECT_RATIOS
                        raw_list_str = str(live_val).strip().replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                        # Try to convert to float, fallback to string if it fails
                        cleaned_parts = [p.strip() for p in raw_list_str.split(',') if p.strip()]
                        try:
                            converted_val = [float(p) for p in cleaned_parts]
                        except ValueError:
                             converted_val = cleaned_parts # Keep as strings if they can't be floats
                    else: # Default to string
                        converted_val = str(live_val)
                else: # Fallback if no default type info
                    converted_val = str(live_val)
                
                config_to_save[key] = converted_val
            
            except (ValueError, TypeError) as e:
                self.log(f"Warning: Could not convert value for '{key}'. Not saved. Error: {e}")

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
        elif widget_type == "QDoubleSpinBox":
            widget = QtWidgets.QDoubleSpinBox()
            if "range" in definition:
                widget.setRange(*definition["range"])
            if "step" in definition:
                widget.setSingleStep(definition["step"])
            if "decimals" in definition:
                widget.setDecimals(definition["decimals"])
            widget.valueChanged.connect(lambda value, k=key: self._update_config_from_widget(k, widget))
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
            top_hbox.addWidget(self._create_form_group(title, keys))
        layout.addLayout(top_hbox)

        self.dataset_manager = DatasetManagerWidget(self)
        self.dataset_manager.datasetsChanged.connect(self._update_epoch_markers_on_graph)
        layout.addWidget(self.dataset_manager)

    def _populate_data_model_tab(self, parent_widget):
        layout = QtWidgets.QHBoxLayout(parent_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 5, 15, 15)

        left_vbox = QtWidgets.QVBoxLayout()
        right_vbox = QtWidgets.QVBoxLayout()

        path_group = self._create_path_group()
        left_vbox.addWidget(path_group)

        core_group = self._create_core_training_group()
        left_vbox.addWidget(core_group)
        left_vbox.addStretch()

        optimizer_group = self._create_form_group("Raven Optimizer", ["WEIGHT_DECAY"])
        right_vbox.addWidget(optimizer_group)
        
        lr_group = self._create_lr_scheduler_group()
        right_vbox.addWidget(lr_group)
        right_vbox.addStretch()

        layout.addLayout(left_vbox, 1)
        layout.addLayout(right_vbox, 1)

        self.widgets["MAX_TRAIN_STEPS"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["GRADIENT_ACCUMULATION_STEPS"].textChanged.connect(self._update_epoch_markers_on_graph)
        self._update_lr_button_states(-1)

    def _create_form_group(self, title, keys):
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QFormLayout(group)
        for key in keys:
            label, widget = self._create_widget(key)
            if label and widget:
                layout.addRow(label, widget)
        return group

    def _create_path_group(self):
        path_group = QtWidgets.QGroupBox("File & Directory Paths")
        path_layout = QtWidgets.QFormLayout(path_group)
        
        self.model_load_strategy_combo = QtWidgets.QComboBox()
        self.model_load_strategy_combo.addItems(["Load Base Model", "Resume from Checkpoint"])
        path_layout.addRow("Mode:", self.model_load_strategy_combo)
        
        self.base_model_sub_widget = QtWidgets.QWidget()
        base_layout = QtWidgets.QFormLayout(self.base_model_sub_widget)
        base_layout.setContentsMargins(0,0,0,0)
        label, widget = self._create_widget("SINGLE_FILE_CHECKPOINT_PATH")
        base_layout.addRow(label, widget)
        path_layout.addRow(self.base_model_sub_widget)

        self.resume_sub_widget = QtWidgets.QWidget()
        resume_layout = QtWidgets.QFormLayout(self.resume_sub_widget)
        resume_layout.setContentsMargins(0,0,0,0)
        label, widget = self._create_widget("RESUME_MODEL_PATH")
        resume_layout.addRow(label, widget)
        label, widget = self._create_widget("RESUME_STATE_PATH")
        resume_layout.addRow(label, widget)
        path_layout.addRow(self.resume_sub_widget)

        label, widget = self._create_widget("OUTPUT_DIR")
        path_layout.addRow(label, widget)
        
        self.model_load_strategy_combo.currentIndexChanged.connect(self.toggle_resume_widgets)
        return path_group

    def _create_core_training_group(self):
        core_group = QtWidgets.QGroupBox("Core Training")
        layout = QtWidgets.QFormLayout(core_group)
        core_keys = ["MAX_TRAIN_STEPS", "SAVE_EVERY_N_STEPS", "GRADIENT_ACCUMULATION_STEPS", "MIXED_PRECISION", "SEED"]
        for key in core_keys:
            label, widget = self._create_widget(key)
            layout.addRow(label, widget)
        
        layout.addRow(self._create_separator())
        label, widget = self._create_widget("NOISE_SCHEDULE_VARIANT")
        layout.addRow(label, widget)
        layout.addRow(self._create_separator())

        self._create_bool_option(layout, "USE_ZERO_TERMINAL_SNR", "Use Zero-Terminal SNR", "Rescales noise schedule for better dynamic range.")
        return core_group

    def _create_lr_scheduler_group(self):
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
        preset_layout.addWidget(QtWidgets.QLabel("<b>Presets:</b>"))
        presets = {
            "Cosine": self.lr_curve_widget.set_cosine_preset,
            "Linear": self.lr_curve_widget.set_linear_preset,
            "Constant": self.lr_curve_widget.set_constant_preset,
            "Step": self.lr_curve_widget.set_step_preset,
            "Cyclical Dip": self.lr_curve_widget.set_cyclical_dip_preset
        }
        for name, func in presets.items():
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(func)
            preset_layout.addWidget(btn)
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
        return lr_group
        
    def _create_separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("border: 1px solid #4a4668;")
        return line

    def _populate_advanced_tab(self, parent_widget):
        main_layout = QtWidgets.QHBoxLayout(parent_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        left_layout = QtWidgets.QVBoxLayout()

        noise_group = QtWidgets.QGroupBox("Noise & Regularization")
        noise_layout = QtWidgets.QVBoxLayout(noise_group)

        self._create_bool_option(noise_layout, "USE_PER_CHANNEL_NOISE", "Use Per-Channel (Color) Noise", "Enables applying noise independently to R, G, and B channels.")
        noise_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))

        self._create_bool_option(noise_layout, "USE_IP_NOISE_GAMMA", "Use Input Perturbation Noise", "Adds noise to input latents for regularization.", self.toggle_ip_noise_gamma_widget)
        self.ip_sub_widget = SubOptionWidget()
        label, widget = self._create_widget("IP_NOISE_GAMMA"); self.ip_sub_widget.get_layout().addWidget(label); self.ip_sub_widget.get_layout().addWidget(widget)
        noise_layout.addWidget(self.ip_sub_widget)
        noise_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))

        self._create_bool_option(noise_layout, "USE_COND_DROPOUT", "Use Text Conditioning Dropout", "Improves classifier-free guidance.", self.toggle_cond_dropout_widget)
        self.cond_sub_widget = SubOptionWidget()
        label, widget = self._create_widget("COND_DROPOUT_PROB"); self.cond_sub_widget.get_layout().addWidget(label); self.cond_sub_widget.get_layout().addWidget(widget)
        noise_layout.addWidget(self.cond_sub_widget)
        left_layout.addWidget(noise_group)

        loss_group = QtWidgets.QGroupBox("Loss Weighting")
        loss_layout = QtWidgets.QVBoxLayout(loss_group)

        self._create_bool_option(loss_layout, "USE_SNR_GAMMA", "Use SNR Gamma Weighting", "Reweights the loss based on Signal-to-Noise Ratio.", self.toggle_snr_gamma_widget)
        self.snr_sub_widget = SubOptionWidget()
        
        snr_sub_layout = QtWidgets.QFormLayout()
        self.snr_sub_widget.get_layout().addLayout(snr_sub_layout)

        snr_strategy_label, snr_strategy_widget = self._create_widget("SNR_STRATEGY")
        snr_sub_layout.addRow(snr_strategy_label, snr_strategy_widget)
        snr_strategy_widget.currentTextChanged.connect(self.toggle_snr_variant_widget)

        gamma_label, gamma_widget = self._create_widget("SNR_GAMMA")
        snr_sub_layout.addRow(gamma_label, gamma_widget)

        self.min_snr_variant_label, self.min_snr_variant_widget = self._create_widget("MIN_SNR_VARIANT")
        snr_sub_layout.addRow(self.min_snr_variant_label, self.min_snr_variant_widget)

        loss_layout.addWidget(self.snr_sub_widget)
        left_layout.addWidget(loss_group)
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
        elif isinstance(widget, QtWidgets.QDoubleSpinBox):
            self.current_config[key] = widget.value()
        elif isinstance(widget, LRCurveWidget):
            self.current_config[key] = widget.get_points()

    def _apply_config_to_widgets(self):
        if self.unet_layer_checkboxes:
            unet_targets = self.current_config.get("UNET_TRAIN_TARGETS", [])
            for key, checkbox in self.unet_layer_checkboxes.items():
                checkbox.setChecked(key in unet_targets)

        if hasattr(self, 'model_load_strategy_combo'):
            is_resuming = self.current_config.get("RESUME_TRAINING", False)
            self.model_load_strategy_combo.setCurrentIndex(1 if is_resuming else 0)
            self.toggle_resume_widgets()

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
                    elif isinstance(widget, QtWidgets.QDoubleSpinBox): widget.setValue(float(value))
                finally:
                    widget.blockSignals(False)

        if hasattr(self, 'lr_curve_widget'):
            self._update_and_clamp_lr_graph()

        for toggle_func in [self.toggle_snr_gamma_widget,
                            self.toggle_ip_noise_gamma_widget, 
                            self.toggle_cond_dropout_widget]:
            toggle_func()

        if hasattr(self, "dataset_manager"):
            datasets_config = self.current_config.get("INSTANCE_DATASETS", [])
            self.dataset_manager.load_datasets_from_config(datasets_config)

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
        
    def toggle_snr_gamma_widget(self):
        if hasattr(self, 'snr_sub_widget'):
            is_checked = self.widgets.get("USE_SNR_GAMMA", QtWidgets.QCheckBox()).isChecked()
            self.snr_sub_widget.setVisible(is_checked)
            if is_checked:
                self.toggle_snr_variant_widget()

    def toggle_ip_noise_gamma_widget(self):
        if hasattr(self, 'ip_sub_widget'): self.ip_sub_widget.setVisible(self.widgets.get("USE_IP_NOISE_GAMMA", QtWidgets.QCheckBox()).isChecked())
        
    def toggle_resume_widgets(self):
        if hasattr(self, 'resume_sub_widget') and hasattr(self, 'base_model_sub_widget'):
            is_resuming = self.model_load_strategy_combo.currentIndex() == 1
            self.resume_sub_widget.setVisible(is_resuming)
            self.base_model_sub_widget.setVisible(not is_resuming)
            
    def toggle_cond_dropout_widget(self):
        if hasattr(self, 'cond_sub_widget'): self.cond_sub_widget.setVisible(self.widgets.get("USE_COND_DROPOUT", QtWidgets.QCheckBox()).isChecked())
        
    def toggle_snr_variant_widget(self, text=None):
        if not hasattr(self, 'min_snr_variant_widget'): return
        strategy = text if text is not None else self.widgets.get("SNR_STRATEGY", QtWidgets.QComboBox()).currentText()
        is_min_snr = (strategy == "Min-SNR")
        self.min_snr_variant_label.setVisible(is_min_snr)
        self.min_snr_variant_widget.setVisible(is_min_snr)

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

        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(config_path)
        train_py_path = os.path.abspath("train.py")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        env_dict = os.environ.copy()
        python_dir = os.path.dirname(sys.executable)
        env_dict["PATH"] = f"{python_dir};{os.path.join(python_dir, 'Scripts')};{env_dict.get('PATH', '')}"
        env_dict["PYTHONPATH"] = f"{script_dir};{env_dict.get('PYTHONPATH', '')}"

        creation_flags = 0
        if os.name == 'nt':
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

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
        if os.name == 'nt':
            prevent_sleep(True)
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
        if os.name == 'nt':
            prevent_sleep(False)

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
        if not hasattr(self, 'lr_curve_widget') or not hasattr(self, 'dataset_manager'):
            return
            
        try:
            total_images = self.dataset_manager.get_total_repeats()
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

# --- NEW DATASET MANAGER WIDGET ---
class DatasetManagerWidget(QtWidgets.QWidget):
    datasetsChanged = QtCore.pyqtSignal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.datasets = []
        self.dataset_widgets = []
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        add_button = QtWidgets.QPushButton("Add Dataset Folder")
        add_button.clicked.connect(self.add_dataset_folder)
        layout.addWidget(add_button)

        self.dataset_vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(self.dataset_vbox)

        bottom_hbox = QtWidgets.QHBoxLayout()
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
        
    def get_total_repeats(self):
        return sum(ds["image_count"] * ds["repeats"] for ds in self.datasets)

    def get_datasets_config(self):
        return [
            {
                "path": ds["path"],
                "repeats": ds["repeats"],
                "mirror_repeats": ds["mirror_repeats"],
                "darken_repeats": ds["darken_repeats"],
                "use_mask": ds["use_mask"],
                "mask_path": ds["mask_path"],
                "mask_focus_factor": ds["mask_focus_factor"],
                "mask_focus_mode": ds["mask_focus_mode"] # <-- ADD THIS LINE
            } for ds in self.datasets
        ]
        
    def load_datasets_from_config(self, datasets_config):
        self.datasets = []
        for d in datasets_config:
            path = d.get("path")
            if path and os.path.exists(path):
                exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
                images = [p for ext in exts for p in Path(path).rglob(f"*{ext}") if "_truncated" not in p.stem and "_mask" not in p.stem]
                if images:
                    self.datasets.append({
                        "path": path, "image_count": len(images),
                        "preview_path": str(random.choice(images)),
                        "repeats": d.get("repeats", 1),
                        "mirror_repeats": d.get("mirror_repeats", False),
                        "darken_repeats": d.get("darken_repeats", False),
                        "use_mask": d.get("use_mask", False),
                        "mask_path": d.get("mask_path", ""),
                        "mask_focus_factor": d.get("mask_focus_factor", 2.0),
                        # <-- ADD THIS LINE WITH A DEFAULT FOR BACKWARD COMPATIBILITY -->
                        "mask_focus_mode": d.get("mask_focus_mode", "Proportional (Multiply)")
                    })
        self.repopulate_dataset_grid()
        self.update_dataset_totals()

    def add_dataset_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if not path: return
        exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        images = [p for ext in exts for p in Path(path).rglob(f"*{ext}") if "_truncated" not in p.stem and "_mask" not in p.stem]
        if not images:
            QtWidgets.QMessageBox.warning(self, "No Images", "No valid images found in the folder (or they were all excluded as masks).")
            return
        
        self.datasets.append({
            "path": path,
            "image_count": len(images),
            "preview_path": str(random.choice(images)),
            "repeats": 1, "mirror_repeats": False, "darken_repeats": False,
            "use_mask": False, "mask_path": "", "mask_focus_factor": 2.0,
            "mask_focus_mode": "Proportional (Multiply)", # <-- ADD THIS LINE
        })
        self.repopulate_dataset_grid()
        self.update_dataset_totals()

    def repopulate_dataset_grid(self):
        for i in reversed(range(self.dataset_vbox.count())):
            item = self.dataset_vbox.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.dataset_widgets = []

        for idx, ds in enumerate(self.datasets):
            main_widget = QtWidgets.QWidget()
            main_layout = QtWidgets.QHBoxLayout(main_widget)
            main_layout.setContentsMargins(0, 10, 0, 5)

            preview_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(ds["preview_path"]).scaled(128, 128, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            preview_label.setPixmap(pixmap)
            main_layout.addWidget(preview_label)

            path_label = QtWidgets.QLabel(f"<b>Path:</b><br>{ds['path']}")
            path_label.setWordWrap(True)
            main_layout.addWidget(path_label, 1)

            repeats_spin = QtWidgets.QSpinBox()
            repeats_spin.setMinimum(1); repeats_spin.setMaximum(10000)
            repeats_spin.setValue(ds["repeats"])
            repeats_spin.valueChanged.connect(lambda v, i=idx: self.update_repeats(i, v))
            
            form_layout = QtWidgets.QFormLayout()
            form_layout.addRow("Repeats:", repeats_spin)
            main_layout.addLayout(form_layout)

            btn_vbox = QtWidgets.QVBoxLayout()
            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.clicked.connect(lambda _, i=idx: self.remove_dataset(i))
            btn_vbox.addWidget(remove_btn)
            clear_btn = QtWidgets.QPushButton("Clear Cached Latents")
            clear_btn.clicked.connect(lambda _, p=ds["path"]: self.confirm_clear_cache(p))
            btn_vbox.addWidget(clear_btn)
            main_layout.addLayout(btn_vbox)

            aug_sub_widget, mask_sub_widget, widget_group = self._create_sub_widgets(idx, ds)
            self.dataset_widgets.append(widget_group)
            
            self.dataset_vbox.addWidget(main_widget)
            self.dataset_vbox.addWidget(aug_sub_widget)
            self.dataset_vbox.addWidget(mask_sub_widget)

            self.toggle_mask_controls(idx, ds.get("use_mask", False))
            if ds.get("use_mask"):
                self.update_mask_preview(idx)



# In the DatasetManagerWidget class...

# In the DatasetManagerWidget class...

    def _create_sub_widgets(self, idx, ds):
        aug_sub_widget = SubOptionWidget()
        aug_layout = aug_sub_widget.get_layout()
        aug_layout.setContentsMargins(35, 5, 5, 5)
        mirror_cb = QtWidgets.QCheckBox("Mirror Repeats")
        mirror_cb.setToolTip("Horizontally mirrors repeated images for this dataset.")
        mirror_cb.setChecked(ds.get("mirror_repeats", False))
        mirror_cb.stateChanged.connect(lambda state, i=idx: self.update_dataset_bool(i, "mirror_repeats", state))
        darken_cb = QtWidgets.QCheckBox("Increase Contrast on Repeats")
        darken_cb.setToolTip("Applies a contrast-enhancing S-curve to repeated images.")
        darken_cb.setChecked(ds.get("darken_repeats", False))
        darken_cb.stateChanged.connect(lambda state, i=idx: self.update_dataset_bool(i, "darken_repeats", state))
        aug_layout.addWidget(mirror_cb)
        aug_layout.addWidget(darken_cb)
        aug_layout.addStretch()

        mask_sub_widget = SubOptionWidget()
        mask_layout = mask_sub_widget.get_layout()
        mask_enable_cb = QtWidgets.QCheckBox("Enable Masked Training")
        mask_enable_cb.setChecked(ds.get("use_mask", False))
        
        help_button = QtWidgets.QPushButton("?")
        help_button.setFixedSize(22, 22)
        help_button.setToolTip("Click for an explanation of Masked Training.")
        
        # --- FINAL CORRECTED STYLESHEET ---
        # This version aggressively overrides the global button styles to force the small size.
        help_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 14px;
                color: #e0e0e0;
                background-color: #4a4668;
                border: 1px solid #5c5a70;
                border-radius: 11px; /* half of 22px size */
                padding: 0px;
                margin: 0px;
                /* Force size override */
                min-height: 20px;
                max-height: 20px;
                min-width: 20px;
                max-width: 20px;
            }
            QPushButton:hover {
                background-color: #6a48d7;
                color: #ffffff;
                border: 1px solid #ab97e6;
            }
            QPushButton:pressed {
                background-color: #383552;
            }
        """)

        help_text = """
        <p><b>Masked Training</b> allows you to increase the training focus on specific parts of your images.</p>
        <p>This is useful for teaching the model specific details, objects, or styles in a concentrated area, while the rest of the image continues to train normally.</p>
        <hr>
        <p><b>How it Works:</b></p>
        <ul>
            <li>You provide a corresponding 'mask' image for each of your training images.</li>

            <li>The mask is an image that acts as a guide:
                <ul>
                    <li style="margin-left:15px;"><b>Black pixels</b> mark the areas that will be trained with normal priority (a focus of 1.0).</li>
                    <li style="margin-left:15px;"><b>Any non-black pixels</b> (white, red, etc.) mark the areas you want to prioritize.</li>
                </ul>
            </li>

            <li>The <b>Focus Factor</b> is a multiplier that determines how much more to prioritize the non-black areas. A factor of 2.0 means these focused areas will have twice the impact on training compared to the black areas.</li>

            <li>Masks must be saved in the specified 'Mask Path' folder and have the same filename as the original image, but with a <b>_mask.png</b> suffix.<br><i>(e.g., for 'my_image.jpg', the mask must be 'my_image_mask.png')</i></li>
        </ul>
        <p>A preview of the mask (visualized in red) will be shown on a sample image when a valid mask path is provided.</p>
        """

        def show_help_dialog():
            dialog = InfoDialog("What is Masked Training?", help_text, self)
            dialog.exec()

        help_button.clicked.connect(show_help_dialog)
        
        mask_enable_layout = QtWidgets.QHBoxLayout()
        mask_enable_layout.setContentsMargins(0, 0, 0, 0)
        mask_enable_layout.addWidget(mask_enable_cb)
        mask_enable_layout.addWidget(help_button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter) # Keep alignment for centering
        mask_enable_layout.addStretch()
        
        mask_path_edit = QtWidgets.QLineEdit(ds.get("mask_path", ""))
        mask_path_edit.setPlaceholderText("Path to mask folder...")
        mask_path_edit.textChanged.connect(lambda text, i=idx: self.update_mask_path(i, text))
        mask_browse_btn = QtWidgets.QPushButton("Browse...")
        mask_browse_btn.clicked.connect(lambda _, i=idx: self.browse_for_mask_folder(i))
        mask_preview_label = QtWidgets.QLabel()
        mask_preview_label.setFixedSize(128, 128)
        mask_preview_label.setStyleSheet("border: 1px solid #4a4668; background-color: #1a1926;")
        mask_factor_spinbox = QtWidgets.QDoubleSpinBox()
        mask_factor_spinbox.setRange(1.0, 100.0); mask_factor_spinbox.setSingleStep(0.1)
        mask_factor_spinbox.setDecimals(2); mask_factor_spinbox.setValue(ds.get("mask_focus_factor", 2.0))
        mask_factor_spinbox.valueChanged.connect(lambda v, i=idx: self.update_mask_factor(i, v))
        mask_mode_combo = QtWidgets.QComboBox()
        mask_mode_combo.addItems(["Proportional (Multiply)", "Uniform (Add)"])
        mask_mode_combo.setToolTip("Proportional: loss * factor\nUniform: loss + factor")
        mask_mode_combo.setCurrentText(ds.get("mask_focus_mode", "Proportional (Multiply)"))
        mask_mode_combo.currentTextChanged.connect(lambda text, i=idx: self.update_mask_mode(i, text))
        
        widget_group = {
            "mask_path_edit": mask_path_edit, "mask_preview": mask_preview_label,
            "mask_factor_spin": mask_factor_spinbox,
            "mask_mode_combo": mask_mode_combo,
        }
        
        
        mask_enable_cb.stateChanged.connect(lambda state, i=idx: self.toggle_mask_controls(i, state))
        
        sub_vbox = QtWidgets.QVBoxLayout()
        sub_vbox.addLayout(mask_enable_layout)
        path_hbox = QtWidgets.QHBoxLayout()
        path_hbox.addWidget(mask_path_edit, 1); path_hbox.addWidget(mask_browse_btn)
        sub_vbox.addLayout(path_hbox)
        factor_hbox = QtWidgets.QHBoxLayout()
        factor_hbox.addWidget(QtWidgets.QLabel("Focus Factor:"))
        factor_hbox.addWidget(mask_factor_spinbox)
        factor_hbox.addWidget(QtWidgets.QLabel("Mode:"))
        factor_hbox.addWidget(mask_mode_combo)
        factor_hbox.addStretch()
        sub_vbox.addLayout(factor_hbox)
        mask_layout.addLayout(sub_vbox, 1)
        mask_layout.addWidget(mask_preview_label)

        return aug_sub_widget, mask_sub_widget, widget_group
    
    def update_repeats(self, idx, val):
        self.datasets[idx]["repeats"] = val
        self.update_dataset_totals()
    def update_mask_path(self, idx, text):
        self.datasets[idx]["mask_path"] = text
        self.update_mask_preview(idx)
    def update_mask_factor(self, idx, val):
        self.datasets[idx]["mask_focus_factor"] = val
    def update_mask_mode(self, idx, text):
        self.datasets[idx]["mask_focus_mode"] = text
    def update_dataset_bool(self, idx, key, state):
        self.datasets[idx][key] = bool(state)
    def remove_dataset(self, idx):
        del self.datasets[idx]
        self.repopulate_dataset_grid()
        self.update_dataset_totals()
    def update_dataset_totals(self):
        total = sum(ds["image_count"] for ds in self.datasets)
        total_rep = self.get_total_repeats()
        self.total_label.setText(str(total))
        self.total_repeats_label.setText(str(total_rep))
        self.datasetsChanged.emit()

    def confirm_clear_cache(self, path):
        reply = QtWidgets.QMessageBox.question(self, "Confirm", "Delete all cached latents in this dataset?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.clear_cached_latents(path)

    def clear_cached_latents(self, path):
        root = Path(path)
        deleted = False
        for cache_dir in list(root.rglob(".precomputed_embeddings_cache")):
            if cache_dir.is_dir():
                try:
                    shutil.rmtree(cache_dir)
                    self.parent_gui.log(f"Deleted cache directory: {cache_dir}")
                    deleted = True
                except Exception as e:
                    self.parent_gui.log(f"Error deleting {cache_dir}: {e}")
        if not deleted:
            self.parent_gui.log("No cached latent directories found to delete.")
            
    def browse_for_mask_folder(self, idx):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Mask Folder")
        if path:
            self.dataset_widgets[idx]["mask_path_edit"].setText(path)

    def update_mask_preview(self, idx):
        preview_label = self.dataset_widgets[idx]["mask_preview"]
        ds = self.datasets[idx]
        preview_label.clear()

        base_img_path = Path(ds.get("preview_path", ""))
        mask_path_dir = Path(ds.get("mask_path", ""))

        if not base_img_path.exists() or not mask_path_dir.is_dir():
            return
            
        mask_filename = f"{base_img_path.stem}_mask.png"
        mask_file_path = mask_path_dir / mask_filename
        
        if not mask_file_path.exists():
            return
            
        try:
            base_pixmap = QtGui.QPixmap(str(base_img_path))
            mask_image = QtGui.QImage(str(mask_file_path))

            if base_pixmap.isNull() or mask_image.isNull(): return

            size = QtCore.QSize(128, 128)
            combined = QtGui.QPixmap(size)
            combined.fill(QtCore.Qt.GlobalColor.transparent)

            scaled_base = base_pixmap.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            draw_rect = scaled_base.rect().translated((size.width() - scaled_base.width()) // 2, (size.height() - scaled_base.height()) // 2)

            scaled_mask_image = mask_image.scaled(scaled_base.size(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

            mask_pixmap = QtGui.QPixmap.fromImage(scaled_mask_image)
            
            mask_bitmap = mask_pixmap.createMaskFromColor(
                QtGui.QColor('black'), QtCore.Qt.MaskMode.MaskInColor
            )

            tint_color = QtGui.QColor(229, 57, 53, 150) # semi-transparent red
            tinted_overlay = QtGui.QPixmap(scaled_base.size())
            tinted_overlay.fill(tint_color)
            tinted_overlay.setMask(mask_bitmap) 

            painter = QtGui.QPainter(combined)
            painter.drawPixmap(draw_rect, scaled_base)      
            painter.drawPixmap(draw_rect, tinted_overlay) 
            painter.end()
            
            preview_label.setPixmap(combined)
        except Exception as e:
            self.parent_gui.log(f"Error creating mask preview for {mask_file_path}: {e}")

    def toggle_mask_controls(self, idx, state):
        is_checked = bool(state)
        self.datasets[idx]["use_mask"] = is_checked
        
        widgets = self.dataset_widgets[idx]
        widgets["mask_path_edit"].setVisible(is_checked)
        widgets["mask_path_edit"].parent().findChild(QtWidgets.QPushButton).setVisible(is_checked) 
        widgets["mask_preview"].setVisible(is_checked)
        widgets["mask_factor_spin"].setVisible(is_checked)
        widgets["mask_factor_spin"].parent().findChild(QtWidgets.QLabel).setVisible(is_checked)
        
        widgets["mask_mode_combo"].setVisible(is_checked)
        mode_label = widgets["mask_mode_combo"].parent().findChild(QtWidgets.QLabel, "Mode:")
        if mode_label:
            mode_label.setVisible(is_checked)

        if is_checked:
            self.update_mask_preview(idx)
        else:
            widgets["mask_preview"].clear()
            
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = TrainingGUI()
    main_win.show()
    sys.exit(app.exec())