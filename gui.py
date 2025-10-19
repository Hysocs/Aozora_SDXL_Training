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
import ctypes
import math
import config as default_config
from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def prevent_sleep(enable=True):
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    kernel32 = ctypes.windll.kernel32
    if enable:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        print("Sleep prevention enabled.")
    else:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("Sleep prevention disabled.")

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
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSIjMmMyYTNlIiBkPSJNOSAxNi4xN0w0LjgzIDEybC0xLjQyIDEuNDFMOSAxOUwyMSA3bC0xLjQxLTEuNDF6Ii8+PC9zdmc+");
}
QCheckBox::indicator:disabled {
    border: 1px solid #5c5a70;
    background-color: #383552;
}
QComboBox { background-color: #1a1926; border: 1px solid #4a4668; padding: 6px; border-radius: 4px; min-height: 32px; max-height: 32px; }
QComboBox:on { border: 1px solid #ab97e6; }
QComboBox::drop-down { border-left: 1px solid #4a4668; }
QComboBox QAbstractItemView { background-color: #383552; border: 1px solid #ab97e6; selection-background-color: #ab97e6; selection-color: #1a1926; }
QSpinBox, QDoubleSpinBox {
    background-color: #1a1926;
    border: 1px solid #4a4668;
    padding: 6px;
    color: #e0e0e0;
    border-radius: 4px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #ab97e6; }
QTabWidget::pane { border: 1px solid #4a4668; border-top: none; }
QTabBar::tab { background: #2c2a3e; border: 1px solid #4a4668; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 10px 20px; color: #e0e0e0; font-weight: bold; min-height: 40px; }
QTabBar::tab:selected { background: #383552; color: #ffffff; border-bottom: 3px solid #ab97e6; }
QTabBar::tab:!selected:hover { background: #4a4668; }
QScrollArea { border: none; }
"""




class GraphPanel(QtWidgets.QWidget):
    """Custom QPainter-based graph panel for plotting data."""
    
    def __init__(self, title, y_label, parent=None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.setMinimumHeight(180)
        
        # Data storage - list of (x, y, color, label, linewidth) tuples for each line
        self.lines = []  # Each line: {'data': deque, 'color': QColor, 'label': str, 'linewidth': int}
        
        # Display settings
        self.padding = {'top': 35, 'bottom': 40, 'left': 70, 'right': 20}
        self.bg_color = QtGui.QColor("#1a1926")
        self.graph_bg_color = QtGui.QColor("#2c2a3e")
        self.grid_color = QtGui.QColor("#4a4668")
        self.text_color = QtGui.QColor("#e0e0e0")
        self.title_color = QtGui.QColor("#ab97e6")
        
        # Auto-scaling
        self.x_min = 0
        self.x_max = 100
        self.y_min = 0
        self.y_max = 1
        
        self.setMouseTracking(True)
        self.tooltip_pos = None
    
    def add_line(self, color, label, max_points=500, linewidth=2):
        """Add a new line to the graph."""
        self.lines.append({
            'data': deque(maxlen=max_points),
            'color': QtGui.QColor(color),
            'label': label,
            'linewidth': linewidth
        })
        return len(self.lines) - 1
    
    def append_data(self, line_index, x, y):
        """Append a data point to a specific line."""
        if 0 <= line_index < len(self.lines):
            self.lines[line_index]['data'].append((x, y))
            self._update_bounds()
    
    def clear_all_data(self):
        """Clear all data from all lines."""
        for line in self.lines:
            line['data'].clear()
        self._update_bounds()
        self.update()
    
    def _update_bounds(self):
        """Update min/max bounds based on current data."""
        all_x = []
        all_y = []
        
        for line in self.lines:
            for x, y in line['data']:
                all_x.append(x)
                all_y.append(y)
        
        if all_x and all_y:
            self.x_min = min(all_x)
            self.x_max = max(all_x)
            self.y_min = min(all_y)
            self.y_max = max(all_y)
            
            # Add 5% padding to y-axis
            y_range = self.y_max - self.y_min
            if y_range == 0:
                y_range = 1
            self.y_min -= y_range * 0.05
            self.y_max += y_range * 0.05
        else:
            self.x_min = 0
            self.x_max = 100
            self.y_min = 0
            self.y_max = 1
    
    def _to_screen_coords(self, x, y):
        """Convert data coordinates to screen coordinates."""
        graph_width = self.width() - self.padding['left'] - self.padding['right']
        graph_height = self.height() - self.padding['top'] - self.padding['bottom']
        
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        
        if x_range == 0:
            x_range = 1
        if y_range == 0:
            y_range = 1
        
        screen_x = self.padding['left'] + ((x - self.x_min) / x_range) * graph_width
        screen_y = self.padding['top'] + graph_height - ((y - self.y_min) / y_range) * graph_height
        
        return QtCore.QPointF(screen_x, screen_y)
    
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Draw graph area
        graph_rect = QtCore.QRect(
            self.padding['left'],
            self.padding['top'],
            self.width() - self.padding['left'] - self.padding['right'],
            self.height() - self.padding['top'] - self.padding['bottom']
        )
        painter.fillRect(graph_rect, self.graph_bg_color)
        
        # Draw grid and labels
        self._draw_grid_and_axes(painter, graph_rect)
        
        # Draw title
        self._draw_title(painter)
        
        # Draw legend
        self._draw_legend(painter)
        
        # Draw data lines
        self._draw_data_lines(painter)
    
    def _draw_grid_and_axes(self, painter, rect):
        """Draw grid lines and axis labels."""
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        
        # Draw horizontal grid lines
        num_h_lines = 5
        for i in range(num_h_lines):
            y = rect.top() + (i / (num_h_lines - 1)) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            
            # Y-axis labels
            y_val = self.y_max - (i / (num_h_lines - 1)) * (self.y_max - self.y_min)
            label = self._format_number(y_val)
            
            painter.setPen(self.text_color)
            painter.drawText(
                QtCore.QRect(5, int(y - 10), self.padding['left'] - 10, 20),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                label
            )
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        
        # Draw vertical grid lines
        num_v_lines = 6
        for i in range(num_v_lines):
            x = rect.left() + (i / (num_v_lines - 1)) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            
            # X-axis labels
            x_val = self.x_min + (i / (num_v_lines - 1)) * (self.x_max - self.x_min)
            label = str(int(x_val))
            
            painter.setPen(self.text_color)
            painter.drawText(
                QtCore.QRect(int(x - 30), rect.bottom() + 5, 60, 20),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                label
            )
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        
        # Draw axis labels
        painter.setPen(self.text_color)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        
        # Y-axis label
        painter.save()
        painter.translate(15, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(
            QtCore.QRect(-50, -10, 100, 20),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self.y_label
        )
        painter.restore()
        
        # X-axis label
        painter.drawText(
            QtCore.QRect(0, self.height() - 20, self.width(), 20),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "Step"
        )
    
    def _draw_title(self, painter):
        """Draw the graph title."""
        painter.setPen(self.title_color)
        font = painter.font()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        
        painter.drawText(
            QtCore.QRect(0, 5, self.width(), 25),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            self.title
        )
    
    def _draw_legend(self, painter):
        """Draw the legend."""
        if not self.lines:
            return
        
        legend_x = self.width() - self.padding['right'] - 120
        legend_y = self.padding['top'] + 10
        
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        for line in self.lines:
            if not line['data']:
                continue
            
            # Draw colored line with appropriate thickness
            painter.setPen(QtGui.QPen(line['color'], line['linewidth']))
            painter.drawLine(legend_x, legend_y + 5, legend_x + 20, legend_y + 5)
            
            # Draw label
            painter.setPen(self.text_color)
            painter.drawText(
                QtCore.QRect(legend_x + 25, legend_y, 80, 15),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                line['label']
            )
            
            legend_y += 20
    
    def _draw_data_lines(self, painter):
        """Draw the actual data lines."""
        for line in self.lines:
            if len(line['data']) < 2:
                continue
            
            # Use the line's specified width
            painter.setPen(QtGui.QPen(line['color'], line['linewidth']))
            
            points = [self._to_screen_coords(x, y) for x, y in line['data']]
            
            # Draw line segments
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
    
    def _format_number(self, value):
        """Format number for display."""
        if abs(value) < 0.01 or abs(value) > 10000:
            return f"{value:.1e}"
        elif abs(value) < 1:
            return f"{value:.4f}"
        else:
            return f"{value:.2f}"


class LiveMetricsWidget(QtWidgets.QWidget):
    """Widget for displaying live training metrics with custom QPainter graphs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_points = 500
        
        # Performance optimization
        self.pending_update = False
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._perform_update)
        self.update_interval_ms = 500
        
        # Pending data buffer
        self.pending_data = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Control buttons
        control_layout = QtWidgets.QHBoxLayout()
        
        self.clear_button = QtWidgets.QPushButton("Clear Data")
        self.clear_button.clicked.connect(self.clear_data)
        control_layout.addWidget(self.clear_button)
        
        self.pause_button = QtWidgets.QPushButton("Pause Updates")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        control_layout.addWidget(self.pause_button)
        
        control_layout.addWidget(QtWidgets.QLabel("Update Speed:"))
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["Fast (100ms)", "Normal (500ms)", "Slow (1000ms)", "Very Slow (2000ms)"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        control_layout.addWidget(self.speed_combo)
        
        control_layout.addStretch()
        
        self.stats_label = QtWidgets.QLabel("No data yet")
        self.stats_label.setStyleSheet("color: #ab97e6; font-weight: bold;")
        control_layout.addWidget(self.stats_label)
        
        layout.addLayout(control_layout)
        
        # Create graph panels
        self.lr_graph = GraphPanel("Learning Rate", "Learning Rate")
        self.lr_line_idx = self.lr_graph.add_line("#6a48d7", "LR", self.max_points)
        layout.addWidget(self.lr_graph, 1)

        self.grad_graph = GraphPanel("Gradient Norms", "Gradient Norm")
        self.grad_raw_idx = self.grad_graph.add_line("#e53935", "Raw", self.max_points, linewidth=4)
        self.grad_clipped_idx = self.grad_graph.add_line("#ffdd57", "Clipped", self.max_points, linewidth=2)
        layout.addWidget(self.grad_graph, 1)

        self.loss_graph = GraphPanel("Training Loss", "Loss")
        self.loss_line_idx = self.loss_graph.add_line("#ab97e6", "Loss", self.max_points)
        layout.addWidget(self.loss_graph, 1)
        
        # Store latest values for stats
        self.latest_step = 0
        self.latest_lr = 0.0
        self.latest_loss = 0.0
        self.latest_grad = 0.0



    def _on_pause_toggled(self, checked):
        """Handle pause button toggle."""
        if checked:
            self.update_timer.stop()
        else:
            if self.pending_update:
                self.update_timer.start(self.update_interval_ms)
    
    def _on_speed_changed(self, index):
        """Update the refresh interval based on speed selection."""
        speeds = [100, 500, 1000, 2000]
        self.update_interval_ms = speeds[index]
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.update_timer.start(self.update_interval_ms)
    
    def parse_and_update(self, text):
        """Parse console output and extract metrics. Batches updates for performance."""
        if self.pause_button.isChecked():
            return
        
        data_added = False
        
        # Parse step report format:
        # --- Step: {step} | Loss: {loss} | LR: {lr} ---
        step_match = re.search(r'--- Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*LR:\s*([\d.e+-]+)\s*---', text)
        if step_match:
            step = int(step_match.group(1))
            loss = float(step_match.group(2))
            lr = float(step_match.group(3))
            
            self.pending_data.append(('step', step, loss, lr))
            self.latest_step = step
            self.latest_lr = lr
            self.latest_loss = loss
            data_added = True
        
        # Parse gradient norm format:
        # Grad Norm (Raw/Clipped): {raw} / {clipped}
        grad_match = re.search(r'Grad Norm \(Raw/Clipped\):\s*([\d.]+)\s*/\s*([\d.]+)', text)
        if grad_match:
            raw_norm = float(grad_match.group(1))
            clipped_norm = float(grad_match.group(2))
            
            self.pending_data.append(('grad', raw_norm, clipped_norm))
            self.latest_grad = raw_norm
            data_added = True
        
        if data_added:
            self.pending_update = True
            # Start timer if not already running
            if not self.update_timer.isActive() and not self.pause_button.isChecked():
                self.update_timer.start(self.update_interval_ms)
    
    def _perform_update(self):
        """Actually perform the plot update. Called by timer."""
        if not self.pending_update or not self.pending_data:
            self.update_timer.stop()
            return
        
        # Process all pending data
        last_step = None
        for data in self.pending_data:
            if data[0] == 'step':
                _, step, loss, lr = data
                last_step = step
                self.lr_graph.append_data(self.lr_line_idx, step, lr)
                self.loss_graph.append_data(self.loss_line_idx, step, loss)
            elif data[0] == 'grad' and last_step is not None:
                _, raw_norm, clipped_norm = data
                self.grad_graph.append_data(self.grad_raw_idx, last_step, raw_norm)
                self.grad_graph.append_data(self.grad_clipped_idx, last_step, clipped_norm)
        
        # Clear pending data
        self.pending_data.clear()
        
        # Update stats label
        self.stats_label.setText(
            f"Latest - Step: {self.latest_step} | LR: {self.latest_lr:.2e} | "
            f"Loss: {self.latest_loss:.4f} | Grad: {self.latest_grad:.4f}"
        )
        
        # Trigger repaints
        self.lr_graph.update()
        self.grad_graph.update()
        self.loss_graph.update()
        
        self.pending_update = False
    
    def clear_data(self):
        """Clear all stored data and reset plots."""
        self.update_timer.stop()
        self.pending_update = False
        self.pending_data.clear()
        
        # Clear all graphs
        self.lr_graph.clear_all_data()
        self.grad_graph.clear_all_data()
        self.loss_graph.clear_all_data()
        
        # Reset stats
        self.latest_step = 0
        self.latest_lr = 0.0
        self.latest_loss = 0.0
        self.latest_grad = 0.0
        
        self.stats_label.setText("No data yet")
    
    def showEvent(self, event):
        """Resume updates when tab becomes visible."""
        super().showEvent(event)
        if self.pending_update and not self.pause_button.isChecked():
            self.update_timer.start(self.update_interval_ms)
    
    def hideEvent(self, event):
        """Pause updates when tab is hidden to save CPU."""
        super().hideEvent(event)
        self.update_timer.stop()


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
        if flat_part_duration < 0: return
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
    metricsSignal = pyqtSignal(str)
    cacheCreatedSignal = pyqtSignal()  # NEW: Signal for cache creation
    
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
            if os.name == 'nt':
                flags |= (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.HIGH_PRIORITY_CLASS)
            self.process = subprocess.Popen(
                [self.executable] + self.args,
                cwd=self.working_dir,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                creationflags=flags
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
                    
                    # Emit for metrics parsing
                    self.metricsSignal.emit(line)
                    
                    # NEW: Detect cache creation
                    if "saved latents cache" in line.lower() or "caching complete" in line.lower():
                        self.cacheCreatedSignal.emit()
            
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
        "VAE_PATH": {"label": "Separate VAE (Optional)", "tooltip": "Path to a separate VAE file. Leave empty to use the VAE from the base model.", "widget": "Path", "file_type": "file_safetensors"},
        "OUTPUT_DIR": {"label": "Output Directory", "tooltip": "Folder where checkpoints will be saved.", "widget": "Path", "file_type": "folder"},
        "CACHING_BATCH_SIZE": {"label": "Caching Batch Size", "tooltip": "Adjust based on VRAM (e.g., 2-8).", "widget": "QSpinBox", "range": (1, 64)},
        "NUM_WORKERS": {"label": "Dataloader Workers", "tooltip": "Set to 0 on Windows if you have issues.", "widget": "QSpinBox", "range": (0, 16)},
        "TARGET_PIXEL_AREA": {"label": "Target Pixel Area", "tooltip": "e.g., 1024*1024=1048576. Buckets are resolutions near this total area.", "widget": "QLineEdit"},
        "SHOULD_UPSCALE": {"label": "Upscale Images", "tooltip": "If enabled, upscale small images closer to bucket limit while maintaining aspect ratio.", "widget": "QCheckBox"},
        "MAX_AREA_TOLERANCE": {"label": "Max Area Tolerance:", "tooltip": "When upscaling, allow up to this multiplier over target area (e.g., 1.1 = 10% over).", "widget": "QLineEdit"},
        "PREDICTION_TYPE": {"label": "Prediction Type:", "tooltip": "v_prediction or epsilon. Must match the base model.", "widget": "QComboBox", "options": ["v_prediction", "epsilon"]},
        "BETA_SCHEDULE": {"label": "Beta Schedule:", "tooltip": "Noise schedule for the diffuser.", "widget": "QComboBox", "options": ["scaled_linear", "linear", "squared", "squaredcos_cap_v2"]},
        "MAX_TRAIN_STEPS": {"label": "Max Training Steps:", "tooltip": "Total number of training steps.", "widget": "QLineEdit"},
        "LEARNING_RATE": {"label": "Base Learning Rate:", "tooltip": "The base learning rate (not used if custom curve is active, kept for future reference).", "widget": "QLineEdit"},
        "BATCH_SIZE": {"label": "Batch Size:", "tooltip": "Number of samples per batch.", "widget": "QSpinBox", "range": (1, 32)},
        "SAVE_EVERY_N_STEPS": {"label": "Save Every N Steps:", "tooltip": "How often to save a checkpoint.", "widget": "QLineEdit"},
        "GRADIENT_ACCUMULATION_STEPS": {"label": "Gradient Accumulation:", "tooltip": "Simulates a larger batch size.", "widget": "QLineEdit"},
        "MIXED_PRECISION": {"label": "Mixed Precision:", "tooltip": "bfloat16 for modern GPUs, float16 for older.", "widget": "QComboBox", "options": ["bfloat16", "float16"]},
        "CLIP_GRAD_NORM": {"label": "Gradient Clipping:", "tooltip": "Maximum gradient norm. Set to 0 to disable.", "widget": "QLineEdit"},
        "SEED": {"label": "Seed:", "tooltip": "Ensures reproducible training.", "widget": "QLineEdit"},
        "RESUME_MODEL_PATH": {"label": "Resume Model:", "tooltip": "The .safetensors checkpoint file.", "widget": "Path", "file_type": "file_safetensors"},
        "RESUME_STATE_PATH": {"label": "Resume State:", "tooltip": "The .pt optimizer state file.", "widget": "Path", "file_type": "file_pt"},
        "UNET_EXCLUDE_TARGETS": {"label": "Exclude Layers (Keywords):", "tooltip": "Comma-separated keywords for layers to exclude from training (e.g., 'conv1, conv2, norm').", "widget": "QLineEdit"},
        "LR_GRAPH_MIN": {"label": "Graph Min LR:", "tooltip": "The minimum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "LR_GRAPH_MAX": {"label": "Graph Max LR:", "tooltip": "The maximum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "NOISE_SCHEDULER": {"label": "Noise Scheduler:", "tooltip": "The noise scheduler to use for training. EulerDiscrete is experimental.", "widget": "QComboBox", "options": ["DDPMScheduler", "DDIMScheduler", "EulerDiscreteScheduler (Experimental)"]},
        "MEMORY_EFFICIENT_ATTENTION": {"label": "Attention Backend:", "tooltip": "Select the attention mechanism to use.", "widget": "QComboBox", "options": ["xformers", "sdpa"]},
        "USE_ZERO_TERMINAL_SNR": {"label": "Use Zero-Terminal SNR", "tooltip": "Rescales noise schedule for better dynamic range.", "widget": "QCheckBox"},
        "GRAD_SPIKE_THRESHOLD_HIGH": {"label": "Spike Threshold (High):", "tooltip": "Trigger detector if gradient norm exceeds this value.", "widget": "QLineEdit"},
        "GRAD_SPIKE_THRESHOLD_LOW": {"label": "Spike Threshold (Low):", "tooltip": "Trigger detector if gradient norm is below this value.", "widget": "QLineEdit"},
        
        # NEW: Noise Enhancement Parameters
        "NOISE_OFFSET": {"label": "Noise Offset:", "tooltip": "Improves learning of very dark/bright images. Range: 0.0-0.15. Try 0.05 first, 0.1 for high-contrast styles.", "widget": "QLineEdit"},
        "USE_PYRAMID_NOISE": {"label": "Use Multi-Res Noise", "tooltip": "Helps learn multi-scale details. Good for complex lighting and artistic styles.", "widget": "QCheckBox"},
        "PYRAMID_DISCOUNT": {"label": "Pyramid Discount:", "tooltip": "How much each pyramid level contributes (0.8-0.95). Higher = more fine detail contribution.", "widget": "QLineEdit"},
        "PYRAMID_ITERATIONS": {"label": "Pyramid Iterations:", "tooltip": "Number of pyramid levels (5-12). More = better multi-scale learning, slightly slower.", "widget": "QSpinBox", "range": (1, 20)},
    }

    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA SDXL Trainer (Simplified)")
        self.setMinimumSize(QtCore.QSize(1000, 800))
        self.resize(1350, 1000)
        self.config_dir = "configs"
        self.widgets = {}
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
        self.param_info_label = QtWidgets.QLabel("Parameters: (awaiting training start)")
        self.param_info_label.setObjectName("ParamInfoLabel")
        self.param_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.param_info_label.setContentsMargins(0, 5, 0, 0)
        self.log_textbox = QtWidgets.QTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setMinimumHeight(200)
        self.tab_view = QtWidgets.QTabWidget()
        dataset_content_widget = QtWidgets.QWidget()
        self._populate_dataset_tab(dataset_content_widget)
        dataset_scroll = QtWidgets.QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setWidget(dataset_content_widget)
        self.tab_view.addTab(dataset_scroll, "Dataset")
        model_content_widget = QtWidgets.QWidget()
        self._populate_model_training_tab(model_content_widget)
        model_scroll = QtWidgets.QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setWidget(model_content_widget)
        self.tab_view.addTab(model_scroll, "Model && Training Parameters")
        
        self.live_metrics_widget = LiveMetricsWidget()
        self.tab_view.addTab(self.live_metrics_widget, "Live Metrics")
        
        console_tab_widget = QtWidgets.QWidget()
        console_layout = QtWidgets.QVBoxLayout(console_tab_widget)
        self._populate_console_tab(console_layout)
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
        config_to_save = {}
        for key in self.default_config.keys():
            if key in ["RESUME_TRAINING", "INSTANCE_DATASETS", "OPTIMIZER_TYPE", "RAVEN_PARAMS", "ADAFACTOR_PARAMS"]:
                continue
            
            live_val = self.current_config.get(key)
            if live_val is None:
                continue

            default_val = self.default_config.get(key)
            try:
                if key == "LR_CUSTOM_CURVE":
                    rounded_curve = [[round(p[0], 8), round(p[1], 10)] for p in live_val]
                    config_to_save[key] = rounded_curve
                    continue
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
                    else:
                        converted_val = str(live_val)
                else:
                    converted_val = str(live_val)
                config_to_save[key] = converted_val
            except (ValueError, TypeError) as e:
                self.log(f"Warning: Could not convert value for '{key}'. Not saved. Error: {e}")

        config_to_save["RESUME_TRAINING"] = self.model_load_strategy_combo.currentIndex() == 1
        config_to_save["INSTANCE_DATASETS"] = self.dataset_manager.get_datasets_config()
        config_to_save["OPTIMIZER_TYPE"] = self.widgets["OPTIMIZER_TYPE"].currentText().lower()

        # Save Raven params
        raven_params = {}
        try:
            betas_str = self.widgets['RAVEN_betas'].text().strip()
            raven_params["betas"] = [float(x.strip()) for x in betas_str.split(',')]
        except (ValueError, IndexError):
            raven_params["betas"] = default_config.RAVEN_PARAMS["betas"]
        try:
            raven_params["eps"] = float(self.widgets['RAVEN_eps'].text().strip())
        except ValueError:
            raven_params["eps"] = default_config.RAVEN_PARAMS["eps"]
        raven_params["weight_decay"] = self.widgets['RAVEN_weight_decay'].value()
        raven_params["debias_strength"] = self.widgets['RAVEN_debias_strength'].value()
        raven_params["use_grad_centralization"] = self.widgets['RAVEN_use_grad_centralization'].isChecked()
        raven_params["gc_alpha"] = self.widgets['RAVEN_gc_alpha'].value()
        config_to_save["RAVEN_PARAMS"] = raven_params

        # Save Adafactor params
        adafactor_params = {}
        try:
            eps_str = self.widgets['ADAFACTOR_eps'].text().strip()
            adafactor_params["eps"] = [float(x.strip()) for x in eps_str.split(',')]
        except (ValueError, IndexError):
            adafactor_params["eps"] = default_config.ADAFACTOR_PARAMS["eps"]
        adafactor_params["clip_threshold"] = self.widgets['ADAFACTOR_clip_threshold'].value()
        adafactor_params["decay_rate"] = self.widgets['ADAFACTOR_decay_rate'].value()
        if self.widgets['ADAFACTOR_beta1_enabled'].isChecked():
            adafactor_params["beta1"] = self.widgets['ADAFACTOR_beta1_value'].value()
        else:
            adafactor_params["beta1"] = None
        adafactor_params["weight_decay"] = self.widgets['ADAFACTOR_weight_decay'].value()
        adafactor_params["scale_parameter"] = self.widgets['ADAFACTOR_scale_parameter'].isChecked()
        adafactor_params["relative_step"] = self.widgets['ADAFACTOR_relative_step'].isChecked()
        adafactor_params["warmup_init"] = self.widgets['ADAFACTOR_warmup_init'].isChecked()
        config_to_save["ADAFACTOR_PARAMS"] = adafactor_params

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
        elif widget_type == "QSpinBox":
            widget = QtWidgets.QSpinBox()
            if "range" in definition:
                widget.setRange(*definition["range"])
            widget.valueChanged.connect(lambda value, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "QComboBox":
            widget = QtWidgets.QComboBox()
            widget.addItems(definition["options"])
            widget.currentTextChanged.connect(lambda text, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "QCheckBox":
            widget = QtWidgets.QCheckBox()
            widget.stateChanged.connect(lambda state, k=key: self._update_config_from_widget(k, widget))
            self.widgets[key] = widget
            return label, widget
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
            "Aspect Ratio Bucketing": ["TARGET_PIXEL_AREA", "SHOULD_UPSCALE", "MAX_AREA_TOLERANCE"]
        }
        
        for title, keys in groups.items():
            top_hbox.addWidget(self._create_form_group(title, keys))
        
        layout.addLayout(top_hbox)
        
        self.dataset_manager = DatasetManagerWidget(self)
        self.dataset_manager.datasetsChanged.connect(self._update_epoch_markers_on_graph)
        layout.addWidget(self.dataset_manager)
        
        if "SHOULD_UPSCALE" in self.widgets and "MAX_AREA_TOLERANCE" in self.widgets:
            self.widgets["SHOULD_UPSCALE"].stateChanged.connect(
                lambda state: self.widgets["MAX_AREA_TOLERANCE"].setEnabled(bool(state))
            )
    
    def _populate_model_training_tab(self, parent_widget):
        layout = QtWidgets.QHBoxLayout(parent_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 5, 15, 15)
        left_vbox = QtWidgets.QVBoxLayout()
        right_vbox = QtWidgets.QVBoxLayout()

        path_group = self._create_path_group()
        left_vbox.addWidget(path_group)

        core_group = self._create_core_training_group()
        left_vbox.addWidget(core_group)

        optimizer_group = self._create_optimizer_group()
        left_vbox.addWidget(optimizer_group)
        left_vbox.addStretch()

        lr_group = self._create_lr_scheduler_group()
        right_vbox.addWidget(lr_group)

        unet_group = self._create_unet_group()
        right_vbox.addWidget(unet_group)

        advanced_group = self._create_advanced_group()
        right_vbox.addWidget(advanced_group)

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
        label, widget = self._create_widget("VAE_PATH")
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
        core_group = QtWidgets.QGroupBox("Core Training Parameters")
        layout = QtWidgets.QFormLayout(core_group)
        core_keys = [
            "PREDICTION_TYPE", 
            "BETA_SCHEDULE", 
            "MAX_TRAIN_STEPS",
            "LEARNING_RATE",
            "BATCH_SIZE",
            "SAVE_EVERY_N_STEPS", 
            "GRADIENT_ACCUMULATION_STEPS",
            "CLIP_GRAD_NORM",
            "MIXED_PRECISION", 
            "SEED"
        ]
        for key in core_keys:
            label, widget = self._create_widget(key)
            layout.addRow(label, widget)
        return core_group

    def _create_optimizer_group(self):
        optimizer_group = QtWidgets.QGroupBox("Optimizer")
        main_layout = QtWidgets.QVBoxLayout(optimizer_group)
        
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel("Optimizer Type:"))
        
        self.widgets["OPTIMIZER_TYPE"] = QtWidgets.QComboBox()
        self.widgets["OPTIMIZER_TYPE"].addItems(["Raven", "Adafactor"])
        self.widgets["OPTIMIZER_TYPE"].currentTextChanged.connect(self._toggle_optimizer_widgets)
        selector_layout.addWidget(self.widgets["OPTIMIZER_TYPE"], 1)
        main_layout.addLayout(selector_layout)

        self.raven_settings_group = QtWidgets.QGroupBox("Raven Settings")
        raven_layout = QtWidgets.QFormLayout(self.raven_settings_group)
        
        self.widgets['RAVEN_betas'] = QtWidgets.QLineEdit()
        self.widgets['RAVEN_eps'] = QtWidgets.QLineEdit()
        self.widgets['RAVEN_weight_decay'] = QtWidgets.QDoubleSpinBox()
        self.widgets['RAVEN_weight_decay'].setRange(0.0, 1.0)
        self.widgets['RAVEN_weight_decay'].setSingleStep(0.001)
        self.widgets['RAVEN_weight_decay'].setDecimals(3)

        self.widgets['RAVEN_debias_strength'] = QtWidgets.QDoubleSpinBox()
        self.widgets['RAVEN_debias_strength'].setRange(0.0, 1.0)
        self.widgets['RAVEN_debias_strength'].setSingleStep(0.01)
        self.widgets['RAVEN_debias_strength'].setDecimals(3)
        self.widgets['RAVEN_debias_strength'].setToolTip("Controls the strength of bias correction. 1.0 = full correction, 0.3 = 30% (softer start).")
        
        self.widgets['RAVEN_use_grad_centralization'] = QtWidgets.QCheckBox("Enable Gradient Centralization")
        self.widgets['RAVEN_use_grad_centralization'].setToolTip("Improves convergence by centering gradients. Recommended for better training stability.")
        
        self.widgets['RAVEN_gc_alpha'] = QtWidgets.QDoubleSpinBox()
        self.widgets['RAVEN_gc_alpha'].setRange(0.0, 1.0)
        self.widgets['RAVEN_gc_alpha'].setSingleStep(0.1)
        self.widgets['RAVEN_gc_alpha'].setDecimals(1)
        self.widgets['RAVEN_gc_alpha'].setToolTip("Strength of gradient centralization. 1.0 = full strength, 0.5 = half strength.")
        
        self.widgets['RAVEN_use_grad_centralization'].stateChanged.connect(
            lambda state: self.widgets['RAVEN_gc_alpha'].setEnabled(bool(state))
        )
        
        raven_layout.addRow("Betas (b1, b2):", self.widgets['RAVEN_betas'])
        raven_layout.addRow("Epsilon (eps):", self.widgets['RAVEN_eps'])
        raven_layout.addRow("Weight Decay:", self.widgets['RAVEN_weight_decay'])
        raven_layout.addRow("Debias Strength:", self.widgets['RAVEN_debias_strength'])
        raven_layout.addRow(self.widgets['RAVEN_use_grad_centralization'])
        raven_layout.addRow("GC Alpha:", self.widgets['RAVEN_gc_alpha'])
        
        main_layout.addWidget(self.raven_settings_group)
        
        self.adafactor_settings_group = QtWidgets.QGroupBox("Adafactor Settings")
        adafactor_layout = QtWidgets.QFormLayout(self.adafactor_settings_group)
        
        self.widgets['ADAFACTOR_eps'] = QtWidgets.QLineEdit()
        self.widgets['ADAFACTOR_clip_threshold'] = QtWidgets.QDoubleSpinBox()
        self.widgets['ADAFACTOR_clip_threshold'].setRange(0.1, 10.0)
        self.widgets['ADAFACTOR_clip_threshold'].setSingleStep(0.1)
        self.widgets['ADAFACTOR_clip_threshold'].setDecimals(2)
        
        self.widgets['ADAFACTOR_decay_rate'] = QtWidgets.QDoubleSpinBox()
        self.widgets['ADAFACTOR_decay_rate'].setRange(-1.0, 0.0)
        self.widgets['ADAFACTOR_decay_rate'].setSingleStep(0.01)
        self.widgets['ADAFACTOR_decay_rate'].setDecimals(3)
        
        beta1_widget = QtWidgets.QWidget()
        beta1_layout = QtWidgets.QHBoxLayout(beta1_widget)
        beta1_layout.setContentsMargins(0,0,0,0)
        self.widgets['ADAFACTOR_beta1_enabled'] = QtWidgets.QCheckBox("Enable")
        self.widgets['ADAFACTOR_beta1_value'] = QtWidgets.QDoubleSpinBox()
        self.widgets['ADAFACTOR_beta1_value'].setRange(0.0, 1.0)
        self.widgets['ADAFACTOR_beta1_value'].setSingleStep(0.01)
        self.widgets['ADAFACTOR_beta1_value'].setDecimals(3)
        beta1_layout.addWidget(self.widgets['ADAFACTOR_beta1_enabled'])
        beta1_layout.addWidget(self.widgets['ADAFACTOR_beta1_value'], 1)
        self.widgets['ADAFACTOR_beta1_enabled'].stateChanged.connect(
            lambda state: self.widgets['ADAFACTOR_beta1_value'].setEnabled(bool(state))
        )

        self.widgets['ADAFACTOR_weight_decay'] = QtWidgets.QDoubleSpinBox()
        self.widgets['ADAFACTOR_weight_decay'].setRange(0.0, 1.0)
        self.widgets['ADAFACTOR_weight_decay'].setSingleStep(0.001)
        self.widgets['ADAFACTOR_weight_decay'].setDecimals(3)
        
        self.widgets['ADAFACTOR_scale_parameter'] = QtWidgets.QCheckBox("Scale Parameter")
        self.widgets['ADAFACTOR_relative_step'] = QtWidgets.QCheckBox("Relative Step")
        self.widgets['ADAFACTOR_warmup_init'] = QtWidgets.QCheckBox("Warmup Init")
        
        adafactor_layout.addRow("Eps (e1, e2):", self.widgets['ADAFACTOR_eps'])
        adafactor_layout.addRow("Clip Threshold:", self.widgets['ADAFACTOR_clip_threshold'])
        adafactor_layout.addRow("Decay Rate:", self.widgets['ADAFACTOR_decay_rate'])
        adafactor_layout.addRow("Beta1:", beta1_widget)
        adafactor_layout.addRow("Weight Decay:", self.widgets['ADAFACTOR_weight_decay'])
        adafactor_layout.addRow(self.widgets['ADAFACTOR_scale_parameter'])
        adafactor_layout.addRow(self.widgets['ADAFACTOR_relative_step'])
        adafactor_layout.addRow(self.widgets['ADAFACTOR_warmup_init'])
        
        main_layout.addWidget(self.adafactor_settings_group)
        
        return optimizer_group
    
    def _toggle_optimizer_widgets(self):
        selected_optimizer = self.widgets["OPTIMIZER_TYPE"].currentText()
        is_raven = (selected_optimizer == "Raven")
        self.raven_settings_group.setVisible(is_raven)
        self.adafactor_settings_group.setVisible(not is_raven)

    def _create_lr_scheduler_group(self):
        lr_group = QtWidgets.QGroupBox("Learning Rate Scheduler")
        lr_layout = QtWidgets.QVBoxLayout(lr_group)
        
        self.lr_curve_widget = LRCurveWidget()
        self.widgets['LR_CUSTOM_CURVE'] = self.lr_curve_widget
        self.lr_curve_widget.pointsChanged.connect(
            lambda pts: self._update_config_from_widget("LR_CUSTOM_CURVE", self.lr_curve_widget)
        )
        self.lr_curve_widget.selectionChanged.connect(self._update_lr_button_states)
        lr_layout.addWidget(self.lr_curve_widget)
        
        lr_controls_layout = QtWidgets.QHBoxLayout()
        self.add_point_btn = QtWidgets.QPushButton("Add Point")
        self.add_point_btn.clicked.connect(self.lr_curve_widget.add_point)
        self.remove_point_btn = QtWidgets.QPushButton("Remove Selected")
        self.remove_point_btn.clicked.connect(self.lr_curve_widget.remove_selected_point)
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
    
    def _create_unet_group(self):
        unet_group = QtWidgets.QGroupBox("UNet Layer Exclusion")
        layout = QtWidgets.QVBoxLayout(unet_group)
        
        info_label = QtWidgets.QLabel(
            "Enter comma-separated keywords for layers to <b>exclude</b> from training.<br>"
            "Example: <i>conv1, conv2, norm</i><br><br>"
            "Any layer name containing these keywords will be frozen."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #e0e0e0; padding: 5px;")
        layout.addWidget(info_label)
        
        label, widget = self._create_widget("UNET_EXCLUDE_TARGETS")
        layout.addWidget(label)
        layout.addWidget(widget)
        
        return unet_group
    
    def _create_advanced_group(self):
            """Creates the 'Advanced Settings' group box with visual separators and subheadings."""
            advanced_group = QtWidgets.QGroupBox("Advanced Settings")
            layout = QtWidgets.QFormLayout(advanced_group)

            # --- General Advanced Settings ---
            label, widget = self._create_widget("NOISE_SCHEDULER")
            layout.addRow(label, widget)

            label, widget = self._create_widget("MEMORY_EFFICIENT_ATTENTION")
            layout.addRow(label, widget)

            label, widget = self._create_widget("USE_ZERO_TERMINAL_SNR")
            layout.addRow(label, widget)

            # --- Separator and Noise Enhancements Subheading ---
            separator1 = QtWidgets.QFrame()
            separator1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator1.setStyleSheet("border: 1px solid #4a4668; margin: 10px 0;")
            layout.addRow(separator1)

            noise_heading = QtWidgets.QLabel("<b>Noise Enhancements</b>")
            noise_heading.setStyleSheet("color: #ab97e6; margin-top: 5px;")
            layout.addRow(noise_heading)
            
            noise_info_label = QtWidgets.QLabel("<i>Techniques to improve learning on high-contrast or artistic datasets.</i>")
            noise_info_label.setStyleSheet("color: #7a788c; font-size: 12px; margin-bottom: 5px;")
            layout.addRow(noise_info_label)

            # --- Noise Enhancement Widgets ---
            label, widget = self._create_widget("NOISE_OFFSET")
            layout.addRow(label, widget)

            label, widget = self._create_widget("USE_PYRAMID_NOISE")
            layout.addRow(label, widget)

            label, widget = self._create_widget("PYRAMID_DISCOUNT")
            layout.addRow(label, widget)

            label, widget = self._create_widget("PYRAMID_ITERATIONS")
            layout.addRow(label, widget)

            # Connect checkbox to enable/disable related pyramid settings
            if "USE_PYRAMID_NOISE" in self.widgets:
                self.widgets["USE_PYRAMID_NOISE"].stateChanged.connect(
                    lambda state: self._toggle_pyramid_settings(bool(state))
                )

            # --- Separator and Gradient Spike Detection Subheading ---
            separator2 = QtWidgets.QFrame()
            separator2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator2.setStyleSheet("border: 1px solid #4a4668; margin: 10px 0;")
            layout.addRow(separator2)

            spike_heading = QtWidgets.QLabel("<b>Gradient Spike Detection</b>")
            spike_heading.setStyleSheet("color: #ab97e6; margin-top: 5px;")
            layout.addRow(spike_heading)

            # --- Gradient Spike Detection Widgets ---
            label, widget = self._create_widget("GRAD_SPIKE_THRESHOLD_HIGH")
            layout.addRow(label, widget)

            label, widget = self._create_widget("GRAD_SPIKE_THRESHOLD_LOW")
            layout.addRow(label, widget)

            return advanced_group

    
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
        button_layout = QtWidgets.QHBoxLayout()
        
        clear_button = QtWidgets.QPushButton("Clear Console")
        clear_button.clicked.connect(self.clear_console_log)
        button_layout.addWidget(clear_button)
        button_layout.addStretch()
        layout.addLayout(button_layout, stretch=0)
    
    def _browse_path(self, entry_widget, file_type):
        path = ""
        current_path = os.path.dirname(entry_widget.text()) if entry_widget.text() else ""
        if file_type == "folder":
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", current_path)
        elif file_type == "file_safetensors":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", current_path, "Safetensors Files (*.safetensors)")
        elif file_type == "file_pt":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select State", current_path, "PyTorch State Files (*.pt)")
        if path:
            entry_widget.setText(path.replace('\\', '/'))
    
    def _update_config_from_widget(self, key, widget):
        if isinstance(widget, QtWidgets.QLineEdit):
            self.current_config[key] = widget.text().strip()
        elif isinstance(widget, QtWidgets.QCheckBox):
            self.current_config[key] = widget.isChecked()
        elif isinstance(widget, QtWidgets.QComboBox):
            self.current_config[key] = widget.currentText()
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            self.current_config[key] = widget.value()
        elif isinstance(widget, LRCurveWidget):
            self.current_config[key] = widget.get_points()
    
    def _apply_config_to_widgets(self):
        for widget in self.widgets.values():
            widget.blockSignals(True)
        try:
            if hasattr(self, 'model_load_strategy_combo'):
                is_resuming = self.current_config.get("RESUME_TRAINING", False)
                self.model_load_strategy_combo.setCurrentIndex(1 if is_resuming else 0)
                self.toggle_resume_widgets()
            
            for key, widget in self.widgets.items():
                if key in ["OPTIMIZER_TYPE", "LR_CUSTOM_CURVE"] or key.startswith(("RAVEN_", "ADAFACTOR_")):
                    continue
                
                value = self.current_config.get(key)
                if value is None: continue

                if isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QtWidgets.QComboBox):
                    widget.setCurrentText(str(value))
                elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    widget.setValue(float(value) if isinstance(widget, QtWidgets.QDoubleSpinBox) else int(value))

            optimizer_type = self.current_config.get("OPTIMIZER_TYPE", default_config.OPTIMIZER_TYPE)
            self.widgets["OPTIMIZER_TYPE"].setCurrentText(optimizer_type.capitalize())
            
            user_raven_params = self.current_config.get("RAVEN_PARAMS", {})
            full_raven_params = {**default_config.RAVEN_PARAMS, **user_raven_params}
            self.widgets["RAVEN_betas"].setText(', '.join(map(str, full_raven_params["betas"])))
            self.widgets["RAVEN_eps"].setText(str(full_raven_params["eps"]))
            self.widgets["RAVEN_weight_decay"].setValue(full_raven_params["weight_decay"])
            self.widgets["RAVEN_debias_strength"].setValue(full_raven_params.get("debias_strength", 1.0))
            
            use_gc = full_raven_params.get("use_grad_centralization", False)
            gc_alpha = full_raven_params.get("gc_alpha", 1.0)
            self.widgets["RAVEN_use_grad_centralization"].setChecked(use_gc)
            self.widgets["RAVEN_gc_alpha"].setValue(gc_alpha)
            self.widgets["RAVEN_gc_alpha"].setEnabled(use_gc)

            user_adafactor_params = self.current_config.get("ADAFACTOR_PARAMS", {})
            full_adafactor_params = {**default_config.ADAFACTOR_PARAMS, **user_adafactor_params}
            self.widgets["ADAFACTOR_eps"].setText(', '.join(map(str, full_adafactor_params["eps"])))
            self.widgets["ADAFACTOR_clip_threshold"].setValue(full_adafactor_params["clip_threshold"])
            self.widgets["ADAFACTOR_decay_rate"].setValue(full_adafactor_params["decay_rate"])
            beta1_val = full_adafactor_params.get("beta1", None)
            beta1_enabled = beta1_val is not None
            self.widgets["ADAFACTOR_beta1_enabled"].setChecked(beta1_enabled)
            self.widgets["ADAFACTOR_beta1_value"].setValue(beta1_val if beta1_enabled else 0.0)
            self.widgets["ADAFACTOR_beta1_value"].setEnabled(beta1_enabled)
            self.widgets["ADAFACTOR_weight_decay"].setValue(full_adafactor_params["weight_decay"])
            self.widgets["ADAFACTOR_scale_parameter"].setChecked(full_adafactor_params["scale_parameter"])
            self.widgets["ADAFACTOR_relative_step"].setChecked(full_adafactor_params["relative_step"])
            self.widgets["ADAFACTOR_warmup_init"].setChecked(full_adafactor_params["warmup_init"])
            
            if "SHOULD_UPSCALE" in self.widgets and "MAX_AREA_TOLERANCE" in self.widgets:
                should_upscale = self.current_config.get("SHOULD_UPSCALE", False)
                self.widgets["MAX_AREA_TOLERANCE"].setEnabled(bool(should_upscale))

            if "USE_PYRAMID_NOISE" in self.widgets and "PYRAMID_DISCOUNT" in self.widgets:
                use_pyramid = self.current_config.get("USE_PYRAMID_NOISE", False)
                self.widgets["PYRAMID_DISCOUNT"].setEnabled(bool(use_pyramid))
                self.widgets["PYRAMID_ITERATIONS"].setEnabled(bool(use_pyramid))
                            
            if hasattr(self, 'lr_curve_widget'):
                self._update_and_clamp_lr_graph()
            
            self._toggle_optimizer_widgets()
            
            if hasattr(self, "dataset_manager"):
                datasets_config = self.current_config.get("INSTANCE_DATASETS", [])
                self.dataset_manager.load_datasets_from_config(datasets_config)

        finally:
            for widget in self.widgets.values():
                widget.blockSignals(False)

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
    
    def clear_console_log(self):
        self.log_textbox.clear()
        self.log("Console cleared.")
    
    def toggle_resume_widgets(self):
        if hasattr(self, 'resume_sub_widget') and hasattr(self, 'base_model_sub_widget'):
            is_resuming = self.model_load_strategy_combo.currentIndex() == 1
            self.resume_sub_widget.setVisible(is_resuming)
            self.base_model_sub_widget.setVisible(not is_resuming)

    def _toggle_pyramid_settings(self, enabled):
        """Enable or disable pyramid noise settings based on the checkbox state."""
        if "PYRAMID_DISCOUNT" in self.widgets:
            self.widgets["PYRAMID_DISCOUNT"].setEnabled(enabled)
        if "PYRAMID_ITERATIONS" in self.widgets:
            self.widgets["PYRAMID_ITERATIONS"].setEnabled(enabled)

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
        
        self.live_metrics_widget.clear_data()
        
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
        self.process_runner.metricsSignal.connect(self.live_metrics_widget.parse_and_update)
        
        # NEW: Connect cache creation signal to update dataset UI
        self.process_runner.cacheCreatedSignal.connect(self.dataset_manager.refresh_cache_buttons)
        
        if os.name == 'nt':
            prevent_sleep(True)
        self.process_runner.start()
        self.log(f"INFO: Starting train.py with config: {config_path}")
    
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
        

        if hasattr(self, 'dataset_manager'):
            self.dataset_manager.refresh_cache_buttons()
        
        if os.name == 'nt':
            prevent_sleep(False)

class NoScrollSpinBox(QtWidgets.QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

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
        
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.grid_container = QtWidgets.QWidget()
        self.dataset_grid = QtWidgets.QGridLayout(self.grid_container)
        self.dataset_grid.setSpacing(15)
        self.dataset_grid.setContentsMargins(5, 5, 5, 5)
        
        scroll_area.setWidget(self.grid_container)
        layout.addWidget(scroll_area)
        
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
    
    def get_total_repeats(self):
        return sum(ds["image_count"] * ds["repeats"] for ds in self.datasets)
    
    def get_datasets_config(self):
        return [{"path": ds["path"], "repeats": ds["repeats"]} for ds in self.datasets]
    
    def _load_dataset_images(self, path):
        """Load all images and their captions from a dataset path."""
        exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        images = [p for ext in exts for p in Path(path).rglob(f"*{ext}")]
        
        dataset_info = []
        for img_path in images:
            caption_path = img_path.with_suffix('.txt')
            caption = ""
            if caption_path.exists():
                try:
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                except:
                    caption = "[Error reading caption]"
            else:
                caption = "[No caption file]"
            
            dataset_info.append({
                "image_path": str(img_path),
                "caption": caption
            })
        
        return dataset_info
    
    def _cache_exists(self, path):
        """Check if cache directory exists."""
        cache_dir = Path(path) / ".precomputed_embeddings_cache"
        return cache_dir.exists() and cache_dir.is_dir()
    
    def load_datasets_from_config(self, datasets_config):
        self.datasets = []
        for d in datasets_config:
            path = d.get("path")
            if path and os.path.exists(path):
                images_data = self._load_dataset_images(path)
                if images_data:
                    self.datasets.append({
                        "path": path,
                        "images_data": images_data,
                        "image_count": len(images_data),
                        "current_preview_idx": 0,
                        "repeats": d.get("repeats", 1),
                    })
        self.repopulate_dataset_grid()
        self.update_dataset_totals()
    
    def add_dataset_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder", "")
        if not path: return
        
        images_data = self._load_dataset_images(path)
        if not images_data:
            QtWidgets.QMessageBox.warning(self, "No Images", "No valid images found in the folder.")
            return
        
        self.datasets.append({
            "path": path,
            "images_data": images_data,
            "image_count": len(images_data),
            "current_preview_idx": 0,
            "repeats": 1,
        })
        self.repopulate_dataset_grid()
        self.update_dataset_totals()
    
    def _cycle_preview(self, idx, direction):
        """Cycle through preview images. Direction: 1 for next, -1 for previous."""
        ds = self.datasets[idx]
        ds["current_preview_idx"] = (ds["current_preview_idx"] + direction) % len(ds["images_data"])
        self._update_preview_for_card(idx)
    
    def _update_preview_for_card(self, idx):
        """Update just the preview and caption for a specific card."""
        if idx >= len(self.dataset_widgets):
            return
        
        ds = self.datasets[idx]
        widgets = self.dataset_widgets[idx]
        
        current_data = ds["images_data"][ds["current_preview_idx"]]
        
        pixmap = QtGui.QPixmap(current_data["image_path"]).scaled(
            183, 183, 
            QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        widgets["preview_label"].setPixmap(pixmap)
        
        caption_text = current_data["caption"]
        if len(caption_text) > 200:
            caption_text = caption_text[:200] + "..."
        widgets["caption_label"].setText(caption_text)
        
        widgets["counter_label"].setText(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")
    
    def repopulate_dataset_grid(self):
        while self.dataset_grid.count():
            item = self.dataset_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.dataset_widgets = []
        
        dataset_count = len(self.datasets)
        if dataset_count == 1:
            COLUMNS = 1
        elif dataset_count == 2:
            COLUMNS = 2
        else:
            COLUMNS = 3
        
        for idx, ds in enumerate(self.datasets):
            row = idx // COLUMNS
            col = idx % COLUMNS
            
            card = QtWidgets.QGroupBox()
            card.setStyleSheet("""
                QGroupBox {
                    border: 2px solid #4a4668;
                    border-radius: 8px;
                    margin-top: 5px;
                    padding: 12px;
                    background-color: #383552;
                }
            """)
            
            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setSpacing(10)
            
            top_section = QtWidgets.QHBoxLayout()
            top_section.setSpacing(12)
            
            preview_section = QtWidgets.QVBoxLayout()
            preview_section.setSpacing(5)
            
            image_container = QtWidgets.QHBoxLayout()
            image_container.addStretch()
            
            preview_label = QtWidgets.QLabel()
            preview_label.setFixedSize(183, 183)
            preview_label.setStyleSheet("border: 1px solid #4a4668; background-color: #2c2a3e;")
            preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            preview_label.setScaledContents(False)
            
            current_data = ds["images_data"][ds["current_preview_idx"]]
            pixmap = QtGui.QPixmap(current_data["image_path"]).scaled(
                183, 183, 
                QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                QtCore.Qt.TransformationMode.SmoothTransformation
            )
            preview_label.setPixmap(pixmap)
            image_container.addWidget(preview_label)
            image_container.addStretch()
            
            preview_section.addLayout(image_container)
            
            counter_nav_layout = QtWidgets.QHBoxLayout()
            counter_nav_layout.setSpacing(8)
            
            left_arrow = QtWidgets.QPushButton("◄")
            left_arrow.setFixedHeight(22)
            left_arrow.setMinimumWidth(35)
            left_arrow.clicked.connect(lambda _, i=idx: self._cycle_preview(i, -1))
            left_arrow.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 0px;
                }
            """)
            counter_nav_layout.addWidget(left_arrow)
            
            counter_label = QtWidgets.QLabel(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")
            counter_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            counter_label.setStyleSheet("color: #ab97e6; font-size: 12px; font-weight: bold;")
            counter_nav_layout.addWidget(counter_label, 1)
            
            right_arrow = QtWidgets.QPushButton("►")
            right_arrow.setFixedHeight(22)
            right_arrow.setMinimumWidth(35)
            right_arrow.clicked.connect(lambda _, i=idx: self._cycle_preview(i, 1))
            right_arrow.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    padding: 0px;
                }
            """)
            counter_nav_layout.addWidget(right_arrow)
            
            preview_section.addLayout(counter_nav_layout)
            
            top_section.addLayout(preview_section)
            
            caption_container = QtWidgets.QWidget()
            caption_container.setStyleSheet("""
                QWidget {
                    background-color: #2c2a3e;
                    border: 1px solid #4a4668;
                    border-radius: 4px;
                }
            """)
            caption_layout = QtWidgets.QVBoxLayout(caption_container)
            caption_layout.setContentsMargins(8, 8, 8, 8)
            
            caption_title = QtWidgets.QLabel("<b>Caption Preview:</b>")
            caption_title.setStyleSheet("color: #ab97e6; font-size: 11px;")
            caption_layout.addWidget(caption_title)
            
            caption_label = QtWidgets.QLabel()
            caption_text = current_data["caption"]
            if len(caption_text) > 200:
                caption_text = caption_text[:200] + "..."
            caption_label.setText(caption_text)
            caption_label.setWordWrap(True)
            caption_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
            caption_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            caption_layout.addWidget(caption_label, 1)
            
            top_section.addWidget(caption_container, 1)
            card_layout.addLayout(top_section)
            
            separator = QtWidgets.QFrame()
            separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator.setStyleSheet("border: 1px solid #4a4668;")
            card_layout.addWidget(separator)
            
            path_label = QtWidgets.QLabel()
            path_short = Path(ds['path']).name
            if len(path_short) > 30:
                path_short = path_short[:27] + "..."
            path_label.setText(f"<b>Folder:</b> {path_short}")
            path_label.setToolTip(ds['path'])
            path_label.setStyleSheet("color: #e0e0e0;")
            card_layout.addWidget(path_label)
            
            count_label = QtWidgets.QLabel(f"<b>Images:</b> {ds['image_count']}")
            count_label.setStyleSheet("color: #e0e0e0;")
            card_layout.addWidget(count_label)
            
            repeats_total_label = QtWidgets.QLabel(f"<b>Total (with repeats):</b> {ds['image_count'] * ds['repeats']}")
            repeats_total_label.setStyleSheet("color: #ab97e6;")
            card_layout.addWidget(repeats_total_label)
            
            repeats_container = QtWidgets.QWidget()
            repeats_layout = QtWidgets.QHBoxLayout(repeats_container)
            repeats_layout.setContentsMargins(0, 5, 0, 0)
            repeats_layout.addWidget(QtWidgets.QLabel("Repeats:"))
            
            repeats_spin = NoScrollSpinBox()
            repeats_spin.setMinimum(1)
            repeats_spin.setMaximum(10000)
            repeats_spin.setValue(ds["repeats"])
            repeats_spin.setStyleSheet("""
                QSpinBox::up-button, QSpinBox::down-button {
                    width: 20px;
                }
            """)
            repeats_spin.valueChanged.connect(lambda v, i=idx: self.update_repeats(i, v))
            repeats_layout.addWidget(repeats_spin, 1)
            card_layout.addWidget(repeats_container)
            
            btn_layout = QtWidgets.QHBoxLayout()
            btn_layout.setSpacing(5)

            remove_btn = QtWidgets.QPushButton("Remove")
            remove_btn.setStyleSheet("min-height: 24px; max-height: 24px; padding: 4px 15px;")
            remove_btn.clicked.connect(lambda _, i=idx: self.remove_dataset(i))
            btn_layout.addWidget(remove_btn)

            clear_btn = QtWidgets.QPushButton("Clear Cache")
            clear_btn.setStyleSheet("min-height: 24px; max-height: 24px; padding: 4px 15px;")
            clear_btn.clicked.connect(lambda _, p=ds["path"]: self.confirm_clear_cache(p))

            cache_exists = self._cache_exists(ds["path"])
            clear_btn.setEnabled(cache_exists)
            if not cache_exists:
                clear_btn.setToolTip("No cache found")
                clear_btn.setStyleSheet("min-height: 24px; max-height: 24px; padding: 4px 15px;")

            btn_layout.addWidget(clear_btn)
            card_layout.addLayout(btn_layout)
            
            self.dataset_grid.addWidget(card, row, col)
            
            self.dataset_widgets.append({
                "preview_label": preview_label,
                "caption_label": caption_label,
                "counter_label": counter_label,
                "repeats_total_label": repeats_total_label,
                "clear_btn": clear_btn
            })
        
        if dataset_count % COLUMNS != 0:
            for empty_col in range((dataset_count % COLUMNS), COLUMNS):
                self.dataset_grid.setColumnStretch(empty_col, 1)

    def update_repeats(self, idx, val):
        self.datasets[idx]["repeats"] = val
        
        if idx < len(self.dataset_widgets):
            ds = self.datasets[idx]
            total = ds['image_count'] * ds['repeats']
            self.dataset_widgets[idx]["repeats_total_label"].setText(
                f"<b>Total (with repeats):</b> {total}"
            )
        
        self.update_dataset_totals()

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
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm", 
            "Delete all cached latents in this dataset?", 
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
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
        
        if deleted:
            self.refresh_cache_buttons()
        
        if not deleted:
            self.parent_gui.log("No cached latent directories found to delete.")
    
    def refresh_cache_buttons(self):
        """Refresh the enabled state of all Clear Cache buttons."""
        for idx, ds in enumerate(self.datasets):
            if idx < len(self.dataset_widgets):
                cache_exists = self._cache_exists(ds["path"])
                clear_btn = self.dataset_widgets[idx]["clear_btn"]
                clear_btn.setEnabled(cache_exists)
                if not cache_exists:
                    clear_btn.setToolTip("No cache found")
                else:
                    clear_btn.setToolTip("")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = TrainingGUI()
    main_win.show()
    sys.exit(app.exec())