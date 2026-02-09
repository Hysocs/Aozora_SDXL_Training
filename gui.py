
import json
import os
import re
from PyQt6 import QtWidgets, QtCore, QtGui
import subprocess
from PyQt6.QtCore import QThread, pyqtSignal, QObject
import copy
import sys
from pathlib import Path
import random
import shutil
import ctypes
import math
from collections import deque
from datetime import datetime
from PyQt6.QtWidgets import QFileIconProvider
from PyQt6.QtCore import QFileInfo

# Note: config import assumes a file named config.py exists in the same directory
try:
    import config as default_config
except ImportError:
    # Fallback if config.py is missing
    class default_config:
        RAVEN_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "debias_strength": 1.0, "use_grad_centralization": False, "gc_alpha": 1.0}
        TITAN_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "debias_strength": 1.0, "use_grad_centralization": False, "gc_alpha": 1.0}
        VELORMS_PARAMS = {"momentum": 0.86, "leak": 0.16, "weight_decay": 0.01, "eps": 1e-8}
        OPTIMIZER_TYPE = "Raven"
        TIMESTEP_ALLOCATION = {"bin_size": 100, "counts": []}

def prevent_sleep(enable=True):
    try:
        if os.name == 'nt':
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            kernel32 = ctypes.windll.kernel32
            if enable:
                kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
            else:
                kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass 

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
QGroupBox:disabled {
    border-color: #383552;
    color: #5c5a70;
}
QGroupBox::title:disabled {
    background-color: #2c2a3e;
    color: #5c5a70;
    border: 1px solid #383552;
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
    color: #5c5a70;
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
QTextEdit, QPlainTextEdit {
    background-color: #1a1926;
    border: 1px solid #4a4668;
    color: #e0e0e0;
    font-family: 'Consolas', 'Courier New', monospace;
    border-radius: 4px;
    padding: 5px;
}
QPlainTextEdit:focus { border: 1px solid #ab97e6; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #ab97e6; background-color: #2c2a3e; border-radius: 4px; }
QCheckBox::indicator:checked { background-color: #ab97e6; image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSIjMmMyYTNlIiBkPSJNOSAxNi4xN0w0LjgzIDEybC0xLjQyIDEuNDFMOSAxOUwyMSA3bC0xLjQxLTEuNDF6Ii8+PC9zdmc+"); }
QCheckBox:disabled { color: #5c5a70; }
QComboBox { background-color: #1a1926; border: 1px solid #4a4668; padding: 6px; border-radius: 4px; min-height: 32px; max-height: 32px; }
QComboBox:on { border: 1px solid #ab97e6; }
QComboBox:disabled { background-color: #242233; color: #5c5a70; border-color: #383552; }
QComboBox::drop-down { border-left: 1px solid #4a4668; }
QComboBox QAbstractItemView { background-color: #383552; border: 1px solid #ab97e6; selection-background-color: #ab97e6; selection-color: #1a1926; }

QSpinBox, QDoubleSpinBox {
    background-color: #1a1926;
    border: 1px solid #4a4668;
    padding: 6px;
    padding-right: 30px; 
    color: #e0e0e0;
    border-radius: 4px;
    min-height: 24px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #ab97e6; }
QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: #242233;
    color: #5c5a70;
    border: 1px solid #383552;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 30px; 
    border-left: 1px solid #4a4668;
    background-color: #383552;
    border-top-right-radius: 4px;
    margin-bottom: 0px; 
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 30px; 
    border-left: 1px solid #4a4668;
    background-color: #383552;
    border-bottom-right-radius: 4px;
    margin-top: 0px; 
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #4a4668;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 5px solid #e0e0e0;
    margin: 0px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #e0e0e0;
    margin: 0px;
}
QTabWidget::pane { border: 1px solid #4a4668; border-top: none; }
QTabBar::tab { background: #2c2a3e; border: 1px solid #4a4668; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 10px 20px; color: #e0e0e0; font-weight: bold; min-height: 40px; }
QTabBar::tab:selected { background: #383552; color: #ffffff; border-bottom: 3px solid #ab97e6; }
QTabBar::tab:!selected:hover { background: #4a4668; }
QScrollArea { border: none; }
QHeaderView::section { background-color: #383552; color: #e0e0e0; border: 1px solid #4a4668; padding: 4px; }
QTableWidget { gridline-color: #4a4668; background-color: #1a1926; }
QTableWidget::item:selected { background-color: #ab97e6; color: #1a1926; }
QSlider::groove:horizontal {
    border: 1px solid #4a4668;
    height: 6px;
    background: #1a1926;
    margin: 2px 0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ab97e6;
    border: 1px solid #ab97e6;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
"""

# --- Helper Classes ---

class NoScrollSpinBox(QtWidgets.QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoScrollDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class NoScrollComboBox(QtWidgets.QComboBox):
    def wheelEvent(self, event):
        # Ignore the event so it propagates to the parent (scroll area)
        event.ignore()

class NumericTableWidgetItem(QtWidgets.QTableWidgetItem):
    """Sorts by numeric value instead of string."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)

class DateTableWidgetItem(QtWidgets.QTableWidgetItem):
    """Sorts by timestamp but displays readable date."""
    def __init__(self, display_text, timestamp):
        super().__init__(display_text)
        self.timestamp = timestamp
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

class CustomFolderDialog(QtWidgets.QDialog):
    def __init__(self, start_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset Folder")
        self.resize(1100, 700)
        self.selected_path = None
        self.history = []
        self.history_idx = -1
        self.is_navigating_history = False
        self.icon_provider = QFileIconProvider()
        
        if start_path and os.path.exists(start_path):
            self.current_path = str(Path(start_path).parent) if os.path.isfile(start_path) else str(Path(start_path))
        else:
            self.current_path = os.getcwd()
            
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        nav_bar = QtWidgets.QHBoxLayout()
        nav_bar.setSpacing(5)
        
        def create_nav_btn(std_icon, tooltip):
            btn = QtWidgets.QPushButton()
            btn.setIcon(self.style().standardIcon(std_icon))
            btn.setFixedWidth(35)
            btn.setToolTip(tooltip)
            return btn
        
        self.btn_back = create_nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack, "Back")
        self.btn_back.clicked.connect(self.go_back)
        self.btn_back.setEnabled(False)
        self.btn_fwd = create_nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward, "Forward")
        self.btn_fwd.clicked.connect(self.go_forward)
        self.btn_fwd.setEnabled(False)
        self.btn_up = create_nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp, "Up Directory")
        self.btn_up.clicked.connect(self.go_up)
        self.btn_refresh = create_nav_btn(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload, "Refresh")
        self.btn_refresh.clicked.connect(lambda: self.load_directory(self.current_path))
        
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setText(self.current_path)
        self.path_edit.returnPressed.connect(self.on_path_entered)
        self.path_edit.setStyleSheet("padding-left: 5px; height: 30px;")

        nav_bar.addWidget(self.btn_back)
        nav_bar.addWidget(self.btn_fwd)
        nav_bar.addWidget(self.btn_up)
        nav_bar.addWidget(self.path_edit)
        nav_bar.addWidget(self.btn_refresh)
        layout.addLayout(nav_bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background-color: #4a4668; }")
        layout.addWidget(splitter)
        
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setIconSize(QtCore.QSize(24, 24))
        self.sidebar.setStyleSheet("""
            QListWidget { background-color: #242233; border: 1px solid #4a4668; border-radius: 4px; outline: none; }
            QListWidget::item { padding: 6px; color: #e0e0e0; margin: 2px; }
            QListWidget::item:hover { background-color: #383552; border-radius: 4px; }
            QListWidget::item:selected { background-color: #6a48d7; color: white; border-radius: 4px; }
        """)
        self.sidebar.itemClicked.connect(self.on_sidebar_click)
        self._populate_sidebar()
        splitter.addWidget(self.sidebar)
        
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Images", "Date Modified", "HiddenPath"])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnHidden(3, True)
        
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QtCore.QSize(20, 20))
        self.table.setStyleSheet("""
            QTableWidget { background-color: #1a1926; alternate-background-color: #242233; border: 1px solid #4a4668; border-radius: 4px; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { background-color: #383552; padding: 6px; border: none; border-right: 1px solid #4a4668; font-weight: bold; }
        """)
        
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(self.on_double_click)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(1, 1)

        bottom_bar = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #888;")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.select_btn = QtWidgets.QPushButton("Select Current Folder")
        self.select_btn.setStyleSheet("background-color: #6a48d7; color: white; padding: 6px 15px;")
        self.select_btn.clicked.connect(self.select_current)
        bottom_bar.addWidget(self.status_label)
        bottom_bar.addStretch()
        bottom_bar.addWidget(self.cancel_btn)
        bottom_bar.addWidget(self.select_btn)
        layout.addLayout(bottom_bar)
        
        self.load_directory(self.current_path)

    def _populate_sidebar(self):
        def add_item(text, path, icon_type="folder"):
            if os.path.exists(path):
                info = QFileInfo(path)
                icon = self.icon_provider.icon(info)
                item = QtWidgets.QListWidgetItem(icon, text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                self.sidebar.addItem(item)
        paths = QtCore.QStandardPaths
        add_item("Desktop", paths.writableLocation(paths.StandardLocation.DesktopLocation))
        add_item("Documents", paths.writableLocation(paths.StandardLocation.DocumentsLocation))
        add_item("Pictures", paths.writableLocation(paths.StandardLocation.PicturesLocation))
        add_item("Downloads", paths.writableLocation(paths.StandardLocation.DownloadLocation))
        sep = QtWidgets.QListWidgetItem("───── Drives ─────")
        sep.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        sep.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sidebar.addItem(sep)
        for volume in QtCore.QStorageInfo.mountedVolumes():
            if volume.isValid() and volume.isReady():
                name = volume.name() if volume.name() else volume.rootPath()
                add_item(name, volume.rootPath())

    def on_sidebar_click(self, item):
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if path: self.load_directory(path)

    def on_path_entered(self):
        path = self.path_edit.text()
        if os.path.exists(path) and os.path.isdir(path): self.load_directory(path)
        else: QtWidgets.QMessageBox.warning(self, "Invalid Path", "Path does not exist or is not a directory.")

    def load_directory(self, path):
        if not self.is_navigating_history:
            if self.history_idx == -1 or self.history[self.history_idx] != path:
                self.history = self.history[:self.history_idx+1]
                self.history.append(path)
                self.history_idx += 1
        self.btn_back.setEnabled(self.history_idx > 0)
        self.btn_fwd.setEnabled(self.history_idx < len(self.history) - 1)
        self.is_navigating_history = False 
        self.table.setSortingEnabled(False)
        self.current_path = path
        self.path_edit.setText(path)
        self.table.setRowCount(0)
        self.status_label.setText("Scanning...")
        QtWidgets.QApplication.processEvents() 
        img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        folder_count = 0
        try:
            entries = []
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir(): entries.append(entry)
            self.table.setRowCount(len(entries))
            folder_count = len(entries)
            for row, entry in enumerate(entries):
                info = QFileInfo(entry.path)
                icon = self.icon_provider.icon(info)
                name_item = QtWidgets.QTableWidgetItem(icon, entry.name)
                self.table.setItem(row, 0, name_item)
                try:
                    count = 0
                    with os.scandir(entry.path) as sub_it:
                        for f in sub_it:
                            if f.is_file() and os.path.splitext(f.name)[1].lower() in img_exts: count += 1
                    count_item = NumericTableWidgetItem(str(count))
                    count_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    if count > 0:
                        count_item.setForeground(QtGui.QColor("#ab97e6"))
                        count_item.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold))
                    else: count_item.setForeground(QtGui.QColor("#5c5a70"))
                    self.table.setItem(row, 1, count_item)
                except PermissionError: self.table.setItem(row, 1, NumericTableWidgetItem("-1"))
                try:
                    ts = entry.stat().st_mtime
                    dt_obj = datetime.fromtimestamp(ts)
                    display_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                    date_item = DateTableWidgetItem(display_str, ts)
                    date_item.setForeground(QtGui.QColor("#aaaaaa"))
                    self.table.setItem(row, 2, date_item)
                except Exception: self.table.setItem(row, 2, DateTableWidgetItem("N/A", 0))
                self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(entry.path))
            self.status_label.setText(f"{folder_count} folders found.")
        except PermissionError:
            QtWidgets.QMessageBox.warning(self, "Access Denied", f"Cannot access {path}")
            self.go_back() 
        except Exception as e: QtWidgets.QMessageBox.warning(self, "Error", str(e))
        self.table.setSortingEnabled(True)

    def go_up(self):
        parent = os.path.dirname(self.current_path)
        if parent and parent != self.current_path: self.load_directory(parent)
            
    def go_back(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.is_navigating_history = True
            self.load_directory(self.history[self.history_idx])

    def go_forward(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.is_navigating_history = True
            self.load_directory(self.history[self.history_idx])

    def on_double_click(self, row, col):
        path = self.table.item(row, 3).text()
        self.load_directory(path)

    def select_current(self):
        rows = self.table.selectionModel().selectedRows()
        if rows: self.selected_path = self.table.item(rows[0].row(), 3).text()
        else: self.selected_path = self.current_path
        self.accept()

# --- Graph Classes ---
class GraphPanel(QtWidgets.QWidget):
    def __init__(self, title, y_label, parent=None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.smoothing_window_size = 15
        self.lines = []
        self.padding = {'top': 35, 'bottom': 40, 'left': 70, 'right': 20}
        self.bg_color = QtGui.QColor("#1a1926")
        self.graph_bg_color = QtGui.QColor("#2c2a3e")
        self.grid_color = QtGui.QColor("#4a4668")
        self.text_color = QtGui.QColor("#e0e0e0")
        self.title_color = QtGui.QColor("#ab97e6")
        self.x_min, self.x_max = 0, 100
        self.y_min, self.y_max = 0, 1
        self.smoothing_level = 0.0
        self.fill_enabled = True
        self.zoom_level = 1.0
        self.pan_offset = 0

    def add_line(self, color, label, max_points=2000, linewidth=2):
        self.lines.append({
            'data': deque(maxlen=max_points),
            'smoothed_data': deque(maxlen=max_points),
            'smoothing_window': deque(maxlen=self.smoothing_window_size),
            'color': QtGui.QColor(color),
            'label': label,
            'linewidth': linewidth
        })
        return len(self.lines) - 1

    def append_data(self, line_index, x, y):
        if 0 <= line_index < len(self.lines):
            line = self.lines[line_index]
            line['data'].append((x, y))
            line['smoothing_window'].append(y)
            smoothed_y = sum(line['smoothing_window']) / len(line['smoothing_window'])
            line['smoothed_data'].append((x, smoothed_y))
            self._update_bounds()

    def clear_all_data(self):
        for line in self.lines:
            line['data'].clear()
            line['smoothed_data'].clear()
            line['smoothing_window'].clear()
        self.pan_offset = 0
        self._update_bounds()
        self.update()

    def _get_visible_data_slice(self, data_deque):
        if not data_deque: return []
        total_points = len(data_deque)
        visible_points_count = max(2, int(total_points / self.zoom_level))
        start_index = self.pan_offset
        end_index = min(start_index + visible_points_count, total_points)
        return list(data_deque)[start_index:end_index]

    def _update_bounds(self):
        all_visible_y = []
        visible_data = []
        for line in self.lines:
            visible_raw = self._get_visible_data_slice(line['data'])
            if visible_raw:
                visible_data = visible_raw
                for _, y in visible_raw: all_visible_y.append(y)
        if visible_data:
            self.x_min = visible_data[0][0]
            self.x_max = visible_data[-1][0]
            if all_visible_y:
                self.y_min = min(all_visible_y)
                self.y_max = max(all_visible_y)
                y_range = self.y_max - self.y_min
                if y_range == 0: y_range = 1
                self.y_min -= y_range * 0.05
                self.y_max += y_range * 0.05
            else: self.y_min, self.y_max = 0, 1
        else:
            self.x_min, self.x_max = 0, 100
            self.y_min, self.y_max = 0, 1

    def _to_screen_coords(self, x, y):
        graph_width = self.width() - self.padding['left'] - self.padding['right']
        graph_height = self.height() - self.padding['top'] - self.padding['bottom']
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        if x_range == 0: x_range = 1
        if y_range == 0: y_range = 1
        screen_x = self.padding['left'] + ((x - self.x_min) / x_range) * graph_width
        screen_y = self.padding['top'] + graph_height - ((y - self.y_min) / y_range) * graph_height
        return QtCore.QPointF(screen_x, screen_y)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        graph_rect = QtCore.QRect(
            self.padding['left'], self.padding['top'],
            self.width() - self.padding['left'] - self.padding['right'],
            self.height() - self.padding['top'] - self.padding['bottom']
        )
        painter.fillRect(graph_rect, self.graph_bg_color)
        self._draw_grid_and_axes(painter, graph_rect)
        self._draw_title(painter)
        self._draw_legend(painter)
        self._draw_data_lines(painter, graph_rect)

    def _draw_data_lines(self, painter, rect):
        for line in self.lines:
            visible_raw = self._get_visible_data_slice(line['data'])
            visible_smoothed = self._get_visible_data_slice(line['smoothed_data'])
            if len(visible_raw) < 2 or len(visible_raw) != len(visible_smoothed): continue
            display_points = []
            for i in range(len(visible_raw)):
                raw_x, raw_y = visible_raw[i]
                _, smoothed_y = visible_smoothed[i]
                display_y = raw_y * (1 - self.smoothing_level) + smoothed_y * self.smoothing_level
                display_points.append(self._to_screen_coords(raw_x, display_y))
            if self.fill_enabled:
                fill_poly = QtGui.QPolygonF(display_points)
                fill_poly.append(QtCore.QPointF(display_points[-1].x(), rect.bottom()))
                fill_poly.append(QtCore.QPointF(display_points[0].x(), rect.bottom()))
                fill_color = QtGui.QColor(line['color'])
                fill_color.setAlpha(40)
                painter.setBrush(fill_color)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(fill_poly)
            painter.setPen(QtGui.QPen(line['color'], line['linewidth']))
            painter.drawPolyline(QtGui.QPolygonF(display_points))

    def _draw_grid_and_axes(self, painter, rect):
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        num_h_lines = 5
        for i in range(num_h_lines):
            y = rect.top() + (i / (num_h_lines - 1)) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            y_val = self.y_max - (i / (num_h_lines - 1)) * (self.y_max - self.y_min)
            label = self._format_number(y_val)
            painter.setPen(self.text_color)
            painter.drawText(
                QtCore.QRect(5, int(y - 10), self.padding['left'] - 10, 20),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                label
            )
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        num_v_lines = 6
        for i in range(num_v_lines):
            x = rect.left() + (i / (num_v_lines - 1)) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            x_val = self.x_min + (i / (num_v_lines - 1)) * (self.x_max - self.x_min)
            label = str(int(x_val))
            painter.setPen(self.text_color)
            painter.drawText(
                QtCore.QRect(int(x - 30), rect.bottom() + 5, 60, 20),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                label
            )
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        painter.setPen(self.text_color)
        font = painter.font(); font.setPointSize(9); painter.setFont(font)
        painter.save()
        painter.translate(15, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-50, -10, 100, 20), QtCore.Qt.AlignmentFlag.AlignCenter, self.y_label)
        painter.restore()
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20), QtCore.Qt.AlignmentFlag.AlignCenter, "Step")
    
    def _draw_title(self, painter):
        painter.setPen(self.title_color)
        font = painter.font(); font.setPointSize(11); font.setBold(True); painter.setFont(font)
        painter.drawText(QtCore.QRect(0, 5, self.width(), 25), QtCore.Qt.AlignmentFlag.AlignCenter, self.title)
    
    def _draw_legend(self, painter):
        if not self.lines: return
        legend_x = self.width() - self.padding['right'] - 120
        legend_y = self.padding['top'] + 10
        font = painter.font(); font.setPointSize(8); painter.setFont(font)
        for line in self.lines:
            if not line['data']: continue
            painter.setPen(QtGui.QPen(line['color'], line['linewidth']))
            painter.drawLine(legend_x, legend_y + 5, legend_x + 20, legend_y + 5)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(legend_x + 25, legend_y, 80, 15), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, line['label'])
            legend_y += 20
    
    def _format_number(self, value):
        if abs(value) < 0.01 or abs(value) > 10000: return f"{value:.1e}"
        elif abs(value) < 1: return f"{value:.4f}"
        else: return f"{value:.2f}"
            
    def set_smoothing(self, value):
        self.smoothing_level = value / 100.0
        self.update()

    def set_fill(self, enabled):
        self.fill_enabled = enabled
        self.update()

    def set_zoom(self, value):
        self.zoom_level = value / 100.0
        self._update_bounds()
        self.update()

    def set_pan(self, value):
        self.pan_offset = value
        self._update_bounds()
        self.update()

    def get_pan_range(self):
        if not self.lines or not self.lines[0]['data']: return 0, 0, 1
        total_points = len(self.lines[0]['data'])
        visible_points_count = max(2, int(total_points / self.zoom_level))
        max_pan = total_points - visible_points_count
        return 0, max(0, max_pan), 1


class LiveMetricsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_points = 20000
        self.pending_update = False
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._perform_update)
        self.update_interval_ms = 500
        self.pending_data = []
        self.graphs = {}
        self._setup_ui()
    
    def _create_graph_with_controls(self, name, title, y_label):
        container = QtWidgets.QGroupBox(title)
        container.setStyleSheet("QGroupBox { margin-top: 10px; padding: 5px; } QGroupBox::title { subcontrol-position: top center; }")
        layout = QtWidgets.QVBoxLayout(container)
        graph = GraphPanel(title, y_label)
        self.graphs[name] = {'widget': graph, 'lines': {}}
        layout.addWidget(graph, 1)
        controls_layout = QtWidgets.QHBoxLayout()
        fill_check = QtWidgets.QCheckBox("Fill")
        fill_check.setChecked(True)
        fill_check.stateChanged.connect(lambda state, g=graph: g.set_fill(state == QtCore.Qt.CheckState.Checked.value))
        controls_layout.addWidget(fill_check)
        controls_layout.addWidget(QtWidgets.QLabel("Raw ↔ Smooth"))
        smoothing_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        smoothing_slider.setRange(0, 100)
        smoothing_slider.setValue(0)
        smoothing_slider.valueChanged.connect(graph.set_smoothing)
        controls_layout.addWidget(smoothing_slider, 1)
        controls_layout.addWidget(QtWidgets.QLabel("Zoom"))
        zoom_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        zoom_slider.setRange(100, 1000)
        zoom_slider.setValue(100)
        controls_layout.addWidget(zoom_slider, 1)
        layout.addLayout(controls_layout)
        pan_scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        pan_scrollbar.setVisible(False)
        pan_scrollbar.valueChanged.connect(graph.set_pan)
        layout.addWidget(pan_scrollbar)
        zoom_slider.valueChanged.connect(lambda val, g=graph, sb=pan_scrollbar: self._update_pan_scrollbar(g, sb, val))
        self.graphs[name]['scrollbar'] = pan_scrollbar
        self.graphs[name]['zoom_slider'] = zoom_slider
        return container

    def _update_pan_scrollbar(self, graph, scrollbar, zoom_value):
        graph.set_zoom(zoom_value)
        min_pan, max_pan, step = graph.get_pan_range()
        scrollbar.blockSignals(True)
        scrollbar.setRange(min_pan, max_pan)
        scrollbar.setPageStep(step)
        if scrollbar.value() > max_pan: scrollbar.setValue(max_pan)
        scrollbar.blockSignals(False)
        scrollbar.setVisible(zoom_value > 100)
        graph.set_pan(scrollbar.value())

    def _add_line_to_graph(self, graph_name, line_name, color, linewidth=2):
        graph_widget = self.graphs[graph_name]['widget']
        line_idx = graph_widget.add_line(color, line_name, self.max_points, linewidth)
        self.graphs[graph_name]['lines'][line_name] = line_idx

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        control_layout = QtWidgets.QHBoxLayout()
        self.clear_button = QtWidgets.QPushButton("Clear Data")
        self.clear_button.clicked.connect(self.clear_data)
        control_layout.addWidget(self.clear_button)
        self.pause_button = QtWidgets.QPushButton("Pause Updates")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._on_pause_toggled)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(QtWidgets.QLabel("Update Speed:"))
        self.speed_combo = NoScrollComboBox()
        self.speed_combo.addItems(["Fast (100ms)", "Normal (500ms)", "Slow (1000ms)", "Very Slow (2000ms)"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        control_layout.addWidget(self.speed_combo)
        control_layout.addStretch()
        self.stats_label = QtWidgets.QLabel("No data yet")
        self.stats_label.setStyleSheet("color: #ab97e6; font-weight: bold;")
        control_layout.addWidget(self.stats_label)
        main_layout.addLayout(control_layout)
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.addWidget(self._create_graph_with_controls("step_loss", "Per-Step Loss", "Loss"), 0, 0)
        grid_layout.addWidget(self._create_graph_with_controls("timestep", "Timestep", "Value"), 0, 1)
        grid_layout.addWidget(self._create_graph_with_controls("optim_loss", "Optimizer Loss (Avg)", "Loss"), 1, 0)
        grid_layout.addWidget(self._create_graph_with_controls("lr", "Learning Rate", "LR"), 1, 1)
        grid_layout.addWidget(self._create_graph_with_controls("grad_norm", "Gradient Norms", "Norm"), 2, 0, 1, 2)
        self._add_line_to_graph("step_loss", "Step Loss", "#4CAF50")
        self._add_line_to_graph("timestep", "Timestep", "#2196F3")
        self._add_line_to_graph("optim_loss", "Avg Loss", "#ab97e6")
        self._add_line_to_graph("lr", "LR", "#6a48d7")
        self._add_line_to_graph("grad_norm", "Raw", "#e53935", linewidth=3)
        self._add_line_to_graph("grad_norm", "Clipped", "#ffdd57", linewidth=2)
        main_layout.addLayout(grid_layout)
        self.latest_global_step, self.latest_optim_step, self.latest_lr = 0, 0, 0.0
        self.latest_step_loss, self.latest_optim_loss, self.latest_grad = 0.0, 0.0, 0.0
        self.latest_timestep = 0

    def _on_pause_toggled(self, checked):
        if checked: self.update_timer.stop()
        elif self.pending_update: self.update_timer.start(self.update_interval_ms)
    
    def _on_speed_changed(self, index):
        speeds = [100, 500, 1000, 2000]
        self.update_interval_ms = speeds[index]
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.update_timer.start(self.update_interval_ms)

    def parse_and_update(self, text):
        if self.pause_button.isChecked(): return
        data_added = False
        progress_match = re.search(r'Training\s*\|.*\|\s*(\d+)/(\d+)\s*\[.*?\]\s*\[Loss:\s*([\d.e+-]+),\s*Timestep:\s*(\d+)\]', text)
        if progress_match:
            step, loss, timestep = int(progress_match.group(1)) - 1, float(progress_match.group(3)), int(progress_match.group(4))
            self.pending_data.append(('progress_step', step, loss, timestep))
            self.latest_global_step, self.latest_step_loss, self.latest_timestep = step, loss, timestep
            data_added = True
        step_match = re.search(r'--- Optimizer Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.e+-]+)\s*\|\s*LR:\s*([\d.e+-]+)\s*---', text)
        if step_match:
            step, avg_loss, lr = int(step_match.group(1)), float(step_match.group(2)), float(step_match.group(3))
            self.pending_data.append(('optim_step', step, avg_loss, lr))
            self.latest_optim_step, self.latest_lr, self.latest_optim_loss = step, lr, avg_loss
            data_added = True
        grad_match = re.search(r'Grad Norm \(Raw/Clipped\):\s*([\d.]+)\s*/\s*([\d.]+)', text)
        if grad_match:
            raw_norm, clipped_norm = float(grad_match.group(1)), float(grad_match.group(2))
            self.pending_data.append(('grad', raw_norm, clipped_norm))
            self.latest_grad = raw_norm
            data_added = True
        if data_added:
            self.pending_update = True
            if not self.update_timer.isActive() and not self.pause_button.isChecked():
                self.update_timer.start(self.update_interval_ms)
    
    def _perform_update(self):
        if not self.pending_update or not self.pending_data:
            self.update_timer.stop()
            return
        last_optim_step = self.latest_optim_step
        for data in self.pending_data:
            if data[0] == 'progress_step':
                _, step, loss, timestep = data
                self.graphs['step_loss']['widget'].append_data(self.graphs['step_loss']['lines']['Step Loss'], step, loss)
                self.graphs['timestep']['widget'].append_data(self.graphs['timestep']['lines']['Timestep'], step, timestep)
            elif data[0] == 'optim_step':
                _, step, avg_loss, lr = data
                last_optim_step = step
                self.graphs['optim_loss']['widget'].append_data(self.graphs['optim_loss']['lines']['Avg Loss'], step, avg_loss)
                self.graphs['lr']['widget'].append_data(self.graphs['lr']['lines']['LR'], step, lr)
            elif data[0] == 'grad' and last_optim_step is not None:
                _, raw_norm, clipped_norm = data
                self.graphs['grad_norm']['widget'].append_data(self.graphs['grad_norm']['lines']['Raw'], last_optim_step, raw_norm)
                self.graphs['grad_norm']['widget'].append_data(self.graphs['grad_norm']['lines']['Clipped'], last_optim_step, clipped_norm)
        self.pending_data.clear()
        stats_text = (f"Step: {self.latest_global_step} | Step Loss: {self.latest_step_loss:.4f} | Timestep: {self.latest_timestep} | "
                      f"Avg Loss: {self.latest_optim_loss:.4f} | LR: {self.latest_lr:.2e} | Grad: {self.latest_grad:.4f}")
        self.stats_label.setText(stats_text)
        for name, graph_data in self.graphs.items():
            self._update_pan_scrollbar(graph_data['widget'], graph_data['scrollbar'], graph_data['zoom_slider'].value())
        self.pending_update = False
    
    def clear_data(self):
        self.update_timer.stop()
        self.pending_update = False
        self.pending_data.clear()
        for name, graph_data in self.graphs.items():
            graph_data['widget'].clear_all_data()
            self._update_pan_scrollbar(graph_data['widget'], graph_data['scrollbar'], graph_data['zoom_slider'].value())
        self.latest_global_step, self.latest_optim_step, self.latest_lr = 0, 0, 0.0
        self.latest_step_loss, self.latest_optim_loss, self.latest_grad = 0.0, 0.0, 0.0
        self.latest_timestep = 0
        self.stats_label.setText("No data yet")
    
    def showEvent(self, event):
        super().showEvent(event)
        if self.pending_update and not self.pause_button.isChecked():
            self.update_timer.start(self.update_interval_ms)
    
    def hideEvent(self, event):
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

        self.padding = {'top': 40, 'bottom': 60, 'left': 60, 'right': 20}
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
        
        points_changed = False
        for p in self._points:
            if p[1] > self.max_lr_bound:
                p[1] = self.max_lr_bound
                points_changed = True
            elif p[1] < self.min_lr_bound:
                p[1] = self.min_lr_bound
                points_changed = True
        
        if points_changed:
            self.pointsChanged.emit(self._points)
        
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
        
        # Draw Grid Labels
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
            
            # Y-Axis Labels (Aligned Right within the padding)
            painter.drawText(QtCore.QRect(0, int(y - 10), self.padding['left'] - 5, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, label)
            
            step_val = int(self.max_steps * (i / 4.0))
            label_x = str(step_val)
            x = rect.left() + (i / 4.0) * rect.width()
            
            # X-Axis Labels (Straight line, no stagger)
            painter.drawText(QtCore.QRect(int(x - 50), rect.bottom() + 5, 100, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, label_x)
                             
        small_font = self.font()
        small_font.setPointSize(8)
        painter.setFont(small_font)
        
        # Epoch Labels (Dotted Lines) - These are distinct from grid labels
        for norm_x, step_count in self.epoch_data:
            x = rect.left() + norm_x * rect.width()
            label_rect = QtCore.QRect(int(x - 40), rect.bottom() + 25, 80, 15)
            painter.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(step_count))
            
        painter.setFont(original_font)
        font.setBold(True); painter.setFont(font)
        painter.drawText(self.rect().adjusted(0, 5, 0, 0), QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop, "Learning Rate Schedule")
    
    def draw_curve(self, painter):
        if not self._visual_points: return
        poly_points = [self._to_pixel_coords(p[0], p[1]) for p in self._visual_points]
        poly = QtGui.QPolygonF(poly_points)
        painter.setPen(QtGui.QPen(self.line_color, 2))
        painter.drawPolyline(poly)
        fill_poly = QtGui.QPolygonF(poly)
        last_x_norm = self._visual_points[-1][0]
        fill_poly.append(self._to_pixel_coords(last_x_norm, self.min_lr_bound))
        first_x_norm = self._visual_points[0][0]
        fill_poly.append(self._to_pixel_coords(first_x_norm, self.min_lr_bound))
        fill_color = QtGui.QColor(self.point_fill_color)
        fill_color.setAlpha(50)  
        painter.setBrush(fill_color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawPolygon(fill_poly)
    
    def draw_points_and_labels(self, painter):
        for i, p in enumerate(self._visual_points):
            pixel_pos = self._to_pixel_coords(p[0], p[1])
            is_selected = (i == self._selected_point_index)
            painter.setBrush(self.selected_point_color if is_selected else self.point_fill_color)
            painter.setPen(self.point_color)
            painter.drawEllipse(pixel_pos, self.point_radius, self.point_radius)
            if is_selected or i == self._dragging_point_index:
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
    
    def set_generated_preset(self, mode, num_restarts, initial_warmup_pct, restart_rampup_pct):
        num_restarts = max(1, int(num_restarts))
        initial_warmup_pct = max(0.0, min(1.0, float(initial_warmup_pct)))
        restart_rampup_pct = max(0.0, min(1.0, float(restart_rampup_pct)))
        epsilon = 1e-5 
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        points = []
        segment_length = 1.0 / num_restarts
        points_per_segment = 20 
        for i in range(num_restarts):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            current_warmup_pct = initial_warmup_pct if i == 0 else restart_rampup_pct
            warmup_len = segment_length * current_warmup_pct
            if warmup_len < epsilon: warmup_len = epsilon
            if warmup_len > segment_length: warmup_len = segment_length
            decay_len = segment_length - warmup_len
            points.append([segment_start, min_lr])
            peak_x = min(segment_start + warmup_len, segment_end)
            points.append([peak_x, max_lr])
            if decay_len > epsilon:
                if mode == "Cosine":
                    for j in range(1, points_per_segment + 1):
                        progress = j / points_per_segment
                        x = peak_x + (progress * decay_len)
                        factor = 0.5 * (1 + math.cos(math.pi * progress))
                        y = min_lr + (max_lr - min_lr) * factor
                        points.append([x, y])
                else: points.append([segment_end, min_lr])
            else:
                if i < num_restarts - 1: points.append([segment_end, max_lr])
                else: points.append([segment_end, max_lr])
        final_points = []
        for p in points:
            x = max(0.0, min(1.0, p[0]))
            y = max(min_lr, min(max_lr, p[1]))
            final_points.append([x, y])
        if not final_points or final_points[0][0] != 0.0: final_points.insert(0, [0.0, min_lr])
        if final_points[-1][0] < 1.0: final_points.append([1.0, final_points[-1][1]])
        unique_points = []
        last_x = -1.0
        for p in final_points:
            if p[0] > last_x + 1e-9:
                unique_points.append(p)
                last_x = p[0]
        self.apply_preset(unique_points)


class TimestepHistogramWidget(QtWidgets.QWidget):
    allocationChanged = QtCore.pyqtSignal(dict)  # Emits {"bin_size": int, "counts": [int]}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        # Left padding 35, Right padding 20
        self.padding = {'top': 40, 'bottom': 50, 'left': 35, 'right': 20}
        
        # Internal State
        self.bin_size = 100
        self.max_tickets = 1000  # Default budget
        self.counts = []  # List of integers, one per bin
        self._dragging_bin_index = -1

        # Colors
        self.bg_color = QtGui.QColor("#1a1926")
        self.grid_color = QtGui.QColor("#4a4668")
        
        self.bar_color_even = QtGui.QColor("#6a48d7")  # Main Theme Purple
        self.bar_color_odd = QtGui.QColor("#5839b0")   # Slightly Darker Purple for contrast
        
        self.bar_hover_color = QtGui.QColor("#ab97e6") # Light Accent Purple (Title color)
        self.text_color = QtGui.QColor("#e0e0e0")
        self.alert_color = QtGui.QColor("#e53935")
        
        # Disabled Colors
        self.disabled_bar_color = QtGui.QColor("#383552")
        self.disabled_text_color = QtGui.QColor("#5c5a70")

        self.setMouseTracking(True)
        self.setToolTip("Click and drag bars to allocate timestep tickets.\nTotal tickets must equal Max Training Steps.")
        
        self._init_bins()

    def set_total_steps(self, steps):
        try:
            steps = int(steps)
        except ValueError:
            return

        if steps <= 0: steps = 1
        
        # Save the new max
        self.max_tickets = steps
        
        # Get current state
        current_sum = sum(self.counts)
        num_bins = len(self.counts)

        # If the graph is empty or has no tickets yet, just initialize a flat distribution
        if num_bins == 0 or current_sum == 0:
            self._init_bins()
            return

        # --- SMART SCALING ALGORITHM (Largest Remainder Method) ---
        raw_new_counts = [(c / current_sum) * steps for c in self.counts]
        new_counts = [int(x) for x in raw_new_counts]
        current_new_sum = sum(new_counts)
        deficit = steps - current_new_sum

        if deficit > 0:
            fractional_parts = []
            for i, raw_val in enumerate(raw_new_counts):
                frac = raw_val - new_counts[i]
                fractional_parts.append((frac, i))
            fractional_parts.sort(key=lambda x: x[0], reverse=True)
            for i in range(deficit):
                idx_to_increment = fractional_parts[i][1]
                new_counts[idx_to_increment] += 1

        self.counts = new_counts
        self.update()
        self._emit_change()

    def set_bin_size(self, size):
        if size <= 0: return
        self.bin_size = size
        self._init_bins()
        self.update()
        self._emit_change()
        
    def set_allocation(self, allocation):
        if not allocation or "bin_size" not in allocation or "counts" not in allocation:
            self._init_bins()
            return
        self.bin_size = allocation["bin_size"]
        loaded_counts = allocation["counts"]
        expected_bins = math.ceil(1000 / self.bin_size)
        if len(loaded_counts) != expected_bins:
            self._init_bins()
        else:
            self.counts = loaded_counts
            current_sum = sum(self.counts)
            if current_sum > 0:
                self.max_tickets = current_sum
        self.update()

    def get_allocation(self):
        return {"bin_size": self.bin_size, "counts": self.counts}

    def generate_from_weights(self, weights):
        """Applies a specific weight distribution to the current max_tickets"""
        num_bins = len(self.counts)
        if num_bins == 0 or not weights: return
        
        total_weight = sum(weights)
        if total_weight == 0: total_weight = 1
        
        raw_counts = [(w / total_weight) * self.max_tickets for w in weights]
        new_counts = [int(c) for c in raw_counts]
        
        diff = self.max_tickets - sum(new_counts)
        # Distribute remainder
        remainders = sorted([(raw_counts[i] - new_counts[i], i) for i in range(num_bins)], key=lambda x: x[0], reverse=True)
        for i in range(diff):
            new_counts[remainders[i][1]] += 1
            
        self.counts = new_counts
        self.update()
        self._emit_change()

    def _init_bins(self):
        # Re-initialize bins (Uniform distribution)
        if self.bin_size <= 0: self.bin_size = 100
        num_bins = math.ceil(1000 / self.bin_size)
        if num_bins <= 0: num_bins = 1
        
        base_count = self.max_tickets // num_bins
        remainder = self.max_tickets % num_bins
        
        self.counts = [base_count + 1 if i < remainder else base_count for i in range(num_bins)]
        self._emit_change()

    def _emit_change(self):
        self.allocationChanged.emit(self.get_allocation())

    def _get_bin_rect(self, index, graph_rect):
        num_bins = len(self.counts)
        if num_bins == 0: return QtCore.QRectF()
        bin_width = graph_rect.width() / num_bins
        x = graph_rect.left() + (index * bin_width)
        max_val = max(self.counts) if self.counts else 1
        y_scale_max = max(max_val * 1.2, self.max_tickets * 0.2) 
        bar_height_ratio = self.counts[index] / y_scale_max
        bar_pixel_height = bar_height_ratio * graph_rect.height()
        y = graph_rect.bottom() - bar_pixel_height
        return QtCore.QRectF(x, y, bin_width, bar_pixel_height)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        is_enabled = self.isEnabled()
        self._draw_grid_and_axes(painter, rect, is_enabled)
        self._draw_bars(painter, rect, is_enabled)
        self._draw_labels(painter, rect, is_enabled)
        self._draw_header(painter, rect, is_enabled)

    def _draw_grid_and_axes(self, painter, rect, is_enabled):
        grid_color = self.grid_color if is_enabled else self.disabled_text_color
        text_color = self.text_color if is_enabled else self.disabled_text_color
        painter.setPen(QtGui.QPen(grid_color, 1, QtCore.Qt.PenStyle.DashLine))
        
        for i in range(5):
            y = rect.bottom() - (i / 4.0) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        
        num_labels = 6 
        for i in range(num_labels):
            x = rect.left() + (i / (num_labels - 1)) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            
            value = int((i / (num_labels - 1)) * 1000)
            painter.setPen(text_color)
            label_rect = QtCore.QRectF(x - 30, rect.bottom() + 5, 60, 20)
            painter.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(value))
            painter.setPen(QtGui.QPen(grid_color, 1, QtCore.Qt.PenStyle.DashLine))

    def _draw_bars(self, painter, rect, is_enabled):
        original_font = painter.font()
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)

        for i in range(len(self.counts)):
            bar_rect = self._get_bin_rect(i, rect)
            is_odd = (i % 2 != 0)
            
            if is_enabled:
                painter.setBrush(self.bar_hover_color if i == self._dragging_bin_index else (self.bar_color_odd if is_odd else self.bar_color_even))
            else:
                painter.setBrush(self.disabled_bar_color)
                
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRect(bar_rect)
            
            if is_enabled:
                painter.setPen(self.text_color)
                offset_y = 15 if is_odd else 0
                text_y = bar_rect.top() - 20 - offset_y
                if text_y < rect.top():
                    if bar_rect.height() > 40:
                        text_y = bar_rect.top() + 5 + offset_y
                        painter.setPen(QtCore.Qt.GlobalColor.black)
                    else:
                        text_y = rect.top() + 5
                        painter.setPen(self.text_color)
                text_rect = QtCore.QRectF(bar_rect.left(), text_y, bar_rect.width(), 20)
                painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(self.counts[i]))
        
        painter.setFont(original_font)

    def _draw_labels(self, painter, rect, is_enabled):
        painter.setPen(self.text_color if is_enabled else self.disabled_text_color)
        painter.save()
        painter.translate(12, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-100, -10, 200, 20), QtCore.Qt.AlignmentFlag.AlignCenter, "Tickets Count")
        painter.restore()
        
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20), 
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Timestep Range")

    def _draw_header(self, painter, rect, is_enabled):
        used = sum(self.counts)
        total = self.max_tickets
        font = painter.font()
        font.setBold(True); font.setPointSize(11)
        painter.setFont(font)
        status_text = f"Tickets Used: {used} / {total}"
        if used == total: painter.setPen(QtGui.QColor("#66bb6a"))
        elif used > total: painter.setPen(self.alert_color)
        else: painter.setPen(self.text_color)
        painter.drawText(self.rect().adjusted(0, 5, 0, 0), QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop, status_text)

    def mousePressEvent(self, event):
        if not self.isEnabled() or event.button() != QtCore.Qt.MouseButton.LeftButton: return
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        num_bins = len(self.counts)
        if num_bins == 0: return
        rel_x = event.pos().x() - rect.left()
        if 0 <= rel_x <= rect.width():
            self._dragging_bin_index = int(rel_x / (rect.width() / num_bins))
            self.update()

    def mouseMoveEvent(self, event):
        if not self.isEnabled(): return
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        rel_x = event.pos().x() - rect.left()
        num_bins = len(self.counts)
        if num_bins == 0: return
        
        if self._dragging_bin_index == -1:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if 0 <= rel_x <= rect.width() and 0 <= event.pos().y() <= self.height() else QtCore.Qt.CursorShape.ArrowCursor)
            return

        idx = self._dragging_bin_index
        if not (0 <= idx < num_bins): return
        
        max_val = max(self.counts) if self.counts else 1
        y_scale_max = max(max_val * 1.2, self.max_tickets * 0.2)
        target_val = int(((rect.bottom() - event.pos().y()) / rect.height()) * y_scale_max)
        
        current_val = self.counts[idx]
        available = self.max_tickets - (sum(self.counts) - current_val)
        new_val = max(0, min(target_val, available))
        
        if new_val != current_val:
            self.counts[idx] = new_val
            self._emit_change()
            self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_bin_index = -1
        self.update()

class ProcessRunner(QThread):
    logSignal = pyqtSignal(str)
    paramInfoSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(str, bool)
    finishedSignal = pyqtSignal(int)
    errorSignal = pyqtSignal(str)
    metricsSignal = pyqtSignal(str)
    cacheCreatedSignal = pyqtSignal()
    
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
                if not line or "NOTE: Redirects are currently not supported" in line: continue
                if line.startswith("GUI_PARAM_INFO::"):
                    self.paramInfoSignal.emit(line.replace('GUI_PARAM_INFO::', '').strip())
                else:
                    is_progress = '\r' in line or bool(re.match(r'^\s*\d+%\|\S*\|', line))
                    if any(keyword in line.lower() for keyword in ["memory inaccessible", "cuda out of memory", "access violation", "nan/inf"]):
                        self.logSignal.emit(f"*** ERROR DETECTED: {line} ***")
                    else:
                        self.progressSignal.emit(line.split('\r')[-1], is_progress)
                    self.metricsSignal.emit(line)
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
        "UNCONDITIONAL_DROPOUT": {"label": "Unconditional Dropout", "tooltip": "Randomly replace captions with empty strings to train unconditional generation (allows negative prompts).", "widget": "QCheckBox"},
        "UNCONDITIONAL_DROPOUT_CHANCE": {"label": "Dropout Chance", "tooltip": "Probability (0.0-1.0) of dropping the caption. 0.1 is standard.", "widget": "QDoubleSpinBox", "range": (0.0, 1.0), "step": 0.05, "decimals": 2},
        "TARGET_PIXEL_AREA": {"label": "Target Pixel Area", "tooltip": "e.g., 1024*1024=1048576. Buckets are resolutions near this total area.", "widget": "QLineEdit"},
        "SHOULD_UPSCALE": {"label": "Upscale Images", "tooltip": "If enabled, upscale small images closer to bucket limit while maintaining aspect ratio.", "widget": "QCheckBox"},
        "MAX_AREA_TOLERANCE": {"label": "Max Area Tolerance:", "tooltip": "When upscaling, allow up to this multiplier over target area (e.g., 1.1 = 10% over).", "widget": "QLineEdit"},
        "PREDICTION_TYPE": {"label": "Prediction Type:", "tooltip": "v_prediction or epsilon. Must match the base model.", "widget": "QComboBox", "options": ["v_prediction", "epsilon"]},
        "BETA_SCHEDULE": {"label": "Beta Schedule:", "tooltip": "Noise schedule for the diffuser.", "widget": "QComboBox", "options": ["scaled_linear", "linear", "squared", "squaredcos_cap_v2"]},
        "MAX_TRAIN_STEPS": {"label": "Max Training Steps:", "tooltip": "Total number of training steps.", "widget": "QLineEdit"},
        "BATCH_SIZE": {"label": "Batch Size:", "tooltip": "Number of samples per batch.", "widget": "QSpinBox", "range": (1, 32)},
        "SAVE_EVERY_N_STEPS": {"label": "Save Every N (Optimizer Steps)", "tooltip": "Determines when the training run will save a checkpoint based on optimizer steps.", "widget": "QLineEdit"},
        "GRADIENT_ACCUMULATION_STEPS": {"label": "Gradient Accumulation:", "tooltip": "Simulates a larger batch size.", "widget": "QLineEdit"},
        "MIXED_PRECISION": {"label": "Mixed Precision:", "tooltip": "bfloat16 for modern GPUs, float16 for older.", "widget": "QComboBox", "options": ["bfloat16", "float16"]},
        "CLIP_GRAD_NORM": {"label": "Gradient Clipping:", "tooltip": "Maximum gradient norm. Set to 0 to disable.", "widget": "QLineEdit"},
        "SEED": {"label": "Seed:", "tooltip": "Ensures reproducible training.", "widget": "QLineEdit"},
        "RESUME_MODEL_PATH": {"label": "Resume Model:", "tooltip": "The .safetensors checkpoint file.", "widget": "Path", "file_type": "file_safetensors"},
        "RESUME_STATE_PATH": {"label": "Resume State:", "tooltip": "The .pt optimizer state file.", "widget": "Path", "file_type": "file_pt"},
        "UNET_EXCLUDE_TARGETS": {"label": "Exclude Layers (Keywords):", "tooltip": "Keywords for layers to exclude from training (comma-separated).", "widget": "QPlainTextEdit", "height": 100},
        "LR_GRAPH_MIN": {"label": "Graph Min LR:", "tooltip": "The minimum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "LR_GRAPH_MAX": {"label": "Graph Max LR:", "tooltip": "The maximum learning rate displayed on the Y-axis.", "widget": "QLineEdit"},
        "NOISE_SCHEDULER": {"label": "Noise Scheduler:", "tooltip": "The noise scheduler to use for training. EulerDiscrete is experimental.", "widget": "QComboBox", "options": ["DDPMScheduler", "DDIMScheduler", "EulerDiscreteScheduler (Experimental)"]},
        "MEMORY_EFFICIENT_ATTENTION": {"label": "Attention Backend:", "tooltip": "Select the attention mechanism to use.", "widget": "QComboBox", "options": ["sdpa", "flex_attn", "cudnn","xformers (Only if no Flash)"]},
        "USE_ZERO_TERMINAL_SNR": {"label": "Use Zero-Terminal SNR", "tooltip": "Rescales noise schedule for better dynamic range.", "widget": "QCheckBox"},
        "GRAD_SPIKE_THRESHOLD_HIGH": {"label": "Spike Threshold (High):", "tooltip": "Trigger detector if gradient norm exceeds this value.", "widget": "QLineEdit"},
        "GRAD_SPIKE_THRESHOLD_LOW": {"label": "Spike Threshold (Low):", "tooltip": "Trigger detector if gradient norm is below this value.", "widget": "QLineEdit"},
        "NOISE_OFFSET": {"label": "Noise Offset Strength:", "tooltip": "Improves learning of dark/bright images. Try 0.05.", "widget": "QDoubleSpinBox", "range": (0.0, 0.5), "step": 0.01, "decimals": 3},
        "LOSS_TYPE": {
            "label": "Loss Type:", 
            "tooltip": "Select the loss function strategy.", 
            "widget": "QComboBox", 
            "options": ["MSE", "Semantic", "Huber_Adaptive"]
        },
        "SEMANTIC_CHAR_WEIGHT": {
            "label": "Character Region Weight:",
            "tooltip": "How strongly to weight the character/blob regions (salient colored areas). 0.0 = ignore character shape, 1.0 = full character detection, 2.0 = boosted. Stacks with detail via Soft Max.",
            "widget": "QDoubleSpinBox",
            "range": (0.0, 2.0),
            "step": 0.05,
            "decimals": 2
        },
        "SEMANTIC_DETAIL_WEIGHT": {
            "label": "Lineart/Detail Weight:",
            "tooltip": "How strongly to weight fine details, edges, and lineart. 0.0 = ignore lines, 1.0 = full edge detection, 2.0 = boosted line emphasis. Stacks with character via Soft Max without cancellation.",
            "widget": "QDoubleSpinBox",
            "range": (0.0, 2.0),
            "step": 0.05,
            "decimals": 2
        },
        # --- NEW HUBER PARAMS ---
        "LOSS_HUBER_BETA": {
            "label": "Huber Beta:", 
            "tooltip": "Threshold where quadratic loss turns linear (0.5 is standard). Lower = more robust to outliers.", 
            "widget": "QDoubleSpinBox", 
            "range": (0.01, 1.0), 
            "step": 0.05, 
            "decimals": 2
        },
        "LOSS_ADAPTIVE_DAMPING": {
            "label": "Adaptive Damping:", 
            "tooltip": "Controls sensitivity to magnitude. Prevents division by zero.", 
            "widget": "QDoubleSpinBox", 
            "range": (0.001, 1.0), 
            "step": 0.01, 
            "decimals": 3
        },
        # --- NEW VAE PARAMS ---
        "VAE_SHIFT_FACTOR": {"label": "VAE Shift Factor:", "tooltip": "Latent shift mean.", "widget": "QDoubleSpinBox", "range": (-10.0, 10.0), "step": 0.0001, "decimals": 4},
        "VAE_SCALING_FACTOR": {"label": "VAE Scaling Factor:", "tooltip": "Latent scaling factor.", "widget": "QDoubleSpinBox", "range": (0.0, 10.0), "step": 0.0001, "decimals": 5},
        "VAE_LATENT_CHANNELS": {"label": "Latent Channels:", "tooltip": "4 for Standard/EQ, 32 for Flux/NoobAI.", "widget": "QSpinBox", "range": (4, 128)},
        "RF_SHIFT_FACTOR": {
            "label": "RF Shift Factor:",
            "tooltip": "Shift factor for SD3/Flux schedules. (e.g., 3.0 for SD3, 1.15 for Flux-Schnell).",
            "widget": "QDoubleSpinBox",
            "range": (0.0, 100.0),
            "step": 0.01,
            "decimals": 4
        },
    }

    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA SDXL Trainer")
        self.setMinimumSize(QtCore.QSize(1000, 800))
        self.resize(1500, 1000)
        self.config_dir = "configs"
        self.state_file = os.path.join(self.config_dir, "gui_state.json")
        self.widgets = {}
        self.process_runner = None
        self.current_config = {}
        self.last_line_is_progress = False
        self.default_config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        self.presets = {}
        self.optimizer_steps_label = None
        self.epochs_label = None
        self.last_browsed_path = os.getcwd()
        self._initialize_configs()
        self._setup_ui()
        
        self.config_dropdown.blockSignals(True)
        
        # Load last selected config
        last_config = self._load_gui_state()
        if last_config and last_config in self.presets:
            index = self.config_dropdown.findData(last_config)
            if index != -1:
                self.config_dropdown.setCurrentIndex(index)
                self.load_selected_config(index)
            else:
                self._load_first_config()
        else:
            self._load_first_config()
        
        self.config_dropdown.blockSignals(False)

    def _load_first_config(self):
        """Helper to load the first available config"""
        if self.config_dropdown.count() > 0:
            self.config_dropdown.setCurrentIndex(0)
            self.load_selected_config(0)
        else:
            self.log("CRITICAL: No configs found or created. Using temporary defaults.")
            self.current_config = copy.deepcopy(self.default_config)
            self._apply_config_to_widgets()

    def _load_gui_state(self):
        """Load the last selected config from state file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return state.get("last_config")
        except (json.JSONDecodeError, IOError) as e:
            self.log(f"Warning: Could not load GUI state. Error: {e}")
        return None

    def _save_gui_state(self):
        """Save the currently selected config to state file"""
        try:
            index = self.config_dropdown.currentIndex()
            if index >= 0:
                selected_key = self.config_dropdown.itemData(index) if self.config_dropdown.itemData(index) else self.config_dropdown.itemText(index).replace(" ", "_").lower()
                state = {"last_config": selected_key}
                os.makedirs(self.config_dir, exist_ok=True)
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=4)
        except Exception as e:
            self.log(f"Warning: Could not save GUI state. Error: {e}")

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
        self._save_gui_state()

    def closeEvent(self, event):
        """Handle window close - save state and clean up"""
        self._save_gui_state()
        
        # Stop any running training process
        if self.process_runner and self.process_runner.isRunning():
            reply = QtWidgets.QMessageBox.question(
                self, 
                "Training in Progress",
                "Training is currently running. Do you want to stop it and exit?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.stop_training()
                if self.process_runner:
                    self.process_runner.quit()
                    self.process_runner.wait(2000)
                if os.name == 'nt':
                    prevent_sleep(False)
                event.accept()
            else:
                event.ignore()
                return
        
        if hasattr(self, 'dataset_manager'):
            for thread in self.dataset_manager.loader_threads:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(1000)
        
        if os.name == 'nt':
            prevent_sleep(False)
        
        event.accept()
    
    def paintEvent(self, event: QtGui.QPaintEvent):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
    
    def _initialize_configs(self):
        os.makedirs(self.config_dir, exist_ok=True)
        json_files = [f for f in os.listdir(self.config_dir) if f.endswith(".json") and f != "gui_state.json"]
        if not json_files:
            default_save_path = os.path.join(self.config_dir, "default.json")
            with open(default_save_path, 'w') as f: json.dump(self.default_config, f, indent=4)
            self.log("No configs found. Created 'default.json'.")
        self.presets = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json") and filename != "gui_state.json":
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.config_dir, filename)
                try:
                    with open(path, 'r') as f: self.presets[name] = json.load(f)
                except (json.JSONDecodeError, IOError) as e: self.log(f"Warning: Could not load config '{filename}'. Error: {e}")
    
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
        self._setup_bottom_bar()
    
    def _setup_corner_widget(self):
        corner_hbox = QtWidgets.QHBoxLayout()
        corner_hbox.setContentsMargins(10, 5, 10, 5)
        corner_hbox.setSpacing(10)
        self.config_dropdown = NoScrollComboBox()
        if not self.presets: self.log("CRITICAL: No presets loaded to populate dropdown.")
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
    
    def _setup_bottom_bar(self):
        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.setContentsMargins(0, 5, 5, 5)
        calc_group = QtWidgets.QGroupBox() 
        calc_group.setStyleSheet("QGroupBox { margin-top: 0px; border: 1px solid #4a4668; border-radius: 6px; padding: 0px 8px; }")
        calc_layout = QtWidgets.QHBoxLayout(calc_group)
        calc_layout.setContentsMargins(0, 0, 0, 0) 
        calc_layout.setSpacing(10)
        self.optimizer_steps_label = QtWidgets.QLabel("N/A")
        self.epochs_label = QtWidgets.QLabel("N/A")
        label_style = "color: #ab97e6; font-weight: bold; font-size: 14px;"
        self.optimizer_steps_label.setStyleSheet(label_style)
        self.epochs_label.setStyleSheet(label_style)
        calc_layout.addWidget(QtWidgets.QLabel("Total Optimizer Steps:"))
        calc_layout.addWidget(self.optimizer_steps_label)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        calc_layout.addWidget(separator)
        calc_layout.addWidget(QtWidgets.QLabel("Total Epochs:"))
        calc_layout.addWidget(self.epochs_label)
        bottom_layout.addWidget(calc_group)
        bottom_layout.addStretch()
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QtWidgets.QPushButton("Stop Training")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        bottom_layout.addWidget(self.start_button)
        bottom_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(bottom_layout)
    
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
        # Skip keys handled manually
        skip_keys = [
            "RESUME_TRAINING", "INSTANCE_DATASETS", "OPTIMIZER_TYPE", 
            "RAVEN_PARAMS", "TITAN_PARAMS", "VELORMS_PARAMS", "GRYPHON_PARAMS", 
            "NOISE_TYPE", "NOISE_OFFSET", 
            "LOSS_TYPE", "SEMANTIC_CHAR_WEIGHT", "SEMANTIC_DETAIL_WEIGHT",
            "LOSS_HUBER_BETA", "LOSS_ADAPTIVE_DAMPING",  # <--- ADDED HERE
            "TIMESTEP_ALLOCATION", "TIMESTEP_WEIGHTING_CURVE"
        ]
        
        for key in self.default_config.keys():
            if key in skip_keys: continue
            live_val = self.current_config.get(key)
            if live_val is None: continue
            
            try:
                if key == "LR_CUSTOM_CURVE":
                    config_to_save[key] = [[round(p[0], 8), round(p[1], 10)] for p in live_val]
                    continue
                config_to_save[key] = live_val
            except Exception as e:
                self.log(f"Warning: Could not save '{key}': {e}")
        
        # --- NEW: Explicitly save RF Params ---
        # We do this explicitly because they might not be in your default_config import yet
        if "RF_SHIFT_FACTOR" in self.widgets:
            config_to_save["RF_SHIFT_FACTOR"] = self.widgets["RF_SHIFT_FACTOR"].value()
        # --------------------------------------

        config_to_save["TRAINING_MODE"] = self.training_mode_combo.currentText()
        config_to_save["RESUME_TRAINING"] = self.model_load_strategy_combo.currentIndex() == 1
        config_to_save["INSTANCE_DATASETS"] = self.dataset_manager.get_datasets_config()
        config_to_save["OPTIMIZER_TYPE"] = self.widgets["OPTIMIZER_TYPE"].currentData()
        config_to_save["NOISE_SCHEDULER"] = self.widgets["NOISE_SCHEDULER"].currentText()
        config_to_save["NOISE_TYPE"] = self.widgets["NOISE_TYPE"].currentText()
        config_to_save["NOISE_OFFSET"] = self.widgets["NOISE_OFFSET"].value()
        config_to_save["LOSS_TYPE"] = self.widgets["LOSS_TYPE"].currentText()
        config_to_save["SEMANTIC_CHAR_WEIGHT"] = self.widgets["SEMANTIC_CHAR_WEIGHT"].value()
        config_to_save["SEMANTIC_DETAIL_WEIGHT"] = self.widgets["SEMANTIC_DETAIL_WEIGHT"].value()
        config_to_save["LOSS_HUBER_BETA"] = self.widgets["LOSS_HUBER_BETA"].value()             # <--- ADDED
        config_to_save["LOSS_ADAPTIVE_DAMPING"] = self.widgets["LOSS_ADAPTIVE_DAMPING"].value() # <--- ADDED
        config_to_save["TIMESTEP_MODE"] = self.ts_mode_combo.currentText()
        
        if hasattr(self, 'timestep_histogram'):
            config_to_save["TIMESTEP_ALLOCATION"] = self.timestep_histogram.get_allocation()

        # ... (Rest of the function remains the same: RAVEN, TITAN, VAE params) ...
        # (Be sure to include the rest of the existing function here)
        
        raven_params = {}
        try:
            r_betas = self.widgets['RAVEN_betas'].text().strip()
            raven_params["betas"] = [float(x.strip()) for x in r_betas.split(',')]
        except: raven_params["betas"] = [0.9, 0.999]
        try: raven_params["eps"] = float(self.widgets['RAVEN_eps'].text().strip())
        except: raven_params["eps"] = 1e-8
        raven_params["weight_decay"] = self.widgets['RAVEN_weight_decay'].value()
        raven_params["debias_strength"] = self.widgets['RAVEN_debias_strength'].value()
        raven_params["use_grad_centralization"] = self.widgets['RAVEN_use_grad_centralization'].isChecked()
        raven_params["gc_alpha"] = self.widgets['RAVEN_gc_alpha'].value()
        config_to_save["RAVEN_PARAMS"] = raven_params

        titan_params = {}
        try:
            t_betas = self.widgets['TITAN_betas'].text().strip()
            titan_params["betas"] = [float(x.strip()) for x in t_betas.split(',')]
        except: titan_params["betas"] = [0.9, 0.999]
        try: titan_params["eps"] = float(self.widgets['TITAN_eps'].text().strip())
        except: titan_params["eps"] = 1e-8
        titan_params["weight_decay"] = self.widgets['TITAN_weight_decay'].value()
        titan_params["debias_strength"] = self.widgets['TITAN_debias_strength'].value()
        titan_params["use_grad_centralization"] = self.widgets['TITAN_use_grad_centralization'].isChecked()
        titan_params["gc_alpha"] = self.widgets['TITAN_gc_alpha'].value()
        config_to_save["TITAN_PARAMS"] = titan_params

        velorms_params = {}
        velorms_params["momentum"] = self.widgets['VELORMS_momentum'].value()
        velorms_params["leak"] = self.widgets['VELORMS_leak'].value()
        velorms_params["weight_decay"] = self.widgets['VELORMS_weight_decay'].value()
        try: velorms_params["eps"] = float(self.widgets['VELORMS_eps'].text().strip())
        except: velorms_params["eps"] = 1e-8
        config_to_save["VELORMS_PARAMS"] = velorms_params

        config_to_save["VAE_SHIFT_FACTOR"] = self.widgets["VAE_SHIFT_FACTOR"].value()
        config_to_save["VAE_SCALING_FACTOR"] = self.widgets["VAE_SCALING_FACTOR"].value()
        config_to_save["VAE_LATENT_CHANNELS"] = self.widgets["VAE_LATENT_CHANNELS"].value()

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
            with open(save_path, 'w') as f: json.dump(config_to_save, f, indent=4)
            self.log(f"Successfully saved settings to {os.path.basename(save_path)}")
            self.presets[selected_key] = config_to_save
        except Exception as e: self.log(f"CRITICAL ERROR: Could not write to {save_path}. Error: {e}")
    
    def save_as_config(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset As", "Enter preset name (alphanumeric, underscores):")
        if ok and name:
            if not re.match(r'^[a-zA-Z0-9_]+$', name):
                self.log("Error: Preset name must be alphanumeric with underscores only.")
                return
            save_path = os.path.join(self.config_dir, f"{name}.json")
            if os.path.exists(save_path):
                reply = QtWidgets.QMessageBox.question(self, "Overwrite?", f"Preset '{name}' exists. Overwrite?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
                if reply != QtWidgets.QMessageBox.StandardButton.Yes: return
            config_to_save = self._prepare_config_to_save()
            try:
                with open(save_path, 'w') as f: json.dump(config_to_save, f, indent=4)
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
            except Exception as e: self.log(f"Error saving preset: {e}")
    
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
        elif widget_type == "QPlainTextEdit":
            widget = QtWidgets.QPlainTextEdit()
            if "height" in definition: widget.setFixedHeight(definition["height"])
            widget.textChanged.connect(lambda: self._update_config_from_widget(key, widget))
        elif widget_type == "QSpinBox":
            widget = NoScrollSpinBox()
            if "range" in definition: widget.setRange(*definition["range"])
            widget.valueChanged.connect(lambda value, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "QDoubleSpinBox":
            widget = NoScrollDoubleSpinBox()
            if "range" in definition: widget.setRange(*definition["range"])
            if "step" in definition: widget.setSingleStep(definition["step"])
            if "decimals" in definition: widget.setDecimals(definition["decimals"])
            widget.valueChanged.connect(lambda value, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "QComboBox":
            widget = NoScrollComboBox()
            widget.addItems(definition["options"])
            widget.currentTextChanged.connect(lambda text, k=key: self._update_config_from_widget(k, widget))
        elif widget_type == "QCheckBox":
            widget = QtWidgets.QCheckBox(definition["label"])
            widget.setToolTip(definition["tooltip"])
            widget.stateChanged.connect(lambda state, k=key: self._update_config_from_widget(k, widget))
            self.widgets[key] = widget
            return None, widget
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
    
    def _add_widget_to_form(self, form_layout, key):
        label, widget = self._create_widget(key)
        if widget:
            if label: form_layout.addRow(label, widget)
            else: form_layout.addRow(widget)

    def _add_stacked_path_widget(self, layout, key):
        """Custom helper to create a stacked layout (Label -> Input -> Browse) for paths."""
        definition = self.UI_DEFINITIONS[key]
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(2)

        label = QtWidgets.QLabel(definition["label"])
        label.setToolTip(definition["tooltip"])
        label.setStyleSheet("font-weight: bold; color: #ab97e6; margin-bottom: 2px;")
        vbox.addWidget(label)

        input_field = QtWidgets.QLineEdit()
        input_field.textChanged.connect(lambda text, k=key: self._update_config_from_widget(k, input_field))
        self.widgets[key] = input_field
        vbox.addWidget(input_field)

        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self._browse_path(input_field, definition["file_type"]))
        vbox.addWidget(browse_btn)

        layout.addWidget(container)
    
    def _populate_dataset_tab(self, parent_widget):
        layout = QtWidgets.QVBoxLayout(parent_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        top_hbox = QtWidgets.QHBoxLayout()
        top_hbox.setSpacing(20)
        
        groups = {
            "Batching & DataLoaders": ["CACHING_BATCH_SIZE", "NUM_WORKERS"],
            "Unconditional Dropout": ["UNCONDITIONAL_DROPOUT", "UNCONDITIONAL_DROPOUT_CHANCE"],
            "Aspect Ratio Bucketing": ["TARGET_PIXEL_AREA", "SHOULD_UPSCALE", "MAX_AREA_TOLERANCE"]
        }
        
        for title, keys in groups.items(): 
            top_hbox.addWidget(self._create_form_group(title, keys))
        layout.addLayout(top_hbox)
        
        self.dataset_manager = DatasetManagerWidget(self)
        self.dataset_manager.datasetsChanged.connect(self._update_training_calculations)
        self.dataset_manager.datasetsChanged.connect(self._update_epoch_markers_on_graph)
        layout.addWidget(self.dataset_manager)
        
        if "SHOULD_UPSCALE" in self.widgets and "MAX_AREA_TOLERANCE" in self.widgets:
            self.widgets["SHOULD_UPSCALE"].stateChanged.connect(lambda state: self.widgets["MAX_AREA_TOLERANCE"].setEnabled(bool(state)))

        if "UNCONDITIONAL_DROPOUT" in self.widgets:
            chk = self.widgets["UNCONDITIONAL_DROPOUT"]
            def update_dropout_widgets(state):
                enabled = bool(state)
                if "UNCONDITIONAL_DROPOUT_CHANCE" in self.widgets:
                    self.widgets["UNCONDITIONAL_DROPOUT_CHANCE"].setEnabled(enabled)

            chk.stateChanged.connect(update_dropout_widgets)
            update_dropout_widgets(chk.isChecked())
    
    def _populate_model_training_tab(self, parent_widget):
        main_layout = QtWidgets.QVBoxLayout(parent_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 5, 15, 15)
        
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(20)
        
        col1_container = QtWidgets.QWidget()
        col1_layout = QtWidgets.QVBoxLayout(col1_container)
        col1_layout.setContentsMargins(0,0,0,0)
        col1_layout.setSpacing(10)
        
        mode_group = QtWidgets.QGroupBox("Training Mode")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self.training_mode_combo = NoScrollComboBox()
        self.training_mode_combo.addItems(["Standard (SDXL)", "Rectified Flow (SDXL)"])
        self.training_mode_combo.currentTextChanged.connect(self._on_training_mode_changed)
        mode_layout.addWidget(self.training_mode_combo)
        col1_layout.addWidget(mode_group)
        
        col1_layout.addWidget(self._create_path_group())
        col1_layout.addStretch()
        
        row1.addWidget(col1_container, 1)
        row1.addWidget(self._create_lr_scheduler_group(), 2)
        
        self.timestep_group = self._create_timestep_sampling_group()
        row1.addWidget(self.timestep_group, 2)
        
        main_layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(20)
        row2.addWidget(self._create_core_training_group(), 1)
        self.scheduler_group = self._create_scheduler_config_group()
        row2.addWidget(self.scheduler_group, 1)

        # --- UPDATED LAYOUT FOR RIGHT COLUMN (VAE + UNET) ---
        right_col_widget = QtWidgets.QWidget()
        right_col_layout = QtWidgets.QVBoxLayout(right_col_widget)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.setSpacing(20)
        
        # VAE Group on Top
        right_col_layout.addWidget(self._create_vae_group())
        # UNet Group Below
        right_col_layout.addWidget(self._create_unet_group())
        
        row2.addWidget(right_col_widget, 1)
        # ----------------------------------------------------

        main_layout.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(20)
        row3.addWidget(self._create_optimizer_group(), 1)
        row3.addWidget(self._create_loss_group(), 1)
        self.noise_group = self._create_noise_enhancements_group()
        row3.addWidget(self.noise_group, 1)
        main_layout.addLayout(row3)

        row4 = QtWidgets.QHBoxLayout()
        row4.setSpacing(20)
        row4.addWidget(self._create_advanced_group(), 1)
        row4.addStretch(2)
        main_layout.addLayout(row4)

        self.widgets["MAX_TRAIN_STEPS"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["MAX_TRAIN_STEPS"].textChanged.connect(self._update_training_calculations)
        self.widgets["GRADIENT_ACCUMULATION_STEPS"].textChanged.connect(self._update_epoch_markers_on_graph)
        self.widgets["GRADIENT_ACCUMULATION_STEPS"].textChanged.connect(self._update_training_calculations)
        self.widgets["BATCH_SIZE"].valueChanged.connect(self._update_training_calculations)
        self._update_lr_button_states(-1)

    def _on_training_mode_changed(self, text):
        is_rf = "Rectified Flow (SDXL)" in text
        
        # Existing logic for scheduler/noise groups
        if hasattr(self, 'scheduler_group'): self.scheduler_group.setEnabled(not is_rf)
        if hasattr(self, 'noise_group'): self.noise_group.setEnabled(not is_rf)
        
        disabled_tooltip = "This configuration is hardcoded for SDXL Rectified Flow training."
        if is_rf:
            if hasattr(self, 'scheduler_group'): self.scheduler_group.setToolTip(disabled_tooltip)
            if hasattr(self, 'noise_group'): self.noise_group.setToolTip(disabled_tooltip)
        else:
            if hasattr(self, 'scheduler_group'): self.scheduler_group.setToolTip("")
            if hasattr(self, 'noise_group'): self.noise_group.setToolTip("")
        
        if hasattr(self, 'dataset_manager'): self.dataset_manager.refresh_cache_buttons()

        # --- NEW: Toggle Visibility of RF Params ---
        if hasattr(self, 'rf_params_container'):
            self.rf_params_container.setVisible(is_rf)
            if is_rf:
                self._update_rf_settings_state()
        # -------------------------------------------

    def _create_form_group(self, title, keys):
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QFormLayout(group)
        for key in keys: self._add_widget_to_form(layout, key)
        return group
    
    def _create_path_group(self):
        path_group = QtWidgets.QGroupBox("File & Directory Paths")
        path_layout = QtWidgets.QVBoxLayout(path_group)
        
        form_layout = QtWidgets.QFormLayout()
        self.model_load_strategy_combo = NoScrollComboBox()
        self.model_load_strategy_combo.addItems(["Load Base Model", "Resume from Checkpoint"])
        form_layout.addRow("Mode:", self.model_load_strategy_combo)
        path_layout.addLayout(form_layout)
        
        self.path_stacked_widget = QtWidgets.QStackedWidget()
        
        base_model_widget = QtWidgets.QWidget()
        base_layout = QtWidgets.QVBoxLayout(base_model_widget)
        base_layout.setContentsMargins(0, 5, 0, 0)
        base_layout.setSpacing(10)
        self._add_stacked_path_widget(base_layout, "SINGLE_FILE_CHECKPOINT_PATH")
        self._add_stacked_path_widget(base_layout, "VAE_PATH")
        self.path_stacked_widget.addWidget(base_model_widget)
        
        resume_widget = QtWidgets.QWidget()
        resume_layout = QtWidgets.QVBoxLayout(resume_widget)
        resume_layout.setContentsMargins(0, 5, 0, 0)
        resume_layout.setSpacing(10)
        self._add_stacked_path_widget(resume_layout, "RESUME_MODEL_PATH")
        self._add_stacked_path_widget(resume_layout, "RESUME_STATE_PATH")
        self.path_stacked_widget.addWidget(resume_widget)
        
        path_layout.addWidget(self.path_stacked_widget)
        
        self._add_stacked_path_widget(path_layout, "OUTPUT_DIR")
        
        self.model_load_strategy_combo.currentIndexChanged.connect(self.toggle_resume_widgets)
        return path_group
    
    def _create_core_training_group(self):
        core_group = QtWidgets.QGroupBox("Core Training Parameters")
        layout = QtWidgets.QFormLayout(core_group)
        
        # Standard Params
        core_keys = ["MAX_TRAIN_STEPS", "BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "SAVE_EVERY_N_STEPS", "MIXED_PRECISION", "CLIP_GRAD_NORM", "SEED"]
        for key in core_keys: 
            self._add_widget_to_form(layout, key)

        # --- NEW: Rectified Flow Specific Container ---
        self.rf_params_container = QtWidgets.QWidget()
        rf_layout = QtWidgets.QFormLayout(self.rf_params_container)
        rf_layout.setContentsMargins(0, 0, 0, 0) # Tight fit
        
        self._add_widget_to_form(rf_layout, "RF_SHIFT_FACTOR")
        

        # Add the container to the main layout
        layout.addRow(self.rf_params_container)
        # ----------------------------------------------

        return core_group
    
    def _build_optimizer_form(self, prefix):
        """Helper to create a layout of widgets for a specific optimizer prefix."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        layout.setContentsMargins(0, 5, 0, 0)
        
        self.widgets[f'{prefix}_betas'] = QtWidgets.QLineEdit()
        self.widgets[f'{prefix}_eps'] = QtWidgets.QLineEdit()
        
        wd = NoScrollDoubleSpinBox()
        wd.setRange(0.0, 1.0)
        wd.setSingleStep(0.00001)
        wd.setDecimals(6)
        self.widgets[f'{prefix}_weight_decay'] = wd
        
        debias = NoScrollDoubleSpinBox()
        debias.setRange(0.0, 1.0); debias.setSingleStep(0.01); debias.setDecimals(3)
        debias.setToolTip("Controls the strength of bias correction.")
        self.widgets[f'{prefix}_debias_strength'] = debias
        
        use_gc = QtWidgets.QCheckBox("Enable Gradient Centralization")
        use_gc.setToolTip("Improves convergence by centering gradients.")
        self.widgets[f'{prefix}_use_grad_centralization'] = use_gc
        
        gc_alpha = NoScrollDoubleSpinBox()
        gc_alpha.setRange(0.0, 1.0); gc_alpha.setSingleStep(0.1); gc_alpha.setDecimals(1)
        gc_alpha.setToolTip("Strength of gradient centralization.")
        self.widgets[f'{prefix}_gc_alpha'] = gc_alpha
        
        use_gc.stateChanged.connect(lambda state: gc_alpha.setEnabled(bool(state)))
        
        layout.addRow("Betas (b1, b2):", self.widgets[f'{prefix}_betas'])
        layout.addRow("Epsilon (eps):", self.widgets[f'{prefix}_eps'])
        layout.addRow("Weight Decay:", self.widgets[f'{prefix}_weight_decay'])
        layout.addRow("Debias Strength:", self.widgets[f'{prefix}_debias_strength'])
        layout.addRow(use_gc)
        layout.addRow("GC Alpha:", self.widgets[f'{prefix}_gc_alpha'])
        
        return container

    def _build_velorms_form(self, prefix):
        """Helper for VeloRMS specific optimizer settings."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        layout.setContentsMargins(0, 5, 0, 0)

        # Momentum
        mom = NoScrollDoubleSpinBox()
        mom.setRange(0.0, 0.999)
        mom.setSingleStep(0.01)
        mom.setDecimals(3)
        mom.setToolTip("Momentum factor (default: 0.86)")
        self.widgets[f'{prefix}_momentum'] = mom
        
        # Leak
        leak = NoScrollDoubleSpinBox()
        leak.setRange(0.0, 1.0)
        leak.setSingleStep(0.01)
        leak.setDecimals(3)
        leak.setToolTip("Gradient leakage into RMS (default: 0.16)")
        self.widgets[f'{prefix}_leak'] = leak
        
        # Weight Decay
        wd = NoScrollDoubleSpinBox()
        wd.setRange(0.0, 1.0)
        wd.setSingleStep(0.00001)
        wd.setDecimals(6)
        wd.setToolTip("Weight decay coefficient (default: 0.01)")
        self.widgets[f'{prefix}_weight_decay'] = wd

        # Eps
        eps = QtWidgets.QLineEdit()
        eps.setToolTip("Term added to denominator for stability (default: 1e-8)")
        self.widgets[f'{prefix}_eps'] = eps
        
        layout.addRow("Momentum:", mom)
        layout.addRow("Leak:", leak)
        layout.addRow("Weight Decay:", wd)
        layout.addRow("Epsilon (eps):", eps)
        
        return container

    def _create_optimizer_group(self):
        optimizer_group = QtWidgets.QGroupBox("Optimizer")
        main_layout = QtWidgets.QVBoxLayout(optimizer_group)
        
        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.addWidget(QtWidgets.QLabel("Optimizer Type:"))
        
        self.widgets["OPTIMIZER_TYPE"] = NoScrollComboBox()
        self.widgets["OPTIMIZER_TYPE"].addItem("Raven: Balanced (~12GB VRAM)", "raven")
        self.widgets["OPTIMIZER_TYPE"].addItem("Titan: VRAM Savings (~6GB VRAM, Slower)", "titan")
        self.widgets["OPTIMIZER_TYPE"].addItem("VeloRMS: CPU Offload (Experimental)", "velorms")
        
        self.widgets["OPTIMIZER_TYPE"].currentIndexChanged.connect(self._toggle_optimizer_widgets)
        selector_layout.addWidget(self.widgets["OPTIMIZER_TYPE"], 1)
        main_layout.addLayout(selector_layout)
        
        self.optimizer_settings_group = QtWidgets.QGroupBox("Optimizer Settings")
        group_layout = QtWidgets.QVBoxLayout(self.optimizer_settings_group)
        
        self.optimizer_stack = QtWidgets.QStackedWidget()
        self.raven_page = self._build_optimizer_form("RAVEN")
        self.optimizer_stack.addWidget(self.raven_page)
        self.titan_page = self._build_optimizer_form("TITAN")
        self.optimizer_stack.addWidget(self.titan_page)
        self.velorms_page = self._build_velorms_form("VELORMS")
        self.optimizer_stack.addWidget(self.velorms_page)
        
        group_layout.addWidget(self.optimizer_stack)
        main_layout.addWidget(self.optimizer_settings_group)
        
        return optimizer_group
    
    def _toggle_optimizer_widgets(self):
        selected_id = self.widgets["OPTIMIZER_TYPE"].currentData()
        
        if selected_id == "titan":
            self.optimizer_stack.setCurrentIndex(1)
        elif selected_id == "velorms":
            self.optimizer_stack.setCurrentIndex(2)
        else:
            self.optimizer_stack.setCurrentIndex(0) 
            
        if hasattr(self, 'optimizer_settings_group'):
            self.optimizer_settings_group.setVisible(True)

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
        lr_controls_layout.addWidget(self.add_point_btn); lr_controls_layout.addWidget(self.remove_point_btn); lr_controls_layout.addStretch()
        lr_layout.addLayout(lr_controls_layout)
        presets_group = QtWidgets.QWidget()
        presets_grid = QtWidgets.QGridLayout(presets_group)
        presets_grid.setContentsMargins(0, 0, 0, 0)
        presets_grid.setSpacing(10)
        presets_grid.addWidget(QtWidgets.QLabel("Restarts:"), 0, 0)
        restarts_spin = NoScrollSpinBox()
        restarts_spin.setRange(1, 50); restarts_spin.setValue(1); restarts_spin.setToolTip("Number of restart cycles (1 = standard single cycle)")
        presets_grid.addWidget(restarts_spin, 0, 1)
        presets_grid.addWidget(QtWidgets.QLabel("Initial Warmup %:"), 1, 0)
        initial_warmup_spin = NoScrollDoubleSpinBox()
        initial_warmup_spin.setRange(0.0, 100.0); initial_warmup_spin.setSingleStep(1.0); initial_warmup_spin.setValue(5.0); initial_warmup_spin.setSuffix("%"); initial_warmup_spin.setToolTip("Percentage of the first cycle spent warming up")
        presets_grid.addWidget(initial_warmup_spin, 1, 1)
        presets_grid.addWidget(QtWidgets.QLabel("Restart Rampup %:"), 2, 0)
        restart_rampup_spin = NoScrollDoubleSpinBox()
        restart_rampup_spin.setRange(0.0, 100.0); restart_rampup_spin.setSingleStep(1.0); restart_rampup_spin.setValue(0.0); restart_rampup_spin.setSuffix("%"); restart_rampup_spin.setToolTip("Percentage of subsequent cycles spent ramping up")
        presets_grid.addWidget(restart_rampup_spin, 2, 1)
        apply_cosine_btn = QtWidgets.QPushButton("Apply Cosine")
        apply_cosine_btn.clicked.connect(lambda: self.lr_curve_widget.set_generated_preset("Cosine", restarts_spin.value(), initial_warmup_spin.value() / 100.0, restart_rampup_spin.value() / 100.0))
        presets_grid.addWidget(apply_cosine_btn, 3, 0)
        apply_linear_btn = QtWidgets.QPushButton("Apply Linear")
        apply_linear_btn.clicked.connect(lambda: self.lr_curve_widget.set_generated_preset("Linear", restarts_spin.value(), initial_warmup_spin.value() / 100.0, restart_rampup_spin.value() / 100.0))
        presets_grid.addWidget(apply_linear_btn, 3, 1)
        lr_layout.addWidget(presets_group)
        graph_bounds_layout = QtWidgets.QFormLayout()
        self._add_widget_to_form(graph_bounds_layout, "LR_GRAPH_MIN")
        self._add_widget_to_form(graph_bounds_layout, "LR_GRAPH_MAX")
        lr_layout.addLayout(graph_bounds_layout)
        self.widgets["LR_GRAPH_MIN"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["LR_GRAPH_MAX"].textChanged.connect(self._update_and_clamp_lr_graph)
        return lr_group

    def _create_scheduler_config_group(self):
        group = QtWidgets.QGroupBox("Scheduler Configuration")
        layout = QtWidgets.QFormLayout(group)
        self._add_widget_to_form(layout, "NOISE_SCHEDULER")
        self._add_widget_to_form(layout, "PREDICTION_TYPE")
        self._add_widget_to_form(layout, "BETA_SCHEDULE")
        _label, snr_checkbox = self._create_widget("USE_ZERO_TERMINAL_SNR")
        if snr_checkbox: layout.addRow(snr_checkbox)
        return group

    def _create_noise_enhancements_group(self):
        group = QtWidgets.QGroupBox("Noise Configuration")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(10)
        form_layout = QtWidgets.QFormLayout()
        self.widgets["NOISE_TYPE"] = NoScrollComboBox()
        self.widgets["NOISE_TYPE"].addItems(["Default", "Offset"])
        self.widgets["NOISE_TYPE"].currentTextChanged.connect(self._toggle_noise_widgets)
        form_layout.addRow("Noise Type:", self.widgets["NOISE_TYPE"])
        layout.addLayout(form_layout)
        self.offset_noise_container = QtWidgets.QWidget()
        offset_layout = QtWidgets.QFormLayout(self.offset_noise_container)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        self._add_widget_to_form(offset_layout, "NOISE_OFFSET")
        layout.addWidget(self.offset_noise_container)
        layout.addStretch(1)
        return group

    def _toggle_noise_widgets(self):
        noise_type = self.widgets["NOISE_TYPE"].currentText()
        self.offset_noise_container.setVisible(noise_type == "Offset")

    def _create_loss_group(self):
        group = QtWidgets.QGroupBox("Loss Configuration")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(10)
        
        form_layout = QtWidgets.QFormLayout()
        
        # 1. Loss Type Dropdown
        self.widgets["LOSS_TYPE"] = NoScrollComboBox()
        # Ensure these options match exactly what the training code expects
        self.widgets["LOSS_TYPE"].addItems(["MSE", "Semantic", "Huber_Adaptive"])
        self.widgets["LOSS_TYPE"].currentTextChanged.connect(self._toggle_loss_widgets)
        form_layout.addRow("Loss Type:", self.widgets["LOSS_TYPE"])
        layout.addLayout(form_layout)
        
        # 2. Container for Semantic Loss (Hidden by default)
        self.semantic_loss_container = QtWidgets.QWidget()
        semantic_layout = QtWidgets.QFormLayout(self.semantic_loss_container)
        semantic_layout.setContentsMargins(0, 0, 0, 0)
        self._add_widget_to_form(semantic_layout, "SEMANTIC_CHAR_WEIGHT")
        self._add_widget_to_form(semantic_layout, "SEMANTIC_DETAIL_WEIGHT")
        layout.addWidget(self.semantic_loss_container)

        # 3. Container for Huber Adaptive Loss (Hidden by default)
        self.huber_loss_container = QtWidgets.QWidget()
        huber_layout = QtWidgets.QFormLayout(self.huber_loss_container)
        huber_layout.setContentsMargins(0, 0, 0, 0)
        self._add_widget_to_form(huber_layout, "LOSS_HUBER_BETA")
        self._add_widget_to_form(huber_layout, "LOSS_ADAPTIVE_DAMPING")
        layout.addWidget(self.huber_loss_container)
        
        layout.addStretch(1)
        return group

    def _toggle_loss_widgets(self):
        loss_type = self.widgets["LOSS_TYPE"].currentText()
        
        # Show Semantic controls only if Semantic is selected
        self.semantic_loss_container.setVisible(loss_type == "Semantic")
        
        # Show Huber controls only if Huber_Adaptive is selected
        self.huber_loss_container.setVisible(loss_type == "Huber_Adaptive")

    def _add_slider_row(self, layout, label_text, min_val, max_val, default_val, divider):
            """Helper to create a Label | Slider | SpinBox row."""
            container = QtWidgets.QWidget()
            h_layout = QtWidgets.QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl = QtWidgets.QLabel(label_text)
            lbl.setFixedWidth(90)
            h_layout.addWidget(lbl)
            
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * divider), int(max_val * divider))
            slider.setValue(int(default_val * divider))
            
            spin = NoScrollDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setSingleStep(0.1)
            spin.setValue(default_val)
            spin.setDecimals(2)
            # CHANGED: Increased width from 70 to 120 to fix "skinny" look
            spin.setFixedWidth(120)

            # Sync Logic
            def on_slider_change(val):
                new_float = val / divider
                if abs(spin.value() - new_float) > (0.5 / divider):
                    spin.setValue(new_float)
                self._update_timestep_distribution() # Trigger update

            def on_spin_change(val):
                new_int = int(val * divider)
                if abs(slider.value() - new_int) > 1:
                    slider.setValue(new_int)
                self._update_timestep_distribution() # Trigger update

            slider.valueChanged.connect(on_slider_change)
            spin.valueChanged.connect(on_spin_change)

            h_layout.addWidget(slider)
            h_layout.addWidget(spin)
            
            layout.addRow(container)
            return slider, spin
    
    def _reset_timestep_sliders(self):
        mode = self.ts_mode_combo.currentText()
        if mode == "Wave":
            self.spin_wave_amp.setValue(0.0)
            self.spin_wave_freq.setValue(1.0)
            self.spin_wave_phase.setValue(0.0)
        elif mode == "Logit-Normal":
            self.spin_ln_mu.setValue(0.0)
            self.spin_ln_sigma.setValue(1.0)
        elif mode == "Beta":
            self.spin_beta_alpha.setValue(3.0)
            self.spin_beta_beta.setValue(3.0)

    def _create_timestep_sampling_group(self):
        group = QtWidgets.QGroupBox("Timestep Ticket Allocation")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(10)

        # 1. Histogram Visualization (Top)
        self.timestep_histogram = TimestepHistogramWidget()
        self.widgets["TIMESTEP_ALLOCATION"] = self.timestep_histogram
        layout.addWidget(self.timestep_histogram)
        
        self.timestep_histogram.allocationChanged.connect(
            lambda data: self._update_config_from_widget("TIMESTEP_ALLOCATION", self.timestep_histogram)
        )

        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setContentsMargins(5, 5, 5, 5)

        # 2. Bin Size and Mode Dropdown
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Bin Size:"))
        self.bin_size_combo = NoScrollComboBox()
        self.bin_size_combo.addItems(["30", "40", "50", "100", "200", "250", "500"])
        self.bin_size_combo.setCurrentText("100")
        row1.addWidget(self.bin_size_combo)
        row1.addSpacing(20)
        row1.addWidget(QtWidgets.QLabel("Distribution Mode:"))
        self.ts_mode_combo = NoScrollComboBox()
        self.ts_mode_combo.addItems(["Wave", "Logit-Normal", "Beta"])
        row1.addWidget(self.ts_mode_combo, 1)
        controls_layout.addLayout(row1)

        # 3. Stacked Widget for Mode-Specific Sliders
        self.ts_slider_stack = QtWidgets.QStackedWidget()
        
        # --- Wave Mode Page (Page 0) ---
        wave_page = QtWidgets.QWidget()
        wave_layout = QtWidgets.QFormLayout(wave_page)
        wave_layout.setContentsMargins(0, 0, 0, 0)
        
        self.slider_wave_freq, self.spin_wave_freq = self._add_slider_row(
            wave_layout, "Frequency:", 0.0, 5.0, 1.0, 100.0
        )
        self.slider_wave_phase, self.spin_wave_phase = self._add_slider_row(
            wave_layout, "Phase:", 0.0, 6.28, 0.0, 100.0
        )
        self.slider_wave_amp, self.spin_wave_amp = self._add_slider_row(
            wave_layout, "Amplitude:", 0.0, 1.0, 0.0, 100.0
        )
        self.ts_slider_stack.addWidget(wave_page)

        # --- Logit-Normal Mode Page (Page 1) ---
        logit_page = QtWidgets.QWidget()
        logit_layout = QtWidgets.QFormLayout(logit_page)
        logit_layout.setContentsMargins(0, 0, 0, 0)
        
        self.slider_ln_mu, self.spin_ln_mu = self._add_slider_row(
            logit_layout, "Center (Mu):", -3.0, 3.0, 0.0, 100.0
        )
        self.slider_ln_sigma, self.spin_ln_sigma = self._add_slider_row(
            logit_layout, "Spread (Sigma):", 0.1, 3.0, 1.0, 100.0
        )
        self.ts_slider_stack.addWidget(logit_page)

        # --- Beta Mode Page (Page 2) ---
        beta_page = QtWidgets.QWidget()
        beta_layout = QtWidgets.QFormLayout(beta_page)
        beta_layout.setContentsMargins(0, 0, 0, 0)
        
        self.slider_beta_alpha, self.spin_beta_alpha = self._add_slider_row(
            beta_layout, "Alpha:", 0.1, 10.0, 3.0, 100.0
        )
        self.slider_beta_beta, self.spin_beta_beta = self._add_slider_row(
            beta_layout, "Beta:", 0.1, 10.0, 3.0, 100.0
        )
        self.ts_slider_stack.addWidget(beta_page)
        
        controls_layout.addWidget(self.ts_slider_stack)

        # 4. Stacked Widget for Presets Buttons
        self.ts_button_stack = QtWidgets.QStackedWidget()
        
        def create_preset_btn(text, callback):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(callback)
            btn.setStyleSheet("padding: 4px; font-size: 11px;")
            return btn

        # --- Wave Presets ---
        wave_btn_page = QtWidgets.QWidget()
        wave_btn_layout = QtWidgets.QGridLayout(wave_btn_page)
        wave_btn_layout.setSpacing(8)
        wave_btn_layout.setContentsMargins(0, 0, 0, 0)
        wave_btn_layout.addWidget(create_preset_btn("Uniform (Flat)", lambda: self._apply_timestep_preset("Uniform")), 0, 0)
        wave_btn_layout.addWidget(create_preset_btn("Peak Ends", lambda: self._apply_timestep_preset("Peak Ends")), 0, 1)
        wave_btn_layout.addWidget(create_preset_btn("Peak Middle", lambda: self._apply_timestep_preset("Peak Middle")), 0, 2)
        self.ts_button_stack.addWidget(wave_btn_page)

        # --- Logit-Normal Presets ---
        ln_btn_page = QtWidgets.QWidget()
        ln_btn_layout = QtWidgets.QGridLayout(ln_btn_page)
        ln_btn_layout.setSpacing(8)
        ln_btn_layout.setContentsMargins(0, 0, 0, 0)
        ln_btn_layout.addWidget(create_preset_btn("Bell Curve", lambda: self._apply_timestep_preset("Bell Curve")), 0, 0)
        ln_btn_layout.addWidget(create_preset_btn("Detail (Early)", lambda: self._apply_timestep_preset("Detail")), 0, 1)
        ln_btn_layout.addWidget(create_preset_btn("Structure (Late)", lambda: self._apply_timestep_preset("Structure")), 0, 2)
        ln_btn_layout.addWidget(create_preset_btn("Logit-Normal (RF/SD3 Recommended)", lambda: self._apply_timestep_preset("Logit-Normal (RF/SD3 Recommended)")), 1, 0)
        self.ts_button_stack.addWidget(ln_btn_page)

        # --- Beta Presets ---
        beta_btn_page = QtWidgets.QWidget()
        beta_btn_layout = QtWidgets.QGridLayout(beta_btn_page)
        beta_btn_layout.setSpacing(8)
        beta_btn_layout.setContentsMargins(0, 0, 0, 0)
        beta_btn_layout.addWidget(create_preset_btn("Symmetric (3,3)", lambda: self._apply_timestep_preset("Beta Symmetric")), 0, 0)
        beta_btn_layout.addWidget(create_preset_btn("Right Skew (2,5)", lambda: self._apply_timestep_preset("Beta Right Skew")), 0, 1)
        beta_btn_layout.addWidget(create_preset_btn("Left Skew (5,2)", lambda: self._apply_timestep_preset("Beta Left Skew")), 0, 2)
        beta_btn_layout.addWidget(create_preset_btn("U-Shape (0.5,0.5)", lambda: self._apply_timestep_preset("Beta U-Shape")), 1, 0)
        self.ts_button_stack.addWidget(beta_btn_page)

        controls_layout.addWidget(self.ts_button_stack)
        layout.addLayout(controls_layout)

        # --- Signal Connections ---
        self.bin_size_combo.currentTextChanged.connect(lambda txt: self.timestep_histogram.set_bin_size(int(txt)))
        
        # Mode switching (Link both Slider Stack and Button Stack)
        self.ts_mode_combo.currentIndexChanged.connect(self.ts_slider_stack.setCurrentIndex)
        self.ts_mode_combo.currentIndexChanged.connect(self.ts_button_stack.setCurrentIndex)
        self.ts_mode_combo.currentIndexChanged.connect(self._update_timestep_distribution)

        return group

    def _apply_timestep_preset(self, name):
        """Sets spinbox values to replicate a preset configuration. SpinBox changes trigger updates."""
        # Block signals briefly? No, we want the signals to fire to update the slider and the graph.
        
        if name == "Uniform":
            self.ts_mode_combo.setCurrentText("Wave")
            self.spin_wave_amp.setValue(0.0) # Flat
            
        elif name == "Bell Curve":
            self.ts_mode_combo.setCurrentText("Logit-Normal")
            self.spin_ln_mu.setValue(0.0)
            self.spin_ln_sigma.setValue(1.0)
            
        elif name == "Detail": # Bias Early (Negative Mu)
            self.ts_mode_combo.setCurrentText("Logit-Normal")
            self.spin_ln_mu.setValue(-1.0)
            self.spin_ln_sigma.setValue(0.8)
            
        elif name == "Structure": # Bias Late (Positive Mu)
            self.ts_mode_combo.setCurrentText("Logit-Normal")
            self.spin_ln_mu.setValue(1.0)
            self.spin_ln_sigma.setValue(0.8)

        elif name == "Peak Ends": # U-Shape (High-Low-High)
            self.ts_mode_combo.setCurrentText("Wave")
            self.spin_wave_freq.setValue(1.0)
            self.spin_wave_phase.setValue(0.0) # Phase 0 is Cos(x), starts at 1
            self.spin_wave_amp.setValue(0.8)
            
        elif name == "Peak Middle": # Hill-Shape (Low-High-Low)
            self.ts_mode_combo.setCurrentText("Wave")
            self.spin_wave_freq.setValue(1.0)
            self.spin_wave_phase.setValue(3.14) # Phase PI flips Cos(x), starts at -1
            self.spin_wave_amp.setValue(0.6)

        elif name == "Logit-Normal (RF/SD3 Recommended)":
            self.ts_mode_combo.setCurrentText("Logit-Normal")
            self.spin_ln_mu.setValue(-0.5)
            self.spin_ln_sigma.setValue(1.0)

        elif name == "Beta Symmetric":
            self.ts_mode_combo.setCurrentText("Beta")
            self.spin_beta_alpha.setValue(3.0)
            self.spin_beta_beta.setValue(3.0)

        elif name == "Beta Right Skew":
            self.ts_mode_combo.setCurrentText("Beta")
            self.spin_beta_alpha.setValue(2.0)
            self.spin_beta_beta.setValue(5.0)

        elif name == "Beta Left Skew":
            self.ts_mode_combo.setCurrentText("Beta")
            self.spin_beta_alpha.setValue(5.0)
            self.spin_beta_beta.setValue(2.0)

        elif name == "Beta U-Shape":
            self.ts_mode_combo.setCurrentText("Beta")
            self.spin_beta_alpha.setValue(0.5)
            self.spin_beta_beta.setValue(0.5)

    def _update_timestep_distribution(self):
        """Calculates weights based on current sliders/spinboxes and sends to histogram."""
        mode = self.ts_mode_combo.currentText()
        weights = []
        
        bin_size = int(self.bin_size_combo.currentText())
        num_bins = math.ceil(1000 / bin_size)
        if num_bins <= 0: num_bins = 1

        if mode == "Wave":
            freq = self.spin_wave_freq.value()
            phase = self.spin_wave_phase.value()
            amp = self.spin_wave_amp.value()
            
            # W(t) = 1 + A * cos(2*pi*f*t + p)
            # t is normalized 0 to 1
            for i in range(num_bins):
                t = i / max(1, num_bins - 1)
                val = 1.0 + amp * math.cos(2 * math.pi * freq * t + phase)
                weights.append(max(0.0, val))
                
        elif mode == "Logit-Normal":
            mu = self.spin_ln_mu.value()
            sigma = self.spin_ln_sigma.value()
            
            def logit(p):
                return math.log(p / (1.0 - p))
            
            def normal_cdf(x):
                return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
            
            for i in range(num_bins):
                # Calculate bin boundaries in normalized [0, 1] space
                t_start = i * bin_size
                t_end = min((i + 1) * bin_size, 1000)
                
                eps = 1e-6
                x_start = max(t_start / 1000.0, eps)
                x_end = min(t_end / 1000.0, 1.0 - eps)
                
                # CDF(end) - CDF(start) = Probability mass in bin
                prob_start = normal_cdf((logit(x_start) - mu) / sigma)
                prob_end = normal_cdf((logit(x_end) - mu) / sigma)
                
                weight = prob_end - prob_start
                weights.append(max(0.0, weight))

        elif mode == "Beta":
            alpha = self.spin_beta_alpha.value()
            beta = self.spin_beta_beta.value()

            # Beta PDF: f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
            # We omit B(a,b) because we normalize weights anyway.
            for i in range(num_bins):
                # Sample at the center of the bin
                t_center = (i * bin_size) + (bin_size / 2)
                x = t_center / 1000.0
                
                # Clamp slightly to avoid 0^neg or 1^neg errors
                eps = 1e-4
                x = max(eps, min(1.0 - eps, x))
                
                val = (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))
                weights.append(max(0.0, val))

        self.timestep_histogram.generate_from_weights(weights)
    
    def _create_unet_group(self):
        unet_group = QtWidgets.QGroupBox("UNet Layer Exclusion")
        layout = QtWidgets.QFormLayout(unet_group)
        self._add_widget_to_form(layout, "UNET_EXCLUDE_TARGETS")
        return unet_group

    def _create_vae_group(self):
        vae_group = QtWidgets.QGroupBox("VAE Configuration")
        layout = QtWidgets.QVBoxLayout(vae_group)
        layout.setSpacing(10)
        
        # Inputs
        form_layout = QtWidgets.QFormLayout()
        self._add_widget_to_form(form_layout, "VAE_LATENT_CHANNELS")
        self._add_widget_to_form(form_layout, "VAE_SHIFT_FACTOR")
        self._add_widget_to_form(form_layout, "VAE_SCALING_FACTOR")
        layout.addLayout(form_layout)
        
        # Presets Buttons
        presets_label = QtWidgets.QLabel("Presets:")
        presets_label.setStyleSheet("color: #ab97e6; font-weight: bold;")
        layout.addWidget(presets_label)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_sdxl = QtWidgets.QPushButton("Standard SDXL")
        btn_sdxl.clicked.connect(lambda: self._apply_vae_preset(0.0, 0.13025, 4))
        btn_flux = QtWidgets.QPushButton("Flux/NoobAI (32ch)")
        btn_flux.clicked.connect(lambda: self._apply_vae_preset(0.0760, 0.6043, 32))
        btn_eq = QtWidgets.QPushButton("EQ VAE")
        btn_eq.clicked.connect(lambda: self._apply_vae_preset(0.1726, 0.1280, 4))
        
        btn_layout.addWidget(btn_sdxl)
        btn_layout.addWidget(btn_flux)
        btn_layout.addWidget(btn_eq)
        layout.addLayout(btn_layout)
        
        # Auto Detect Button
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet("border: 1px solid #4a4668; margin: 5px 0;")
        layout.addWidget(separator)
        
        btn_detect = QtWidgets.QPushButton("Run Auto-Detect Tool")
        btn_detect.setStyleSheet("background-color: #383552; color: #ab97e6; border: 1px solid #6a48d7;")
        btn_detect.setToolTip("Launches a separate tool to scan your VAE and suggest values.")
        btn_detect.clicked.connect(self.launch_vae_detector)
        layout.addWidget(btn_detect)

        return vae_group

    def _apply_vae_preset(self, shift, scale, channels):
        """Helper to set VAE GUI values from buttons"""
        self.widgets["VAE_SHIFT_FACTOR"].setValue(shift)
        self.widgets["VAE_SCALING_FACTOR"].setValue(scale)
        self.widgets["VAE_LATENT_CHANNELS"].setValue(channels)
        self.log(f"Applied VAE Preset: Shift={shift}, Scale={scale}, Ch={channels}")

    def launch_vae_detector(self):
        """
        Launches the external VAE detection script, passing the current config path.
        """
        # --- GET CURRENT CONFIG PATH ---
        index = self.config_dropdown.currentIndex()
        if index < 0:
            QtWidgets.QMessageBox.warning(self, "No Config Selected", "Please select a configuration from the dropdown first.")
            return

        selected_key = self.config_dropdown.itemData(index)
        config_path = os.path.join(self.config_dir, f"{selected_key}.json")

        if not os.path.exists(config_path):
            QtWidgets.QMessageBox.critical(self, "Config Not Found", f"The configuration file could not be found:\n{config_path}\n\nPlease save the configuration first.")
            return

        # --- FIND THE SCRIPT ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "vae_diagnostics.py"),
            os.path.join(script_dir, "tools", "vae_diagnostics.py")
        ]
        
        target_script = None
        for p in possible_paths:
            if os.path.exists(p):
                target_script = p
                break
        
        if not target_script:
            self.log("Error: Could not find 'vae_diagnostics.py' in root or tools/ folder.")
            QtWidgets.QMessageBox.warning(self, "Tool Not Found", "Could not find 'vae_diagnostics.py'.")
            return

        # --- LAUNCH THE SCRIPT WITH ARGUMENT ---
        try:
            command = [sys.executable, target_script, "--config", os.path.abspath(config_path)]
            
            if os.name == 'nt':
                subprocess.Popen(command, cwd=os.path.dirname(target_script), creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(command, cwd=os.path.dirname(target_script))
                
            self.log(f"Launched VAE Detector with config: {selected_key}.json")
        except Exception as e:
            self.log(f"Error launching tool: {e}")

    def _create_advanced_group(self):
        advanced_group = QtWidgets.QGroupBox("Miscellaneous")
        layout = QtWidgets.QFormLayout(advanced_group)
        self._add_widget_to_form(layout, "MEMORY_EFFICIENT_ATTENTION")
        separator = QtWidgets.QFrame(); separator.setFrameShape(QtWidgets.QFrame.Shape.HLine); separator.setStyleSheet("border: 1px solid #4a4668; margin: 10px 0;")
        layout.addRow(separator)
        spike_heading = QtWidgets.QLabel("<b>Gradient Spike Detection</b>"); spike_heading.setStyleSheet("color: #ab97e6; margin-top: 5px;")
        layout.addRow(spike_heading)
        self._add_widget_to_form(layout, "GRAD_SPIKE_THRESHOLD_HIGH")
        self._add_widget_to_form(layout, "GRAD_SPIKE_THRESHOLD_LOW")
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
        start_path = entry_widget.text() if entry_widget.text() else self.last_browsed_path
        if not os.path.exists(start_path): start_path = self.last_browsed_path
        if file_type == "folder": path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", start_path)
        elif file_type == "file_safetensors": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", start_path, "Safetensors Files (*.safetensors)")
        elif file_type == "file_pt": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select State", start_path, "PyTorch State Files (*.pt)")
        if path:
            entry_widget.setText(path.replace('\\', '/'))
            if os.path.isfile(path): self.last_browsed_path = os.path.dirname(path)
            else: self.last_browsed_path = path
    
    def _update_config_from_widget(self, key, widget):
        if isinstance(widget, QtWidgets.QLineEdit): self.current_config[key] = widget.text().strip()
        elif isinstance(widget, QtWidgets.QPlainTextEdit): self.current_config[key] = widget.toPlainText().strip()
        elif isinstance(widget, QtWidgets.QCheckBox): self.current_config[key] = widget.isChecked()
        elif isinstance(widget, QtWidgets.QComboBox): self.current_config[key] = widget.currentText()
        elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)): self.current_config[key] = widget.value()
        elif isinstance(widget, LRCurveWidget): self.current_config[key] = widget.get_points()
        elif isinstance(widget, TimestepHistogramWidget): self.current_config[key] = widget.get_allocation()
    
    def _apply_config_to_widgets(self):
        for widget in self.widgets.values(): widget.blockSignals(True)
        try:
            # ... (Standard Loading Logic) ...
            mode = self.current_config.get("TRAINING_MODE", "Standard (SDXL)")
            self.training_mode_combo.setCurrentText(mode)
            self._on_training_mode_changed(mode) 
            if hasattr(self, 'model_load_strategy_combo'):
                is_resuming = self.current_config.get("RESUME_TRAINING", False)
                self.model_load_strategy_combo.setCurrentIndex(1 if is_resuming else 0)
                self.toggle_resume_widgets(1 if is_resuming else 0)
            
            # Exclude special keys handled manually
            special_keys = [
                "OPTIMIZER_TYPE", "LR_CUSTOM_CURVE", "NOISE_TYPE", "LOSS_TYPE", "TIMESTEP_ALLOCATION", "TIMESTEP_MODE"
            ] + [k for k in self.widgets.keys() if k.startswith(("RAVEN_", "TITAN_", "VELORMS_", "SEMANTIC_LOSS_", "LOSS_HUBER_", "LOSS_ADAPTIVE_", "NOISE_OFFSET"))]
            
            for key, widget in self.widgets.items():
                if key in special_keys: continue
                value = self.current_config.get(key)
                if value is None: continue
                if isinstance(widget, QtWidgets.QLineEdit): widget.setText(str(value))
                elif isinstance(widget, QtWidgets.QPlainTextEdit): widget.setPlainText(str(value))
                elif isinstance(widget, QtWidgets.QCheckBox): widget.setChecked(bool(value))
                elif isinstance(widget, QtWidgets.QComboBox): widget.setCurrentText(str(value))
                elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)): widget.setValue(float(value) if isinstance(widget, QtWidgets.QDoubleSpinBox) else int(value))
            
            # --- OPTIMIZER LOADING ---
            optimizer_type = self.current_config.get("OPTIMIZER_TYPE", default_config.OPTIMIZER_TYPE).lower()
            index = self.widgets["OPTIMIZER_TYPE"].findData(optimizer_type)
            if index >= 0: self.widgets["OPTIMIZER_TYPE"].setCurrentIndex(index)
            else: self.widgets["OPTIMIZER_TYPE"].setCurrentIndex(0)
            
            # Load RAVEN Params
            r_params = {**default_config.RAVEN_PARAMS, **self.current_config.get("RAVEN_PARAMS", {})}
            self.widgets["RAVEN_betas"].setText(', '.join(map(str, r_params.get("betas", [0.9, 0.999]))))
            self.widgets["RAVEN_eps"].setText(str(r_params.get("eps", 1e-8)))
            self.widgets["RAVEN_weight_decay"].setValue(r_params.get("weight_decay", 0.01))
            self.widgets["RAVEN_debias_strength"].setValue(r_params.get("debias_strength", 1.0))
            self.widgets["RAVEN_use_grad_centralization"].setChecked(r_params.get("use_grad_centralization", False))
            self.widgets["RAVEN_gc_alpha"].setValue(r_params.get("gc_alpha", 1.0))
            self.widgets["RAVEN_gc_alpha"].setEnabled(r_params.get("use_grad_centralization", False))

            # Load TITAN Params
            t_defaults = getattr(default_config, "TITAN_PARAMS", default_config.RAVEN_PARAMS)
            t_params = {**t_defaults, **self.current_config.get("TITAN_PARAMS", {})}
            self.widgets["TITAN_betas"].setText(', '.join(map(str, t_params.get("betas", [0.9, 0.999]))))
            self.widgets["TITAN_eps"].setText(str(t_params.get("eps", 1e-8)))
            self.widgets["TITAN_weight_decay"].setValue(t_params.get("weight_decay", 0.01))
            self.widgets["TITAN_debias_strength"].setValue(t_params.get("debias_strength", 1.0))
            self.widgets["TITAN_use_grad_centralization"].setChecked(t_params.get("use_grad_centralization", False))
            self.widgets["TITAN_gc_alpha"].setValue(t_params.get("gc_alpha", 1.0))
            self.widgets["TITAN_gc_alpha"].setEnabled(t_params.get("use_grad_centralization", False))

            # Load VELORMS Params
            v_defaults = getattr(default_config, "VELORMS_PARAMS", {"momentum": 0.86, "leak": 0.16, "weight_decay": 0.01, "eps": 1e-8})
            v_params = {**v_defaults, **self.current_config.get("VELORMS_PARAMS", {})}
            self.widgets["VELORMS_momentum"].setValue(v_params.get("momentum", 0.86))
            self.widgets["VELORMS_leak"].setValue(v_params.get("leak", 0.16))
            self.widgets["VELORMS_weight_decay"].setValue(v_params.get("weight_decay", 0.01))
            self.widgets["VELORMS_eps"].setText(str(v_params.get("eps", 1e-8)))

            self._toggle_optimizer_widgets()
            
            # --- MANUAL TOGGLE UPDATES ---
            noise_type = self.current_config.get("NOISE_TYPE", "Default")
            self.widgets["NOISE_TYPE"].setCurrentText(noise_type)
            self.widgets["NOISE_OFFSET"].setValue(self.current_config.get("NOISE_OFFSET", 0.0))
            self._toggle_noise_widgets()
            
            loss_type = self.current_config.get("LOSS_TYPE", "MSE")
            self.widgets["LOSS_TYPE"].setCurrentText(loss_type)
            
            # Semantic Params
            self.widgets["SEMANTIC_CHAR_WEIGHT"].setValue(self.current_config.get("SEMANTIC_CHAR_WEIGHT", 0.5))
            self.widgets["SEMANTIC_DETAIL_WEIGHT"].setValue(self.current_config.get("SEMANTIC_DETAIL_WEIGHT", 0.8))
            
            # Huber Params
            self.widgets["LOSS_HUBER_BETA"].setValue(self.current_config.get("LOSS_HUBER_BETA", 0.5))
            self.widgets["LOSS_ADAPTIVE_DAMPING"].setValue(self.current_config.get("LOSS_ADAPTIVE_DAMPING", 0.1))
            self._toggle_loss_widgets()

            if "SHOULD_UPSCALE" in self.widgets and "MAX_AREA_TOLERANCE" in self.widgets:
                self.widgets["MAX_AREA_TOLERANCE"].setEnabled(bool(self.current_config.get("SHOULD_UPSCALE", False)))
            
            if "UNCONDITIONAL_DROPOUT" in self.widgets:
                is_dropout = self.widgets["UNCONDITIONAL_DROPOUT"].isChecked()
                if "UNCONDITIONAL_DROPOUT_CHANCE" in self.widgets:
                    self.widgets["UNCONDITIONAL_DROPOUT_CHANCE"].setEnabled(is_dropout)

            if hasattr(self, 'lr_curve_widget'): self._update_and_clamp_lr_graph()
            
            if hasattr(self, 'timestep_histogram'):
                allocation = self.current_config.get("TIMESTEP_ALLOCATION")
                if allocation:
                    if "bin_size" in allocation:
                        self.bin_size_combo.blockSignals(True)
                        self.bin_size_combo.setCurrentText(str(allocation["bin_size"]))
                        self.bin_size_combo.blockSignals(False)
                    self.timestep_histogram.set_allocation(allocation)
                
                # Load saved mode (default to Wave if missing)
                ts_mode = self.current_config.get("TIMESTEP_MODE", "Wave")
                
                # Temporarily disconnect the automatic distribution generation
                self.ts_mode_combo.currentIndexChanged.disconnect(self._update_timestep_distribution)
                
                # Set the mode (this will still switch the slider/button stacks because those connections remain active)
                self.ts_mode_combo.setCurrentText(ts_mode)
                
                # Re-connect the generation trigger
                self.ts_mode_combo.currentIndexChanged.connect(self._update_timestep_distribution)
                
                # Reset sliders to their default values for the loaded mode (so we don't accidentally keep old tweaks)
                self._reset_timestep_sliders()
            if ts_mode and hasattr(self, 'ts_mode_combo'):
                self.ts_mode_combo.setCurrentText(ts_mode)
                
            if hasattr(self, "dataset_manager"): 
                self.dataset_manager.load_datasets_from_config(self.current_config.get("INSTANCE_DATASETS", []))
            self._update_training_calculations()
        finally:
            for widget in self.widgets.values(): widget.blockSignals(False)

    def restore_defaults(self):
        index = self.config_dropdown.currentIndex()
        if index < 0: return
        selected_key = self.config_dropdown.itemData(index) if self.config_dropdown.itemData(index) else self.config_dropdown.itemText(index).replace(" ", "_").lower()
        reply = QtWidgets.QMessageBox.question(self, "Restore Defaults", f"This will overwrite '{selected_key}.json' with hardcoded defaults. Are you sure?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.presets[selected_key] = copy.deepcopy(self.default_config)
            self.save_config()
            self.load_selected_config(index)
            self.log(f"Restored '{selected_key}.json' to defaults.")
    
    def clear_console_log(self):
        self.log_textbox.clear()
        self.log("Console cleared.")

    def toggle_resume_widgets(self, index):
        if hasattr(self, 'path_stacked_widget'): self.path_stacked_widget.setCurrentIndex(index)

    def _update_training_calculations(self):
        if not all(k in self.widgets for k in ["MAX_TRAIN_STEPS", "GRADIENT_ACCUMULATION_STEPS", "BATCH_SIZE"]) or not hasattr(self, 'dataset_manager'): return
        try:
            max_train_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
            grad_accum_steps = int(self.widgets["GRADIENT_ACCUMULATION_STEPS"].text())
            batch_size = self.widgets["BATCH_SIZE"].value()
            total_images_with_repeats = self.dataset_manager.get_total_repeats()
            
            optimizer_steps = max_train_steps // grad_accum_steps if grad_accum_steps > 0 else 0
            steps_per_epoch = total_images_with_repeats // batch_size if total_images_with_repeats > 0 and batch_size > 0 else 0
            total_epochs = max_train_steps / steps_per_epoch if steps_per_epoch > 0 else float('inf')
            
            self.optimizer_steps_label.setText(f"{optimizer_steps:,}")
            self.epochs_label.setText("∞ (Not enough images for one batch)" if total_epochs == float('inf') else f"{total_epochs:.2f}")

            if hasattr(self, 'timestep_histogram'):
                self.timestep_histogram.set_total_steps(max_train_steps)

        except (ValueError, KeyError):
            self.optimizer_steps_label.setText("Invalid Input"); self.epochs_label.setText("Invalid Input")

    def _update_and_clamp_lr_graph(self):
        if not hasattr(self, 'lr_curve_widget'): return
        try: steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except (ValueError, KeyError): steps = 1
        try: min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except (ValueError, KeyError): min_lr = 0.0
        try: max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
        except (ValueError, KeyError): max_lr = 1e-6
        

        if "LR_CUSTOM_CURVE" in self.current_config:
            clamped_points = []
            for p in self.current_config["LR_CUSTOM_CURVE"]:
                clamped_y = max(min_lr, min(max_lr, p[1]))
                clamped_points.append([p[0], clamped_y])
            self.current_config["LR_CUSTOM_CURVE"] = clamped_points

        
        self.lr_curve_widget.set_bounds(steps, min_lr, max_lr)
        self.lr_curve_widget.set_points(self.current_config.get("LR_CUSTOM_CURVE", []))
        self._update_epoch_markers_on_graph()
    
    def _update_epoch_markers_on_graph(self):
        if not hasattr(self, 'lr_curve_widget') or not hasattr(self, 'dataset_manager'): return
        try:
            total_images = self.dataset_manager.get_total_repeats(); max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except (ValueError, KeyError):
            self.lr_curve_widget.set_epoch_data([]); return
        epoch_data = []
        if total_images > 0 and max_steps > 0:
            steps_per_epoch = total_images; current_epoch_step = steps_per_epoch
            while current_epoch_step < max_steps:
                epoch_data.append((current_epoch_step / max_steps, int(current_epoch_step)))
                current_epoch_step += steps_per_epoch
        self.lr_curve_widget.set_epoch_data(epoch_data)

    def _update_rf_settings_state(self, text=None):
        """Disables Shift Factor if mode is standard"""
        if "RF_SHIFT_FACTOR" not in self.widgets:
            return
        self.widgets["RF_SHIFT_FACTOR"].setEnabled(True)   

    def _update_lr_button_states(self, selected_index):
        if hasattr(self, 'remove_point_btn'):
            is_removable = selected_index > 0 and selected_index < len(self.lr_curve_widget.get_points()) - 1
            self.remove_point_btn.setEnabled(is_removable)
    
    def log(self, message): self.append_log(message.strip(), replace=False)
    
    def append_log(self, text, replace=False):
        scrollbar = self.log_textbox.verticalScrollBar(); scroll_at_bottom = (scrollbar.value() >= scrollbar.maximum() - 4)
        cursor = self.log_textbox.textCursor(); cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if replace: cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor); cursor.removeSelectedText(); cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.log_textbox.setTextCursor(cursor); self.log_textbox.insertPlainText(text.rstrip() + '\n')
        if scroll_at_bottom: scrollbar.setValue(scrollbar.maximum())

    def handle_process_output(self, text, is_progress):
        if text:
            self.append_log(text, replace=is_progress and self.last_line_is_progress)
            self.last_line_is_progress = is_progress
    
    def get_current_cache_folder_name(self):
        mode = self.training_mode_combo.currentText()
        if "Rectified Flow (SDXL)" in mode: return ".precomputed_embeddings_cache_rf_noobai"
        else: return ".precomputed_embeddings_cache"

    def start_training(self):
        self.save_config()
        self.log("\n" + "="*50 + "\nStarting training process...\n" + "="*50)
        selected_key = self.config_dropdown.itemData(self.config_dropdown.currentIndex()) if self.config_dropdown.itemData(self.config_dropdown.currentIndex()) else self.config_dropdown.itemText(self.config_dropdown.currentIndex()).replace(" ", "_").lower()
        config_path = os.path.join(self.config_dir, f"{selected_key}.json")
        if not os.path.exists(config_path):
            self.log(f"CRITICAL ERROR: Config file not found: {config_path}. Aborting."); return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.abspath(config_path)
        mode = self.training_mode_combo.currentText()
        script_name = "train.py"
        self.log("INFO: Training started. Executing train.py")
        train_py_path = os.path.abspath(script_name)
        if not os.path.exists(train_py_path):
             self.log(f"CRITICAL ERROR: Training script not found: {train_py_path}. Aborting.")
             return
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.live_metrics_widget.clear_data()
        env_dict = os.environ.copy(); python_dir = os.path.dirname(sys.executable)
        env_dict["PATH"] = f"{python_dir};{os.path.join(python_dir, 'Scripts')};{env_dict.get('PATH', '')}"
        env_dict["PYTHONPATH"] = f"{script_dir};{env_dict.get('PYTHONPATH', '')}"
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        self.process_runner = ProcessRunner(executable=sys.executable, args=["-u", train_py_path, "--config", config_path], working_dir=script_dir, env=env_dict, creation_flags=creation_flags)
        self.process_runner.logSignal.connect(self.log)
        self.process_runner.paramInfoSignal.connect(lambda info: self.param_info_label.setText(f"Trainable Parameters: {info}"))
        self.process_runner.progressSignal.connect(self.handle_process_output)
        self.process_runner.finishedSignal.connect(self.training_finished)
        self.process_runner.errorSignal.connect(self.log)
        self.process_runner.metricsSignal.connect(self.live_metrics_widget.parse_and_update)
        self.process_runner.cacheCreatedSignal.connect(self.dataset_manager.refresh_cache_buttons)
        if os.name == 'nt': prevent_sleep(True)
        self.process_runner.start()
        self.log(f"INFO: Starting {script_name} with config: {config_path}")
    
    def stop_training(self):
        if self.process_runner and self.process_runner.isRunning(): self.process_runner.stop()
        else: self.log("No active training process to stop.")
    
    def training_finished(self, exit_code=0):
        if self.process_runner: self.process_runner.quit(); self.process_runner.wait(); self.process_runner = None
        status = "successfully" if exit_code == 0 else f"with an error (Code: {exit_code})"
        self.log(f"\n" + "="*50 + f"\nTraining finished {status}.\n" + "="*50)
        self.param_info_label.setText("Parameters: (training complete)" if exit_code == 0 else "Parameters: (training failed or stopped)")
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        if hasattr(self, 'dataset_manager'): self.dataset_manager.refresh_cache_buttons()
        if os.name == 'nt': prevent_sleep(False)

class DatasetLoaderThread(QThread):
    finished = pyqtSignal(list, str) 
    def __init__(self, path):
        super().__init__()
        self.path = path
    def run(self):
        images_data = []
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        try:
            path_obj = Path(self.path)
            for file_path in path_obj.rglob("*"):
                if file_path.suffix.lower() in exts:
                    caption_path = file_path.with_suffix('.txt')
                    images_data.append({
                        "image_path": str(file_path),
                        "caption_path": str(caption_path) if caption_path.exists() else None,
                        "caption_loaded": False,
                        "caption": ""
                    })
        except Exception as e: print(f"Error scanning {self.path}: {e}")
        self.finished.emit(images_data, self.path)

class DatasetManagerWidget(QtWidgets.QWidget):
    datasetsChanged = QtCore.pyqtSignal()
    
    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.datasets = []
        self.dataset_widgets = []
        self.loader_threads = [] 
        self._init_ui()
        
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        top_bar = QtWidgets.QHBoxLayout()
        add_native_btn = QtWidgets.QPushButton("Add Folder (Standard)")
        add_native_btn.setToolTip("Use the standard Windows folder picker")
        add_native_btn.clicked.connect(self.add_dataset_folder_native)
        top_bar.addWidget(add_native_btn)
        add_custom_btn = QtWidgets.QPushButton("Add Folder (Smart)")
        add_custom_btn.setStyleSheet("background-color: #383552; color: #ab97e6; border: 1px solid #6a48d7;")
        add_custom_btn.setToolTip("Use the custom explorer to see image counts inside subfolders before selecting")
        add_custom_btn.clicked.connect(self.add_dataset_folder_custom)
        top_bar.addWidget(add_custom_btn)
        top_bar.addWidget(QtWidgets.QLabel("Sort By:"))
        self.sort_combo = NoScrollComboBox()
        self.sort_combo.addItems(["Default (Order Added)", "Name (A-Z)", "Name (Z-A)", "Image Count (High → Low)", "Image Count (Low → High)"])
        self.sort_combo.currentIndexChanged.connect(self.sort_datasets)
        top_bar.addWidget(self.sort_combo)
        self.loading_label = QtWidgets.QLabel("")
        self.loading_label.setStyleSheet("color: #ffdd57; font-weight: bold;")
        top_bar.addWidget(self.loading_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)
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

    def get_total_repeats(self): return sum(ds["image_count"] * ds["repeats"] for ds in self.datasets)
    def get_datasets_config(self): return [{"path": ds["path"], "repeats": ds["repeats"]} for ds in self.datasets]
    def _format_caption_for_display(self, text, max_chars=50):
        """Force wrap long strings by inserting manual line breaks"""
        import textwrap
        # Break long words, preserve existing newlines
        wrapped = textwrap.fill(text, width=max_chars, 
                            break_long_words=True,
                            replace_whitespace=False,
                            drop_whitespace=False)
        return wrapped
    def _start_loading_path(self, path, repeats=1):
        self.loading_label.setText(f"Scanning: {Path(path).name}...")
        loader = DatasetLoaderThread(path)
        loader.repeats = repeats 
        loader.finished.connect(self._on_loader_finished)
        self.loader_threads.append(loader)
        loader.start()

    def _on_loader_finished(self, images_data, path):
        sender = self.sender()
        repeats = getattr(sender, 'repeats', 1)
        if sender in self.loader_threads: self.loader_threads.remove(sender)
        if not self.loader_threads: self.loading_label.setText("")
        if not images_data:
            if self.isVisible(): QtWidgets.QMessageBox.warning(self, "No Images", f"No valid images found in {path}.")
            return
        self.datasets.append({"path": path, "images_data": images_data, "image_count": len(images_data), "current_preview_idx": 0, "repeats": repeats})
        self.sort_datasets()
        self.update_dataset_totals()

    def sort_datasets(self):
        criteria = self.sort_combo.currentText()
        if criteria == "Name (A-Z)": self.datasets.sort(key=lambda x: Path(x['path']).name.lower())
        elif criteria == "Name (Z-A)": self.datasets.sort(key=lambda x: Path(x['path']).name.lower(), reverse=True)
        elif criteria == "Image Count (High → Low)": self.datasets.sort(key=lambda x: x['image_count'], reverse=True)
        elif criteria == "Image Count (Low → High)": self.datasets.sort(key=lambda x: x['image_count'])
        self.repopulate_dataset_grid()

    def load_datasets_from_config(self, datasets_config):
        self.datasets = []
        for t in self.loader_threads: t.quit()
        self.loader_threads = []
        self.repopulate_dataset_grid()
        for d in datasets_config:
            path = d.get("path")
            if path and os.path.exists(path): self._start_loading_path(path, d.get("repeats", 1))

    def add_dataset_folder_native(self):
        start_path = self.parent_gui.last_browsed_path
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder", start_path)
        if path:
            self.parent_gui.last_browsed_path = path
            self._start_loading_path(path, 1)

    def add_dataset_folder_custom(self):
        start_path = self.parent_gui.last_browsed_path
        dialog = CustomFolderDialog(start_path, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.selected_path:
            path = dialog.selected_path
            self.parent_gui.last_browsed_path = path
            self._start_loading_path(path, 1)

    def _cycle_preview(self, idx, direction):
        ds = self.datasets[idx]
        ds["current_preview_idx"] = (ds["current_preview_idx"] + direction) % len(ds["images_data"])
        self._update_preview_for_card(idx)
        
    def _update_preview_for_card(self, idx):
        if idx >= len(self.dataset_widgets): 
            return
        ds = self.datasets[idx]
        widgets = self.dataset_widgets[idx]
        current_data = ds["images_data"][ds["current_preview_idx"]]
        
        if not current_data["caption_loaded"]:
            if current_data["caption_path"]:
                try:
                    with open(current_data["caption_path"], 'r', encoding='utf-8') as f: 
                        current_data["caption"] = f.read().strip()
                except Exception: 
                    current_data["caption"] = "[Error reading caption]"
            else: 
                current_data["caption"] = "[No caption file]"
            current_data["caption_loaded"] = True
        
        # Update image preview
        pixmap = QtGui.QPixmap(current_data["image_path"])
        if not pixmap.isNull(): 
            pixmap = pixmap.scaled(183, 183, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        widgets["preview_label"].setPixmap(pixmap)
        
        # Update caption - NO TRUNCATION, full text with scrolling
        widgets["caption_text"].setPlainText(current_data["caption"])
        widgets["counter_label"].setText(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")
        
    def repopulate_dataset_grid(self):
        while self.dataset_grid.count():
            item = self.dataset_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.dataset_widgets = []
        dataset_count = len(self.datasets)
        COLUMNS = 1 if dataset_count == 1 else (2 if dataset_count == 2 else 3)
        for idx, ds in enumerate(self.datasets):
            row, col = idx // COLUMNS, idx % COLUMNS
            card = QtWidgets.QGroupBox()
            card.setStyleSheet("QGroupBox { border: 2px solid #4a4668; border-radius: 8px; margin-top: 5px; padding: 12px; background-color: #383552; }")
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
            image_container.addWidget(preview_label)
            image_container.addStretch()
            preview_section.addLayout(image_container)
            counter_nav_layout = QtWidgets.QHBoxLayout()
            counter_nav_layout.setSpacing(8)
            left_arrow = QtWidgets.QPushButton("◄")
            left_arrow.setFixedHeight(22); left_arrow.setMinimumWidth(35)
            left_arrow.clicked.connect(lambda _, i=idx: self._cycle_preview(i, -1))
            left_arrow.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 0px; }")
            counter_nav_layout.addWidget(left_arrow)
            counter_label = QtWidgets.QLabel(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")
            counter_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            counter_label.setStyleSheet("color: #ab97e6; font-size: 12px; font-weight: bold;")
            counter_nav_layout.addWidget(counter_label, 1)
            right_arrow = QtWidgets.QPushButton("►")
            right_arrow.setFixedHeight(22); right_arrow.setMinimumWidth(35)
            right_arrow.clicked.connect(lambda _, i=idx: self._cycle_preview(i, 1))
            right_arrow.setStyleSheet("QPushButton { font-size: 16px; font-weight: bold; padding: 0px; }")
            counter_nav_layout.addWidget(right_arrow)
            preview_section.addLayout(counter_nav_layout)
            top_section.addLayout(preview_section)


            caption_container = QtWidgets.QWidget()
            caption_container.setStyleSheet("QWidget { background-color: #2c2a3e; border: 1px solid #4a4668; border-radius: 4px; }")
            caption_container.setMinimumWidth(150)
            caption_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

            caption_layout = QtWidgets.QVBoxLayout(caption_container)
            caption_layout.setContentsMargins(8, 8, 8, 8)

            caption_title = QtWidgets.QLabel("<b>Caption Preview:</b>")
            caption_title.setStyleSheet("color: #ab97e6; font-size: 11px;")
            caption_layout.addWidget(caption_title)

            # QTextEdit with full height support and scrolling
            caption_text = QtWidgets.QTextEdit("Loading...")
            caption_text.setReadOnly(True)
            caption_text.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            caption_text.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
            caption_text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            caption_text.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            caption_text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            caption_text.setMinimumHeight(183)  # Minimum to match image, but can grow
            caption_text.setStyleSheet("""
                QTextEdit { 
                    background-color: #2c2a3e; 
                    color: #e0e0e0; 
                    font-size: 12px; 
                    border: none; 
                    padding: 0px; 
                }
            """)
            caption_layout.addWidget(caption_text, 1)

            top_section.addWidget(caption_container, 1)


            card_layout.addLayout(top_section)
            separator = QtWidgets.QFrame()
            separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            separator.setStyleSheet("border: 1px solid #4a4668;")
            card_layout.addWidget(separator)
            path_label = QtWidgets.QLabel()
            path_short = Path(ds['path']).name
            if len(path_short) > 30: path_short = path_short[:27] + "..."
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
            repeats_spin.setMinimum(1); repeats_spin.setMaximum(10000); repeats_spin.setValue(ds["repeats"])
            repeats_spin.setStyleSheet("QSpinBox::up-button, QSpinBox::down-button { width: 20px; }")
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
            if not cache_exists: clear_btn.setToolTip("No cache found")
            btn_layout.addWidget(clear_btn)
            card_layout.addLayout(btn_layout)
            self.dataset_grid.addWidget(card, row, col)
            self.dataset_widgets.append({
                "preview_label": preview_label, 
                "caption_text": caption_text,  # Changed from caption_label
                "counter_label": counter_label, 
                "repeats_total_label": repeats_total_label, 
                "clear_btn": clear_btn
            })
            self._update_preview_for_card(idx)
        if dataset_count % COLUMNS != 0:
            for empty_col in range((dataset_count % COLUMNS), COLUMNS): self.dataset_grid.setColumnStretch(empty_col, 1)
                
    def _cache_exists(self, path):
        cache_name = self.parent_gui.get_current_cache_folder_name()
        cache_dir = Path(path) / cache_name
        return cache_dir.exists() and cache_dir.is_dir()

    def update_repeats(self, idx, val):
        self.datasets[idx]["repeats"] = val
        if idx < len(self.dataset_widgets):
            ds = self.datasets[idx]
            total = ds['image_count'] * ds['repeats']
            self.dataset_widgets[idx]["repeats_total_label"].setText(f"<b>Total (with repeats):</b> {total}")
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
        cache_name = self.parent_gui.get_current_cache_folder_name()
        reply = QtWidgets.QMessageBox.question(self, "Confirm", f"Delete cached latents in '{cache_name}' for this dataset?", QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes: self.clear_cached_latents(path)
        
    def clear_cached_latents(self, path):
        root = Path(path)
        target_cache_name = self.parent_gui.get_current_cache_folder_name()
        deleted = False
        for cache_dir in list(root.rglob(target_cache_name)):
            if cache_dir.is_dir():
                try: 
                    shutil.rmtree(cache_dir)
                    self.parent_gui.log(f"Deleted cache directory: {cache_dir}")
                    deleted = True
                except Exception as e: self.parent_gui.log(f"Error deleting {cache_dir}: {e}")
        if deleted: self.refresh_cache_buttons()
        if not deleted: self.parent_gui.log(f"No cache directories found matching '{target_cache_name}'.")
        
    def refresh_cache_buttons(self):
        for idx, ds in enumerate(self.datasets):
            if idx < len(self.dataset_widgets):
                cache_exists = self._cache_exists(ds["path"])
                clear_btn = self.dataset_widgets[idx]["clear_btn"]
                clear_btn.setEnabled(cache_exists)
                if not cache_exists: clear_btn.setToolTip("No cache found")
                else: clear_btn.setToolTip("")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = TrainingGUI()
    main_win.show()
    sys.exit(app.exec())
