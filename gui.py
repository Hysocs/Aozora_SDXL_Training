import json
import os
import re
import math
import copy
import sys
import shutil
import ctypes
from pathlib import Path
from collections import deque
from datetime import datetime

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QFileIconProvider
from PyQt6.QtCore import QFileInfo
import subprocess

try:
    import config as default_config
except ImportError:
    class default_config:
        RAVEN_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "debias_strength": 1.0, "use_grad_centralization": False, "gc_alpha": 1.0}
        TITAN_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "debias_strength": 1.0, "use_grad_centralization": False, "gc_alpha": 1.0}
        VELORMS_PARAMS = {"momentum": 0.86, "leak": 0.16, "weight_decay": 0.01, "eps": 1e-8}
        OPTIMIZER_TYPE = "Raven"
        TIMESTEP_ALLOCATION = {"bin_size": 100, "counts": []}


# ──────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (matches Semantic Loss Visualizer)
# ──────────────────────────────────────────────────────────────────
DARK_BG    = "#0d0f14"
PANEL_BG   = "#141720"
GRAPH_BG   = "#0e1016"   # Just subtly darker than the panels, creating a gentle sunken effect without being a void
BORDER     = "#252a38"
ACCENT     = "#7c6af7"
ACCENT2    = "#3ecfcf"
TEXT_PRI   = "#e8eaf0"
TEXT_SEC   = "#7a7f9a"
DANGER     = "#f74b6a"
SUCCESS    = "#3ef78e"
WARN       = "#f7c948"

# "Ghosted" disabled states (blended deeply into DARK_BG)
TEXT_MUTED   = "#383b4075"  # Extremely faint grey for disabled text
BORDER_MUTED = "#191c2675"  # Barely visible border for disabled elements

STYLESHEET = f"""
QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRI};
    font-family: 'Consolas', 'Segoe UI', monospace;
    font-size: 10pt; /* Using pt instead of px fixes QFont warnings */
}}

/* ── GROUP BOX ── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 16px;
    padding: 10px 8px 8px 8px;
    color: {TEXT_SEC};
    font-size: 8pt; /* Using pt instead of px */
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {ACCENT};
    font-weight: bold;
}}
QGroupBox:disabled {{ border: 1px solid {BORDER_MUTED}; background: transparent; }}
QGroupBox:disabled::title {{ color: {TEXT_MUTED}; }}

/* ── BUTTONS ── */
QPushButton {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRI};
    padding: 6px 14px;
    min-height: 28px;
    max-height: 36px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: white; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; background: transparent; border: 1px solid {BORDER_MUTED}; }}

#StartButton {{ background: {ACCENT}; color: white; font-weight: bold; border-color: {ACCENT}; }}
#StartButton:hover {{ background: #9580ff; }}
#StopButton {{ background: {DANGER}; color: white; font-weight: bold; border-color: {DANGER}; }}
#StopButton:hover {{ background: #ff6b85; }}

/* ── TEXT INPUTS & COMBO BOXES ── */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT_PRI};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:on {{ border-color: {ACCENT}; }}
QPlainTextEdit, QTextEdit {{ font-family: 'Consolas', monospace; padding: 4px; }}
QComboBox {{ min-height: 28px; max-height: 36px; }}
QComboBox::drop-down {{ border-left: 1px solid {BORDER}; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_BG};
    border: 1px solid {ACCENT};
    selection-background-color: {ACCENT};
    selection-color: white;
}}

/* Ghosted Disabled Inputs */
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QComboBox:disabled {{
    color: {TEXT_MUTED};
    background: transparent;
    border: 1px solid {BORDER_MUTED};
}}
QComboBox::drop-down:disabled {{ border-left: 1px solid {BORDER_MUTED}; }}

/* ── SPIN BOXES ── */
QSpinBox, QDoubleSpinBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    padding-right: 28px;
    color: {TEXT_PRI};
    min-height: 24px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 26px;
    border-left: 1px solid {BORDER};
    background: {PANEL_BG};
}}
QSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 4px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {BORDER}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 4px solid {TEXT_PRI}; width: 0; height: 0;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 4px solid {TEXT_PRI}; width: 0; height: 0;
}}

/* Ghosted Disabled Spin Boxes */
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {TEXT_MUTED};
    background: transparent;
    border: 1px solid {BORDER_MUTED};
}}
QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled,
QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {{
    background: transparent;
    border-left: 1px solid {BORDER_MUTED};
}}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ border-bottom-color: {TEXT_MUTED}; }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ border-top-color: {TEXT_MUTED}; }}

/* ── CHECKBOXES ── */
QCheckBox {{ spacing: 8px; color: {TEXT_PRI}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

/* Ghosted Disabled Checkboxes */
QCheckBox:disabled {{ color: {TEXT_MUTED}; }}
QCheckBox::indicator:disabled {{
    background: transparent;
    border: 1px solid {BORDER_MUTED};
}}
QCheckBox::indicator:checked:disabled {{
    background: {TEXT_MUTED};
    border: 1px solid {BORDER_MUTED};
}}

/* ── SLIDERS & MISC ── */
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; margin: 2px 0; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; border: 1px solid {ACCENT};
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: white; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

QSlider::groove:horizontal:disabled {{ background: {BORDER_MUTED}; }}
QSlider::handle:horizontal:disabled {{ background: {TEXT_MUTED}; border: none; }}
QSlider::sub-page:horizontal:disabled {{ background: transparent; }}

QLabel:disabled {{ color: {TEXT_MUTED}; }}
#TitleLabel {{
    color: {ACCENT};
    font-size: 20pt; /* Using pt instead of px */
    font-weight: bold;
    padding: 12px;
    border-bottom: 1px solid {BORDER};
    font-family: 'Consolas', monospace;
}}

/* ── TABS, SCROLLBARS, TABLES ── */
QTabWidget::pane {{ border: 1px solid {BORDER}; border-top: none; }}
QTabBar::tab {{
    background: {DARK_BG};
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 8px 18px;
    color: {TEXT_SEC};
    font-weight: bold;
    min-height: 36px;
}}
QTabBar::tab:selected {{ background: {PANEL_BG}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:!selected:hover {{ background: {BORDER}; color: {TEXT_PRI}; }}
QScrollArea {{ border: none; }}
QHeaderView::section {{
    background: {PANEL_BG};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QTableWidget {{ gridline-color: {BORDER}; background: {PANEL_BG}; }}
QTableWidget::item:selected {{ background: {ACCENT}; color: white; }}

QScrollBar:vertical {{ background: {DARK_BG}; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSplitter::handle {{ background: {BORDER}; }}
"""


# ──────────────────────────────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────────────────────────────
def prevent_sleep(enable=True):
    try:
        if os.name == 'nt':
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            kernel32 = ctypes.windll.kernel32
            state = (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if enable else ES_CONTINUOUS
            kernel32.SetThreadExecutionState(state)
    except Exception:
        pass


# ── No-scroll wrappers ────────────────────────────────────────────
class NoScrollSpinBox(QtWidgets.QSpinBox):
    def wheelEvent(self, e): e.ignore()

class NoScrollDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, e): e.ignore()

class NoScrollComboBox(QtWidgets.QComboBox):
    def wheelEvent(self, e): e.ignore()


# ── Table items ───────────────────────────────────────────────────
class NumericTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other):
        try: return float(self.text()) < float(other.text())
        except ValueError: return super().__lt__(other)

class DateTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __init__(self, display_text, timestamp):
        super().__init__(display_text)
        self.timestamp = timestamp
    def __lt__(self, other):
        return self.timestamp < other.timestamp


# ──────────────────────────────────────────────────────────────────
#  WIDGET FACTORY HELPERS
# ──────────────────────────────────────────────────────────────────
def make_spin(lo, hi, val=None, *, scroll=False):
    w = QtWidgets.QSpinBox() if scroll else NoScrollSpinBox()
    w.setRange(lo, hi)
    if val is not None: w.setValue(val)
    return w

def make_dspin(lo, hi, val=None, step=0.1, decimals=2, *, scroll=False):
    w = QtWidgets.QDoubleSpinBox() if scroll else NoScrollDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    if val is not None: w.setValue(val)
    return w

def make_combo(items, *, scroll=False):
    w = QtWidgets.QComboBox() if scroll else NoScrollComboBox()
    w.addItems(items)
    return w

def make_btn(text, callback=None, style=None):
    b = QtWidgets.QPushButton(text)
    if callback: b.clicked.connect(callback)
    if style: b.setStyleSheet(style)
    return b

def make_label(text, color=None, bold=False, size=None):
    lbl = QtWidgets.QLabel(text)
    parts = []
    if color: parts.append(f"color: {color};")
    if bold: parts.append("font-weight: bold;")
    if size: parts.append(f"font-size: {size}pt;")
    if parts: lbl.setStyleSheet(" ".join(parts))
    return lbl

def make_separator(horizontal=True):
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.HLine if horizontal else QtWidgets.QFrame.Shape.VLine)
    f.setStyleSheet(f"border: 1px solid {BORDER}; margin: 4px 0;")
    return f

def group_box(title, layout_type=QtWidgets.QVBoxLayout):
    gb = QtWidgets.QGroupBox(title)
    lay = layout_type(gb)
    return gb, lay

def form_row(layout, label_text, widget, tooltip=None):
    lbl = QtWidgets.QLabel(label_text)
    if tooltip:
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    layout.addRow(lbl, widget)

def path_row(layout, label_text, line_edit, browse_callback, tooltip=None):
    """Stacked label → lineedit → browse button."""
    container = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout(container)
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(2)
    lbl = make_label(label_text, color=ACCENT, bold=True)
    if tooltip: lbl.setToolTip(tooltip)
    vbox.addWidget(lbl)
    if tooltip: line_edit.setToolTip(tooltip)
    vbox.addWidget(line_edit)
    btn = make_btn("Browse...")
    btn.clicked.connect(browse_callback)
    vbox.addWidget(btn)
    layout.addWidget(container)

def hbox(*widgets, spacing=8):
    container = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        if w is None: lay.addStretch()
        else: lay.addWidget(w)
    return container, lay


# ──────────────────────────────────────────────────────────────────
#  CUSTOM FOLDER DIALOG
# ──────────────────────────────────────────────────────────────────
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
        self.current_path = (str(Path(start_path).parent) if start_path and os.path.isfile(start_path)
                             else (start_path if start_path and os.path.exists(start_path) else os.getcwd()))
        self._build_ui()
        self.load_directory(self.current_path)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Nav bar
        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(5)
        def nav_btn(icon, tip, cb):
            b = QtWidgets.QPushButton()
            b.setIcon(self.style().standardIcon(icon))
            b.setFixedWidth(35)
            b.setToolTip(tip)
            b.clicked.connect(cb)
            return b

        self.btn_back    = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack, "Back", self.go_back)
        self.btn_fwd     = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward, "Forward", self.go_forward)
        self.btn_up      = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp, "Up", self.go_up)
        self.btn_refresh = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload, "Refresh",
                                   lambda: self.load_directory(self.current_path))
        self.btn_back.setEnabled(False)
        self.btn_fwd.setEnabled(False)

        self.path_edit = QtWidgets.QLineEdit(self.current_path)
        self.path_edit.returnPressed.connect(self._on_path_entered)

        for w in [self.btn_back, self.btn_fwd, self.btn_up, self.path_edit, self.btn_refresh]:
            nav.addWidget(w) if w is not self.path_edit else nav.addWidget(w, 1)
        layout.addLayout(nav)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        # Sidebar
        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setIconSize(QtCore.QSize(24, 24))
        self.sidebar.setStyleSheet(f"""
            QListWidget {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px; outline: none; }}
            QListWidget::item {{ padding: 6px; color: {TEXT_PRI}; margin: 2px; }}
            QListWidget::item:hover {{ background: {BORDER}; border-radius: 4px; }}
            QListWidget::item:selected {{ background: {ACCENT}; color: white; border-radius: 4px; }}
        """)
        self.sidebar.itemClicked.connect(lambda item: item.data(QtCore.Qt.ItemDataRole.UserRole) and self.load_directory(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        self._populate_sidebar()
        splitter.addWidget(self.sidebar)

        # File table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Images", "Date Modified", "HiddenPath"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnHidden(3, True)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QtCore.QSize(20, 20))
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda r, _: self.load_directory(self.table.item(r, 3).text()))
        self.table.setStyleSheet(f"""
            QTableWidget {{ background: {PANEL_BG}; alternate-background-color: {DARK_BG}; border: 1px solid {BORDER}; border-radius: 4px; }}
            QTableWidget::item {{ padding: 4px; }}
            QHeaderView::section {{ background: {BORDER}; padding: 6px; border: none; border-right: 1px solid {BORDER}; font-weight: bold; }}
        """)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # Bottom bar
        bottom = QtWidgets.QHBoxLayout()
        self.status_label = make_label("", color=TEXT_SEC)
        cancel_btn = make_btn("Cancel", self.reject)
        select_btn = make_btn("Select Current Folder", self.select_current)
        select_btn.setStyleSheet(f"background: {ACCENT}; color: white; padding: 6px 15px;")
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(cancel_btn)
        bottom.addWidget(select_btn)
        layout.addLayout(bottom)

    def _populate_sidebar(self):
        paths = QtCore.QStandardPaths
        for text, loc in [("Desktop", paths.StandardLocation.DesktopLocation),
                          ("Documents", paths.StandardLocation.DocumentsLocation),
                          ("Pictures", paths.StandardLocation.PicturesLocation),
                          ("Downloads", paths.StandardLocation.DownloadLocation)]:
            p = paths.writableLocation(loc)
            if os.path.exists(p):
                item = QtWidgets.QListWidgetItem(self.icon_provider.icon(QFileInfo(p)), text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
                self.sidebar.addItem(item)
        sep = QtWidgets.QListWidgetItem("───── Drives ─────")
        sep.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        sep.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sidebar.addItem(sep)
        for vol in QtCore.QStorageInfo.mountedVolumes():
            if vol.isValid() and vol.isReady():
                name = vol.name() or vol.rootPath()
                item = QtWidgets.QListWidgetItem(self.icon_provider.icon(QFileInfo(vol.rootPath())), name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, vol.rootPath())
                self.sidebar.addItem(item)

    def _on_path_entered(self):
        p = self.path_edit.text()
        if os.path.isdir(p): self.load_directory(p)
        else: QtWidgets.QMessageBox.warning(self, "Invalid Path", "Path does not exist or is not a directory.")

    def load_directory(self, path):
        if not self.is_navigating_history:
            if self.history_idx == -1 or self.history[self.history_idx] != path:
                self.history = self.history[:self.history_idx + 1]
                self.history.append(path)
                self.history_idx += 1
        self.btn_back.setEnabled(self.history_idx > 0)
        self.btn_fwd.setEnabled(self.history_idx < len(self.history) - 1)
        self.is_navigating_history = False
        self.current_path = path
        self.path_edit.setText(path)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.status_label.setText("Scanning...")
        QtWidgets.QApplication.processEvents()

        img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        try:
            entries = [e for e in os.scandir(path) if e.is_dir()]
            self.table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                info = QFileInfo(entry.path)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(self.icon_provider.icon(info), entry.name))
                try:
                    count = sum(1 for f in os.scandir(entry.path)
                                if f.is_file() and os.path.splitext(f.name)[1].lower() in img_exts)
                    ci = NumericTableWidgetItem(str(count))
                    ci.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    ci.setForeground(QtGui.QColor(ACCENT if count > 0 else TEXT_SEC))
                    if count > 0: ci.setFont(QtGui.QFont("Consolas", 9, QtGui.QFont.Weight.Bold))
                    self.table.setItem(row, 1, ci)
                except PermissionError:
                    self.table.setItem(row, 1, NumericTableWidgetItem("-1"))
                try:
                    ts = entry.stat().st_mtime
                    di = DateTableWidgetItem(datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"), ts)
                    di.setForeground(QtGui.QColor(TEXT_SEC))
                    self.table.setItem(row, 2, di)
                except Exception:
                    self.table.setItem(row, 2, DateTableWidgetItem("N/A", 0))
                self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(entry.path))
            self.status_label.setText(f"{len(entries)} folders found.")
        except PermissionError:
            QtWidgets.QMessageBox.warning(self, "Access Denied", f"Cannot access {path}")
            self.go_back()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
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

    def select_current(self):
        rows = self.table.selectionModel().selectedRows()
        self.selected_path = (self.table.item(rows[0].row(), 3).text() if rows else self.current_path)
        self.accept()


# ──────────────────────────────────────────────────────────────────
#  GRAPH PANEL
# ──────────────────────────────────────────────────────────────────
class GraphPanel(QtWidgets.QWidget):
    def __init__(self, title, y_label, parent=None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.smoothing_window_size = 15
        self.lines = []
        self.padding = {'top': 35, 'bottom': 40, 'left': 70, 'right': 20}
        self.bg_color = QtGui.QColor(PANEL_BG)
        self.graph_bg_color = QtGui.QColor(GRAPH_BG)
        self.grid_color = QtGui.QColor(BORDER)
        self.text_color = QtGui.QColor(TEXT_PRI)
        self.title_color = QtGui.QColor(ACCENT)
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
            'linewidth': linewidth,
        })
        return len(self.lines) - 1

    def append_data(self, line_index, x, y):
        if 0 <= line_index < len(self.lines):
            line = self.lines[line_index]
            line['data'].append((x, y))
            line['smoothing_window'].append(y)
            line['smoothed_data'].append((x, sum(line['smoothing_window']) / len(line['smoothing_window'])))
            self._update_bounds()

    def clear_all_data(self):
        for line in self.lines:
            line['data'].clear(); line['smoothed_data'].clear(); line['smoothing_window'].clear()
        self.pan_offset = 0
        self._update_bounds()
        self.update()

    def _get_visible_slice(self, data_deque):
        if not data_deque: return []
        total = len(data_deque)
        visible = max(2, int(total / self.zoom_level))
        end = min(self.pan_offset + visible, total)
        return list(data_deque)[self.pan_offset:end]

    def _update_bounds(self):
        all_y, visible_data = [], []
        for line in self.lines:
            vis = self._get_visible_slice(line['data'])
            if vis:
                visible_data = vis
                all_y.extend(y for _, y in vis)
        if visible_data:
            self.x_min = visible_data[0][0]
            self.x_max = visible_data[-1][0]
            if all_y:
                yr = max(all_y) - min(all_y) or 1
                self.y_min = min(all_y) - yr * 0.05
                self.y_max = max(all_y) + yr * 0.05
            else:
                self.y_min, self.y_max = 0, 1
        else:
            self.x_min, self.x_max = 0, 100
            self.y_min, self.y_max = 0, 1

    def _to_screen(self, x, y):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        xr = self.x_max - self.x_min or 1
        yr = self.y_max - self.y_min or 1
        sx = self.padding['left'] + ((x - self.x_min) / xr) * gw
        sy = self.padding['top'] + gh - ((y - self.y_min) / yr) * gh
        return QtCore.QPointF(sx, sy)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        gr = QtCore.QRect(self.padding['left'], self.padding['top'],
                          self.width() - self.padding['left'] - self.padding['right'],
                          self.height() - self.padding['top'] - self.padding['bottom'])
        painter.fillRect(gr, self.graph_bg_color)
        self._draw_grid(painter, gr)
        self._draw_lines(painter, gr)
        self._draw_title(painter)
        self._draw_legend(painter)

    def _draw_grid(self, painter, rect):
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        for i in range(5):
            y = rect.top() + (i / 4) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            y_val = self.y_max - (i / 4) * (self.y_max - self.y_min)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(5, int(y - 10), self.padding['left'] - 10, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             self._fmt(y_val))
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        for i in range(6):
            x = rect.left() + (i / 5) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            x_val = self.x_min + (i / 5) * (self.x_max - self.x_min)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(int(x - 30), rect.bottom() + 5, 60, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(x_val)))
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        painter.setPen(self.text_color)
        f = painter.font(); f.setPixelSize(12); painter.setFont(f)
        painter.save()
        painter.translate(15, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-50, -10, 100, 20), QtCore.Qt.AlignmentFlag.AlignCenter, self.y_label)
        painter.restore()
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Step")

    def _draw_lines(self, painter, rect):
        for line in self.lines:
            raw = self._get_visible_slice(line['data'])
            smo = self._get_visible_slice(line['smoothed_data'])
            if len(raw) < 2 or len(raw) != len(smo): continue
            pts = [self._to_screen(raw[i][0], raw[i][1] * (1 - self.smoothing_level) + smo[i][1] * self.smoothing_level)
                   for i in range(len(raw))]
            if self.fill_enabled:
                poly = QtGui.QPolygonF(pts)
                poly.append(QtCore.QPointF(pts[-1].x(), rect.bottom()))
                poly.append(QtCore.QPointF(pts[0].x(), rect.bottom()))
                fc = QtGui.QColor(line['color']); fc.setAlpha(40)
                painter.setBrush(fc); painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(poly)
            painter.setPen(QtGui.QPen(line['color'], line['linewidth']))
            painter.drawPolyline(QtGui.QPolygonF(pts))

    def _draw_title(self, painter):
        painter.setPen(self.title_color)
        f = painter.font(); f.setPixelSize(15); f.setBold(True); painter.setFont(f)
        painter.drawText(QtCore.QRect(0, 5, self.width(), 25), QtCore.Qt.AlignmentFlag.AlignCenter, self.title)

    def _draw_legend(self, painter):
        lx = self.width() - self.padding['right'] - 120
        ly = self.padding['top'] + 10
        f = painter.font(); f.setPixelSize(11); painter.setFont(f)
        for line in self.lines:
            if not line['data']: continue
            painter.setPen(QtGui.QPen(line['color'], line['linewidth']))
            painter.drawLine(lx, ly + 5, lx + 20, ly + 5)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(lx + 25, ly, 80, 15),
                             QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, line['label'])
            ly += 20

    def _fmt(self, v):
        if abs(v) < 0.01 or abs(v) > 10000: return f"{v:.1e}"
        if abs(v) < 1: return f"{v:.4f}"
        return f"{v:.2f}"

    def set_smoothing(self, v): self.smoothing_level = v / 100.0; self.update()
    def set_fill(self, e): self.fill_enabled = e; self.update()
    def set_zoom(self, v): self.zoom_level = v / 100.0; self._update_bounds(); self.update()
    def set_pan(self, v): self.pan_offset = v; self._update_bounds(); self.update()

    def get_pan_range(self):
        if not self.lines or not self.lines[0]['data']: return 0, 0, 1
        total = len(self.lines[0]['data'])
        visible = max(2, int(total / self.zoom_level))
        return 0, max(0, total - visible), 1


# ──────────────────────────────────────────────────────────────────
#  LIVE METRICS WIDGET
# ──────────────────────────────────────────────────────────────────
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

    def _make_graph_container(self, name, title, y_label):
        gb, lay = group_box(title)
        gb.setStyleSheet(f"QGroupBox {{ margin-top: 10px; }} QGroupBox::title {{ subcontrol-position: top center; }}")
        graph = GraphPanel(title, y_label)
        self.graphs[name] = {'widget': graph, 'lines': {}}
        lay.addWidget(graph, 1)

        ctrl = QtWidgets.QHBoxLayout()
        fill_chk = QtWidgets.QCheckBox("Fill")
        fill_chk.setChecked(True)
        fill_chk.stateChanged.connect(lambda s, g=graph: g.set_fill(s == QtCore.Qt.CheckState.Checked.value))
        ctrl.addWidget(fill_chk)
        ctrl.addWidget(make_label("Smooth:"))
        smooth_sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        smooth_sl.setRange(0, 100)
        smooth_sl.valueChanged.connect(graph.set_smoothing)
        ctrl.addWidget(smooth_sl, 1)
        ctrl.addWidget(make_label("Zoom:"))
        zoom_sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        zoom_sl.setRange(100, 1000)
        zoom_sl.setValue(100)
        ctrl.addWidget(zoom_sl, 1)
        lay.addLayout(ctrl)

        pan_sb = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Horizontal)
        pan_sb.setVisible(False)
        pan_sb.valueChanged.connect(graph.set_pan)
        lay.addWidget(pan_sb)
        zoom_sl.valueChanged.connect(lambda v, g=graph, sb=pan_sb: self._sync_pan(g, sb, v))

        self.graphs[name]['scrollbar'] = pan_sb
        self.graphs[name]['zoom_slider'] = zoom_sl
        return gb

    def _sync_pan(self, graph, scrollbar, zoom_value):
        graph.set_zoom(zoom_value)
        mn, mx, step = graph.get_pan_range()
        scrollbar.blockSignals(True)
        scrollbar.setRange(mn, mx)
        scrollbar.setPageStep(step)
        if scrollbar.value() > mx: scrollbar.setValue(mx)
        scrollbar.blockSignals(False)
        scrollbar.setVisible(zoom_value > 100)
        graph.set_pan(scrollbar.value())

    def _add_line(self, graph_name, line_name, color, linewidth=2):
        g = self.graphs[graph_name]
        idx = g['widget'].add_line(color, line_name, self.max_points, linewidth)
        g['lines'][line_name] = idx

    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        ctrl = QtWidgets.QHBoxLayout()
        self.clear_btn = make_btn("Clear Data", self.clear_data)
        self.pause_btn = QtWidgets.QPushButton("Pause Updates")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(lambda c: self.update_timer.stop() if c else (self.update_timer.start(self.update_interval_ms) if self.pending_update else None))
        self.speed_combo = make_combo(["Fast (100ms)", "Normal (500ms)", "Slow (1000ms)", "Very Slow (2000ms)"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.stats_label = make_label("No data yet", color=ACCENT2, bold=True)
        for w in [self.clear_btn, self.pause_btn, make_label("Speed:"), self.speed_combo, None, self.stats_label]:
            (ctrl.addStretch() if w is None else ctrl.addWidget(w))
        main.addLayout(ctrl)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._make_graph_container("step_loss", "Per-Step Loss", "Loss"), 0, 0)
        grid.addWidget(self._make_graph_container("timestep", "Timestep", "Value"), 0, 1)
        grid.addWidget(self._make_graph_container("optim_loss", "Optimizer Loss (Avg)", "Loss"), 1, 0)
        grid.addWidget(self._make_graph_container("lr", "Learning Rate", "LR"), 1, 1)
        grid.addWidget(self._make_graph_container("grad_norm", "Gradient Norms", "Norm"), 2, 0, 1, 2)
        main.addLayout(grid)

        self._add_line("step_loss", "Step Loss", "#4CAF50")
        self._add_line("timestep", "Timestep", ACCENT2)
        self._add_line("optim_loss", "Avg Loss", ACCENT)
        self._add_line("lr", "LR", "#6a48d7")
        self._add_line("grad_norm", "Raw", DANGER, linewidth=3)
        self._add_line("grad_norm", "Clipped", WARN, linewidth=2)

        self.latest_global_step = self.latest_optim_step = self.latest_timestep = 0
        self.latest_lr = self.latest_step_loss = self.latest_optim_loss = self.latest_grad = 0.0

    def _on_speed_changed(self, i):
        self.update_interval_ms = [100, 500, 1000, 2000][i]
        if self.update_timer.isActive():
            self.update_timer.stop(); self.update_timer.start(self.update_interval_ms)

    def parse_and_update(self, text):
        if self.pause_btn.isChecked(): return
        added = False
        m = re.search(r'Training\s*\|.*\|\s*(\d+)/(\d+)\s*\[.*?\]\s*\[Loss:\s*([\d.e+-]+),\s*Timestep:\s*(\d+)\]', text)
        if m:
            step, loss, ts = int(m.group(1)) - 1, float(m.group(3)), int(m.group(4))
            self.pending_data.append(('progress_step', step, loss, ts))
            self.latest_global_step, self.latest_step_loss, self.latest_timestep = step, loss, ts
            added = True
        m = re.search(r'--- Optimizer Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.e+-]+)\s*\|\s*LR:\s*([\d.e+-]+)\s*---', text)
        if m:
            step, avg_loss, lr = int(m.group(1)), float(m.group(2)), float(m.group(3))
            self.pending_data.append(('optim_step', step, avg_loss, lr))
            self.latest_optim_step, self.latest_lr, self.latest_optim_loss = step, lr, avg_loss
            added = True
        m = re.search(r'Grad Norm \(Raw/Clipped\):\s*([\d.]+)\s*/\s*([\d.]+)', text)
        if m:
            self.pending_data.append(('grad', float(m.group(1)), float(m.group(2))))
            self.latest_grad = float(m.group(1))
            added = True
        if added:
            self.pending_update = True
            if not self.update_timer.isActive() and not self.pause_btn.isChecked():
                self.update_timer.start(self.update_interval_ms)

    def _perform_update(self):
        if not self.pending_update or not self.pending_data:
            self.update_timer.stop(); return
        last_optim_step = self.latest_optim_step
        for data in self.pending_data:
            t = data[0]
            if t == 'progress_step':
                _, step, loss, ts = data
                self.graphs['step_loss']['widget'].append_data(self.graphs['step_loss']['lines']['Step Loss'], step, loss)
                self.graphs['timestep']['widget'].append_data(self.graphs['timestep']['lines']['Timestep'], step, ts)
            elif t == 'optim_step':
                _, step, avg_loss, lr = data
                last_optim_step = step
                self.graphs['optim_loss']['widget'].append_data(self.graphs['optim_loss']['lines']['Avg Loss'], step, avg_loss)
                self.graphs['lr']['widget'].append_data(self.graphs['lr']['lines']['LR'], step, lr)
            elif t == 'grad' and last_optim_step is not None:
                _, raw, clipped = data
                self.graphs['grad_norm']['widget'].append_data(self.graphs['grad_norm']['lines']['Raw'], last_optim_step, raw)
                self.graphs['grad_norm']['widget'].append_data(self.graphs['grad_norm']['lines']['Clipped'], last_optim_step, clipped)
        self.pending_data.clear()
        self.stats_label.setText(
            f"Step: {self.latest_global_step} | Loss: {self.latest_step_loss:.4f} | "
            f"Timestep: {self.latest_timestep} | Avg Loss: {self.latest_optim_loss:.4f} | "
            f"LR: {self.latest_lr:.2e} | Grad: {self.latest_grad:.4f}")
        for name, gd in self.graphs.items():
            self._sync_pan(gd['widget'], gd['scrollbar'], gd['zoom_slider'].value())
        self.pending_update = False

    def clear_data(self):
        self.update_timer.stop()
        self.pending_update = False
        self.pending_data.clear()
        for name, gd in self.graphs.items():
            gd['widget'].clear_all_data()
            self._sync_pan(gd['widget'], gd['scrollbar'], gd['zoom_slider'].value())
        self.latest_global_step = self.latest_optim_step = self.latest_timestep = 0
        self.latest_lr = self.latest_step_loss = self.latest_optim_loss = self.latest_grad = 0.0
        self.stats_label.setText("No data yet")

    def showEvent(self, e):
        super().showEvent(e)
        if self.pending_update and not self.pause_btn.isChecked():
            self.update_timer.start(self.update_interval_ms)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.update_timer.stop()


# ──────────────────────────────────────────────────────────────────
#  LR CURVE WIDGET
# ──────────────────────────────────────────────────────────────────
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
        # Colors
        self.bg_color = QtGui.QColor(GRAPH_BG)
        self.grid_color = QtGui.QColor(BORDER)
        self.epoch_grid_color = QtGui.QColor(TEXT_SEC)
        self.line_color = QtGui.QColor(ACCENT)
        self.point_color = QtGui.QColor(TEXT_PRI)
        self.point_fill_color = QtGui.QColor(ACCENT)
        self.selected_point_color = QtGui.QColor(WARN)
        self.text_color = QtGui.QColor(TEXT_PRI)
        self.setMouseTracking(True)

    def set_epoch_data(self, d): self.epoch_data = d; self.update()

    def set_bounds(self, max_steps, min_lr, max_lr):
        self.max_steps = max(max_steps, 1)
        self.min_lr_bound = min_lr
        self.max_lr_bound = max_lr if max_lr > min_lr else min_lr + 1e-9
        changed = False
        for p in self._points:
            clamped = max(self.min_lr_bound, min(self.max_lr_bound, p[1]))
            if clamped != p[1]: p[1] = clamped; changed = True
        if changed: self.pointsChanged.emit(self._points)
        self._update_visual(); self.update()

    def set_points(self, points):
        self._points = sorted(points, key=lambda p: p[0])
        self._update_visual(); self.update()

    def _update_visual(self):
        self._visual_points = [[p[0], max(self.min_lr_bound, min(self.max_lr_bound, p[1]))] for p in self._points]

    def get_points(self): return self._points

    def _log_range(self):
        safe_max = max(self.max_lr_bound, 1e-12)
        log_max = math.log(safe_max)
        eff_min = max(self.min_lr_bound if self.min_lr_bound > 0 else safe_max / self.LOG_FLOOR_DIVISOR, 1e-12)
        return log_max, math.log(eff_min)

    def _to_pixel(self, nx, lr):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        px = self.padding['left'] + nx * gw
        if lr <= self.min_lr_bound: return QtCore.QPointF(px, self.padding['top'] + gh)
        log_max, log_min = self._log_range()
        lr_range = log_max - log_min
        if lr_range <= 0: return QtCore.QPointF(px, self.padding['top'])
        ny = (math.log(lr) - log_min) / lr_range
        return QtCore.QPointF(px, self.padding['top'] + (1 - ny) * gh)

    def _to_data(self, px, py):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        nx = (px - self.padding['left']) / gw
        ny = 1 - ((max(self.padding['top'], min(py, self.padding['top'] + gh)) - self.padding['top']) / gh)
        log_max, log_min = self._log_range()
        lr_range = log_max - log_min
        lr = math.exp(log_min + ny * lr_range) if lr_range > 0 else self.min_lr_bound
        return max(0.0, min(1.0, nx)), max(self.min_lr_bound, min(self.max_lr_bound, lr))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        gr = QtCore.QRect(self.padding['left'], self.padding['top'],
                          self.width() - self.padding['left'] - self.padding['right'],
                          self.height() - self.padding['top'] - self.padding['bottom'])
        self._draw_grid(painter, gr)
        self._draw_curve(painter)
        self._draw_points(painter)

    def _draw_grid(self, painter, rect):
        painter.setPen(self.grid_color)
        log_max, log_min = self._log_range()
        log_range = log_max - log_min
        f = self.font(); f.setPixelSize(13); painter.setFont(f)
        for i in range(5):
            y = rect.top() + (i / 4) * rect.height()
            painter.setPen(self.grid_color)
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            ny = 1.0 - i / 4
            lr_val = (self.max_lr_bound if i == 0 else self.min_lr_bound if i == 4
                      else math.exp(log_min + ny * log_range) if log_range > 0 else self.max_lr_bound)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(0, int(y - 10), self.padding['left'] - 5, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, f"{lr_val:.1e}")
            x_step = rect.left() + (i / 4) * rect.width()
            painter.drawText(QtCore.QRect(int(x_step - 50), rect.bottom() + 5, 100, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(self.max_steps * i / 4)))

        ep_pen = QtGui.QPen(self.epoch_grid_color)
        ep_pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        painter.setPen(ep_pen)
        f2 = self.font(); f2.setPixelSize(11); painter.setFont(f2)
        for nx, steps in self.epoch_data:
            x = rect.left() + nx * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(int(x - 40), rect.bottom() + 25, 80, 15),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(steps))
            painter.setPen(ep_pen)

        painter.setPen(self.text_color)
        f3 = self.font(); f3.setBold(True); f3.setPixelSize(13); painter.setFont(f3)
        painter.drawText(self.rect().adjusted(0, 5, 0, 0),
                         QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                         "Learning Rate Schedule")

    def _draw_curve(self, painter):
        if not self._visual_points: return
        pts = [self._to_pixel(p[0], p[1]) for p in self._visual_points]
        painter.setPen(QtGui.QPen(self.line_color, 2))
        painter.drawPolyline(QtGui.QPolygonF(pts))
        fill = QtGui.QPolygonF(pts)
        fill.append(self._to_pixel(self._visual_points[-1][0], self.min_lr_bound))
        fill.append(self._to_pixel(self._visual_points[0][0], self.min_lr_bound))
        fc = QtGui.QColor(self.point_fill_color); fc.setAlpha(50)
        painter.setBrush(fc); painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawPolygon(fill)

    def _draw_points(self, painter):
        for i, p in enumerate(self._visual_points):
            pos = self._to_pixel(p[0], p[1])
            is_sel = i == self._selected_point_index
            painter.setBrush(self.selected_point_color if is_sel else self.point_fill_color)
            painter.setPen(self.point_color)
            painter.drawEllipse(pos, self.point_radius, self.point_radius)
            if is_sel or i == self._dragging_point_index:
                op = self._points[i]
                painter.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 30, 100, 20),
                                 QtCore.Qt.AlignmentFlag.AlignCenter,
                                 f"({int(op[0] * self.max_steps)}, {op[1]:.1e})")

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton: return
        new_sel = -1
        for i, p in enumerate(self._visual_points):
            if (QtCore.QPointF(event.pos()) - self._to_pixel(p[0], p[1])).manhattanLength() < self.point_radius * 1.5:
                self._dragging_point_index = i; new_sel = i; break
        if self._selected_point_index != new_sel:
            self._selected_point_index = new_sel
            self.selectionChanged.emit(new_sel)
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging_point_index != -1:
            nx, lr = self._to_data(event.pos().x(), event.pos().y())
            i = self._dragging_point_index
            if i == 0 or i == len(self._points) - 1:
                nx = 0.0 if i == 0 else 1.0
            else:
                nx = max(self._points[i-1][0], min(self._points[i+1][0], nx))
            self._points[i] = [nx, lr]
            self._update_visual(); self.pointsChanged.emit(self._points); self.update()
        else:
            on_pt = any((QtCore.QPointF(event.pos()) - self._to_pixel(p[0], p[1])).manhattanLength() < self.point_radius * 1.5 for p in self._visual_points)
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor if on_pt else QtCore.Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton: return
        self._dragging_point_index = -1
        self.set_points(self._points); self.pointsChanged.emit(self._points)

    def add_point(self):
        if len(self._points) < 2: return
        max_gap, insert_idx = 0, -1
        for i in range(len(self._points) - 1):
            gap = self._points[i+1][0] - self._points[i][0]
            if gap > max_gap: max_gap = gap; insert_idx = i + 1
        if insert_idx != -1:
            prev, nxt = self._points[insert_idx-1], self._points[insert_idx]
            _, log_min = self._log_range()
            new_lr = math.exp(max(log_min, (math.log(max(prev[1], 1e-12)) + math.log(max(nxt[1], 1e-12))) / 2))
            self._points.insert(insert_idx, [(prev[0] + nxt[0]) / 2, new_lr])
            self.set_points(self._points); self.pointsChanged.emit(self._points)

    def remove_selected_point(self):
        i = self._selected_point_index
        if 0 < i < len(self._points) - 1:
            self._points.pop(i)
            self._selected_point_index = -1
            self.selectionChanged.emit(-1)
            self.set_points(self._points); self.pointsChanged.emit(self._points)

    def apply_preset(self, pts): self.set_points(pts); self.pointsChanged.emit(pts)

    def set_generated_preset(self, mode, num_restarts, warmup_pct, restart_rampup_pct):
        num_restarts = max(1, int(num_restarts))
        warmup_pct = max(0.0, min(1.0, float(warmup_pct)))
        restart_rampup_pct = max(0.0, min(1.0, float(restart_rampup_pct)))
        eps = 1e-5
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        points = []
        seg_len = 1.0 / num_restarts

        for i in range(num_restarts):
            seg_start = i * seg_len
            seg_end = (i + 1) * seg_len
            wp = warmup_pct if i == 0 else restart_rampup_pct
            warmup_len = max(eps, min(seg_len * wp, seg_len))
            decay_len = seg_len - warmup_len
            peak_x = min(seg_start + warmup_len, seg_end)
            points.append([seg_start, min_lr])
            points.append([peak_x, max_lr])
            if decay_len > eps:
                if mode == "Cosine":
                    for j in range(1, 21):
                        t = j / 20
                        x = peak_x + t * decay_len
                        y = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))
                        points.append([x, y])
                else:
                    points.append([seg_end, min_lr])
            else:
                points.append([seg_end, max_lr])

        # Clean and deduplicate
        final, last_x = [], -1.0
        for p in sorted(points, key=lambda p: p[0]):
            x = max(0.0, min(1.0, p[0]))
            y = max(min_lr, min(max_lr, p[1]))
            if x > last_x + 1e-9: final.append([x, y]); last_x = x
        if not final or final[0][0] != 0.0: final.insert(0, [0.0, min_lr])
        if final[-1][0] < 1.0: final.append([1.0, final[-1][1]])
        self.apply_preset(final)


# ──────────────────────────────────────────────────────────────────
#  TIMESTEP HISTOGRAM WIDGET
# ──────────────────────────────────────────────────────────────────
class TimestepHistogramWidget(QtWidgets.QWidget):
    allocationChanged = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.padding = {'top': 40, 'bottom': 50, 'left': 35, 'right': 20}
        self.bin_size = 100
        self.max_tickets = 1000
        self.counts = []
        self._dragging_bin_index = -1
        self.bg_color = QtGui.QColor(GRAPH_BG)
        self.bar_color_even = QtGui.QColor(ACCENT)
        self.bar_color_odd  = QtGui.QColor("#5839b0")
        self.bar_hover_color = QtGui.QColor(ACCENT2)
        self.disabled_bar_color = QtGui.QColor(BORDER)
        self.text_color = QtGui.QColor(TEXT_PRI)
        self.grid_color = QtGui.QColor(BORDER)
        self.alert_color = QtGui.QColor(DANGER)
        self.disabled_text_color = QtGui.QColor(TEXT_SEC)
        self.setMouseTracking(True)
        self._init_bins()

    def set_total_steps(self, steps):
        steps = max(int(steps), 1)
        self.max_tickets = steps
        cur = sum(self.counts)
        if not self.counts or cur == 0: self._init_bins(); return
        raw = [(c / cur) * steps for c in self.counts]
        new = [int(x) for x in raw]
        diff = steps - sum(new)
        fracs = sorted(enumerate(raw[i] - new[i] for i in range(len(new))), key=lambda x: x[1], reverse=True)
        for i in range(diff): new[fracs[i][0]] += 1
        self.counts = new
        self.update(); self._emit_change()

    def set_bin_size(self, size):
        if size <= 0: return
        self.bin_size = size
        self._init_bins(); self.update(); self._emit_change()

    def set_allocation(self, alloc):
        if not alloc or "bin_size" not in alloc or "counts" not in alloc:
            self._init_bins(); return
        self.bin_size = alloc["bin_size"]
        expected = math.ceil(1000 / self.bin_size)
        if len(alloc["counts"]) != expected:
            self._init_bins()
        else:
            self.counts = alloc["counts"]
            s = sum(self.counts)
            if s > 0: self.max_tickets = s
        self.update()

    def get_allocation(self): return {"bin_size": self.bin_size, "counts": self.counts}

    def generate_from_weights(self, weights):
        n = len(self.counts)
        if n == 0 or not weights: return
        tw = sum(weights) or 1
        raw = [(w / tw) * self.max_tickets for w in weights]
        new = [int(c) for c in raw]
        diff = self.max_tickets - sum(new)
        fracs = sorted(enumerate(raw[i] - new[i] for i in range(n)), key=lambda x: x[1], reverse=True)
        for i in range(diff): new[fracs[i][0]] += 1
        self.counts = new; self.update(); self._emit_change()

    def _init_bins(self):
        self.bin_size = max(self.bin_size, 1)
        n = max(math.ceil(1000 / self.bin_size), 1)
        base, rem = divmod(self.max_tickets, n)
        self.counts = [base + (1 if i < rem else 0) for i in range(n)]
        self._emit_change()

    def _emit_change(self): self.allocationChanged.emit(self.get_allocation())

    def _bin_rect(self, idx, rect):
        n = len(self.counts)
        if n == 0: return QtCore.QRectF()
        bw = rect.width() / n
        x = rect.left() + idx * bw
        y_scale = max((max(self.counts) if self.counts else 1) * 1.2, self.max_tickets * 0.2)
        h = (self.counts[idx] / y_scale) * rect.height()
        return QtCore.QRectF(x, rect.bottom() - h, bw, h)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        enabled = self.isEnabled()
        self._paint_grid(painter, rect, enabled)
        self._paint_bars(painter, rect, enabled)
        self._paint_labels(painter, enabled)
        self._paint_header(painter, enabled)

    def _paint_grid(self, painter, rect, enabled):
        gc = self.grid_color if enabled else self.disabled_text_color
        tc = self.text_color if enabled else self.disabled_text_color
        painter.setPen(QtGui.QPen(gc, 1, QtCore.Qt.PenStyle.DashLine))
        for i in range(5):
            y = rect.bottom() - (i / 4) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        for i in range(6):
            x = rect.left() + (i / 5) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.setPen(tc)
            painter.drawText(QtCore.QRectF(x - 30, rect.bottom() + 5, 60, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(i / 5 * 1000)))
            painter.setPen(QtGui.QPen(gc, 1, QtCore.Qt.PenStyle.DashLine))

    def _paint_bars(self, painter, rect, enabled):
        f = painter.font(); f.setPixelSize(11); painter.setFont(f)
        for i, count in enumerate(self.counts):
            br = self._bin_rect(i, rect)
            color = (self.bar_hover_color if i == self._dragging_bin_index else
                     (self.bar_color_odd if i % 2 else self.bar_color_even)) if enabled else self.disabled_bar_color
            painter.setBrush(color); painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRect(br)
            if enabled:
                painter.setPen(self.text_color)
                ty = br.top() - 20 - (15 if i % 2 else 0)
                if ty < rect.top(): ty = br.top() + 5 if br.height() > 40 else rect.top() + 5
                painter.drawText(QtCore.QRectF(br.left(), ty, br.width(), 20), QtCore.Qt.AlignmentFlag.AlignCenter, str(count))

    def _paint_labels(self, painter, enabled):
        painter.setPen(self.text_color if enabled else self.disabled_text_color)
        painter.save()
        painter.translate(12, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-100, -10, 200, 20), QtCore.Qt.AlignmentFlag.AlignCenter, "Tickets Count")
        painter.restore()
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Timestep Range")

    def _paint_header(self, painter, enabled):
        used, total = sum(self.counts), self.max_tickets
        f = painter.font(); f.setBold(True); f.setPixelSize(13); painter.setFont(f)
        painter.setPen(QtGui.QColor(SUCCESS if used == total else DANGER if used > total else TEXT_PRI))
        painter.drawText(self.rect().adjusted(0, 5, 0, 0),
                         QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                         f"Tickets Used: {used} / {total}")

    def mousePressEvent(self, event):
        if not self.isEnabled() or event.button() != QtCore.Qt.MouseButton.LeftButton: return
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        rel_x = event.pos().x() - rect.left()
        n = len(self.counts)
        if n and 0 <= rel_x <= rect.width():
            self._dragging_bin_index = int(rel_x / (rect.width() / n)); self.update()

    def mouseMoveEvent(self, event):
        if not self.isEnabled(): return
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        n = len(self.counts)
        rel_x = event.pos().x() - rect.left()
        if self._dragging_bin_index == -1:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if n and 0 <= rel_x <= rect.width() else QtCore.Qt.CursorShape.ArrowCursor)
            return
        idx = self._dragging_bin_index
        if not (0 <= idx < n): return
        y_scale = max((max(self.counts) if self.counts else 1) * 1.2, self.max_tickets * 0.2)
        target = int(((rect.bottom() - event.pos().y()) / rect.height()) * y_scale)
        avail = self.max_tickets - (sum(self.counts) - self.counts[idx])
        new_val = max(0, min(target, avail))
        if new_val != self.counts[idx]:
            self.counts[idx] = new_val; self._emit_change(); self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_bin_index = -1; self.update()


# ──────────────────────────────────────────────────────────────────
#  PROCESS RUNNER THREAD
# ──────────────────────────────────────────────────────────────────
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
                cwd=self.working_dir, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, creationflags=flags)
            self.logSignal.emit(f"INFO: Started subprocess (PID: {self.process.pid})")
            for line in iter(self.process.stdout.readline, ''):
                line = line.strip()
                if not line or "NOTE: Redirects are currently not supported" in line: continue
                if line.startswith("GUI_PARAM_INFO::"):
                    self.paramInfoSignal.emit(line.replace('GUI_PARAM_INFO::', '').strip())
                else:
                    is_progress = '\r' in line or bool(re.match(r'^\s*\d+%\|\S*\|', line))
                    if any(kw in line.lower() for kw in ["memory inaccessible", "cuda out of memory", "access violation", "nan/inf"]):
                        self.logSignal.emit(f"*** ERROR DETECTED: {line} ***")
                    else:
                        self.progressSignal.emit(line.split('\r')[-1], is_progress)
                    self.metricsSignal.emit(line)
                    if "saved latents cache" in line.lower() or "caching complete" in line.lower():
                        self.cacheCreatedSignal.emit()
            self.finishedSignal.emit(self.process.wait())
        except Exception as e:
            self.errorSignal.emit(f"Subprocess error: {str(e)}")
            self.finishedSignal.emit(-1)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait()
            self.logSignal.emit("Process stopped.")


# ──────────────────────────────────────────────────────────────────
#  DATASET LOADER THREAD
# ──────────────────────────────────────────────────────────────────
class DatasetLoaderThread(QThread):
    finished = pyqtSignal(list, str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.repeats = 1

    def run(self):
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        images_data = []
        try:
            for file_path in Path(self.path).rglob("*"):
                if file_path.suffix.lower() in exts:
                    cap_path = file_path.with_suffix('.txt')
                    images_data.append({"image_path": str(file_path),
                                        "caption_path": str(cap_path) if cap_path.exists() else None,
                                        "caption_loaded": False, "caption": ""})
        except Exception as e:
            print(f"Error scanning {self.path}: {e}")
        self.finished.emit(images_data, self.path)


# ──────────────────────────────────────────────────────────────────
#  DATASET MANAGER WIDGET
# ──────────────────────────────────────────────────────────────────
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
        layout.setContentsMargins(0, 0, 0, 0)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(make_btn("Add Folder (Standard)", self.add_dataset_folder_native))
        smart_btn = make_btn("Add Folder (Smart)", self.add_dataset_folder_custom)
        smart_btn.setStyleSheet(f"background: {PANEL_BG}; color: {ACCENT}; border: 1px solid {ACCENT};")
        top.addWidget(smart_btn)
        top.addWidget(make_label("Sort By:"))
        self.sort_combo = make_combo(["Default (Order Added)", "Name (A-Z)", "Name (Z-A)",
                                      "Image Count (High → Low)", "Image Count (Low → High)"])
        self.sort_combo.currentIndexChanged.connect(self.sort_datasets)
        top.addWidget(self.sort_combo)
        self.loading_label = make_label("", color=WARN, bold=True)
        top.addWidget(self.loading_label)
        top.addStretch()
        layout.addLayout(top)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.grid_container = QtWidgets.QWidget()
        self.dataset_grid = QtWidgets.QGridLayout(self.grid_container)
        self.dataset_grid.setSpacing(15)
        self.dataset_grid.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(self.grid_container)
        layout.addWidget(scroll)

        bot = QtWidgets.QHBoxLayout()
        bot.addStretch(1)
        self.total_label = make_label("0", color=ACCENT2, bold=True)
        self.total_repeats_label = make_label("0", color=ACCENT, bold=True)
        for lbl, val in [("Total Images:", self.total_label), ("With Repeats:", self.total_repeats_label)]:
            bot.addWidget(make_label(lbl))
            bot.addWidget(val)
        bot.addStretch()
        layout.addLayout(bot)

    def get_total_repeats(self): return sum(d["image_count"] * d["repeats"] for d in self.datasets)
    def get_datasets_config(self): return [{"path": d["path"], "repeats": d["repeats"]} for d in self.datasets]

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
        self.datasets.append({"path": path, "images_data": images_data, "image_count": len(images_data),
                               "current_preview_idx": 0, "repeats": repeats})
        self.sort_datasets(); self.update_dataset_totals()

    def sort_datasets(self):
        c = self.sort_combo.currentText()
        if   "A-Z"   in c: self.datasets.sort(key=lambda x: Path(x['path']).name.lower())
        elif "Z-A"   in c: self.datasets.sort(key=lambda x: Path(x['path']).name.lower(), reverse=True)
        elif "High"  in c: self.datasets.sort(key=lambda x: x['image_count'], reverse=True)
        elif "Low"   in c: self.datasets.sort(key=lambda x: x['image_count'])
        self.repopulate_dataset_grid()

    def load_datasets_from_config(self, datasets_config):
        self.datasets = []
        for t in self.loader_threads: t.quit()
        self.loader_threads = []
        self.repopulate_dataset_grid()
        for d in datasets_config:
            p = d.get("path")
            if p and os.path.exists(p): self._start_loading_path(p, d.get("repeats", 1))

    def add_dataset_folder_native(self):
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder", self.parent_gui.last_browsed_path)
        if p: self.parent_gui.last_browsed_path = p; self._start_loading_path(p, 1)

    def add_dataset_folder_custom(self):
        dialog = CustomFolderDialog(self.parent_gui.last_browsed_path, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.selected_path:
            self.parent_gui.last_browsed_path = dialog.selected_path
            self._start_loading_path(dialog.selected_path, 1)

    def _cycle_preview(self, idx, direction):
        ds = self.datasets[idx]
        ds["current_preview_idx"] = (ds["current_preview_idx"] + direction) % len(ds["images_data"])
        self._update_preview_for_card(idx)

    def _update_preview_for_card(self, idx):
        if idx >= len(self.dataset_widgets): return
        ds = self.datasets[idx]
        wdg = self.dataset_widgets[idx]
        data = ds["images_data"][ds["current_preview_idx"]]
        if not data["caption_loaded"]:
            if data["caption_path"]:
                try:
                    with open(data["caption_path"], 'r', encoding='utf-8') as f: data["caption"] = f.read().strip()
                except Exception: data["caption"] = "[Error reading caption]"
            else: data["caption"] = "[No caption file]"
            data["caption_loaded"] = True
        px = QtGui.QPixmap(data["image_path"])
        if not px.isNull(): px = px.scaled(183, 183, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        wdg["preview_label"].setPixmap(px)
        wdg["caption_text"].setPlainText(data["caption"])
        wdg["counter_label"].setText(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")

    def repopulate_dataset_grid(self):
        while self.dataset_grid.count():
            item = self.dataset_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.dataset_widgets = []
        n = len(self.datasets)
        COLS = 1 if n == 1 else (2 if n == 2 else 3)

        for idx, ds in enumerate(self.datasets):
            card = QtWidgets.QGroupBox()
            card.setStyleSheet(f"QGroupBox {{ border: 2px solid {BORDER}; border-radius: 8px; margin-top: 5px; padding: 12px; background: {DARK_BG}; }}")
            cl = QtWidgets.QVBoxLayout(card)
            cl.setSpacing(10)

            # Preview + caption
            top_h = QtWidgets.QHBoxLayout()
            top_h.setSpacing(12)

            preview_sec = QtWidgets.QVBoxLayout()
            preview_sec.setSpacing(5)
            img_h = QtWidgets.QHBoxLayout()
            img_h.addStretch()
            preview_lbl = QtWidgets.QLabel()
            preview_lbl.setFixedSize(183, 183)
            preview_lbl.setStyleSheet(f"border: 1px solid {BORDER}; background: {GRAPH_BG};")
            preview_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            img_h.addWidget(preview_lbl)
            img_h.addStretch()
            preview_sec.addLayout(img_h)

            nav_h = QtWidgets.QHBoxLayout()
            nav_h.setSpacing(8)
            left_btn = make_btn("◄")
            left_btn.setFixedHeight(22); left_btn.setMinimumWidth(35)
            left_btn.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 0;")
            left_btn.clicked.connect(lambda _, i=idx: self._cycle_preview(i, -1))
            counter_lbl = make_label(f"1/{len(ds['images_data'])}", color=ACCENT, bold=True)
            counter_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            right_btn = make_btn("►")
            right_btn.setFixedHeight(22); right_btn.setMinimumWidth(35)
            right_btn.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 0;")
            right_btn.clicked.connect(lambda _, i=idx: self._cycle_preview(i, 1))
            nav_h.addWidget(left_btn); nav_h.addWidget(counter_lbl, 1); nav_h.addWidget(right_btn)
            preview_sec.addLayout(nav_h)
            top_h.addLayout(preview_sec)

            # Caption
            cap_container = QtWidgets.QWidget()
            cap_container.setStyleSheet(f"QWidget {{ background: {GRAPH_BG}; border: 1px solid {BORDER}; border-radius: 4px; }}")
            cap_container.setMinimumWidth(150)
            cap_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            cap_layout = QtWidgets.QVBoxLayout(cap_container)
            cap_layout.setContentsMargins(8, 8, 8, 8)
            cap_layout.addWidget(make_label("<b>Caption Preview:</b>", color=ACCENT, size=11))
            caption_text = QtWidgets.QTextEdit("Loading...")
            caption_text.setReadOnly(True)
            caption_text.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            caption_text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            caption_text.setMinimumHeight(183)
            caption_text.setStyleSheet(f"QTextEdit {{ background: {GRAPH_BG}; color: {TEXT_PRI}; font-size: 9pt; border: none; }}")
            cap_layout.addWidget(caption_text, 1)
            top_h.addWidget(cap_container, 1)
            cl.addLayout(top_h)
            cl.addWidget(make_separator())

            # Info
            path_short = (Path(ds['path']).name[:27] + "...") if len(Path(ds['path']).name) > 30 else Path(ds['path']).name
            path_lbl = make_label(f"<b>Folder:</b> {path_short}")
            path_lbl.setToolTip(ds['path'])
            cl.addWidget(path_lbl)
            cl.addWidget(make_label(f"<b>Images:</b> {ds['image_count']}"))
            repeats_total_lbl = make_label(f"<b>Total (with repeats):</b> {ds['image_count'] * ds['repeats']}", color=ACCENT)
            cl.addWidget(repeats_total_lbl)

            # Repeats
            rep_h = QtWidgets.QHBoxLayout()
            rep_h.setContentsMargins(0, 5, 0, 0)
            rep_h.addWidget(make_label("Repeats:"))
            rep_spin = NoScrollSpinBox()
            rep_spin.setStyleSheet(f"background: #080a0e; color: {TEXT_PRI}; border: 1px solid {BORDER}; border-radius: 4px; padding: 2px 4px;")
            rep_spin.setRange(1, 10000); rep_spin.setValue(ds["repeats"])
            rep_spin.valueChanged.connect(lambda v, i=idx: self.update_repeats(i, v))
            rep_h.addWidget(rep_spin, 1)
            cl.addLayout(rep_h)

            # Buttons
            btn_h = QtWidgets.QHBoxLayout()
            btn_h.setSpacing(5)
            rm_btn = make_btn("Remove"); rm_btn.setStyleSheet("min-height: 24px; max-height: 24px; padding: 4px 15px;")
            rm_btn.clicked.connect(lambda _, i=idx: self.remove_dataset(i))
            clear_btn = make_btn("Clear Cache"); clear_btn.setStyleSheet("min-height: 24px; max-height: 24px; padding: 4px 15px;")
            clear_btn.clicked.connect(lambda _, p=ds["path"]: self.confirm_clear_cache(p))
            cache_exists = self._cache_exists(ds["path"])
            clear_btn.setEnabled(cache_exists)
            if not cache_exists: clear_btn.setToolTip("No cache found")
            btn_h.addWidget(rm_btn); btn_h.addWidget(clear_btn)
            cl.addLayout(btn_h)

            row, col = divmod(idx, COLS)
            self.dataset_grid.addWidget(card, row, col)
            self.dataset_widgets.append({
                "preview_label": preview_lbl, "caption_text": caption_text,
                "counter_label": counter_lbl, "repeats_total_label": repeats_total_lbl,
                "clear_btn": clear_btn
            })
            self._update_preview_for_card(idx)

        if n % COLS != 0:
            for ec in range(n % COLS, COLS): self.dataset_grid.setColumnStretch(ec, 1)

    def _cache_exists(self, path):
        name = self.parent_gui.get_current_cache_folder_name()
        d = Path(path) / name
        return d.exists() and d.is_dir()

    def update_repeats(self, idx, val):
        self.datasets[idx]["repeats"] = val
        if idx < len(self.dataset_widgets):
            ds = self.datasets[idx]
            self.dataset_widgets[idx]["repeats_total_label"].setText(f"<b>Total (with repeats):</b> {ds['image_count'] * ds['repeats']}")
        self.update_dataset_totals()

    def remove_dataset(self, idx):
        del self.datasets[idx]; self.repopulate_dataset_grid(); self.update_dataset_totals()

    def update_dataset_totals(self):
        self.total_label.setText(str(sum(d["image_count"] for d in self.datasets)))
        self.total_repeats_label.setText(str(self.get_total_repeats()))
        self.datasetsChanged.emit()

    def confirm_clear_cache(self, path):
        name = self.parent_gui.get_current_cache_folder_name()
        if QtWidgets.QMessageBox.question(self, "Confirm", f"Delete cached latents in '{name}' for this dataset?",
                                          QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes:
            self.clear_cached_latents(path)

    def clear_cached_latents(self, path):
        name = self.parent_gui.get_current_cache_folder_name()
        deleted = False
        for d in list(Path(path).rglob(name)):
            if d.is_dir():
                try: shutil.rmtree(d); self.parent_gui.log(f"Deleted cache: {d}"); deleted = True
                except Exception as e: self.parent_gui.log(f"Error deleting {d}: {e}")
        if deleted: self.refresh_cache_buttons()
        else: self.parent_gui.log(f"No cache directories found matching '{name}'.")

    def refresh_cache_buttons(self):
        for idx, ds in enumerate(self.datasets):
            if idx < len(self.dataset_widgets):
                exists = self._cache_exists(ds["path"])
                btn = self.dataset_widgets[idx]["clear_btn"]
                btn.setEnabled(exists)
                btn.setToolTip("" if exists else "No cache found")


# ──────────────────────────────────────────────────────────────────
#  TRAINING GUI  (main window)
# ──────────────────────────────────────────────────────────────────
# Widget definitions: (label, tooltip, widget_type, *extra)
UI_DEFS = {
    "SINGLE_FILE_CHECKPOINT_PATH": ("Base Model (.safetensors)", "Path to the base SDXL model.", "path", "file_safetensors"),
    "VAE_PATH":                    ("Separate VAE (Optional)", "Leave empty to use the VAE from the base model.", "path", "file_safetensors"),
    "OUTPUT_DIR":                  ("Output Directory", "Folder where checkpoints will be saved.", "path", "folder"),
    "CACHING_BATCH_SIZE":          ("Caching Batch Size", "Adjust based on VRAM (e.g., 2-8).", "spin", 1, 64),
    "NUM_WORKERS":                 ("Dataloader Workers", "Set to 0 on Windows if you have issues.", "spin", 0, 16),
    "UNCONDITIONAL_DROPOUT":       ("Unconditional Dropout", "Randomly replace captions with empty strings.", "check"),
    "UNCONDITIONAL_DROPOUT_CHANCE":("Dropout Chance", "Probability (0.0-1.0) of dropping the caption.", "dspin", 0.0, 1.0, 0.05, 2),
    "TARGET_PIXEL_AREA":           ("Target Pixel Area", "e.g., 1024*1024=1048576.", "line"),
    "SHOULD_UPSCALE":              ("Upscale Images", "Upscale small images closer to bucket limit.", "check"),
    "MAX_AREA_TOLERANCE":          ("Max Area Tolerance", "Multiplier over target area when upscaling.", "line"),
    "PREDICTION_TYPE":             ("Prediction Type", "v_prediction or epsilon. Must match the base model.", "combo", ["v_prediction", "epsilon"]),
    "BETA_SCHEDULE":               ("Beta Schedule", "Noise schedule for the diffuser.", "combo", ["scaled_linear", "linear", "squared", "squaredcos_cap_v2"]),
    "MAX_TRAIN_STEPS":             ("Max Training Steps", "Total number of training steps.", "line"),
    "BATCH_SIZE":                  ("Batch Size", "Number of samples per batch.", "spin", 1, 32),
    "SAVE_EVERY_N_STEPS":          ("Save Every N (Optimizer Steps)", "When to save a checkpoint.", "line"),
    "GRADIENT_ACCUMULATION_STEPS": ("Gradient Accumulation", "Simulates a larger batch size.", "line"),
    "MIXED_PRECISION":             ("Mixed Precision", "bfloat16 for modern GPUs, float16 for older.", "combo", ["bfloat16", "float16"]),
    "CLIP_GRAD_NORM":              ("Gradient Clipping", "Maximum gradient norm. 0 to disable.", "line"),
    "SEED":                        ("Seed", "Ensures reproducible training.", "line"),
    "RESUME_MODEL_PATH":           ("Resume Model", "The .safetensors checkpoint file.", "path", "file_safetensors"),
    "RESUME_STATE_PATH":           ("Resume State", "The .pt optimizer state file.", "path", "file_pt"),
    "UNET_EXCLUDE_TARGETS":        ("Exclude Layers (Keywords)", "Keywords for layers to exclude (comma-separated).", "textedit", 100),
    "LR_GRAPH_MIN":                ("Graph Min LR", "Minimum learning rate displayed on the Y-axis.", "line"),
    "LR_GRAPH_MAX":                ("Graph Max LR", "Maximum learning rate displayed on the Y-axis.", "line"),
    "MEMORY_EFFICIENT_ATTENTION":  ("Attention Backend", "Select the attention mechanism to use.", "combo", ["sdpa", "flex_attn", "cudnn", "xformers (Only if no Flash)"]),
    "USE_ZERO_TERMINAL_SNR":       ("Use Zero-Terminal SNR", "Rescales noise schedule for better dynamic range.", "check"),
    "GRAD_SPIKE_THRESHOLD_HIGH":   ("Spike Threshold (High)", "Trigger detector if gradient norm exceeds this.", "line"),
    "GRAD_SPIKE_THRESHOLD_LOW":    ("Spike Threshold (Low)", "Trigger detector if gradient norm is below this.", "line"),
    "LOSS_TYPE":                   ("Loss Type", "Select the loss function strategy.", "combo", ["MSE", "Semantic", "Huber_Adaptive"]),
    "SEMANTIC_SEP_WEIGHT":         ("Separation Region Weight", "Weight for blob/color separation regions.", "dspin", 0.0, 2.0, 0.05, 2),
    "SEMANTIC_DETAIL_WEIGHT":      ("Lineart/Detail Weight", "Weight for fine details, edges, and lineart.", "dspin", 0.0, 2.0, 0.05, 2),
    "SEMANTIC_ENTROPY_WEIGHT":     ("Entropy/Texture Weight", "Weight for texture complexity. Requires scikit-image.", "dspin", 0.0, 2.0, 0.05, 2),
    "LOSS_HUBER_BETA":             ("Huber Beta", "Threshold where quadratic loss turns linear (0.5 standard).", "dspin", 0.01, 1.0, 0.05, 2),
    "LOSS_ADAPTIVE_DAMPING":       ("Adaptive Damping", "Controls sensitivity to magnitude.", "dspin", 0.001, 1.0, 0.01, 3),
    "VAE_SHIFT_FACTOR":            ("VAE Shift Factor", "Latent shift mean.", "dspin", -10.0, 10.0, 0.0001, 4),
    "VAE_SCALING_FACTOR":          ("VAE Scaling Factor", "Latent scaling factor.", "dspin", 0.0, 10.0, 0.0001, 5),
    "VAE_LATENT_CHANNELS":         ("Latent Channels", "4 for Standard/EQ, 32 for Flux/NoobAI.", "spin", 4, 128),
    "RF_SHIFT_FACTOR":             ("RF Shift Factor", "Shift factor for SD3/Flux schedules.", "dspin", 0.0, 100.0, 0.01, 4),
}


class TrainingGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA SDXL Trainer")
        self.setMinimumSize(1000, 800)
        self.resize(1500, 1000)
        self.config_dir = "configs"
        self.state_file = os.path.join(self.config_dir, "gui_state.json")
        self.widgets = {}
        self.process_runner = None
        self.current_config = {}
        self.last_line_is_progress = False
        self.default_config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        self.presets = {}
        self.last_browsed_path = os.getcwd()

        self._initialize_configs()
        self._setup_ui()

        self.config_dropdown.blockSignals(True)
        last = self._load_gui_state()
        if last and last in self.presets:
            idx = self.config_dropdown.findData(last)
            if idx != -1:
                self.config_dropdown.setCurrentIndex(idx)
                self.load_selected_config(idx)
            else:
                self._load_first_config()
        else:
            self._load_first_config()
        self.config_dropdown.blockSignals(False)

    # ── Config management ─────────────────────────────────────────
    def _initialize_configs(self):
        os.makedirs(self.config_dir, exist_ok=True)
        if not any(f.endswith(".json") and f != "gui_state.json" for f in os.listdir(self.config_dir)):
            with open(os.path.join(self.config_dir, "default.json"), 'w') as f:
                json.dump(self.default_config, f, indent=4)
        self.presets = {}
        for fn in os.listdir(self.config_dir):
            if fn.endswith(".json") and fn != "gui_state.json":
                name = os.path.splitext(fn)[0]
                try:
                    with open(os.path.join(self.config_dir, fn), 'r') as f:
                        self.presets[name] = json.load(f)
                except Exception as e:
                    self.log(f"Warning: Could not load '{fn}': {e}")

    def _load_gui_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f: return json.load(f).get("last_config")
        except Exception: pass
        return None

    def _save_gui_state(self):
        try:
            idx = self.config_dropdown.currentIndex()
            if idx >= 0:
                key = self.config_dropdown.itemData(idx) or self.config_dropdown.itemText(idx).replace(" ", "_").lower()
                os.makedirs(self.config_dir, exist_ok=True)
                with open(self.state_file, 'w') as f: json.dump({"last_config": key}, f, indent=4)
        except Exception as e: self.log(f"Warning: Could not save GUI state: {e}")

    def _load_first_config(self):
        if self.config_dropdown.count() > 0:
            self.config_dropdown.setCurrentIndex(0)
            self.load_selected_config(0)
        else:
            self.current_config = copy.deepcopy(self.default_config)
            self._apply_config_to_widgets()

    def load_selected_config(self, index):
        key = self.config_dropdown.itemData(index) or self.config_dropdown.itemText(index).replace(" ", "_").lower()
        if key in self.presets:
            self.current_config = {**copy.deepcopy(self.default_config), **self.presets[key]}
            self.log(f"Loaded config: '{key}.json'")
        else:
            self.current_config = copy.deepcopy(self.default_config)
            self.log(f"Warning: preset '{key}' not found, using defaults.")
        self._apply_config_to_widgets()
        self._save_gui_state()

    def save_config(self):
        idx = self.config_dropdown.currentIndex()
        if idx < 0: self.log("Error: No configuration selected."); return
        key = self.config_dropdown.itemData(idx) or self.config_dropdown.itemText(idx).replace(" ", "_").lower()
        cfg = self._collect_config()
        try:
            with open(os.path.join(self.config_dir, f"{key}.json"), 'w') as f: json.dump(cfg, f, indent=4)
            self.presets[key] = cfg
            self.log(f"Saved config: '{key}.json'")
        except Exception as e: self.log(f"Error saving config: {e}")

    def save_as_config(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset As", "Enter preset name (alphanumeric, underscores):")
        if not ok or not name: return
        if not re.match(r'^[a-zA-Z0-9_]+$', name): self.log("Error: Invalid preset name."); return
        save_path = os.path.join(self.config_dir, f"{name}.json")
        if os.path.exists(save_path):
            if QtWidgets.QMessageBox.question(self, "Overwrite?", f"Preset '{name}' exists. Overwrite?",
                                              QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        cfg = self._collect_config()
        try:
            with open(save_path, 'w') as f: json.dump(cfg, f, indent=4)
            self.presets[name] = cfg
            self.config_dropdown.blockSignals(True)
            self.config_dropdown.clear()
            for pn in sorted(self.presets.keys()):
                self.config_dropdown.addItem(pn.replace("_", " ").title(), pn)
            self.config_dropdown.setCurrentIndex(self.config_dropdown.findData(name))
            self.config_dropdown.blockSignals(False)
            self.log(f"Saved preset: '{name}.json'")
        except Exception as e: self.log(f"Error saving preset: {e}")

    def restore_defaults(self):
        idx = self.config_dropdown.currentIndex()
        if idx < 0: return
        key = self.config_dropdown.itemData(idx) or self.config_dropdown.itemText(idx).replace(" ", "_").lower()
        if QtWidgets.QMessageBox.question(self, "Restore Defaults",
                                          f"Overwrite '{key}.json' with hardcoded defaults?",
                                          QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes:
            self.presets[key] = copy.deepcopy(self.default_config)
            self.save_config(); self.load_selected_config(idx)
            self.log(f"Restored '{key}.json' to defaults.")

    # ── Widget creation helpers ───────────────────────────────────
    def _make_widget(self, key):
        """Create a widget from UI_DEFS and register it."""
        if key not in UI_DEFS: return None, None
        d = UI_DEFS[key]
        label_text, tooltip, wtype = d[0], d[1], d[2]
        extra = d[3:]
        lbl = QtWidgets.QLabel(label_text)
        lbl.setToolTip(tooltip)

        if wtype == "line":
            w = QtWidgets.QLineEdit()
            w.textChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "textedit":
            w = QtWidgets.QPlainTextEdit()
            if extra: w.setFixedHeight(extra[0])
            w.textChanged.connect(lambda: self._sync_widget(key))
        elif wtype == "spin":
            lo, hi = extra[0], extra[1]
            w = NoScrollSpinBox(); w.setRange(lo, hi)
            w.valueChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "dspin":
            lo, hi, step, dec = extra[0], extra[1], extra[2], extra[3]
            w = NoScrollDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step); w.setDecimals(dec)
            w.valueChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "combo":
            w = NoScrollComboBox(); w.addItems(extra[0])
            w.currentTextChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "check":
            w = QtWidgets.QCheckBox(label_text); w.setToolTip(tooltip)
            w.stateChanged.connect(lambda _, k=key: self._sync_widget(k))
            self.widgets[key] = w
            return None, w
        elif wtype == "path":
            file_type = extra[0]
            line = QtWidgets.QLineEdit()
            line.textChanged.connect(lambda _, k=key: self._sync_widget(k))
            self.widgets[key] = line
            container = QtWidgets.QWidget()
            vb = QtWidgets.QVBoxLayout(container); vb.setContentsMargins(0,0,0,0); vb.setSpacing(2)
            lb = make_label(label_text, color=ACCENT, bold=True); lb.setToolTip(tooltip)
            vb.addWidget(lb); vb.addWidget(line)
            browse = make_btn("Browse...", lambda _, ft=file_type, le=line: self._browse_path(le, ft))
            vb.addWidget(browse)
            return None, container
        else:
            return None, None

        w.setToolTip(tooltip)
        self.widgets[key] = w
        return lbl, w

    def _add_to_form(self, form_layout, key):
        lbl, w = self._make_widget(key)
        if w: form_layout.addRow(lbl, w) if lbl else form_layout.addRow(w)

    def _sync_widget(self, key):
        w = self.widgets.get(key)
        if not w: return
        if isinstance(w, QtWidgets.QLineEdit): self.current_config[key] = w.text().strip()
        elif isinstance(w, QtWidgets.QPlainTextEdit): self.current_config[key] = w.toPlainText().strip()
        elif isinstance(w, QtWidgets.QCheckBox): self.current_config[key] = w.isChecked()
        elif isinstance(w, QtWidgets.QComboBox): self.current_config[key] = w.currentText()
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)): self.current_config[key] = w.value()
        elif isinstance(w, LRCurveWidget): self.current_config[key] = w.get_points()
        elif isinstance(w, TimestepHistogramWidget): self.current_config[key] = w.get_allocation()

    def _set_widget(self, key, value):
        w = self.widgets.get(key)
        if not w or value is None: return
        w.blockSignals(True)
        if isinstance(w, QtWidgets.QLineEdit): w.setText(str(value))
        elif isinstance(w, QtWidgets.QPlainTextEdit): w.setPlainText(str(value))
        elif isinstance(w, QtWidgets.QCheckBox): w.setChecked(bool(value))
        elif isinstance(w, QtWidgets.QComboBox): w.setCurrentText(str(value))
        elif isinstance(w, QtWidgets.QDoubleSpinBox): w.setValue(float(value))
        elif isinstance(w, QtWidgets.QSpinBox): w.setValue(int(value))
        w.blockSignals(False)

    def _browse_path(self, entry, file_type):
        start = entry.text() if entry.text() else self.last_browsed_path
        if not os.path.exists(start): start = self.last_browsed_path
        if file_type == "folder": path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", start)
        elif file_type == "file_safetensors": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", start, "Safetensors (*.safetensors)")
        elif file_type == "file_pt": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select State", start, "PyTorch State (*.pt)")
        else: path = ""
        if path:
            entry.setText(path.replace('\\', '/'))
            self.last_browsed_path = os.path.dirname(path) if os.path.isfile(path) else path

    def _form_group(self, title, keys, layout_cls=QtWidgets.QFormLayout):
        gb, _ = group_box(title, layout_cls)
        lay = gb.layout()
        for k in keys: self._add_to_form(lay, k)
        return gb

    # ── UI Setup ──────────────────────────────────────────────────
    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(5, 5, 5, 5)

        title = make_label("AOZORA SDXL Trainer", color=ACCENT, bold=True)
        title.setObjectName("TitleLabel")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        main.addWidget(title)

        self.log_textbox = QtWidgets.QTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setMinimumHeight(200)
        self.log_textbox.setStyleSheet(f"background: {DARK_BG}; color: {TEXT_PRI}; font-family: Consolas;")

        self.tab_view = QtWidgets.QTabWidget()

        # Dataset tab
        ds_widget = QtWidgets.QWidget()
        self._build_dataset_tab(ds_widget)
        ds_scroll = QtWidgets.QScrollArea()
        ds_scroll.setWidgetResizable(True)
        ds_scroll.setWidget(ds_widget)
        self.tab_view.addTab(ds_scroll, "Dataset")

        # Model & Training tab
        mt_widget = QtWidgets.QWidget()
        self._build_model_training_tab(mt_widget)
        mt_scroll = QtWidgets.QScrollArea()
        mt_scroll.setWidgetResizable(True)
        mt_scroll.setWidget(mt_widget)
        self.tab_view.addTab(mt_scroll, "Model && Training Parameters")

        # Live Metrics tab
        self.live_metrics_widget = LiveMetricsWidget()
        self.tab_view.addTab(self.live_metrics_widget, "Live Metrics")

        # Console tab
        console_widget = QtWidgets.QWidget()
        console_layout = QtWidgets.QVBoxLayout(console_widget)
        console_layout.setContentsMargins(15, 15, 15, 15)
        console_layout.addWidget(self.log_textbox, stretch=1)
        console_layout.addWidget(make_btn("Clear Console", self.clear_console_log), stretch=0)
        self.tab_view.addTab(console_widget, "Training Console")

        main.addWidget(self.tab_view)
        self._build_corner_widget()
        self._build_bottom_bar()

    def _build_corner_widget(self):
        corner = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(corner)
        lay.setContentsMargins(10, 5, 10, 5)
        lay.setSpacing(8)

        self.config_dropdown = NoScrollComboBox()
        for name in sorted(self.presets.keys()):
            self.config_dropdown.addItem(name.replace("_", " ").title(), name)
        self.config_dropdown.currentIndexChanged.connect(self.load_selected_config)
        lay.addWidget(self.config_dropdown)
        lay.addWidget(make_btn("Save Config", self.save_config))
        lay.addWidget(make_btn("Save As...", self.save_as_config))
        restore_btn = make_btn("↺", self.restore_defaults)
        restore_btn.setFixedHeight(32)
        restore_btn.setToolTip("Restore Selected Config to Defaults")
        lay.addWidget(restore_btn)
        self.tab_view.setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)

    def _build_bottom_bar(self):
        lay = QtWidgets.QHBoxLayout()
        lay.setContentsMargins(0, 5, 5, 5)

        calc_gb = QtWidgets.QGroupBox()
        calc_gb.setStyleSheet(f"QGroupBox {{ margin-top: 0px; border: 1px solid {BORDER}; border-radius: 6px; padding: 0px 8px; }}")
        calc_lay = QtWidgets.QHBoxLayout(calc_gb)
        calc_lay.setContentsMargins(0, 0, 0, 0)
        calc_lay.setSpacing(10)
        self.optimizer_steps_label = make_label("N/A", color=ACCENT, bold=True, size=12)
        self.epochs_label = make_label("N/A", color=ACCENT, bold=True, size=12)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setStyleSheet(f"border: 1px solid {BORDER};")
        calc_lay.addWidget(make_label("Total Optimizer Steps:"))
        calc_lay.addWidget(self.optimizer_steps_label)
        calc_lay.addWidget(sep)
        calc_lay.addWidget(make_label("Total Epochs:"))
        calc_lay.addWidget(self.epochs_label)
        lay.addWidget(calc_gb)
        lay.addStretch()

        self.start_button = make_btn("Start Training", self.start_training)
        self.start_button.setObjectName("StartButton")
        self.stop_button = make_btn("Stop Training", self.stop_training)
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setVisible(False)
        lay.addWidget(self.start_button)
        lay.addWidget(self.stop_button)
        self.layout().addLayout(lay)

    # ── Dataset Tab ───────────────────────────────────────────────
    def _build_dataset_tab(self, parent):
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(15, 15, 15, 15)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(20)
        top.addWidget(self._form_group("Batching & DataLoaders", ["CACHING_BATCH_SIZE", "NUM_WORKERS"]))
        top.addWidget(self._form_group("Unconditional Dropout", ["UNCONDITIONAL_DROPOUT", "UNCONDITIONAL_DROPOUT_CHANCE"]))
        top.addWidget(self._form_group("Aspect Ratio Bucketing", ["TARGET_PIXEL_AREA", "SHOULD_UPSCALE", "MAX_AREA_TOLERANCE"]))
        layout.addLayout(top)

        self.dataset_manager = DatasetManagerWidget(self)
        self.dataset_manager.datasetsChanged.connect(self._update_training_calculations)
        self.dataset_manager.datasetsChanged.connect(self._update_epoch_markers_on_graph)
        layout.addWidget(self.dataset_manager)

        # Conditional enables
        if "SHOULD_UPSCALE" in self.widgets and "MAX_AREA_TOLERANCE" in self.widgets:
            self.widgets["SHOULD_UPSCALE"].stateChanged.connect(
                lambda s: self.widgets["MAX_AREA_TOLERANCE"].setEnabled(bool(s)))
        if "UNCONDITIONAL_DROPOUT" in self.widgets and "UNCONDITIONAL_DROPOUT_CHANCE" in self.widgets:
            chk = self.widgets["UNCONDITIONAL_DROPOUT"]
            chk.stateChanged.connect(lambda s: self.widgets["UNCONDITIONAL_DROPOUT_CHANCE"].setEnabled(bool(s)))

    # ── Model & Training Tab ──────────────────────────────────────
    def _build_model_training_tab(self, parent):
        lay = QtWidgets.QVBoxLayout(parent)
        lay.setSpacing(15)
        lay.setContentsMargins(15, 5, 15, 15)

        # Row 1: Training Mode + Paths | LR Scheduler | Timestep
        r1 = QtWidgets.QHBoxLayout(); r1.setSpacing(20)

        col1 = QtWidgets.QWidget()
        col1_lay = QtWidgets.QVBoxLayout(col1); col1_lay.setContentsMargins(0,0,0,0); col1_lay.setSpacing(10)
        mode_gb, mode_lay = group_box("Training Mode", QtWidgets.QHBoxLayout)
        self.training_mode_combo = make_combo(["Standard (SDXL)", "Rectified Flow (SDXL)"])
        self.training_mode_combo.currentTextChanged.connect(self._on_training_mode_changed)
        mode_lay.addWidget(self.training_mode_combo)
        col1_lay.addWidget(mode_gb)
        col1_lay.addWidget(self._build_path_group())
        col1_lay.addStretch()
        r1.addWidget(col1, 1)
        r1.addWidget(self._build_lr_scheduler_group(), 2)
        self.timestep_group = self._build_timestep_group()
        r1.addWidget(self.timestep_group, 2)
        lay.addLayout(r1)

        # Row 2: Core Training | Scheduler Config | VAE + UNet
        r2 = QtWidgets.QHBoxLayout(); r2.setSpacing(20)
        r2.addWidget(self._build_core_training_group(), 1)
        self.scheduler_group = self._build_scheduler_group()
        r2.addWidget(self.scheduler_group, 1)
        right_col = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_col); right_lay.setContentsMargins(0,0,0,0); right_lay.setSpacing(20)
        right_lay.addWidget(self._build_vae_group())
        right_lay.addWidget(self._form_group("UNet Layer Exclusion", ["UNET_EXCLUDE_TARGETS"]))
        r2.addWidget(right_col, 1)
        lay.addLayout(r2)

        # Row 3: Optimizer | Loss | Miscellaneous  <-- MODIFIED
        r3 = QtWidgets.QHBoxLayout(); r3.setSpacing(20)
        r3.addWidget(self._build_optimizer_group(), 1)
        r3.addWidget(self._build_loss_group(), 1)
        r3.addWidget(self._build_advanced_group(), 1) # <-- MOVED: Misc group added here
        lay.addLayout(r3)

        # Row 4: Advanced <-- REMOVED
        # The entire layout for Row 4 has been removed since its contents were moved to Row 3.

        # Wire up recalculation
        self.widgets["MAX_TRAIN_STEPS"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["MAX_TRAIN_STEPS"].textChanged.connect(self._update_training_calculations)
        self.widgets["GRADIENT_ACCUMULATION_STEPS"].textChanged.connect(self._update_epoch_markers_on_graph)
        self.widgets["GRADIENT_ACCUMULATION_STEPS"].textChanged.connect(self._update_training_calculations)
        self.widgets["BATCH_SIZE"].valueChanged.connect(self._update_training_calculations)
        self._update_lr_button_states(-1)

    def _build_path_group(self):
        gb, lay = group_box("File & Directory Paths", QtWidgets.QVBoxLayout)
        form = QtWidgets.QFormLayout()
        self.model_load_strategy_combo = make_combo(["Load Base Model", "Resume from Checkpoint"])
        form.addRow("Mode:", self.model_load_strategy_combo)
        lay.addLayout(form)

        self.path_stacked_widget = QtWidgets.QStackedWidget()

        # Base model page
        base_page = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(base_page); bl.setContentsMargins(0, 5, 0, 0); bl.setSpacing(10)
        for key in ["SINGLE_FILE_CHECKPOINT_PATH", "VAE_PATH"]:
            _, w = self._make_widget(key)
            if w: bl.addWidget(w)
        self.path_stacked_widget.addWidget(base_page)

        # Resume page
        resume_page = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(resume_page); rl.setContentsMargins(0, 5, 0, 0); rl.setSpacing(10)
        for key in ["RESUME_MODEL_PATH", "RESUME_STATE_PATH"]:
            _, w = self._make_widget(key)
            if w: rl.addWidget(w)
        self.path_stacked_widget.addWidget(resume_page)

        lay.addWidget(self.path_stacked_widget)

        # Output dir
        _, out_w = self._make_widget("OUTPUT_DIR")
        if out_w: lay.addWidget(out_w)

        self.model_load_strategy_combo.currentIndexChanged.connect(self.path_stacked_widget.setCurrentIndex)
        return gb

    def _build_core_training_group(self):
        gb, lay = group_box("Core Training Parameters", QtWidgets.QFormLayout)
        for key in ["MAX_TRAIN_STEPS", "BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "SAVE_EVERY_N_STEPS",
                    "MIXED_PRECISION", "CLIP_GRAD_NORM", "SEED"]:
            self._add_to_form(lay, key)
        # RF shift factor container
        self.rf_params_container = QtWidgets.QWidget()
        rf_lay = QtWidgets.QFormLayout(self.rf_params_container)
        rf_lay.setContentsMargins(0, 0, 0, 0)
        self._add_to_form(rf_lay, "RF_SHIFT_FACTOR")
        lay.addRow(self.rf_params_container)
        return gb

    def _build_lr_scheduler_group(self):
        gb, lay = group_box("Learning Rate Scheduler")
        self.lr_curve_widget = LRCurveWidget()
        self.widgets['LR_CUSTOM_CURVE'] = self.lr_curve_widget
        self.lr_curve_widget.pointsChanged.connect(lambda pts: self._sync_widget("LR_CUSTOM_CURVE"))
        self.lr_curve_widget.selectionChanged.connect(self._update_lr_button_states)
        lay.addWidget(self.lr_curve_widget)

        btn_row = QtWidgets.QHBoxLayout()
        self.add_point_btn = make_btn("Add Point", self.lr_curve_widget.add_point)
        self.remove_point_btn = make_btn("Remove Selected", self.lr_curve_widget.remove_selected_point)
        btn_row.addWidget(self.add_point_btn); btn_row.addWidget(self.remove_point_btn); btn_row.addStretch()
        lay.addLayout(btn_row)

        # Presets
        pg = QtWidgets.QWidget()
        pgrid = QtWidgets.QGridLayout(pg); pgrid.setContentsMargins(0,0,0,0); pgrid.setSpacing(8)
        pgrid.addWidget(make_label("Restarts:"), 0, 0)
        restarts_spin = NoScrollSpinBox(); restarts_spin.setRange(1, 50); restarts_spin.setValue(1)
        pgrid.addWidget(restarts_spin, 0, 1)
        pgrid.addWidget(make_label("Initial Warmup %:"), 1, 0)
        init_warmup = NoScrollDoubleSpinBox(); init_warmup.setRange(0, 100); init_warmup.setValue(5.0); init_warmup.setSuffix("%")
        pgrid.addWidget(init_warmup, 1, 1)
        pgrid.addWidget(make_label("Restart Rampup %:"), 2, 0)
        restart_ramp = NoScrollDoubleSpinBox(); restart_ramp.setRange(0, 100); restart_ramp.setValue(0.0); restart_ramp.setSuffix("%")
        pgrid.addWidget(restart_ramp, 2, 1)
        pgrid.addWidget(make_btn("Apply Cosine", lambda: self.lr_curve_widget.set_generated_preset(
            "Cosine", restarts_spin.value(), init_warmup.value() / 100, restart_ramp.value() / 100)), 3, 0)
        pgrid.addWidget(make_btn("Apply Linear", lambda: self.lr_curve_widget.set_generated_preset(
            "Linear", restarts_spin.value(), init_warmup.value() / 100, restart_ramp.value() / 100)), 3, 1)
        lay.addWidget(pg)

        bounds_lay = QtWidgets.QFormLayout()
        self._add_to_form(bounds_lay, "LR_GRAPH_MIN")
        self._add_to_form(bounds_lay, "LR_GRAPH_MAX")
        lay.addLayout(bounds_lay)
        self.widgets["LR_GRAPH_MIN"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["LR_GRAPH_MAX"].textChanged.connect(self._update_and_clamp_lr_graph)
        return gb

    def _build_scheduler_group(self):
        gb, lay = group_box("Scheduler Configuration", QtWidgets.QFormLayout)
        
        # Add a static label instead of the dropdown
        scheduler_label = make_label("DDPMScheduler", color=TEXT_SEC)
        scheduler_label.setToolTip("Hardcoded in train.py for Standard SDXL mode.")
        lay.addRow(make_label("Noise Scheduler:"), scheduler_label)

        # Add the remaining widgets
        for key in ["PREDICTION_TYPE", "BETA_SCHEDULE", "USE_ZERO_TERMINAL_SNR"]:
            self._add_to_form(lay, key)
        return gb

    def _build_loss_group(self):
        gb, lay = group_box("Loss Configuration")
        form = QtWidgets.QFormLayout()
        self.widgets["LOSS_TYPE"] = make_combo(["MSE", "Semantic", "Huber_Adaptive"])
        self.widgets["LOSS_TYPE"].currentTextChanged.connect(self._toggle_loss_widgets)
        form.addRow("Loss Type:", self.widgets["LOSS_TYPE"])
        lay.addLayout(form)

        self.semantic_loss_container = QtWidgets.QWidget()
        sl = QtWidgets.QFormLayout(self.semantic_loss_container); sl.setContentsMargins(0, 0, 0, 0)
        for key in ["SEMANTIC_SEP_WEIGHT", "SEMANTIC_DETAIL_WEIGHT", "SEMANTIC_ENTROPY_WEIGHT"]:
            self._add_to_form(sl, key)
        lay.addWidget(self.semantic_loss_container)

        self.huber_loss_container = QtWidgets.QWidget()
        hl = QtWidgets.QFormLayout(self.huber_loss_container); hl.setContentsMargins(0, 0, 0, 0)
        for key in ["LOSS_HUBER_BETA", "LOSS_ADAPTIVE_DAMPING"]:
            self._add_to_form(hl, key)
        lay.addWidget(self.huber_loss_container)
        lay.addStretch(1)
        return gb

    def _toggle_loss_widgets(self):
        lt = self.widgets["LOSS_TYPE"].currentText()
        self.semantic_loss_container.setVisible(lt == "Semantic")
        self.huber_loss_container.setVisible(lt == "Huber_Adaptive")

    def _build_optimizer_group(self):
        gb, lay = group_box("Optimizer")
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(make_label("Optimizer Type:"))
        self.widgets["OPTIMIZER_TYPE"] = make_combo([])
        self.widgets["OPTIMIZER_TYPE"].addItem("Raven: Balanced (~12GB VRAM)", "raven")
        self.widgets["OPTIMIZER_TYPE"].addItem("Titan: VRAM Savings (~6GB VRAM, Slower)", "titan")
        self.widgets["OPTIMIZER_TYPE"].addItem("VeloRMS: CPU Offload (Experimental)", "velorms")
        self.widgets["OPTIMIZER_TYPE"].currentIndexChanged.connect(self._toggle_optimizer_widgets)
        sel_row.addWidget(self.widgets["OPTIMIZER_TYPE"], 1)
        lay.addLayout(sel_row)

        self.optimizer_settings_group, opt_lay = group_box("Optimizer Settings")
        self.optimizer_stack = QtWidgets.QStackedWidget()
        self.optimizer_stack.addWidget(self._build_adam_optimizer_form("RAVEN"))
        self.optimizer_stack.addWidget(self._build_adam_optimizer_form("TITAN"))
        self.optimizer_stack.addWidget(self._build_velorms_form())
        opt_lay.addWidget(self.optimizer_stack)
        lay.addWidget(self.optimizer_settings_group)
        return gb

    def _build_adam_optimizer_form(self, prefix):
        container = QtWidgets.QWidget()
        lay = QtWidgets.QFormLayout(container); lay.setContentsMargins(0, 5, 0, 0)
        self.widgets[f'{prefix}_betas'] = QtWidgets.QLineEdit()
        self.widgets[f'{prefix}_eps'] = QtWidgets.QLineEdit()
        self.widgets[f'{prefix}_weight_decay'] = make_dspin(0.0, 1.0, step=0.00001, decimals=6)
        self.widgets[f'{prefix}_debias_strength'] = make_dspin(0.0, 1.0, step=0.01, decimals=3)
        self.widgets[f'{prefix}_use_grad_centralization'] = QtWidgets.QCheckBox("Enable Gradient Centralization")
        self.widgets[f'{prefix}_gc_alpha'] = make_dspin(0.0, 1.0, step=0.1, decimals=1)
        gc = self.widgets[f'{prefix}_use_grad_centralization']
        gca = self.widgets[f'{prefix}_gc_alpha']
        gc.stateChanged.connect(lambda s: gca.setEnabled(bool(s)))
        for lbl, key in [("Betas (b1, b2):", f'{prefix}_betas'), ("Epsilon:", f'{prefix}_eps'),
                          ("Weight Decay:", f'{prefix}_weight_decay'), ("Debias Strength:", f'{prefix}_debias_strength')]:
            lay.addRow(lbl, self.widgets[key])
        lay.addRow(gc)
        lay.addRow("GC Alpha:", gca)
        return container

    def _build_velorms_form(self):
        container = QtWidgets.QWidget()
        lay = QtWidgets.QFormLayout(container); lay.setContentsMargins(0, 5, 0, 0)
        self.widgets["VELORMS_momentum"] = make_dspin(0.0, 0.999, step=0.01, decimals=3)
        self.widgets["VELORMS_leak"] = make_dspin(0.0, 1.0, step=0.01, decimals=3)
        self.widgets["VELORMS_weight_decay"] = make_dspin(0.0, 1.0, step=0.00001, decimals=6)
        self.widgets["VELORMS_eps"] = QtWidgets.QLineEdit()
        for lbl, key in [("Momentum:", "VELORMS_momentum"), ("Leak:", "VELORMS_leak"),
                          ("Weight Decay:", "VELORMS_weight_decay"), ("Epsilon:", "VELORMS_eps")]:
            lay.addRow(lbl, self.widgets[key])
        return container

    def _toggle_optimizer_widgets(self):
        idx_map = {"titan": 1, "velorms": 2}
        self.optimizer_stack.setCurrentIndex(
            idx_map.get(self.widgets["OPTIMIZER_TYPE"].currentData(), 0))

    def _build_vae_group(self):
        gb, lay = group_box("VAE Configuration")
        form = QtWidgets.QFormLayout()
        for key in ["VAE_LATENT_CHANNELS", "VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"]:
            self._add_to_form(form, key)
        lay.addLayout(form)
        lay.addWidget(make_label("Presets:", color=ACCENT, bold=True))
        btn_row = QtWidgets.QHBoxLayout()
        for label, args in [("Standard SDXL", (0.0, 0.13025, 4)),
                             ("Flux/NoobAI (32ch)", (0.0760, 0.6043, 32)),
                             ("EQ VAE", (0.1726, 0.1280, 4))]:
            btn_row.addWidget(make_btn(label, lambda _, a=args: self._apply_vae_preset(*a)))
        lay.addLayout(btn_row)
        lay.addWidget(make_separator())
        detect_btn = make_btn("Run Auto-Detect Tool", self.launch_vae_detector)
        detect_btn.setStyleSheet(f"background: {PANEL_BG}; color: {ACCENT}; border: 1px solid {ACCENT};")
        lay.addWidget(detect_btn)
        return gb

    def _apply_vae_preset(self, shift, scale, channels):
        self.widgets["VAE_SHIFT_FACTOR"].setValue(shift)
        self.widgets["VAE_SCALING_FACTOR"].setValue(scale)
        self.widgets["VAE_LATENT_CHANNELS"].setValue(channels)
        self.log(f"Applied VAE Preset: Shift={shift}, Scale={scale}, Ch={channels}")

    def launch_vae_detector(self):
        idx = self.config_dropdown.currentIndex()
        if idx < 0: QtWidgets.QMessageBox.warning(self, "No Config Selected", "Please select a configuration first."); return
        key = self.config_dropdown.itemData(idx)
        cfg_path = os.path.join(self.config_dir, f"{key}.json")
        if not os.path.exists(cfg_path):
            QtWidgets.QMessageBox.critical(self, "Config Not Found", f"Save the configuration first.\n{cfg_path}"); return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target = next((p for p in [os.path.join(script_dir, "vae_diagnostics.py"),
                                    os.path.join(script_dir, "tools", "vae_diagnostics.py")] if os.path.exists(p)), None)
        if not target: self.log("Error: 'vae_diagnostics.py' not found."); return
        try:
            cmd = [sys.executable, target, "--config", os.path.abspath(cfg_path)]
            kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE} if os.name == 'nt' else {}
            subprocess.Popen(cmd, cwd=os.path.dirname(target), **kw)
            self.log(f"Launched VAE Detector with config: {key}.json")
        except Exception as e: self.log(f"Error launching tool: {e}")

    def _build_advanced_group(self):
        gb, lay = group_box("Miscellaneous", QtWidgets.QFormLayout)
        self._add_to_form(lay, "MEMORY_EFFICIENT_ATTENTION")
        lay.addRow(make_separator())
        lay.addRow(make_label("<b>Gradient Spike Detection</b>", color=ACCENT))
        self._add_to_form(lay, "GRAD_SPIKE_THRESHOLD_HIGH")
        self._add_to_form(lay, "GRAD_SPIKE_THRESHOLD_LOW")
        return gb

    # ── Timestep Sampling Group ───────────────────────────────────
    def _build_timestep_group(self):
        gb, lay = group_box("Timestep Ticket Allocation")
        self.timestep_histogram = TimestepHistogramWidget()
        self.widgets["TIMESTEP_ALLOCATION"] = self.timestep_histogram
        self.timestep_histogram.allocationChanged.connect(lambda _: self._sync_widget("TIMESTEP_ALLOCATION"))
        lay.addWidget(self.timestep_histogram)

        # Controls
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(make_label("Bin Size:"))
        self.bin_size_combo = make_combo(["30", "40", "50", "100", "200", "250", "500"])
        self.bin_size_combo.setCurrentText("100")
        r1.addWidget(self.bin_size_combo)
        r1.addSpacing(20)
        r1.addWidget(make_label("Distribution Mode:"))
        self.ts_mode_combo = make_combo(["Wave", "Logit-Normal", "Beta"])
        r1.addWidget(self.ts_mode_combo, 1)
        lay.addLayout(r1)

        # Slider stack
        self.ts_slider_stack = QtWidgets.QStackedWidget()

        # Wave page
        wave_page = QtWidgets.QWidget()
        wl = QtWidgets.QFormLayout(wave_page); wl.setContentsMargins(0, 0, 0, 0)
        self.slider_wave_freq, self.spin_wave_freq = self._add_slider_row(wl, "Frequency:", 0.0, 5.0, 1.0, 100.0)
        self.slider_wave_phase, self.spin_wave_phase = self._add_slider_row(wl, "Phase:", 0.0, 6.28, 0.0, 100.0)
        self.slider_wave_amp, self.spin_wave_amp = self._add_slider_row(wl, "Amplitude:", 0.0, 1.0, 0.0, 100.0)
        self.ts_slider_stack.addWidget(wave_page)

        # Logit-Normal page
        logit_page = QtWidgets.QWidget()
        ll = QtWidgets.QFormLayout(logit_page); ll.setContentsMargins(0, 0, 0, 0)
        self.slider_ln_mu, self.spin_ln_mu = self._add_slider_row(ll, "Center (Mu):", -3.0, 3.0, 0.0, 100.0)
        self.slider_ln_sigma, self.spin_ln_sigma = self._add_slider_row(ll, "Spread (Sigma):", 0.1, 3.0, 1.0, 100.0)
        self.ts_slider_stack.addWidget(logit_page)

        # Beta page
        beta_page = QtWidgets.QWidget()
        bl = QtWidgets.QFormLayout(beta_page); bl.setContentsMargins(0, 0, 0, 0)
        self.slider_beta_alpha, self.spin_beta_alpha = self._add_slider_row(bl, "Alpha:", 0.1, 10.0, 3.0, 100.0)
        self.slider_beta_beta, self.spin_beta_beta = self._add_slider_row(bl, "Beta:", 0.1, 10.0, 3.0, 100.0)
        self.ts_slider_stack.addWidget(beta_page)
        lay.addWidget(self.ts_slider_stack)

        # Preset buttons stack
        self.ts_button_stack = QtWidgets.QStackedWidget()
        presets_by_mode = [
            [("Uniform (Flat)", "Uniform"), ("Peak Ends", "Peak Ends"), ("Peak Middle", "Peak Middle")],
            [("Bell Curve", "Bell Curve"), ("Detail (Early)", "Detail"), ("Structure (Late)", "Structure"),
             ("Logit-Normal (RF/SD3 Recommended)", "Logit-Normal (RF/SD3 Recommended)")],
            [("Symmetric (3,3)", "Beta Symmetric"), ("Right Skew (2,5)", "Beta Right Skew"),
             ("Left Skew (5,2)", "Beta Left Skew"), ("U-Shape (0.5,0.5)", "Beta U-Shape")],
        ]
        for page_presets in presets_by_mode:
            page = QtWidgets.QWidget()
            pgrid = QtWidgets.QGridLayout(page); pgrid.setSpacing(8); pgrid.setContentsMargins(0, 0, 0, 0)
            for i, (label, preset_name) in enumerate(page_presets):
                b = make_btn(label, lambda _, pn=preset_name: self._apply_timestep_preset(pn))
                b.setStyleSheet("padding: 4px; font-size: 8pt;")
                pgrid.addWidget(b, i // 3, i % 3)
            self.ts_button_stack.addWidget(page)
        lay.addWidget(self.ts_button_stack)

        self.bin_size_combo.currentTextChanged.connect(lambda t: (self.timestep_histogram.set_bin_size(int(t)), self._update_timestep_distribution()))
        self.ts_mode_combo.currentIndexChanged.connect(self.ts_slider_stack.setCurrentIndex)
        self.ts_mode_combo.currentIndexChanged.connect(self.ts_button_stack.setCurrentIndex)
        self.ts_mode_combo.currentIndexChanged.connect(lambda _: self._update_timestep_distribution())
        return gb

    def _add_slider_row(self, layout, label_text, min_val, max_val, default_val, divider):
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container); h.setContentsMargins(0, 0, 0, 0)
        lbl = make_label(label_text); lbl.setFixedWidth(90)
        h.addWidget(lbl)
        sl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        sl.setRange(int(min_val * divider), int(max_val * divider))
        sl.setValue(int(default_val * divider))
        sp = NoScrollDoubleSpinBox(); sp.setRange(min_val, max_val); sp.setSingleStep(0.1)
        sp.setValue(default_val); sp.setDecimals(2); sp.setFixedWidth(120)
        sl.valueChanged.connect(lambda v: sp.setValue(v / divider) if abs(sp.value() - v / divider) > 0.5 / divider else None)
        sp.valueChanged.connect(lambda v: (sl.setValue(int(v * divider)), self._update_timestep_distribution()))
        sl.valueChanged.connect(lambda _: self._update_timestep_distribution())
        h.addWidget(sl); h.addWidget(sp)
        layout.addRow(container)
        return sl, sp

    def _apply_timestep_preset(self, name):
        # Block signals during preset application to avoid partial updates
        self.ts_mode_combo.blockSignals(True)

        mode_map = {
            "Uniform":         "Wave",
            "Peak Ends":       "Wave",
            "Peak Middle":     "Wave",
            "Bell Curve":      "Logit-Normal",
            "Detail":          "Logit-Normal",
            "Structure":       "Logit-Normal",
            "Logit-Normal (RF/SD3 Recommended)": "Logit-Normal",
            "Beta Symmetric":  "Beta",
            "Beta Right Skew": "Beta",
            "Beta Left Skew":  "Beta",
            "Beta U-Shape":    "Beta",
        }
        values = {
            "Uniform":         dict(wave_amp=0.0, wave_freq=1.0, wave_phase=0.0),
            "Peak Ends":       dict(wave_freq=1.0, wave_phase=0.0, wave_amp=0.8),
            "Peak Middle":     dict(wave_freq=1.0, wave_phase=3.14, wave_amp=0.6),
            "Bell Curve":      dict(ln_mu=0.0, ln_sigma=1.0),
            "Detail":          dict(ln_mu=-1.0, ln_sigma=0.8),
            "Structure":       dict(ln_mu=1.0, ln_sigma=0.8),
            "Logit-Normal (RF/SD3 Recommended)": dict(ln_mu=-0.5, ln_sigma=1.0),
            "Beta Symmetric":  dict(beta_alpha=3.0, beta_beta=3.0),
            "Beta Right Skew": dict(beta_alpha=2.0, beta_beta=5.0),
            "Beta Left Skew":  dict(beta_alpha=5.0, beta_beta=2.0),
            "Beta U-Shape":    dict(beta_alpha=0.5, beta_beta=0.5),
        }
        if name not in mode_map:
            self.ts_mode_combo.blockSignals(False)
            return

        target_mode = mode_map[name]
        vals = values[name]

        # Set mode index directly
        mode_idx = {"Wave": 0, "Logit-Normal": 1, "Beta": 2}.get(target_mode, 0)
        self.ts_mode_combo.setCurrentIndex(mode_idx)
        self.ts_slider_stack.setCurrentIndex(mode_idx)
        self.ts_button_stack.setCurrentIndex(mode_idx)
        self.ts_mode_combo.blockSignals(False)

        # Apply spin values and sync sliders - block all signals first
        slider_spin_pairs = [
            (self.slider_wave_freq, self.spin_wave_freq, "wave_freq", 100.0),
            (self.slider_wave_phase, self.spin_wave_phase, "wave_phase", 100.0),
            (self.slider_wave_amp, self.spin_wave_amp, "wave_amp", 100.0),
            (self.slider_ln_mu, self.spin_ln_mu, "ln_mu", 100.0),
            (self.slider_ln_sigma, self.spin_ln_sigma, "ln_sigma", 100.0),
            (self.slider_beta_alpha, self.spin_beta_alpha, "beta_alpha", 100.0),
            (self.slider_beta_beta, self.spin_beta_beta, "beta_beta", 100.0),
        ]
        for sl, sp, key, divider in slider_spin_pairs:
            sl.blockSignals(True); sp.blockSignals(True)
            if key in vals:
                sp.setValue(vals[key])
                sl.setValue(int(vals[key] * divider))
            sl.blockSignals(False); sp.blockSignals(False)

        # Force a single distribution update
        self._update_timestep_distribution()

    def _update_timestep_distribution(self):
        mode = self.ts_mode_combo.currentText()
        n = max(math.ceil(1000 / int(self.bin_size_combo.currentText())), 1)
        weights = []

        if mode == "Wave":
            freq, phase, amp = self.spin_wave_freq.value(), self.spin_wave_phase.value(), self.spin_wave_amp.value()
            for i in range(n):
                t = i / max(1, n - 1)
                weights.append(max(0.0, 1.0 + amp * math.cos(2 * math.pi * freq * t + phase)))
        elif mode == "Logit-Normal":
            mu, sigma = self.spin_ln_mu.value(), self.spin_ln_sigma.value()
            bin_size = int(self.bin_size_combo.currentText())
            def logit(p): return math.log(p / (1 - p))
            def ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            for i in range(n):
                t_s, t_e = i * bin_size, min((i + 1) * bin_size, 1000)
                eps = 1e-6
                w = ncdf((logit(min(t_e / 1000, 1 - eps)) - mu) / sigma) - ncdf((logit(max(t_s / 1000, eps)) - mu) / sigma)
                weights.append(max(0.0, w))
        elif mode == "Beta":
            alpha, beta = self.spin_beta_alpha.value(), self.spin_beta_beta.value()
            bin_size = int(self.bin_size_combo.currentText())
            for i in range(n):
                x = max(1e-4, min(1 - 1e-4, ((i * bin_size) + bin_size / 2) / 1000))
                weights.append(max(0.0, x ** (alpha - 1) * (1 - x) ** (beta - 1)))

        self.timestep_histogram.generate_from_weights(weights)

    # ── Training Mode ─────────────────────────────────────────────
    def _on_training_mode_changed(self, text):
        is_rf = "Rectified Flow" in text
        # The problematic 'for' loop and 'if' check for noise_group have been removed.
        if hasattr(self, 'scheduler_group'): 
            self.scheduler_group.setEnabled(not is_rf)
        if hasattr(self, 'rf_params_container'): 
            self.rf_params_container.setVisible(is_rf)
        if hasattr(self, 'dataset_manager'): 
            self.dataset_manager.refresh_cache_buttons()

    # ── Config Apply / Collect ────────────────────────────────────
    def _apply_config_to_widgets(self):
        for w in self.widgets.values(): w.blockSignals(True)
        try:
            mode = self.current_config.get("TRAINING_MODE", "Standard (SDXL)")
            self.training_mode_combo.blockSignals(True)
            self.training_mode_combo.setCurrentText(mode)
            self.training_mode_combo.blockSignals(False)
            self._on_training_mode_changed(mode)

            is_resuming = self.current_config.get("RESUME_TRAINING", False)
            self.model_load_strategy_combo.setCurrentIndex(1 if is_resuming else 0)

            # Standard widget keys
            skip = {"OPTIMIZER_TYPE", "LR_CUSTOM_CURVE", "NOISE_TYPE", "LOSS_TYPE", "TIMESTEP_ALLOCATION"}
            skip |= {k for k in self.widgets if k.startswith(("RAVEN_", "TITAN_", "VELORMS_"))}
            for key, w in self.widgets.items():
                if key in skip: continue
                self._set_widget(key, self.current_config.get(key))

            # Optimizer
            opt_type = self.current_config.get("OPTIMIZER_TYPE", default_config.OPTIMIZER_TYPE).lower()
            idx = self.widgets["OPTIMIZER_TYPE"].findData(opt_type)
            self.widgets["OPTIMIZER_TYPE"].setCurrentIndex(idx if idx >= 0 else 0)

            # Load adam param groups
            for prefix, def_attr in [("RAVEN", "RAVEN_PARAMS"), ("TITAN", "TITAN_PARAMS")]:
                defaults = getattr(default_config, def_attr, {})
                params = {**defaults, **self.current_config.get(def_attr, {})}
                self.widgets[f"{prefix}_betas"].setText(', '.join(map(str, params.get("betas", [0.9, 0.999]))))
                self.widgets[f"{prefix}_eps"].setText(str(params.get("eps", 1e-8)))
                self.widgets[f"{prefix}_weight_decay"].setValue(params.get("weight_decay", 0.01))
                self.widgets[f"{prefix}_debias_strength"].setValue(params.get("debias_strength", 1.0))
                self.widgets[f"{prefix}_use_grad_centralization"].setChecked(params.get("use_grad_centralization", False))
                self.widgets[f"{prefix}_gc_alpha"].setValue(params.get("gc_alpha", 1.0))
                self.widgets[f"{prefix}_gc_alpha"].setEnabled(params.get("use_grad_centralization", False))

            vp = {**getattr(default_config, "VELORMS_PARAMS", {}), **self.current_config.get("VELORMS_PARAMS", {})}
            self.widgets["VELORMS_momentum"].setValue(vp.get("momentum", 0.86))
            self.widgets["VELORMS_leak"].setValue(vp.get("leak", 0.16))
            self.widgets["VELORMS_weight_decay"].setValue(vp.get("weight_decay", 0.01))
            self.widgets["VELORMS_eps"].setText(str(vp.get("eps", 1e-8)))
            self._toggle_optimizer_widgets()

            # Loss
            self.widgets["LOSS_TYPE"].setCurrentText(self.current_config.get("LOSS_TYPE", "MSE"))
            for key, default in [("SEMANTIC_SEP_WEIGHT", 0.5), ("SEMANTIC_DETAIL_WEIGHT", 0.8),
                                  ("SEMANTIC_ENTROPY_WEIGHT", 0.8), ("LOSS_HUBER_BETA", 0.5), ("LOSS_ADAPTIVE_DAMPING", 0.1)]:
                self.widgets[key].setValue(self.current_config.get(key, default))
            self._toggle_loss_widgets()

            # Conditional enables
            if "SHOULD_UPSCALE" in self.widgets:
                self.widgets["MAX_AREA_TOLERANCE"].setEnabled(self.widgets["SHOULD_UPSCALE"].isChecked())
            if "UNCONDITIONAL_DROPOUT" in self.widgets:
                self.widgets["UNCONDITIONAL_DROPOUT_CHANCE"].setEnabled(self.widgets["UNCONDITIONAL_DROPOUT"].isChecked())

            # LR curve
            self._update_and_clamp_lr_graph()

            # Timestep histogram
            alloc = self.current_config.get("TIMESTEP_ALLOCATION")
            if alloc:
                if "bin_size" in alloc:
                    self.bin_size_combo.blockSignals(True)
                    self.bin_size_combo.setCurrentText(str(alloc["bin_size"]))
                    self.bin_size_combo.blockSignals(False)
                self.timestep_histogram.set_allocation(alloc)
            ts_mode = self.current_config.get("TIMESTEP_MODE", "Wave")
            self.ts_mode_combo.blockSignals(True)
            self.ts_mode_combo.setCurrentText(ts_mode)
            self.ts_mode_combo.blockSignals(False)

            # Datasets
            if hasattr(self, 'dataset_manager'):
                self.dataset_manager.load_datasets_from_config(self.current_config.get("INSTANCE_DATASETS", []))
            self._update_training_calculations()
        finally:
            for w in self.widgets.values(): w.blockSignals(False)

    def _collect_config(self):
        cfg = {}
        skip_keys = {"RESUME_TRAINING", "INSTANCE_DATASETS", "OPTIMIZER_TYPE",
                     "RAVEN_PARAMS", "TITAN_PARAMS", "VELORMS_PARAMS",
                     "NOISE_TYPE", "NOISE_OFFSET", "LOSS_TYPE",
                     "SEMANTIC_SEP_WEIGHT", "SEMANTIC_DETAIL_WEIGHT", "SEMANTIC_ENTROPY_WEIGHT",
                     "LOSS_HUBER_BETA", "LOSS_ADAPTIVE_DAMPING", "TIMESTEP_ALLOCATION", "TIMESTEP_WEIGHTING_CURVE"}

        for key, val in self.current_config.items():
            if key in skip_keys: continue
            if val is None: continue
            cfg[key] = [[round(p[0], 8), round(p[1], 10)] for p in val] if key == "LR_CUSTOM_CURVE" else val

        cfg["TRAINING_MODE"] = self.training_mode_combo.currentText()
        cfg["RESUME_TRAINING"] = self.model_load_strategy_combo.currentIndex() == 1
        cfg["INSTANCE_DATASETS"] = self.dataset_manager.get_datasets_config()
        cfg["OPTIMIZER_TYPE"] = self.widgets["OPTIMIZER_TYPE"].currentData()
        cfg["LOSS_TYPE"] = self.widgets["LOSS_TYPE"].currentText()
        for key in ["SEMANTIC_SEP_WEIGHT", "SEMANTIC_DETAIL_WEIGHT", "SEMANTIC_ENTROPY_WEIGHT",
                    "LOSS_HUBER_BETA", "LOSS_ADAPTIVE_DAMPING"]:
            cfg[key] = self.widgets[key].value()
        cfg["TIMESTEP_MODE"] = self.ts_mode_combo.currentText()
        if hasattr(self, 'timestep_histogram'): cfg["TIMESTEP_ALLOCATION"] = self.timestep_histogram.get_allocation()

        # Optimizer params
        for prefix, key in [("RAVEN", "RAVEN_PARAMS"), ("TITAN", "TITAN_PARAMS")]:
            try:
                betas = [float(x) for x in self.widgets[f"{prefix}_betas"].text().split(',')]
            except: betas = [0.9, 0.999]
            try: eps = float(self.widgets[f"{prefix}_eps"].text())
            except: eps = 1e-8
            cfg[key] = {
                "betas": betas, "eps": eps,
                "weight_decay": self.widgets[f"{prefix}_weight_decay"].value(),
                "debias_strength": self.widgets[f"{prefix}_debias_strength"].value(),
                "use_grad_centralization": self.widgets[f"{prefix}_use_grad_centralization"].isChecked(),
                "gc_alpha": self.widgets[f"{prefix}_gc_alpha"].value(),
            }
        try: veps = float(self.widgets["VELORMS_eps"].text())
        except: veps = 1e-8
        cfg["VELORMS_PARAMS"] = {
            "momentum": self.widgets["VELORMS_momentum"].value(),
            "leak": self.widgets["VELORMS_leak"].value(),
            "weight_decay": self.widgets["VELORMS_weight_decay"].value(),
            "eps": veps,
        }
        for key in ["VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"]:
            cfg[key] = self.widgets[key].value()
        cfg["VAE_LATENT_CHANNELS"] = self.widgets["VAE_LATENT_CHANNELS"].value()
        if "RF_SHIFT_FACTOR" in self.widgets: cfg["RF_SHIFT_FACTOR"] = self.widgets["RF_SHIFT_FACTOR"].value()
        return cfg

    # ── Calculations & Graph Updates ──────────────────────────────
    def _update_training_calculations(self):
        try:
            max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
            grad_accum = int(self.widgets["GRADIENT_ACCUMULATION_STEPS"].text())
            batch = self.widgets["BATCH_SIZE"].value()
            total_images = self.dataset_manager.get_total_repeats()
            opt_steps = max_steps // grad_accum if grad_accum else 0
            steps_per_epoch = total_images // batch if total_images and batch else 0
            epochs = max_steps / steps_per_epoch if steps_per_epoch else float('inf')
            self.optimizer_steps_label.setText(f"{opt_steps:,}")
            self.epochs_label.setText("∞" if epochs == float('inf') else f"{epochs:.2f}")
            if hasattr(self, 'timestep_histogram'): self.timestep_histogram.set_total_steps(max_steps)
        except (ValueError, KeyError):
            self.optimizer_steps_label.setText("Invalid"); self.epochs_label.setText("Invalid")

    def _update_and_clamp_lr_graph(self):
        if not hasattr(self, 'lr_curve_widget'): return
        try: steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except: steps = 1
        try: min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except: min_lr = 0.0
        try: max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
        except: max_lr = 1e-6
        if "LR_CUSTOM_CURVE" in self.current_config:
            self.current_config["LR_CUSTOM_CURVE"] = [
                [p[0], max(min_lr, min(max_lr, p[1]))] for p in self.current_config["LR_CUSTOM_CURVE"]]
        self.lr_curve_widget.set_bounds(steps, min_lr, max_lr)
        self.lr_curve_widget.set_points(self.current_config.get("LR_CUSTOM_CURVE", []))
        self._update_epoch_markers_on_graph()

    def _update_epoch_markers_on_graph(self):
        if not hasattr(self, 'lr_curve_widget') or not hasattr(self, 'dataset_manager'): return
        try:
            total = self.dataset_manager.get_total_repeats()
            max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except: self.lr_curve_widget.set_epoch_data([]); return
        data = []
        if total > 0 and max_steps > 0:
            step = total
            while step < max_steps: data.append((step / max_steps, int(step))); step += total
        self.lr_curve_widget.set_epoch_data(data)

    def _update_lr_button_states(self, sel_idx):
        if hasattr(self, 'remove_point_btn'):
            removable = 0 < sel_idx < len(self.lr_curve_widget.get_points()) - 1
            self.remove_point_btn.setEnabled(removable)

    # ── Training Control ──────────────────────────────────────────
    def get_current_cache_folder_name(self):
        return (".precomputed_embeddings_cache_rf_noobai"
                if "Rectified Flow" in self.training_mode_combo.currentText()
                else ".precomputed_embeddings_cache")

    def start_training(self):
        self.save_config()
        self.log("\n" + "=" * 50 + "\nStarting training process...\n" + "=" * 50)
        key = self.config_dropdown.itemData(self.config_dropdown.currentIndex()) or ""
        config_path = os.path.abspath(os.path.join(self.config_dir, f"{key}.json"))
        if not os.path.exists(config_path): self.log(f"CRITICAL: Config not found: {config_path}"); return
        train_py_path = os.path.abspath("train.py")
        if not os.path.exists(train_py_path): self.log(f"CRITICAL: train.py not found."); return

        self.start_button.setVisible(False); self.stop_button.setVisible(True)
        self.live_metrics_widget.clear_data()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        env = os.environ.copy()
        python_dir = os.path.dirname(sys.executable)
        env["PATH"] = f"{python_dir};{os.path.join(python_dir, 'Scripts')};{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{script_dir};{env.get('PYTHONPATH', '')}"

        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        self.process_runner = ProcessRunner(sys.executable, ["-u", train_py_path, "--config", config_path],
                                            script_dir, env, flags)
        self.process_runner.logSignal.connect(self.log)
        self.process_runner.progressSignal.connect(self.handle_process_output)
        self.process_runner.finishedSignal.connect(self.training_finished)
        self.process_runner.errorSignal.connect(self.log)
        self.process_runner.metricsSignal.connect(self.live_metrics_widget.parse_and_update)
        self.process_runner.cacheCreatedSignal.connect(self.dataset_manager.refresh_cache_buttons)
        if os.name == 'nt': prevent_sleep(True)
        self.process_runner.start()
        self.log(f"INFO: Starting train.py with config: {config_path}")

    def stop_training(self):
        if self.process_runner and self.process_runner.isRunning(): self.process_runner.stop()
        else: self.log("No active training process to stop.")

    def training_finished(self, exit_code=0):
        if self.process_runner: self.process_runner.quit(); self.process_runner.wait(); self.process_runner = None
        status = "successfully" if exit_code == 0 else f"with error (Code: {exit_code})"
        self.log("\n" + "=" * 50 + f"\nTraining finished {status}.\n" + "=" * 50)
        self.start_button.setVisible(True); self.stop_button.setVisible(False)
        if hasattr(self, 'dataset_manager'): self.dataset_manager.refresh_cache_buttons()
        if os.name == 'nt': prevent_sleep(False)

    # ── Logging ───────────────────────────────────────────────────
    def log(self, message): self.append_log(message.strip(), replace=False)

    def append_log(self, text, replace=False):
        sb = self.log_textbox.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4
        cur = self.log_textbox.textCursor()
        cur.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if replace:
            cur.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
            cur.removeSelectedText()
            cur.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.log_textbox.setTextCursor(cur)
        self.log_textbox.insertPlainText(text.rstrip() + '\n')
        if at_bottom: sb.setValue(sb.maximum())

    def handle_process_output(self, text, is_progress):
        if text:
            self.append_log(text, replace=is_progress and self.last_line_is_progress)
            self.last_line_is_progress = is_progress

    def clear_console_log(self): self.log_textbox.clear(); self.log("Console cleared.")

    # ── Window Events ─────────────────────────────────────────────
    def closeEvent(self, event):
        self._save_gui_state()
        if self.process_runner and self.process_runner.isRunning():
            reply = QtWidgets.QMessageBox.question(self, "Training in Progress",
                                                   "Training is running. Stop it and exit?",
                                                   QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.stop_training()
                if self.process_runner: self.process_runner.quit(); self.process_runner.wait(2000)
                if os.name == 'nt': prevent_sleep(False)
                event.accept()
            else:
                event.ignore(); return
        if hasattr(self, 'dataset_manager'):
            for t in self.dataset_manager.loader_threads:
                if t.isRunning(): t.quit(); t.wait(1000)
        if os.name == 'nt': prevent_sleep(False)
        event.accept()

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)


# ──────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = TrainingGUI()
    win.show()
    sys.exit(app.exec())