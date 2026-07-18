"""Central theme primitives for the training GUI.

Widgets should consume semantic roles from ``THEME`` instead of embedding colors.
Keeping the palette and QSS here makes visual changes cheap and prevents custom
painted controls from drifting away from standard Qt widgets.
"""

from dataclasses import dataclass

from PyQt6 import QtGui, QtWidgets


@dataclass(frozen=True, slots=True)
class Theme:
    # Surfaces: deliberately close in value, from application to raised control.
    window: str = "#11151c"
    nested_group: str = "#11141d"
    deeply_nested: str = "#0b0e14"
    surface: str = "#12161e"
    surface_raised: str = "#181d27"
    surface_hover: str = "#202633"
    border: str = "#2b3242"
    border_muted: str = "#1b202b"

    # Content and interaction.
    text: str = "#e6e9f0"
    text_muted: str = "#8991a7"
    text_disabled: str = "#50586b"
    accent: str = "#7c6af7"
    accent_hover: str = "#9585ff"
    accent_deep: str = "#5e4bc4"
    accent_alt: str = "#43caca"
    danger: str = "#f05b72"
    danger_hover: str = "#ff7187"
    success: str = "#49d98a"
    warning: str = "#e8bd4c"

    @property
    def chart(self) -> str:
        """Base group surface."""
        return self.surface

    @property
    def canvas(self) -> str:
        """Nested panels and custom-painted chart canvases."""
        return self.nested_group

    def color(self, role: str) -> QtGui.QColor:
        return QtGui.QColor(getattr(self, role))


THEME = Theme()


def set_role(widget: QtWidgets.QWidget, role: str) -> QtWidgets.QWidget:
    """Assign a QSS role without installing a per-widget stylesheet."""
    widget.setProperty("uiRole", role)
    return widget


def make_stylesheet(theme: Theme = THEME) -> str:
    t = theme
    return f"""
QWidget {{
    background-color: transparent;
    color: {t.text};
    font-family: 'Consolas', 'Segoe UI', monospace;
    font-size: 10pt;
}}
#TrainingGUI, QDialog, QMessageBox {{ background-color: {t.window}; }}
QLabel {{ background-color: transparent; }}
QWidget[uiRole="transparent"], QWidget[uiRole="chartOverlay"] {{ background: transparent; }}
QWidget[uiRole="panel"] {{ background: {t.canvas}; border: 1px solid {t.border}; border-radius: 4px; }}
QWidget[uiRole="deepPanel"] {{ background: {t.deeply_nested}; border: 1px solid {t.border}; border-radius: 4px; }}
QWidget[uiRole="navigation"] {{ background: {t.window}; border-bottom: 1px solid {t.border}; }}
QFrame[uiRole="mainFrame"] {{ background: {t.window}; border: 1px solid {t.border}; border-top: none; }}
QWidget[uiRole="footer"] {{ background: {t.window}; border-top: 1px solid {t.border}; }}
QLabel[uiRole="preview"] {{ background: {t.canvas}; border: 1px solid {t.border}; }}
QListWidget[uiRole="quickFocus"] {{ background: {t.surface}; }}
QWidget[uiRole="datasetEntry"] {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 5px; }}
QWidget[uiRole="datasetEntry"][selected="true"] {{ background: {t.surface_hover}; border-color: {t.accent}; }}

QGroupBox {{
    background-color: {t.chart};
    border: 1px solid {t.border};
    border-radius: 6px;
    margin-top: 16px;
    padding: 10px 8px 8px 8px;
    color: {t.text_muted};
    font-size: 8pt;
}}
QGroupBox[uiRole="section"] {{ background-color: {t.surface}; }}
QGroupBox[uiRole="flat"] {{ margin-top: 0; padding: 0; }}
QGroupBox QGroupBox {{ background-color: {t.nested_group}; }}
QGroupBox[groupSurface="inner"] {{ background-color: #11141d; }}
QGroupBox[density="compact"] {{ font-size: 8pt; }}
QGroupBox[density="compact"] QPushButton {{ padding: 3px 6px; min-height: 22px; }}
QGroupBox[density="compact"] QLineEdit,
QGroupBox[density="compact"] QSpinBox,
QGroupBox[density="compact"] QDoubleSpinBox {{ padding: 3px 5px; min-height: 20px; }}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {t.accent};
    background-color: transparent;
    font-weight: bold;
    font-size: 10pt;
}}
QGroupBox[nested="true"]::title {{ background-color: transparent; }}
QGroupBox[density="compact"]::title {{ font-size: 8pt; }}
QGroupBox:disabled {{ border-color: {t.border_muted}; }}
QGroupBox:disabled::title {{ color: {t.text_disabled}; }}

QPushButton {{
    background: {t.surface_raised}; border: 1px solid {t.border}; border-radius: 4px;
    color: {t.text}; padding: 6px 14px; min-height: 28px; max-height: 36px;
}}
QPushButton:hover {{ border-color: {t.accent}; color: {t.accent}; }}
QPushButton:pressed, QPushButton:checked {{ background: {t.accent}; color: white; }}
QPushButton:disabled {{ color: {t.text_disabled}; background: {t.window}; border-color: {t.border_muted}; }}
QPushButton[uiRole="icon"] {{ padding: 0; min-width: 24px; max-width: 32px; font-weight: bold; }}
QPushButton[uiRole="compact"] {{ padding: 4px; font-size: 8pt; }}
QPushButton[uiRole="segmentLeft"] {{ padding: 3px 8px; border-top-right-radius: 0; border-bottom-right-radius: 0; }}
QPushButton[uiRole="segmentRight"] {{
    padding: 3px; border-left: none; border-top-left-radius: 0; border-bottom-left-radius: 0;
    background: {t.accent}; color: white; font-weight: bold;
}}
QPushButton[uiRole="segmentRight"]:hover {{ background: {t.accent_hover}; color: white; }}
QPushButton[uiRole="accent"], #StartButton {{ background: {t.accent}; color: white; border-color: {t.accent}; font-weight: bold; }}
QPushButton[uiRole="accent"]:hover, #StartButton:hover {{ background: {t.accent_hover}; color: white; }}
QPushButton[uiRole="danger"], #StopButton {{ background: {t.danger}; color: white; border-color: {t.danger}; font-weight: bold; }}
QPushButton[uiRole="danger"]:hover, #StopButton:hover {{ background: {t.danger_hover}; color: white; }}
QPushButton[uiRole="warning"] {{ background: {t.warning}; color: {t.window}; border-color: {t.warning}; font-weight: bold; }}
#FollowOutputButton:checked {{ background: {t.accent}; color: white; border-color: {t.accent}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {t.surface_raised}; border: 1px solid {t.border}; border-radius: 4px;
    padding: 5px 8px; color: {t.text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:on,
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {t.accent}; }}
QPlainTextEdit, QTextEdit {{ font-family: 'Consolas', monospace; padding: 4px; }}
QPlainTextEdit[uiRole="consoleText"] {{ background: #11141d; color: {t.text}; }}
QTextEdit[uiRole="caption"] {{ background: transparent; border: none; font-size: 9pt; }}
QComboBox {{ min-height: 28px; max-height: 36px; }}
QComboBox::drop-down {{ border-left: 1px solid {t.border}; width: 20px; }}
QComboBox QAbstractItemView {{ background: {t.surface_raised}; border: 1px solid {t.accent}; selection-background-color: {t.accent}; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 28px; min-height: 24px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 26px; border-left: 1px solid {t.border}; background: {t.surface_raised};
}}
QSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 4px; }}
QSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 4px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {t.surface_hover}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 4px solid {t.text}; width: 0; height: 0; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 4px solid {t.text}; width: 0; height: 0; }}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled, QComboBox:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {t.text_disabled}; background: {t.window}; border-color: {t.border_muted}; }}

QCheckBox {{ spacing: 8px; color: {t.text}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {t.border}; border-radius: 3px; background: {t.surface_raised}; }}
QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QCheckBox:disabled {{ color: {t.text_disabled}; }}
QCheckBox::indicator:disabled {{ background: {t.window}; border-color: {t.border_muted}; }}
QSlider::groove:horizontal {{ height: 4px; background: {t.border}; border-radius: 2px; margin: 2px 0; }}
QSlider::handle:horizontal {{ background: {t.accent}; border: 1px solid {t.accent}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {t.accent_hover}; }}
QSlider::sub-page:horizontal {{ background: {t.accent}; border-radius: 2px; }}

QLabel:disabled {{ color: {t.text_disabled}; }}
#TitleLabel {{ color: {t.accent}; font-size: 20pt; font-weight: bold; padding: 12px; border-bottom: 1px solid {t.border}; }}
QTabWidget::pane {{ border: 1px solid {t.border}; border-top: none; }}
QTabBar::tab {{ background: {t.window}; border: 1px solid {t.border}; border-bottom: none; border-radius: 4px 4px 0 0; padding: 8px 18px; color: {t.text_muted}; font-weight: bold; min-height: 36px; }}
QTabBar::tab:selected {{ background: {t.surface_raised}; color: {t.accent}; border-bottom: 2px solid {t.accent}; }}
QTabBar::tab:!selected:hover {{ background: {t.surface_hover}; color: {t.text}; }}
QScrollArea {{ border: none; }}
QScrollArea[uiRole="settingsSidebar"] {{ background: transparent; }}
QScrollArea[uiRole="settingsSidebar"] QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QScrollArea[uiRole="mainContent"] {{ background: transparent; }}
QScrollArea[uiRole="mainContent"] QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QScrollArea[uiRole="tabOverflow"] {{ background: transparent; }}
QScrollArea[uiRole="tabOverflow"] QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QHeaderView::section {{ background: {t.surface_raised}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}
QTableWidget, QListWidget {{ gridline-color: {t.border}; background: {t.surface_raised}; alternate-background-color: {t.surface}; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: {t.accent}; color: white; }}
QScrollBar:vertical {{ background: {t.window}; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background: {t.border}; border-radius: 4px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background: {t.accent}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSplitter::handle {{ background: {t.border}; }}
QToolTip {{ background: {t.surface_raised}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}
QMenu {{ background: {t.nested_group}; border: 1px solid {t.border}; border-radius: 6px; padding: 3px; }}
QGroupBox[uiRole="datasetNavigator"] {{ background: {t.nested_group}; }}
QGroupBox[uiRole="datasetNavigator"]::title {{ background: transparent; }}
QListWidget[uiRole="quickFocus"],
QListWidget[uiRole="quickFocus"]::viewport {{ background: {t.nested_group}; }}
"""
