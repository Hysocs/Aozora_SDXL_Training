# unet_explorer_pyqt6.py
# Run with: pip install pyqt6 torch diffusers safetensors
# Then: python unet_explorer_pyqt6.py

import sys
import re
from pathlib import Path

import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QFrame,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor

from diffusers import StableDiffusionXLPipeline


class ModelLoaderThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16" if "fp16" in str(self.model_path).lower() else None,
            )
            self.finished.emit(pipe.unet)
        except Exception as e:
            self.error.emit(str(e))


class UNetExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.unet = None
        self.setWindowTitle("SDXL UNet Block Explorer — PyQt6 (Illustrious / Pony / Any SDXL)")
        self.resize(1200, 900)

        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-size: 14px; }
            QPushButton { background-color: #007acc; color: white; padding: 10px; border-radius: 6px; font-weight: bold; }
            QPushButton:hover { background-color: #005a99; }
            QTreeWidget { background-color: #2d2d2d; color: #ffffff; alternate-background-color: #262626; }
            QTreeWidget::item { padding: 6px; }
            QTreeWidget::item:selected { background-color: #007acc; }
            QHeaderView::section { background-color: #3c3c3c; color: white; padding: 8px; }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("SDXL UNet Block Explorer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #00ddff; padding: 20px;")
        layout.addWidget(title)

        btn = QPushButton("Load .safetensors Model File")
        btn.clicked.connect(self.load_model)
        btn.setFixedHeight(50)
        layout.addWidget(btn)

        self.status = QLabel("No model loaded")
        self.status.setStyleSheet("color: #aaaaaa; padding: 10px;")
        layout.addWidget(self.status)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setAlternatingRowColors(True)
        self.tree.setHeaderLabels(["Size (M)", "Layer Name"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.SortOrder.DescendingOrder)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tree)
        layout.addWidget(scroll)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SDXL .safetensors model", "", "Safetensors (*.safetensors)"
        )
        if not path:
            return

        self.status.setText("Loading model... Please wait")
        self.tree.clear()

        self.loader = ModelLoaderThread(path)
        self.loader.finished.connect(self.on_model_loaded)
        self.loader.error.connect(lambda e: self.status.setText(f"Error: {e}"))
        self.loader.start()

    def on_model_loaded(self, unet):
        self.unet = unet
        model_name = Path(self.loader.model_path).name
        total_params = sum(p.numel() for p in unet.parameters()) / 1e6
        self.status.setText(f"Loaded: {model_name} — {total_params:.1f}M params total")

        self.populate_tree()

    def populate_tree(self):
        self.tree.clear()

        # Prefixes for the major architectural blocks in SDXL UNet
        block_prefixes = [
            "conv_in",
            "down_blocks.0", "down_blocks.1", "down_blocks.2",
            "mid_block",
            "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3",
            "conv_out",
            "time_embedding",      # timestep embedding (small but visible)
            "label_emb",           # class labels if any (usually not present)
        ]

        # Build dict: block_name → list of (full_name, numel)
        blocks = {}
        other = []

        for name, param in self.unet.named_parameters():
            assigned = False
            for prefix in block_prefixes:
                if name.startswith(prefix + ".") or name == prefix:
                    block_name = prefix
                    blocks.setdefault(block_name, []).append((name, param.numel()))
                    assigned = True
                    break
            if not assigned:
                other.append((name, param.numel()))

        # Sort blocks in logical order (same order SDXL uses)
        order = {
            "conv_in": 0,
            "down_blocks.0": 1,
            "down_blocks.1": 2,
            "down_blocks.2": 3,
            "mid_block": 10,
            "up_blocks.0": 20,
            "up_blocks.1": 21,
            "up_blocks.2": 22,
            "up_blocks.3": 23,
            "conv_out": 30,
            "time_embedding": -1,
            "label_emb": -2,
        }
        sorted_blocks = sorted(blocks.items(), key=lambda x: order.get(x[0], 999))

        # Add everything
        for block_name, layers in sorted_blocks:
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            total_mb = sum(size for _, size in layers_sorted) / 1e6

            parent = QTreeWidgetItem(self.tree)
            nice_name = block_name.replace(".", " → ")  # makes it readable
            parent.setText(0, f"{total_mb:.2f}M")
            parent.setText(1, f"├── {nice_name} ({len(layers)} params)")
            parent.setFont(1, QFont("Segoe UI", 10, QFont.Weight.Bold))
            parent.setBackground(0, QColor("#333333"))
            parent.setBackground(1, QColor("#333333"))

            for name, size in layers_sorted:
                mb = size / 1e6
                child = QTreeWidgetItem(parent)
                child.setText(0, f"{mb:.3f}M")
                child.setText(1, name)

            self.tree.expandItem(parent)

        # Add "Other" at the end if anything is left
        if other:
            total_other = sum(size for _, size in other) / 1e6
            parent = QTreeWidgetItem(self.tree)
            parent.setText(0, f"{total_other:.2f}M")
            parent.setText(1, f"├── Other ({len(other)} params)")
            parent.setBackground(0, QColor("#aa4444"))
            parent.setBackground(1, QColor("#aa4444"))
            for name, size in sorted(other, key=lambda x: x[1], reverse=True):
                child = QTreeWidgetItem(parent)
                child.setText(0, f"{size/1e6:.3f}M")
                child.setText(1, name)

        # Optional: expand the huge blocks so you can see the attention layers
        # self.tree.expandAll()

        # Sort blocks logically
        def block_sort_key(b):
            if b == "conv_in": return -1
            if b == "conv_out": return 100
            if "down" in b: return int(b.split(".")[1])
            if "up" in b: return 50 + int(b.split(".")[1])
            if b == "mid_block": return 25
            return 999

        sorted_blocks = sorted(blocks.items(), key=lambda x: block_sort_key(x[0]))

        for block_name, layers in sorted_blocks:
            # Sort layers by param count descending
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)

            total_mb = sum(size for _, size in layers_sorted) / 1e6

            parent = QTreeWidgetItem(self.tree)
            parent.setText(0, f"{total_mb:.2f}M")
            parent.setText(1, f"├── {block_name} ({len(layers)} layers)")
            parent.setFont(1, QFont("Segoe UI", 10, QFont.Weight.Bold))
            parent.setBackground(0, QColor("#333333"))
            parent.setBackground(1, QColor("#333333"))

            for name, size in layers_sorted:
                mb = size / 1e6
                child = QTreeWidgetItem(parent)
                child.setText(0, f"{mb:.3f}")
                child.setText(1, name)

            self.tree.expandItem(parent) if block_name in ["conv_out", "down_blocks.2", "up_blocks.3"] else None

        self.tree.expandAll()  # Optional: start expanded or collapsed as you like


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UNetExplorer()
    window.show()
    sys.exit(app.exec())