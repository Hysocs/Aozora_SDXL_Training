import json
import os
import re
from PyQt6 import QtWidgets, QtCore, QtGui
import config as default_config
import copy
import sys
# --- STYLESHEET (Dark Purple Theme) ---
STYLESHEET = """
/* --- DARK VIOLET THEME --- */
QWidget {
    background-color: #2c2a3e; /* Dark violet-blue */
    color: #e0e0e0; /* Light grey text */
    font-family: 'Segoe UI', 'Calibri', 'Helvetica Neue', sans-serif;
    font-size: 15px; /* Slightly larger base font */
}
TrainingGUI {
    border: 2px solid #1a1926; /* Very dark border */
    border-radius: 8px;
}
#TitleLabel {
    color: #ab97e6; /* Light purple */
    font-size: 28px;
    font-weight: bold;
    padding: 15px;
    border-bottom: 2px solid #4a4668; /* Purple separator */
}
QGroupBox {
    border: 1px solid #4a4668; /* Purple border */
    border-radius: 8px;
    margin-top: 20px;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center; /* Centered title */
    padding: 0 10px;
    background-color: #383552; /* Slightly lighter purple */
    color: #e0e0e0;
    font-weight: bold;
    border-radius: 4px;
}
QPushButton {
    background-color: transparent;
    border: 2px solid #ab97e6; /* Light purple border */
    color: #ab97e6; /* Light purple text */
    padding: 10px 15px;
    min-height: 32px;
    max-height: 32px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover { background-color: #383552; color: #ffffff; }
QPushButton:pressed { background-color: #4a4668; }
QPushButton:disabled { color: #5c5a70; border-color: #5c5a70; background-color: transparent; }
#StartButton { background-color: #6a48d7; border-color: #6a48d7; color: #ffffff; } /* Vibrant purple */
#StartButton:hover { background-color: #7e5be0; border-color: #7e5be0; }
#StopButton { background-color: #e53935; border-color: #e53935; color: #ffffff; } /* Red */
#StopButton:hover { background-color: #f44336; border-color: #f44336; }
QLineEdit {
    background-color: #1a1926; /* Very dark */
    border: 1px solid #4a4668;
    padding: 6px;
    color: #e0e0e0;
    border-radius: 4px;
}
QLineEdit:focus { border: 1px solid #ab97e6; }
/* --- ADD THIS RULE --- */
QLineEdit:disabled {
    background-color: #242233; /* A muted, slightly lighter dark background */
    color: #7a788c; /* Muted grey/purple text */
    border: 1px solid #383552; /* A less prominent border */
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
    padding: 5px; /* Adjusted padding */
}
QCheckBox {
    spacing: 8px; /* Space between indicator and text */
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
QTabWidget::pane { border: 1px solid #4a4668; border-top: none; }
QTabBar::tab { background: #2c2a3e; border: 1px solid #4a4668; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 10px 20px; color: #e0e0e0; font-weight: bold; min-height: 40px; }
QTabBar::tab:selected { background: #383552; color: #ffffff; border-bottom: 3px solid #ab97e6; }
QTabBar::tab:!selected:hover { background: #4a4668; }
QScrollArea { border: none; } /* Remove border from scroll area */
"""
class SubOptionWidget(QtWidgets.QWidget):
    """
    A custom widget that contains indented sub-options and draws a vertical
    line on the left to visually group them under a parent checkbox.
    """
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
class TrainingGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA SDXL Trainer (UNet-Only)")
        self.setMinimumSize(QtCore.QSize(1000, 800))
        self.resize(1200, 950)
        self.config_dir = "configs"
        self.config_path = os.path.join(self.config_dir, "user_config.json")
        self.default_path = os.path.join(self.config_dir, "default_config.json")
        self.widgets = {}
        self.unet_layer_checkboxes = {}
        self.training_process = None
        self.tabs_loaded = set()
        self.current_config = {}
        self.last_line_is_progress = False
        self.default_config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        self.presets = {}
        self._ensure_config_files()
        self._load_presets()
        self._read_config_from_file()
        self._setup_ui()
        self._apply_config_to_widgets()
        self._on_tab_change(0)
    def paintEvent(self, event: QtGui.QPaintEvent):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
    def _ensure_config_files(self):
        os.makedirs(self.config_dir, exist_ok=True)
        if not os.path.exists(self.default_path):
            with open(self.default_path, 'w') as f:
                json.dump(self.default_config, f, indent=4)
        rtx3060_90_data = {
            "SINGLE_FILE_CHECKPOINT_PATH": "",
            "INSTANCE_DATA_DIR": "",
            "OUTPUT_DIR": "",
            "FORCE_RECACHE_LATENTS": False,
            "CACHING_BATCH_SIZE": 4,
            "BATCH_SIZE": 1,
            "NUM_WORKERS": 0,
            "TARGET_PIXEL_AREA": 1048576,
            "BUCKET_ASPECT_RATIOS": [
                0.5,
                0.56,
                0.66,
                0.75,
                1.0,
                1.25,
                1.33,
                1.5,
                1.77,
                2.0
            ],
            "MAX_TRAIN_STEPS": 250000,
            "GRADIENT_ACCUMULATION_STEPS": 64,
            "MIXED_PRECISION": "bfloat16",
            "SEED": 42,
            "SAVE_EVERY_N_STEPS": 5000,
            "RESUME_TRAINING": True,
            "RESUME_MODEL_PATH": "",
            "RESUME_STATE_PATH": "",
            "UNET_LEARNING_RATE": 8e-07,
            "LR_SCHEDULER_TYPE": "cosine",
            "LR_WARMUP_PERCENT": 0.1,
            "CLIP_GRAD_NORM": 1.0,
            "UNET_TRAIN_TARGETS": [
                "attn1",
                "attn2",
                "mid_block.attentions",
                "ff",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "time_embedding",
                "time_emb_proj",
                "add_embedding",
                "conv_in",
                "conv_shortcut",
                "downsamplers",
                "upsamplers",
                "conv_out",
                "norm",
                "norm1",
                "norm2",
                "norm3",
                "conv_norm_out"
            ],
            "USE_MIN_SNR_GAMMA": True,
            "MIN_SNR_GAMMA": 5.0,
            "MIN_SNR_VARIANT": "corrected",
            "USE_ZERO_TERMINAL_SNR": True,
            "USE_IP_NOISE_GAMMA": True,
            "IP_NOISE_GAMMA": 0.1,
            "USE_RESIDUAL_SHIFTING": True,
            "USE_COND_DROPOUT": True,
            "COND_DROPOUT_PROB": 0.1
        }
        rtx3060_90_path = os.path.join(self.config_dir, "rtx3060_90.json")
        if not os.path.exists(rtx3060_90_path):
            with open(rtx3060_90_path, 'w') as f:
                json.dump(rtx3060_90_data, f, indent=4)
    def _load_presets(self):
        self.presets = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json") and filename not in ["default_config.json", "user_config.json"]:
                name = os.path.splitext(filename)[0]
                path = os.path.join(self.config_dir, filename)
                with open(path, 'r') as f:
                    self.presets[name] = json.load(f)
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
        self.tab_view.addTab(QtWidgets.QWidget(), "1. Data & Model")
        self.tab_view.addTab(QtWidgets.QWidget(), "2. Training Parameters")
        self.tab_view.addTab(QtWidgets.QWidget(), "3. Advanced")
        self.tab_view.addTab(QtWidgets.QWidget(), "4. Training Console")
        corner_hbox = QtWidgets.QHBoxLayout()
        corner_hbox.setContentsMargins(10, 5, 10, 5)
        corner_hbox.setSpacing(10)
        self.config_dropdown = QtWidgets.QComboBox()
        self.config_dropdown.addItem("Default")
        if os.path.exists(self.config_path):
            self.config_dropdown.addItem("User Config")
        display_names = {
            "rtx3060_90": "RTX 3060 (90%)",
        }
        for name in sorted(self.presets.keys()):
            display = display_names.get(name, name.replace("_", " ").title())
            self.config_dropdown.addItem(display, name)
        self.config_dropdown.currentIndexChanged.connect(self.load_selected_config)
        corner_hbox.addWidget(self.config_dropdown)
        self.save_button = QtWidgets.QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)
        corner_hbox.addWidget(self.save_button)
        self.restore_button = QtWidgets.QPushButton("↺")
        self.restore_button.setToolTip("Restore Defaults")
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
        selected_display = self.config_dropdown.currentText()
        selected_key = self.config_dropdown.itemData(index)
        if selected_display == "Default":
            with open(self.default_path, 'r') as f:
                self.current_config = json.load(f)
        elif selected_display == "User Config":
            self._read_config_from_file()
        else:
            self.current_config = copy.deepcopy(self.presets[selected_key])
        self._apply_config_to_widgets()
    def _on_tab_change(self, index):
        if self.tab_view.widget(index).layout() is not None: return
        tab_widget = self.tab_view.widget(index)
        tab_layout = QtWidgets.QVBoxLayout(tab_widget)
        tab_layout.setContentsMargins(0,0,0,0)
        tab_name = self.tab_view.tabText(index)
        if tab_name in {"1. Data & Model", "2. Training Parameters", "3. Advanced"}:
            scroll_area = QtWidgets.QScrollArea()
            scroll_area.setWidgetResizable(True)
            tab_layout.addWidget(scroll_area)
            content_widget = QtWidgets.QWidget()
            scroll_area.setWidget(content_widget)
            if tab_name == "1. Data & Model": self._populate_data_model_tab(content_widget)
            elif tab_name == "2. Training Parameters": self._populate_training_params_tab(content_widget)
            elif tab_name == "3. Advanced": self._populate_advanced_tab(content_widget)
        elif tab_name == "4. Training Console":
            self._populate_console_tab(tab_layout)
        self._apply_config_to_widgets()
    def _populate_data_model_tab(self, parent_widget):
        layout = QtWidgets.QHBoxLayout(parent_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 5, 15, 15)
        # --- Left Column ---
        left_vbox = QtWidgets.QVBoxLayout()
        paths_group = QtWidgets.QGroupBox("File & Directory Paths")
        paths_layout = QtWidgets.QFormLayout(paths_group)
        self._create_path_option(paths_layout, "SINGLE_FILE_CHECKPOINT_PATH", "Base Model (.safetensors)", "Path to the base SDXL model.", "file_safetensors")
        self._create_path_option(paths_layout, "INSTANCE_DATA_DIR", "Dataset Directory", "Folder containing training images.", "folder")
        self._create_path_option(paths_layout, "OUTPUT_DIR", "Output Directory", "Folder where checkpoints will be saved.", "folder")
        left_vbox.addWidget(paths_group)
        batch_group = QtWidgets.QGroupBox("Batching & DataLoaders")
        batch_layout = QtWidgets.QFormLayout(batch_group)
        self._create_entry_option(batch_layout, "CACHING_BATCH_SIZE", "Caching Batch Size", "Adjust based on VRAM (e.g., 2-4).")
        # --- MODIFIED SECTION START ---
        # Create the batch size option with an updated tooltip explaining why it's disabled.
        self._create_entry_option(batch_layout, "BATCH_SIZE", "Training Batch Size", "Batches are not supported; this is fixed to 1.")
       
        # Get a direct reference to the QLineEdit widget for batch size.
        batch_size_widget = self.widgets.get("BATCH_SIZE")
        if batch_size_widget:
            # Set the text to "1" to explicitly show the default value.
            batch_size_widget.setText("1")
           
            # Disable the widget to make it non-editable (greyed out).
            batch_size_widget.setEnabled(False)
           
            # Ensure the underlying configuration value is also set to 1.
            # This guarantees consistency between the UI and the training script.
            self.current_config["BATCH_SIZE"] = 1
        # --- MODIFIED SECTION END ---
        self._create_entry_option(batch_layout, "NUM_WORKERS", "Dataloader Workers", "Set to 0 on Windows if you have issues.")
        self._create_bool_option(batch_layout, "FORCE_RECACHE_LATENTS", "Force Recache Latents", "Re-creates VAE latent caches on next run.")
        left_vbox.addWidget(batch_group)
        left_vbox.addStretch()
        # --- Right Column ---
        right_vbox = QtWidgets.QVBoxLayout()
        bucket_group = QtWidgets.QGroupBox("Aspect Ratio Bucketing")
        bucket_layout = QtWidgets.QFormLayout(bucket_group)
        self._create_entry_option(bucket_layout, "TARGET_PIXEL_AREA", "Target Pixel Area", "e.g., 1024*1024=1048576. Buckets are resolutions near this total area.")
        self._create_entry_option(bucket_layout, "BUCKET_ASPECT_RATIOS", "Aspect Ratios (comma-sep)", "e.g., 1.0, 1.5, 0.66. Defines bucket shapes.")
        right_vbox.addWidget(bucket_group)
        right_vbox.addStretch()
        # --- Final Layout Assembly ---
        layout.addLayout(left_vbox, stretch=1)
        layout.addLayout(right_vbox, stretch=1)
    def _populate_training_params_tab(self, parent_widget):
        layout = QtWidgets.QHBoxLayout(parent_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(15, 5, 15, 15)
        left_layout = QtWidgets.QVBoxLayout()
        core_group = QtWidgets.QGroupBox("Core Training")
        core_layout = QtWidgets.QFormLayout(core_group)
        self._create_entry_option(core_layout, "MAX_TRAIN_STEPS", "Max Training Steps:", "Total number of training steps.")
        self._create_entry_option(core_layout, "SAVE_EVERY_N_STEPS", "Save Every N Steps:", "How often to save a checkpoint.")
        self._create_entry_option(core_layout, "GRADIENT_ACCUMULATION_STEPS", "Gradient Accumulation:", "Simulates a larger batch size.")
        self._create_dropdown_option(core_layout, "MIXED_PRECISION", "Mixed Precision:", ["bfloat16", "fp16"], "bfloat16 for modern GPUs, fp16 for older.")
        self._create_entry_option(core_layout, "SEED", "Seed:", "Ensures reproducible training.")
        left_layout.addWidget(core_group)
        resume_group = QtWidgets.QGroupBox("Resume From Checkpoint")
        resume_group_layout = QtWidgets.QVBoxLayout(resume_group)
        self._create_bool_option(resume_group_layout, "RESUME_TRAINING", "Enable Resuming", "Allows resuming from a checkpoint.", self.toggle_resume_widgets)
        self.resume_sub_widget = SubOptionWidget()
        self._create_path_option(self.resume_sub_widget, "RESUME_MODEL_PATH", "Resume Model:", "The .safetensors checkpoint file.", "file_safetensors")
        self._create_path_option(self.resume_sub_widget, "RESUME_STATE_PATH", "Resume State:", "The .pt optimizer state file.", "file_pt")
        resume_group_layout.addWidget(self.resume_sub_widget)
        left_layout.addWidget(resume_group)
        left_layout.addStretch()
        right_layout = QtWidgets.QVBoxLayout()
        lr_group = QtWidgets.QGroupBox("Learning Rate & Optimizer")
        lr_layout = QtWidgets.QFormLayout(lr_group)
        self._create_entry_option(lr_layout, "UNET_LEARNING_RATE", "UNet Learning Rate:", "e.g., 8e-7")
        self._create_dropdown_option(lr_layout, "LR_SCHEDULER_TYPE", "LR Scheduler:", ["cosine", "linear"], "How LR changes over time.")
        self._create_entry_option(lr_layout, "LR_WARMUP_PERCENT", "LR Warmup Percent:", "e.g., 0.1 for 10% of total steps")
        self._create_entry_option(lr_layout, "CLIP_GRAD_NORM", "Clip Gradient Norm:", "Helps prevent unstable training. 1.0 is safe.")
        right_layout.addWidget(lr_group)
        right_layout.addStretch()
        layout.addLayout(left_layout, stretch=1)
        layout.addLayout(right_layout, stretch=1)
    def _populate_advanced_tab(self, parent_widget):
        main_layout = QtWidgets.QHBoxLayout(parent_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        left_layout = QtWidgets.QVBoxLayout()
        adv_group = QtWidgets.QGroupBox("Advanced (v-prediction)")
        adv_layout = QtWidgets.QVBoxLayout(adv_group)
        self._create_bool_option(adv_layout, "USE_RESIDUAL_SHIFTING", "Use Residual Shifting Schedulers", "High-Noise Timestep Sampling with Curved Schedules")
        self._create_bool_option(adv_layout, "USE_ZERO_TERMINAL_SNR", "Use Zero-Terminal SNR Rescaling", "Rescales noise schedule for better dynamic range in v-pred.")
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_MIN_SNR_GAMMA", "Use Min-SNR Gamma", "Recommended for v-prediction models.", self.toggle_min_snr_gamma_widget)
        self.min_snr_sub_widget = SubOptionWidget()
        self._create_entry_option(self.min_snr_sub_widget, "MIN_SNR_GAMMA", "Gamma Value:", "Common range is 5.0 to 20.0.")
        self._create_dropdown_option(self.min_snr_sub_widget, "MIN_SNR_VARIANT", "Variant:", ["standard", "corrected", "debiased"], "Select the Min-SNR weighting variant.")
        adv_layout.addWidget(self.min_snr_sub_widget)
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_IP_NOISE_GAMMA", "Use Input Perturbation Noise", "Adds Gaussian noise to latents for regularization.", self.toggle_ip_noise_gamma_widget)
        self.ip_sub_widget = SubOptionWidget()
        self._create_entry_option(self.ip_sub_widget, "IP_NOISE_GAMMA", "Gamma Value:", "Common range is 0.05 to 0.25.")
        adv_layout.addWidget(self.ip_sub_widget)
        adv_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.Shape.HLine))
        self._create_bool_option(adv_layout, "USE_COND_DROPOUT", "Use Text Conditioning Dropout", "Drops conditioning randomly for regularization.", self.toggle_cond_dropout_widget)
        self.cond_sub_widget = SubOptionWidget()
        self._create_entry_option(self.cond_sub_widget, "COND_DROPOUT_PROB", "Dropout Probability:", "e.g., 0.1 (5-15% common).")
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
                "norm": "Attention GroupNorm",
                "norm1": "ResNet GroupNorm1",
                "norm2": "ResNet GroupNorm2",
                "norm3": "Transformer Norm3",
                "conv_norm_out": "Output GroupNorm"
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
    def _create_entry_option(self, layout_or_widget, key, label, tooltip_text):
        entry = QtWidgets.QLineEdit()
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        if isinstance(layout_or_widget, SubOptionWidget): layout_or_widget.addRow(label_widget, entry)
        else: layout_or_widget.addRow(label_widget, entry)
        self.widgets[key] = entry
        entry.textChanged.connect(lambda text, k=key: self._update_config_from_widget(k, entry))
    def _create_path_option(self, layout_or_widget, key, label, tooltip_text, file_type):
        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        entry = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse...")
        hbox.addWidget(entry, stretch=1)
        hbox.addWidget(button)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        if isinstance(layout_or_widget, SubOptionWidget): layout_or_widget.addRow(label_widget, container)
        else: layout_or_widget.addRow(label_widget, container)
        button.clicked.connect(lambda: self._browse_path(entry, file_type))
        self.widgets[key] = entry
        entry.textChanged.connect(lambda text, k=key: self._update_config_from_widget(k, entry))
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
        if isinstance(layout, QtWidgets.QVBoxLayout): layout.addWidget(checkbox)
        else: layout.addRow(checkbox)
        self.widgets[key] = checkbox
        checkbox.stateChanged.connect(lambda state, k=key: self._update_config_from_widget(k, checkbox))
        if command: checkbox.stateChanged.connect(command)
    def _create_dropdown_option(self, layout_or_widget, key, label, values, tooltip_text):
        dropdown = QtWidgets.QComboBox()
        dropdown.addItems(values)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        if isinstance(layout_or_widget, SubOptionWidget): layout_or_widget.addRow(label_widget, dropdown)
        else: layout_or_widget.addRow(label_widget, dropdown)
        self.widgets[key] = dropdown
        dropdown.currentTextChanged.connect(lambda text, k=key: self._update_config_from_widget(k, dropdown))
    def _update_unet_targets_config(self):
        if not self.unet_layer_checkboxes: return
        self.current_config["UNET_TRAIN_TARGETS"] = [k for k, cb in self.unet_layer_checkboxes.items() if cb.isChecked()]
    def _update_config_from_widget(self, key, widget):
        if key not in self.current_config: return
        if isinstance(widget, QtWidgets.QLineEdit): self.current_config[key] = widget.text().strip()
        elif isinstance(widget, QtWidgets.QCheckBox): self.current_config[key] = widget.isChecked()
        elif isinstance(widget, QtWidgets.QComboBox): self.current_config[key] = widget.currentText()
    def _read_config_from_file(self):
        config = {}
        with open(self.default_path, 'r') as f:
            config = json.load(f)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    config.update(user_config)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not read {self.config_path}. Using defaults. Error: {e}")
        self.current_config = config
    def _apply_config_to_widgets(self):
        if self.unet_layer_checkboxes:
            unet_targets = self.current_config.get("UNET_TRAIN_TARGETS", [])
            for key, checkbox in self.unet_layer_checkboxes.items():
                checkbox.setChecked(key in unet_targets)
        for key, widget in self.widgets.items():
            if key in self.current_config:
                value = self.current_config.get(key)
                if value is None: continue
                widget.blockSignals(True)
                try:
                    if isinstance(widget, QtWidgets.QLineEdit): widget.setText(", ".join(map(str, value)) if isinstance(value, list) else str(value))
                    elif isinstance(widget, QtWidgets.QCheckBox): widget.setChecked(bool(value))
                    elif isinstance(widget, QtWidgets.QComboBox): widget.setCurrentText(str(value))
                finally: widget.blockSignals(False)
        if "USE_MIN_SNR_GAMMA" in self.widgets: self.toggle_min_snr_gamma_widget()
        if "RESUME_TRAINING" in self.widgets: self.toggle_resume_widgets()
        if "USE_IP_NOISE_GAMMA" in self.widgets: self.toggle_ip_noise_gamma_widget()
        if "USE_COND_DROPOUT" in self.widgets: self.toggle_cond_dropout_widget()
    def save_config(self):
        config_to_save = {}
        with open(self.default_path, 'r') as f:
            default_map = json.load(f)
        for key, default_val in default_map.items():
            live_val = self.current_config.get(key)
            if live_val is None:
                self.log(f"Info: No value found for '{key}' in current UI state. Skipping.")
                continue
            converted_val = None
            try:
                if key == "UNET_TRAIN_TARGETS": converted_val = live_val
                elif isinstance(default_val, bool): converted_val = bool(live_val)
                elif isinstance(default_val, int): converted_val = int(str(live_val)) if str(live_val).strip() else 0
                elif isinstance(default_val, float): converted_val = float(str(live_val)) if str(live_val).strip() else 0.0
                elif key == "BUCKET_ASPECT_RATIOS":
                    raw_list_str = str(live_val).strip().replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                    converted_val = [float(p.strip()) for p in raw_list_str.split(',') if p.strip()]
                else: converted_val = str(live_val)
                config_to_save[key] = converted_val
            except (ValueError, TypeError) as e:
                self.log(f"Warning: Could not convert value for '{key}'. It will not be saved. Error: {e}")
        index = self.config_dropdown.currentIndex()
        selected_display = self.config_dropdown.currentText()
        selected_key = self.config_dropdown.itemData(index)
        if selected_display == "Default" or selected_display == "User Config":
            save_path = self.config_path
            if selected_display == "Default":
                self.log("Saving changes from Default to User Config...")
        else:
            save_path = os.path.join(self.config_dir, f"{selected_key}.json")
        try:
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            self.log(f"Successfully saved all current settings to {os.path.basename(save_path)}")
            if selected_key is not None:
                with open(save_path, 'r') as f:
                    self.presets[selected_key] = json.load(f)
            if os.path.basename(save_path) == "user_config.json" and "User Config" not in [self.config_dropdown.itemText(i) for i in range(self.config_dropdown.count())]:
                self.config_dropdown.insertItem(1, "User Config")
        except Exception as e:
            self.log(f"CRITICAL ERROR: Could not write to {save_path}. Error: {e}")
    def restore_defaults(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
            self.log(f"Removed {self.config_path}. Restoring defaults.")
        self._read_config_from_file()
        self._apply_config_to_widgets()
    def toggle_all_unet_checkboxes(self, state):
        for cb in self.unet_layer_checkboxes.values(): cb.setChecked(state)
        self._update_unet_targets_config()
    def toggle_min_snr_gamma_widget(self):
        if "USE_MIN_SNR_GAMMA" in self.widgets: self.min_snr_sub_widget.setVisible(self.widgets["USE_MIN_SNR_GAMMA"].isChecked())
    def toggle_ip_noise_gamma_widget(self):
        if "USE_IP_NOISE_GAMMA" in self.widgets: self.ip_sub_widget.setVisible(self.widgets["USE_IP_NOISE_GAMMA"].isChecked())
    def toggle_resume_widgets(self):
        if "RESUME_TRAINING" in self.widgets: self.resume_sub_widget.setVisible(self.widgets["RESUME_TRAINING"].isChecked())
    def toggle_cond_dropout_widget(self):
        if "USE_COND_DROPOUT" in self.widgets: self.cond_sub_widget.setVisible(self.widgets["USE_COND_DROPOUT"].isChecked())
    def is_progress_bar(self, line):
        pattern = r'^\s*\d+%\|\S*\| \d+/\d+ \[\d+:\d+<\S+,\s*\S+\]$'
        return bool(re.match(pattern, line))
    def append_log(self, text, replace=False):
        """
        Appends text to the log box with intelligent scrolling.
        If the user is already at the bottom, it stays at the bottom.
        If the user has scrolled up, the scroll position is preserved.
        """
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
    def log(self, message):
        self.append_log(message.strip(), replace=False)
    def start_training(self):
        self.save_config()
        self.log("\n" + "="*50 + "\nStarting training process...\n" + "="*50)
        self.param_info_label.setText("Verifying training parameters...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.training_process = QtCore.QProcess(self)
        self.training_process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.training_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.training_process.finished.connect(self.training_finished)
        # --- IMPROVEMENTS START HERE ---
        # 1. Set the working directory explicitly to the script's location.
        # This ensures that files like 'user_config.json' and 'train.py' are found reliably.
        script_path = os.path.dirname(os.path.abspath(__file__))
        self.training_process.setWorkingDirectory(script_path)
        self.log(f"INFO: Set working directory for training process to: {script_path}")
        # 2. Create a clean environment for the child process.
        # This helps prevent conflicts with other Python/CUDA libraries on your system.
        env = QtCore.QProcessEnvironment.systemEnvironment()
       
        # Get the Python path from the current executable to ensure it's prioritized.
        python_dir = os.path.dirname(sys.executable)
       
        # Prepend the script's python directory and its Scripts subdirectory to the system PATH.
        # This ensures it finds the correct python.exe and any installed packages first.
        original_path = env.value("Path")
        new_path = f"{python_dir};{os.path.join(python_dir, 'Scripts')};{original_path}"
        env.insert("Path", new_path)
       
        self.training_process.setProcessEnvironment(env)
        self.log("INFO: Configured isolated environment for the training process.")
        # --- IMPROVEMENTS END HERE ---
        try:
            # Launch the process
            self.training_process.start(sys.executable, ["-u", "train.py"])
            if not self.training_process.waitForStarted(5000):
                self.log(f"ERROR: Failed to start training process: {self.training_process.errorString()}")
                self.training_finished()
               
        except Exception as e:
            self.log(f"CRITICAL ERROR: Could not launch Python process: {e}")
            self.training_finished()
    def handle_stdout(self):
        text = bytes(self.training_process.readAllStandardOutput()).decode('utf-8', errors='ignore')
        for line in text.splitlines():
            if "NOTE: Redirects are currently not supported in Windows or MacOs." in line:
                continue
            if line.startswith("GUI_PARAM_INFO::"):
                self.param_info_label.setText(f"Trainable Parameters: {line.replace('GUI_PARAM_INFO::', '').strip()}")
            else:
                if '\r' in line:
                    parts = line.split('\r')
                    if parts:
                        current_progress = parts[-1].lstrip()
                        if current_progress:
                            replace = self.last_line_is_progress
                            self.append_log(current_progress, replace=replace)
                            self.last_line_is_progress = True
                else:
                    replace = self.is_progress_bar(line) and self.last_line_is_progress
                    self.append_log(line, replace=replace)
                    self.last_line_is_progress = self.is_progress_bar(line)
    def stop_training(self):
        if self.training_process and self.training_process.state() == QtCore.QProcess.ProcessState.Running:
            self.log("--- Sending termination signal to training process... ---")
            self.training_process.kill()
        else: self.log("No active training process to stop.")
    def training_finished(self):
        if not self.training_process: return
        exit_code = self.training_process.exitCode()
        status = "successfully" if exit_code == 0 else f"with an error (Code: {exit_code})"
        self.log(f"\n" + "="*50 + f"\nTraining process finished {status}.\n" + "="*50)
        if exit_code == 0: self.param_info_label.setText("Parameters: (training complete)")
        else: self.param_info_label.setText("Parameters: (training failed or stopped)")
        self.training_process = None
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = TrainingGUI()
    main_win.show()
    sys.exit(app.exec())