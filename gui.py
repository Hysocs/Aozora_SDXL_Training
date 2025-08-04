import sys
import json
import os
from PyQt6 import QtWidgets, QtCore, QtGui
import config as default_config

# --- STYLESHEET (No changes needed here, the UI change below fixes the appearance) ---
STYLESHEET = """
/* --- AOZORA(SKY BLUE) THEME --- */
QWidget {
    background-color: #1e2a3a;
    color: #f0f8ff;
    font-family: 'Segoe UI', 'Calibri', 'Helvetica Neue', sans-serif;
    font-size: 14px;
}
TrainingGUI {
    border: 2px solid #2c3e50;
    border-radius: 8px;
}
#TitleLabel {
    color: #87CEEB;
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    border-bottom: 1px solid #2c3e50;
}
QGroupBox {
    border: 1px solid #2c3e50;
    border-radius: 6px;
    margin-top: 15px;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    background-color: #1e2a3a;
    color: #f0f8ff;
    font-weight: bold;
}
QPushButton {
    background-color: transparent;
    border: 1px solid #87CEEB;
    color: #87CEEB;
    padding: 8px 12px;
    min-height: 24px;
    border-radius: 5px;
    font-weight: bold;
}
QPushButton:hover { background-color: #2c3e50; color: #ffffff; }
QPushButton:pressed { background-color: #3b4d61; }
QPushButton:disabled { color: #5c6a79; border-color: #5c6a79; background-color: transparent; }
#StartButton { background-color: #00BFFF; border-color: #00BFFF; color: #1e2a3a; }
#StartButton:hover { background-color: #1E90FF; border-color: #1E90FF; }
#StopButton { background-color: #e74c3c; border-color: #e74c3c; color: #ffffff; }
#StopButton:hover { background-color: #c0392b; border-color: #c0392b; }
QLineEdit {
    background-color: #2c3e50;
    border: 1px solid #3b4d61;
    padding: 5px;
    color: #f0f8ff;
    border-radius: 4px;
}
QLineEdit:focus { border: 1px solid #87CEEB; }
#ParamInfoLabel {
    background-color: #141f2b;
    color: #87CEEB;
    font-weight: bold;
    font-size: 13px;
    border: 1px solid #2c3e50;
    border-radius: 4px;
    padding: 4px;
}
QPlainTextEdit {
    background-color: #141f2b;
    border: 1px solid #2c3e50;
    color: #f0f8ff;
    font-family: 'Consolas', 'Courier New', monospace;
    border-radius: 4px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #87CEEB;
    background-color: #2c3e50;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #87CEEB;
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSIjMWUyYTNhIiBkPSJNOSAxNi4xN0w0LjgzIDEybC0xLjQyIDEuNDFMOSAxOUwyMSA3bC0xLjQxLTEuNDF6Ii8+PC9zdmc+");
}
QCheckBox::indicator:disabled {
    border: 1px solid #5c6a79;
    background-color: #3b4d61;
}
QComboBox { background-color: #2c3e50; border: 1px solid #3b4d61; padding: 5px; border-radius: 4px; }
QComboBox:on { border: 1px solid #87CEEB; }
QComboBox::drop-down { border-left: 1px solid #3b4d61; }
QComboBox QAbstractItemView { background-color: #2c3e50; border: 1px solid #87CEEB; selection-background-color: #87CEEB; selection-color: #1e2a3a; }
QTabWidget::pane { border: 1px solid #2c3e50; border-top: none; }
QTabBar::tab { background: #1e2a3a; border: 1px solid #2c3e50; border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px; padding: 8px 15px; color: #f0f8ff; font-weight: bold; }
QTabBar::tab:selected { background: #2c3e50; color: #ffffff; border-bottom: 3px solid #87CEEB; }
QTabBar::tab:!selected:hover { background: #3b4d61; }
"""

class TrainingGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA SDXL Trainer (UNet-Only)")
        self.setMinimumSize(QtCore.QSize(1000, 900))
        self.config_path = "user_config.json"
        self.widgets = {}
        self.unet_layer_checkboxes = {}
        self.training_process = None
        self.tabs_loaded = set()
        self.current_config = {}
        self._read_config_from_file()
        self._setup_ui()
        self._apply_config_to_widgets() # Apply the config after the UI is set up
        self._on_tab_change(0)


    def paintEvent(self, event: QtGui.QPaintEvent):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)

    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(2, 2, 2, 2)
        title_label = QtWidgets.QLabel("AOZORA SDXL Trainer")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)
        self.tab_view = QtWidgets.QTabWidget()
        self.tab_view.currentChanged.connect(self._on_tab_change)
        self.main_layout.addWidget(self.tab_view)
        self.tab_view.addTab(QtWidgets.QWidget(), "1. Data & Model")
        self.tab_view.addTab(QtWidgets.QWidget(), "2. Training Parameters")
        self.tab_view.addTab(QtWidgets.QWidget(), "3. UNet Layer Targeting")
        self.param_info_label = QtWidgets.QLabel("Parameters: (awaiting training start)")
        self.param_info_label.setObjectName("ParamInfoLabel")
        self.param_info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.param_info_label.setContentsMargins(0, 5, 0, 0)
        self.main_layout.addWidget(self.param_info_label)
        self.log_textbox = QtWidgets.QPlainTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setMinimumHeight(250)
        self.main_layout.addWidget(self.log_textbox, stretch=1)
        button_layout = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("Save Config")
        self.save_button.clicked.connect(self.save_config)
        self.restore_button = QtWidgets.QPushButton("Restore Defaults")
        self.restore_button.clicked.connect(self.restore_defaults)
        self.start_button = QtWidgets.QPushButton("Start Training")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QtWidgets.QPushButton("Stop Training")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.restore_button)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(button_layout)

    def _on_tab_change(self, index):
        tab_name = self.tab_view.tabText(index)
        if tab_name not in self.tabs_loaded:
            tab_widget = self.tab_view.widget(index)
            if tab_name == "1. Data & Model": self._populate_data_model_tab(tab_widget)
            elif tab_name == "2. Training Parameters": self._populate_training_params_tab(tab_widget)
            elif tab_name == "3. UNet Layer Targeting": self._populate_layer_targeting_tab(tab_widget)
            self._apply_config_to_widgets()
            self.tabs_loaded.add(tab_name)

    def _populate_data_model_tab(self, tab):
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setSpacing(15)
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
        self._create_entry_option(batch_layout, "BATCH_SIZE", "Training Batch Size", "Almost always 1 for this script.")
        self.widgets["BATCH_SIZE"].setEnabled(False)
        self._create_entry_option(batch_layout, "NUM_WORKERS", "Dataloader Workers", "Set to 0 on Windows if you have issues.")
        self._create_bool_option(batch_layout, "FORCE_RECACHE_LATENTS", "Force Recache Latents", "Re-creates VAE latent caches on next run.")
        left_vbox.addWidget(batch_group)
        left_vbox.addStretch()
        right_vbox = QtWidgets.QVBoxLayout()
        bucket_group = QtWidgets.QGroupBox("Aspect Ratio Bucketing")
        bucket_layout = QtWidgets.QFormLayout(bucket_group)
        self._create_entry_option(bucket_layout, "TARGET_PIXEL_AREA", "Target Pixel Area", "e.g., 1024*1024=1048576. Buckets are resolutions near this total area.")
        self._create_entry_option(bucket_layout, "BUCKET_ASPECT_RATIOS", "Aspect Ratios (comma-sep)", "e.g., 1.0, 1.5, 0.66. Defines bucket shapes.")
        right_vbox.addWidget(bucket_group)
        right_vbox.addStretch()
        layout.addLayout(left_vbox, stretch=1)
        layout.addLayout(right_vbox, stretch=1)

    # === REFACTORED: Now uses a QCheckBox inside the group box for consistent styling ===
    def _populate_training_params_tab(self, tab):
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setSpacing(15)
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
        resume_layout = QtWidgets.QFormLayout(resume_group)
        resume_layout.setContentsMargins(5, 0, 5, 5)
        # The checkbox now controls the other widgets in this group
        self._create_bool_option(resume_layout, "RESUME_TRAINING", "Enable Resuming", "Allows resuming from a checkpoint.", self.toggle_resume_widgets)
        self._create_path_option(resume_layout, "RESUME_MODEL_PATH", "Resume Model:", "The .safetensors checkpoint file.", "file_safetensors")
        self._create_path_option(resume_layout, "RESUME_STATE_PATH", "Resume State:", "The .pt optimizer state file.", "file_pt")
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

        adv_group = QtWidgets.QGroupBox("Advanced (v-prediction)")
        adv_layout = QtWidgets.QFormLayout(adv_group)
        adv_layout.setContentsMargins(5, 0, 5, 5)
        self._create_bool_option(adv_layout, "USE_MIN_SNR_GAMMA", "Use Min-SNR Gamma", "Recommended for v-prediction models.", self.toggle_min_snr_gamma_widget)
        self._create_entry_option(adv_layout, "MIN_SNR_GAMMA", "Min-SNR Gamma Value:", "Common range is 5.0 to 20.0.")
        right_layout.addWidget(adv_group)
        right_layout.addStretch()

        layout.addLayout(left_layout, stretch=1)
        layout.addLayout(right_layout, stretch=1)


    def _populate_layer_targeting_tab(self, tab):
        main_layout = QtWidgets.QVBoxLayout(tab)
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addStretch()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.toggle_all_unet_checkboxes(True))
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.toggle_all_unet_checkboxes(False))
        top_layout.addWidget(select_all_btn)
        top_layout.addWidget(deselect_all_btn)
        main_layout.addLayout(top_layout)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        all_unet_targets = {
            "Attention Blocks": {"attn1": "Self-Attention", "attn2": "Cross-Attention", "mid_block.attentions": "Mid-Block Attention", "ff": "Feed-Forward Networks"},
            "Attention Sub-Layers (Advanced)": {"to_q": "Query Projection", "to_k": "Key Projection", "to_v": "Value Projection", "to_out.0": "Output Projection", "proj_in": "Transformer Input Projection", "proj_out": "Transformer Output Projection"},
            "UNet Embeddings (Non-Text)": {"time_embedding": "Time Embedding", "time_emb_proj": "Time Embedding Projection", "add_embedding": "Added Conditional Embedding"},
            "Convolutional & ResNet Layers": {"conv_in": "Input Conv", "conv1": "ResNet Conv1", "conv2": "ResNet Conv2", "conv_shortcut": "ResNet Skip Conv", "downsamplers": "Downsampler Convs", "upsamplers": "Upsampler Convs", "conv_out": "Output Conv"},
            "Normalization Layers (Experimental)": {"norm1": "ResNet GroupNorm1", "norm2": "ResNet GroupNorm2", "conv_norm_out": "Output GroupNorm"},
        }
        for group_name, targets in all_unet_targets.items():
            label = QtWidgets.QLabel(f"<b>{group_name}</b>")
            label.setStyleSheet("margin-top: 10px; border: none;")
            scroll_layout.addWidget(label)
            for key, text in targets.items():
                cb = QtWidgets.QCheckBox(f"{text} (keyword: '{key}')")
                cb.setStyleSheet("margin-left: 15px;")
                scroll_layout.addWidget(cb)
                self.unet_layer_checkboxes[key] = cb
                cb.stateChanged.connect(self._update_unet_targets_config)
        scroll_layout.addStretch()
        main_layout.addWidget(scroll_area)

    def _create_entry_option(self, layout, key, label, tooltip_text):
        entry = QtWidgets.QLineEdit()
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        layout.addRow(label_widget, entry)
        self.widgets[key] = entry
        entry.textChanged.connect(lambda text, k=key: self._update_config_from_widget(k, entry))

    def _create_path_option(self, layout, key, label, tooltip_text, file_type):
        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        entry = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse...")
        hbox.addWidget(entry, stretch=1)
        hbox.addWidget(button)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        layout.addRow(label_widget, container)
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
        layout.addRow(checkbox)
        self.widgets[key] = checkbox
        checkbox.stateChanged.connect(lambda state, k=key: self._update_config_from_widget(k, checkbox))
        if command:
            checkbox.stateChanged.connect(command)

    def _create_dropdown_option(self, layout, key, label, values, tooltip_text):
        dropdown = QtWidgets.QComboBox()
        dropdown.addItems(values)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        layout.addRow(label_widget, dropdown)
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
        config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    config.update(user_config)
            except (json.JSONDecodeError, TypeError) as e:
                # This is a safe fallback, but we should probably inform the user.
                # For now, we'll just log it. A pop-up could be used for a more direct notification.
                print(f"Warning: Could not read {self.config_path}. Using defaults. Error: {e}")
        self.current_config = config


    def _apply_config_to_widgets(self):
        # This function is now safe to call even if the widgets aren't fully populated yet.
        if self.unet_layer_checkboxes:
            unet_targets = self.current_config.get("UNET_TRAIN_TARGETS", [])
            for key, checkbox in self.unet_layer_checkboxes.items():
                checkbox.setChecked(key in unet_targets)

        for key, widget in self.widgets.items():
            if key in self.current_config:
                value = self.current_config.get(key)
                if value is None:
                    continue
                widget.blockSignals(True)
                try:
                    if isinstance(widget, QtWidgets.QLineEdit): widget.setText(", ".join(map(str, value)) if isinstance(value, list) else str(value))
                    elif isinstance(widget, QtWidgets.QCheckBox): widget.setChecked(bool(value))
                    elif isinstance(widget, QtWidgets.QComboBox): widget.setCurrentIndex(widget.findText(str(value)))
                finally:
                    widget.blockSignals(False)

        # These need to be called to ensure the UI state is consistent
        if "USE_MIN_SNR_GAMMA" in self.widgets:
            self.toggle_min_snr_gamma_widget()
        if "RESUME_TRAINING" in self.widgets:
            self.toggle_resume_widgets()


    # === REWRITTEN: This function now correctly handles typing and saving ---
    # === NEW VERSION: Saves ALL current values from the GUI to the config file ===
    def save_config(self):
        """
        Converts all current UI values to their correct python types and saves
        the entire configuration to user_config.json.
        """
        config_to_save = {}
        # We use the default_config file as a "manifest" of all expected keys and their intended data types.
        default_map = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}

        # We iterate over the default map's keys to ensure we process every setting
        for key, default_val in default_map.items():
            # Get the current value from our live config dictionary, which is updated by the UI
            live_val = self.current_config.get(key)

            # If a value is somehow missing from the live config, skip it to avoid errors.
            if live_val is None:
                self.log(f"Info: No value found for '{key}' in current UI state. Skipping.")
                continue

            converted_val = None
            try:
                # This value is updated and stored as a list of strings directly
                if key == "UNET_TRAIN_TARGETS":
                    converted_val = live_val
                # Handle booleans from checkboxes
                elif isinstance(default_val, bool):
                    converted_val = bool(live_val)
                # Handle integer numbers from QLineEdit
                elif isinstance(default_val, int):
                    # Handle case where user leaves the box empty
                    converted_val = int(str(live_val)) if str(live_val).strip() else 0
                # Handle floating point numbers from QLineEdit
                elif isinstance(default_val, float):
                    # Handle case where user leaves the box empty
                    converted_val = float(str(live_val)) if str(live_val).strip() else 0.0
                # Handle list of floats specifically for aspect ratios
                elif key == "BUCKET_ASPECT_RATIOS":
                    raw_list_str = str(live_val).strip().replace('[', '').replace(']', '').replace("'", "").replace('"', '')
                    converted_val = [float(p.strip()) for p in raw_list_str.split(',') if p.strip()]
                # Handle all other types (which are mostly strings for paths and dropdowns)
                else:
                    converted_val = str(live_val)

                # --- THIS IS THE KEY CHANGE ---
                # We no longer check if it's different from default. We save every value.
                config_to_save[key] = converted_val

            except (ValueError, TypeError) as e:
                self.log(f"Warning: Could not convert value for '{key}'. It will not be saved. Error: {e}")

        # Write the complete dictionary to the file
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            self.log("Successfully saved all current settings to user_config.json")
        except Exception as e:
            self.log(f"CRITICAL ERROR: Could not write to {self.config_path}. Error: {e}")
            
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
        if "MIN_SNR_GAMMA" in self.widgets and "USE_MIN_SNR_GAMMA" in self.widgets:
            is_enabled = self.widgets["USE_MIN_SNR_GAMMA"].isChecked()
            self.widgets["MIN_SNR_GAMMA"].setEnabled(is_enabled)

    def toggle_resume_widgets(self):
        if "RESUME_MODEL_PATH" in self.widgets and "RESUME_TRAINING" in self.widgets:
            is_enabled = self.widgets["RESUME_TRAINING"].isChecked()
            self.widgets["RESUME_MODEL_PATH"].setEnabled(is_enabled)
            self.widgets["RESUME_STATE_PATH"].setEnabled(is_enabled)

    def log(self, message):
        self.log_textbox.appendPlainText(message.strip())
        self.log_textbox.verticalScrollBar().setValue(self.log_textbox.verticalScrollBar().maximum())

    def start_training(self):
        self.save_config()
        self.log("\n" + "="*50 + "\nStarting training process...\n" + "="*50)
        self.param_info_label.setText("Verifying training parameters...")
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.training_process = QtCore.QProcess(self)
        self.training_process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.training_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.training_process.finished.connect(self.training_finished)
        try:
            self.training_process.start(sys.executable, ["-u", "train.py"])
            if not self.training_process.waitForStarted(5000):
                self.log(f"ERROR: Failed to start training process: {self.training_process.errorString()}")
                self.training_finished()
        except Exception as e:
            self.log(f"CRITICAL ERROR: Could not launch Python process: {e}")
            self.training_finished()

    def handle_stdout(self):
        text = bytes(self.training_process.readAllStandardOutput()).decode('utf-8', errors='ignore')
        log_output = []
        for line in text.splitlines():
            if line.startswith("GUI_PARAM_INFO::"): self.param_info_label.setText(f"Trainable Parameters: {line.replace('GUI_PARAM_INFO::', '').strip()}")
            else: log_output.append(line)
        if log_output: self.log('\n'.join(log_output))

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