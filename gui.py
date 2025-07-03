import sys
import json
import os
from PyQt6 import QtWidgets, QtCore, QtGui
import config as default_config

# AOZORA(Sky Blue) Modern Theme Stylesheet
STYLESHEET = """
/* --- AOZORA(SKY BLUE) THEME --- */

/* --- General Window & Font --- */
QWidget {
    background-color: #1e2a3a; /* Dark Slate Blue */
    color: #f0f8ff; /* AliceBlue */
    font-family: 'Segoe UI', 'Calibri', 'Helvetica Neue', sans-serif;
    font-size: 14px;
}

TrainingGUI {
    border: 2px solid #2c3e50;
    border-radius: 8px;
}

/* --- Title --- */
#TitleLabel {
    color: #87CEEB; /* SkyBlue */
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    border-bottom: 1px solid #2c3e50; /* Softer separator */
}

/* --- Group Boxes --- */
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

/* --- Buttons --- */
QPushButton {
    background-color: transparent;
    border: 1px solid #87CEEB;
    color: #87CEEB;
    padding: 8px 12px;
    min-height: 24px;
    border-radius: 5px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2c3e50;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #3b4d61;
}
QPushButton:disabled {
    color: #5c6a79;
    border-color: #5c6a79;
    background-color: transparent;
}

/* --- Special Buttons --- */
#StartButton {
    background-color: #00BFFF; /* DeepSkyBlue */
    border-color: #00BFFF;
    color: #1e2a3a; /* Dark text for contrast on bright button */
}
#StartButton:hover {
    background-color: #1E90FF; /* DodgerBlue */
    border-color: #1E90FF;
}
#StopButton {
    background-color: #e74c3c; /* A softer, modern red */
    border-color: #e74c3c;
    color: #ffffff;
}
#StopButton:hover {
    background-color: #c0392b; /* Darker red on hover */
    border-color: #c0392b;
}

/* --- Input Fields --- */
QLineEdit {
    background-color: #2c3e50;
    border: 1px solid #3b4d61;
    padding: 5px;
    color: #f0f8ff;
    border-radius: 4px;
}
QLineEdit:focus {
    border: 1px solid #87CEEB;
}

/* --- Log Text Area --- */
QPlainTextEdit {
    background-color: #141f2b; /* Even darker for focus */
    border: 1px solid #2c3e50;
    color: #f0f8ff;
    font-family: 'Consolas', 'Courier New', monospace;
    border-radius: 4px;
}

/* --- Checkboxes & Dropdowns --- */
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #87CEEB;
    background-color: #2c3e50;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background-color: #87CEEB;
    image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSIjMWUyYTNhIiBkPSJNOSAxNi4xN0w0LjgzIDEybC0xLjQyIDEuNDFMOSAxOUwyMSA3bC0xLjQxLTEuNDF6Ii8+PC9zdmc+"); /* Dark checkmark */
}
QCheckBox::indicator:disabled {
    border: 1px solid #5c6a79;
    background-color: #3b4d61;
}

QComboBox {
    background-color: #2c3e50;
    border: 1px solid #3b4d61;
    padding: 5px;
    border-radius: 4px;
}
QComboBox:on {
    border: 1px solid #87CEEB;
}
QComboBox::drop-down {
    border-left: 1px solid #3b4d61;
}
QComboBox QAbstractItemView {
    background-color: #2c3e50;
    border: 1px solid #87CEEB;
    selection-background-color: #87CEEB;
    selection-color: #1e2a3a;
}

/* --- Tabs --- */
QTabWidget::pane {
    border: 1px solid #2c3e50;
    border-top: none;
}
QTabBar::tab {
    background: #1e2a3a;
    border: 1px solid #2c3e50;
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 8px 15px;
    color: #f0f8ff;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #2c3e50;
    color: #ffffff;
    border-bottom: 3px solid #87CEEB;
}
QTabBar::tab:!selected:hover {
    background: #3b4d61;
}
"""

class TrainingGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORASDXL Trainer")
        self.setMinimumSize(QtCore.QSize(900, 750))
        self.config_path = "user_config.json"
        self.widgets = {}
        self.unet_layer_checkboxes = {}
        self.training_process = None
        self.resume_widgets_container = None
        self.tabs_loaded = set()
        self.loaded_config = {}
        self._read_config_from_file()
        self._setup_ui()
        self._on_tab_change(0)

    def paintEvent(self, event: QtGui.QPaintEvent):
        """Allows stylesheets to use properties like 'border' on the main window."""
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
        
    def _setup_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(2, 2, 2, 2)

        # Use a modern, clean title label instead of ASCII art
        title_label = QtWidgets.QLabel("AOZORA SDXL Trainer")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title_label)

        self.tab_view = QtWidgets.QTabWidget()
        self.tab_view.currentChanged.connect(self._on_tab_change)
        self.main_layout.addWidget(self.tab_view)
        self.tab_view.addTab(QtWidgets.QWidget(), "1. Data & Model")
        self.tab_view.addTab(QtWidgets.QWidget(), "2. Training Parameters")
        self.tab_view.addTab(QtWidgets.QWidget(), "3. Layer Targeting")

        self.log_textbox = QtWidgets.QPlainTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setMinimumHeight(250)
        # Add the log box with a stretch factor so it expands to fill available vertical space
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
        button_layout.addStretch() # Pushes start/stop buttons to the right
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        self.main_layout.addLayout(button_layout)

    def _on_tab_change(self, index):
        """Lazy-loads tab content for faster startup and a cleaner UI build process."""
        tab_name = self.tab_view.tabText(index)
        if tab_name not in self.tabs_loaded:
            tab_widget = self.tab_view.widget(index)
            if tab_name == "1. Data & Model": self._populate_data_model_tab(tab_widget)
            elif tab_name == "2. Training Parameters": self._populate_training_params_tab(tab_widget)
            elif tab_name == "3. Layer Targeting": self._populate_layer_targeting_tab(tab_widget)
            self._apply_config_to_widgets()
            self.tabs_loaded.add(tab_name)

    def _populate_data_model_tab(self, tab):
        layout = QtWidgets.QHBoxLayout(tab)
        
        paths_group = QtWidgets.QGroupBox("File & Directory Paths")
        paths_layout = QtWidgets.QFormLayout(paths_group)
        self._create_path_option(paths_layout, "SINGLE_FILE_CHECKPOINT_PATH", "Base Model (.safetensors)", "Path to the base SDXL model.", "file_safetensors")
        self._create_path_option(paths_layout, "INSTANCE_DATA_DIR", "Dataset Directory", "Folder containing training images.", "folder")
        self._create_path_option(paths_layout, "OUTPUT_DIR", "Output Directory", "Folder where checkpoints will be saved.", "folder")

        data_group = QtWidgets.QGroupBox("Data Processing & Batching")
        data_layout = QtWidgets.QFormLayout(data_group)
        self._create_bool_option(data_layout, "FORCE_RECACHE_LATENTS", "Force Recache Latents", "Re-creates VAE latent caches.")
        self._create_entry_option(data_layout, "CACHING_BATCH_SIZE", "Caching Batch Size", "Adjust based on VRAM.")
        self._create_entry_option(data_layout, "BATCH_SIZE", "Training Batch Size", "Almost always 1 for this script.")
        self.widgets["BATCH_SIZE"].setEnabled(False)
        self._create_entry_option(data_layout, "NUM_WORKERS", "Dataloader Workers", "Set to 0 on Windows if you have issues.")
        self._create_entry_option(data_layout, "BUCKET_SIZES", "Bucket Sizes (comma-separated)", "Resolutions for image bucketing.")
        
        layout.addWidget(paths_group)
        layout.addWidget(data_group)
        layout.addStretch()

    def _populate_training_params_tab(self, tab):
        layout = QtWidgets.QHBoxLayout(tab)
        left_layout = QtWidgets.QVBoxLayout()
        
        core_group = QtWidgets.QGroupBox("Core Training")
        core_layout = QtWidgets.QFormLayout(core_group)
        self._create_entry_option(core_layout, "MAX_TRAIN_STEPS", "Max Training Steps:", "Total number of training steps.")
        self._create_entry_option(core_layout, "GRADIENT_ACCUMULATION_STEPS", "Gradient Accumulation:", "Simulates a larger batch size.")
        self._create_dropdown_option(core_layout, "MIXED_PRECISION", "Mixed Precision:", ["bfloat16", "fp16"], "bfloat16 for modern GPUs, fp16 for older.")
        self._create_entry_option(core_layout, "SEED", "Seed:", "Ensures reproducible training.")
        self._create_entry_option(core_layout, "SAVE_EVERY_N_STEPS", "Save Every N Steps:", "How often to save a checkpoint.")
        self._create_bool_option(core_layout, "RESUME_TRAINING", "Resume from Checkpoint", "Allows resuming from a checkpoint.", self.toggle_resume_widgets)
        
        # Container for resume options, to be shown/hidden
        self.resume_widgets_container = QtWidgets.QWidget()
        resume_layout = QtWidgets.QFormLayout(self.resume_widgets_container)
        resume_layout.setContentsMargins(0, 5, 0, 0)
        self._create_path_option(resume_layout, "RESUME_MODEL_PATH", "Resume Model:", "The .safetensors checkpoint file.", "file_safetensors")
        self._create_path_option(resume_layout, "RESUME_STATE_PATH", "Resume State:", "The .pt optimizer state file.", "file_pt")
        core_layout.addRow(self.resume_widgets_container)
        self.resume_widgets_container.hide()
        left_layout.addWidget(core_group)
        left_layout.addStretch()

        right_layout = QtWidgets.QVBoxLayout()
        lr_group = QtWidgets.QGroupBox("Learning Rate & Optimizer")
        lr_layout = QtWidgets.QFormLayout(lr_group)
        self._create_entry_option(lr_layout, "UNET_LEARNING_RATE", "UNet Learning Rate:", "e.g., 8e-7")
        self._create_entry_option(lr_layout, "TEXT_ENCODER_LEARNING_RATE", "Text Encoder LR:", "e.g., 4e-7")
        self._create_dropdown_option(lr_layout, "LR_SCHEDULER_TYPE", "LR Scheduler:", ["cosine", "linear"], "How LR changes over time.")
        self._create_entry_option(lr_layout, "LR_WARMUP_PERCENT", "LR Warmup Percent:", "e.g., 0.1 for 10%")
        self._create_entry_option(lr_layout, "CLIP_GRAD_NORM", "Clip Gradient Norm:", "Helps prevent unstable training. 1.0 is safe.")

        adv_group = QtWidgets.QGroupBox("Advanced")
        adv_layout = QtWidgets.QFormLayout(adv_group)
        self._create_bool_option(adv_layout, "USE_MIN_SNR_GAMMA", "Use Min-SNR Gamma", "Recommended for v-prediction models.", self.toggle_min_snr_gamma_widget)
        self._create_entry_option(adv_layout, "MIN_SNR_GAMMA", "Min-SNR Gamma Value:", "Common range is 5.0 to 20.0.")
        self.widgets["MIN_SNR_GAMMA"].setEnabled(False)

        right_layout.addWidget(lr_group)
        right_layout.addWidget(adv_group)
        right_layout.addStretch()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

    def _populate_layer_targeting_tab(self, tab):
        main_layout = QtWidgets.QVBoxLayout(tab)
        
        # Top layout for buttons
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addStretch()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(lambda: self.toggle_all_unet_checkboxes(True))
        deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.toggle_all_unet_checkboxes(False))
        top_layout.addWidget(select_all_btn)
        top_layout.addWidget(deselect_all_btn)
        main_layout.addLayout(top_layout)

        # Scroll area for the many layer checkboxes
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)

        all_unet_targets = {
            "Attention Blocks": {"attn1": "Self-Attention", "attn2": "Cross-Attention", "mid_block.attentions": "Mid-Block Attention", "ff": "Feed-Forward Networks"},
            "Attention Sub-Layers (Advanced)": {"to_q": "Query Projection", "to_k": "Key Projection", "to_v": "Value Projection", "to_out.0": "Output Projection"},
            "UNet Embeddings (Non-Text)": {"time_embedding": "Time Embedding", "time_emb_proj": "Time Embedding Projection", "add_embedding": "Added Conditional Embedding"},
            "Convolutional & ResNet Layers": {"conv_in": "Input Conv", "resnets.conv1": "ResNet Conv1", "resnets.conv2": "ResNet Conv2", "resnets.conv_shortcut": "ResNet Skip Conv", "downsamplers.0.conv": "Downsampler Convs", "upsamplers.0.conv": "Upsampler Convs", "conv_out": "Output Conv"},
            "Normalization Layers (Experimental)": {"norm1": "ResNet GroupNorm1", "norm2": "ResNet GroupNorm2", "conv_norm_out": "Output GroupNorm"}
        }
        for group_name, targets in all_unet_targets.items():
            label = QtWidgets.QLabel(f"<b>{group_name}</b>")
            label.setStyleSheet("margin-top: 10px; border: none;")
            scroll_layout.addWidget(label)
            for key, text in targets.items():
                cb = QtWidgets.QCheckBox(f"{text} (keyword: '{key}')")
                cb.setStyleSheet("margin-left: 15px;") # Indent checkboxes
                scroll_layout.addWidget(cb)
                self.unet_layer_checkboxes[key] = cb
        scroll_layout.addStretch()

        main_layout.addWidget(scroll_area)

        te_group = QtWidgets.QGroupBox("Text Encoder Training")
        te_layout = QtWidgets.QFormLayout(te_group)
        self._create_dropdown_option(te_layout, "TEXT_ENCODER_TRAIN_TARGET", "Target:", ["none", "token_embedding_only", "full"], "Controls which parts of the text encoders are trained.")
        main_layout.addWidget(te_group)

    def _create_entry_option(self, layout, key, label, tooltip_text):
        entry = QtWidgets.QLineEdit()
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        layout.addRow(label_widget, entry)
        self.widgets[key] = entry

    def _create_path_option(self, layout, key, label, tooltip_text, file_type):
        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        entry = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("Browse...")
        hbox.addWidget(entry, stretch=1) # Allow entry to expand
        hbox.addWidget(button)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        layout.addRow(label_widget, container)
        button.clicked.connect(lambda: self._browse_path(entry, file_type))
        self.widgets[key] = entry

    def _browse_path(self, entry_widget, file_type):
        path = ""
        if file_type == "folder":
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        elif file_type == "file_safetensors":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", "", "Safetensors Files (*.safetensors)")
        elif file_type == "file_pt":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select State", "", "PyTorch State Files (*.pt)")
        if path:
            entry_widget.setText(path)

    def _create_bool_option(self, layout, key, label, tooltip_text, command=None):
        checkbox = QtWidgets.QCheckBox(label)
        checkbox.setToolTip(tooltip_text)
        layout.addRow(checkbox)
        self.widgets[key] = checkbox
        if command:
            checkbox.stateChanged.connect(command)

    def _create_dropdown_option(self, layout, key, label, values, tooltip_text):
        dropdown = QtWidgets.QComboBox()
        dropdown.addItems(values)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip_text)
        layout.addRow(label_widget, dropdown)
        self.widgets[key] = dropdown

    def _read_config_from_file(self):
        # Start with default config values
        config = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    # Update default config with user's saved settings
                    config.update(user_config)
            except (json.JSONDecodeError, TypeError) as e:
                self.log(f"Warning: Could not read {self.config_path}. Error: {e}. Using default settings.")
        self.loaded_config = config

    def _apply_config_to_widgets(self):
        # Apply UNET_TRAIN_TARGETS separately as they are special
        unet_targets = self.loaded_config.get("UNET_TRAIN_TARGETS", [])
        for key, checkbox in self.unet_layer_checkboxes.items():
            checkbox.setChecked(key in unet_targets)

        for key, widget in self.widgets.items():
            value = self.loaded_config.get(key)
            if value is None: continue

            if isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(", ".join(map(str, value)) if isinstance(value, list) else str(value))
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)

        # Ensure dependent widgets update their state after loading config
        if "USE_MIN_SNR_GAMMA" in self.widgets: self.toggle_min_snr_gamma_widget()
        if "RESUME_TRAINING" in self.widgets: self.toggle_resume_widgets()

    def save_config(self):
        config_to_save = {}
        default_map = {k: v for k, v in default_config.__dict__.items() if not k.startswith('__')}

        # Handle UNET_TRAIN_TARGETS checkboxes
        if self.unet_layer_checkboxes:
            config_to_save["UNET_TRAIN_TARGETS"] = [k for k, cb in self.unet_layer_checkboxes.items() if cb.isChecked()]

        for key, widget in self.widgets.items():
            default_val = default_map.get(key)
            val = None

            if isinstance(widget, QtWidgets.QLineEdit):
                val_str = widget.text().strip()
                # This complex logic handles converting the string from a QLineEdit
                # back into its proper type (int, float, list of ints) for JSON serialization.
                if isinstance(default_val, list):
                    parts = [item.strip() for item in val_str.split(',') if item.strip()]
                    try: val = [int(p) for p in parts]
                    except ValueError: val = parts
                elif isinstance(default_val, (int, float)):
                    try: val = type(default_val)(val_str)
                    except (ValueError, TypeError): val = val_str
                else:
                    val = val_str
            elif isinstance(widget, QtWidgets.QCheckBox):
                val = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                val = widget.currentText()
            
            # Only save if the value is different from the default to keep the JSON file clean
            if val != default_val:
                config_to_save[key] = val

        with open(self.config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        self.log("Configuration saved to user_config.json")

    def restore_defaults(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
            self.log(f"Removed {self.config_path}. Restoring defaults.")
        self._read_config_from_file()
        self._apply_config_to_widgets()

    def toggle_all_unet_checkboxes(self, state):
        for cb in self.unet_layer_checkboxes.values():
            cb.setChecked(state)

    def toggle_min_snr_gamma_widget(self):
        if "MIN_SNR_GAMMA" in self.widgets and "USE_MIN_SNR_GAMMA" in self.widgets:
            self.widgets["MIN_SNR_GAMMA"].setEnabled(self.widgets["USE_MIN_SNR_GAMMA"].isChecked())

    def toggle_resume_widgets(self):
        if "RESUME_TRAINING" in self.widgets and self.resume_widgets_container:
            is_checked = self.widgets["RESUME_TRAINING"].isChecked()
            self.resume_widgets_container.setVisible(is_checked)
            self.widgets["RESUME_MODEL_PATH"].setEnabled(is_checked)
            self.widgets["RESUME_STATE_PATH"].setEnabled(is_checked)

    def log(self, message):
        self.log_textbox.appendPlainText(message.strip())
        self.log_textbox.verticalScrollBar().setValue(self.log_textbox.verticalScrollBar().maximum())

    def start_training(self):
        self.save_config()
        self.log("\n" + "="*50 + "\nStarting training process...\n" + "="*50)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.training_process = QtCore.QProcess(self)
        self.training_process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.training_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.training_process.finished.connect(self.training_finished)
        
        # Use sys.executable to ensure the same Python env is used. -u is for unbuffered output.
        try:
            self.training_process.start(sys.executable, ["-u", "train.py"])
            if not self.training_process.waitForStarted(5000):
                self.log(f"ERROR: Failed to start training process: {self.training_process.errorString()}")
                self.training_finished()
        except Exception as e:
            self.log(f"CRITICAL ERROR: Could not launch Python process: {e}")
            self.training_finished()

    def handle_stdout(self):
        data = self.training_process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='ignore')
        self.log(text)

    def stop_training(self):
        if self.training_process and self.training_process.state() == QtCore.QProcess.ProcessState.Running:
            self.log("--- Sending termination signal to training process... ---")
            self.training_process.kill()
        else:
            self.log("No active training process to stop.")

    def training_finished(self):
        exit_code = self.training_process.exitCode()
        status = "successfully" if exit_code == 0 else f"with an error (Code: {exit_code})"
        self.log(f"\n" + "="*50 + f"\nTraining process finished {status}.\n" + "="*50)

        self.training_process = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    main_win = TrainingGUI()
    main_win.show()
    sys.exit(app.exec())