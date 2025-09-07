# Aozora SDXL Trainer

A user-friendly graphical interface (GUI) for fine-tuning Stable Diffusion XL (SDXL) models. This tool is built on a powerful, memory-efficient training script that allows for targeted layer training, latent caching, and easy configuration management.

The goal of this project is to provide a simple yet powerful way for users to fine-tune SDXL without needing to edit scripts or manage complex command-line arguments. Everything is controlled through the GUI.

- **Please note that the model only works with vpred models supporting the standard format, merged or lora merged models may not train and layer selection is hand crafted for my model**

<img src="https://i.imgur.com/IK57j6r.png" alt="Gui" width="850"/>
<img src="https://i.imgur.com/Sv54FgU.png" alt="Gui" width="850"/>


## Features

- **Intuitive GUI:** All training parameters are accessible through a clean, tabbed interface.
- **Layer-Specific Training:** Precisely control which parts of the UNet and Text Encoders are trained.
- **Efficient Latent Caching:** Pre-computes and caches VAE latents to accelerate the training process.
- **Resume Training:** Easily resume a training run from a saved checkpoint and optimizer state.
- **Configuration Management:** Save your settings to a `user_config.json` file and restore defaults with a single click.
- **Memory Optimization:** Utilizes gradient checkpointing, mixed precision (`bf16`/`fp16`), and Adafactor to train on consumer-grade GPUs.
- **Automated Setup:** Comes with a `setup.bat` script to create a portable environment and install all dependencies.

## Prerequisites

- **OS:** Windows 10/11
- **Python:** Python 3.10 (The setup script is configured for this version)
- **GPU:** An NVIDIA GPU with at least 12GB of VRAM is recommended. More VRAM allows for more aggressive training configurations.
- **NVIDIA Drivers:** Recent NVIDIA drivers and CUDA toolkit are required.

---

## Installation

The installation is fully automated using the provided batch script.

### Clone the Repository

Download this project to your computer. You can do this by clicking the green "Code" button on GitHub and selecting "Download ZIP", or by using Git:

    git clone https://github.com/Hysocs/Aozora_SDXL_Training.git
    cd Aozora_SDXL_Training

### Run the Setup Script

Navigate to the project directory and double-click `setup.bat`.

You will be presented with a choice:

    ====================================================
           SDXL Training Environment Installer
    ====================================================

    Please choose your installation type:

      [1] Install WITH Flash Attention (Recommended for NVIDIA RTX cards)
      [2] Install WITHOUT Flash Attention (For AMD or older NVIDIA cards)

    Enter your choice [1 or 2]:

- **Option 1 (Flash Attention):** Recommended for modern NVIDIA GPUs (RTX 20, 30, 40 series). It significantly speeds up training and reduces VRAM usage.
- **Option 2 (No Flash Attention):** Choose this if you have an older NVIDIA GPU, an AMD GPU, or if the Flash Attention installation fails.

The script will then:
- Create a local Python virtual environment in a folder named `portable_Venv`.
- Install all necessary libraries, including PyTorch with CUDA support.
- Download and install Flash Attention if selected.

This process will take several minutes as it downloads large files. Once it's finished, you'll see a "SETUP COMPLETE!" message.

---

## How to Use

### Step 1: Prepare Your Data and Model

Before launching the GUI, get your files in order.

1. **Base Model:** Place your base SDXL model (e.g., `Aozora-XL_vPredV1-Final.safetensors`) in the main project folder or a location you can easily navigate to.

2. **Dataset:** Create a dataset folder. Inside this folder, you can have subfolders for organization. The script will search recursively for images.
   - Each image should have a corresponding `.txt` file with the same name containing its caption.
   - If no `.txt` file is found, the image's filename (with underscores replaced by spaces) will be used as the caption.

**Example Directory Structure:**

    Aozora_SDXL_Training/
    ├── DatasetV1/
    │   ├── concept_one/
    │   │   ├── image001.png
    │   │   ├── image001.txt   (caption: "a photo of a red car")
    │   │   ├── image002.webp
    │   │   └── image002.txt   (caption: "a drawing of a blue boat")
    │   └── concept_two/
    │       ├── my_character.jpg
    │       └── my_character.txt (caption: "1girl, solo, best quality")
    ├── Aozora-XL_vPredV1-Final.safetensors
    ├── gui.py
    ├── train.py
    ├── setup.bat
    └── start_gui.bat

### Step 2: Launch the GUI

Double-click the `start_gui.bat` file. This will activate the local Python environment and open the aozora Trainer GUI.

### Step 3: Configure Training

The GUI is organized into three tabs. Go through each one to set up your training run.

#### Tab 1: Data & Model

- **Base Model:** Click "Browse..." and select your base `.safetensors` model file.
- **Dataset Directory:** Click "Browse..." and select the main folder containing your training images (e.g., `DatasetV1`).
- **Output Directory:** Choose a folder where the final model and checkpoints will be saved.
- **Force Recache Latents:** Check this box if you have changed your dataset or want to re-process all images. The first time you run the trainer on a new dataset, this happens automatically.

#### Tab 2: Training Parameters

- **Max Training Steps:** The total number of steps the training will run for.
- **Save Every N Steps:** How often to save a full checkpoint. These are useful for resuming or for testing progress.
- **UNet Learning Rate:** The learning rate for the main image generation network (UNet). A good starting point is `8e-7`.
- **Text Encoder LR:** The learning rate for the text encoders. A lower value like `4e-7` is often a safe bet.
- **Resume from Checkpoint:** Check this box to enable the fields for resuming a previous run. You will need to provide the path to both the `.safetensors` checkpoint model and the `.pt` training state file.

#### Tab 3: Layer Targeting

This is where you can fine-tune what gets trained.

- **UNet Layers:** Check the boxes for the UNet layers you want to train. For a general-purpose fine-tune, training all **Attention Blocks** is a common strategy. You can use the "Select All" and "Deselect All" buttons for convenience.
- **Text Encoder Training:**
  - `none`: Does not train the text encoders at all. Fastest, but learns less about your text concepts.
  - `token_embedding_only`: **(Recommended)** Only trains the word embeddings. This is very effective for teaching the model new concepts or styles associated with specific words.
  - `full`: Trains the entire text encoders. This is very powerful but requires more VRAM and can be prone to overfitting.

### Step 4: Start Training

1. **Save Config:** Click the **Save Config** button to save your current settings to `user_config.json`. This is good practice and allows you to easily load your settings next time.
2. **Start Training:** Click the **Start Training** button.

The log window at the bottom of the GUI will fill with information. You will first see the latent caching process (if needed), followed by the main training loop with a progress bar.

You can click the **Stop Training** button at any time to gracefully terminate the process.

## Output

- **Final Model:** A fully trained `.safetensors` model will be saved in your specified **Output Directory** (e.g., `Aozora-XL_vPredV1-Final_trained_step34860.safetensors`).
- **Checkpoints:** Intermediate models and training states (`.pt` files) will be saved inside a `checkpoints` subfolder within your output directory. These are used for resuming training.