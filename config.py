# ====================================================================================
# DEFAULT CONFIGURATION
# All values here are the default settings for the UNet-only trainer.
# The GUI will load these and save user changes to a separate user_config.json file.
# ====================================================================================

# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./Aozora-XL_vPredV1-Final.safetensors"
INSTANCE_DATA_DIR = "./DatasetV1/"
OUTPUT_DIR = "./sdxl_finetune_output"

# --- Caching & Data ---
FORCE_RECACHE_LATENTS = False
CACHING_BATCH_SIZE = 3
BATCH_SIZE = 1 # Recommended to keep at 1 for this script
NUM_WORKERS = 4 # Set to 0 on Windows if you encounter DataLoader errors

# --- NEW: Aspect Ratio Bucketing ---
# The total number of pixels for each bucket (e.g., 1024*1024 = 1,048,576)
TARGET_PIXEL_AREA = 1327104
BUCKET_ASPECT_RATIOS = [1.0, 1.5, 0.66, 1.33, 0.75, 1.77, 0.56]

# --- Training Parameters ---
MAX_TRAIN_STEPS = 35000
GRADIENT_ACCUMULATION_STEPS = 64
MIXED_PRECISION = "bfloat16" # Use "fp16" for older GPUs (e.g., 20-series)
SEED = 42
SAVE_EVERY_N_STEPS = 5000

# --- Resume from Checkpoint (Manual Paths) ---
RESUME_TRAINING = False
# These paths are used ONLY if RESUME_TRAINING is True
RESUME_MODEL_PATH = "" # e.g., "./sdxl_finetune_output/checkpoints/checkpoint_step_5000.safetensors"
RESUME_STATE_PATH = "" # e.g., "./sdxl_finetune_output/checkpoints/training_state_step_5000.pt"

# --- Learning Rate & Optimizer ---
UNET_LEARNING_RATE = 8e-7
LR_SCHEDULER_TYPE = "cosine"
LR_WARMUP_PERCENT = 0.25 # e.g., 0.1 for 10% of total steps
CLIP_GRAD_NORM = 1.0

# --- Layer Training Targets (UNet-Only) ---
# These are the default layers that will be trained. The GUI shows all options.
UNET_TRAIN_TARGETS = [
    "attn1",
    "attn2",
    "ff",
    "time_emb_proj",
    "conv_in",
    "conv_out",
    "time_embedding",
]

# --- Advanced ---
USE_MIN_SNR_GAMMA = True
MIN_SNR_GAMMA = 20.0