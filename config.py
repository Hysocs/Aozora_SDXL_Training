# ====================================================================================
# DEFAULT CONFIGURATION
# All values here are the default settings, matched to the original non-GUI script.
# The GUI will load these and save user changes to a separate user_config.json file.
# ====================================================================================

# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./Aozora-XL_vPredV1-Final.safetensors"
INSTANCE_DATA_DIR = "./DatasetV1/"
OUTPUT_DIR = "./sdxl_finetune_output"

# --- Caching & Data ---
FORCE_RECACHE_LATENTS = False
CACHING_BATCH_SIZE = 3
BUCKET_SIZES = [512, 576, 640, 704, 768, 800, 832, 896, 960, 1024, 1152]
BATCH_SIZE = 1
NUM_WORKERS = 4

# --- Training Parameters ---
MAX_TRAIN_STEPS = 34860
GRADIENT_ACCUMULATION_STEPS = 64
MIXED_PRECISION = "bfloat16"
SEED = 42
SAVE_EVERY_N_STEPS = 5000

# --- Resume from Checkpoint (Manual Paths) ---
RESUME_TRAINING = False
# These paths are used ONLY if RESUME_TRAINING is True
RESUME_MODEL_PATH = "" # e.g., "./sdxl_finetune_output/checkpoints/checkpoint_step_5000.safetensors"
RESUME_STATE_PATH = "" # e.g., "./sdxl_finetune_output/checkpoints/training_state_step_5000.pt"

# --- Learning Rate & Optimizer ---
UNET_LEARNING_RATE = 8e-7
TEXT_ENCODER_LEARNING_RATE = 4e-7
LR_SCHEDULER_TYPE = "cosine"
LR_WARMUP_PERCENT = 0.25
CLIP_GRAD_NORM = 1.0

# --- Layer Training Targets (matched to original script) ---
# These are the default layers that will be trained. The GUI shows all options.
UNET_TRAIN_TARGETS = [
    "attn1",          # Self-Attention
    "attn2",          # Cross-Attention
    "ff",             # Feed-Forward Networks
    "time_emb_proj",  # Time Embedding Projection
    "conv_in",        # UNet Input Convolution
    "conv_out",       # UNet Output Convolution
    "time_embedding", # Main Time Embedding (Linear)
]
# Options for Text Encoders: "none", "token_embedding_only", "full"
TEXT_ENCODER_TRAIN_TARGET = "token_embedding_only"

# --- Advanced ---
USE_MIN_SNR_GAMMA = True
MIN_SNR_GAMMA = 20.0