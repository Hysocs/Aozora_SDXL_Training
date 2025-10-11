# ==================================================================================== 
# DEFAULT CONFIGURATION 
# Simplified configuration matching the refactored training script
# ====================================================================================

# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./model.safetensors"
VAE_PATH = ""  # Optional separate VAE path, leave empty to use VAE from model
OUTPUT_DIR = "./output"

# --- Resume Training ---
RESUME_TRAINING = False
RESUME_MODEL_PATH = ""
RESUME_STATE_PATH = ""

# --- Dataset Configuration ---
INSTANCE_DATASETS = [
    {
        "path": "./data",
        "repeats": 1,
    }
]

# --- Caching & Data Loaders ---
CACHING_BATCH_SIZE = 2
NUM_WORKERS = 4

# --- Aspect Ratio Bucketing ---
TARGET_PIXEL_AREA = 1048576  # 1024*1024

# --- Core Training Parameters ---
NOISE_SCHEDULER = "DDPMScheduler"
PREDICTION_TYPE = "v_prediction"
BETA_SCHEDULE = "scaled_linear"
MAX_TRAIN_STEPS = 10000
LEARNING_RATE = 3e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "float16"
CLIP_GRAD_NORM = 1.0
SEED = 42

# --- Saving ---
SAVE_EVERY_N_STEPS = 1000

# --- UNet Layer Exclusion (Blacklist) ---
UNET_EXCLUDE_TARGETS = "conv1, conv2"  # Comma-separated keywords to exclude from training

# --- Learning Rate Scheduler ---
LR_CUSTOM_CURVE = [
    [0.0, 0.0],
    [0.05, 8.0e-7],
    [0.85, 8.0e-7],
    [1.0, 1.0e-7]
]
LR_GRAPH_MIN = 0.0
LR_GRAPH_MAX = 1.0e-6

# --- Advanced ---
MEMORY_EFFICIENT_ATTENTION = "xformers"
USE_ZERO_TERMINAL_SNR = True

# --- Optimizer Configuration ---
OPTIMIZER_TYPE = "raven"  # "raven" or "adafactor"

# Raven Optimizer Parameters
RAVEN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "debias_strength": 0.3,
    "use_grad_centralization": False,  # NEW
    "gc_alpha": 1.0  # NEW
}

# Adafactor Optimizer Parameters
ADAFACTOR_PARAMS = {
    "eps": [1e-30, 1e-3],
    "clip_threshold": 1.0,
    "decay_rate": -0.8,
    "beta1": None,
    "weight_decay": 0.01,
    "scale_parameter": True,
    "relative_step": False,
    "warmup_init": False
}

# --- Gradient Spike Detection ---
GRAD_SPIKE_THRESHOLD_HIGH = 75.0
GRAD_SPIKE_THRESHOLD_LOW = 0.2