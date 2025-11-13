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
NUM_WORKERS = 0
SHOULD_UPSCALE = False
# --- Aspect Ratio Bucketing ---
TARGET_PIXEL_AREA = 1048576  # 1024*1024
MAX_AREA_TOLERANCE = 1.1

# --- Core Training Parameters ---
NOISE_SCHEDULER = "DDPMScheduler"
PREDICTION_TYPE = "v_prediction"
BETA_SCHEDULE = "scaled_linear"
MAX_TRAIN_STEPS = 10000
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

TIMESTEP_SAMPLING_METHOD = "Random Integer (Default)"

TIMESTEP_SAMPLING_MIN = 0
TIMESTEP_SAMPLING_MAX = 999
TIMESTEP_SAMPLING_GRAD_MIN = 0.1
TIMESTEP_SAMPLING_GRAD_MAX = 0.9

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
    "use_grad_centralization": False,
    "gc_alpha": 1.0
}

# --- Gradient Spike Detection ---
GRAD_SPIKE_THRESHOLD_HIGH = 75.0
GRAD_SPIKE_THRESHOLD_LOW = 0.2

# --- Noise Enhancements ---
NOISE_TYPE = "Semantic"  # Options: "Default", "Offset", "Semantic"
SEMANTIC_NOISE_NORMALIZE = True

# --- Parameters for "Offset" Noise ---
NOISE_OFFSET = 0.05 # Only used if NOISE_TYPE is "Offset"

# --- Parameters for "Semantic" Noise ---
# These settings are only active when NOISE_TYPE is "Semantic"
SEMANTIC_NOISE_BLEND = 0.5     # 0.0 = 100% Character, 1.0 = 100% Detail, 0.5 = Equal mix
SEMANTIC_NOISE_GLOBAL_STRENGTH = 0.8  # Overall modulation strength (1.0 = standard, 2.0 = strong)