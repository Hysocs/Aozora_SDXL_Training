# ====================================================================================
# DEFAULT CONFIGURATION
# Updated for Custom Timestep Ticket Allocation (Bar Chart)
# ====================================================================================

# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./model.safetensors"
VAE_PATH = ""
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
VRAM_CHUNK_ENABLED = True
VRAM_CHUNK_SIZE = 96
VRAM_CHUNK_CHANCE = 1.0
# --- Aspect Ratio Bucketing ---
SHOULD_UPSCALE = False
TARGET_PIXEL_AREA = 1048576  # 1024*1024
MAX_AREA_TOLERANCE = 1.1

# --- Core Training Parameters ---
NOISE_SCHEDULER = "DDPMScheduler"
PREDICTION_TYPE = "v_prediction"
BETA_SCHEDULE = "scaled_linear"
MAX_TRAIN_STEPS = 10000
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "bfloat16"
CLIP_GRAD_NORM = 1.0
SEED = 42

# --- Saving ---
SAVE_EVERY_N_STEPS = 1000

# --- UNet Layer Exclusion (Blacklist) ---
UNET_EXCLUDE_TARGETS = "conv1, conv2"

# --- Learning Rate Scheduler ---
LR_CUSTOM_CURVE = [
    [0.0, 0.0],
    [0.05, 8.0e-7],
    [0.85, 8.0e-7],
    [1.0, 1.0e-7]
]
LR_GRAPH_MIN = 0.0
LR_GRAPH_MAX = 1.0e-6

# --- Timestep Allocation (Ticket System) ---
# "bin_size": resolution of the bars (e.g., 50 means 0-50, 50-100...)
# "counts": list of integers representing tickets per bin.
TIMESTEP_ALLOCATION = {
    "bin_size": 100,
    "counts": [] # Populated by GUI based on MAX_TRAIN_STEPS
}

# --- Optimizer Configuration ---
OPTIMIZER_TYPE = "raven"
RAVEN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "debias_strength": 0.3,
    "use_grad_centralization": False,
    "gc_alpha": 1.0
}
TITAN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "debias_strength": 0.3,
    "use_grad_centralization": False,
    "gc_alpha": 1.0
}
# --- Noise Configuration ---
NOISE_TYPE = "Default"
NOISE_OFFSET = 0.05

# --- Loss Configuration ---
LOSS_TYPE = "Default"
SEMANTIC_LOSS_BLEND = 0.5
SEMANTIC_LOSS_STRENGTH = 0.8

# --- Advanced & Miscellaneous ---
MEMORY_EFFICIENT_ATTENTION = "xformers"
USE_ZERO_TERMINAL_SNR = True
GRAD_SPIKE_THRESHOLD_HIGH = 75.0
GRAD_SPIKE_THRESHOLD_LOW = 0.2