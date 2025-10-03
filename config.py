# ==================================================================================== 
# DEFAULT CONFIGURATION 
# All values here are the default settings for the UNet-only trainer. 
# The GUI will load these and save user changes to a separate user_config.json file. 
# ====================================================================================

# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./Aozora-XL_vPredV1-Final.safetensors"
OUTPUT_DIR = "./sdxl_finetune_output"

# --- Dataset Configuration ---
INSTANCE_DATASETS = [
    {
        "path": "./DatasetV1/",
        "repeats": 1,
    }
]

# --- Caching & Data Loaders ---
CACHING_BATCH_SIZE = 3
BATCH_SIZE = 1
NUM_WORKERS = 4

# --- Aspect Ratio Bucketing ---
TARGET_PIXEL_AREA = 1327104

# --- Training Parameters ---
PREDICTION_TYPE = "v_prediction"
BETA_SCHEDULE = "scaled_linear"
MAX_TRAIN_STEPS = 35000
GRADIENT_ACCUMULATION_STEPS = 64
CLIP_GRAD_NORM = 1.25
MIXED_PRECISION = "bfloat16"
SEED = 42
SAVE_EVERY_N_STEPS = 5000

# --- Resume from Checkpoint (Manual Paths) ---
RESUME_TRAINING = False
RESUME_MODEL_PATH = ""
RESUME_STATE_PATH = ""

# --- Learning Rate & Optimizer ---
LR_CUSTOM_CURVE = [
    [0.0, 0.0],
    [0.05, 8.0e-7],
    [0.85, 8.0e-7],
    [1.0, 1.0e-7]
]
LR_GRAPH_MIN = 0.0
LR_GRAPH_MAX = 1.0e-6
WEIGHT_DECAY = 0.01

# --- Layer Training Targets (UNet-Only) ---
UNET_TRAIN_TARGETS = [
    "attn1",
    "attn2",
    "ff",
    "time_emb_proj",
    "conv_in",
    "conv_out",
    "time_embedding",
]

# TIMESTEP CURRICULUM
TIMESTEP_CURRICULUM_MODE = "Fixed"
TIMESTEP_CURRICULUM_START_RANGE = "0, 999"
TIMESTEP_CURRICULUM_END_PERCENT = 80

# DYNAMIC CHALLENGE BALANCING
DYNAMIC_CHALLENGE_PROBE_PERCENT = 5
DYNAMIC_CHALLENGE_TARGET_GRAD_NORM_RANGE = "0.25, 0.90"

# --- Advanced ---
MEMORY_EFFICIENT_ATTENTION = "xformers"
USE_SNR_GAMMA = True
SNR_STRATEGY = "Min-SNR"
SNR_GAMMA = 20.0
MIN_SNR_VARIANT = "min_snr_gamma"
USE_ZERO_TERMINAL_SNR = True
USE_IP_NOISE_GAMMA = True
IP_NOISE_GAMMA = 0.01
USE_COND_DROPOUT = True
COND_DROPOUT_PROB = 0.1

# --- EMA (Exponential Moving Average) Configuration ---
USE_EMA = True
EMA_DECAY = 0.9999  # Standard decay rate (0.999-0.9999 typical range)

# --- Optimizer Configuration ---
OPTIMIZER_TYPE = "Raven"
RAVEN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "use_grad_centralization": False,
    "gc_alpha": 1.0,
}
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