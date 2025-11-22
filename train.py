import inspect
import re
import os
from pathlib import Path
import gc
import json
from collections import defaultdict, deque
import random
import time
import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, Sampler
from diffusers import StableDiffusionXLPipeline, DDPMScheduler, AutoencoderKL, EulerDiscreteScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file, load_file
from PIL import Image, TiffImagePlugin, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
import logging
import warnings
import argparse
import numpy as np
import cv2
from diffusers.models.attention_processor import AttnProcessor2_0
import multiprocessing
from multiprocessing import Pool, cpu_count
import config as default_config
import threading
import queue
import tomesd

# --- Import the custom optimizer ---
from optimizer.raven import RavenAdamW

# --- Global Settings ---
warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# region Helper Functions & Classes
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"INFO: Set random seed to {seed}")

# --- Semantic Map Functions ---
# --- MODIFICATION: These functions are now used for loss weighting, not noise modulation. ---

def generate_character_map(pil_image):
    """Generates a map focusing on the overall character/object region using color and structure."""
    np_image_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    bilateral = cv2.bilateralFilter(np_image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    mean_a, mean_b = np.mean(a_channel), np.mean(b_channel)
    saliency_a = np.abs(a_channel.astype(np.float32) - mean_a)
    saliency_b = np.abs(b_channel.astype(np.float32) - mean_b)
    color_saliency = saliency_a + saliency_b
    
    if color_saliency.max() > 0:
        color_saliency_norm = color_saliency / color_saliency.max()
    else:
        color_saliency_norm = np.zeros_like(color_saliency, dtype=np.float32)
    
    color_saliency_uint8 = (color_saliency_norm * 255).astype(np.uint8)
    
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(color_saliency_uint8, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    eroded_float = eroded.astype(np.float32) / 255.0
    final_map = cv2.GaussianBlur(eroded_float, (11, 11), 0)
    
    return Image.fromarray(final_map)

def generate_detail_map(pil_image):
    """Generates a map focusing on edges and lineart."""
    np_image_gray = np.array(pil_image.convert("L"))
    sobelx = cv2.Sobel(np_image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(np_image_gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    if magnitude.max() > 0:
        magnitude_norm = magnitude / magnitude.max()
    else:
        magnitude_norm = np.zeros_like(magnitude, dtype=np.float32)
    
    final_map = cv2.GaussianBlur(magnitude_norm.astype(np.float32), (1, 1), 0)
    
    return Image.fromarray(final_map)

# --- MODIFICATION: Renamed from _generate_semantic_noise_map_for_batch ---
def _generate_semantic_map_for_batch(images, blend_factor, strength, target_size, device, dtype, num_channels):
    """
    Generates semantic maps for loss weighting.
    - blend_factor: 0.0 = 100% character map, 1.0 = 100% detail map
    - strength: Overall multiplier for the map. The final map is NOT multiplied by this, it's just used for weighting.
    """
    batch_maps = []
    
    for img in images:
        if img is None:
            # If an image is missing, create a zero map which will result in a weight of 1.0 later
            weight_map_tensor = torch.zeros(target_size, dtype=dtype)
            batch_maps.append(weight_map_tensor)
            continue
            
        char_map_pil = generate_character_map(img)
        detail_map_pil = generate_detail_map(img)

        char_map_np = np.array(char_map_pil).astype(np.float32)
        detail_map_np = np.array(detail_map_pil).astype(np.float32)

        # Raw weighted combination
        combined_map = ((1.0 - blend_factor) * char_map_np) + (blend_factor * detail_map_np)
        
        # Normalize by theoretical max to ensure a clean 0-1 range
        if combined_map.max() > 0:
            combined_map = combined_map / combined_map.max()
        
        # Apply gamma correction for smoother falloff
        weight_map = np.power(combined_map, 0.8)
        
        # Resize and convert to tensor
        combined_map_pil = Image.fromarray(weight_map, mode='F').resize(target_size, Image.Resampling.LANCZOS)
        weight_map_tensor = torch.from_numpy(np.array(combined_map_pil)).float()
        batch_maps.append(weight_map_tensor)

    # Stack, add channel dimension, move to device, and expand for all channels
    final_map_batch = torch.stack(batch_maps).unsqueeze(1).to(device, dtype=dtype)
    return final_map_batch.expand(-1, num_channels, -1, -1)

# --- MODIFICATION: Simplified noise generation ---
def generate_train_noise(latents, config):
    """
    Generates training noise. 'Semantic' noise is no longer handled here.
    """
    base_noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)

    if config.NOISE_TYPE == "Offset":
        strength = float(getattr(config, "NOISE_OFFSET", 0.1))
        if strength <= 0:
            return base_noise
        b, c, h, w = base_noise.shape
        # Create a single offset value for all pixels in each batch item
        offset = torch.randn(b, 1, 1, 1, device=base_noise.device, dtype=base_noise.dtype)
        return base_noise + strength * offset
    
    else: # "Default"
        return base_noise

class TrainingConfig:
    def __init__(self):
        # --- MODIFICATION: Assumes new config values like LOSS_TYPE will be in default_config ---
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        self._load_from_user_config()
        self._type_check_and_correct()
        self.compute_dtype = torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16

    def _load_from_user_config(self):
        parser = argparse.ArgumentParser(description="Load a specific training configuration.")
        parser.add_argument("--config", type=str, help="Path to the user configuration JSON file.")
        args, _ = parser.parse_known_args()
        if args.config:
            path = Path(args.config)
            if path.exists():
                print(f"INFO: Loading configuration from {path}")
                try:
                    with open(path, 'r') as f:
                        user_config = json.load(f)
                    for key, value in user_config.items():
                        setattr(self, key, value)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"ERROR: Could not parse {path}: {e}. Using defaults.")
            else:
                print(f"WARNING: Config {path} not found. Using defaults.")
        else:
            print("INFO: No config file specified. Using defaults from config.py.")

    def _type_check_and_correct(self):
        print("INFO: Checking and correcting configuration types...")
        for key, value in list(self.__dict__.items()):
            if key in ["RESUME_MODEL_PATH", "RESUME_STATE_PATH"] and getattr(self, "RESUME_TRAINING", False):
                if not value or not Path(value).exists():
                    raise FileNotFoundError(f"RESUME_TRAINING is enabled, but {key}='{value}' is not a valid file path.")
            if key == "UNET_EXCLUDE_TARGETS":
                if isinstance(value, str):
                    setattr(self, key, [item.strip() for item in value.split(',') if item.strip()])
                elif isinstance(value, list):
                    setattr(self, key, [item for item in value if item])
                continue
            default_value = getattr(default_config, key, None)
            if default_value is None or isinstance(value, type(default_value)):
                continue
            expected_type = type(default_value)
            if expected_type == bool and isinstance(value, str):
                setattr(self, key, value.lower() in ['true', '1', 't', 'y', 'yes'])
                continue
            try:
                if expected_type == int:
                    setattr(self, key, int(float(value)))
                else:
                    setattr(self, key, expected_type(value))
            except (ValueError, TypeError):
                print(f"WARNING: Could not convert '{key}' value '{value}' to {expected_type.__name__}. "
                    f"Using default value: {default_value}")
                setattr(self, key, default_value)
        print("INFO: Configuration types checked and corrected.")

class CustomCurveLRScheduler:
    def __init__(self, optimizer, curve_points, max_train_steps):
        self.optimizer = optimizer
        self.curve_points = sorted(curve_points, key=lambda p: p[0])
        self.max_train_steps = max(max_train_steps, 1)
        self.current_training_step = 0
        if not self.curve_points:
            raise ValueError("LR_CUSTOM_CURVE cannot be empty")
        if self.curve_points[0][0] != 0.0:
            self.curve_points.insert(0, [0.0, self.curve_points[0][1]])
        if self.curve_points[-1][0] != 1.0:
            self.curve_points.append([1.0, self.curve_points[-1][1]])
        self._update_lr()

    def _interpolate_lr(self, normalized_position):
        normalized_position = max(0.0, min(1.0, normalized_position))
        for i in range(len(self.curve_points) - 1):
            x1, y1 = self.curve_points[i]
            x2, y2 = self.curve_points[i + 1]
            if x1 <= normalized_position <= x2:
                if x2 - x1 == 0:
                    return y1
                t = (normalized_position - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        return self.curve_points[-1][1]

    def _update_lr(self):
        normalized_position = self.current_training_step / max(self.max_train_steps - 1, 1)
        lr = self._interpolate_lr(normalized_position)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self, training_step):
        self.current_training_step = training_step
        self._update_lr()

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

class TrainingDiagnostics:
    def __init__(self, accumulation_steps):
        self.accumulation_steps = accumulation_steps
        self.losses = deque(maxlen=accumulation_steps)

    def step(self, loss):
        if loss is not None:
            self.losses.append(loss)

    def get_average_loss(self):
        if not self.losses: return 0.0
        return sum(self.losses) / len(self.losses)

    def reset(self):
        self.losses.clear()

class AsyncReporter:
    def __init__(self, total_steps, test_param_name):
        self.total_steps = total_steps
        self.test_param_name = test_param_name
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _format_time(self, seconds):
        if seconds is None or not math.isfinite(seconds):
            return "N/A"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def _generate_progress_string(self, current_step, timing_data):
        bar_width = 30
        percentage = (current_step + 1) / self.total_steps
        filled_length = int(bar_width * percentage)
        bar = '#' * filled_length + '-' * (bar_width - filled_length)
        
        s_per_step = timing_data.get('raw_step_time', 0)
        time_spent_str = self._format_time(timing_data.get('elapsed_time'))
        eta_str = self._format_time(timing_data.get('eta'))
        
        loss_val = timing_data.get('loss', 0.0)
        timestep_val = timing_data.get('timestep', 'N/A')
        
        progress_bar = f'Training |{bar}| {current_step + 1}/{self.total_steps} [{percentage:.2%}]'
        step_info = f" [Loss: {loss_val:.4f}, Timestep: {timestep_val}]"
        timing_info = f" [{s_per_step:.2f}s/step, ETA: {eta_str}, Elapsed: {time_spent_str}]"
        
        return progress_bar + step_info + timing_info

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_type, data = self.task_queue.get(timeout=1)
                if task_type == 'log_step':
                    self._handle_log_step(**data)
                elif task_type == 'anomaly':
                    self._handle_anomaly(**data)
                self.task_queue.task_done()
            except queue.Empty:
                continue

    def _handle_log_step(self, global_step, timing_data, diag_data):
        progress_str = self._generate_progress_string(global_step, timing_data)
        print(progress_str, end='\r')

        if diag_data:
            print()
            update_status = "[OK]" if diag_data['update_delta'] > 1e-12 else "[NO UPDATE!]"
            vram_reserved_gb = torch.cuda.memory_reserved() / 1e9
            vram_allocated_gb = torch.cuda.memory_allocated() / 1e9
            
            report_str = (
                f"--- Optimizer Step: {diag_data['optim_step']:<5} | Loss: {diag_data['avg_loss']:<8.5f} | LR: {diag_data['current_lr']:.2e} ---\n"
                f"  Time: {diag_data['optim_step_time']:.2f}s/step | Avg Speed: {diag_data['avg_optim_step_time']:.2f}s/step\n"
                f"  Grad Norm (Raw/Clipped): {diag_data['raw_grad_norm']:<8.4f} / {diag_data['clipped_grad_norm']:<8.4f}\n"
                f"  VRAM: Training={vram_reserved_gb:.2f}GB | Model={vram_allocated_gb:.2f}GB\n"
                f"  Test Param: {self.test_param_name}\n"
                f"  |- Update Magnitude : {diag_data['update_delta']:.4e} {update_status}"
            )
            print(report_str)

    def _handle_anomaly(self, global_step, raw_grad_norm, clipped_grad_norm, low_thresh, high_thresh, paths):
        anomaly_str = (
            f"\n\n{'='*20} GRADIENT ANOMALY DETECTED {'='*20}\n"
            f"  Step: {global_step}\n"
            f"  Raw Gradient Norm: {raw_grad_norm:.4f}\n"
            f"  Clipped Gradient Norm: {clipped_grad_norm:.4f}\n"
            f"  Thresholds (Low/High): {low_thresh} / {high_thresh}\n"
            f"  Images in Accumulated Batch:\n"
            + "".join([f"    - {Path(p).stem}\n" for p in paths]) +
            f"{'='*65}\n"
        )
        print(anomaly_str)

    def log_step(self, global_step, timing_data, diag_data=None):
        data = {'global_step': global_step, 'timing_data': timing_data, 'diag_data': diag_data}
        self.task_queue.put(('log_step', data))

    def check_and_report_anomaly(self, global_step, raw_grad_norm, clipped_grad_norm, config, paths):
        low = getattr(config, "GRAD_SPIKE_THRESHOLD_LOW", 0.0)
        high = getattr(config, "GRAD_SPIKE_THRESHOLD_HIGH", 1000.0)
        if raw_grad_norm > high or raw_grad_norm < low:
            data = {
                'global_step': global_step, 'raw_grad_norm': raw_grad_norm, 'clipped_grad_norm': clipped_grad_norm,
                'low_thresh': low, 'high_thresh': high, 'paths': list(paths)
            }
            self.task_queue.put(('anomaly', data))

    def shutdown(self):
        print("\nShutting down async reporter. Waiting for pending tasks...")
        self.task_queue.join()
        self.stop_event.set()
        self.worker_thread.join()
        print("Async reporter shut down.")

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.total_images = len(self.dataset)

    def __iter__(self):
        g = torch.Generator()
        # SEED + EPOCH: Ensures every epoch has a unique, random order,
        # preventing the model from memorizing the sequence.
        g.manual_seed(self.seed + self.epoch)
        
        indices = list(range(self.total_images))
        
        # --- MODE A: BATCH SIZE 1 (Total Aspect Ratio Mixing) ---
        # If batch size is 1, we ignore buckets completely.
        # We shuffle the entire list of images globally.
        # This results in [Tall, Wide, Square, Wide, Tall] sequences.
        if self.batch_size == 1:
            if self.shuffle:
                indices = torch.randperm(len(indices), generator=g).tolist()
            
            # Wrap each index in a list to create a "batch" of 1
            batches = [[i] for i in indices]

        # --- MODE B: BATCH SIZE > 1 (Bucketed Mixing) ---
        # If batch size > 1, we MUST group by resolution to allow tensor stacking.
        # However, we Shuffle the Batches at the end so we don't process
        # all tall batches at once.
        else:
            # 1. Shuffle indices so buckets fill up randomly
            if self.shuffle:
                indices = torch.randperm(len(indices), generator=g).tolist()

            # 2. Group into buckets (Tall, Wide, Square)
            buckets = defaultdict(list)
            for idx in indices:
                key = self.dataset.bucket_keys[idx]
                buckets[key].append(idx)

            # 3. Create Batches
            batches = []
            for key in buckets:
                bucket_indices = buckets[key]
                for i in range(0, len(bucket_indices), self.batch_size):
                    batch = bucket_indices[i : i + self.batch_size]
                    batches.append(batch)
            
            # 4. Shuffle the Batches (Aspect Ratio Mixing for Batches)
            # This ensures: Batch_1(Tall) -> Batch_2(Wide) -> Batch_3(Square)
            if self.shuffle:
                batch_indices = torch.randperm(len(batches), generator=g).tolist()
                batches = [batches[i] for i in batch_indices]

        self.epoch += 1
        yield from batches

    def __len__(self):
        return (self.total_images + self.batch_size - 1) // self.batch_size

class ResolutionCalculator:
    def __init__(self, target_area, stride=64, should_upscale=False, max_area_tolerance=1.1):
        self.target_area = target_area
        self.stride = stride
        self.should_upscale = should_upscale
        self.max_area = target_area * max_area_tolerance

    def calculate_resolution(self, width, height):
        aspect_ratio = width / height
        if not self.should_upscale:
            h = int(math.sqrt(self.target_area / aspect_ratio) // self.stride) * self.stride
            w = int(h * aspect_ratio // self.stride) * self.stride
            return (w if w > 0 else self.stride, h if h > 0 else self.stride)
        current_area = width * height
        if current_area > self.max_area:
            scale_factor = math.sqrt(self.target_area / current_area)
        elif current_area < self.target_area:
            scale_factor = math.sqrt(self.target_area / current_area)
        else:
            scale_factor = 1.0
        new_w = int((width * scale_factor) // self.stride) * self.stride
        new_h = int((height * scale_factor) // self.stride) * self.stride
        new_w = max(new_w, self.stride)
        new_h = max(new_h, self.stride)
        if new_w * new_h > self.max_area:
            scale_down = math.sqrt(self.max_area / (new_w * new_h))
            new_w = int((new_w * scale_down) // self.stride) * self.stride
            new_h = int((new_h * scale_down) // self.stride) * self.stride
            new_w = max(new_w, self.stride)
            new_h = max(new_h, self.stride)
        return (new_w, new_h)

def resize_to_fit(image, target_w, target_h):
    w, h = image.size
    if w / target_w < h / target_h:
        w_new, h_new = target_w, int(h * target_w / w)
    else:
        w_new, h_new = int(w * target_h / h), target_h
    return image.resize((w_new, h_new), Image.Resampling.LANCZOS).crop(((w_new - target_w) // 2, (h_new - target_h) // 2, (w_new + target_w) // 2, (h_new + target_h) // 2))

def validate_and_assign_resolution(args):
    ip, calculator = args
    try:
        with Image.open(ip) as img:
            img.verify()
        with Image.open(ip) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                print(f"\n[INVALID] Image {ip.name} has invalid dimensions. Skipping.")
                return None
        cp = ip.with_suffix('.txt')
        if cp.exists():
            with open(cp, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            if not caption:
                print(f"\n[WARNING] Caption file for {ip.name} is EMPTY. Image will be skipped.")
                return None
        else:
            fallback_caption = ip.stem.replace('_', ' ')
            print(f"\n[CAPTION WARNING] No .txt file for {ip.name}. Using fallback: '{fallback_caption}'")
            caption = fallback_caption
        return {"ip": ip, "caption": caption, "target_resolution": calculator.calculate_resolution(w, h), "original_size": (w, h)}
    except Exception as e:
        print(f"\n[CORRUPT IMAGE OR READ ERROR] Skipping {ip}, Reason: {e}")
        return None

def compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device):
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    for caption in captions:
        with torch.no_grad():
            i1 = t1(caption, padding="max_length", max_length=t1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            i2 = t2(caption, padding="max_length", max_length=t2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            e1 = te1(i1, output_hidden_states=True).hidden_states[-2]
            out2 = te2(i2, output_hidden_states=True)
            prompt_embeds = torch.cat((e1, out2.hidden_states[-2]), dim=-1)
            pooled_embeds = out2[0]
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_embeds)
    return torch.cat(prompt_embeds_list), torch.cat(pooled_prompt_embeds_list)

def check_if_caching_needed(config):
    needs_caching = False
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists():
            print(f"WARNING: Dataset path {root} does not exist, skipping.")
            continue
        image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
                      for p in root.rglob(f"*{ext}")]
        if not image_paths:
            print(f"INFO: No images found in {root}, skipping.")
            continue
        cache_dir = root / ".precomputed_embeddings_cache"
        if not cache_dir.exists():
            print(f"INFO: Cache directory doesn't exist for {root}, caching needed.")
            needs_caching = True
            continue
        cached_stems = {p.stem for p in cache_dir.rglob("*.pt")}
        uncached = [p for p in image_paths if p.stem not in cached_stems]
        if uncached:
            print(f"INFO: Found {len(uncached)} uncached images in {root}")
            needs_caching = True
        else:
            print(f"INFO: All {len(image_paths)} images already cached in {root}")
    return needs_caching

def load_vae_only(config, device):
    vae_path = config.VAE_PATH
    if not vae_path or not Path(vae_path).exists():
        return None
    print(f"INFO: Loading dedicated VAE from: {vae_path}")
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)
    print(f"INFO: VAE in float32 for high-quality caching")
    print(f"INFO: Moving VAE to {device}...")
    vae = vae.to(device)
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"INFO: VAE loaded. Current VRAM usage: {vram_gb:.2f} GB")
    return vae

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config):
        print("\n" + "="*60)
        print("INFO: All datasets are already cached. Skipping caching step.")
        print("="*60 + "\n")
        return
    print("\n" + "="*60)
    print("STARTING CACHING PROCESS")
    print("="*60 + "\n")
    calc = ResolutionCalculator(
    config.TARGET_PIXEL_AREA,
    stride=64,
    should_upscale=config.SHOULD_UPSCALE,
    max_area_tolerance=getattr(config, 'MAX_AREA_TOLERANCE', 1.1)
    )
    vae.enable_tiling()
    print("INFO: Loading text encoders to device...")
    te1.to(device)
    te2.to(device)
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"INFO: Current VRAM usage: {vram_gb:.2f} GB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        print(f"\n{'='*60}")
        print(f"Processing dataset: {root}")
        print(f"{'='*60}")
        paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
                for p in root.rglob(f"*{ext}")]
        if not paths:
            print(f"WARNING: No images found in {root}.")
            continue
        cache_dir = root / ".precomputed_embeddings_cache"
        cache_dir.mkdir(exist_ok=True)
        stems = {p.stem for p in cache_dir.rglob("*.pt")}
        to_process = [p for p in paths if p.stem not in stems]
        if not to_process:
            print(f"INFO: All {len(paths)} images already cached for this dataset.")
            continue
        print(f"INFO: Found {len(to_process)} images to cache (out of {len(paths)} total)")
        print("INFO: Validating images and calculating resolutions...")
        with Pool(processes=min(cpu_count(), 8)) as pool:
            results = list(tqdm(
                pool.imap(validate_and_assign_resolution, [(p, calc) for p in to_process]),
                total=len(to_process),
                desc="Validating"
            ))
        metadata = [r for r in results if r]
        if not metadata:
            print(f"WARNING: No valid images found to process in {root}.")
            continue
        print(f"INFO: {len(metadata)} images validated successfully")
        res_meta = {m['ip'].stem: m['target_resolution'] for m in metadata}
        metadata_path = root / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(res_meta, f, indent=4)
        print(f"INFO: Saved metadata to {metadata_path}")
        grouped = defaultdict(list)
        for m in metadata:
            grouped[m["target_resolution"]].append(m)
        print(f"INFO: Images grouped into {len(grouped)} resolution buckets")
        batches = [(res, grouped[res][i:i + config.CACHING_BATCH_SIZE])
                  for res in grouped
                  for i in range(0, len(grouped[res]), config.CACHING_BATCH_SIZE)]
        random.shuffle(batches)
        print(f"INFO: Processing {len(batches)} batches...")
        for batch_idx, ((w, h), batch_meta) in enumerate(tqdm(batches, desc="Caching")):
            captions = [m['caption'] for m in batch_meta]
            embeds, pooled = compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device)
            images, valid_meta = [], []
            for m in batch_meta:
                try:
                    with Image.open(m['ip']) as img:
                        img = img.convert("RGB")
                    processed_img = transform(resize_to_fit(img, w, h))
                    images.append(processed_img)
                    valid_meta.append(m)
                except Exception as e:
                    print(f"\n[ERROR] Skipping {m['ip']}: {e}")
            if not images: continue
            image_batch_tensor = torch.stack(images).to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(image_batch_tensor).latent_dist.sample() * vae.config.scaling_factor
            for j, m in enumerate(valid_meta):
                cache_path = cache_dir / f"{m['ip'].stem}.pt"
                torch.save({
                    "original_size": m["original_size"], "target_size": (w, h),
                    "embeds": embeds[j].cpu().to(torch.float32),
                    "pooled": pooled[j].cpu().to(torch.float32),
                    "latents": latents[j].cpu().to(torch.float32)
                }, cache_path)
            if (batch_idx + 1) % 10 == 0: torch.cuda.empty_cache()
    print("\nINFO: Moving text encoders back to CPU...")
    te1.cpu(); te2.cpu(); gc.collect(); torch.cuda.empty_cache()
    print("\n" + "="*60); print("CACHING COMPLETE"); print("="*60 + "\n")

class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.latent_files = []
        # --- MODIFICATION: Check for semantic loss, not semantic noise ---
        self.use_semantic_loss = (getattr(config, 'LOSS_TYPE', 'Default') == "Semantic")
        
        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / ".precomputed_embeddings_cache"
            if not cache_dir.exists(): continue
            files = list(cache_dir.glob("*.pt"))
            self.latent_files.extend(files * int(ds.get("repeats", 1)))
        if not self.latent_files: raise ValueError("No cached files found.")
        print("INFO: Shuffling the entire dataset order...")
        random.shuffle(self.latent_files)
        
        self.bucket_keys = []
        all_meta = {}
        for ds in config.INSTANCE_DATASETS:
            meta_path = Path(ds["path"]) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f: all_meta.update(json.load(f))
        for f in self.latent_files:
            key = all_meta.get(f.stem)
            self.bucket_keys.append(tuple(key) if key else None)
        print(f"INFO: Dataset initialized with {len(self.latent_files)} samples.")
        if self.use_semantic_loss:
             print("INFO: Semantic loss is ENABLED. Original images will be loaded during training.")

    def __len__(self): return len(self.latent_files)

    def _find_original_image_path(self, latent_path):
        """Finds the original image file corresponding to a latent file."""
        latent_p = Path(latent_path)
        stem = latent_p.stem
        parent_dir = latent_p.parent.parent
        
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            image_path = parent_dir / (stem + ext)
            if image_path.exists():
                return image_path
        return None

    def __getitem__(self, i):
        try:
            latent_path = self.latent_files[i]
            data = torch.load(latent_path, map_location="cpu")
            if (torch.isnan(data["latents"]).any() or torch.isinf(data["latents"]).any()):
                return None
            
            item_data = {
                "latents": data["latents"], "embeds": data["embeds"], "pooled": data["pooled"],
                "original_sizes": data["original_size"], "target_sizes": data["target_size"],
                "latent_path": str(latent_path)
            }
            
            # --- MODIFICATION: Load original image if semantic loss is enabled ---
            if self.use_semantic_loss:
                original_image_path = self._find_original_image_path(latent_path)
                if original_image_path:
                    item_data["original_image"] = Image.open(original_image_path).convert("RGB")
                else:
                    print(f"WARNING: Could not find original image for {latent_path.stem}. Semantic loss will be skipped for this item.")
                    item_data["original_image"] = None

            return item_data
        except Exception as e:
            print(f"WARNING: Skipping {self.latent_files[i]}: {e}")
            return None

class TimestepSampler:
    """
    Handles different timestep sampling strategies, including a robust, goal-seeking mode 
    that differentiates between single outliers and genuine instability trends.
    """
    def __init__(self, config, noise_scheduler, device):
        self.method = config.TIMESTEP_SAMPLING_METHOD
        self.config = config
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps

        if "Dynamic" in self.method:
            self.target_min_grad = self.config.TIMESTEP_SAMPLING_GRAD_MIN
            self.target_max_grad = self.config.TIMESTEP_SAMPLING_GRAD_MAX
            
            self.current_min_ts = float(self.config.TIMESTEP_SAMPLING_MIN)
            self.current_max_ts = float(self.config.TIMESTEP_SAMPLING_MAX)
            
            self.global_min_ts = 0.0
            self.global_max_ts = float(self.num_train_timesteps - 1)
            
            self.adjustment_strength = 0.05
            self.smoothing_factor = 0.9
            self.max_shift_per_step = 50.0
            
            self.smoothed_grad_norm = (self.target_min_grad + self.target_max_grad) / 2.0
            self.last_timestep_avg = self.num_train_timesteps / 2.0
            self.consecutive_spike_count = 0
            
            print("INFO: Initialized Trend-Aware Dynamic Sampler.")
            print(f"      Target Grad Norm Range: [{self.target_min_grad:.3f}, {self.target_max_grad:.3f}]")
            print(f"      Adjustment Strength: {self.adjustment_strength}, Max Shift: {int(self.max_shift_per_step)}")

    def sample(self, batch_size):
        """Public method to get a batch of timesteps based on the current strategy."""
        if "Uniform Continuous" in self.method:
            t_continuous = torch.rand(batch_size, device=self.device)
            return (t_continuous * (self.num_train_timesteps - 1)).long()
        
        elif "Dynamic" in self.method:
            min_ts = int(self.current_min_ts)
            max_ts = int(self.current_max_ts)
            if min_ts >= max_ts:
                min_ts = max(0, max_ts - 1)
            
            return torch.randint(min_ts, max_ts + 1, (batch_size,), device=self.device, dtype=torch.long)
        
        else: # Default: Random Integer
            min_ts = self.config.TIMESTEP_SAMPLING_MIN
            max_ts = self.config.TIMESTEP_SAMPLING_MAX
            return torch.randint(min_ts, max_ts + 1, (batch_size,), device=self.device, dtype=torch.long)
            
    def update(self, raw_grad_norm):
        """Update the dynamic range based on the raw gradient norm, handling outliers intelligently."""
        if "Dynamic" not in self.method:
            return

        is_spike = raw_grad_norm > self.target_max_grad
        if is_spike:
            self.consecutive_spike_count += 1
        else:
            self.consecutive_spike_count = 0
            
        self.smoothed_grad_norm = (self.smoothing_factor * self.smoothed_grad_norm) + \
                                  ((1 - self.smoothing_factor) * raw_grad_norm)
        
        base_shift = self.adjustment_strength * self.num_train_timesteps
        shift_direction = 0.0

        if self.consecutive_spike_count > 1:
            shift_direction = -1.5
        elif self.consecutive_spike_count == 1:
            shift_direction = -0.5
        else:
            if self.smoothed_grad_norm < self.target_min_grad:
                shift_direction = 1.0
            elif self.smoothed_grad_norm > self.target_max_grad:
                shift_direction = -1.0
        
        if shift_direction != 0.0:
            final_shift = min(base_shift * abs(shift_direction), self.max_shift_per_step)
            final_shift *= (shift_direction / abs(shift_direction))

            self.current_min_ts += final_shift
            self.current_max_ts += final_shift

        self.current_min_ts = max(self.global_min_ts, self.current_min_ts)
        self.current_max_ts = min(self.global_max_ts, self.current_max_ts)
        
        min_range_size = 50.0
        current_range_size = self.current_max_ts - self.current_min_ts
        if current_range_size < min_range_size:
            center = (self.current_min_ts + self.current_max_ts) / 2.0
            self.current_min_ts = center - (min_range_size / 2.0)
            self.current_max_ts = center + (min_range_size / 2.0)
        
        self.current_min_ts = max(self.global_min_ts, self.current_min_ts)
        self.current_max_ts = min(self.global_max_ts, self.current_max_ts)
        if self.current_min_ts >= self.current_max_ts:
            self.current_min_ts = self.current_max_ts - 1.0

    def record_timesteps(self, timesteps):
        """Records the timesteps used in a batch for the next dynamic update."""
        if "Dynamic" in self.method:
            self.last_timestep_avg = timesteps.float().mean().item()

def custom_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    return {k: (torch.stack([item[k] for item in batch]) if isinstance(batch[0][k], torch.Tensor)
                else [item[k] for item in batch]) for k in batch[0]}

def _generate_hf_to_sd_unet_key_mapping(hf_keys):
    final_map = {}
    for hf_key in hf_keys:
        key = hf_key
        if "resnets" in key:
            new_key = re.sub(r"^down_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"input_blocks.{3*int(m.group(1)) + int(m.group(2)) + 1}.0.", key)
            new_key = re.sub(r"^mid_block\.resnets\.(\d+)\.", lambda m: f"middle_block.{2*int(m.group(1))}.", new_key)
            new_key = re.sub(r"^up_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"output_blocks.{3*int(m.group(1)) + int(m.group(2))}.0.", new_key)
            new_key = new_key.replace("norm1.", "in_layers.0.").replace("conv1.", "in_layers.2.").replace("norm2.", "out_layers.0.").replace("conv2.", "out_layers.3.").replace("time_emb_proj.", "emb_layers.1.").replace("conv_shortcut.", "skip_connection.")
            final_map[hf_key] = new_key; continue
        if "attentions" in key:
            new_key = re.sub(r"^down_blocks\.(\d+)\.attentions\.(\d+)\.", lambda m: f"input_blocks.{3*int(m.group(1)) + int(m.group(2)) + 1}.1.", key)
            new_key = re.sub(r"^mid_block\.attentions\.0\.", "middle_block.1.", new_key)
            new_key = re.sub(r"^up_blocks\.(\d+)\.attentions\.(\d+)\.", lambda m: f"output_blocks.{3*int(m.group(1)) + int(m.group(2))}.1.", new_key)
            final_map[hf_key] = new_key; continue
        if "downsamplers" in key: final_map[hf_key] = re.sub(r"^down_blocks\.(\d+)\.downsamplers\.0\.conv\.", lambda m: f"input_blocks.{3*(int(m.group(1))+1)}.0.op.", key); continue
        if "upsamplers" in key: final_map[hf_key] = re.sub(r"^up_blocks\.(\d+)\.upsamplers\.0\.", lambda m: f"output_blocks.{3*int(m.group(1)) + 2}.2.", key); continue
        if key.startswith("conv_in."): final_map[hf_key] = key.replace("conv_in.", "input_blocks.0.0."); continue
        if key.startswith("conv_norm_out."): final_map[hf_key] = key.replace("conv_norm_out.", "out.0."); continue
        if key.startswith("conv_out."): final_map[hf_key] = key.replace("conv_out.", "out.2."); continue
        if key.startswith("time_embedding.linear_1."): final_map[hf_key] = key.replace("time_embedding.linear_1.", "time_embed.0."); continue
        if key.startswith("time_embedding.linear_2."): final_map[hf_key] = key.replace("time_embedding.linear_2.", "time_embed.2."); continue
        if key.startswith("add_embedding.linear_1."): final_map[hf_key] = key.replace("add_embedding.linear_1.", "label_emb.0.0."); continue
        if key.startswith("add_embedding.linear_2."): final_map[hf_key] = key.replace("add_embedding.linear_2.", "label_emb.0.2."); continue
    return final_map

def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar / F.pad(alphas_bar[:-1], (1, 0), value=1.0)
    return 1 - alphas

def filter_scheduler_config(config, scheduler_class):
    return {k: v for k, v in config.items() if k in inspect.signature(scheduler_class.__init__).parameters}
# endregion

def create_optimizer(config, params_to_optimize):
    """Creates the optimizer, supporting RavenAdamW with gradient centralization."""
    optimizer_type = config.OPTIMIZER_TYPE.lower()
    if optimizer_type == "raven":
        print("\n--- Initializing Raven Optimizer ---")
        
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        if curve_points:
            initial_lr = max(point[1] for point in curve_points)
            print(f"INFO: Initializing optimizer with max LR from custom curve: {initial_lr:.2e}")
        else:
            initial_lr = config.LEARNING_RATE 
            print(f"WARNING: LR_CUSTOM_CURVE is empty. Falling back to default LEARNING_RATE: {initial_lr:.2e}")

        raven_params = getattr(config, 'RAVEN_PARAMS', {})
        return RavenAdamW(
            params_to_optimize,
            lr=initial_lr,
            betas=tuple(raven_params.get('betas', [0.9, 0.999])),
            eps=raven_params.get('eps', 1e-8),
            weight_decay=raven_params.get('weight_decay', 0.01),
            debias_strength=raven_params.get('debias_strength', 1.0),
            use_grad_centralization=raven_params.get('use_grad_centralization', False),
            gc_alpha=raven_params.get('gc_alpha', 1.0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'. This script is configured to only use 'raven'.")

def save_model(output_path, unet, base_model_state_dict, key_map, trainable_layer_names):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to: {output_path}")
    unet.to('cpu')
    trained_unet_state_dict = unet.state_dict()
    final_state_dict = base_model_state_dict.copy()

    for hf_key in trainable_layer_names:
        mapped_key = key_map.get(hf_key)
        if mapped_key is None: continue
        sd_key = 'model.diffusion_model.' + mapped_key
        if sd_key in final_state_dict:
            final_state_dict[sd_key] = trained_unet_state_dict[hf_key]
        else:
            print(f"WARNING: Key '{sd_key}' not found in base model state dict. Skipping.")

    save_file(final_state_dict, output_path)
    print(f"Successfully saved model: {output_path.name}")
    unet.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def save_checkpoint(global_step, unet, base_model_state_dict, key_map, trainable_layer_names,
                    optimizer, lr_scheduler, scaler, sampler, config):
    output_dir = Path(config.OUTPUT_DIR)
    model_filename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    
    save_model(output_dir / model_filename, unet, base_model_state_dict, key_map, trainable_layer_names)
    
    training_state = {
        'global_step': global_step,
        'optimizer_state': optimizer.save_cpu_state(),
        'sampler_seed': sampler.seed,
    }

    if scaler is not None:
        training_state['scaler_state_dict'] = scaler.state_dict()
    else:
        training_state['scaler_state_dict'] = None

    torch.save(training_state, output_dir / state_filename)
    print(f"Successfully saved training state: {state_filename}")


def main():
    config = TrainingConfig()
    if config.SEED: set_seed(config.SEED)
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Optimized matmul settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    global_step = 0
    model_to_load = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed = config.SEED
    optimizer_state = None

    if config.RESUME_TRAINING:
        print("\n" + "="*50); print("--- RESUMING TRAINING SESSION ---")
        state_path = Path(config.RESUME_STATE_PATH)
        
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        global_step = training_state['global_step']
        initial_sampler_seed = training_state['sampler_seed']
        optimizer_state = training_state['optimizer_state']
        model_to_load = Path(config.RESUME_MODEL_PATH)
        
        print(f"INFO: Resuming from global step: {global_step}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50); print("--- STARTING FRESH TRAINING SESSION ---"); print("="*50 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if check_if_caching_needed(config):
        print("Loading base model components for caching...")
        base_pipe = StableDiffusionXLPipeline.from_single_file(config.SINGLE_FILE_CHECKPOINT_PATH, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        vae_for_caching = load_vae_only(config, device) or base_pipe.vae.to(device)
        precompute_and_cache_latents(config, base_pipe.tokenizer, base_pipe.tokenizer_2, base_pipe.text_encoder, base_pipe.text_encoder_2, vae_for_caching, device)
        del base_pipe, vae_for_caching; gc.collect(); torch.cuda.empty_cache()

    print(f"\n--- Loading Model ---")
    print(f"INFO: Loading UNet for training from: {model_to_load.name}")
    base_model_state_dict = load_file(model_to_load, device="cpu")
    pipe = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=config.compute_dtype, low_cpu_mem_usage=True)
    unet = pipe.unet
    original_scheduler_config = pipe.scheduler.config
    del pipe; gc.collect(); torch.cuda.empty_cache()
    
    print("\n--- Configuring Noise Scheduler ---")
    SCHEDULER_MAP = {"DDPMScheduler": DDPMScheduler, "DDIMScheduler": DDIMScheduler, "EulerDiscreteScheduler": EulerDiscreteScheduler}
    scheduler_name = getattr(config, 'NOISE_SCHEDULER', 'DDPMScheduler').replace(" (Experimental)", "")
    scheduler_class = SCHEDULER_MAP.get(scheduler_name, DDPMScheduler)
    
    training_scheduler_config = {
        **original_scheduler_config, 
        'prediction_type': config.PREDICTION_TYPE,
        'beta_schedule': config.BETA_SCHEDULE
    }
    noise_scheduler = scheduler_class.from_config(filter_scheduler_config(training_scheduler_config, scheduler_class))
    print(f"INFO: Training with {type(noise_scheduler).__name__}")

    if getattr(config, 'USE_ZERO_TERMINAL_SNR', False):
        print("INFO: Applying Zero Terminal SNR rescaling to scheduler betas.")
        new_betas = rescale_zero_terminal_snr(noise_scheduler.betas)
        noise_scheduler.betas = new_betas
        noise_scheduler.alphas_cumprod = torch.cumprod(1.0 - new_betas, dim=0)

    print("\n--- Initializing Dataset ---")
    dataset = ImageTextLatentDataset(config)
    print(f"INFO: Initializing BucketBatchSampler with seed: {initial_sampler_seed}")
    sampler = BucketBatchSampler(dataset, config.BATCH_SIZE, initial_sampler_seed, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    unet.to(device).enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention() if config.MEMORY_EFFICIENT_ATTENTION == "xformers" else unet.set_attn_processor(AttnProcessor2_0())
    

    print("\n--- UNet Layer Selection Report ---")
    exclusion_keywords = config.UNET_EXCLUDE_TARGETS
    
    trainable_layer_names = []
    frozen_layer_names = []
    
    for name, param in unet.named_parameters():
        should_exclude = any(k in name for k in exclusion_keywords)
        if should_exclude:
            param.requires_grad = False
            frozen_layer_names.append(name)
        else:
            param.requires_grad = True
            trainable_layer_names.append(name)

    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in params_to_optimize)
    frozen_params = total_params - trainable_params
    percentage_trainable = (trainable_params / total_params) * 100 if total_params > 0 else 0

    print(f"  - Exclusion Keywords: {exclusion_keywords}")
    print(f"  - Trainable: {trainable_params / 1e6:.2f}M params ({percentage_trainable:.2f}%)")
    print(f"  - Frozen:    {frozen_params / 1e6:.2f}M params")
    

    
    print(f"GUI_PARAM_INFO::{trainable_params / 1e6:.2f}M ({percentage_trainable:.2f}%) of {total_params / 1e6:.2f}M total")

    optimizer = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(optimizer=optimizer, curve_points=config.LR_CUSTOM_CURVE, max_train_steps=config.MAX_TRAIN_STEPS)

    if config.RESUME_TRAINING:
        print("\n--- Restoring Training States ---")
        if optimizer_state:
            optimizer.load_cpu_state(optimizer_state)
        lr_scheduler.step(global_step)
        print("--- Resume setup complete. Starting training loop. ---")
    else:
        print("\n--- Fresh start setup complete. Starting training loop. ---")

    unet.train()
    key_map = _generate_hf_to_sd_unet_key_mapping(list(unet.state_dict().keys()))
    
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    # Passed "Gradient Check" as the name so the reporter knows we aren't tracking a specific layer anymore
    reporter = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="Gradient Check") 
    timestep_sampler = TimestepSampler(config, noise_scheduler, device)

    print(f"\n--- Using Loss Calculation Method: {getattr(config, 'LOSS_TYPE', 'Default')} ---\n")

    accumulated_latent_paths = []
    training_start_time = time.time()
    global_step_times = deque(maxlen=50)
    optim_step_times = deque(maxlen=20)
    last_step_time = time.time()
    last_optim_step_log_time = time.time()
    done = False
    
    while not done:
        for batch in dataloader:
            if global_step >= config.MAX_TRAIN_STEPS:
                done = True
                break
            if not batch: continue

            if "latent_path" in batch: accumulated_latent_paths.extend(batch["latent_path"])
            
            latents = batch["latents"].to(device, non_blocking=True)
            embeds = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            original_images = batch.get("original_image")

            target = None
            
            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):
                time_ids_list = []
                for s1, s2 in zip(batch["original_sizes"], batch["target_sizes"]):
                    time_id = torch.tensor(list(s1) + [0,0] + list(s2), dtype=torch.float32)
                    time_ids_list.append(time_id.unsqueeze(0).to(device, dtype=config.compute_dtype))
                time_ids = torch.cat(time_ids_list, dim=0)
                
                noise = generate_train_noise(latents, config)
                timesteps = timestep_sampler.sample(latents.shape[0])
                timestep_sampler.record_timesteps(timesteps)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                if config.PREDICTION_TYPE == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                
                pred = unet(noisy_latents, timesteps, embeds, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample

            # Loss Calculation
            per_pixel_loss = F.mse_loss(pred.float(), target.to(pred.dtype).float(), reduction="none")

            if getattr(config, 'LOSS_TYPE', 'Default') == "Semantic":
                if original_images and None not in original_images:
                    b, c, h, w = latents.shape
                    semantic_map = _generate_semantic_map_for_batch(
                        original_images, config.SEMANTIC_LOSS_BLEND, config.SEMANTIC_LOSS_STRENGTH,
                        target_size=(w, h), device=latents.device, dtype=torch.float32, num_channels=c
                    )
                    modulation = 1.0 + (semantic_map * config.SEMANTIC_LOSS_STRENGTH)
                    loss = (per_pixel_loss * modulation).mean()
                else:
                    loss = per_pixel_loss.mean()
            else:
                loss = per_pixel_loss.mean()

            accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
            raw_loss_value = loss.detach().item()
            loss = loss / accumulation_steps
            loss = loss.to(dtype=config.compute_dtype)
            loss.backward()

            diagnostics.step(raw_loss_value)

            diag_data_to_log = None
            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                
                # --- SIMPLIFIED: No parameter clone/check logic here ---
                
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(params_to_optimize, float('inf')).item()
                timestep_sampler.update(raw_grad_norm)
                
                if config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, config.CLIP_GRAD_NORM)
                clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm

                optimizer.step()
                lr_scheduler.step(global_step + 1)
                optimizer.zero_grad(set_to_none=True)

                optim_step_time = time.time() - last_optim_step_log_time
                optim_step_times.append(optim_step_time)
                last_optim_step_log_time = time.time()
                

                update_status = 1.0 if raw_grad_norm > 0 else 0.0

                diag_data_to_log = {
                    'optim_step': (global_step + 1) // config.GRADIENT_ACCUMULATION_STEPS,
                    'avg_loss': diagnostics.get_average_loss(),
                    'current_lr': optimizer.param_groups[0]['lr'],
                    'raw_grad_norm': raw_grad_norm,
                    'clipped_grad_norm': clipped_grad_norm,
                    'update_delta': update_status, # Pseudo-value for reporter
                    'optim_step_time': optim_step_time,
                    'avg_optim_step_time': sum(optim_step_times) / len(optim_step_times)
                }
                reporter.check_and_report_anomaly(global_step + 1, raw_grad_norm, clipped_grad_norm, config, accumulated_latent_paths)
                diagnostics.reset()
                accumulated_latent_paths.clear()

            step_duration = time.time() - last_step_time
            global_step_times.append(step_duration)
            last_step_time = time.time()
            elapsed_time = time.time() - training_start_time
            eta_seconds = (config.MAX_TRAIN_STEPS - (global_step + 1)) * (sum(global_step_times) / len(global_step_times))
            
            timing_data = {
                'raw_step_time': step_duration, 'elapsed_time': elapsed_time, 'eta': eta_seconds,
                'loss': loss.item(), 'timestep': timesteps[0].item()
            }
            
            reporter.log_step(global_step, timing_data=timing_data, diag_data=diag_data_to_log)
            global_step += 1
            
            if config.SAVE_EVERY_N_STEPS > 0 and global_step % config.SAVE_EVERY_N_STEPS == 0:
                 print(f"\n--- Saving checkpoint at step {global_step} ---")
                 save_checkpoint(
                    global_step, unet, base_model_state_dict, key_map, trainable_layer_names,
                    optimizer, lr_scheduler, None, sampler, config
                )

    print("\nTraining complete.")
    reporter.shutdown()
    output_path = OUTPUT_DIR / f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_final.safetensors"
    save_model(output_path, unet, base_model_state_dict, key_map, trainable_layer_names)
    print("All tasks complete. Final model saved.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()