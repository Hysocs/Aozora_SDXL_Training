import inspect
import fnmatch
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
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel
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
from optimizer.raven import RavenAdamW
from optimizer.titan import TitanAdamW
import uuid

warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"INFO: Set random seed to {seed}")

def fix_alpha_channel(img):
    if img.mode == 'P' and 'transparency' in img.info:
        img = img.convert('RGBA')
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

def generate_character_map(pil_image):
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

def _generate_semantic_map_for_batch(images, blend_factor, strength, target_size, device, dtype, num_channels):
    batch_maps = []
    for img in images:
        if img is None:
            weight_map_tensor = torch.zeros(target_size, dtype=dtype)
            batch_maps.append(weight_map_tensor)
            continue
        char_map_pil = generate_character_map(img)
        detail_map_pil = generate_detail_map(img)
        char_map_np = np.array(char_map_pil).astype(np.float32)
        detail_map_np = np.array(detail_map_pil).astype(np.float32)
        combined_map = ((1.0 - blend_factor) * char_map_np) + (blend_factor * detail_map_np)
        if combined_map.max() > 0:
            combined_map = combined_map / combined_map.max()
        weight_map = np.power(combined_map, 0.8)
        combined_map_pil = Image.fromarray(weight_map, mode='F').resize(target_size, Image.Resampling.LANCZOS)
        weight_map_tensor = torch.from_numpy(np.array(combined_map_pil)).float()
        batch_maps.append(weight_map_tensor)
    final_map_batch = torch.stack(batch_maps).unsqueeze(1).to(device, dtype=dtype)
    return final_map_batch.expand(-1, num_channels, -1, -1)

def generate_train_noise(latents, config):
    return torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)

class TrainingConfig:
    def __init__(self):
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        self._load_from_user_config()
        self._type_check_and_correct()
        self.compute_dtype = torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16
        self.is_rectified_flow = getattr(self, "TRAINING_MODE", "Standard (SDXL)") == "Rectified Flow (Experimental)"

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
                print(f"WARNING: Could not convert '{key}' value '{value}' to {expected_type.__name__}. Using default value: {default_value}")
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
        if seconds is None or not math.isfinite(seconds): return "N/A"
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
                if task_type == 'log_step': self._handle_log_step(**data)
                elif task_type == 'anomaly': self._handle_anomaly(**data)
                self.task_queue.task_done()
            except queue.Empty: continue

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
            data = {'global_step': global_step, 'raw_grad_norm': raw_grad_norm, 'clipped_grad_norm': clipped_grad_norm, 'low_thresh': low, 'high_thresh': high, 'paths': list(paths)}
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
        g.manual_seed(self.seed + self.epoch)
        indices = list(range(self.total_images))
        if self.batch_size == 1:
            if self.shuffle: indices = torch.randperm(len(indices), generator=g).tolist()
            batches = [[i] for i in indices]
        else:
            if self.shuffle: indices = torch.randperm(len(indices), generator=g).tolist()
            buckets = defaultdict(list)
            for idx in indices:
                key = self.dataset.bucket_keys[idx]
                buckets[key].append(idx)
            batches = []
            for key in buckets:
                bucket_indices = buckets[key]
                for i in range(0, len(bucket_indices), self.batch_size):
                    batch = bucket_indices[i : i + self.batch_size]
                    batches.append(batch)
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
        current_area = width * height
        
        if not self.should_upscale:
            if current_area < self.target_area:
                 w = int(width // self.stride) * self.stride
                 h = int(height // self.stride) * self.stride
                 return (max(w, self.stride), max(h, self.stride))
            
            h = int(math.sqrt(self.target_area / aspect_ratio) // self.stride) * self.stride
            w = int(h * aspect_ratio // self.stride) * self.stride
            return (w, h)
        
        if current_area > self.max_area: scale_factor = math.sqrt(self.target_area / current_area)
        elif current_area < self.target_area: scale_factor = math.sqrt(self.target_area / current_area)
        else: scale_factor = 1.0
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
    if w / target_w < h / target_h: w_new, h_new = target_w, int(h * target_w / w)
    else: w_new, h_new = int(w * target_h / h), target_h
    return image.resize((w_new, h_new), Image.Resampling.BICUBIC).crop(((w_new - target_w) // 2, (h_new - target_h) // 2, (w_new + target_w) // 2, (h_new + target_h) // 2))

def validate_and_assign_resolution(args):
    ip, calculator = args
    try:
        with Image.open(ip) as img: img.verify()
        with Image.open(ip) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0: return None
        cp = ip.with_suffix('.txt')
        if cp.exists():
            with open(cp, 'r', encoding='utf-8') as f: caption = f.read().strip()
            if not caption: return None
        else: caption = ip.stem.replace('_', ' ')
        return {"ip": ip, "caption": caption, "target_resolution": calculator.calculate_resolution(w, h), "original_size": (w, h)}
    except Exception as e:
        print(f"\n[CORRUPT IMAGE OR READ ERROR] Skipping {ip}, Reason: {e}")
        return None

def compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device):
    prompt_embeds_list = []
    pooled_prompt_embeds_list = [] 
    max_len = t1.model_max_length 
    for caption in captions:
        with torch.no_grad():
            tokens_1 = t1(caption, return_tensors="pt", truncation=False).input_ids.to(device)
            tokens_2 = t2(caption, return_tensors="pt", truncation=False).input_ids.to(device)
            ids1 = tokens_1[0]
            ids2 = tokens_2[0]
            if len(ids1) > 2:
                ids1 = ids1[1:-1]
                ids2 = ids2[1:-1]
            chunk_size = max_len - 2 
            total_len = len(ids1)
            num_chunks = (total_len + chunk_size - 1) // chunk_size
            if num_chunks == 0: num_chunks = 1
            hidden_states_list = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, total_len)
                chunk_ids1 = ids1[start:end]
                chunk_ids2 = ids2[start:end]
                bos = torch.tensor([t1.bos_token_id], device=device)
                eos = torch.tensor([t1.eos_token_id], device=device)
                chunk_ids1 = torch.cat([bos, chunk_ids1, eos], dim=0).unsqueeze(0)
                chunk_ids2 = torch.cat([bos, chunk_ids2, eos], dim=0).unsqueeze(0)
                pad_len = max_len - chunk_ids1.shape[1]
                if pad_len > 0:
                    pad_token = t1.pad_token_id
                    padding = torch.tensor([pad_token] * pad_len, device=device).unsqueeze(0)
                    chunk_ids1 = torch.cat([chunk_ids1, padding], dim=1)
                    chunk_ids2 = torch.cat([chunk_ids2, padding], dim=1)
                out1 = te1(chunk_ids1, output_hidden_states=True)
                out2 = te2(chunk_ids2, output_hidden_states=True)
                h1 = out1.hidden_states[-2]
                h2 = out2.hidden_states[-2]
                chunk_embed = torch.cat((h1, h2), dim=-1)
                hidden_states_list.append(chunk_embed)
                if i == 0: pooled_prompt_embeds_list.append(out2[0])
            final_prompt_embed = torch.cat(hidden_states_list, dim=1)
            prompt_embeds_list.append(final_prompt_embed)
    max_seq_len = max([pe.shape[1] for pe in prompt_embeds_list])
    padded_embeds = []
    for pe in prompt_embeds_list:
        curr_len = pe.shape[1]
        if curr_len < max_seq_len:
            padding = torch.zeros((1, max_seq_len - curr_len, pe.shape[2]), device=device, dtype=pe.dtype)
            pe = torch.cat([pe, padding], dim=1)
        padded_embeds.append(pe)
    return torch.cat(padded_embeds), torch.cat(pooled_prompt_embeds_list)

def check_if_caching_needed(config):
    needs_caching = False
    
    cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists():
            print(f"WARNING: Dataset path {root} does not exist, skipping.")
            continue
        image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        if not image_paths:
            print(f"INFO: No images found in {root}, skipping.")
            continue
        cache_dir = root / cache_folder_name
        if not cache_dir.exists():
            print(f"INFO: Cache directory doesn't exist for {root}, caching needed.")
            needs_caching = True
            continue
        
        cached_te_files = list(cache_dir.glob("*_te.pt"))
        if len(cached_te_files) < len(image_paths):
            print(f"INFO: Found {len(image_paths)} images but only {len(cached_te_files)} cached files in {root}")
            needs_caching = True
        else:
            print(f"INFO: All {len(image_paths)} images appear to be cached in {root}")
            
    return needs_caching

def load_vae_only(config, device):
    vae_path = config.VAE_PATH
    if not vae_path or not Path(vae_path).exists(): 
        return None
    
    print(f"INFO: Loading dedicated VAE from: {vae_path}")
    
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)
    
    if config.is_rectified_flow:
        vae.config.shift_factor = 0.1726
        vae.config.scaling_factor = 0.1280
        print(f"INFO: [RF MODE] VAE configured with EQB7 parameters:")
        print(f"      - Shift Factor: {vae.config.shift_factor}")
        print(f"      - Scale Factor: {vae.config.scaling_factor}")
    else:
        vae.config.shift_factor = None
        vae.config.scaling_factor = 0.13025
        print(f"INFO: [STANDARD MODE] VAE configured with SDXL parameters:")
        print(f"      - Scale Factor: {vae.config.scaling_factor}")
    
    vae.enable_tiling()
    vae.enable_slicing()
    
    vae = vae.to(device)
    return vae

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config):
        print("\n" + "="*60)
        print("INFO: Datasets already cached.")
        print("="*60 + "\n")
        return
    
    cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    
    print("\n" + "="*60)
    print(f"STARTING HIGH-FIDELITY (FP32) CACHING FOR: {config.TRAINING_MODE}")
    print(f"Using cache folder: {cache_folder_name}")
    print("="*60 + "\n")
    
    calc = ResolutionCalculator(
        config.TARGET_PIXEL_AREA, 
        stride=64, 
        should_upscale=config.SHOULD_UPSCALE
    )
    
    vae.to(device, dtype=torch.float32)
    
    if config.is_rectified_flow:
        assert abs(vae.config.shift_factor - 0.1726) < 1e-6, \
            f"VAE shift_factor mismatch! Expected 0.1726, got {vae.config.shift_factor}"
        assert abs(vae.config.scaling_factor - 0.1280) < 1e-6, \
            f"VAE scaling_factor mismatch! Expected 0.1280, got {vae.config.scaling_factor}"
    else:
        expected_scale = 0.13025
        assert abs(vae.config.scaling_factor - expected_scale) < 1e-6, \
            f"VAE scaling_factor mismatch! Expected {expected_scale}, got {vae.config.scaling_factor}"
    
    vae.enable_tiling()
    vae.enable_slicing()
    
    te1.to(device)
    te2.to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        cache_dir = root / cache_folder_name
        cache_dir.mkdir(exist_ok=True)
        
        paths = [
            p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] 
            for p in root.rglob(f"*{ext}")
        ]
        
        to_process = []
        for p in paths:
            relative_path = p.relative_to(root)
            safe_stem = str(relative_path.with_suffix('')).replace(
                os.sep, '_'
            ).replace('/', '_').replace('\\', '_')
            
            if not (cache_dir / f"{safe_stem}_te.pt").exists():
                to_process.append(p)

        if not to_process:
            print(f"INFO: All images in {root.name} are already cached.")
            continue
            
        print(f"INFO: Validating and Bucketing {len(to_process)} images from {root.name}...")
        
        with Pool(processes=min(cpu_count(), 8)) as pool:
            results = list(tqdm(
                pool.imap(
                    validate_and_assign_resolution, 
                    [(p, calc) for p in to_process]
                ), 
                total=len(to_process)
            ))
        
        metadata = [r for r in results if r]
        
        grouped = defaultdict(list)
        for m in metadata:
            grouped[m["target_resolution"]].append(m)
        
        batches = []
        for res in grouped:
            for i in range(0, len(grouped[res]), config.CACHING_BATCH_SIZE):
                batches.append((res, grouped[res][i:i + config.CACHING_BATCH_SIZE]))
        
        random.shuffle(batches)
        
        for batch_idx, ((w, h), batch_meta) in enumerate(
            tqdm(batches, desc=f"Encoding {root.name} (FP32)")
        ):
            captions = [m['caption'] for m in batch_meta]
            
            embeds, pooled = compute_chunked_text_embeddings(
                captions, t1, t2, te1, te2, device
            )
            embeds = embeds.to(dtype=torch.float32)
            pooled = pooled.to(dtype=torch.float32)
            
            images_global = []
            images_full = []
            valid_meta_final = []
            
            for m in batch_meta:
                try:
                    with Image.open(m['ip']) as img:
                        img = fix_alpha_channel(img)
                        
                        img_resized = resize_to_fit(img, w, h)
                        images_global.append(transform(img_resized))
                        
                        raw_w, raw_h = img.size
                        should_cache_full = (raw_w > w + 64) or (raw_h > h + 64)
                        
                        if should_cache_full:
                            images_full.append(transform(img))
                        else:
                            images_full.append(None)
                            
                        valid_meta_final.append(m)
                except Exception as e:
                    print(f"Skipping {m['ip']}: {e}")
            
            if not images_global:
                continue
            
            pixel_values_global = torch.stack(images_global).to(device, dtype=torch.float32)
            
            with torch.no_grad():
                dist = vae.encode(pixel_values_global).latent_dist
                latents_global = dist.sample() * vae.config.scaling_factor
                
                latents_full_list = []
                for img_t in images_full:
                    if img_t is not None:
                        img_t = img_t.unsqueeze(0).to(device, dtype=torch.float32)
                        dist_full = vae.encode(img_t).latent_dist
                        lat_full = dist_full.sample() * vae.config.scaling_factor
                        latents_full_list.append(lat_full.squeeze(0).cpu())
                    else:
                        latents_full_list.append(None)
                
            latents_global = latents_global.cpu()
            
            for j, m in enumerate(valid_meta_final):
                relative_path = m['ip'].relative_to(root)
                safe_filename = str(relative_path.with_suffix('')).replace(
                    os.sep, '_'
                ).replace('/', '_').replace('\\', '_')
                
                path_te = cache_dir / f"{safe_filename}_te.pt"
                path_lat = cache_dir / f"{safe_filename}_lat.pt"
                path_full = cache_dir / f"{safe_filename}_full.pt"
                
                has_full = latents_full_list[j] is not None
                
                torch.save({
                    "original_stem": m['ip'].stem,
                    "relative_path": str(relative_path),
                    "original_size": m["original_size"],
                    "target_size": (w, h),
                    "embeds": embeds[j].cpu(),
                    "pooled": pooled[j].cpu(),
                    "vae_shift": vae.config.shift_factor,
                    "vae_scale": vae.config.scaling_factor,
                    "has_full_res": has_full
                }, path_te)
                
                torch.save(latents_global[j], path_lat)
                
                if has_full:
                    torch.save(latents_full_list[j], path_full)
            
            del pixel_values_global, latents_global, embeds, pooled
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

    print("\nINFO: Caching Complete (Split Files). Cleaning up...")
    te1.cpu()
    te2.cpu()
    gc.collect()
    torch.cuda.empty_cache()

class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.te_files = []
        self.use_semantic_loss = (getattr(config, 'LOSS_TYPE', 'Default') == "Semantic")
        self.enable_chunking = getattr(config, "VRAM_CHUNK_ENABLED", True)
        self.max_latent_dim = int(getattr(config, "VRAM_CHUNK_SIZE", 96))
        self.chunk_chance = float(getattr(config, "VRAM_CHUNK_CHANCE", 1.0))
        self.dataset_roots = {} 
        self.is_rectified_flow = config.is_rectified_flow
        self.worker_rng = None
        self.seed = config.SEED if config.SEED else 42
        
        cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"

        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / cache_folder_name
            if not cache_dir.exists(): continue
            
            files = list(cache_dir.glob("*_te.pt"))
            for f in files:
                self.dataset_roots[str(f)] = root
                
            self.te_files.extend(files * int(ds.get("repeats", 1)))

        if not self.te_files: raise ValueError("No cached files found.")
        
        print("INFO: Shuffling the entire dataset order...")
        random.shuffle(self.te_files)
        
        self.bucket_keys = []
        
        print("INFO: building buckets from cache headers...")
        for f in tqdm(self.te_files, desc="Loading Cache Headers"):
            try:
                header = torch.load(f, map_location='cpu')
                res = header.get('target_size')
                if res:
                    self.bucket_keys.append(tuple(res))
                else:
                    self.bucket_keys.append(None)

            except Exception as e:
                print(f"Error reading cache header for {f}: {e}")
                self.bucket_keys.append(None)
                
        print(f"INFO: Dataset initialized with {len(self.te_files)} samples.")
        if self.enable_chunking:
            print(f"INFO: Spatial Chunking Enabled. Max Latent Dim: {self.max_latent_dim}, Chance: {self.chunk_chance}")
        if self.use_semantic_loss: print("INFO: Semantic loss is ENABLED.")

    def __len__(self): return len(self.te_files)

    def _init_worker_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_rng = random.Random(self.seed)
        else:
            self.worker_rng = random.Random(self.seed + worker_info.id)

    def __getitem__(self, i):
        try:
            if self.worker_rng is None:
                self._init_worker_rng()
                
            path_te = self.te_files[i]
            data_te = torch.load(path_te, map_location="cpu")
            
            cached_shift = data_te.get("vae_shift", 0.0)
            cached_scale = data_te.get("vae_scale", 0.0)
            
            if self.is_rectified_flow:
                if abs(cached_shift - 0.1726) > 1e-4 or abs(cached_scale - 0.1280) > 1e-4:
                     print(f"WARNING: Skipping {path_te}: VAE param mismatch for Rectified Flow.")
                     return None
            else:
                if abs(cached_scale - 0.13025) > 1e-4:
                     print(f"WARNING: Skipping {path_te}: VAE param mismatch for Standard SDXL.")
                     return None
            
            should_chunk = (
                self.enable_chunking and 
                self.worker_rng.random() < self.chunk_chance
            )
            
            path_str = str(path_te)
            path_lat = Path(path_str.replace("_te.pt", "_lat.pt"))
            
            latents_to_use = None
            latents_full = None
            
            if should_chunk and data_te.get("has_full_res", False):
                path_full = Path(path_str.replace("_te.pt", "_full.pt"))
                if path_full.exists():
                    latents_full = torch.load(path_full, map_location="cpu")
            
            if latents_full is None:
                latents_to_use = torch.load(path_lat, map_location="cpu")
            else:
                latents_to_use = latents_full
                
            if (torch.isnan(latents_to_use).any() or torch.isinf(latents_to_use).any()): return None

            final_latents = None
            crop_coords = (0, 0)
            target_size_px = data_te["target_size"] 
            
            if should_chunk and latents_full is not None:
                h_full = latents_to_use.shape[1]
                w_full = latents_to_use.shape[2]
                
                new_h = min(h_full, self.max_latent_dim)
                new_w = min(w_full, self.max_latent_dim)
                
                top_max = h_full - new_h
                left_max = w_full - new_w
                
                crop_top = self.worker_rng.randint(0, top_max) if top_max > 0 else 0
                crop_left = self.worker_rng.randint(0, left_max) if left_max > 0 else 0
                
                final_latents = latents_to_use[:, crop_top:crop_top+new_h, crop_left:crop_left+new_w]
                
                crop_coords = (crop_top * 8, crop_left * 8)
                target_size_px = (new_w * 8, new_h * 8)
            else:
                final_latents = latents_to_use
                crop_coords = (0, 0)
                
                if should_chunk and latents_full is None:
                     h = final_latents.shape[1]
                     w = final_latents.shape[2]
                     if h > self.max_latent_dim or w > self.max_latent_dim:
                        new_h = min(h, self.max_latent_dim)
                        new_w = min(w, self.max_latent_dim)
                        top_max = h - new_h
                        left_max = w - new_w
                        crop_top = self.worker_rng.randint(0, top_max) if top_max > 0 else 0
                        crop_left = self.worker_rng.randint(0, left_max) if left_max > 0 else 0
                        
                        final_latents = final_latents[:, crop_top:crop_top+new_h, crop_left:crop_left+new_w]
                        crop_coords = (crop_top * 8, crop_left * 8)
                        target_size_px = (new_w * 8, new_h * 8)

            item_data = {
                "latents": final_latents, 
                "embeds": data_te["embeds"], 
                "pooled": data_te["pooled"], 
                "original_sizes": data_te["original_size"], 
                "target_sizes": target_size_px, 
                "crop_coords": crop_coords,
                "latent_path": str(path_te)
            }

            if self.use_semantic_loss:
                root = self.dataset_roots.get(str(path_te))
                relative_path = data_te.get("relative_path")
                
                if root and relative_path:
                    original_image_path = Path(root) / relative_path
                    if original_image_path.exists():
                        try:
                            with Image.open(original_image_path) as raw_img: 
                                raw_img = fix_alpha_channel(raw_img)
                                
                                w_px_final = target_size_px[0]
                                h_px_final = target_size_px[1]
                                crop_t_px, crop_l_px = crop_coords
                                
                                if latents_full is not None and should_chunk:
                                    cropped_semantic_img = raw_img.crop((
                                        crop_l_px, 
                                        crop_t_px, 
                                        crop_l_px + w_px_final, 
                                        crop_t_px + h_px_final
                                    ))
                                    item_data["original_image"] = cropped_semantic_img.resize((w_px_final, h_px_final))
                                else:
                                    w_global = latents_to_use.shape[2] * 8
                                    h_global = latents_to_use.shape[1] * 8
                                    bucket_img = resize_to_fit(raw_img, w_global, h_global)
                                    cropped_semantic_img = bucket_img.crop((
                                        crop_l_px, 
                                        crop_t_px, 
                                        crop_l_px + w_px_final, 
                                        crop_t_px + h_px_final
                                    ))
                                    item_data["original_image"] = cropped_semantic_img
                        except Exception as e:
                            print(f"Error processing semantic image {original_image_path}: {e}")
                            item_data["original_image"] = None
                    else:
                        item_data["original_image"] = None
                else:
                    item_data["original_image"] = None
                    
            return item_data
        except Exception as e:
            print(f"WARNING: Skipping {self.te_files[i]}: {e}")
            return None

class TimestepSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.max_train_steps = config.MAX_TRAIN_STEPS
        self.total_tickets_needed = self.max_train_steps * config.BATCH_SIZE
        self.seed = config.SEED if config.SEED else 42
        
        allocation = getattr(config, 'TIMESTEP_ALLOCATION', None)
        self.ticket_pool = self._build_ticket_pool(allocation)
        self.pool_index = 0
        
        print(f"INFO: Initialized Ticket Pool Timestep Sampler")
        print(f"      Total tickets: {len(self.ticket_pool)}, Seed: {self.seed}")
        self._print_sampling_distribution()

    def _build_ticket_pool(self, allocation):
        use_fallback = False
        if not allocation or "counts" not in allocation or "bin_size" not in allocation:
            print("WARNING: TIMESTEP_ALLOCATION missing. Fallback to Uniform.")
            use_fallback = True
        elif sum(allocation["counts"]) == 0:
            print("WARNING: TIMESTEP_ALLOCATION sum is 0. Fallback to Uniform.")
            use_fallback = True

        if use_fallback:
            self.bin_size = 100 
            num_bins = 1000 // self.bin_size
            base = self.max_train_steps // num_bins
            counts = [base] * num_bins
            for i in range(self.max_train_steps % num_bins):
                counts[i] += 1
        else:
            self.bin_size = allocation["bin_size"] 
            counts = allocation["counts"]

        multiplier = self.config.BATCH_SIZE
        pool = []
        rng = np.random.Generator(np.random.PCG64(self.seed))
        
        for i, count in enumerate(counts):
            if count <= 0: continue
            start_t = i * self.bin_size
            end_t = min(1000, (i + 1) * self.bin_size)
            if start_t >= 1000: break
            num_tickets = int(count * multiplier)
            bin_samples = rng.integers(start_t, end_t, size=num_tickets).tolist()
            pool.extend(bin_samples)
            
        random.seed(self.seed)
        random.shuffle(pool)
        
        if len(pool) == 0:
            print("CRITICAL: Pool generation failed. Using random pool.")
            pool = [random.randint(0, 999) for _ in range(self.total_tickets_needed)]
        elif len(pool) < self.total_tickets_needed:
            print(f"WARNING: Ticket pool ({len(pool)}) < Required ({self.total_tickets_needed}). Recycling tickets.")
            while len(pool) < self.total_tickets_needed:
                needed = self.total_tickets_needed - len(pool)
                pool.extend(pool[:needed])
        elif len(pool) > self.total_tickets_needed:
            pool = pool[:self.total_tickets_needed]
            
        return pool

    def set_current_step(self, step):
        consumed_tickets = step * self.config.BATCH_SIZE
        self.pool_index = consumed_tickets % len(self.ticket_pool)
        print(f"INFO: Timestep Sampler resumed at index {self.pool_index} (Step {step})")

    def sample(self, batch_size):
        indices = []
        for _ in range(batch_size):
            if self.pool_index >= len(self.ticket_pool):
                self.pool_index = 0 
            indices.append(self.ticket_pool[self.pool_index])
            self.pool_index += 1
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _print_sampling_distribution(self):
        print("\n" + "="*80)
        print("TIMESTEP TICKET POOL DISTRIBUTION")
        print(f"Total Pool Size: {len(self.ticket_pool):,} | Seed: {self.seed}")
        print("="*80)
        
        timesteps = torch.tensor(self.ticket_pool, dtype=torch.long)
        vis_bin_size = self.bin_size 
        num_bins = math.ceil(1000 / vis_bin_size)
        
        print(f"\n{'Range':<10} | {'Count':<10} | {'%':<6} | Bar")
        print("-" * 80)
        
        max_count = 0
        counts = []
        for i in range(num_bins):
            start = i * vis_bin_size
            end = min(1000, start + vis_bin_size)
            c = ((timesteps >= start) & (timesteps < end)).sum().item()
            counts.append((f"{start}-{end}", c))
            if c > max_count: max_count = c
            
        for label, count in counts:
            pct = (count / len(self.ticket_pool)) * 100
            bar_len = int((count / max(1, max_count)) * 40)
            print(f"{label:<10} | {count:<10,d} | {pct:<6.1f}% | {'#' * bar_len}")
        print("="*80 + "\n")
        
    def update(self, raw_grad_norm):
        pass

def apply_flow_matching_shift(t, shift_factor):
    if shift_factor == 1.0:
        return t
    return (shift_factor * t) / (1.0 + (shift_factor - 1.0) * t)

def custom_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    output = {}
    if "embeds" in batch[0]:
        embeds_list = [item["embeds"] for item in batch]
        if embeds_list[0].dim() == 3: embeds_list = [e.squeeze(0) for e in embeds_list]
        max_len = max([e.shape[0] for e in embeds_list])
        padded_embeds = []
        for e in embeds_list:
            curr_len = e.shape[0]
            if curr_len < max_len:
                padding = torch.zeros((max_len - curr_len, e.shape[1]), dtype=e.dtype)
                e_padded = torch.cat([e, padding], dim=0)
                padded_embeds.append(e_padded)
            else: padded_embeds.append(e)
        output["embeds"] = torch.stack(padded_embeds)
    for k in batch[0]:
        if k == "embeds": continue
        if isinstance(batch[0][k], torch.Tensor): 
            output[k] = torch.stack([item[k] for item in batch])
        else: 
            output[k] = [item[k] for item in batch]
    return output

def create_optimizer(config, params_to_optimize):
    optimizer_type = config.OPTIMIZER_TYPE.lower()
    
    if optimizer_type == "titan":
        print("\n--- Initializing Titan Optimizer (Sync Offloading) ---")
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        if curve_points: initial_lr = max(point[1] for point in curve_points)
        else: initial_lr = config.LEARNING_RATE 
        
        titan_params = getattr(config, 'TITAN_PARAMS', {})
        defaults = default_config.TITAN_PARAMS
        final_params = {**defaults, **titan_params}

        return TitanAdamW(
            params_to_optimize, 
            lr=initial_lr, 
            betas=tuple(final_params.get('betas', [0.9, 0.999])), 
            eps=final_params.get('eps', 1e-8), 
            weight_decay=final_params.get('weight_decay', 0.01), 
            debias_strength=final_params.get('debias_strength', 1.0), 
            use_grad_centralization=final_params.get('use_grad_centralization', False), 
            gc_alpha=final_params.get('gc_alpha', 1.0)
        )

    elif optimizer_type == "raven":
        print("\n--- Initializing Raven Optimizer ---")
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        if curve_points: initial_lr = max(point[1] for point in curve_points)
        else: initial_lr = config.LEARNING_RATE 
        raven_params = getattr(config, 'RAVEN_PARAMS', {})
        return RavenAdamW(params_to_optimize, lr=initial_lr, betas=tuple(raven_params.get('betas', [0.9, 0.999])), eps=raven_params.get('eps', 1e-8), weight_decay=raven_params.get('weight_decay', 0.01), debias_strength=raven_params.get('debias_strength', 1.0), use_grad_centralization=raven_params.get('use_grad_centralization', False), gc_alpha=raven_params.get('gc_alpha', 1.0))
    
    else: 
        raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'")


# ==============================================================================
#  SDXL KEY CONVERSION LOGIC (Official Logic Embedded)
# ==============================================================================

def _get_sdxl_unet_conversion_map():
    """
    Builds the specific string replacement map for SDXL UNets.
    Matches the logic of the official conversion scripts.
    """
    unet_conversion_map = [
        # (stable-diffusion key, HF Diffusers key)
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"),
        ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"),
        ("out.2.bias", "conv_out.bias"),
        # SDXL Specific
        ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
        ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
        ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
        ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ]

    unet_conversion_map_resnet = [
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    ]

    unet_conversion_map_layer = []

    # Dynamic logic for Down/Up blocks
    for i in range(3): # Loop over downblocks
        for j in range(2): # Loop over resnets
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i > 0: # Attentions only in Down 1 and 2
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3): # Upblocks usually have 3 resnets in SDXL
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            if i < 2: # Attentions only in Up 0 and 1
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
                unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # Downsamplers
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # Upsamplers
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    # Handle oddity in output blocks
    unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))

    # Mid Block
    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
    
    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    return unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer

def convert_unet_keys_manually(unet_state_dict):
    """
    Applies the SDXL mapping to the trained state dict.
    Returns a new dictionary with Original keys.
    """
    print("   -> Generating SDXL Key Map...")
    map_static, map_resnet, map_layer = _get_sdxl_unet_conversion_map()
    
    # 1. Initialize with current keys
    mapping = {k: k for k in unet_state_dict.keys()}
    
    # 2. Apply static global mappings (Embeddings, etc)
    for sd_name, hf_name in map_static:
        if hf_name in mapping:
            mapping[hf_name] = sd_name
            
    # 3. Apply Resnet internal mappings
    # We iterate over the KEYS, replacing substrings. 
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
            
    # 4. Apply Block Layer mappings (The complex part)
    # This matches the prefixes (down_blocks.1...) and swaps them
    for k, v in mapping.items():
        for sd_part, hf_part in map_layer:
            # We use replace because it handles the tail (transformer_blocks...) automatically
            if hf_part in v:
                v = v.replace(hf_part, sd_part)
        mapping[k] = v
        
    # 5. Build final dict
    new_state_dict = {}
    for hf_name, sd_name in mapping.items():
        # Ensure the prefix is correct for the final file
        if not sd_name.startswith("model.diffusion_model."):
            final_key = f"model.diffusion_model.{sd_name}"
        else:
            final_key = sd_name
        
        new_state_dict[final_key] = unet_state_dict[hf_name]
        
    return new_state_dict

# ==============================================================================
#  SAVE FUNCTION
# ==============================================================================

def save_model(output_path, unet, base_checkpoint_path, compute_dtype):
    """
    Saves the model by:
    1. Loading the BASE checkpoint (getting VAE/CLIP for free).
    2. converting the TRAINED UNet to SDXL keys manually.
    3. Patching the BASE checkpoint and saving.
    """
    from safetensors.torch import load_file, save_file
    import gc
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nINFO: Saving complete checkpoint to: {output_path}")

    # 1. Load the original checkpoint tensors (Raw load)
    print("   -> Loading base checkpoint dictionary (Preserving VAE/CLIP)...")
    try:
        base_tensors = load_file(str(base_checkpoint_path))
    except Exception as e:
        print(f"ERROR: Could not load base checkpoint: {e}")
        return

    # 2. Prepare the Trained UNet
    print("   -> Capturing trained UNet state...")
    unet_cpu_state = {
        k: v.to("cpu", dtype=torch.float16) 
        for k, v in unet.state_dict().items()
    }
    
    # 3. Convert Keys using the embedded logic
    print("   -> Converting Diffusers keys to SDXL keys...")
    sdxl_unet_state = convert_unet_keys_manually(unet_cpu_state)
    
    # 4. Patch the Base Checkpoint
    print("   -> Merging weights...")
    
    final_tensors = base_tensors.copy()
    update_count = 0
    
    for key, value in sdxl_unet_state.items():
        final_tensors[key] = value
        update_count += 1

    print(f"   -> SUCCESS: Updated {update_count} keys.")

    # =========================================================
    #  DEBUG CHECK: Find which keys were NOT updated
    # =========================================================
    # Filter for UNet keys in the base model (starting with model.diffusion_model)
    base_unet_keys = [k for k in final_tensors.keys() if k.startswith("model.diffusion_model.")]
    
    # Keys we just updated
    updated_keys_set = set(sdxl_unet_state.keys())
    
    # Find mismatches
    missing_keys = [k for k in base_unet_keys if k not in updated_keys_set]
    
    if len(missing_keys) > 0 and len(missing_keys) < 100:
        print(f"INFO: The following {len(missing_keys)} keys exist in Base but were NOT updated (likely unused):")
        for k in missing_keys:
            print(f"      [MISSING] {k}")
    elif len(missing_keys) == 0:
         print("INFO: Perfect match! All base UNet keys were updated.")
    # =========================================================

    # 5. Save
    print("   -> Saving to disk...")
    save_file(final_tensors, str(output_path))
    print("   -> Save Complete.")

    # Cleanup
    del unet_cpu_state, sdxl_unet_state, base_tensors, final_tensors
    gc.collect()

def save_checkpoint(global_step, unet, base_checkpoint_path, optimizer, lr_scheduler, scaler, sampler, config):
    """
    Save checkpoint including model and training state.
    """
    output_dir = Path(config.OUTPUT_DIR)
    model_filename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    
    # Save complete model checkpoint
    save_model(
        output_dir / model_filename,
        unet,
        base_checkpoint_path,
        config.compute_dtype
    )
    
    # Save training state
    training_state = {
        'global_step': global_step,
        'optimizer_state': optimizer.save_cpu_state() if hasattr(optimizer, 'save_cpu_state') else optimizer.state_dict(),
        'sampler_seed': sampler.seed
    }
    
    if scaler is not None:
        training_state['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(training_state, output_dir / state_filename)
    print(f"Successfully saved training state: {state_filename}")


def main():
    config = TrainingConfig()
    
    if config.SEED:
        set_seed(config.SEED)
    
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    global_step = 0
    model_to_load = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed = config.SEED
    optimizer_state = None

    if config.RESUME_TRAINING:
        print("\n" + "="*50)
        print("--- RESUMING TRAINING SESSION ---")
        state_path = Path(config.RESUME_STATE_PATH)
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        global_step = training_state['global_step']
        initial_sampler_seed = training_state['sampler_seed']
        optimizer_state = training_state['optimizer_state']
        model_to_load = Path(config.RESUME_MODEL_PATH)
        print(f"INFO: Resuming from global step: {global_step}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print(f"--- STARTING {config.TRAINING_MODE.upper()} TRAINING SESSION ---")
        print("="*50 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if check_if_caching_needed(config):
        print("Loading base model components for caching...")
        
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            config.SINGLE_FILE_CHECKPOINT_PATH,
            torch_dtype=torch.float32,
            unet=None, 
            low_cpu_mem_usage=True
        )
        
        vae_for_caching = load_vae_only(config, device)
        if vae_for_caching is None:
            vae_for_caching = base_pipe.vae
            if config.is_rectified_flow:
                vae_for_caching.config.shift_factor = 0.1726
                vae_for_caching.config.scaling_factor = 0.1280
            else:
                vae_for_caching.config.scaling_factor = 0.13025
            
            print("INFO: Using pipeline VAE with mode-specific configuration")
            vae_for_caching = vae_for_caching.to(device)
        
        precompute_and_cache_latents(
            config,
            base_pipe.tokenizer,
            base_pipe.tokenizer_2,
            base_pipe.text_encoder,
            base_pipe.text_encoder_2,
            vae_for_caching,
            device
        )
        
        del base_pipe, vae_for_caching
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n--- Loading Model ---")
    print(f"INFO: Loading UNet for training from: {model_to_load.name}")
    
    # Removed incorrect base model loading
    # base_model_state_dict = load_file(model_to_load, device="cpu") 
    
    unet = UNet2DConditionModel.from_single_file(
        model_to_load,
        torch_dtype=config.compute_dtype,
        low_cpu_mem_usage=True
    )
    
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Initializing Dataset ---")
    dataset = ImageTextLatentDataset(config)
    print(f"INFO: Initializing BucketBatchSampler with seed: {initial_sampler_seed}")
    sampler = BucketBatchSampler(
        dataset,
        config.BATCH_SIZE,
        initial_sampler_seed,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS
    )

    unet.enable_gradient_checkpointing()
    unet.to(device)
    
    if config.MEMORY_EFFICIENT_ATTENTION == "xformers":
        unet.enable_xformers_memory_efficient_attention()
    else:
        unet.set_attn_processor(AttnProcessor2_0())
    
    print("\n--- UNet Layer Selection Report ---")
    exclusion_keywords = config.UNET_EXCLUDE_TARGETS
    trainable_layer_names = []
    frozen_layer_names = []
    
    for name, param in unet.named_parameters():
        should_exclude = False
        for keyword in exclusion_keywords:
            pattern = keyword if '*' in keyword else f"*{keyword}*"
            if fnmatch.fnmatch(name, pattern):
                should_exclude = True
                break
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
    
    optimizer = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(
        optimizer=optimizer,
        curve_points=config.LR_CUSTOM_CURVE,
        max_train_steps=config.MAX_TRAIN_STEPS
    )

    if config.RESUME_TRAINING:
        print("\n--- Restoring Training States ---")
        if optimizer_state:
            optimizer.load_cpu_state(optimizer_state)
        lr_scheduler.step(global_step)
        print("--- Resume setup complete. Starting training loop. ---")
    else:
        print("\n--- Fresh start setup complete. Starting training loop. ---")

    unet.train()
    # Removed incorrect key map generation
    # key_map = _generate_hf_to_sd_unet_key_mapping(list(unet.state_dict().keys()))
    
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    reporter = AsyncReporter(
        total_steps=config.MAX_TRAIN_STEPS,
        test_param_name="Gradient Check"
    )
    
    timestep_sampler = TimestepSampler(config, device)

    if config.RESUME_TRAINING and global_step > 0:
        timestep_sampler.set_current_step(global_step)

    scheduler = None
    if config.is_rectified_flow:
        print(f"\n--- Using Loss: Rectified Flow (Velocity Matching) ---")
        print(f"--- VAE: EQB7 (Shift: 0.1726, Scale: 0.1280) ---\n")
    else:
        prediction_type = getattr(config, "PREDICTION_TYPE", "epsilon")
        print(f"\n--- Using Loss: Standard SDXL ({prediction_type}) ---")
        print(f"--- VAE: Standard SDXL (Scale: 0.13025) ---\n")
        
        scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            subfolder="scheduler"
        )
        if prediction_type == "v_prediction":
            scheduler.config.prediction_type = "v_prediction"
        else:
            scheduler.config.prediction_type = "epsilon"

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
            if not batch:
                continue

            if "latent_path" in batch:
                accumulated_latent_paths.extend(batch["latent_path"])
            
            latents = batch["latents"].to(device, non_blocking=True)
            embeds = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            original_images = batch.get("original_image")
            batch_crop_coords = batch.get("crop_coords", [(0, 0)] * len(batch["latents"]))

            with torch.autocast(
                device_type=device.type,
                dtype=config.compute_dtype,
                enabled=True
            ):
                time_ids_list = []
                for s1, crop, s2 in zip(batch["original_sizes"], batch_crop_coords, batch["target_sizes"]):
                    time_id = torch.tensor(
                        [s1[1], s1[0], crop[0], crop[1], s2[1], s2[0]],
                        dtype=torch.float32
                    )
                    time_ids_list.append(time_id.unsqueeze(0))
                time_ids = torch.cat(time_ids_list, dim=0).to(device, dtype=config.compute_dtype)

                noise = generate_train_noise(latents, config)
                
                timesteps = timestep_sampler.sample(latents.shape[0])
                
                target = None
                noisy_latents = None
                timesteps_conditioning = None

                if config.is_rectified_flow:
                    t_continuous = timesteps.float() / 1000.0

                    rf_shift = 2.5 
                    t_shifted = apply_flow_matching_shift(t_continuous, rf_shift)

                    t_expanded = t_shifted.view(-1, 1, 1, 1)
                    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise

                    target = noise - latents

                    timesteps_conditioning = (t_shifted * 1000.0).long()

                else:
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    
                    pred_type = getattr(config, "PREDICTION_TYPE", "epsilon")
                    
                    if pred_type == "epsilon":
                        target = noise
                    elif pred_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type: {pred_type}")
                        
                    timesteps_conditioning = timesteps

                pred = unet(
                    noisy_latents,
                    timesteps_conditioning,
                    embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled,
                        "time_ids": time_ids
                    }
                ).sample

            per_pixel_loss = F.mse_loss(
                pred.float(),
                target.to(pred.dtype).float(),
                reduction="none"
            )

            if getattr(config, 'LOSS_TYPE', 'Default') == "Semantic":
                if original_images and None not in original_images:
                    b, c, h, w = latents.shape
                    semantic_map = _generate_semantic_map_for_batch(
                        original_images,
                        config.SEMANTIC_LOSS_BLEND,
                        config.SEMANTIC_LOSS_STRENGTH,
                        target_size=(w, h),
                        device=latents.device,
                        dtype=torch.float32,
                        num_channels=c
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
                
                if isinstance(optimizer, TitanAdamW):
                    raw_grad_norm = optimizer.clip_grad_norm(
                        config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else 0.0
                    )
                else:
                    raw_grad_norm = torch.nn.utils.clip_grad_norm_(
                        params_to_optimize,
                        float('inf')
                    )
                    timestep_sampler.update(raw_grad_norm.item())
                    
                    if config.CLIP_GRAD_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(
                            params_to_optimize,
                            config.CLIP_GRAD_NORM
                        )
                
                if isinstance(raw_grad_norm, torch.Tensor):
                    raw_grad_norm = raw_grad_norm.item()
                
                clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm
                timestep_sampler.update(raw_grad_norm)

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
                    'update_delta': update_status,
                    'optim_step_time': optim_step_time,
                    'avg_optim_step_time': sum(optim_step_times) / len(optim_step_times)
                }
                
                reporter.check_and_report_anomaly(
                    global_step + 1,
                    raw_grad_norm,
                    clipped_grad_norm,
                    config,
                    accumulated_latent_paths
                )
                diagnostics.reset()
                accumulated_latent_paths.clear()

            step_duration = time.time() - last_step_time
            global_step_times.append(step_duration)
            last_step_time = time.time()
            elapsed_time = time.time() - training_start_time
            eta_seconds = (config.MAX_TRAIN_STEPS - (global_step + 1)) * \
                (sum(global_step_times) / len(global_step_times))
            
            timing_data = {
                'raw_step_time': step_duration,
                'elapsed_time': elapsed_time,
                'eta': eta_seconds,
                'loss': loss.item(),
                'timestep': timesteps[0].item()
            }
            
            reporter.log_step(
                global_step,
                timing_data=timing_data,
                diag_data=diag_data_to_log
            )
            global_step += 1
            
            if config.SAVE_EVERY_N_STEPS > 0 and \
               global_step % config.SAVE_EVERY_N_STEPS == 0:
                print(f"\n--- Saving checkpoint at step {global_step} ---")

                save_checkpoint(
                    global_step,
                    unet,
                    model_to_load,  # Pass the base checkpoint path
                    optimizer,
                    lr_scheduler,
                    None,  # scaler
                    sampler,
                    config
                )

    print("\nTraining complete.")
    reporter.shutdown()
    
    output_path = OUTPUT_DIR / f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_unified_{str(uuid.uuid4())[:4]}.safetensors"
    save_model(
        output_path,
        unet,
        model_to_load,  # Pass the base checkpoint path
        config.compute_dtype
    )
    print("All tasks complete. Final model saved.")

if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()