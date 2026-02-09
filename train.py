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
from PIL import Image, TiffImagePlugin, ImageFile, ImageOps
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
from optimizer.velorms import VeloRMS
import uuid
from diffusers.models.attention_processor import (
        AttnProcessor2_0, 
        FusedAttnProcessor2_0
    )
from tools.semantic import generate_semantic_map_batch_softmax
warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def set_attention_processor(unet, attention_mode="flash_attn"):
    """
    attention_mode: "flash_attn", "fused", "xformers", "sdpa"
    """
    if attention_mode == "flex_attn":
        try:
            from diffusers.models.attention_processor import FlexAttnProcessor
            unet.set_attn_processor(FlexAttnProcessor())
            print("INFO: Using Flex Attention")
        except ImportError:
            print("WARNING: FlexAttention processor not available, falling back to SDPA")
            unet.set_attn_processor(AttnProcessor2_0())

    if attention_mode == "cudnn":
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)  # Disable FA to force CuDNN
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # Fallback
            unet.set_attn_processor(AttnProcessor2_0())
            print("INFO: Using CuDNN SDPA backend (PyTorch 2.5+ optimized)")
        else:
            print("WARNING: CuDNN SDPA requires PyTorch 2.5+, falling back to standard SDPA")
            unet.set_attn_processor(AttnProcessor2_0())
            
            
    elif attention_mode == "xformers (Only if no Flash)":
        unet.enable_xformers_memory_efficient_attention()
        print("INFO: Using xFormers")
        
    else:  # sdpa
        unet.set_attn_processor(AttnProcessor2_0())
        print("INFO: Using SDPA (PyTorch native)")

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

def generate_train_noise(latents, config, step, seed):
    g = torch.Generator(device=latents.device)

    step_seed = (seed + step) % (2**32 - 1)
    g.manual_seed(step_seed)
    
    return torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=g)

class TrainingConfig:
    def __init__(self):
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        self._load_from_user_config()
        self._type_check_and_correct()
        self.compute_dtype = torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16
        self.is_rectified_flow = getattr(self, "TRAINING_MODE", "Standard (SDXL)") == "Rectified Flow (SDXL)"

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
    def __init__(self, optimizer, curve_points, total_micro_steps):
        self.optimizer = optimizer
        self.curve_points = sorted(curve_points, key=lambda p: p[0])
        self.total_micro_steps = max(total_micro_steps, 1)
        self.current_micro_step = 0
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
        normalized_position = self.current_micro_step / max(self.total_micro_steps - 1, 1)
        lr = self._interpolate_lr(normalized_position)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def step(self, micro_step):
        self.current_micro_step = micro_step
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

    def set_epoch(self, epoch):
        self.epoch = epoch

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
        current_area = width * height
        
        # If image is smaller than target and upscaling is disabled, just snap to a nearby grid
        if not self.should_upscale and current_area < self.target_area:
            w = int(width // self.stride) * self.stride
            h = int(height // self.stride) * self.stride
            return max(w, self.stride), max(h, self.stride)

        # 1. Calculate the ideal scale to match the target pixel area
        scale = math.sqrt(self.target_area / current_area)
        
        # 2. Determine the "ideal" floating-point dimensions
        scaled_w = width * scale
        scaled_h = height * scale
        
        target_w = int(round(scaled_w / self.stride) * self.stride)
        target_h = int(round(scaled_h / self.stride) * self.stride)
        
        return max(target_w, self.stride), max(target_h, self.stride)

def smart_resize(image, target_w, target_h):

    
    src_w, src_h = image.size
    
    # 1. Calculate aspect ratios to determine crop direction
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h
    
    if src_aspect > target_aspect:
        # Source is wider than target -> Crop width, keep full height
        crop_w = src_h * target_aspect
        crop_h = src_h
        x_offset = (src_w - crop_w) / 2.0
        y_offset = 0.0
    else:
        # Source is taller than target -> Crop height, keep full width
        crop_w = src_w
        crop_h = src_w / target_aspect
        x_offset = 0.0
        y_offset = (src_h - crop_h) / 2.0
        
    # 3. Define the box (left, top, right, bottom)
    box = (x_offset, y_offset, x_offset + crop_w, y_offset + crop_h)
    
    return image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS, box=box)

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

def compute_text_embeddings_sdxl(captions, t1, t2, te1, te2, device):
    """
    Standard SDXL text embedding computation.
    Returns properly sized embeddings for SDXL UNet.
    """
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    
    for caption in captions:
        with torch.no_grad():
            # Tokenize with proper truncation/padding to 77 tokens
            tokens_1 = t1(
                caption,
                padding="max_length",
                max_length=t1.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            
            tokens_2 = t2(
                caption,
                padding="max_length",
                max_length=t2.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(device)
            
            # Get embeddings from text encoders
            # Use penultimate layer for CLIP ViT-L
            enc_1_output = te1(tokens_1, output_hidden_states=True)
            prompt_embeds_1 = enc_1_output.hidden_states[-2]  # [1, 77, 768]
            
            # Use penultimate layer for OpenCLIP ViT-G  
            enc_2_output = te2(tokens_2, output_hidden_states=True)
            prompt_embeds_2 = enc_2_output.hidden_states[-2]  # [1, 77, 1280]
            
            # Concatenate along feature dimension
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # [1, 77, 2048]
            
            # Pooled embeddings from second encoder (for time embeddings)
            pooled_prompt_embeds = enc_2_output[0]  # [1, 1280]
            
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    
    # Stack batches
    prompt_embeds = torch.cat(prompt_embeds_list, dim=0)  # [B, 77, 2048]
    pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)  # [B, 1280]
    
    return prompt_embeds, pooled_prompt_embeds

def check_if_caching_needed(config):
    needs_caching = False
    
    cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    semantic_cache_folder_name = ".semantic_maps_cache"
    use_semantic_loss = getattr(config, 'LOSS_TYPE', 'MSE') == "Semantic"
    
    # Check for Null Embeddings if Dropout is ON
    if getattr(config, "UNCONDITIONAL_DROPOUT", False):
        if config.INSTANCE_DATASETS:
            ds0 = config.INSTANCE_DATASETS[0]
            if not (Path(ds0["path"]) / cache_folder_name / "null_embeds.pt").exists():
                print("INFO: Null embeddings for unconditional dropout missing. Caching needed.")
                needs_caching = True

    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        
        if not root.exists():
            print(f"WARNING: Dataset path {root} does not exist, skipping.")
            continue
            
        image_paths = [
            p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] 
            for p in root.rglob(f"*{ext}")
        ]
        
        if not image_paths:
            print(f"INFO: No images found in {root}, skipping.")
            continue
            
        cache_dir = root / cache_folder_name
        
        if not cache_dir.exists():
            print(f"INFO: Cache directory doesn't exist for {root}, caching needed.")
            needs_caching = True
            continue
        
        # Check semantic cache if needed
        if use_semantic_loss:
            semantic_cache_dir = root / semantic_cache_folder_name
            
            if not semantic_cache_dir.exists():
                print(f"INFO: Semantic cache directory doesn't exist for {root}, caching needed.")
                needs_caching = True
                continue
        
        cached_te_files = list(cache_dir.glob("*_te.pt"))
        
        if len(cached_te_files) < len(image_paths):
            print(f"INFO: Found {len(image_paths)} images but only {len(cached_te_files)} cached files in {root}")
            needs_caching = True
        else:
            # Check if semantic maps exist for all files
            if use_semantic_loss:
                semantic_cache_dir = root / semantic_cache_folder_name
                missing_semantic = 0
                
                for te_file in cached_te_files:
                    safe_name = te_file.stem.replace('_te', '')
                    sem_path = semantic_cache_dir / f"{safe_name}_sem.pt"
                    
                    if not sem_path.exists():
                        missing_semantic += 1
                
                if missing_semantic > 0:
                    print(f"INFO: {missing_semantic} semantic maps missing in {root}")
                    needs_caching = True
                else:
                    print(f"INFO: All {len(image_paths)} images and semantic maps cached in {root}")
            else:
                print(f"INFO: All {len(image_paths)} images appear to be cached in {root}")
            
    return needs_caching

def load_unet_robust(path, compute_dtype, target_channels=4):
    print(f"INFO: Loading UNet from: {Path(path).name}")
    
    # Base arguments for loading
    load_kwargs = {
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }

    if target_channels is not None and target_channels != 4:
        print(f"INFO: Config requests {target_channels} channels. Forcing UNet input/output dimensions...")
        load_kwargs["in_channels"] = target_channels
        load_kwargs["out_channels"] = target_channels
        load_kwargs["ignore_mismatched_sizes"] = True

    try:
        unet = UNet2DConditionModel.from_single_file(path, **load_kwargs)
        
        print(f"INFO: Loaded UNet (Channels: In={unet.config.in_channels} / Out={unet.config.out_channels})")
        print(f"INFO: UNet attention implementation: {unet.config.attention_type if hasattr(unet.config, 'attention_type') else 'flash_attention_2'}")
        
        return unet

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load UNet with target_channels={target_channels}.")
        print(f"Error Details: {e}")
        # If we forced 32 and it failed, or if we tried 4 and it failed, we raise.
        raise e

def load_vae_robust(path, device, target_channels=None):
    """
    Safely loads a VAE. If target_channels is set in config, it forces that configuration.
    Otherwise, it attempts auto-detection.
    """
    print(f"INFO: Attempting to load VAE from: {path}")
    
    # If config explicitly sets channels, try that first
    if target_channels is not None:
        print(f"INFO: Config requests VAE with {target_channels} channels.")
        try:
            vae = AutoencoderKL.from_single_file(
                path, 
                torch_dtype=torch.float32,
                latent_channels=target_channels,
                ignore_mismatched_sizes=True 
            )
            print(f"INFO: Successfully loaded VAE with {target_channels} channels (forced by config).")
            return vae.to(device)
        except Exception as e:
            print(f"WARNING: Failed to force load VAE with {target_channels} channels: {e}")
            print("INFO: Falling back to auto-detection...")

    # Standard Auto-detection / Fallback logic
    try:
        # 1. Try loading as a standard SDXL VAE first
        vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float32)
        print("INFO: Loaded standard SDXL VAE (4 channels).")
        
    except Exception as e:
        # 2. Check for the specific shape mismatch error
        err_str = str(e)
        if "conv_out.weight" in err_str and "but got torch.Size([64" in err_str:
            print("INFO: Detected Flux/SDXL-Modified VAE (32 channels). Switching configuration...")
            try:
                # Force 32 channels and ignore size mismatches
                vae = AutoencoderKL.from_single_file(
                    path, 
                    torch_dtype=torch.float32,
                    latent_channels=32,
                    ignore_mismatched_sizes=True
                )
                print("INFO: Successfully loaded 32-channel VAE.")
            except Exception as e2:
                print(f"CRITICAL ERROR: Failed to load VAE with 32-channel config: {e2}")
                raise e2
        else:
            raise e
            
    return vae.to(device)

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config):
        print("\n" + "="*60)
        print("INFO: Datasets already cached.")
        print("="*60 + "\n")
        return
    
    cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    
    # Determine if we need semantic map caching
    use_semantic_loss = getattr(config, 'LOSS_TYPE', 'MSE') == "Semantic"
    semantic_cache_folder_name = ".semantic_maps_cache"
    
    print("\n" + "="*60)
    print(f"STARTING VRAM-OPTIMIZED CACHING FOR: {config.TRAINING_MODE}")
    print(f"Using cache folder: {cache_folder_name}")
    if use_semantic_loss:
        print(f"Semantic maps will be cached to: {semantic_cache_folder_name}")
    print(f"VAE Channels: {vae.config.latent_channels}")
    print("="*60 + "\n")
    
    calc = ResolutionCalculator(
        config.TARGET_PIXEL_AREA, 
        stride=64, 
        should_upscale=config.SHOULD_UPSCALE
    )
    
    vae.to(device, dtype=torch.float32)
    vae.enable_tiling()
    vae.enable_slicing()
    
    te1.to(device)
    te2.to(device)
    
    print("INFO: Computing Null Embeddings for Unconditional Dropout...")
    with torch.no_grad():
        null_embeds, null_pooled = compute_text_embeddings_sdxl([""], t1, t2, te1, te2, device)
    
    null_embeds = null_embeds.cpu()
    null_pooled = null_pooled.cpu()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        cache_dir = root / cache_folder_name
        cache_dir.mkdir(exist_ok=True)
        
        # Create semantic cache directory if needed
        semantic_cache_dir = root / semantic_cache_folder_name if use_semantic_loss else None
        if use_semantic_loss:
            semantic_cache_dir.mkdir(exist_ok=True)
        
        torch.save({"embeds": null_embeds, "pooled": null_pooled}, cache_dir / "null_embeds.pt")
        
        paths = [
            p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] 
            for p in root.rglob(f"*{ext}")
        ]
        
        to_process = []
        for p in paths:
            relative_path = p.relative_to(root)
            safe_stem = str(relative_path.with_suffix('')).replace(os.sep, '_').replace('/', '_').replace('\\', '_')
            
            te_exists = (cache_dir / f"{safe_stem}_te.pt").exists()
            lat_exists = (cache_dir / f"{safe_stem}_lat.pt").exists()
            sem_exists = (semantic_cache_dir / f"{safe_stem}_sem.pt").exists() if use_semantic_loss else True
            
            if not te_exists or not lat_exists or (use_semantic_loss and not sem_exists):
                to_process.append(p)

        if not to_process:
            print(f"INFO: All images in {root.name} are already cached (including semantic maps).")
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
        
        for batch_idx, ((w, h), batch_meta) in enumerate(tqdm(batches, desc=f"Encoding {root.name}")):
            captions = [m['caption'] for m in batch_meta]
            
            embeds, pooled = compute_text_embeddings_sdxl(captions, t1, t2, te1, te2, device)
            embeds = embeds.to(dtype=torch.float32)
            pooled = pooled.to(dtype=torch.float32)
            
            images_global = []
            valid_meta_final = []
            original_images_for_semantic = [] if use_semantic_loss else None
            
            for m in batch_meta:
                try:
                    with Image.open(m['ip']) as img:
                        img = fix_alpha_channel(img)
                        img_resized = smart_resize(img, w, h)
                        images_global.append(transform(img_resized))
                        valid_meta_final.append(m)
                        
                        # Store original PIL image for semantic map generation
                        if use_semantic_loss:
                            # Reload original for semantic processing (need full quality)
                            with Image.open(m['ip']) as orig_img:
                                orig_img = fix_alpha_channel(orig_img)
                                original_images_for_semantic.append(orig_img)
                                
                except Exception as e:
                    print(f"Skipping {m['ip']}: {e}")
            
            if not images_global:
                continue
            
            pixel_values_global = torch.stack(images_global).to(device, dtype=torch.float32)

            with torch.no_grad():
                dist = vae.encode(pixel_values_global).latent_dist
                latents_global = dist.mean
                
                if batch_idx == 0:
                    print(f"\nDEBUG - RAW VAE OUTPUT (before normalization):")
                    print(f"  Raw latent mean: {latents_global.mean().item():.4f}")
                    print(f"  Raw latent std: {latents_global.std().item():.4f}")
                
                if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                    latents_global = (latents_global - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    latents_global = latents_global * vae.config.scaling_factor
                
                if batch_idx == 0:
                    print(f"\nDEBUG - NORMALIZED LATENTS (after normalization):")
                    print(f"  Normalized mean: {latents_global.mean().item():.4f} (target: ~0.0)")
                    print(f"  Normalized std: {latents_global.std().item():.4f} (target: ~1.0)")
                    print(f"  VAE shift_factor: {vae.config.shift_factor}")
                    print(f"  VAE scaling_factor: {vae.config.scaling_factor}")
                    print("-" * 60)
                    
                if batch_idx == 0:
                    print("\n=== VAE CHANNEL BALANCE DIAGNOSTIC ===")
                    for i in range(latents_global.shape[1]):
                        ch_mean = latents_global[:, i].mean().item()
                        ch_std = latents_global[:, i].std().item()
                        print(f"Channel {i}: mean={ch_mean:.4f}, std={ch_std:.4f}")
                    print("=" * 50 + "\n")

                if batch_idx == 0:
                    print(f"DEBUG: Caching Latent Shape: {latents_global.shape} (Channels: {latents_global.shape[1]})")
                    print(f"DEBUG: Latent mean: {latents_global.mean().item():.4f}")
                    print(f"DEBUG: Latent std: {latents_global.std().item():.4f}")

                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: mean={latents_global.mean().item():.4f}, std={latents_global.std().item():.4f}")
                
            latents_global = latents_global.cpu()
            
            # Generate and cache semantic maps if needed
            semantic_maps = None
            if use_semantic_loss and original_images_for_semantic:
                char_weight = getattr(config, "SEMANTIC_CHAR_WEIGHT", 1.0)
                detail_weight = getattr(config, "SEMANTIC_DETAIL_WEIGHT", 1.0)
                # Target size is latent size (w/8, h/8 for SDXL)
                latent_w, latent_h = w // 8, h // 8
                
                semantic_maps = generate_semantic_map_batch_softmax(
                    images=original_images_for_semantic,
                    char_weight=char_weight,
                    detail_weight=detail_weight,
                    target_size=(latent_w, latent_h),
                    device="cpu",  # Keep on CPU to save VRAM during caching
                    dtype=torch.float32,
                    num_channels=latents_global.shape[1]  # Match latent channels
                )
                semantic_maps = semantic_maps.cpu()
                
                if batch_idx == 0:
                    print(f"\nDEBUG: Semantic map shape: {semantic_maps.shape}")
                    print(f"DEBUG: Semantic map range: [{semantic_maps.min():.4f}, {semantic_maps.max():.4f}]")
            
            for j, m in enumerate(valid_meta_final):
                relative_path = m['ip'].relative_to(root)
                safe_filename = str(relative_path.with_suffix('')).replace(os.sep, '_').replace('/', '_').replace('\\', '_')
                
                path_te = cache_dir / f"{safe_filename}_te.pt"
                path_lat = cache_dir / f"{safe_filename}_lat.pt"
                path_sem = semantic_cache_dir / f"{safe_filename}_sem.pt" if use_semantic_loss else None
                
                torch.save({
                    "original_stem": m['ip'].stem,
                    "relative_path": str(relative_path),
                    "original_size": m["original_size"],
                    "target_size": (w, h),
                    "embeds": embeds[j].cpu(),
                    "pooled": pooled[j].cpu(),
                    "vae_shift": vae.config.shift_factor,
                    "vae_scale": vae.config.scaling_factor,
                }, path_te)
                
                torch.save(latents_global[j], path_lat)
                
                # Save semantic map if generated
                if semantic_maps is not None:
                    torch.save(semantic_maps[j], path_sem)
            
            del pixel_values_global, latents_global, embeds, pooled
            if semantic_maps is not None:
                del semantic_maps
            if original_images_for_semantic:
                del original_images_for_semantic
            
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

    print("\nINFO: Caching Complete. Cleaning up...")
    te1.cpu()
    te2.cpu()
    gc.collect()
    torch.cuda.empty_cache()

class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.te_files = []
        self.dataset_roots = {} 
        self.is_rectified_flow = config.is_rectified_flow
        self.seed = config.SEED if config.SEED else 42
        self.worker_rng = None
        
        cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
        semantic_cache_folder_name = ".semantic_maps_cache"

        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / cache_folder_name
            if not cache_dir.exists(): 
                continue
            
            files = list(cache_dir.glob("*_te.pt"))
            for f in files:
                self.dataset_roots[str(f)] = root
                
            self.te_files.extend(files * int(ds.get("repeats", 1)))

        if not self.te_files: 
            raise ValueError("No cached files found.")
        
        random.shuffle(self.te_files)
        
        self.bucket_keys = []
        
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
        
        self.dropout_prob = getattr(config, "UNCONDITIONAL_DROPOUT_CHANCE", 0.0) if getattr(config, "UNCONDITIONAL_DROPOUT", False) else 0.0
        self.null_embeds = None
        self.null_pooled = None

        self.target_shift = getattr(config, 'VAE_SHIFT_FACTOR', None)
        self.target_scale = getattr(config, 'VAE_SCALING_FACTOR', None)
        
        # --- SEMANTIC LOSS CONFIGURATION ---
        self.use_semantic_loss = (getattr(config, 'LOSS_TYPE', 'MSE') == "Semantic")
        self.semantic_cache_roots = {}  # Map te_file path to semantic cache dir
        
        if self.use_semantic_loss:
            print("INFO: Semantic loss is ENABLED. Loading from cached semantic maps.")
            # Build mapping to semantic cache directories
            for ds in config.INSTANCE_DATASETS:
                root = Path(ds["path"])
                semantic_cache_dir = root / semantic_cache_folder_name
                cache_dir = root / cache_folder_name
                
                if not semantic_cache_dir.exists():
                    print(f"WARNING: Semantic cache directory not found: {semantic_cache_dir}")
                    print(f"         Please run caching first with LOSS_TYPE='Semantic' in config.")
                    continue
                
                # Map each te_file to its semantic cache directory
                for f in cache_dir.glob("*_te.pt"):
                    self.semantic_cache_roots[str(f)] = semantic_cache_dir
            
            # Verify we have semantic maps for all files
            missing_semantic = []
            for te_file in self.te_files:
                semantic_dir = self.semantic_cache_roots.get(str(te_file))
                if semantic_dir:
                    safe_name = Path(te_file).stem.replace('_te', '')
                    sem_path = semantic_dir / f"{safe_name}_sem.pt"
                    if not sem_path.exists():
                        missing_semantic.append(te_file)
                else:
                    missing_semantic.append(te_file)
            
            if missing_semantic:
                print(f"WARNING: {len(missing_semantic)} files missing semantic maps.")
                print(f"         First few: {[str(p) for p in missing_semantic[:3]]}")
        # --- END SEMANTIC LOSS CONFIGURATION ---

        if self.dropout_prob > 0:
            found_null = False
            for ds in config.INSTANCE_DATASETS:
                root = Path(ds["path"])
                cache_dir = root / cache_folder_name
                null_path = cache_dir / "null_embeds.pt"
                if null_path.exists():
                    try:
                        null_data = torch.load(null_path, map_location="cpu")
                        self.null_embeds = null_data["embeds"]
                        if self.null_embeds.dim() == 3:
                            self.null_embeds = self.null_embeds.squeeze(0)
                        self.null_pooled = null_data["pooled"]
                        if self.null_pooled.dim() == 2:
                            self.null_pooled = self.null_pooled.squeeze(0)
                        found_null = True
                        break
                    except Exception as e:
                        print(f"WARNING: Failed to load null embeddings: {e}")
            
            if not found_null:
                print("WARNING: Unconditional Dropout enabled but null_embeds.pt not found. Disabling dropout.")
                self.dropout_prob = 0.0
                
        print(f"INFO: Dataset initialized with {len(self.te_files)} samples.")

    def _validate_cache_compatibility(self):
        """Check first few cache files match expected VAE config. Halts if mismatch detected."""
        print("INFO: Validating cache VAE compatibility...")
        incompatible_count = 0
        checked = 0
        first_mismatch = None
        
        for f in self.te_files[:20]:  # Check first 20 files
            try:
                data_te = torch.load(f, map_location="cpu")
                cached_shift = data_te.get("vae_shift", 0.0) or 0.0
                cached_scale = data_te.get("vae_scale", 0.0) or 0.13025
                
                scale_match = abs(cached_scale - self.target_scale) < 1e-4
                shift_match = (self.target_shift is None or 
                              abs(cached_shift - self.target_shift) < 1e-4)
                
                if not scale_match or not shift_match:
                    incompatible_count += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            'file': f,
                            'cached_shift': cached_shift,
                            'cached_scale': cached_scale,
                            'expected_shift': self.target_shift,
                            'expected_scale': self.target_scale
                        }
                        
                checked += 1
            except Exception as e:
                print(f"WARNING: Could not check cache file {f}: {e}")
        
        if incompatible_count > 0:
            dataset_paths = list(set(self.dataset_roots.values()))
            
            print(f"\n{'='*70}")
            print("CRITICAL: CACHE VAE MISMATCH DETECTED")
            print(f"{'='*70}")
            print(f"Checked {checked} files, found {incompatible_count} incompatible.")
            print(f"\nFirst mismatch:")
            print(f"  File: {first_mismatch['file']}")
            print(f"  Cached:  shift={first_mismatch['cached_shift']}, scale={first_mismatch['cached_scale']}")
            print(f"  Expected: shift={first_mismatch['expected_shift']}, scale={first_mismatch['expected_scale']}")
            print(f"\nSOLUTION:")
            print(f"  1. Delete cache folder(s): {dataset_paths}")
            print(f"  2. Re-run training to regenerate cache with correct VAE settings")
            print(f"  3. Or adjust VAE_SHIFT_FACTOR/VAE_SCALING_FACTOR to match cache")
            print(f"{'='*70}\n")
            raise ValueError(f"Cache VAE config mismatch. {incompatible_count}/{checked} files incompatible. Please delete cache and regenerate.")
        else:
            print(f"INFO: Cache VAE config validated ({checked} samples checked).")
            
    def __len__(self): 
        return len(self.te_files)

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
            
            path_str = str(path_te)
            path_lat = Path(path_str.replace("_te.pt", "_lat.pt"))
            
            latents = torch.load(path_lat, map_location="cpu")
            
            if torch.isnan(latents).any() or torch.isinf(latents).any(): 
                return None

            final_latents = latents
            
            item_data = {
                "latents": final_latents,
                "embeds": data_te["embeds"].squeeze(0) if data_te["embeds"].dim() == 3 else data_te["embeds"],
                "pooled": data_te["pooled"].squeeze(0) if data_te["pooled"].dim() == 2 else data_te["pooled"],
                "original_sizes": data_te["original_size"],
                "target_sizes": data_te["target_size"],
                "crop_coords": (0, 0),
                "latent_path": str(path_te)
            }
            
            if self.dropout_prob > 0 and self.worker_rng.random() < self.dropout_prob:
                item_data["embeds"] = self.null_embeds
                item_data["pooled"] = self.null_pooled

            # --- LOAD CACHED SEMANTIC MAP ---
            if self.use_semantic_loss:
                semantic_dir = self.semantic_cache_roots.get(str(path_te))
                if semantic_dir:
                    safe_name = Path(path_te).stem.replace('_te', '')
                    sem_path = semantic_dir / f"{safe_name}_sem.pt"
                    
                    if sem_path.exists():
                        semantic_map = torch.load(sem_path, map_location="cpu")
                        item_data["semantic_map"] = semantic_map
                    else:
                        print(f"WARNING: Semantic map not found: {sem_path}")
                        # Create fallback uniform map
                        c, h, w = final_latents.shape
                        item_data["semantic_map"] = torch.ones((c, h, w), dtype=torch.float32)
                else:
                    # Create fallback uniform map
                    c, h, w = final_latents.shape
                    item_data["semantic_map"] = torch.ones((c, h, w), dtype=torch.float32)
            # --- END SEMANTIC MAP LOADING ---

            return item_data
            
        except Exception as e:
            print(f"WARNING: Skipping {self.te_files[i]}: {e}")
            return None
        
class TimestepSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Calculate TOTAL tickets needed for the entire training run (including accumulation)
        # This ensures we have enough tickets for every single micro-step.
        total_micro_steps = config.MAX_TRAIN_STEPS * config.GRADIENT_ACCUMULATION_STEPS
        self.total_tickets_needed = total_micro_steps * config.BATCH_SIZE
        
        self.seed = config.SEED if config.SEED else 42
        
        # RF-specific settings from config
        self.is_rectified_flow = getattr(config, "is_rectified_flow", False)
        self.shift_factor = getattr(config, "RF_SHIFT_FACTOR", 3.0)
        self.use_dynamic_shift = getattr(config, "RF_USE_DYNAMIC_SHIFT", False)
        self.base_pixels = getattr(config, "RF_BASE_PIXELS", 1048576)
        
        # Existing ticket pool logic - unchanged
        allocation = getattr(config, 'TIMESTEP_ALLOCATION', None)
        self.ticket_pool = self._build_ticket_pool(allocation)
        self.pool_index = 0
        
        print(f"INFO: Initialized Ticket Pool Timestep Sampler")
        print(f"      Total tickets: {len(self.ticket_pool)}, Seed: {self.seed}")
        if self.is_rectified_flow:
            print(f"      RF Shift: {self.shift_factor}")
            if self.use_dynamic_shift:
                print(f"      Dynamic Shift: Enabled (Base: {self.base_pixels}px)")
        self._print_sampling_distribution()

    def _build_ticket_pool(self, allocation):
        # Your existing ticket pool logic - completely unchanged
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
            # Base distribution for total needs
            base = self.total_tickets_needed // num_bins // self.config.BATCH_SIZE
            counts = [base] * num_bins
            for i in range((self.total_tickets_needed // self.config.BATCH_SIZE) % num_bins):
                counts[i] += 1
        else:
            self.bin_size = allocation["bin_size"] 
            counts = allocation["counts"]

        multiplier = self.config.BATCH_SIZE
        # Scale the allocation counts (usually meant for MAX_STEPS) to cover ACCUMULATION
        # Logic: (Total Needed / Sum of Counts) scaling factor
        total_counts = sum(counts)
        scale_factor = (self.total_tickets_needed / multiplier) / total_counts if total_counts > 0 else 1.0
        
        pool = []
        rng = np.random.Generator(np.random.PCG64(self.seed))
        
        for i, count in enumerate(counts):
            if count <= 0: 
                continue
            start_t = i * self.bin_size
            end_t = min(1000, (i + 1) * self.bin_size)
            if start_t >= 1000: 
                break
            
            # Scale count to cover full training duration + batch size
            num_tickets = int(count * scale_factor * multiplier)
            if num_tickets <= 0: num_tickets = 1

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

    def get_dynamic_shift(self, height, width):
        """Calculate dynamic shift based on image resolution"""
        if not self.use_dynamic_shift:
            return self.shift_factor
        
        current_pixels = height * width
        ratio = current_pixels / self.base_pixels
        # Square root scaling for area ratio
        dynamic_shift = self.shift_factor * math.sqrt(ratio)
        
        return dynamic_shift

    def apply_flow_shift(self, t_normalized, shift_factor):
        """
        SD3/Flux schedule warping equation.
        t_normalized: [0, 1] tensor
        shift_factor: float (1.0 = no shift, 3.0 = SD3/Flux standard)
        
        Returns: shifted t in [0, 1]
        """
        if shift_factor == 1.0 or shift_factor is None:
            return t_normalized
        # The SD3/Flux equation
        return (shift_factor * t_normalized) / (1.0 + (shift_factor - 1.0) * t_normalized)

    def set_current_step(self, micro_step):
        # Aligns the pool based on TOTAL FORWARD PASSES (micro_step)
        consumed_tickets = micro_step * self.config.BATCH_SIZE
        self.pool_index = consumed_tickets % len(self.ticket_pool)
        print(f"INFO: Timestep Sampler resumed at index {self.pool_index} (Micro Step {micro_step})")

    def sample(self, batch_size):
        # Unchanged - just return timesteps from pool
        indices = []
        for _ in range(batch_size):
            if self.pool_index >= len(self.ticket_pool):
                self.pool_index = 0 
            indices.append(self.ticket_pool[self.pool_index])
            self.pool_index += 1
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _print_sampling_distribution(self):
        # Your existing visualization code - unchanged
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
            if c > max_count: 
                max_count = c
            
        for label, count in counts:
            pct = (count / len(self.ticket_pool)) * 100
            bar_len = int((count / max(1, max_count)) * 40)
            print(f"{label:<10} | {count:<10,d} | {pct:<6.1f}% | {'#' * bar_len}")
        print("="*80 + "\n")
        
    def update(self, raw_grad_norm):
        pass


def custom_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch: 
        return {}
    
    output = {}
    for k in batch[0]:
        if k == "original_image":
            # Pass PIL images as a simple list, do not stack
            output[k] = [item[k] for item in batch]
        elif isinstance(batch[0][k], torch.Tensor):
            output[k] = torch.stack([item[k] for item in batch])
        else:
            output[k] = [item[k] for item in batch]
    
    return output

def create_optimizer(config, params_to_optimize):
    optimizer_type = config.OPTIMIZER_TYPE.lower()

    def print_lr_graph(curve_points, max_steps):
        print("--- Custom Learning Rate Schedule ---")
        if not curve_points:
            print("No custom curve points provided.")
            return

        print("Step Percentage -> Learning Rate")
        for point in curve_points:
            print(f"{point[0]*100:5.1f}% -> {point[1]:.2e}")

        graph_width = 50
        graph_height = 10
        graph = [[' ' for _ in range(graph_width)] for _ in range(graph_height)]
        
        min_lr = min(p[1] for p in curve_points)
        max_lr = max(p[1] for p in curve_points)

        def interpolate(x, x0, y0, x1, y1):
            if x1 == x0:
                return y0
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        for i in range(graph_width):
            step_percent = i / (graph_width -1)
            
            lr = 0
            for j in range(len(curve_points) - 1):
                if curve_points[j][0] <= step_percent <= curve_points[j+1][0]:
                    lr = interpolate(step_percent, curve_points[j][0], curve_points[j][1], curve_points[j+1][0], curve_points[j+1][1])
                    break
            
            if max_lr > min_lr:
                y = int(((lr - min_lr) / (max_lr - min_lr)) * (graph_height - 1))
                graph[graph_height - 1 - y][i] = '*'
            else:
                graph[graph_height -1][i] = '*'


        print("\nLearning Rate Schedule:")
        for i, row in enumerate(graph):
            lr_label = max_lr - i * (max_lr-min_lr)/(graph_height-1) if graph_height > 1 and max_lr > min_lr else max_lr
            print(f"{lr_label:.2e} |{''.join(row)}|")
        print("       " + "-" * (graph_width+2))
        print("       0%" + " " * (graph_width - 5) + "100%")
        print("---")


    if optimizer_type == "titan":
        print("\n--- Initializing Titan Optimizer (Sync Offloading) ---")
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        if curve_points:
            initial_lr = max(point[1] for point in curve_points)
            print_lr_graph(curve_points, config.MAX_TRAIN_STEPS)
        else:
            initial_lr = config.LEARNING_RATE

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
        if curve_points:
            initial_lr = max(point[1] for point in curve_points)
            print_lr_graph(curve_points, config.MAX_TRAIN_STEPS)
        else:
            initial_lr = config.LEARNING_RATE
        raven_params = getattr(config, 'RAVEN_PARAMS', {})
        defaults = getattr(default_config, 'RAVEN_PARAMS', {})
        final_params = {**defaults, **raven_params}

        return RavenAdamW(
            params_to_optimize,
            lr=initial_lr,
            betas=tuple(final_params.get('betas', [0.9, 0.999])),
            eps=final_params.get('eps', 1e-8),
            weight_decay=final_params.get('weight_decay', 0.01),
            debias_strength=final_params.get('debias_strength', 1.0),
            use_grad_centralization=final_params.get('use_grad_centralization', False),
            gc_alpha=final_params.get('gc_alpha', 1.0)
        )

    elif optimizer_type == "velorms":
        print("\n--- Initializing VeloRMS Optimizer ---")
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        if curve_points:
            initial_lr = max(point[1] for point in curve_points)
            print_lr_graph(curve_points, config.MAX_TRAIN_STEPS)
        else:
            initial_lr = config.LEARNING_RATE
        
        velorms_params = getattr(config, 'VELORMS_PARAMS', {})
        defaults = getattr(default_config, 'VELORMS_PARAMS', {})
        # Merge user config with defaults
        final_params = {**defaults, **velorms_params}

        return VeloRMS(
            params_to_optimize, 
            lr=initial_lr,
            momentum=final_params.get('momentum', 0.86),
            leak=final_params.get('leak', 0.16),
            weight_decay=final_params.get('weight_decay', 0.01),
            eps=final_params.get('eps', 1e-8),
            verbose=False
        )
    
    else: 
        raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'")

# ==============================================================================
#  OPTIMIZED SDXL KEY CONVERSION & SAVE LOGIC
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

def get_unet_key_mapping(current_keys):
    """
    Generates a dictionary mapping {HuggingFace_Key: SDXL_Key}.
    Does NOT handle tensors, only strings.
    """
    map_static, map_resnet, map_layer = _get_sdxl_unet_conversion_map()
    
    # 1. Initialize with current keys
    mapping = {k: k for k in current_keys}
    
    # 2. Apply static global mappings
    for sd_name, hf_name in map_static:
        if hf_name in mapping:
            mapping[hf_name] = sd_name
            
    # 3. Apply Resnet internal mappings
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
            
    # 4. Apply Block Layer mappings
    for k, v in mapping.items():
        for sd_part, hf_part in map_layer:
            if hf_part in v:
                v = v.replace(hf_part, sd_part)
        mapping[k] = v
        
    # 5. Final Formatting (Add model.diffusion_model prefix)
    final_mapping = {}
    for hf_name, sd_name in mapping.items():
        if not sd_name.startswith("model.diffusion_model."):
            final_key = f"model.diffusion_model.{sd_name}"
        else:
            final_key = sd_name
        final_mapping[hf_name] = final_key
        
    return final_mapping

def save_model(output_path, unet, base_checkpoint_path, compute_dtype):
    """
    Memory Optimized Save with Dtype Preservation:
    1. Loads Base Checkpoint.
    2. Converts ALL base tensors to training dtype (bfloat16/float16).
    3. Moves trained weights from GPU -> CPU one by one.
    4. Updates Base Checkpoint dict IN-PLACE.
    5. Saves unified checkpoint in training dtype.
    """
    from safetensors.torch import load_file, save_file
    import gc
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nINFO: Saving complete checkpoint to: {output_path}")
    print(f"   -> Target dtype: {compute_dtype}")

    # 1. Build the Key Map (Strings only - negligible RAM)
    print("   -> Generating Key Map...")
    hf_keys = list(unet.state_dict().keys())
    key_map = get_unet_key_mapping(hf_keys)

    # 2. Load Base Checkpoint to CPU
    print("   -> Loading base checkpoint dictionary...")
    try:
        base_tensors = load_file(str(base_checkpoint_path), device="cpu")
    except Exception as e:
        print(f"ERROR: Could not load base checkpoint: {e}")
        return

    # 3. **NEW: Convert ALL base tensors to training dtype**
    print(f"   -> Converting base checkpoint to {compute_dtype}...")
    converted_count = 0
    for key in base_tensors.keys():
        if base_tensors[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            base_tensors[key] = base_tensors[key].to(dtype=compute_dtype)
            converted_count += 1
    print(f"   -> Converted {converted_count} tensors to {compute_dtype}")

    # 4. In-Place Update - Copy ALL UNet layers (frozen + trainable)
    print("   -> Merging ALL UNet weights (frozen + trainable)...")
    
    update_count = 0
    unet_state = unet.state_dict()
    
    for hf_key, target_key in key_map.items():
        if hf_key in unet_state:
            # Move ALL layers (frozen or trained) to CPU with correct dtype
            # This ensures frozen layers are also saved in bfloat16
            tensor_gpu = unet_state[hf_key]
            tensor_cpu = tensor_gpu.detach().to("cpu", dtype=compute_dtype)
            
            # Inject into base checkpoint
            base_tensors[target_key] = tensor_cpu
            update_count += 1
            
            del tensor_cpu

    print(f"   -> Copied {update_count} layers (trainable + frozen in {compute_dtype})")

    # 5. Save to Disk
    print("   -> Saving to disk (SAFETENSORS)...")
    save_file(base_tensors, str(output_path))
    
    print(f"   -> Save Complete in {compute_dtype}")

    # 6. Cleanup
    del base_tensors
    del unet_state
    del key_map
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_checkpoint(global_step, micro_step, unet, base_checkpoint_path, optimizer, lr_scheduler, scaler, sampler, config):
    """
    Save checkpoint including model and training state.
    global_step = Optimizer Steps
    micro_step = Total Forward Passes
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
    # Robust save for custom optimizers vs standard
    if hasattr(optimizer, 'save_cpu_state'):
        optim_state = optimizer.save_cpu_state()
    else:
        optim_state = optimizer.state_dict()

    training_state = {
        'global_step': global_step,    # Optimizer Steps (updates)
        'micro_step': micro_step,      # Total Forward Passes (batches)
        'optimizer_state': optim_state,
        'sampler_seed': sampler.seed,
        'sampler_epoch': sampler.epoch,
        'random_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_cpu_state': torch.get_rng_state(),
        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
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
 
    global_step = 0 # Legacy name, mapped to optimizer_step in usage usually, but we use explicit names below
    micro_step = 0
    optimizer_step = 0
    
    model_to_load = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed = config.SEED
    optimizer_state = None
    initial_epoch = 0 

    if config.RESUME_TRAINING:
        print("\n" + "="*50)
        print("--- RESUMING TRAINING SESSION ---")
        state_path = Path(config.RESUME_STATE_PATH)
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        
        # Load the counters
        # If micro_step wasn't saved in old version, derive it from global_step * accumulation
        global_step_saved = training_state.get('global_step', 0)
        micro_step = training_state.get('micro_step', global_step_saved * config.GRADIENT_ACCUMULATION_STEPS)
        optimizer_step = micro_step // config.GRADIENT_ACCUMULATION_STEPS
        
        initial_sampler_seed = training_state['sampler_seed']
        initial_epoch = training_state.get('sampler_epoch', 0)
        optimizer_state = training_state['optimizer_state']
        model_to_load = Path(config.RESUME_MODEL_PATH)

        print("INFO: Restoring Random Number Generator States...")
        if 'random_state' in training_state:
            random.setstate(training_state['random_state'])
        if 'numpy_state' in training_state:
            np.random.set_state(training_state['numpy_state'])
        if 'torch_cpu_state' in training_state:
            torch.set_rng_state(training_state['torch_cpu_state'])
        if 'torch_cuda_state' in training_state and training_state['torch_cuda_state'] is not None:
            torch.cuda.set_rng_state(training_state['torch_cuda_state'])

        print(f"INFO: Resuming from Micro Step (Batch): {micro_step}")
        print(f"INFO: Corresponding Optimizer Step: {optimizer_step}")
        print(f"INFO: Resuming from Epoch: {initial_epoch}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print(f"--- STARTING {config.TRAINING_MODE.upper()} TRAINING SESSION ---")
        print("="*50 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if check_if_caching_needed(config):
        print("Loading base model components for caching...")
        vae_source = config.VAE_PATH if (config.VAE_PATH and Path(config.VAE_PATH).exists()) else config.SINGLE_FILE_CHECKPOINT_PATH
        target_channels = getattr(config, 'VAE_LATENT_CHANNELS', None)
        vae_for_caching = load_vae_robust(vae_source, device, target_channels=target_channels)

        is_32_ch = vae_for_caching.config.latent_channels == 32
        detected_shift = 0.0760 if is_32_ch else 0.0
        detected_scale = 0.6043 if is_32_ch else 0.13025
        conf_shift = getattr(config, 'VAE_SHIFT_FACTOR', None)
        conf_scale = getattr(config, 'VAE_SCALING_FACTOR', None)
        vae_for_caching.config.shift_factor = conf_shift if conf_shift is not None else detected_shift
        vae_for_caching.config.scaling_factor = conf_scale if conf_scale is not None else detected_scale
        
        vae_for_caching.enable_tiling()
        vae_for_caching.enable_slicing()

        base_pipe = StableDiffusionXLPipeline.from_single_file(
            config.SINGLE_FILE_CHECKPOINT_PATH,
            vae=vae_for_caching, 
            unet=None, 
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
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
    
    target_channels = getattr(config, 'VAE_LATENT_CHANNELS', 4)
    unet = load_unet_robust(model_to_load, config.compute_dtype, target_channels)
    
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
    
    sampler.set_epoch(initial_epoch)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS
    )

    unet.enable_gradient_checkpointing()
    unet.to(device)
    set_attention_processor(unet, config.MEMORY_EFFICIENT_ATTENTION)
    
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
    
    # LR Scheduler: initialized with MAX_TRAIN_STEPS (interpreted as Micro Steps)
    lr_scheduler = CustomCurveLRScheduler(
        optimizer=optimizer,
        curve_points=config.LR_CUSTOM_CURVE,
        total_micro_steps=config.MAX_TRAIN_STEPS
    )

    if config.RESUME_TRAINING:
        print("\n--- Restoring Training States ---")
        if optimizer_state:
            try:
                if hasattr(optimizer, 'load_cpu_state'): optimizer.load_cpu_state(optimizer_state)
                else: optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"WARNING: Optimizer load failed ({e}). Starting optimizer fresh.")

        # Sync LR scheduler
        lr_scheduler.step(micro_step)
        print("--- Resume setup complete. Starting training loop. ---")
    else:
        print("\n--- Fresh start setup complete. Starting training loop. ---")

    unet.train()
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    
    # Reporter: Total Steps = MAX_TRAIN_STEPS (interpreted as Micro Steps)
    reporter = AsyncReporter(
        total_steps=config.MAX_TRAIN_STEPS,
        test_param_name="Gradient Check"
    )
    
    timestep_sampler = TimestepSampler(config, device)
    if config.RESUME_TRAINING and micro_step > 0:
        timestep_sampler.set_current_step(micro_step)

    scheduler = None
    if config.is_rectified_flow:
        print(f"\n--- Using Loss: SDXL Rectified Flow (Velocity Matching) ---")
    else:
        prediction_type = getattr(config, "PREDICTION_TYPE", "epsilon")
        print(f"\n--- Using Loss: Standard SDXL ({prediction_type}) ---")
        scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        scheduler.config.prediction_type = prediction_type if prediction_type == "v_prediction" else "epsilon"

    accumulated_latent_paths = []
    training_start_time = time.time()
    global_step_times = deque(maxlen=50)
    optim_step_times = deque(maxlen=20)
    last_step_time = time.time()
    last_optim_step_log_time = time.time()
    done = False
    
    # Data skipping for resume
    batches_to_skip = 0
    if config.RESUME_TRAINING:
        batches_in_epoch = len(dataloader)
        batches_to_skip = micro_step % batches_in_epoch
        print(f"INFO: Skipping {batches_to_skip} batches in current epoch to align states.")

    while not done:
        for batch in dataloader:
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            # Loop Limit: Break if we reach MAX_TRAIN_STEPS (treated as Micro Steps)
            if micro_step >= config.MAX_TRAIN_STEPS:
                done = True
                break
                
            if not batch: continue

            # --- START MICRO STEP PROCESSING ---
            micro_step += 1
            
            if "latent_path" in batch: accumulated_latent_paths.extend(batch["latent_path"])
            
            latents = batch["latents"].to(device, non_blocking=True)
            embeds = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            batch_crop_coords = batch.get("crop_coords", [(0, 0)] * len(batch["latents"]))

            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):
                time_ids_list = []
                for s1, crop, s2 in zip(batch["original_sizes"], batch_crop_coords, batch["target_sizes"]):
                    time_ids_list.append(torch.tensor([s1[1], s1[0], crop[0], crop[1], s2[1], s2[0]], dtype=torch.float32).unsqueeze(0))
                time_ids = torch.cat(time_ids_list, dim=0).to(device, dtype=config.compute_dtype)

                noise = generate_train_noise(latents, config, micro_step, initial_sampler_seed)
                timesteps = timestep_sampler.sample(latents.shape[0])
                
                target, noisy_latents, timesteps_conditioning = None, None, None

                if config.is_rectified_flow:
                    t_normalized = timesteps.float() / 1000.0
                    shift_factor = getattr(config, "RF_SHIFT_FACTOR", 3.0)
                    current_shift = timestep_sampler.get_dynamic_shift(batch["target_sizes"][0][0], batch["target_sizes"][0][1]) if timestep_sampler.use_dynamic_shift else shift_factor
                    t_shifted = timestep_sampler.apply_flow_shift(t_normalized, current_shift)
                    t_expanded = t_shifted.view(-1, 1, 1, 1)
                    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
                    target = noise - latents
                    timesteps_conditioning = (t_shifted * 1000).long().clamp(0, 999)
                else:
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    target = scheduler.get_velocity(latents, noise, timesteps) if scheduler.config.prediction_type == "v_prediction" else noise
                    timesteps_conditioning = timesteps

                pred = unet(noisy_latents, timesteps_conditioning, embeds, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample

            loss_type = getattr(config, "LOSS_TYPE", "MSE")
            if loss_type == "Semantic":
                per_pixel_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
                semantic_maps = batch.get("semantic_map")
                if semantic_maps is not None:
                    loss = (per_pixel_loss * (1.0 + semantic_maps.to(device, dtype=torch.float32))).mean()
                else:
                    loss = per_pixel_loss.mean()
            else:
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

            raw_loss_value = loss.detach().item()
            scaled_loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()
    
            diagnostics.step(raw_loss_value)
            
            # Update LR every micro step
            lr_scheduler.step(micro_step)
            
            # --- ACCUMULATION & OPTIMIZER STEP ---
            is_accumulation_step = (micro_step % config.GRADIENT_ACCUMULATION_STEPS == 0)
            diag_data_to_log = None
            
            if is_accumulation_step:
                if isinstance(optimizer, TitanAdamW): raw_grad_norm = optimizer.clip_grad_norm(config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf'))
                else: raw_grad_norm = torch.nn.utils.clip_grad_norm_(params_to_optimize, config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf'))
                if isinstance(raw_grad_norm, torch.Tensor): raw_grad_norm = raw_grad_norm.item()
                clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm
                timestep_sampler.update(raw_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                optimizer_step += 1
                
                optim_step_time = time.time() - last_optim_step_log_time
                optim_step_times.append(optim_step_time)
                last_optim_step_log_time = time.time()

                diag_data_to_log = {
                    'optim_step': optimizer_step,
                    'avg_loss': diagnostics.get_average_loss(),
                    'current_lr': optimizer.param_groups[0]['lr'],
                    'raw_grad_norm': raw_grad_norm,
                    'clipped_grad_norm': clipped_grad_norm,
                    'update_delta': 1.0 if raw_grad_norm > 0 else 0.0,
                    'optim_step_time': optim_step_time,
                    'avg_optim_step_time': sum(optim_step_times) / len(optim_step_times)
                }
                
                reporter.check_and_report_anomaly(optimizer_step, raw_grad_norm, clipped_grad_norm, config, accumulated_latent_paths)
                diagnostics.reset()
                accumulated_latent_paths.clear()

                # Save Checkpoint based on Optimizer Step
                if config.SAVE_EVERY_N_STEPS > 0 and optimizer_step > 0 and (optimizer_step % config.SAVE_EVERY_N_STEPS == 0):
                    print(f"\n--- Saving checkpoint at optimizer step {optimizer_step} (Micro Step {micro_step}) ---")
                    save_checkpoint(optimizer_step, micro_step, unet, model_to_load, optimizer, lr_scheduler, None, sampler, config)

            # --- LOGGING UPDATES EVERY MICRO STEP ---
            step_duration = time.time() - last_step_time
            global_step_times.append(step_duration)
            last_step_time = time.time()
            elapsed_time = time.time() - training_start_time
            eta_seconds = (config.MAX_TRAIN_STEPS - micro_step) * (sum(global_step_times) / len(global_step_times)) if global_step_times else 0
            
            timing_data = {
                'raw_step_time': step_duration,
                'elapsed_time': elapsed_time,
                'eta': eta_seconds,
                'loss': raw_loss_value,
                'timestep': timesteps_conditioning[0].item() if timesteps_conditioning is not None else 0
            }
            
            # Pass micro_step to reporter so it updates 1/250, 2/250, etc.
            reporter.log_step(micro_step, timing_data=timing_data, diag_data=diag_data_to_log)

    print("\nTraining complete.")
    reporter.shutdown()
    output_path = OUTPUT_DIR / f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_unified_{str(uuid.uuid4())[:4]}.safetensors"
    save_model(output_path, unet, model_to_load, config.compute_dtype)
    print("All tasks complete. Final model saved.")

if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()