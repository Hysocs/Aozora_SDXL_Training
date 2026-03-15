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
from optimizer.raven import RavenAdamW
from optimizer.titan import TitanAdamW
from optimizer.velorms import VeloRMS
import uuid
from diffusers.models.attention_processor import (
        AttnProcessor2_0, 
        FusedAttnProcessor2_0
    )
from tools.semantic import generate_latent_semantic_map_batch

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
        self.REF_MODE = getattr(self, "REF_MODE", "Full") # "Full"

    def _load_from_user_config(self):
        parser = argparse.ArgumentParser(description="Load a specific training configuration.")
        parser.add_argument("--config", type=str, help="Path to the user configuration JSON file.")
        args, _ = parser.parse_known_args()
        if args.config:
            path = Path(args.config)
            if path.exists():
                print(f"INFO: Loading configuration from {path}")
                try:
                    with open(path, 'r', encoding='utf-8') as f:
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
            if key in["RESUME_MODEL_PATH", "RESUME_STATE_PATH"] and getattr(self, "RESUME_TRAINING", False):
                if not value or not Path(value).exists():
                    raise FileNotFoundError(f"RESUME_TRAINING is enabled, but {key}='{value}' is not a valid file path.")
            if key == "UNET_EXCLUDE_TARGETS":
                if isinstance(value, str):
                    setattr(self, key,[item.strip() for item in value.split(',') if item.strip()])
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
            self.curve_points.insert(0,[0.0, self.curve_points[0][1]])
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
            scale = param_group.get('lr_scale', 1.0)
            param_group['lr'] = lr * scale

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
        progress_bar = f'Training |{bar}| {current_step + 1}/{self.total_steps}[{percentage:.2%}]'
        step_info = f"[Loss: {loss_val:.4f}, Timestep: {timestep_val}]"
        timing_info = f"[{s_per_step:.2f}s/step, ETA: {eta_str}, Elapsed: {time_spent_str}]"
        return progress_bar + step_info + timing_info

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_type, data = self.task_queue.get(timeout=1)
                if task_type == 'log_step': self._handle_log_step(**data)
                elif task_type == 'anomaly': self._handle_anomaly(**data)
                self.task_queue.task_done()
            except queue.Empty: continue
            
    def _format_metric(self, value, threshold=100.0):
        if abs(value) > threshold:
            return f"[HIGH]"
        return f"{value:.4f}"
    
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
                f"  Grad Norm (Raw/Clipped): {self._format_metric(diag_data['raw_grad_norm'])} / {self._format_metric(diag_data['clipped_grad_norm'])}\n"
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
            batches =[]
            for key in buckets:
                bucket_indices = buckets[key]
                for i in range(0, len(bucket_indices), self.batch_size):
                    batch = bucket_indices[i : i + self.batch_size]
                    batches.append(batch)
            if self.shuffle:
                batch_indices = torch.randperm(len(batches), generator=g).tolist()
                batches =[batches[i] for i in batch_indices]
        self.epoch += 1
        yield from batches

    def __len__(self):
        return (self.total_images + self.batch_size - 1) // self.batch_size
    
class ResolutionCalculator:
    def __init__(self, target_area, stride=64, should_upscale=True, max_area_tolerance=1.0):
        self.target_area = target_area
        self.stride = stride
        self.should_upscale = should_upscale

    def calculate_resolution(self, width, height):
        current_area = width * height
        if not self.should_upscale and current_area <= self.target_area:
            return self._quantize_to_stride(width, height)
        exact_scale = math.sqrt(self.target_area / current_area)
        scaled_w = width * exact_scale
        scaled_h = height * exact_scale
        final_w, final_h = self._quantize_to_stride(scaled_w, scaled_h)
        final_area = final_w * final_h
        if final_area > self.target_area:
            adjustment = math.sqrt(self.target_area / final_area) * 0.999 
            final_w, final_h = self._quantize_to_stride(scaled_w * adjustment, scaled_h * adjustment)
        return final_w, final_h
    
    def _quantize_to_stride(self, w, h):
        return int(round(w / self.stride) * self.stride), int(round(h / self.stride) * self.stride)

def smart_resize(image, target_w, target_h):
    return image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

def validate_and_assign_resolution(args):
    ip, target_area, stride, should_upscale = args 
    try:
        with Image.open(ip) as img: 
            img.verify()
        with Image.open(ip) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0: 
                return None
        current_area = w * h
        if not should_upscale and current_area < target_area:
            target_w = int(w // stride) * stride
            target_h = int(h // stride) * stride
            target_w, target_h = max(target_w, stride), max(target_h, stride)
        else:
            scale = math.sqrt(target_area / current_area)
            scaled_w = w * scale
            scaled_h = h * scale
            target_w = int(round(scaled_w / stride) * stride)
            target_h = int(round(scaled_h / stride) * stride)
            target_w, target_h = max(target_w, stride), max(target_h, stride)
        
        cp = ip.with_suffix('.txt')
        if cp.exists():
            with open(cp, 'r', encoding='utf-8', errors='ignore') as f: 
                caption = f.read().strip()
            if not caption: 
                caption = ip.stem.replace('_', ' ')
        else: 
            caption = ip.stem.replace('_', ' ')
        
        return {
            "ip": ip, 
            "caption": caption, 
            "target_resolution": (target_w, target_h),
            "original_size": (w, h),
            "original_area": w * h,
            "target_area": target_w * target_h
        }
    except Exception as e:
        print(f"\n[CORRUPT IMAGE OR READ ERROR] Skipping {ip}, Reason: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_text_embeddings_sdxl(captions, t1, t2, te1, te2, device):
    prompt_embeds_list =[]
    pooled_prompt_embeds_list =[]
    
    for caption in captions:
        with torch.no_grad():
            tokens_1 = t1(
                caption, padding="max_length", max_length=t1.model_max_length,
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            
            tokens_2 = t2(
                caption, padding="max_length", max_length=t2.model_max_length,
                truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            
            enc_1_output = te1(tokens_1, output_hidden_states=True)
            prompt_embeds_1 = enc_1_output.hidden_states[-2]  
            
            enc_2_output = te2(tokens_2, output_hidden_states=True)
            prompt_embeds_2 = enc_2_output.hidden_states[-2]  
            
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  
            pooled_prompt_embeds = enc_2_output[0]  
            
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    
    prompt_embeds = torch.cat(prompt_embeds_list, dim=0)  
    pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=0)  
    return prompt_embeds, pooled_prompt_embeds

def check_if_caching_needed(config):
    needs_caching = False
    cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    semantic_cache_folder_name = ".semantic_maps_cache"
    use_semantic_loss = getattr(config, 'LOSS_TYPE', 'MSE') == "Semantic"
    
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
            
        image_paths = [p for ext in['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        if not image_paths:
            print(f"INFO: No images found in {root}, skipping.")
            continue
            
        cache_dir = root / cache_folder_name
        if not cache_dir.exists():
            print(f"INFO: Cache directory doesn't exist for {root}, caching needed.")
            needs_caching = True
            continue
        
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
            if use_semantic_loss:
                semantic_cache_dir = root / semantic_cache_folder_name
                missing_semantic = 0
                for te_file in cached_te_files:
                    safe_name = te_file.stem.replace('_te', '')
                    sem_path = semantic_cache_dir / f"{safe_name}_sem.pt"
                    if not sem_path.exists(): missing_semantic += 1
                
                if missing_semantic > 0:
                    print(f"INFO: {missing_semantic} semantic maps missing in {root}")
                    needs_caching = True
                else: print(f"INFO: All {len(image_paths)} images and semantic maps cached in {root}")
            else:
                print(f"INFO: All {len(image_paths)} images appear to be cached in {root}")
            
    return needs_caching

def load_unet_robust(path, compute_dtype):
    print(f"INFO: Loading UNet from: {Path(path).name}")
    try:
        from safetensors.torch import load_file
        sd = load_file(str(path), device="cpu")
        detected_channels = None
        for k in["conv_in.weight", "model.diffusion_model.input_blocks.0.0.weight"]:
            if k in sd:
                detected_channels = sd[k].shape[1]
                break
        del sd
        if detected_channels is None: detected_channels = 4
    except Exception:
        detected_channels = 4

    print(f"INFO: Detected UNet input channels: {detected_channels}")

    load_kwargs = {
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": False,
    }

    if detected_channels != 4:
        load_kwargs["in_channels"] = detected_channels
        load_kwargs["ignore_mismatched_sizes"] = True

    unet = UNet2DConditionModel.from_single_file(str(path), **load_kwargs)
    print(f"INFO: Loaded UNet (in_channels={unet.config.in_channels})")
    return unet

def load_vae_robust(path, device, target_channels=None):
    print(f"INFO: Attempting to load VAE from: {path}")
    if target_channels is None:
        try:
            from safetensors.torch import load_file
            tensors = load_file(path, device="cpu")
            for k in["first_stage_model.quant_conv.weight", "quant_conv.weight"]:
                if k in tensors:
                    target_channels = tensors[k].shape[0] // 2
                    break
        except Exception: pass
            
    target_channels = target_channels or 4
    try:
        if target_channels != 4:
            print(f"INFO: Loading VAE with {target_channels} channels...")
            vae = AutoencoderKL.from_single_file(
                path, torch_dtype=torch.float32, latent_channels=target_channels,
                ignore_mismatched_sizes=True, low_cpu_mem_usage=False
            )
        else:
            vae = AutoencoderKL.from_single_file(
                path, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
        print(f"INFO: Successfully loaded VAE with {target_channels} channels.")
        return vae.to(device)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load VAE with {target_channels}-channel config: {e}")
        raise e

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config):
        print("\n" + "="*60)
        print("INFO: Datasets already cached.")
        print("="*60 + "\n")
        return
    
    cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    use_semantic_loss = getattr(config, 'LOSS_TYPE', 'MSE') == "Semantic"
    semantic_cache_folder_name = ".semantic_maps_cache"
    
    print("\n" + "="*60)
    print(f"STARTING VRAM-OPTIMIZED CACHING FOR: {config.TRAINING_MODE}")
    print(f"Using cache folder: {cache_folder_name}")
    if use_semantic_loss: print(f"Semantic maps will be cached to: {semantic_cache_folder_name}")
    print(f"VAE Channels: {vae.config.latent_channels}")
    print("="*60 + "\n")
    
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
        
        semantic_cache_dir = root / semantic_cache_folder_name if use_semantic_loss else None
        if use_semantic_loss: semantic_cache_dir.mkdir(exist_ok=True)
        
        torch.save({"embeds": null_embeds, "pooled": null_pooled}, cache_dir / "null_embeds.pt")
        
        paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        
        to_process =[]
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
                pool.imap(validate_and_assign_resolution,[(p, config.TARGET_PIXEL_AREA, 64, config.SHOULD_UPSCALE) for p in to_process]), 
                total=len(to_process)
            ))
        metadata = [r for r in results if r]
        
        grouped = defaultdict(list)
        for m in metadata: grouped[m["target_resolution"]].append(m)
        
        batches =[]
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
            valid_meta_final =[]
            original_images_for_semantic =[] if use_semantic_loss else None
            
            for m in batch_meta:
                try:
                    with Image.open(m['ip']) as img:
                        img = fix_alpha_channel(img)
                        img_resized = smart_resize(img, w, h)
                        images_global.append(transform(img_resized))
                        valid_meta_final.append(m)
                        if use_semantic_loss: original_images_for_semantic.append(img_resized.copy())
                except Exception as e: print(f"Skipping {m['ip']}: {e}")
            
            if not images_global: continue
            pixel_values_global = torch.stack(images_global).to(device, dtype=torch.float32)

            with torch.no_grad():
                dist = vae.encode(pixel_values_global).latent_dist
                latents_global = dist.mean
                if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                    latents_global = (latents_global - vae.config.shift_factor) * vae.config.scaling_factor
                else:
                    latents_global = latents_global * vae.config.scaling_factor
                
            latents_global = latents_global.cpu()
            
            semantic_maps = None
            if use_semantic_loss and original_images_for_semantic:
                sep_weight = getattr(config, "SEMANTIC_SEP_WEIGHT", 1.0)
                detail_weight = getattr(config, "SEMANTIC_DETAIL_WEIGHT", 1.0)
                entropy_weight = getattr(config, "SEMANTIC_ENTROPY_WEIGHT", 0.0)
                
                semantic_maps = generate_latent_semantic_map_batch(
                    latents=latents_global, images=original_images_for_semantic,
                    sep_weight=sep_weight, detail_weight=detail_weight,
                    entropy_weight=entropy_weight, device="cpu", dtype=torch.float32
                )
                semantic_maps = semantic_maps.cpu()
            
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
                if semantic_maps is not None: torch.save(semantic_maps[j], path_sem)
            
            del pixel_values_global, latents_global, embeds, pooled
            if semantic_maps is not None: del semantic_maps
            if original_images_for_semantic: del original_images_for_semantic
            if batch_idx % 20 == 0: torch.cuda.empty_cache()

    print("\nINFO: Caching Complete. Cleaning up...")
    te1.cpu()
    te2.cpu()
    gc.collect()
    torch.cuda.empty_cache()

class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.te_files =[]
        self.dataset_roots = {} 
        self.is_rectified_flow = config.is_rectified_flow
        self.seed = config.SEED if config.SEED else 42
        self.worker_rng = None
        
        cache_folder_name = ".precomputed_embeddings_cache_rf_noobai" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
        semantic_cache_folder_name = ".semantic_maps_cache"

        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / cache_folder_name
            if not cache_dir.exists(): continue
            files = list(cache_dir.glob("*_te.pt"))
            for f in files: self.dataset_roots[str(f)] = root
            self.te_files.extend(files * int(ds.get("repeats", 1)))

        if not self.te_files: raise ValueError("No cached files found.")
        random.shuffle(self.te_files)
        self.bucket_keys =[]
        
        for f in tqdm(self.te_files, desc="Loading Cache Headers"):
            try:
                header = torch.load(f, map_location='cpu')
                res = header.get('target_size')
                if res: self.bucket_keys.append(tuple(res))
                else: self.bucket_keys.append(None)
            except Exception as e:
                print(f"Error reading cache header for {f}: {e}")
                self.bucket_keys.append(None)
        
        self.dropout_prob = getattr(config, "UNCONDITIONAL_DROPOUT_CHANCE", 0.0) if getattr(config, "UNCONDITIONAL_DROPOUT", False) else 0.0
        self.null_embeds = None
        self.null_pooled = None
        self.target_shift = getattr(config, 'VAE_SHIFT_FACTOR', None)
        self.target_scale = getattr(config, 'VAE_SCALING_FACTOR', None)
        self.use_semantic_loss = (getattr(config, 'LOSS_TYPE', 'MSE') == "Semantic")
        self.semantic_cache_roots = {}  
        
        if self.use_semantic_loss:
            print("INFO: Semantic loss is ENABLED. Loading from cached semantic maps.")
            for ds in config.INSTANCE_DATASETS:
                root = Path(ds["path"])
                semantic_cache_dir = root / semantic_cache_folder_name
                cache_dir = root / cache_folder_name
                if not semantic_cache_dir.exists():
                    print(f"WARNING: Semantic cache directory not found: {semantic_cache_dir}")
                    continue
                for f in cache_dir.glob("*_te.pt"):
                    self.semantic_cache_roots[str(f)] = semantic_cache_dir

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
                        if self.null_embeds.dim() == 3: self.null_embeds = self.null_embeds.squeeze(0)
                        self.null_pooled = null_data["pooled"]
                        if self.null_pooled.dim() == 2: self.null_pooled = self.null_pooled.squeeze(0)
                        found_null = True
                        break
                    except Exception as e:
                        print(f"WARNING: Failed to load null embeddings: {e}")
            
            if not found_null:
                print("WARNING: Unconditional Dropout enabled but null_embeds.pt not found. Disabling dropout.")
                self.dropout_prob = 0.0
                
        print(f"INFO: Dataset initialized with {len(self.te_files)} samples.")

    def _init_worker_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: self.worker_rng = random.Random(self.seed)
        else: self.worker_rng = random.Random(self.seed + worker_info.id)

    def __len__(self): return len(self.te_files)

    def __getitem__(self, i):
        try:
            if self.worker_rng is None: self._init_worker_rng()
            path_te = self.te_files[i]
            data_te = torch.load(path_te, map_location="cpu")
            path_str = str(path_te)
            path_lat = Path(path_str.replace("_te.pt", "_lat.pt"))
            latents = torch.load(path_lat, map_location="cpu")
            if torch.isnan(latents).any() or torch.isinf(latents).any(): return None

            item_data = {
                "latents": latents,
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

            if self.use_semantic_loss:
                semantic_dir = self.semantic_cache_roots.get(str(path_te))
                if semantic_dir:
                    safe_name = Path(path_te).stem.replace('_te', '')
                    sem_path = semantic_dir / f"{safe_name}_sem.pt"
                    if sem_path.exists():
                        item_data["semantic_map"] = torch.load(sem_path, map_location="cpu")
                    else:
                        c, h, w = latents.shape
                        item_data["semantic_map"] = torch.ones((c, h, w), dtype=torch.float32)
                else:
                    c, h, w = latents.shape
                    item_data["semantic_map"] = torch.ones((c, h, w), dtype=torch.float32)

            return item_data
        except Exception as e:
            print(f"WARNING: Skipping {self.te_files[i]}: {e}")
            return None
        
class TimestepSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.total_tickets_needed = config.MAX_TRAIN_STEPS * config.BATCH_SIZE
        self.seed = config.SEED if config.SEED else 42
        self.is_rectified_flow = getattr(config, "is_rectified_flow", False)
        self.shift_factor = getattr(config, "RF_SHIFT_FACTOR", 3.0)
        self.use_dynamic_shift = getattr(config, "RF_USE_DYNAMIC_SHIFT", False)
        self.base_pixels = getattr(config, "RF_BASE_PIXELS", 1048576)
        
        allocation = getattr(config, 'TIMESTEP_ALLOCATION', None)
        self.ticket_pool = self._build_ticket_pool(allocation)
        self.pool_index = 0

    def _build_ticket_pool(self, allocation):
        use_fallback = False
        if not allocation or "counts" not in allocation or "bin_size" not in allocation: use_fallback = True
        elif sum(allocation["counts"]) == 0: use_fallback = True

        if use_fallback:
            self.bin_size = 100 
            num_bins = 1000 // self.bin_size
            base = self.total_tickets_needed // num_bins // self.config.BATCH_SIZE
            counts = [base] * num_bins
            for i in range((self.total_tickets_needed // self.config.BATCH_SIZE) % num_bins): counts[i] += 1
        else:
            self.bin_size = allocation["bin_size"] 
            counts = allocation["counts"]

        multiplier = self.config.BATCH_SIZE
        total_counts = sum(counts)
        scale_factor = (self.total_tickets_needed / multiplier) / total_counts if total_counts > 0 else 1.0
        
        pool =[]
        rng = np.random.Generator(np.random.PCG64(self.seed))
        
        for i, count in enumerate(counts):
            if count <= 0: continue
            start_t = i * self.bin_size
            end_t = min(1000, (i + 1) * self.bin_size)
            if start_t >= 1000: break
            num_tickets = int(count * scale_factor * multiplier)
            if num_tickets <= 0: num_tickets = 1
            bin_samples = rng.integers(start_t, end_t, size=num_tickets).tolist()
            pool.extend(bin_samples)
            
        random.seed(self.seed)
        random.shuffle(pool)
        if len(pool) < self.total_tickets_needed:
            while len(pool) < self.total_tickets_needed: pool.extend(pool[:self.total_tickets_needed - len(pool)])
        elif len(pool) > self.total_tickets_needed:
            pool = pool[:self.total_tickets_needed]
        return pool

    def get_dynamic_shift(self, height, width):
        if not self.use_dynamic_shift: return self.shift_factor
        current_pixels = height * width
        ratio = current_pixels / self.base_pixels
        return self.shift_factor * math.sqrt(ratio)

    def apply_flow_shift(self, t_normalized, shift_factor):
        if shift_factor == 1.0 or shift_factor is None: return t_normalized
        return (shift_factor * t_normalized) / (1.0 + (shift_factor - 1.0) * t_normalized)

    def set_current_step(self, micro_step):
        consumed_tickets = micro_step * self.config.BATCH_SIZE
        self.pool_index = consumed_tickets % len(self.ticket_pool)

    def sample(self, batch_size):
        indices =[]
        for _ in range(batch_size):
            if self.pool_index >= len(self.ticket_pool): self.pool_index = 0 
            indices.append(self.ticket_pool[self.pool_index])
            self.pool_index += 1
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def update(self, raw_grad_norm): pass

def custom_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    output = {}
    for k in batch[0]:
        if k == "original_image": output[k] = [item[k] for item in batch]
        elif isinstance(batch[0][k], torch.Tensor): output[k] = torch.stack([item[k] for item in batch])
        else: output[k] = [item[k] for item in batch]
    return output

def create_optimizer(config, params_to_optimize):
    optimizer_type = config.OPTIMIZER_TYPE.lower()

    if optimizer_type == "titan":
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        initial_lr = max(point[1] for point in curve_points) if curve_points else config.LEARNING_RATE
        titan_params = getattr(config, 'TITAN_PARAMS', {})
        defaults = default_config.TITAN_PARAMS
        final_params = {**defaults, **titan_params}
        return TitanAdamW(
            params_to_optimize, lr=initial_lr, betas=tuple(final_params.get('betas',[0.9, 0.999])),
            eps=final_params.get('eps', 1e-8), weight_decay=final_params.get('weight_decay', 0.01),
            debias_strength=final_params.get('debias_strength', 1.0),
            use_grad_centralization=final_params.get('use_grad_centralization', False),
            gc_alpha=final_params.get('gc_alpha', 1.0)
        )
    elif optimizer_type == "raven":
        curve_points = getattr(config, 'LR_CUSTOM_CURVE',[])
        initial_lr = max(point[1] for point in curve_points) if curve_points else config.LEARNING_RATE
        raven_params = getattr(config, 'RAVEN_PARAMS', {})
        defaults = getattr(default_config, 'RAVEN_PARAMS', {})
        final_params = {**defaults, **raven_params}
        return RavenAdamW(
            params_to_optimize, lr=initial_lr, betas=tuple(final_params.get('betas',[0.9, 0.999])),
            eps=final_params.get('eps', 1e-8), weight_decay=final_params.get('weight_decay', 0.01),
            debias_strength=final_params.get('debias_strength', 1.0),
            use_grad_centralization=final_params.get('use_grad_centralization', False),
            gc_alpha=final_params.get('gc_alpha', 1.0)
        )
    elif optimizer_type == "velorms":
        curve_points = getattr(config, 'LR_CUSTOM_CURVE',[])
        initial_lr = max(point[1] for point in curve_points) if curve_points else config.LEARNING_RATE
        velorms_params = getattr(config, 'VELORMS_PARAMS', {})
        defaults = getattr(default_config, 'VELORMS_PARAMS', {})
        final_params = {**defaults, **velorms_params}
        return VeloRMS(
            params_to_optimize, lr=initial_lr, momentum=final_params.get('momentum', 0.86),
            leak=final_params.get('leak', 0.16), weight_decay=final_params.get('weight_decay', 0.01),
            eps=final_params.get('eps', 1e-8), verbose=False
        )
    else: 
        raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'")

def _get_sdxl_unet_conversion_map():
    unet_conversion_map =[
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
        ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
        ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
        ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
        ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ]
    unet_conversion_map_resnet =[
        ("in_layers.0", "norm1"), ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"), ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"), ("skip_connection", "conv_shortcut"),
    ]
    unet_conversion_map_layer =[]
    
    for i in range(3): 
        for j in range(2): 
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))
            if i > 0: 
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))
        for j in range(3): 
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))
            if i < 2: 
                hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
                unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))
        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))
            
    unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))
    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
    
    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    return unet_conversion_map, unet_conversion_map_resnet, unet_conversion_map_layer

def get_unet_key_mapping(current_keys):
    map_static, map_resnet, map_layer = _get_sdxl_unet_conversion_map()
    mapping = {k: k for k in current_keys}
    
    for sd_name, hf_name in map_static:
        if hf_name in mapping: mapping[hf_name] = sd_name
            
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in map_resnet: v = v.replace(hf_part, sd_part)
            mapping[k] = v
            
    for k, v in mapping.items():
        for sd_part, hf_part in map_layer:
            if hf_part in v: v = v.replace(hf_part, sd_part)
        mapping[k] = v
        
    final_mapping = {}
    for hf_name, sd_name in mapping.items():
        if not sd_name.startswith("model.diffusion_model."): final_key = f"model.diffusion_model.{sd_name}"
        else: final_key = sd_name
        final_mapping[hf_name] = final_key
        
    return final_mapping

def save_model(output_path, unet, base_checkpoint_path, compute_dtype):
    from safetensors.torch import load_file, save_file
    import gc
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    hf_keys = list(unet.state_dict().keys())
    key_map = get_unet_key_mapping(hf_keys)

    try: base_tensors = load_file(str(base_checkpoint_path), device="cpu")
    except Exception as e:
        print(f"ERROR: Could not load base checkpoint: {e}")
        return

    unet_state = unet.state_dict()
    for key in base_tensors.keys():
        if base_tensors[key].dtype in[torch.float32, torch.float16, torch.bfloat16]:
            base_tensors[key] = base_tensors[key].to(dtype=compute_dtype)

    for hf_key, target_key in key_map.items():
        if hf_key in unet_state:
            tensor_cpu = unet_state[hf_key].detach().to("cpu", dtype=compute_dtype)
            base_tensors[target_key] = tensor_cpu
            del tensor_cpu

    save_file(base_tensors, str(output_path))
    del base_tensors
    del unet_state
    del key_map
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def save_checkpoint_pt(global_step, micro_step, unet, base_checkpoint_path, optimizer, lr_scheduler, scaler, sampler, config):
    output_dir = Path(config.OUTPUT_DIR)
    model_filename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    
    save_model(output_dir / model_filename, unet, base_checkpoint_path, config.compute_dtype)
    
    if hasattr(optimizer, 'save_cpu_state'): optim_state = optimizer.save_cpu_state()
    else: optim_state = optimizer.state_dict()

    training_state = {
        'global_step': global_step,    
        'micro_step': micro_step,      
        'optimizer_state': optim_state,
        'sampler_seed': sampler.seed,
        'sampler_epoch': sampler.epoch,
        'random_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_cpu_state': torch.get_rng_state(),
        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    
    if scaler is not None: training_state['scaler_state_dict'] = scaler.state_dict()
    torch.save(training_state, output_dir / state_filename)
    print(f"Successfully saved training state: {state_filename}")

def zero_grad_channels(conv_in_weight, channel_slice):
    """
    Hook to zero out gradients for specific input channel groups
    so the loss cannot flow back through those conv_in weights.
    Slice example: slice(0, 4) zeros grad for ch 0-3 input weights.
    """
    def hook(grad):
        grad = grad.clone()
        grad[:, channel_slice, :, :] = 0.0
        return grad
        
    return conv_in_weight.register_hook(hook)


def main():
    config = TrainingConfig()

    if config.SEED:
        set_seed(config.SEED)

    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    micro_step     = 0
    optimizer_step = 0

    model_to_load        = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed = config.SEED
    optimizer_state      = None
    initial_epoch        = 0

    if config.RESUME_TRAINING:
        print("\n" + "="*50)
        print("--- RESUMING TRAINING SESSION ---")
        state_path     = Path(config.RESUME_STATE_PATH)
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)

        global_step_saved    = training_state.get('global_step', 0)
        micro_step           = training_state.get('micro_step', global_step_saved * config.GRADIENT_ACCUMULATION_STEPS)
        optimizer_step       = micro_step // config.GRADIENT_ACCUMULATION_STEPS
        initial_sampler_seed = training_state['sampler_seed']
        initial_epoch        = training_state.get('sampler_epoch', 0)
        optimizer_state      = training_state['optimizer_state']
        model_to_load        = Path(config.RESUME_MODEL_PATH)

        print("INFO: Restoring Random Number Generator States...")
        if 'random_state'     in training_state: random.setstate(training_state['random_state'])
        if 'numpy_state'      in training_state: np.random.set_state(training_state['numpy_state'])
        if 'torch_cpu_state'  in training_state: torch.set_rng_state(training_state['torch_cpu_state'])
        if 'torch_cuda_state' in training_state and training_state['torch_cuda_state'] is not None:
            torch.cuda.set_rng_state(training_state['torch_cuda_state'])

        print(f"INFO: Resuming from Micro Step: {micro_step} | Optimizer Step: {optimizer_step} | Epoch: {initial_epoch}")
        print("="*50 + "\n")
    else:
        print("\n" + "="*50)
        print(f"--- STARTING {config.TRAINING_MODE.upper()} TRAINING SESSION ---")
        print("="*50 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if check_if_caching_needed(config):
        print("Loading base model components for caching...")

        target_channels = getattr(config, 'VAE_LATENT_CHANNELS', 4)
        conf_shift      = getattr(config, 'VAE_SHIFT_FACTOR', None)
        conf_scale      = getattr(config, 'VAE_SCALING_FACTOR', None)
        vae_source      = config.VAE_PATH if (config.VAE_PATH and Path(config.VAE_PATH).exists()) else config.SINGLE_FILE_CHECKPOINT_PATH
        vae_for_caching = load_vae_robust(vae_source, device, target_channels=target_channels)

        file_shift = getattr(vae_for_caching.config, 'shift_factor',   None)
        file_scale = getattr(vae_for_caching.config, 'scaling_factor', None)
        vae_for_caching.config.shift_factor   = conf_shift if conf_shift is not None else file_shift
        vae_for_caching.config.scaling_factor = conf_scale if conf_scale is not None else file_scale
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
            base_pipe.tokenizer, base_pipe.tokenizer_2,
            base_pipe.text_encoder, base_pipe.text_encoder_2,
            vae_for_caching, device
        )
        del base_pipe, vae_for_caching
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n--- Loading Model ---")
    print(f"INFO: Loading UNet from: {model_to_load.name}")
    unet = load_unet_robust(model_to_load, config.compute_dtype)

    if unet.config.in_channels != 12:
        print(f"WARNING: UNet has {unet.config.in_channels} channels. Expected 12.")

    print(f"\nINFO: 12-channel UNet Architecture")
    print("INFO: Ch 0-3  = Noisy latents at current t")
    print("INFO: Ch 4-7  = Blind x0 guess at t_curr  — 'what is fully clean?'")
    print("INFO: Ch 8-11 = Blind x0 guess at t_mid   — 'what is halfway to clean?'")
    print("INFO: Blind pass outputs decoded OUTSIDE no_grad — full gradient to conv_in")
    print("INFO: Loss terms use live tensors — direct gradient to anchor channels")

    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Initializing Dataset ---")
    dataset = ImageTextLatentDataset(config)
    sampler = BucketBatchSampler(dataset, config.BATCH_SIZE, initial_sampler_seed, shuffle=True)
    sampler.set_epoch(initial_epoch)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    unet.enable_gradient_checkpointing()
    unet.to(device)
    set_attention_processor(unet, getattr(config, "MEMORY_EFFICIENT_ATTENTION", "sdpa"))

    print("\n--- UNet Layer Selection ---")
    exclusion_keywords = getattr(config, "UNET_EXCLUDE_TARGETS", [])
    for name, param in unet.named_parameters():
        should_exclude = any(fnmatch.fnmatch(name, kw if '*' in kw else f"*{kw}*") for kw in exclusion_keywords)
        param.requires_grad = not should_exclude

    total_params     = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.2f}%) | Frozen: {(total_params-trainable_params)/1e6:.2f}M")

    conv_in_params        = list(unet.conv_in.parameters())
    conv_in_ids           = {id(p) for p in conv_in_params}
    other_params          = [p for p in unet.parameters() if p.requires_grad and id(p) not in conv_in_ids]
    conv_in_lr_multiplier = getattr(config, "CONV_IN_LR_MULTIPLIER", 5)
    curve_points          = getattr(config, 'LR_CUSTOM_CURVE', [])
    base_lr               = max(p[1] for p in curve_points) if curve_points else 1e-5
    conv_in_lr            = base_lr * conv_in_lr_multiplier

    params_to_optimize = [
        {"params": conv_in_params, "lr": conv_in_lr, "lr_scale": conv_in_lr_multiplier},
        {"params": other_params,   "lr_scale": 1.0},
    ]
    print(f"\n  conv_in: {conv_in_lr_multiplier}x LR boost | Base: {base_lr:.2e} | conv_in: {conv_in_lr:.2e}")

    optimizer    = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(optimizer=optimizer, curve_points=config.LR_CUSTOM_CURVE, total_micro_steps=config.MAX_TRAIN_STEPS)

    if config.RESUME_TRAINING:
        print("\n--- Restoring Training States ---")
        if optimizer_state:
            try:
                if hasattr(optimizer, 'load_cpu_state'): optimizer.load_cpu_state(optimizer_state)
                else: optimizer.load_state_dict(optimizer_state)
            except Exception as e:
                print(f"WARNING: Optimizer load failed ({e}). Starting fresh.")
        lr_scheduler.step(micro_step)
        print("--- Resume complete. Starting training. ---")
    else:
        print("\n--- Fresh start. Starting training. ---")

    unet.train()
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    reporter    = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="Gradient Check")

    timestep_sampler = TimestepSampler(config, device)
    if config.RESUME_TRAINING and micro_step > 0:
        timestep_sampler.set_current_step(micro_step)

    scheduler = None
    if config.is_rectified_flow:
        print(f"\n--- Loss: SDXL Rectified Flow (Velocity Matching) ---")
    else:
        prediction_type = getattr(config, "PREDICTION_TYPE", "epsilon")
        print(f"\n--- Loss: Standard SDXL ({prediction_type}) ---")
        scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        scheduler.config.prediction_type = prediction_type if prediction_type == "v_prediction" else "epsilon"

    loss_type = getattr(config, "LOSS_TYPE", "MSE")
    print(f"--- Loss Type: {loss_type} ---\n")

    accumulated_latent_paths = []
    training_start_time      = time.time()
    global_step_times        = deque(maxlen=50)
    optim_step_times         = deque(maxlen=20)
    last_step_time           = time.time()
    last_optim_step_log_time = time.time()
    done                     = False

    conv_in_diag     = {"ch_0_3_norm": 0.0, "ch_4_7_norm": 0.0, "ch_8_11_norm": 0.0, "adaptive_mult": float(conv_in_lr_multiplier), "t_mid": 0}
    contrastive_diag = None

    batches_to_skip = 0
    if config.RESUME_TRAINING:
        batches_to_skip = micro_step % len(dataloader)
        print(f"INFO: Skipping {batches_to_skip} batches to align state.")

    while not done:
        for batch in dataloader:
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            if micro_step >= config.MAX_TRAIN_STEPS:
                done = True
                break

            if not batch: continue

            micro_step += 1

            if "latent_path" in batch: accumulated_latent_paths.extend(batch["latent_path"])

            latents           = batch["latents"].to(device, non_blocking=True).detach()
            embeds            = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled            = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            batch_crop_coords = batch.get("crop_coords", [(0, 0)] * len(batch["latents"]))

            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):
                time_ids_list = []
                for s1, crop, s2 in zip(batch["original_sizes"], batch_crop_coords, batch["target_sizes"]):
                    time_ids_list.append(torch.tensor([s1[1], s1[0], crop[0], crop[1], s2[1], s2[0]], dtype=torch.float32).unsqueeze(0))
                time_ids = torch.cat(time_ids_list, dim=0).to(device, dtype=config.compute_dtype)

                noise     = generate_train_noise(latents, config, micro_step, initial_sampler_seed)
                timesteps = timestep_sampler.sample(latents.shape[0])

                if config.is_rectified_flow:
                    t_normalized  = timesteps.float() / 1000.0
                    shift_factor  = getattr(config, "RF_SHIFT_FACTOR", 3.0)
                    current_shift = timestep_sampler.get_dynamic_shift(batch["target_sizes"][0][0], batch["target_sizes"][0][1]) if timestep_sampler.use_dynamic_shift else shift_factor
                    t_shifted     = timestep_sampler.apply_flow_shift(t_normalized, current_shift)
                    t_expanded    = t_shifted.view(-1, 1, 1, 1)
                    noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
                    target        = noise - latents
                    timesteps_conditioning = (t_shifted * 1000).long().clamp(0, 999)
                else:
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    target        = scheduler.get_velocity(latents, noise, timesteps) if scheduler.config.prediction_type == "v_prediction" else noise
                    timesteps_conditioning = timesteps
                    t_shifted  = timesteps.float() / 1000.0
                    t_expanded = t_shifted.view(-1, 1, 1, 1)

                B, C, H, W = noisy_latents.shape

                # ── In-distribution placeholders for blind passes ─────────────────
                blind_input = torch.cat([noisy_latents.detach(), noisy_latents.detach(), noisy_latents.detach()], dim=1)

                # ── t_mid: random fraction of t_curr ─────────────────────────────
                t_curr_val = t_shifted.mean().item()
                t_mid_val  = t_curr_val * random.uniform(0.3, 0.7)
                t_mid_ts   = torch.full_like(t_shifted, t_mid_val * 1000).long().clamp(0, 999)
                conv_in_diag["t_mid"] = int(t_mid_val * 1000)

                # ── Blind passes — UNet inside no_grad, decode OUTSIDE ────────────
                unet.disable_gradient_checkpointing()
                with torch.no_grad():

                    blind_pred_curr = unet(
                        blind_input,
                        timesteps_conditioning.detach(), embeds.detach(),
                        added_cond_kwargs={"text_embeds": pooled.detach(), "time_ids": time_ids.detach()}
                    ).sample

                    blind_pred_mid = unet(
                        blind_input,
                        t_mid_ts.detach(), embeds.detach(),
                        added_cond_kwargs={"text_embeds": pooled.detach(), "time_ids": time_ids.detach()}
                    ).sample

                    del blind_input

                # ── Decode outside no_grad — gradient graph alive ─────────────────
                t_mid_exp   = t_mid_ts.float().view(-1, 1, 1, 1) / 1000.0
                current_ref = (noisy_latents.detach().float() - t_expanded.detach().float() * blind_pred_curr.float()).to(config.compute_dtype)
                mid_ref     = (noisy_latents.detach().float() - t_mid_exp.float() * blind_pred_mid.float()).to(config.compute_dtype)
                del blind_pred_curr, blind_pred_mid, t_mid_exp, t_mid_ts

                unet.enable_gradient_checkpointing()

                # ── Conv_in norm tracking ─────────────────────────────────────────
                conv_in_weight = unet.conv_in.weight
                ch_0_3_norm   = conv_in_weight[:, :4,  :, :].norm().item()
                ch_4_7_norm   = conv_in_weight[:, 4:8, :, :].norm().item()
                ch_8_11_norm  = conv_in_weight[:, 8:,  :, :].norm().item()

                # Adaptive LR: drive ch 4-11 toward 70% of ch 0-3 norm
                ratio         = (min(ch_4_7_norm, ch_8_11_norm) / ch_0_3_norm) if ch_0_3_norm > 1e-6 else 0.0
                target_ratio  = 0.70
                adaptive_mult = conv_in_lr_multiplier * (max(0.1, target_ratio - ratio) / target_ratio)
                adaptive_mult = max(1.0, adaptive_mult)

                for pg in optimizer.param_groups:
                    if any(id(p) in conv_in_ids for p in pg['params']):
                        pg['lr_scale'] = adaptive_mult
                        break

                conv_in_diag.update({
                    "ch_0_3_norm":   ch_0_3_norm,
                    "ch_4_7_norm":   ch_4_7_norm,
                    "ch_8_11_norm":  ch_8_11_norm,
                    "adaptive_mult": adaptive_mult
                })

                # ── Guided pass — full gradient flow ─────────────────────────────
                model_input = torch.cat([noisy_latents, current_ref, mid_ref], dim=1)

                pred = unet(
                    model_input, timesteps_conditioning, embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}
                ).sample

            # ── Loss ─────────────────────────────────────────────────────────────
            if loss_type == "DestinationLoss":
                clean        = latents.float()
                guided_x0    = noisy_latents.float() - t_expanded.float() * pred.float()
                guided_dist  = F.mse_loss(guided_x0, clean, reduction="mean")

                # Anchor-forcing loss: register hook to zero ch 0-3 gradients
                # so this loss CANNOT be satisfied via the base channels.
                # The only path to minimize destination_dist + midpoint_dist
                # is to strengthen conv_in ch 4-11.
                anchor_hook      = zero_grad_channels(unet.conv_in.weight, slice(0, 4))
                destination_dist = F.mse_loss(current_ref.float(), clean, reduction="mean")
                midpoint_dist    = F.mse_loss(mid_ref.float(),     clean, reduction="mean")
                anchor_hook.remove()

                anchor_weight = getattr(config, "ANCHOR_LOSS_WEIGHT", 3.0)
                loss = guided_dist + anchor_weight * (destination_dist + midpoint_dist)

                contrastive_diag = {
                    "mode":             "DestinationLoss",
                    "guided_dist":      guided_dist.item(),
                    "destination_dist": destination_dist.item(),
                    "midpoint_dist":    midpoint_dist.item(),
                    "loss":             loss.item(),
                }
                del clean, guided_x0, guided_dist, destination_dist, midpoint_dist

            elif loss_type == "Semantic":
                base_loss     = F.mse_loss(pred.float(), target.float(), reduction="none")
                semantic_maps = batch.get("semantic_map")
                loss          = (base_loss * (1.0 + semantic_maps.to(device, dtype=torch.float32))).mean() if semantic_maps is not None else base_loss.mean()
                contrastive_diag = None

            elif loss_type == "MSE_Perturb":
                value_weight = 1.0 + torch.abs(target.float()) * 0.5
                loss         = ((pred.float() - target.float()) ** 2 * value_weight).mean()
                contrastive_diag = None

            else:  # MSE fallback
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                contrastive_diag = None

            del current_ref, mid_ref

            raw_loss_value = loss.detach().item()
            scaled_loss    = loss / config.GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            del scaled_loss, loss, pred, noise
            if micro_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.cuda.empty_cache()

            diagnostics.step(raw_loss_value)
            lr_scheduler.step(micro_step)

            if micro_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                if isinstance(optimizer, TitanAdamW):
                    raw_grad_norm = optimizer.clip_grad_norm(config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf'))
                else:
                    all_params    = [p for group in params_to_optimize for p in (group["params"] if isinstance(group, dict) else [group])]
                    raw_grad_norm = torch.nn.utils.clip_grad_norm_(all_params, config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf'))

                if isinstance(raw_grad_norm, torch.Tensor): raw_grad_norm = raw_grad_norm.item()
                clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm
                timestep_sampler.update(raw_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                optim_step_time = time.time() - last_optim_step_log_time
                optim_step_times.append(optim_step_time)
                last_optim_step_log_time = time.time()

                ch_0_3         = conv_in_diag["ch_0_3_norm"]
                ch_4_7         = conv_in_diag["ch_4_7_norm"]
                ch_8_11        = conv_in_diag["ch_8_11_norm"]
                ratio_4_7_pct  = (ch_4_7  / ch_0_3 * 100) if ch_0_3 > 1e-6 else 0.0
                ratio_8_11_pct = (ch_8_11 / ch_0_3 * 100) if ch_0_3 > 1e-6 else 0.0

                print(
                    f"\n[12ch conv_in] Optimizer Step {optimizer_step} | "
                    f"Ch 0-3: {ch_0_3:.4f} | "
                    f"Ch 4-7: {ch_4_7:.4f} ({ratio_4_7_pct:.1f}%) | "
                    f"Ch 8-11: {ch_8_11:.4f} ({ratio_8_11_pct:.1f}%) | "
                    f"LR mult: {conv_in_diag['adaptive_mult']:.1f}x | "
                    f"t_mid: {conv_in_diag['t_mid']}"
                )

                if contrastive_diag is not None and contrastive_diag.get("mode") == "DestinationLoss":
                    print(
                        f"[DestinationLoss] Step {optimizer_step}\n"
                        f"  Guided dist     : {contrastive_diag['guided_dist']:.4f}  (want lowest)\n"
                        f"  Destination dist: {contrastive_diag['destination_dist']:.4f}\n"
                        f"  Midpoint dist   : {contrastive_diag['midpoint_dist']:.4f}\n"
                        f"  Loss            : {contrastive_diag['loss']:.4f}"
                    )

                diag_data_to_log = {
                    'optim_step':          optimizer_step,
                    'avg_loss':            diagnostics.get_average_loss(),
                    'current_lr':          optimizer.param_groups[-1]['lr'],
                    'raw_grad_norm':       raw_grad_norm,
                    'clipped_grad_norm':   clipped_grad_norm,
                    'update_delta':        1.0 if raw_grad_norm > 0 else 0.0,
                    'optim_step_time':     optim_step_time,
                    'avg_optim_step_time': sum(optim_step_times) / len(optim_step_times)
                }

                reporter.check_and_report_anomaly(optimizer_step, raw_grad_norm, clipped_grad_norm, config, accumulated_latent_paths)
                diagnostics.reset()
                accumulated_latent_paths.clear()

                if config.SAVE_EVERY_N_STEPS > 0 and optimizer_step > 0 and optimizer_step % config.SAVE_EVERY_N_STEPS == 0:
                    print(f"\n--- Saving checkpoint at optimizer step {optimizer_step} ---")
                    save_checkpoint_pt(optimizer_step, micro_step, unet, model_to_load, optimizer, lr_scheduler, None, sampler, config)
            else:
                diag_data_to_log = None

            step_duration = time.time() - last_step_time
            global_step_times.append(step_duration)
            last_step_time = time.time()
            elapsed_time   = time.time() - training_start_time
            eta_seconds    = (config.MAX_TRAIN_STEPS - micro_step) * (sum(global_step_times) / len(global_step_times)) if global_step_times else 0

            reporter.log_step(micro_step, timing_data={
                'raw_step_time': step_duration,
                'elapsed_time':  elapsed_time,
                'eta':           eta_seconds,
                'loss':          raw_loss_value,
                'timestep':      timesteps_conditioning[0].item() if timesteps_conditioning is not None else 0
            }, diag_data=diag_data_to_log)

    print("\nTraining complete.")
    reporter.shutdown()
    output_path = OUTPUT_DIR / f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_unified_{str(uuid.uuid4())[:4]}.safetensors"
    save_model(output_path, unet, model_to_load, config.compute_dtype)
    print("All tasks complete. Final model saved.")



if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()