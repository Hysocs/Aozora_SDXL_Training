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
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from PIL import Image, TiffImagePlugin, ImageFile, ImageOps
from torchvision import transforms
from tqdm.auto import tqdm
import logging
import warnings
import argparse
import numpy as np
import cv2
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

warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FLUX_BN_EPS = 1e-4
BN_MEAN_SUFFIXES = [
    "bn.running_mean",
    "normalize.bn.running_mean",
    "normalize.running_mean",
]
BN_VAR_SUFFIXES = [
    "bn.running_var",
    "normalize.bn.running_var",
    "normalize.running_var",
]


def set_attention_processor(unet, attention_mode="flash_attn"):

    if attention_mode == "cudnn":
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            unet.set_attn_processor(AttnProcessor2_0())
            print("INFO: Using CuDNN SDPA backend (PyTorch 2.5+ optimized)")
        else:
            print("WARNING: CuDNN SDPA requires PyTorch 2.5+, falling back to standard SDPA")
            unet.set_attn_processor(AttnProcessor2_0())
    elif attention_mode == "xformers (Only if no Flash)":
        unet.enable_xformers_memory_efficient_attention()
        print("INFO: Using xFormers")
        
    if attention_mode == "pytorch29_optimized":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            unet.set_attn_processor(AttnProcessor2_0())
            print("INFO: Using PyTorch 2.9 Optimized SDPA (Flash + MemEfficient + Math)")
            print(f"      CUDA Version: {torch.version.cuda}")
            print(f"      PyTorch Version: {torch.__version__}")
            
        except Exception as e:
            print(f"WARNING: PyTorch 2.9 optimization failed: {e}")
            unet.set_attn_processor(AttnProcessor2_0())

    else:
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
    if img.mode in ('RGBA', 'PA', 'LA'):
        rgb_img = img.convert('RGB')
        return rgb_img
    return img.convert("RGB")

def generate_noise(latents, generator, device, dtype=None, step=None, seed=None):
    # Optional deterministic step-seeding
    if step is not None and seed is not None:
        step_seed = (seed + step) % (2**32 - 1)
        generator.manual_seed(step_seed)

    return torch.randn(latents.shape, device=device, dtype=torch.float32, generator=generator)


class TrainingConfig:
    def __init__(self):
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        self._load_from_user_config()
        self._type_check_and_correct()
        self.NOISE_MODE = "normal"
        self.compute_dtype = torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16
        self.is_rectified_flow = getattr(self, "PREDICTION_TYPE", "epsilon") == "rectified_flow"

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
                    if user_config.get("LOSS_TYPE") == "DestinationLoss":
                        user_config["LOSS_TYPE"] = "PatchMSE"
                    for key, value in user_config.items():
                        setattr(self, key, value)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"ERROR: Could not parse {path}: {e}. Using defaults.")
            else:
                print(f"WARNING: Config {path} not found. Using defaults.")

    def _type_check_and_correct(self):
        for key, value in list(self.__dict__.items()):
            if key in ["RESUME_MODEL_PATH", "RESUME_STATE_PATH"] and getattr(self, "RESUME_TRAINING", False):
                if not value or not Path(value).exists():
                    raise FileNotFoundError(f"RESUME_TRAINING is enabled, but {key}='{value}' is not a valid file path.")
            if key == "UNET_EXCLUDE_TARGETS":
                if isinstance(value, str): setattr(self, key, [item.strip() for item in value.split(',') if item.strip()])
                elif isinstance(value, list): setattr(self, key, [item for item in value if item])
                continue
            default_value = getattr(default_config, key, None)
            if default_value is None or isinstance(value, type(default_value)): continue
            expected_type = type(default_value)
            if expected_type == bool and isinstance(value, str):
                setattr(self, key, value.lower() in ['true', '1', 't', 'y', 'yes'])
                continue
            try:
                if expected_type == int: setattr(self, key, int(float(value)))
                else: setattr(self, key, expected_type(value))
            except (ValueError, TypeError):
                setattr(self, key, default_value)


class CustomCurveLRScheduler:
    def __init__(self, optimizer, curve_points, total_micro_steps):
        self.optimizer = optimizer
        self.curve_points = sorted(curve_points, key=lambda p: p[0])
        self.total_micro_steps = max(total_micro_steps, 1)
        self.current_micro_step = 0
        if not self.curve_points: raise ValueError("LR_CUSTOM_CURVE cannot be empty")
        if self.curve_points[0][0] != 0.0: self.curve_points.insert(0, [0.0, self.curve_points[0][1]])
        if self.curve_points[-1][0] != 1.0: self.curve_points.append([1.0, self.curve_points[-1][1]])
        self._update_lr()

    def _interpolate_lr(self, normalized_position):
        normalized_position = max(0.0, min(1.0, normalized_position))
        for i in range(len(self.curve_points) - 1):
            x1, y1 = self.curve_points[i]
            x2, y2 = self.curve_points[i + 1]
            if x1 <= normalized_position <= x2:
                if x2 - x1 == 0: return y1
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
        if loss is not None: self.losses.append(loss)

    def get_average_loss(self):
        if not self.losses: return 0.0
        return sum(self.losses) / len(self.losses)

    def reset(self):
        self.losses.clear()


# =========================================================================
# ASYNC REPORTER: 100% OFFLOADED LOGGING
# =========================================================================
class AsyncReporter:
    def __init__(self, total_steps, test_param_name):
        self.total_steps = total_steps
        self.test_param_name = test_param_name
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._last_line_len = 0

    def _clear_line(self):
        if self._last_line_len > 0:
            print('\r' + ' ' * self._last_line_len + '\r', end='', flush=True)
            self._last_line_len = 0

    def _format_time(self, seconds):
        if seconds is None or not math.isfinite(seconds): return "N/A"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_type, data = self.task_queue.get(timeout=0.05)
                if task_type == 'log_step': self._handle_log_step(**data)
                elif task_type == 'message': self._handle_message(**data)
                self.task_queue.task_done()
            except queue.Empty: continue

    def _handle_log_step(self, global_step, timing_data, diag_data):
        if diag_data:
            self._clear_line()
            update_status = "[OK]" if diag_data['update_delta'] > 1e-12 else "[NO UPDATE!]"
            print(
                f"\n--- Optimizer Step: {diag_data['optim_step']:<5} | Loss: {diag_data['avg_loss']:<8.5f} | LR: {diag_data['current_lr']:.2e} ---\n"
                f"  Time: {diag_data['optim_step_time']:.2f}s/step | Avg Speed: {diag_data['avg_optim_step_time']:.2f}s/step\n"
                f"  Grad Norm (Raw/Clipped): {diag_data['raw_grad_norm']:<8.4f} / {diag_data['clipped_grad_norm']:<8.4f}\n"
                f"  VRAM: Training={torch.cuda.memory_reserved()/1e9:.2f}GB | Model={torch.cuda.memory_allocated()/1e9:.2f}GB\n"
                f"  |- Update Magnitude : {diag_data['update_delta']:.4e} {update_status}\n"
            )

        bar_width = 30
        percentage = (global_step + 1) / self.total_steps
        filled_length = int(bar_width * percentage)
        bar = '#' * filled_length + '-' * (bar_width - filled_length)
        s_per_step = timing_data.get('raw_step_time', 0)
        time_spent_str = self._format_time(timing_data.get('elapsed_time'))
        eta_str = self._format_time(timing_data.get('eta'))
        loss_val = timing_data.get('loss', 0.0)
        timestep_val = timing_data.get('timestep', 'N/A')
        prog_str = f'Training |{bar}| {global_step + 1}/{self.total_steps}[{percentage:.2%}][Loss: {loss_val:.4f}, Timestep: {timestep_val}][{s_per_step:.2f}s/step, ETA: {eta_str}, Elapsed: {time_spent_str}]'
        print('\r' + prog_str, end='', flush=True)
        self._last_line_len = len(prog_str)

    def _handle_message(self, text):
        self._clear_line()
        print(text)

    def log_step(self, global_step, timing_data, diag_data=None):
        self.task_queue.put(('log_step', {'global_step': global_step, 'timing_data': timing_data, 'diag_data': diag_data}))

    def log_message(self, text):
        self.task_queue.put(('message', {'text': text}))

    def shutdown(self):
        self._clear_line()
        print("\nShutting down async reporter. Waiting for pending tasks...")
        self.task_queue.join()
        self.stop_event.set()
        self.worker_thread.join()


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

        if self.batch_size == 1:
            indices = torch.randperm(self.total_images, generator=g).tolist()
            batches = [[i] for i in indices]
        else:
            indices = torch.randperm(self.total_images, generator=g).tolist()
            buckets = defaultdict(list)
            for idx in indices:
                buckets[self.dataset.bucket_keys[idx]].append(idx)

            bucket_batches = {}
            for key in sorted(buckets):
                bucket_indices = buckets[key]
                chunks = [
                    bucket_indices[i:i + self.batch_size]
                    for i in range(0, len(bucket_indices), self.batch_size)
                ]
                if self.shuffle and len(chunks) > 1:
                    chunk_order = torch.randperm(len(chunks), generator=g).tolist()
                    chunks = [chunks[i] for i in chunk_order]
                bucket_batches[key] = chunks

            if self.shuffle:
                batches = []
                last_key = None
                while bucket_batches:
                    candidates = [key for key in bucket_batches if key != last_key]
                    if not candidates:
                        candidates = list(bucket_batches)

                    max_remaining = max(len(bucket_batches[key]) for key in candidates)
                    top_candidates = [
                        key for key in candidates
                        if len(bucket_batches[key]) == max_remaining
                    ]
                    key_index = torch.randint(len(top_candidates), (1,), generator=g).item()
                    key = top_candidates[key_index]

                    batches.append(bucket_batches[key].pop(0))
                    last_key = key
                    if not bucket_batches[key]:
                        del bucket_batches[key]
            else:
                batches = [
                    batch
                    for key in sorted(bucket_batches)
                    for batch in bucket_batches[key]
                ]

        self.epoch += 1
        yield from batches

    def __len__(self):
        return self.total_images


STANDARD_SDXL_BUCKETS = [
    (1024, 1024),
    (1152, 896), (896, 1152),
    (1216, 832), (832, 1216),
    (1344, 768), (768, 1344),
    (1440, 720), (720, 1440),
    (1536, 640), (640, 1536),
    (1600, 512), (512, 1600),
    (896, 896), (768, 768),
]


def get_optimal_bucket(orig_w, orig_h, target_area=None, stride=64):
    orig_ar = orig_w / max(orig_h, 1)
    orig_area = orig_w * orig_h
    
    if target_area is None:
        target_area = 1024 * 1024
    
    def bucket_score(bw, bh):
        bucket_ar = bw / max(bh, 1)
        bucket_area = bw * bh
        ar_error = abs(bucket_ar - orig_ar) / max(orig_ar, 0.01)
        if bucket_area > 0 and target_area > 0:
            area_error = abs(math.log(bucket_area / target_area))
        else:
            area_error = 100.0
        return ar_error * 10.0 + area_error
    
    best_bucket = min(STANDARD_SDXL_BUCKETS, key=lambda b: bucket_score(b[0], b[1]))
    bw, bh = best_bucket
    
    if bw > orig_w and bh > orig_h:
        fitting_buckets = [(w, h) for w, h in STANDARD_SDXL_BUCKETS 
                          if w <= orig_w and h <= orig_h]
        if fitting_buckets:
            best_bucket = max(fitting_buckets, key=lambda b: b[0] * b[1])
        else:
            best_bucket = min(STANDARD_SDXL_BUCKETS, key=lambda b: b[0] * b[1])
    
    return best_bucket

def get_multi_bucket_resolutions(orig_w, orig_h, target_area=None, should_upscale=False, max_extra=0):
    primary = get_optimal_bucket(orig_w, orig_h, target_area, 64)
    if max_extra <= 0:
        return [primary]

    orig_ar = orig_w / max(orig_h, 1)
    if target_area is None:
        target_area = 1024 * 1024

    def bucket_score(bucket):
        bw, bh = bucket
        bucket_ar = bw / max(bh, 1)
        bucket_area = bw * bh
        ar_error = abs(bucket_ar - orig_ar) / max(orig_ar, 0.01)
        area_error = abs(math.log(bucket_area / target_area)) if bucket_area > 0 and target_area > 0 else 100.0
        return ar_error * 10.0 + area_error

    candidates = []
    for bucket in STANDARD_SDXL_BUCKETS:
        if bucket == primary:
            continue
        bw, bh = bucket
        if not should_upscale and bw > orig_w and bh > orig_h:
            continue
        candidates.append(bucket)

    candidates.sort(key=bucket_score)
    return [primary] + candidates[:max_extra]

def make_bucket_variant_metadata(base_meta, target_w, target_h, variant_index=0):
    orig_w, orig_h = base_meta["original_size"]
    scale = max(target_w / max(orig_w, 1), target_h / max(orig_h, 1))
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))
    crop_left = max(0, (scaled_w - target_w) // 2)
    crop_top = max(0, (scaled_h - target_h) // 2)
    meta = dict(base_meta)
    meta.update({
        "target_resolution": (target_w, target_h),
        "scaled_size": (scaled_w, scaled_h),
        "crop_coords": (crop_top, crop_left),
        "bucket_variant_index": variant_index,
        "cache_suffix": "" if variant_index == 0 else f"_mb{variant_index}",
    })
    return meta

def smart_resize(image, target_w, target_h):
    orig_w, orig_h = image.size
    scale_x = target_w / max(orig_w, 1)
    scale_y = target_h / max(orig_h, 1)
    scale = max(scale_x, scale_y)
    
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    new_w = max(new_w, target_w)
    new_h = max(new_h, target_h)
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    crop_left = (new_w - target_w) // 2
    crop_top = (new_h - target_h) // 2
    crop_right = crop_left + target_w
    crop_bottom = crop_top + target_h
    
    cropped = resized.crop((crop_left, crop_top, crop_right, crop_bottom))
    assert cropped.size == (target_w, target_h), \
        f"smart_resize failed: expected ({target_w},{target_h}), got {cropped.size}"
    return cropped

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

        if not should_upscale and (w * h) < target_area:
            target_w, target_h = get_optimal_bucket(w, h, w * h, stride)
        else:
            target_w, target_h = get_optimal_bucket(w, h, target_area, stride)

        scale = max(target_w / w, target_h / h)
        scaled_w = int(round(w * scale))
        scaled_h = int(round(h * scale))
        
        crop_left = max(0, (scaled_w - target_w) // 2)
        crop_top = max(0, (scaled_h - target_h) // 2)

        caption = read_caption_for_image(ip)

        return {
            "ip": ip, 
            "caption": caption, 
            "target_resolution": (target_w, target_h),
            "original_size": (w, h), 
            "scaled_size": (scaled_w, scaled_h),
            "crop_coords": (crop_top, crop_left),
            "original_area": w * h, 
            "target_area": target_w * target_h,
            "was_upscaled": should_upscale and (w * h) < target_area
        }
    except Exception as e:
        print(f"\n[CORRUPT IMAGE OR READ ERROR] Skipping {ip}, Reason: {e}")
        return None

def caption_chunking_enabled(config):
    return bool(getattr(config, "CAPTION_CHUNKING_ENABLED", False))


def read_caption_for_image(ip):
    cp = ip.with_suffix('.txt')
    caption = ip.stem.replace('_', ' ')
    if cp.exists():
        with open(cp, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
            if content:
                caption = content
    return caption


def get_tokenizer_max_length(tokenizer):
    return int(getattr(tokenizer, "model_max_length", 77) or 77)


def get_caption_token_ids(tokenizer, caption):
    tokenized = tokenizer(caption, add_special_tokens=False, truncation=False)
    input_ids = tokenized.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return input_ids


def get_caption_chunk_count(caption, tokenizer):
    max_len = get_tokenizer_max_length(tokenizer)
    chunk_payload_len = max(1, max_len - 2)
    return max(1, math.ceil(len(get_caption_token_ids(tokenizer, caption)) / chunk_payload_len))


def get_max_caption_chunks_for_config(config, t1, t2):
    if not caption_chunking_enabled(config):
        return 1

    max_chunks = 1
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists():
            continue
        for ip in (p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")):
            caption = read_caption_for_image(ip)
            max_chunks = max(
                max_chunks,
                get_caption_chunk_count(caption, t1),
                get_caption_chunk_count(caption, t2),
            )
    return max_chunks


def build_chunked_token_tensor(tokenizer, caption, total_chunks, device):
    max_len = get_tokenizer_max_length(tokenizer)
    payload_len = max(1, max_len - 2)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    ids = get_caption_token_ids(tokenizer, caption)

    chunks = []
    for i in range(max(1, int(total_chunks or 1))):
        payload = ids[i * payload_len:(i + 1) * payload_len]
        chunk = [bos] + payload + [eos]
        chunk += [pad] * (max_len - len(chunk))
        chunks.append(chunk[:max_len])
    return torch.tensor(chunks, dtype=torch.long, device=device)


def encode_caption_chunks_sdxl(caption, t1, t2, te1, te2, device, total_chunks):
    tokens_1 = build_chunked_token_tensor(t1, caption, total_chunks, device)
    tokens_2 = build_chunked_token_tensor(t2, caption, total_chunks, device)
    enc_1_output = te1(tokens_1, output_hidden_states=True)
    enc_2_output = te2(tokens_2, output_hidden_states=True)
    hidden_1 = enc_1_output.hidden_states[-2].reshape(1, -1, enc_1_output.hidden_states[-2].shape[-1])
    hidden_2 = enc_2_output.hidden_states[-2].reshape(1, -1, enc_2_output.hidden_states[-2].shape[-1])
    return torch.cat([hidden_1, hidden_2], dim=-1), enc_2_output[0][:1]


def compute_text_embeddings_sdxl(captions, t1, t2, te1, te2, device, allow_chunking=False, total_chunks=None):
    prompt_embeds_list, pooled_prompt_embeds_list = [], []
    if allow_chunking:
        if total_chunks is None:
            total_chunks = max(
                1,
                *(max(get_caption_chunk_count(caption, t1), get_caption_chunk_count(caption, t2)) for caption in captions),
            )
        total_chunks = max(1, int(total_chunks))
    for caption in captions:
        with torch.no_grad():
            if allow_chunking:
                prompt_embeds, pooled_prompt_embeds = encode_caption_chunks_sdxl(caption, t1, t2, te1, te2, device, total_chunks)
            else:
                tokens_1 = t1(caption, padding="max_length", max_length=t1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                tokens_2 = t2(caption, padding="max_length", max_length=t2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
                enc_1_output = te1(tokens_1, output_hidden_states=True)
                enc_2_output = te2(tokens_2, output_hidden_states=True)
                prompt_embeds = torch.cat([enc_1_output.hidden_states[-2], enc_2_output.hidden_states[-2]], dim=-1)
                pooled_prompt_embeds = enc_2_output[0]
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    return torch.cat(prompt_embeds_list, dim=0), torch.cat(pooled_prompt_embeds_list, dim=0)

def get_text_conditioning_scale_range(config):
    if not bool(getattr(config, "TEXT_CONDITIONING_SCALE_ENABLED", False)):
        return 1.0, 1.0
    min_scale = float(getattr(config, "TEXT_CONDITIONING_SCALE_MIN", 1.0))
    max_scale = float(getattr(config, "TEXT_CONDITIONING_SCALE_MAX", 1.0))
    min_scale = min(max(min_scale, 0.0), 1.0)
    max_scale = min(max(max_scale, 0.0), 2.0)
    if min_scale > max_scale:
        min_scale, max_scale = max_scale, min_scale
    return min_scale, max_scale

def text_conditioning_scale_enabled(config):
    min_scale, max_scale = get_text_conditioning_scale_range(config)
    return min_scale < 1.0 or max_scale > 1.0

def null_conditioning_cache_needed(config):
    return bool(getattr(config, "UNCONDITIONAL_DROPOUT", False)) or text_conditioning_scale_enabled(config)

def get_caption_cache_options(config):
    vae_source = get_vae_source_for_config(config)
    vae_source_path = ""
    vae_source_size = None
    vae_source_mtime_ns = None
    if vae_source:
        try:
            vae_source_resolved = Path(vae_source).resolve()
            vae_source_path = str(vae_source_resolved)
            if vae_source_resolved.exists():
                stat = vae_source_resolved.stat()
                vae_source_size = stat.st_size
                vae_source_mtime_ns = stat.st_mtime_ns
        except OSError:
            vae_source_path = str(vae_source)

    return {
        "version": 10,
        "unconditional_dropout": bool(getattr(config, "UNCONDITIONAL_DROPOUT", False)),
        "caption_chunking_enabled": caption_chunking_enabled(config),
        "multi_bucket_enabled": bool(getattr(config, "MULTI_BUCKET_ENABLED", False)),
        "multi_bucket_extra_buckets": int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0) if getattr(config, "MULTI_BUCKET_ENABLED", False) else 0,
        "vae_normalization_mode": getattr(config, "VAE_NORMALIZATION_MODE", "scalar"),
        "vae_shift_factor": getattr(config, "VAE_SHIFT_FACTOR", None),
        "vae_scaling_factor": getattr(config, "VAE_SCALING_FACTOR", None),
        "vae_latent_channels": getattr(config, "VAE_LATENT_CHANNELS", None),
        "vae_path": str(getattr(config, "VAE_PATH", "") or ""),
        "vae_source_path": vae_source_path,
        "vae_source_size": vae_source_size,
        "vae_source_mtime_ns": vae_source_mtime_ns,
    }

def check_if_caching_needed(config):
    needs_caching = False
    cache_folder_name = ".precomputed_embeddings_cache_rf" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    expected_cache_options = get_caption_cache_options(config)

    if null_conditioning_cache_needed(config):
        if config.INSTANCE_DATASETS and not (Path(config.INSTANCE_DATASETS[0]["path"]) / cache_folder_name / "null_embeds.pt").exists():
            needs_caching = True

    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists(): continue
        image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        if not image_paths: continue

        cache_dir = root / cache_folder_name
        if not cache_dir.exists():
            needs_caching = True; continue
            
        index_path = cache_dir / "dataset_index.json"
        if not index_path.exists():
            needs_caching = True; continue
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            if index_data.get("cache_options") != expected_cache_options:
                needs_caching = True
            indexed_files = index_data.get("files", [])
            if any("scaled_size" not in item for item in indexed_files):
                needs_caching = True
            if len(indexed_files) < len(image_paths):
                needs_caching = True
            for item in indexed_files[: min(len(indexed_files), 10)]:
                te_path = item.get("te_path")
                lat_path = item.get("lat_path")
                if not te_path or not lat_path or not Path(te_path).exists() or not Path(lat_path).exists():
                    needs_caching = True
                    break
        except Exception:
            needs_caching = True

        cached_te_files = list(cache_dir.glob("*_te.pt"))
        if len(cached_te_files) < len(image_paths):
            needs_caching = True
        else:
            for f in cached_te_files[: min(len(cached_te_files), 10)]:
                try:
                    data_te = torch.load(f, map_location="cpu", weights_only=True)
                    if data_te.get("cache_options") != expected_cache_options:
                        needs_caching = True
                        break
                except Exception:
                    needs_caching = True
                    break
    return needs_caching

def load_unet_robust(path, compute_dtype):
    print(f"INFO: Loading UNet from: {Path(path).name}")
    in_channels = 4
    out_channels = 4
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            key_in = "model.diffusion_model.input_blocks.0.0.weight"
            key_out = "model.diffusion_model.out.2.weight"
            if key_in in f.keys():
                shape_in = f.get_slice(key_in).get_shape()
                in_channels = shape_in[1]
            if key_out in f.keys():
                shape_out = f.get_slice(key_out).get_shape()
                out_channels = shape_out[0]
    except Exception as e:
        print(f"WARNING: Could not peek into safetensors for channel sizes, falling back to defaults. Error: {e}")

    print(f"INFO: Detected UNet configuration - in_channels: {in_channels}, out_channels: {out_channels}")

    try:
        unet = UNet2DConditionModel.from_single_file(
            path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            in_channels=in_channels,
            out_channels=out_channels
        )
        print(f"INFO: Loaded UNet (Channels: In={unet.config.in_channels} / Out={unet.config.out_channels})")
        return unet
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load UNet. {e}")
        raise e

def load_vae_robust(path, device, target_channels=None):
    print(f"INFO: Attempting to load VAE from: {path}")
    if target_channels is None:
        try:
            tensors = load_file(path, device="cpu")
            for k in ["first_stage_model.quant_conv.weight", "quant_conv.weight"]:
                if k in tensors:
                    target_channels = tensors[k].shape[0] // 2
                    break
        except Exception: pass

    target_channels = target_channels or 4
    try:
        if target_channels != 4:
            vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float32, latent_channels=target_channels, ignore_mismatched_sizes=True, low_cpu_mem_usage=False)
        else:
            vae = AutoencoderKL.from_single_file(path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        print(f"INFO: Successfully loaded VAE with {target_channels} channels.")
        return vae.to(device)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load VAE: {e}")
        raise e

def find_tensor_by_suffix(safetensors_path, suffixes):
    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for suffix in suffixes:
            matches = [k for k in keys if k == suffix or k.endswith("." + suffix)]
            if matches:
                key = sorted(matches, key=len)[0]
                return f.get_tensor(key).float(), key
    return None, None

def extract_flux_bn_stats_from_safetensor(safetensors_path):
    mean, mean_key = find_tensor_by_suffix(safetensors_path, BN_MEAN_SUFFIXES)
    var, var_key = find_tensor_by_suffix(safetensors_path, BN_VAR_SUFFIXES)

    if mean is None or var is None:
        raise RuntimeError(
            f"Could not find Flux BN stats in {safetensors_path}. "
            "Expected keys ending with bn.running_mean and bn.running_var."
        )
    if mean.numel() != 128 or var.numel() != 128:
        raise RuntimeError(
            f"Flux BN stats found but wrong shape. "
            f"mean={tuple(mean.shape)}, var={tuple(var.shape)}. Expected 128 elements."
        )

    print(
        f"INFO: Loaded Flux VAE BN stats from {Path(safetensors_path).name}\n"
        f"      mean key: {mean_key}\n"
        f"      var key:  {var_key}\n"
        f"      mean range: [{mean.min().item():+.5f}, {mean.max().item():+.5f}]\n"
        f"      var  range: [{var.min().item():+.5f}, {var.max().item():+.5f}]"
    )
    return mean, var

def flux_bn32_to_bn128_layout(latents):
    if latents.dim() != 4 or latents.shape[1] != 32:
        raise RuntimeError(f"flux_bn32 expects [N, 32, H, W] latents before BN, got shape {tuple(latents.shape)}")
    if latents.shape[2] % 2 != 0 or latents.shape[3] % 2 != 0:
        raise RuntimeError(f"flux_bn32 requires even latent height/width, got shape {tuple(latents.shape)}")

    n, c, h, w = latents.shape
    return (
        latents.view(n, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(n, c * 4, h // 2, w // 2)
    )

def flux_bn128_to_bn32_layout(latents):
    if latents.dim() != 4 or latents.shape[1] != 128:
        raise RuntimeError(f"flux_bn32 decode expects [N, 128, H, W] BN latents, got shape {tuple(latents.shape)}")

    n, c, h, w = latents.shape
    return (
        latents.view(n, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(n, c // 4, h * 2, w * 2)
    )

def apply_flux_bn32_norm(latents, mean_128, var_128):
    latents_bn128 = flux_bn32_to_bn128_layout(latents)
    mean_128 = mean_128.to(device=latents_bn128.device, dtype=latents_bn128.dtype)
    var_128 = var_128.to(device=latents_bn128.device, dtype=latents_bn128.dtype)
    latents_bn128 = F.batch_norm(latents_bn128, mean_128, var_128, training=False, momentum=0.1, eps=FLUX_BN_EPS)
    return flux_bn128_to_bn32_layout(latents_bn128)

def invert_flux_bn32_norm(latents, mean_128, var_128):
    latents_bn128 = flux_bn32_to_bn128_layout(latents)
    mean_128 = mean_128.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
    sigma_128 = torch.sqrt(var_128.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1) + FLUX_BN_EPS)
    return flux_bn128_to_bn32_layout(latents_bn128 * sigma_128 + mean_128)

def get_vae_source_for_config(config):
    vae_path = getattr(config, "VAE_PATH", None)
    if vae_path and Path(vae_path).exists():
        return vae_path
    return getattr(config, "SINGLE_FILE_CHECKPOINT_PATH", None)

def denormalize_latents_for_vae_decode(latents, config, vae=None, bn_mean_128=None, bn_var_128=None):
    mode = str(getattr(config, "VAE_NORMALIZATION_MODE", "scalar")).lower()

    if mode == "flux_bn32":
        if bn_mean_128 is None or bn_var_128 is None:
            vae_source = get_vae_source_for_config(config)
            if not vae_source or not Path(vae_source).exists():
                raise RuntimeError("VAE_NORMALIZATION_MODE='flux_bn32' requires VAE_PATH or SINGLE_FILE_CHECKPOINT_PATH.")
            bn_mean_128, bn_var_128 = extract_flux_bn_stats_from_safetensor(vae_source)
        return invert_flux_bn32_norm(latents, bn_mean_128, bn_var_128)

    if mode != "scalar":
        raise RuntimeError(f"Unknown VAE_NORMALIZATION_MODE: {mode}")

    shift = getattr(config, "VAE_SHIFT_FACTOR", None)
    scale = getattr(config, "VAE_SCALING_FACTOR", None)
    if scale is None and vae is not None:
        scale = getattr(vae.config, "scaling_factor", 1.0)
    if shift is None and vae is not None:
        shift = getattr(vae.config, "shift_factor", None)

    scale = 1.0 if scale is None else scale
    if shift is not None:
        return latents / scale + shift
    return latents / scale

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config):
        print("\n" + "="*60 + "\nINFO: Datasets already cached.\n" + "="*60 + "\n")
        return

    cache_folder_name = ".precomputed_embeddings_cache_rf" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
    expected_cache_options = get_caption_cache_options(config)

    vae.to(device, dtype=torch.float32)
    vae.enable_tiling()
    vae.enable_slicing()
    print(f"INFO: VAE shift={vae.config.shift_factor}, scale={vae.config.scaling_factor}, channels={vae.config.latent_channels}")

    vae_norm_mode = str(getattr(config, "VAE_NORMALIZATION_MODE", "scalar")).lower()
    bn_mean_128 = None
    bn_var_128 = None
    if vae_norm_mode == "flux_bn32":
        vae_source = get_vae_source_for_config(config)
        if not vae_source or not Path(vae_source).exists():
            raise RuntimeError("VAE_NORMALIZATION_MODE='flux_bn32' requires VAE_PATH or SINGLE_FILE_CHECKPOINT_PATH.")
        bn_mean_128, bn_var_128 = extract_flux_bn_stats_from_safetensor(vae_source)
        print(
            "INFO: Using ComfyUI-style Flux BN32 latent normalization\n"
            "      applies BN in [N,128,H/2,W/2] layout, then stores [N,32,H,W]\n"
            f"      mean_128 range: [{bn_mean_128.min().item():+.5f}, {bn_mean_128.max().item():+.5f}]\n"
            f"      var_128 range:  [{bn_var_128.min().item():+.5f}, {bn_var_128.max().item():+.5f}]"
        )
    elif vae_norm_mode != "scalar":
        raise RuntimeError(f"Unknown VAE_NORMALIZATION_MODE: {vae_norm_mode}")

    te1.to(device)
    te2.to(device)
    allow_caption_chunking = caption_chunking_enabled(config)
    caption_total_chunks = get_max_caption_chunks_for_config(config, t1, t2) if allow_caption_chunking else 1
    if allow_caption_chunking:
        print(
            f"INFO: Caption chunking enabled for cached text embeddings: "
            f"{caption_total_chunks} chunk(s), {caption_total_chunks * min(get_tokenizer_max_length(t1), get_tokenizer_max_length(t2))} tokens max."
        )

    if null_conditioning_cache_needed(config):
        with torch.no_grad():
            null_embeds, null_pooled = compute_text_embeddings_sdxl(
                [""],
                t1, t2, te1, te2, device,
                allow_chunking=allow_caption_chunking,
                total_chunks=caption_total_chunks,
            )
    else:
        null_embeds, null_pooled = None, None

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        cache_dir = root / cache_folder_name
        cache_dir.mkdir(exist_ok=True)

        if null_embeds is not None and null_pooled is not None:
            torch.save({"embeds": null_embeds.cpu(), "pooled": null_pooled.cpu()}, cache_dir / "null_embeds.pt")

        paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        multi_bucket_extra = (
            max(0, int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0))
            if getattr(config, "MULTI_BUCKET_ENABLED", False)
            else 0
        )
        if multi_bucket_extra > 0:
            print(f"INFO: Multi-bucket cache enabled: up to {multi_bucket_extra} extra bucket(s) per image.")
        force_recaching = True
        index_path = cache_dir / "dataset_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    force_recaching = json.load(f).get("cache_options") != expected_cache_options
            except Exception:
                force_recaching = True

        if force_recaching:
            stale_files = list(cache_dir.glob("*_te*.pt")) + list(cache_dir.glob("*_lat.pt"))
            stale_files += [cache_dir / "null_embeds.pt"]
            for stale_file in stale_files:
                if not stale_file.exists():
                    continue
                try:
                    stale_file.unlink()
                except OSError as e:
                    print(f"WARNING: Could not remove stale cache file {stale_file}: {e}")

        to_process = []
        for p in paths:
            safe_stem = str(p.relative_to(root).with_suffix('')).replace(os.sep, '_')
            if force_recaching or not (cache_dir / f"{safe_stem}_te.pt").exists() or not (cache_dir / f"{safe_stem}_lat.pt").exists():
                to_process.append(p)

        if to_process:
            with Pool(processes=min(cpu_count(), 8)) as pool:
                results = list(tqdm(pool.imap(validate_and_assign_resolution, [(p, config.TARGET_PIXEL_AREA, 64, config.SHOULD_UPSCALE) for p in to_process]), total=len(to_process)))

            expanded_results = []
            for m in (r for r in results if r):
                target_area = (m["original_area"] if not config.SHOULD_UPSCALE and m["original_area"] < config.TARGET_PIXEL_AREA else config.TARGET_PIXEL_AREA)
                buckets_for_image = get_multi_bucket_resolutions(
                    m["original_size"][0],
                    m["original_size"][1],
                    target_area,
                    config.SHOULD_UPSCALE,
                    multi_bucket_extra,
                )
                for variant_index, (target_w, target_h) in enumerate(buckets_for_image):
                    expanded_results.append(make_bucket_variant_metadata(m, target_w, target_h, variant_index))

            grouped = defaultdict(list)
            for m in expanded_results: grouped[m["target_resolution"]].append(m)

            batches = [(res, grouped[res][i:i + config.CACHING_BATCH_SIZE]) for res in grouped for i in range(0, len(grouped[res]), config.CACHING_BATCH_SIZE)]
            random.shuffle(batches)

            variants_total_by_image = defaultdict(int)
            variants_done_by_image = defaultdict(int)
            started_images = set()
            for m in expanded_results:
                variants_total_by_image[str(m["ip"].relative_to(root))] += 1

            touched_images = 0
            total_images = len(variants_total_by_image)

            def mark_cache_variant_done(meta, progress_bar):
                nonlocal touched_images
                image_key = str(meta["ip"].relative_to(root))
                total_variants = variants_total_by_image[image_key]
                if total_variants <= 0 or variants_done_by_image[image_key] >= total_variants:
                    return
                if image_key not in started_images:
                    started_images.add(image_key)
                    touched_images += 1
                variants_done_by_image[image_key] += 1
                progress_bar.set_postfix_str(
                    f"image_step {variants_done_by_image[image_key]}/{total_variants}, images {touched_images}/{total_images}"
                )
                progress_bar.update(1)

            with tqdm(total=len(expanded_results), desc=f"Encoding {root.name}", unit="step") as encode_pbar:
                for batch_idx, ((w, h), batch_meta) in enumerate(batches):
                    images_global, valid_meta_final, skipped_meta = [], [], []
                    for m in batch_meta:
                        try:
                            with Image.open(m['ip']) as img:
                                img_resized = smart_resize(fix_alpha_channel(img), w, h)
                                images_global.append(transform(img_resized))
                                valid_meta_final.append(m)
                        except Exception as e:
                            skipped_meta.append(m)
                            tqdm.write(f"[SKIP] {m['ip'].name}: {e}")

                    for m in skipped_meta:
                        mark_cache_variant_done(m, encode_pbar)

                    if not images_global: continue

                    embeds_all, pooled_all = compute_text_embeddings_sdxl(
                        [m["caption"] for m in valid_meta_final],
                        t1, t2, te1, te2, device,
                        allow_chunking=allow_caption_chunking,
                        total_chunks=caption_total_chunks,
                    )

                    with torch.no_grad():
                        latents_global = vae.encode(torch.stack(images_global).to(device, dtype=torch.float32)).latent_dist.mean
                        raw_min = latents_global.min().item()
                        raw_max = latents_global.max().item()
                        raw_mean = latents_global.mean().item()
                        raw_std = latents_global.std().item()

                        if vae_norm_mode == "flux_bn32":
                            if latents_global.shape[1] != 32:
                                raise RuntimeError(
                                    f"flux_bn32 expects 32-channel latents, got shape {tuple(latents_global.shape)}"
                                )
                            latents_global = apply_flux_bn32_norm(latents_global, bn_mean_128, bn_var_128)
                        elif getattr(vae.config, 'shift_factor', None) is not None:
                            latents_global = (latents_global - vae.config.shift_factor) * vae.config.scaling_factor
                        else:
                            latents_global = latents_global * vae.config.scaling_factor

                        norm_min = latents_global.min().item()
                        norm_max = latents_global.max().item()
                        norm_mean = latents_global.mean().item()
                        norm_std = latents_global.std().item()

                    latents_global = latents_global.cpu()

                    for j, m in enumerate(valid_meta_final):
                        safe_filename = str(m['ip'].relative_to(root).with_suffix('')).replace(os.sep, '_') + m.get("cache_suffix", "")
                        torch.save({
                            "original_stem": m['ip'].stem, "relative_path": str(m['ip'].relative_to(root)),
                            "original_size": m["original_size"], "scaled_size": m.get("scaled_size", m["original_size"]),
                            "target_size": (w, h),
                            "crop_coords": m.get("crop_coords", (0, 0)),
                            "bucket_variant_index": m.get("bucket_variant_index", 0),
                            "caption": m["caption"],
                            "embeds": embeds_all[j].to(dtype=torch.float32).cpu(),
                            "pooled": pooled_all[j].to(dtype=torch.float32).cpu(),
                            "cache_options": expected_cache_options,
                            "vae_normalization_mode": vae_norm_mode,
                            "vae_shift": vae.config.shift_factor, "vae_scale": vae.config.scaling_factor,
                            "flux_bn_eps": FLUX_BN_EPS if vae_norm_mode == "flux_bn32" else None,
                        }, cache_dir / f"{safe_filename}_te.pt")
                        torch.save(latents_global[j].to(dtype=torch.float32).cpu(), cache_dir / f"{safe_filename}_lat.pt")
                        mark_cache_variant_done(m, encode_pbar)

                    if batch_idx % 20 == 0: torch.cuda.empty_cache()
                
        print(f"INFO: Building fast JSON index for {root.name}...")
        index_data = []
        cached_files = list(cache_dir.glob("*_te.pt"))
        for f in cached_files:
            try:
                data_te = torch.load(f, map_location='cpu')
                
                index_data.append({
                    "te_path": str(f),
                    "lat_path": str(f).replace("_te.pt", "_lat.pt"),
                    "target_size": data_te.get('target_size'),
                    "original_size": data_te.get('original_size'),
                    "scaled_size": data_te.get('scaled_size', data_te.get('original_size')),
                    "crop_coords": data_te.get('crop_coords', (0,0)),
                    "bucket_variant_index": data_te.get("bucket_variant_index", 0),
                })
            except Exception as e:
                print(f"WARNING: Could not index {f}: {e}")
                
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump({"version": 10, "cache_options": expected_cache_options, "files": index_data}, f)
        print(f"INFO: Saved dataset index to {index_path}")

    te1.cpu(); te2.cpu(); gc.collect(); torch.cuda.empty_cache()


class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.items = []
        self.bucket_keys = []
        self.seed = config.SEED if config.SEED else 42
        self.worker_rng = None
        cache_folder_name = ".precomputed_embeddings_cache_rf" if config.is_rectified_flow else ".precomputed_embeddings_cache_standard_sdxl"
        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            index_path = root / cache_folder_name / "dataset_index.json"
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    
                repeats = int(ds.get("repeats", 1))
                for _ in range(repeats):
                    for item in index_data["files"]:
                        self.items.append(item)
                        self.bucket_keys.append(tuple(item["target_size"]))
            else:
                print(f"WARNING: Index missing at {index_path}. Please re-run caching!")

        if not self.items: raise ValueError("No cached files found.")

        combined = list(zip(self.items, self.bucket_keys))
        random.shuffle(combined)
        self.items, self.bucket_keys = zip(*combined)
        self.items = list(self.items)
        self.bucket_keys = list(self.bucket_keys)
        self.null_embeds = None
        self.null_pooled = None
        self.cond_scale_min, self.cond_scale_max = get_text_conditioning_scale_range(config)
        self.cond_scale_enabled = self.cond_scale_min < 1.0 or self.cond_scale_max < 1.0
        self.dropout_prob = (
            min(max(float(getattr(config, "UNCONDITIONAL_DROPOUT_CHANCE", 0.0)), 0.0), 1.0)
            if getattr(config, "UNCONDITIONAL_DROPOUT", False)
            else 0.0
        )
        if self.dropout_prob > 0 or self.cond_scale_enabled:
            try:
                null_data = torch.load(Path(config.INSTANCE_DATASETS[0]["path"]) / cache_folder_name / "null_embeds.pt", map_location="cpu", weights_only=True)
                self.null_embeds = null_data["embeds"].squeeze(0) if null_data["embeds"].dim() == 3 else null_data["embeds"]
                self.null_pooled = null_data["pooled"].squeeze(0) if null_data["pooled"].dim() == 2 else null_data["pooled"]
            except Exception:
                self.dropout_prob = 0.0
                self.cond_scale_enabled = False

    def __len__(self): return len(self.items)

    def _init_worker_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_rng = random.Random(self.seed + (worker_info.id if worker_info else 0))

    def _load_tensor(self, path):
        if not path: return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def _load_latent_payload(self, path):
        payload = self._load_tensor(path)
        if isinstance(payload, dict):
            return payload.get("latents")
        return payload

    def __getitem__(self, i):
        try:
            if self.worker_rng is None: self._init_worker_rng()
            
            item_data = self.items[i]
            path_te = item_data["te_path"]
            path_lat = item_data["lat_path"]

            data_te = self._load_tensor(path_te)
            latents = self._load_latent_payload(path_lat)
            embeds, pooled = data_te["embeds"], data_te["pooled"]

            if torch.isnan(latents).any() or torch.isinf(latents).any(): return None

            item = {
                "latents": latents,
                "embeds": embeds.squeeze(0) if embeds.dim() == 3 else embeds,
                "pooled": pooled.squeeze(0) if pooled.dim() == 2 else pooled,
                "original_sizes": item_data["original_size"], 
                "scaled_sizes": item_data.get("scaled_size", item_data["original_size"]),
                "target_sizes": item_data["target_size"],
                "crop_coords": item_data.get("crop_coords", (0, 0)), 
                "latent_path": path_te
            }

            if self.dropout_prob > 0 and self.worker_rng.random() < self.dropout_prob:
                item["embeds"], item["pooled"] = self.null_embeds, self.null_pooled
            elif self.cond_scale_enabled:
                scale = self.worker_rng.uniform(self.cond_scale_min, self.cond_scale_max)
                item["embeds"] = self.null_embeds + (item["embeds"] - self.null_embeds) * scale
                item["pooled"] = self.null_pooled + (item["pooled"] - self.null_pooled) * scale

            return item
        except Exception as e:
            print(f"[DATASET] Failed to load item {i}: {e}")
            return None


class TimestepSampler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.total_tickets_needed = config.MAX_TRAIN_STEPS * config.BATCH_SIZE
        self.seed = config.SEED if config.SEED else 42
        self.is_rectified_flow = config.is_rectified_flow
        allocation = getattr(config, 'TIMESTEP_ALLOCATION', None)
        self.ticket_pool = self._build_ticket_pool(allocation)
        self.pool_index = 0

    def _build_ticket_pool(self, allocation):
        if not allocation or "counts" not in allocation or "bin_size" not in allocation or sum(allocation["counts"]) == 0:
            self.bin_size = 100
            counts = [self.total_tickets_needed // (1000 // 100) // self.config.BATCH_SIZE] * (1000 // 100)
            for i in range((self.total_tickets_needed // self.config.BATCH_SIZE) % (1000 // 100)): counts[i] += 1
        else:
            self.bin_size = allocation["bin_size"]
            counts = allocation["counts"]

        scale_factor = (self.total_tickets_needed / self.config.BATCH_SIZE) / sum(counts) if sum(counts) > 0 else 1.0
        pool = []
        rng = np.random.Generator(np.random.PCG64(self.seed))
        for i, count in enumerate(counts):
            if count <= 0: continue
            start_t = i * self.bin_size
            end_t = min(1000, (i + 1) * self.bin_size)
            if start_t >= 1000: break
            num_tickets = max(1, int(count * scale_factor * self.config.BATCH_SIZE))
            pool.extend(rng.integers(start_t, end_t, size=num_tickets).tolist())

        random.seed(self.seed)
        random.shuffle(pool)
        if len(pool) == 0: pool = [random.randint(0, 999) for _ in range(self.total_tickets_needed)]
        elif len(pool) < self.total_tickets_needed:
            while len(pool) < self.total_tickets_needed: pool.extend(pool[:self.total_tickets_needed - len(pool)])
        return pool[:self.total_tickets_needed]

    def set_current_step(self, micro_step):
        self.pool_index = (micro_step * self.config.BATCH_SIZE) % len(self.ticket_pool)

    def sample(self, batch_size):
        indices = []
        for _ in range(batch_size):
            if self.pool_index >= len(self.ticket_pool): self.pool_index = 0
            indices.append(self.ticket_pool[self.pool_index])
            self.pool_index += 1
        return torch.tensor(indices, dtype=torch.long, device=self.device), indices[0]

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


def print_dataset_resolution_sample(dataset, sample_count=5):
    sample_count = min(sample_count, len(dataset.items))
    if sample_count <= 0:
        return

    print(f"INFO: Dataset resolution sample ({sample_count} cached item{'s' if sample_count != 1 else ''}):")
    for item in dataset.items[:sample_count]:
        orig_w, orig_h = item["original_size"]
        targ_w, targ_h = item["target_size"]
        orig_ar = orig_w / orig_h if orig_h else 1.0
        targ_ar = targ_w / targ_h if targ_h else 1.0
        ar_error_pct = (abs(orig_ar - targ_ar) / orig_ar * 100) if orig_ar else 0.0
        stem = Path(item["te_path"]).stem.replace("_te", "")
        variant_label = f", variant {item.get('bucket_variant_index', 0)}" if item.get("bucket_variant_index", 0) else ""
        print(
            f"INFO:   {stem}: original {orig_w}x{orig_h} (AR {orig_ar:.4f}) -> "
            f"target {targ_w}x{targ_h} (AR {targ_ar:.4f}){variant_label}, "
            f"AR diff {ar_error_pct:.2f}%, cropped not stretched"
        )


def create_optimizer(config, params_to_optimize):
    optimizer_type = config.OPTIMIZER_TYPE.lower()
    initial_lr = max(p[1] for p in getattr(config, 'LR_CUSTOM_CURVE', [])) if getattr(config, 'LR_CUSTOM_CURVE', []) else config.LEARNING_RATE

    if optimizer_type == "titan":
        final_params = {**default_config.TITAN_PARAMS, **getattr(config, 'TITAN_PARAMS', {})}
        dtype_str = final_params.get('momentum_dtype', 'bfloat16')
        m_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        return TitanAdamW(params_to_optimize, lr=initial_lr, betas=tuple(final_params.get('betas', [0.9, 0.999])), eps=final_params.get('eps', 1e-8), weight_decay=final_params.get('weight_decay', 0.01), debias_strength=final_params.get('debias_strength', 1.0), use_grad_centralization=final_params.get('use_grad_centralization', False), gc_alpha=final_params.get('gc_alpha', 1.0), momentum_dtype=m_dtype)
    elif optimizer_type == "raven":
        final_params = {**getattr(default_config, 'RAVEN_PARAMS', {}), **getattr(config, 'RAVEN_PARAMS', {})}
        dtype_str = final_params.get('momentum_dtype', 'bfloat16')
        m_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        return RavenAdamW(params_to_optimize, lr=initial_lr, betas=tuple(final_params.get('betas', [0.9, 0.999])), eps=final_params.get('eps', 1e-8), weight_decay=final_params.get('weight_decay', 0.01), debias_strength=final_params.get('debias_strength', 1.0), use_grad_centralization=final_params.get('use_grad_centralization', False), gc_alpha=final_params.get('gc_alpha', 1.0), momentum_dtype=m_dtype)
    elif optimizer_type == "velorms":
        final_params = {**getattr(default_config, 'VELORMS_PARAMS', {}), **getattr(config, 'VELORMS_PARAMS', {})}
        return VeloRMS(params_to_optimize, lr=initial_lr, weight_decay=final_params.get('weight_decay', 0.01), eps=final_params.get('eps', 1e-8), verbose=False)
    raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'")

def _get_sdxl_unet_conversion_map():
    unet_conversion_map = [
        ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"),
        ("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
        ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ]
    unet_conversion_map_resnet = [
        ("in_layers.0", "norm1"), ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"), ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"), ("skip_connection", "conv_shortcut"),
    ]
    unet_conversion_map_layer = []
    for i in range(3):
        for j in range(2):
            unet_conversion_map_layer.append((f"input_blocks.{3 * i + j + 1}.0.", f"down_blocks.{i}.resnets.{j}."))
            if i > 0: unet_conversion_map_layer.append((f"input_blocks.{3 * i + j + 1}.1.", f"down_blocks.{i}.attentions.{j}."))
        for j in range(3):
            unet_conversion_map_layer.append((f"output_blocks.{3 * i + j}.0.", f"up_blocks.{i}.resnets.{j}."))
            if i < 2: unet_conversion_map_layer.append((f"output_blocks.{3 * i + j}.1.", f"up_blocks.{i}.attentions.{j}."))
        if i < 3:
            unet_conversion_map_layer.append((f"input_blocks.{3 * (i + 1)}.0.op.", f"down_blocks.{i}.downsamplers.0.conv."))
            unet_conversion_map_layer.append((f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}.", f"up_blocks.{i}.upsamplers.0."))
    unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))
    unet_conversion_map_layer.append(("middle_block.1.", "mid_block.attentions.0."))
    for j in range(2): unet_conversion_map_layer.append((f"middle_block.{2 * j}.", f"mid_block.resnets.{j}."))
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
        final_mapping[hf_name] = sd_name if sd_name.startswith("model.diffusion_model.") else f"model.diffusion_model.{sd_name}"
    return final_mapping

def save_model(output_path, unet, base_checkpoint_path, compute_dtype):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nINFO: Saving model to: {output_path.name}")
    
    try: base_tensors = load_file(str(base_checkpoint_path), device="cpu")
    except Exception as e:
        print(f"ERROR: Could not load base checkpoint: {e}")
        return

    key_map = get_unet_key_mapping(list(unet.state_dict().keys()))
    unet_state = unet.state_dict()
    
    print(f"INFO: Base checkpoint keys: {len(base_tensors)}")
    print(f"INFO: UNet keys to merge:   {len(key_map)}")

    converted = 0
    for key in base_tensors.keys():
        if base_tensors[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
            base_tensors[key] = base_tensors[key].to(dtype=compute_dtype)
            converted += 1
    print(f"INFO: Converted {converted} base tensors to {compute_dtype}")

    updated, missing = 0, []
    for hf_key, target_key in key_map.items():
        if hf_key in unet_state:
            if target_key not in base_tensors:
                missing.append(target_key)
            tensor_cpu = unet_state[hf_key].detach().to("cpu", dtype=compute_dtype)
            base_tensors[target_key] = tensor_cpu
            updated += 1
            del tensor_cpu

    print(f"INFO: Merged {updated} UNet layers into checkpoint")
    if missing:
        print(f"WARNING: {len(missing)} keys not found in base checkpoint (new keys added):")
        for k in missing[:5]: print(f"  -> {k}")
        if len(missing) > 5: print(f"  ... and {len(missing) - 5} more")

    print(f"INFO: Saving to disk...")
    save_file(base_tensors, str(output_path))
    print(f"INFO: Save complete -> {output_path.name}")

    del base_tensors, unet_state, key_map
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def save_checkpoint_pt(global_step, micro_step, unet, base_checkpoint_path, optimizer, lr_scheduler, scaler, sampler, config):
    output_dir = Path(config.OUTPUT_DIR)
    model_filename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    save_model(output_dir / model_filename, unet, base_checkpoint_path, config.compute_dtype)

    optim_state = optimizer.save_cpu_state() if hasattr(optimizer, 'save_cpu_state') else optimizer.state_dict()
    training_state = {
        'global_step': global_step, 'micro_step': micro_step, 'optimizer_state': optim_state,
        'sampler_seed': sampler.seed, 'sampler_epoch': sampler.epoch,
        'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
        'torch_cpu_state': torch.get_rng_state(),
        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(training_state, output_dir / state_filename)


def consume_force_save_flag(flag_path):
    if not flag_path.exists():
        return False
    try:
        flag_path.unlink()
        return True
    except OSError as e:
        print(f"WARNING: Emergency checkpoint flag found but could not be deleted: {e}")
        return False


def main():
    config = TrainingConfig()
    if config.SEED: set_seed(config.SEED)

    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    force_save_flag = Path(__file__).resolve().with_name("force_save.flag")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_step, micro_step, optimizer_step = 0, 0, 0
    model_to_load = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed, optimizer_state, initial_epoch = config.SEED, None, 0

    if config.RESUME_TRAINING:
        print("\n" + "="*50 + "\n--- RESUMING TRAINING SESSION ---\n")
        training_state = torch.load(Path(config.RESUME_STATE_PATH), map_location="cpu", weights_only=False)
        global_step_saved   = training_state.get('global_step', 0)
        micro_step          = training_state.get('micro_step', global_step_saved * config.GRADIENT_ACCUMULATION_STEPS)
        optimizer_step      = micro_step // config.GRADIENT_ACCUMULATION_STEPS
        initial_sampler_seed = training_state['sampler_seed']
        initial_epoch       = training_state.get('sampler_epoch', 0)
        optimizer_state     = training_state['optimizer_state']
        model_to_load       = Path(config.RESUME_MODEL_PATH)
        if 'random_state'    in training_state: random.setstate(training_state['random_state'])
        if 'numpy_state'     in training_state: np.random.set_state(training_state['numpy_state'])
        if 'torch_cpu_state' in training_state: torch.set_rng_state(training_state['torch_cpu_state'])
        if 'torch_cuda_state' in training_state and training_state['torch_cuda_state'] is not None:
            torch.cuda.set_rng_state(training_state['torch_cuda_state'])
    else:
        mode_str = "RECTIFIED FLOW" if config.is_rectified_flow else "STANDARD SDXL"
        print("\n" + "="*50 + f"\n--- STARTING {mode_str} TRAINING ---\n" + "="*50 + "\n")

    print(f"INFO: Noise type: {getattr(config, 'NOISE_MODE', 'normal')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if check_if_caching_needed(config):
        target_channels = getattr(config, 'VAE_LATENT_CHANNELS', 4)
        conf_shift      = getattr(config, 'VAE_SHIFT_FACTOR', None)
        conf_scale      = getattr(config, 'VAE_SCALING_FACTOR', None)

        vae_source = config.VAE_PATH if (config.VAE_PATH and Path(config.VAE_PATH).exists()) else config.SINGLE_FILE_CHECKPOINT_PATH
        vae_for_caching = load_vae_robust(vae_source, device, target_channels=target_channels)
        vae_for_caching.config.shift_factor   = conf_shift  if conf_shift  is not None else getattr(vae_for_caching.config, 'shift_factor',   None)
        vae_for_caching.config.scaling_factor = conf_scale  if conf_scale  is not None else getattr(vae_for_caching.config, 'scaling_factor', None)
        print(f"INFO: VAE shift={vae_for_caching.config.shift_factor}, scale={vae_for_caching.config.scaling_factor}, channels={vae_for_caching.config.latent_channels}")
        vae_for_caching.enable_tiling()
        vae_for_caching.enable_slicing()

        base_pipe = StableDiffusionXLPipeline.from_single_file(
            config.SINGLE_FILE_CHECKPOINT_PATH, vae=vae_for_caching, unet=None,
            torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        precompute_and_cache_latents(config, base_pipe.tokenizer, base_pipe.tokenizer_2,
                                     base_pipe.text_encoder, base_pipe.text_encoder_2,
                                     vae_for_caching, device)
        del base_pipe, vae_for_caching
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n--- Loading Model ---")
    unet = load_unet_robust(model_to_load, config.compute_dtype)
    gc.collect(); torch.cuda.empty_cache()

    scheduler = None
    if not config.is_rectified_flow:
        prediction_type = getattr(config, "PREDICTION_TYPE", "epsilon")
        print(f"\n--- Using Standard SDXL ({prediction_type}) ---")
        scheduler = DDPMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler"
        )
        scheduler.config.prediction_type = (
            prediction_type if prediction_type == "v_prediction" else "epsilon"
        )
    else:
        print(f"\n--- Using Rectified Flow ---")

    print("\n--- Initializing Dataset ---")
    dataset   = ImageTextLatentDataset(config)
    print_dataset_resolution_sample(dataset, sample_count=5)
    sampler   = BucketBatchSampler(dataset, config.BATCH_SIZE, initial_sampler_seed, shuffle=True)
    sampler.set_epoch(initial_epoch)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)

    unet.enable_gradient_checkpointing()
    unet.to(device)
    set_attention_processor(unet, getattr(config, "MEMORY_EFFICIENT_ATTENTION", "sdpa"))

    exclusion_keywords = getattr(config, "UNET_EXCLUDE_TARGETS", [])
    for name, param in unet.named_parameters():
        should_exclude = any(fnmatch.fnmatch(name, kw if '*' in kw else f"*{kw}*") for kw in exclusion_keywords)
        param.requires_grad = not should_exclude

    total_params     = sum(p.numel() for p in unet.parameters())
    frozen_params    = sum(p.numel() for p in unet.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params
    print(f"\n{'='*50}\nINFO: UNet Parameter Statistics:")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Frozen Parameters:    {frozen_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Percentage Frozen:    {(frozen_params/total_params)*100:.2f}%")
    print("="*50 + "\n")

    params_to_optimize = [{"params": [p for p in unet.parameters() if p.requires_grad], "lr_scale": 1.0}]
    optimizer    = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(optimizer=optimizer, curve_points=config.LR_CUSTOM_CURVE, total_micro_steps=config.MAX_TRAIN_STEPS)
    all_trainable_params = [p for g in params_to_optimize for p in g['params']]

    if config.RESUME_TRAINING:
        if optimizer_state:
            try: optimizer.load_cpu_state(optimizer_state) if hasattr(optimizer, 'load_cpu_state') else optimizer.load_state_dict(optimizer_state)
            except: pass
        lr_scheduler.step(micro_step)

    unet.train()
    diagnostics  = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    reporter     = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="conv_in" if hasattr(unet, 'conv_in') else "first param")
    
    timestep_sampler = TimestepSampler(config, device)
    if config.RESUME_TRAINING and micro_step > 0:
        timestep_sampler.set_current_step(micro_step)

    accumulated_latent_paths  = []
    global_step_times         = deque(maxlen=50)
    optim_step_times          = deque(maxlen=20)
    training_start_time       = time.time()
    last_step_time            = time.time()
    last_optim_step_log_time  = time.time()
    done                      = False
    dataloader_len            = len(dataloader)
    if config.RESUME_TRAINING and dataloader_len > 0:
        batches_to_skip = micro_step % dataloader_len
    else:
        if config.RESUME_TRAINING and dataloader_len <= 0:
            reporter.log_message("WARNING: Resume requested but dataloader is empty; starting without batch skipping.")
        batches_to_skip = 0
    noise_generator = torch.Generator(device=device)

    while not done:
        for batch in dataloader:
            if batches_to_skip > 0: batches_to_skip -= 1; continue
            if micro_step >= config.MAX_TRAIN_STEPS: done = True; break
            if not batch: continue

            micro_step += 1
            diag_data_to_log = None 

            if "latent_path" in batch:
                accumulated_latent_paths.extend(batch["latent_path"])

            latents = batch["latents"].to(device, non_blocking=True).detach()
            embeds  = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled  = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            batch_crop_coords = batch.get("crop_coords", [(0, 0)] * len(batch["latents"]))

            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):

                scaled_sizes = batch.get("scaled_sizes", batch["original_sizes"])
                time_ids_data = [
                    [s1[1], s1[0], crop[0], crop[1], s2[1], s2[0]]
                    for s1, crop, s2 in zip(scaled_sizes, batch_crop_coords, batch["target_sizes"])
                ]
                time_ids = torch.tensor(time_ids_data, dtype=config.compute_dtype, device=device)

                timesteps, first_timestep_int = timestep_sampler.sample(latents.shape[0])
                timestep_str = str(first_timestep_int)
                noise = generate_noise(
                    latents=latents,
                    generator=noise_generator,
                    device=device,
                    dtype=config.compute_dtype,
                    step=micro_step,
                    seed=config.SEED
                )
                if config.is_rectified_flow:
                    jitter = torch.rand(timesteps.shape, device=device, dtype=torch.float32)
                    t_continuous = ((timesteps.float() + jitter) / 1000.0).clamp(0.0, 1.0)

                    t_exp = t_continuous.view(-1, 1, 1, 1)
                    noisy_latents = (1 - t_exp) * latents + t_exp * noise
                    target = noise - latents
                    timesteps_conditioning = (t_continuous * 1000.0)
                else:
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    target = (scheduler.get_velocity(latents, noise, timesteps)
                              if scheduler.config.prediction_type == "v_prediction" else noise)
                    timesteps_conditioning = timesteps

                pred = unet(noisy_latents.to(config.compute_dtype), timesteps_conditioning, embeds,
                            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample

                if getattr(config, "LOSS_TYPE", "MSE") == "PatchMSE" and config.is_rectified_flow:
                    error = pred.float() - target.float()
                    batch_size, _, height, width = error.shape

                    if height <= 1 or width <= 1:
                        loss = error.square().mean()
                    else:
                        min_side = min(height, width)
                        min_crop = max(1, int(min_side * 0.15))
                        max_crop = max(min_crop, int(min_side * 0.75))
                        patch_losses = []

                        for sample_index in range(batch_size):
                            crop_side = int(torch.randint(min_crop, max_crop + 1, (1,), device=error.device).item())
                            top = int(torch.randint(0, height - crop_side + 1, (1,), device=error.device).item())
                            left = int(torch.randint(0, width - crop_side + 1, (1,), device=error.device).item())
                            patch = error[sample_index, :, top:top + crop_side, left:left + crop_side]
                            patch_losses.append(patch.square().mean())

                        loss = torch.stack(patch_losses).mean()
                else:
                    loss = F.mse_loss(pred.float(), target.float())

                (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

            raw_loss_value = loss.detach().item()
            diagnostics.step(raw_loss_value)
            lr_scheduler.step(micro_step)

            if micro_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                raw_grad_norm = (
                    optimizer.clip_grad_norm(config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf'))
                    if isinstance(optimizer, TitanAdamW)
                    else torch.nn.utils.clip_grad_norm_(
                        all_trainable_params,
                        config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float('inf')
                    )
                )
                if isinstance(raw_grad_norm, torch.Tensor): raw_grad_norm = raw_grad_norm.item()
                clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                optim_step_time = time.time() - last_optim_step_log_time
                optim_step_times.append(optim_step_time)
                last_optim_step_log_time = time.time()

                diag_data_to_log = {
                    'optim_step':          optimizer_step,
                    'avg_loss':            diagnostics.get_average_loss(),
                    'current_lr':          optimizer.param_groups[-1]['lr'],
                    'raw_grad_norm':       raw_grad_norm,
                    'clipped_grad_norm':   clipped_grad_norm,
                    'update_delta':        1.0 if raw_grad_norm > 0 else 0.0,
                    'optim_step_time':     optim_step_time,
                    'avg_optim_step_time': sum(optim_step_times) / len(optim_step_times),
                }

                diagnostics.reset()
                accumulated_latent_paths.clear()

                scheduled_save = (
                    config.SAVE_EVERY_N_STEPS > 0 and
                    optimizer_step > 0 and
                    optimizer_step % config.SAVE_EVERY_N_STEPS == 0
                )
                force_save = consume_force_save_flag(force_save_flag)
                if scheduled_save or force_save:
                    reason = "Emergency checkpoint requested" if force_save and not scheduled_save else "Saving checkpoint"
                    reporter.log_message(f"\n--- {reason} at optimizer step {optimizer_step} ---")
                    save_checkpoint_pt(optimizer_step, micro_step, unet, model_to_load,
                                       optimizer, lr_scheduler, None, sampler, config)

            step_duration = time.time() - last_step_time
            global_step_times.append(step_duration)
            last_step_time = time.time()

            reporter.log_step(micro_step, timing_data={
                'raw_step_time': step_duration,
                'elapsed_time':  time.time() - training_start_time,
                'eta':           (config.MAX_TRAIN_STEPS - micro_step) * (sum(global_step_times) / len(global_step_times)) if global_step_times else 0,
                'loss':          raw_loss_value,
                'timestep':      timestep_str,
            }, diag_data=diag_data_to_log)

    reporter.log_message("\nTraining complete.")
    reporter.shutdown()
    save_model(
        OUTPUT_DIR / f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_unified_{str(uuid.uuid4())[:4]}.safetensors",
        unet, model_to_load, config.compute_dtype
    )
    print("All tasks complete. Final model saved.")


if __name__ == "__main__":
    try: multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
