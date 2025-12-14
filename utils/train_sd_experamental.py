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
from optimizer.raven import RavenAdamW

warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- CHANGE 1: ADDED FOR REPRODUCIBILITY ---
# These settings ensure that CUDA convolution operations are deterministic.
# This can have a slight performance cost but is essential for reproducible results.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -----------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    base_noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
    if config.NOISE_TYPE == "Offset":
        strength = float(getattr(config, "NOISE_OFFSET", 0.1))
        if strength <= 0:
            return base_noise
        b, c, h, w = base_noise.shape
        offset = torch.randn(b, 1, 1, 1, device=base_noise.device, dtype=base_noise.dtype)
        return base_noise + strength * offset
    else:
        return base_noise

def map_training_step_to_noise_level(timesteps, max_steps=1000):
    """
    Maps Virtual Timesteps (0-2000) to Scheduler Noise Levels (0-1000).
    0-1000 (Standard): Maps 1:1 to Noise Level.
    1000-2000 (Inverse): Maps (t-1000) to Noise Level.
    """
    mapped_steps = torch.zeros_like(timesteps)
    
    # Phase A: Standard (0-1000)
    mask_std = timesteps < max_steps
    if mask_std.any():
        mapped_steps[mask_std] = timesteps[mask_std]

    # Phase B: Inverse (1000-2000)
    mask_inv = timesteps >= max_steps
    if mask_inv.any():
        mapped_steps[mask_inv] = timesteps[mask_inv] - max_steps

    # Clamp safely to max_steps - 1 for array indexing
    return torch.clamp(mapped_steps, 0, max_steps - 1)

class TrainingConfig:
    def __init__(self):
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        self._load_from_user_config()
        self._type_check_and_correct()
        self.compute_dtype = torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16
        
        # --- Config for Extended Range ---
        if not hasattr(self, 'VIRTUAL_TIMESTEP_MAX'):
            self.VIRTUAL_TIMESTEP_MAX = 2000 # 0-1000 Standard, 1000-2000 Inverse
        if not hasattr(self, 'STANDARD_TIMESTEP_MAX'):
            self.STANDARD_TIMESTEP_MAX = 1000 # The split point
        if not hasattr(self, 'INVERSE_LOSS_SCALE'):
            self.INVERSE_LOSS_SCALE = 1.0
        if not hasattr(self, 'CLIP_GRAD_NORM'):
            self.CLIP_GRAD_NORM = 1.0
        if not hasattr(self, 'INVERSE_TASK_PROBABILITY'):
            self.INVERSE_TASK_PROBABILITY = 0.25
            
    def _load_from_user_config(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str)
        args, _ = parser.parse_known_args()
        if args.config:
            path = Path(args.config)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        user_config = json.load(f)
                    for key, value in user_config.items():
                        setattr(self, key, value)
                except Exception:
                    pass

    def _type_check_and_correct(self):
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
            except Exception:
                setattr(self, key, default_value)

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
        step_info = f" [Loss: {loss_val:.4f}, TS: {timestep_val}]"
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
            vram_reserved_gb = torch.cuda.memory_reserved() / 1e9
            vram_allocated_gb = torch.cuda.memory_allocated() / 1e9
            report_str = (
                f"--- Optimizer Step: {diag_data['optim_step']:<5} | Loss: {diag_data['avg_loss']:<8.5f} | LR: {diag_data['current_lr']:.2e} ---\n"
                f"  Time: {diag_data['optim_step_time']:.2f}s/step | Avg Speed: {diag_data['avg_optim_step_time']:.2f}s/step\n"
                f"  Grad Norm (Raw/Clipped): {diag_data['raw_grad_norm']:<8.4f} / {diag_data['clipped_grad_norm']:<8.4f}\n"
                f"  VRAM: Training={vram_reserved_gb:.2f}GB | Model={vram_allocated_gb:.2f}GB"
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

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = list(range(self.total_images))
        if self.batch_size == 1:
            if self.shuffle:
                indices = torch.randperm(len(indices), generator=g).tolist()
            batches = [[i] for i in indices]
        else:
            if self.shuffle:
                indices = torch.randperm(len(indices), generator=g).tolist()
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
            if w <= 0 or h <= 0: return None
        cp = ip.with_suffix('.txt')
        if cp.exists():
            with open(cp, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            if not caption: return None
        else:
            caption = ip.stem.replace('_', ' ')
        return {"ip": ip, "caption": caption, "target_resolution": calculator.calculate_resolution(w, h), "original_size": (w, h)}
    except Exception:
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
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists(): continue
        image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        if not image_paths: continue
        cache_dir = root / ".precomputed_embeddings_cache"
        if not cache_dir.exists(): return True
        cached_stems = {p.stem for p in cache_dir.rglob("*.pt")}
        uncached = [p for p in image_paths if p.stem not in cached_stems]
        if uncached: return True
    return False

def load_vae_only(config, device):
    vae_path = config.VAE_PATH
    if not vae_path or not Path(vae_path).exists(): return None
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)
    return vae.to(device)

def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    if not check_if_caching_needed(config): return
    calc = ResolutionCalculator(config.TARGET_PIXEL_AREA, stride=64, should_upscale=config.SHOULD_UPSCALE, max_area_tolerance=getattr(config, 'MAX_AREA_TOLERANCE', 1.1))
    vae.enable_tiling()
    te1.to(device)
    te2.to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in root.rglob(f"*{ext}")]
        if not paths: continue
        cache_dir = root / ".precomputed_embeddings_cache"
        cache_dir.mkdir(exist_ok=True)
        stems = {p.stem for p in cache_dir.rglob("*.pt")}
        to_process = [p for p in paths if p.stem not in stems]
        if not to_process: continue
        with Pool(processes=min(cpu_count(), 8)) as pool:
            results = list(tqdm(pool.imap(validate_and_assign_resolution, [(p, calc) for p in to_process]), total=len(to_process)))
        metadata = [r for r in results if r]
        if not metadata: continue
        res_meta = {m['ip'].stem: m['target_resolution'] for m in metadata}
        with open(root / "metadata.json", 'w', encoding='utf-8') as f: json.dump(res_meta, f, indent=4)
        grouped = defaultdict(list)
        for m in metadata: grouped[m["target_resolution"]].append(m)
        batches = [(res, grouped[res][i:i + config.CACHING_BATCH_SIZE]) for res in grouped for i in range(0, len(grouped[res]), config.CACHING_BATCH_SIZE)]
        random.shuffle(batches)
        for batch_idx, ((w, h), batch_meta) in enumerate(tqdm(batches)):
            captions = [m['caption'] for m in batch_meta]
            embeds, pooled = compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device)
            images, valid_meta = [], []
            for m in batch_meta:
                try:
                    with Image.open(m['ip']) as img: img = fix_alpha_channel(img)
                    processed_img = transform(resize_to_fit(img, w, h))
                    images.append(processed_img)
                    valid_meta.append(m)
                except Exception: pass
            if not images: continue
            image_batch_tensor = torch.stack(images).to(device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(image_batch_tensor).latent_dist.sample() * vae.config.scaling_factor
            for j, m in enumerate(valid_meta):
                cache_path = cache_dir / f"{m['ip'].stem}.pt"
                torch.save({"original_size": m["original_size"], "target_size": (w, h), "embeds": embeds[j].cpu().to(torch.float32), "pooled": pooled[j].cpu().to(torch.float32), "latents": latents[j].cpu().to(torch.float32)}, cache_path)
            if (batch_idx + 1) % 10 == 0: torch.cuda.empty_cache()
    te1.cpu(); te2.cpu(); gc.collect(); torch.cuda.empty_cache()

class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.latent_files = []
        self.use_semantic_loss = (getattr(config, 'LOSS_TYPE', 'Default') == "Semantic")
        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / ".precomputed_embeddings_cache"
            if not cache_dir.exists(): continue
            files = list(cache_dir.glob("*.pt"))
            self.latent_files.extend(files * int(ds.get("repeats", 1)))
        if not self.latent_files: raise ValueError("No cached files found.")
        
        # --- CHANGE 2: REMOVED FOR REPRODUCIBILITY ---
        # This line shuffled the dataset in-memory differently on every run,
        # causing non-deterministic data loading. The BucketBatchSampler
        # handles shuffling deterministically using a seed, so this is not needed.
        # random.shuffle(self.latent_files)
        # ----------------------------------------------
        
        self.bucket_keys = []
        all_meta = {}
        for ds in config.INSTANCE_DATASETS:
            meta_path = Path(ds["path"]) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f: all_meta.update(json.load(f))
        for f in self.latent_files:
            key = all_meta.get(f.stem)
            self.bucket_keys.append(tuple(key) if key else None)

    def __len__(self): return len(self.latent_files)

    def _find_original_image_path(self, latent_path):
        latent_p = Path(latent_path)
        stem = latent_p.stem
        parent_dir = latent_p.parent.parent
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            image_path = parent_dir / (stem + ext)
            if image_path.exists(): return image_path
        return None

    def __getitem__(self, i):
        try:
            latent_path = self.latent_files[i]
            data = torch.load(latent_path, map_location="cpu")
            if (torch.isnan(data["latents"]).any() or torch.isinf(data["latents"]).any()): return None
            item_data = {
                "latents": data["latents"], "embeds": data["embeds"], "pooled": data["pooled"],
                "original_sizes": data["original_size"], "target_sizes": data["target_size"],
                "latent_path": str(latent_path)
            }
            if self.use_semantic_loss:
                original_image_path = self._find_original_image_path(latent_path)
                if original_image_path:
                    with Image.open(original_image_path) as raw_img:
                        item_data["original_image"] = fix_alpha_channel(raw_img)
                else:
                    item_data["original_image"] = None
            return item_data
        except Exception:
            return None

class TimestepSampler:
    def __init__(self, config, noise_scheduler, device):
        self.config = config
        self.device = device

        # Get the boundaries from the config
        self.standard_max = config.STANDARD_TIMESTEP_MAX      # e.g., 1000
        self.virtual_max = config.VIRTUAL_TIMESTEP_MAX        # e.g., 2000

        # The new, superior logic: A direct probability for the coin flip
        # This value (0.0 to 1.0) controls the percentage of training steps
        # that will be dedicated to the inverse "noising" task.
        self.inverse_prob = getattr(config, 'INVERSE_TASK_PROBABILITY', 0.5)

        print("INFO: Initialized Timestep Sampler with 'Biased Coin Flip' logic.")
        print(f"      - Probability of an Inverse Task (1000-2000): {self.inverse_prob:.0%}")
        print(f"      - Probability of a Standard Task (0-1000):   {1.0 - self.inverse_prob:.0%}")
        if self.inverse_prob == 1.0:
            print("      >>> Training with 100% Inverse Steps. <<<")

    def sample(self, batch_size):
        """
        Samples timesteps using a two-step process:
        1. A biased "coin flip" determines if the step is Standard or Inverse.
        2. A uniform random sample is drawn from within the chosen range.
        """
        # 1. Perform the biased coin flip for the entire batch
        rand_flips = torch.rand(batch_size, device=self.device)
        is_inverse_step = rand_flips < self.inverse_prob

        # Prepare the output tensor
        timesteps = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        # 2. Sample uniformly within each chosen range
        
        # For all steps that are NOT inverse (the "tails" of the coin flip)
        num_standard = (~is_inverse_step).sum()
        if num_standard > 0:
            # Pick random integers between 0 and 999
            standard_steps = torch.randint(0, self.standard_max, (num_standard,), device=self.device)
            timesteps[~is_inverse_step] = standard_steps

        # For all steps that ARE inverse (the "heads" of the coin flip)
        num_inverse = is_inverse_step.sum()
        if num_inverse > 0:
            # Pick random integers between 1000 and 1999
            inverse_steps = torch.randint(self.standard_max, self.virtual_max, (num_inverse,), device=self.device)
            timesteps[is_inverse_step] = inverse_steps
            
        return timesteps

    def update(self, raw_grad_norm):
        # This method is no longer needed for the new logic.
        pass

    def record_timesteps(self, timesteps):
        # This method is no longer needed for the new logic.
        pass

def custom_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    return {k: (torch.stack([item[k] for item in batch]) if isinstance(batch[0][k], torch.Tensor) else [item[k] for item in batch]) for k in batch[0]}

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

def create_optimizer(config, params_to_optimize):
    optimizer_type = config.OPTIMIZER_TYPE.lower()
    if optimizer_type == "raven":
        curve_points = getattr(config, 'LR_CUSTOM_CURVE', [])
        initial_lr = max(point[1] for point in curve_points) if curve_points else config.LEARNING_RATE
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
        raise ValueError(f"Unsupported optimizer type: '{config.OPTIMIZER_TYPE}'.")

def save_model(output_path, unet, base_model_state_dict, key_map, trainable_layer_names):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unet.to('cpu')
    trained_unet_state_dict = unet.state_dict()
    final_state_dict = base_model_state_dict.copy()
    for hf_key in trainable_layer_names:
        mapped_key = key_map.get(hf_key)
        if mapped_key is None: continue
        sd_key = 'model.diffusion_model.' + mapped_key
        if sd_key in final_state_dict:
            final_state_dict[sd_key] = trained_unet_state_dict[hf_key]
    save_file(final_state_dict, output_path)
    unet.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def save_checkpoint(global_step, unet, base_model_state_dict, key_map, trainable_layer_names, optimizer, lr_scheduler, scaler, sampler, config):
    output_dir = Path(config.OUTPUT_DIR)
    model_filename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    save_model(output_dir / model_filename, unet, base_model_state_dict, key_map, trainable_layer_names)
    training_state = {
        'global_step': global_step,
        'optimizer_state': optimizer.save_cpu_state(),
        'sampler_seed': sampler.seed,
        'scaler_state_dict': scaler.state_dict() if scaler else None
    }
    torch.save(training_state, output_dir / state_filename)

def main():
    config = TrainingConfig()
    if config.SEED: set_seed(config.SEED)
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    global_step = 0
    model_to_load = Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    initial_sampler_seed = config.SEED
    optimizer_state = None

    if config.RESUME_TRAINING:
        state_path = Path(config.RESUME_STATE_PATH)
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        global_step = training_state['global_step']
        initial_sampler_seed = training_state['sampler_seed']
        optimizer_state = training_state['optimizer_state']
        model_to_load = Path(config.RESUME_MODEL_PATH)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if check_if_caching_needed(config):
        base_pipe = StableDiffusionXLPipeline.from_single_file(config.SINGLE_FILE_CHECKPOINT_PATH, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        vae_for_caching = load_vae_only(config, device) or base_pipe.vae.to(device)
        precompute_and_cache_latents(config, base_pipe.tokenizer, base_pipe.tokenizer_2, base_pipe.text_encoder, base_pipe.text_encoder_2, vae_for_caching, device)
        del base_pipe, vae_for_caching; gc.collect(); torch.cuda.empty_cache()

    base_model_state_dict = load_file(model_to_load, device="cpu")
    pipe = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=config.compute_dtype, low_cpu_mem_usage=True)
    unet = pipe.unet
    original_scheduler_config = pipe.scheduler.config
    del pipe; gc.collect(); torch.cuda.empty_cache()
    
    SCHEDULER_MAP = {"DDPMScheduler": DDPMScheduler, "DDIMScheduler": DDIMScheduler, "EulerDiscreteScheduler": EulerDiscreteScheduler}
    scheduler_name = getattr(config, 'NOISE_SCHEDULER', 'DDPMScheduler').replace(" (Experimental)", "")
    scheduler_class = SCHEDULER_MAP.get(scheduler_name, DDPMScheduler)
    training_scheduler_config = {**original_scheduler_config, 'prediction_type': config.PREDICTION_TYPE, 'beta_schedule': config.BETA_SCHEDULE}
    noise_scheduler = scheduler_class.from_config(filter_scheduler_config(training_scheduler_config, scheduler_class))

    if getattr(config, 'USE_ZERO_TERMINAL_SNR', False):
        new_betas = rescale_zero_terminal_snr(noise_scheduler.betas)
        noise_scheduler.betas = new_betas
        noise_scheduler.alphas_cumprod = torch.cumprod(1.0 - new_betas, dim=0)

    dataset = ImageTextLatentDataset(config)
    sampler = BucketBatchSampler(dataset, config.BATCH_SIZE, initial_sampler_seed, shuffle=True)
    
    # --- CHANGE 3: MODIFIED FOR REPRODUCIBILITY ---
    # Using multiple workers (num_workers > 0) introduces OS-level scheduling randomness,
    # making batch order non-deterministic. Setting num_workers=0 ensures data is loaded
    # sequentially, which is crucial for reproducibility.
    # Note: This will make data loading slower. For performance, you can revert this
    # to `config.NUM_WORKERS` but will lose deterministic training.
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=0)
    # ---------------------------------------------

    unet.to(device).enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention() if config.MEMORY_EFFICIENT_ATTENTION == "xformers" else unet.set_attn_processor(AttnProcessor2_0())
    
    exclusion_keywords = config.UNET_EXCLUDE_TARGETS
    trainable_layer_names = []
    
    for name, param in unet.named_parameters():
        should_exclude = any(k in name for k in exclusion_keywords)
        if should_exclude:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_layer_names.append(name)

    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    optimizer = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(optimizer=optimizer, curve_points=config.LR_CUSTOM_CURVE, max_train_steps=config.MAX_TRAIN_STEPS)

    if config.RESUME_TRAINING:
        if optimizer_state: optimizer.load_cpu_state(optimizer_state)
        lr_scheduler.step(global_step)

    unet.train()
    key_map = _generate_hf_to_sd_unet_key_mapping(list(unet.state_dict().keys()))
    
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    reporter = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="Cycle Check")
    timestep_sampler = TimestepSampler(config, noise_scheduler, device)

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
                done = True; break
            if not batch: continue

            if "latent_path" in batch: accumulated_latent_paths.extend(batch["latent_path"])
            
            latents = batch["latents"].to(device, non_blocking=True)
            embeds = batch["embeds"].to(device, non_blocking=True, dtype=config.compute_dtype)
            pooled = batch["pooled"].to(device, non_blocking=True, dtype=config.compute_dtype)
            original_images = batch.get("original_image")
            
            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):
                time_ids_list = []
                for s1, s2 in zip(batch["original_sizes"], batch["target_sizes"]):
                    time_id = torch.tensor(list(s1) + [0,0] + list(s2), dtype=torch.float32)
                    time_ids_list.append(time_id.unsqueeze(0).to(device, dtype=config.compute_dtype))
                time_ids = torch.cat(time_ids_list, dim=0)
                
                # --- NEW HYBRID LOGIC 2.0 (VIRTUAL 0-2000) ---
                timesteps = timestep_sampler.sample(latents.shape[0])
                timestep_sampler.record_timesteps(timesteps)
                
                # Generate Base Noise
                noise = generate_train_noise(latents, config)
                
                # Masks for Splitting
                split_step = config.STANDARD_TIMESTEP_MAX # 1000
                mask_std = timesteps < split_step  # 0-1000: Standard
                mask_inv = timesteps >= split_step # 1000-2000: Inverse
                
                loss_val = torch.tensor(0.0, device=device, dtype=config.compute_dtype)
                
                # 1. Standard Denoising Phase (0 < t < 1000)
                if mask_std.any():
                    # Standard Noising Process
                    noisy_latents_std = noise_scheduler.add_noise(latents[mask_std], noise[mask_std], timesteps[mask_std])
                    
                    # Standard Prediction
                    pred_std = unet(noisy_latents_std, timesteps[mask_std], embeds[mask_std], 
                                   added_cond_kwargs={"text_embeds": pooled[mask_std], "time_ids": time_ids[mask_std]}).sample
                    
                    if config.PREDICTION_TYPE == "v_prediction":
                        target_std = noise_scheduler.get_velocity(latents[mask_std], noise[mask_std], timesteps[mask_std])
                    else:
                        target_std = noise[mask_std]
                        
                    loss_std = F.mse_loss(pred_std.float(), target_std.float(), reduction="mean")
                    loss_val += loss_std

                # 2. Inverse "Noising" Phase (1000 <= t <= 2000)
                if mask_inv.any():
                    # Map 1000-2000 to 0-1000 Noise Levels
                    # 1000 -> 0 (Clean)
                    # 2000 -> 1000 (Max Noise)
                    mapped_noise_steps = map_training_step_to_noise_level(timesteps[mask_inv], max_steps=noise_scheduler.config.num_train_timesteps)
                    
                    # Create Target: What the model SHOULD produce (The noisy latent)
                    target_noisy_latents = noise_scheduler.add_noise(latents[mask_inv], noise[mask_inv], mapped_noise_steps)
                    
                    # Input: The clean latents (Learning to degrade)
                    # Note: We pass the HIGH timestep (e.g. 1500) to the UNet so it knows this is "inverse mode"
                    pred_inv = unet(latents[mask_inv], timesteps[mask_inv], embeds[mask_inv], 
                                   added_cond_kwargs={"text_embeds": pooled[mask_inv], "time_ids": time_ids[mask_inv]}).sample
                    
                    # Loss: Compare Predicted Noisy Latent vs Actual Noisy Latent
                    raw_inv_loss = F.mse_loss(pred_inv.float(), target_noisy_latents.float(), reduction="mean")
                    
                    # Scale Loss
                    loss_val += (raw_inv_loss * config.INVERSE_LOSS_SCALE)

                loss = loss_val

            accumulation_steps = config.GRADIENT_ACCUMULATION_STEPS
            loss = loss / accumulation_steps
            loss = loss.to(dtype=config.compute_dtype)
            loss.backward()

            diagnostics.step(loss.item() * accumulation_steps)

            diag_data_to_log = None
            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(params_to_optimize, float('inf')).item()
                timestep_sampler.update(raw_grad_norm)
                
                clip_val = getattr(config, 'CLIP_GRAD_NORM', 1.0)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, clip_val)
                clipped_grad_norm = min(raw_grad_norm, clip_val)

                optimizer.step()
                lr_scheduler.step(global_step + 1)
                optimizer.zero_grad(set_to_none=True)

                optim_step_time = time.time() - last_optim_step_log_time
                optim_step_times.append(optim_step_time)
                last_optim_step_log_time = time.time()
                
                diag_data_to_log = {
                    'optim_step': (global_step + 1) // config.GRADIENT_ACCUMULATION_STEPS,
                    'avg_loss': diagnostics.get_average_loss(),
                    'current_lr': optimizer.param_groups[0]['lr'],
                    'raw_grad_norm': raw_grad_norm,
                    'clipped_grad_norm': clipped_grad_norm,
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
                'loss': loss.item() * accumulation_steps, 'timestep': timesteps[0].item()
            }
            
            reporter.log_step(global_step, timing_data=timing_data, diag_data=diag_data_to_log)
            global_step += 1
            
            if config.SAVE_EVERY_N_STEPS > 0 and global_step % config.SAVE_EVERY_N_STEPS == 0:
                 save_checkpoint(global_step, unet, base_model_state_dict, key_map, trainable_layer_names, optimizer, lr_scheduler, None, sampler, config)

    reporter.shutdown()
    output_path = OUTPUT_DIR / f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_final.safetensors"
    save_model(output_path, unet, base_model_state_dict, key_map, trainable_layer_names)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()