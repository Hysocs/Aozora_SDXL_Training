import inspect
import re
import os
from pathlib import Path
import glob
import gc
import json
from collections import defaultdict, deque
import random
import time
import shutil
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import _LRScheduler
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from safetensors.torch import save_file, load_file
from PIL import Image, TiffImagePlugin, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
from optimizer.raven import Raven
import logging
import warnings
import config as default_config
import multiprocessing
from multiprocessing import Pool, cpu_count
import argparse
import copy
import signal
import numpy as np
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers.optimization import Adafactor

warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True. Gradients will be None")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.buckets = defaultdict(list)
        
        for i, key in enumerate(dataset.bucket_keys):
            if key is not None:
                self.buckets[key].append(i)
        
        self.buckets_list = [b for b in self.buckets.values() if len(b) > 0]
        tqdm.write(f"INFO: Created {len(self.buckets_list)} buckets for training with sizes: {[len(b) for b in self.buckets_list]}")

    def __iter__(self):
        all_batches = []
        for bucket_indices in self.buckets_list:
            if self.shuffle:
                random.shuffle(bucket_indices)
            
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    all_batches.append(batch)
        
        if self.shuffle:
            random.shuffle(all_batches)
            
        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for bucket in self.buckets_list:
            n = len(bucket)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

class CustomCurveScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_curve_points, max_steps, last_epoch=-1):
        if not lr_curve_points or len(lr_curve_points) < 2:
            raise ValueError("LR_CUSTOM_CURVE must contain at least two points.")
            
        # This part remains the same
        self.absolute_points = sorted([[p[0] * max_steps, p[1]] for p in lr_curve_points])
        
        if self.absolute_points[0][0] > 0:
            self.absolute_points.insert(0, [0, self.absolute_points[0][1]])
        if self.absolute_points[-1][0] < max_steps:
             self.absolute_points.append([max_steps, self.absolute_points[-1][1]])

        super().__init__(optimizer, last_epoch)

    # THE KEY CHANGE IS HERE: We now pass the current step directly
    def get_lr(self, current_step=None):
        # If no step is provided, fall back to the old behavior for compatibility
        if current_step is None:
            current_step = self.last_epoch

        # The rest of the logic is the same, just using the new 'current_step' variable
        p1 = None
        p2 = None
        if current_step >= self.absolute_points[-1][0]:
            return [self.absolute_points[-1][1] for _ in self.optimizer.param_groups]

        for i in range(len(self.absolute_points) - 1):
            if self.absolute_points[i][0] <= current_step < self.absolute_points[i+1][0]:
                p1 = self.absolute_points[i]
                p2 = self.absolute_points[i+1]
                break
        
        if p1 is None or p2 is None:
            # This can happen if current_step is exactly the last point
            return [self.absolute_points[-1][1] for _ in self.optimizer.param_groups]

        step1, lr1 = p1
        step2, lr2 = p2

        if step2 == step1:
            return [lr1 for _ in self.optimizer.param_groups]
            
        progress = (current_step - step1) / (step2 - step1)
        current_lr = lr1 + progress * (lr2 - lr1)

        return [current_lr for _ in self.optimizer.param_groups]

    # We also need a new method to manually set the LR
    def step(self, current_step=None):
        if current_step is None:
            # Fallback to default PyTorch behavior if no step is provided
            super().step()
            return

        new_lrs = self.get_lr(current_step=current_step)
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        
        # We still update last_epoch, though it's not used for calculation anymore
        self.last_epoch = current_step
    
class TrainingConfig:
    def __init__(self):
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        parser = argparse.ArgumentParser(description="Load a specific training configuration.")
        parser.add_argument("--config", type=str, help="Path to the user configuration JSON file.")
        args = parser.parse_args()
        user_config_path = args.config
        if user_config_path:
            path = Path(user_config_path)
            if path.exists():
                print(f"INFO: Loading configuration from {path}")
                try:
                    with open(path, 'r') as f:
                        user_config = json.load(f)
                    for key, value in user_config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"ERROR: Could not read or parse {path}: {e}. Using default settings.")
            else:
                print(f"WARNING: Specified config {path} does not exist. Using defaults.")
        else:
            print("INFO: No configuration file specified. Using default settings.")
            
        float_keys = ["CLIP_GRAD_NORM", "MIN_SNR_GAMMA", "IP_NOISE_GAMMA", "COND_DROPOUT_PROB"]
        int_keys = ["MAX_TRAIN_STEPS", "GRADIENT_ACCUMULATION_STEPS", "SEED", "SAVE_EVERY_N_STEPS", "CACHING_BATCH_SIZE", "BATCH_SIZE", "NUM_WORKERS", "TARGET_PIXEL_AREA"]
        str_keys = ["MIN_SNR_VARIANT", "OPTIMIZER_TYPE"]
        bool_keys = [
            "MIRROR_REPEATS", "DARKEN_REPEATS", 
            "RESUME_TRAINING", "USE_PER_CHANNEL_NOISE", "USE_SNR_GAMMA",
            "USE_ZERO_TERMINAL_SNR", "USE_IP_NOISE_GAMMA", "USE_RESIDUAL_SHIFTING", 
            "USE_COND_DROPOUT", "USE_MASKED_TRAINING"
        ]
       
        for key in float_keys:
            if hasattr(self, key):
                val = getattr(self, key)
                try: setattr(self, key, float(val))
                except (ValueError, TypeError): setattr(self, key, default_config.__dict__[key])
        for key in int_keys:
            if hasattr(self, key):
                val = getattr(self, key)
                try: setattr(self, key, int(val))
                except (ValueError, TypeError): setattr(self, key, default_config.__dict__[key])
        for key in str_keys:
            if hasattr(self, key):
                val = getattr(self, key)
                try: setattr(self, key, str(val).lower())
                except (ValueError, TypeError): setattr(self, key, default_config.__dict__[key])
        for key in bool_keys:
            if hasattr(self, key):
                val = getattr(self, key)
                if isinstance(val, str):
                    setattr(self, key, val.lower() in ['true', '1', 't', 'y', 'yes'])
                else:
                    setattr(self, key, bool(val))
        if hasattr(self, 'RAVEN_PARAMS'):
            for param, val in self.RAVEN_PARAMS.items():
                if isinstance(val, list):
                    self.RAVEN_PARAMS[param] = [float(x) for x in val]
                elif param == 'weight_decay' or param == 'eps':
                    self.RAVEN_PARAMS[param] = float(val)
        if hasattr(self, 'ADAFACTOR_PARAMS'):
            for param, val in self.ADAFACTOR_PARAMS.items():
                if param == 'eps':
                    self.ADAFACTOR_PARAMS[param] = [float(x) for x in val]
                elif param in ['clip_threshold', 'decay_rate', 'weight_decay']:
                    self.ADAFACTOR_PARAMS[param] = float(val)
                elif param == 'beta1' and isinstance(val, str) and val.lower() == 'none':
                    self.ADAFACTOR_PARAMS[param] = None

class AspectRatioBucketing:
    def __init__(self, target_area, aspect_ratios, stride=64):
        self.target_area = target_area
        self.stride = stride
        self.bucket_resolutions = self._generate_bucket_resolutions(aspect_ratios)
        print(f"INFO: Initialized {len(self.bucket_resolutions)} aspect ratio buckets for a target area of ~{target_area/1e6:.2f}M pixels.")
        for i, res in enumerate(self.bucket_resolutions): print(f" - Bucket {i}: {res} (AR: {res[0]/res[1]:.2f})")
    def _generate_bucket_resolutions(self, aspect_ratios):
        resolutions = set()
        for aspect_ratio in aspect_ratios:
            h = math.sqrt(self.target_area / aspect_ratio)
            w = h * aspect_ratio
            h = int(h // self.stride) * self.stride
            w = int(w // self.stride) * self.stride
            if h > 0 and w > 0: resolutions.add((w, h))
        return sorted(list(resolutions), key=lambda x: x[0] * x[1], reverse=True)
    def assign_to_bucket(self, width, height):
        aspect_ratio = width / height
        best_bucket = min(self.bucket_resolutions, key=lambda b: abs(b[0]/b[1] - aspect_ratio))
        return best_bucket

def resize_and_crop(image, target_w, target_h):
    img_w, img_h = image.size
    img_aspect = img_w / img_h
    target_aspect = target_w / target_h
    if img_aspect > target_aspect:
        new_h = target_h
        new_w = int(new_h * img_aspect)
    else:
        new_w = target_w
        new_h = int(new_w / img_aspect)
   
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))

def apply_sigmoid_contrast(image, gain=10, cutoff=0.5):
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = 1 / (1 + np.exp(gain * (cutoff - arr)))
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def validate_image_and_assign_bucket(args):
    ip, bucketer = args
    if "_truncated" in ip.stem: return None
    try:
        with Image.open(ip) as img:
            img.verify()
        with Image.open(ip) as img:
            img.load()
            w, h = img.size
        cp = ip.with_suffix('.txt')
        caption = ip.stem.replace('_', ' ')
        if cp.exists():
            with open(cp, 'r', encoding='utf-8') as f: caption = f.read().strip()
       
        bucket_resolution = bucketer.assign_to_bucket(w, h)
       
        return {
            "ip": ip,
            "caption": caption,
            "bucket_resolution": bucket_resolution,
            "original_size": (w, h)
        }
    except Exception as e:
        tqdm.write(f"\n[CORRUPT IMAGE DETECTED] Path: {ip}")
        tqdm.write(f" └─ Reason: {e}")
        try:
            new_name = ip.with_name(f"{ip.stem}_truncated{ip.suffix}")
            ip.rename(new_name)
            if ip.with_suffix('.txt').exists(): ip.with_suffix('.txt').rename(new_name.with_suffix('.txt'))
            tqdm.write(f" └─ Action: Renamed to quarantine.")
        except Exception as rename_e:
            tqdm.write(f" └─ [ERROR] Could not rename file: {rename_e}")
        return None

def validate_wrapper(args):
    return validate_image_and_assign_bucket(args)

def precompute_and_cache_latents(config, tokenizer1, tokenizer2, text_encoder1, text_encoder2, vae, device_for_encoding):
    if not hasattr(config, "INSTANCE_DATASETS") or not config.INSTANCE_DATASETS:
        config.INSTANCE_DATASETS = [{"path": config.INSTANCE_DATA_DIR, "repeats": 1}]
   
    bucketer = AspectRatioBucketing(config.TARGET_PIXEL_AREA, config.BUCKET_ASPECT_RATIOS)
    vae.to(device_for_encoding); text_encoder1.to(device_for_encoding); text_encoder2.to(device_for_encoding)
    vae.enable_tiling()
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
   
    for dataset in config.INSTANCE_DATASETS:
        data_root_path = Path(dataset["path"])
        print(f"Processing dataset: {data_root_path}")
   
        all_image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] 
                   for p in data_root_path.rglob(f"*{ext}") 
                   if "_mask" not in p.stem]
        if not all_image_paths: 
            print(f"WARNING: No images found in {data_root_path}. Skipping.")
            continue
        existing_stems = {p.stem for p in data_root_path.rglob(".precomputed_embeddings_cache/*.pt")}
        images_to_process = [p for p in all_image_paths if p.stem not in existing_stems]
   
        if not images_to_process:
            print("All images are already cached. Nothing to do.")
            continue
        print(f"Found {len(images_to_process)} images to cache. Now validating and assigning to buckets...")
   
        with Pool(processes=cpu_count()) as pool:
            args = [(ip, bucketer) for ip in images_to_process]
            results = list(tqdm(pool.imap(validate_wrapper, args), total=len(args), desc="Validating and bucketing"))
   
        image_metadata = [res for res in results if res]
        if not image_metadata: 
            print(f"WARNING: No valid images found in {data_root_path}. Skipping.")
            continue
        print(f"Validation complete. Found {len(image_metadata)} valid images to cache.")
   
        grouped_metadata = defaultdict(list)
        for meta in image_metadata:
            grouped_metadata[meta["bucket_resolution"]].append(meta)
   
        all_batches_to_process = []
        for bucket_res, metadata_list in grouped_metadata.items():
            for i in range(0, len(metadata_list), config.CACHING_BATCH_SIZE):
                all_batches_to_process.append((bucket_res, metadata_list[i:i + config.CACHING_BATCH_SIZE]))
        random.shuffle(all_batches_to_process)
   
        print("INFO: Caching all variants (original, flipped, contrast, flipped-contrast) for future use.")

        for (target_w, target_h), batch_metadata in tqdm(all_batches_to_process, desc="Caching embeddings in batches"):
            captions_batch = [meta['caption'] for meta in batch_metadata]
            with torch.no_grad():
                prompt_embeds_batch, pooled_prompt_embeds_batch = compute_chunked_text_embeddings(
                    captions_batch, tokenizer1, tokenizer2, text_encoder1, text_encoder2, device_for_encoding
                )
            
            variants = {}
            image_tensors_variants = { "original": [], "contrast": [], "flipped": [], "flipped_contrast": [] }
            
            for meta in batch_metadata:
                try:
                    img = Image.open(meta['ip']).convert('RGB')
                    contrast_img = apply_sigmoid_contrast(img, gain=10)
                    
                    processed_img = resize_and_crop(img, target_w, target_h)
                    image_tensors_variants["original"].append(img_transform(processed_img))
                    image_tensors_variants["flipped"].append(img_transform(processed_img.transpose(Image.FLIP_LEFT_RIGHT)))
                    
                    processed_contrast = resize_and_crop(contrast_img, target_w, target_h)
                    image_tensors_variants["contrast"].append(img_transform(processed_contrast))
                    image_tensors_variants["flipped_contrast"].append(img_transform(processed_contrast.transpose(Image.FLIP_LEFT_RIGHT)))

                except Exception as e:
                    tqdm.write(f"\n[ERROR] Skipping bad image during variant creation {meta['ip']}: {e}")

            with torch.no_grad():
                for key, tensors in image_tensors_variants.items():
                    if tensors:
                        tensor_batch = torch.stack(tensors).to(device_for_encoding, dtype=vae.dtype)
                        latents_batch = vae.encode(tensor_batch).latent_dist.mean * vae.config.scaling_factor
                        variants[key] = latents_batch
                        del tensor_batch
            
            for j, meta in enumerate(batch_metadata):
                image_path = meta['ip']
                cache_dir = image_path.parent / ".precomputed_embeddings_cache"
                cache_dir.mkdir(exist_ok=True)
                save_data = {
                    "original_size_as_tuple": meta["original_size"],
                    "target_size_as_tuple": (target_h, target_w),
                    "prompt_embeds_cpu": prompt_embeds_batch[j].clone().cpu().to(config.compute_dtype),
                    "pooled_prompt_embeds_cpu": pooled_prompt_embeds_batch[j].clone().cpu().to(config.compute_dtype),
                }
                for key, latents in variants.items():
                    if j < len(latents):
                        save_data[key] = { "latents_cpu": latents[j].clone().cpu().to(config.compute_dtype) }
                torch.save(save_data, cache_dir / f"{image_path.stem}.pt")
            
            del prompt_embeds_batch, pooled_prompt_embeds_batch
            for latents in variants.values(): del latents
            gc.collect(); torch.cuda.empty_cache()
   
    vae.disable_tiling()
    vae.cpu(); text_encoder1.cpu(); text_encoder2.cpu(); gc.collect(); torch.cuda.empty_cache()

def compute_chunked_text_embeddings(captions_batch, tokenizer1, tokenizer2, text_encoder1, text_encoder2, device):
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    
    for caption in captions_batch:
        with torch.no_grad():
            text_inputs1 = tokenizer1(caption, padding="max_length", max_length=tokenizer1.model_max_length, truncation=True, return_tensors="pt")
            text_inputs2 = tokenizer2(caption, padding="max_length", max_length=tokenizer2.model_max_length, truncation=True, return_tensors="pt")
            
            prompt_embeds_out1 = text_encoder1(text_inputs1.input_ids.to(device), output_hidden_states=True)
            prompt_embeds1 = prompt_embeds_out1.hidden_states[-2]
            
            prompt_embeds_out2 = text_encoder2(text_inputs2.input_ids.to(device), output_hidden_states=True)
            prompt_embeds2 = prompt_embeds_out2.hidden_states[-2]
            pooled_prompt_embeds2 = prompt_embeds_out2[0]

            prompt_embeds = torch.cat((prompt_embeds1, prompt_embeds2), dim=-1)
        
        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds2)
    
    return torch.cat(prompt_embeds_list), torch.cat(pooled_prompt_embeds_list)

class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.config = config
        if not hasattr(config, "INSTANCE_DATASETS") or not config.INSTANCE_DATASETS:
            config.INSTANCE_DATASETS = [{"path": config.INSTANCE_DATA_DIR, "repeats": 1}]
            
        self.latent_files = []
        all_unique_files = set()
        
        for dataset in config.INSTANCE_DATASETS:
            root = Path(dataset["path"])
            print(f"Loading dataset '{root.name}' with config: {dataset}")
            files = list(root.rglob(".precomputed_embeddings_cache/*.pt"))
            for f in files:
                all_unique_files.add(f)
            
            repeats = dataset.get("repeats", 1)
            should_mirror = dataset.get("mirror_repeats", False)
            should_darken = dataset.get("darken_repeats", False)
            
            self.latent_files.extend([(f, "original", dataset) for f in files])
            
            if repeats > 1:
                for _ in range(repeats - 1):
                    variant_key = "original"
                    if should_darken and should_mirror: variant_key = "flipped_contrast"
                    elif should_darken: variant_key = "contrast"
                    elif should_mirror: variant_key = "flipped"
                    self.latent_files.extend([(f, variant_key, dataset) for f in files])

        self.latent_files = sorted(self.latent_files, key=lambda x: str(x[0]))
        if not self.latent_files:
            raise ValueError("No cached embedding files found. Please ensure pre-caching was successful.")
        
        tqdm.write("INFO: Pre-caching bucket shapes for all latent files...")
        self.bucket_keys = []
        for file_path, _, _ in tqdm(self.latent_files, desc="Caching bucket keys"):
            try:
                full_data = torch.load(file_path, map_location="cpu")
                latents = full_data.get("original", {}).get("latents_cpu")
                if latents is not None:
                    self.bucket_keys.append((latents.shape[1], latents.shape[2]))
                else:
                    tqdm.write(f"WARNING: 'original' latents not found in {file_path}. Marking as invalid.")
                    self.bucket_keys.append(None)
            except Exception as e:
                tqdm.write(f"WARNING: Could not load bucket key for {file_path}: {e}")
                self.bucket_keys.append(None)
        
        print(f"Dataset initialized with {len(self.latent_files)} samples (including repeats).")

    def __len__(self): return len(self.latent_files)

    def __getitem__(self, i):
        file_path, variant_key, dataset_config = self.latent_files[i]
        try:
            full_data = torch.load(file_path, map_location="cpu")
            if variant_key not in full_data or "latents_cpu" not in full_data.get(variant_key, {}):
                tqdm.write(f"Warning: Variant '{variant_key}' not found in {file_path}. Falling back to 'original'.")
                variant_key = "original"
            
            variant_data = full_data[variant_key]
            latents = variant_data["latents_cpu"]

            mask_latent = torch.ones(1, latents.shape[1], latents.shape[2]) 
            focus_factor = dataset_config.get("mask_focus_factor", 1.0)
            focus_mode = dataset_config.get("mask_focus_mode", "Proportional (Multiply)")
            
            if dataset_config.get("use_mask", False):
                mask_dir = Path(dataset_config["path"]) / "masks" 
                mask_path = mask_dir / f"{file_path.stem}_mask.png"
                if mask_path.exists():
                    with Image.open(mask_path).convert("L") as mask_img:
                        if 'flipped' in variant_key:
                            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)
                        
                        resized_mask = mask_img.resize((latents.shape[2], latents.shape[1]), Image.Resampling.NEAREST)
                        mask_tensor = transforms.ToTensor()(resized_mask)
                        mask_latent = (mask_tensor > 0.1).float()

            for key, tensor in [("latents", latents), 
                                ("prompt_embeds", full_data["prompt_embeds_cpu"]), 
                                ("pooled_embeds", full_data["pooled_prompt_embeds_cpu"])]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise ValueError(f"NaN/Inf in {key} for file: {file_path}")

            return {
                "latents_cpu": latents,
                "prompt_embeds_cpu": full_data["prompt_embeds_cpu"].squeeze(0),
                "pooled_prompt_embeds_cpu": full_data["pooled_prompt_embeds_cpu"].squeeze(0),
                "original_size_as_tuple": full_data["original_size_as_tuple"],
                "target_size_as_tuple": full_data["target_size_as_tuple"],
                "mask_latent_cpu": mask_latent,
                "mask_focus_factor_cpu": focus_factor,
                "mask_focus_mode_cpu": focus_mode
            }
        except Exception as e:
            print(f"WARNING: Skipping bad .pt file {file_path}: {e}")
            return None
        
def custom_collate_fn_latent(batch):
    batch = list(filter(None, batch))
    if not batch: return {}
    return {
        "latents": torch.stack([item["latents_cpu"] for item in batch]),
        "prompt_embeds": torch.stack([item["prompt_embeds_cpu"] for item in batch]),
        "pooled_prompt_embeds": torch.stack([item["pooled_prompt_embeds_cpu"] for item in batch]),
        "original_sizes": [item["original_size_as_tuple"] for item in batch],
        "target_sizes": [item["target_size_as_tuple"] for item in batch],
        "masks": torch.stack([item["mask_latent_cpu"] for item in batch]),
        "mask_focus_factors": torch.tensor([item["mask_focus_factor_cpu"] for item in batch]),
        "mask_focus_modes": [item["mask_focus_mode_cpu"] for item in batch],
    }

class TimestepSampler:
    def __init__(self, config, noise_scheduler, device):
        self.variant = config.NOISE_SCHEDULE_VARIANT
        self.num_timesteps = noise_scheduler.config.num_train_timesteps
        self.device = device
        
        if self.variant == "logsnr_laplace":
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device, dtype=torch.float32)
            clipped_alphas = torch.clamp(alphas_cumprod, min=1e-8, max=1 - 1e-8)
            log_snr = torch.log(clipped_alphas) - torch.log(1 - clipped_alphas)
            scale = 1.0  # Changed from 0.3
            self.weights = torch.exp(-torch.abs(log_snr) / scale)
            self.weights /= self.weights.sum()
            print("INFO: Initialized LogSNR Laplace timestep sampler with scale=1.0.")

    def __call__(self, batch_size):
        if self.variant == "residual_shifting":
            min_timestep = int(0.5 * self.num_timesteps)
            return torch.randint(min_timestep, self.num_timesteps, (batch_size,), device=self.device).long()
        elif self.variant == "logsnr_laplace":
            return torch.multinomial(self.weights, num_samples=batch_size, replacement=True).long()
        else:
            return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

class InputPerturbation:
    def __init__(self, config):
        self.is_enabled = config.USE_IP_NOISE_GAMMA and config.IP_NOISE_GAMMA > 0
        if self.is_enabled:
            self.gamma = config.IP_NOISE_GAMMA
            self.use_per_channel = config.USE_PER_CHANNEL_NOISE
            print(f"INFO: Input Perturbation enabled with gamma={self.gamma}, per-channel={self.use_per_channel}.")

    def __call__(self, latents):
        if not self.is_enabled:
            return latents
        
        if self.use_per_channel:
            noise = torch.randn_like(latents)
        else:
            noise = torch.randn_like(latents[:, :1, :, :]).repeat(1, latents.shape[1], 1, 1)
        return latents + self.gamma * noise

class ConditioningDropout:
    def __init__(self, config):
        self.is_enabled = config.USE_COND_DROPOUT
        if self.is_enabled:
            self.prob = config.COND_DROPOUT_PROB
            print(f"INFO: Conditioning Dropout enabled with probability={self.prob}.")

    def __call__(self, prompt_embeds, pooled_embeds):
        if self.is_enabled and random.random() < self.prob:
            return torch.zeros_like(prompt_embeds), torch.zeros_like(pooled_embeds)
        return prompt_embeds, pooled_embeds

class BaseLoss:
    def __call__(self, pred, target, is_v_pred):
        return F.huber_loss(pred.float(), target.float(), reduction="none", delta=2.0)

class MaskedLoss:
    def __init__(self, config):
        self.is_enabled = any(d.get("use_mask", False) for d in config.INSTANCE_DATASETS)
        if self.is_enabled:
            print("INFO: Masked Loss weighting is enabled for at least one dataset.")
            
    def __call__(self, loss, masks, factors, modes):
        if not self.is_enabled or masks is None:
            return loss.mean(dim=list(range(1, len(loss.shape))))

        factors = factors.view(-1, 1, 1, 1).to(masks.device, dtype=loss.dtype)
        weight_map_mul = torch.where(masks == 1, factors, torch.ones_like(masks))
        loss_mul = loss * weight_map_mul
        
        additive_map = (masks - 1) * factors
        loss_add = loss + additive_map

        is_add_mode = torch.tensor([mode.startswith("Uniform") for mode in modes], device=loss.device).view(-1, 1, 1, 1)
        final_loss = torch.where(is_add_mode, loss_add, loss_mul)
        
        return final_loss.mean(dim=list(range(1, len(final_loss.shape))))

class MinSNRLoss:
    def __init__(self, config, noise_scheduler):
        self.is_enabled = config.USE_SNR_GAMMA
        if self.is_enabled:
            self.gamma = config.SNR_GAMMA 
            self.variant = config.MIN_SNR_VARIANT
            self.alphas_cumprod = noise_scheduler.alphas_cumprod.to("cpu")
            print(f"INFO: Min-SNR Loss weighting enabled with gamma={self.gamma}, variant={self.variant}.")
            
    def __call__(self, loss, timesteps):
        if not self.is_enabled:
            return loss.mean(), None

        timesteps_cpu = timesteps.cpu()
        snr = (self.alphas_cumprod[timesteps_cpu] / (1 - self.alphas_cumprod[timesteps_cpu])).to(loss.device)
        snr = torch.clamp(snr, min=1e-8)
        
        if self.variant == "debiased":
            snr_loss_weights = 1 / (snr + 1)
        elif self.variant == "standard":
            snr_loss_weights = torch.clamp(self.gamma / snr, max=1.0)
        elif self.variant == "corrected":
            snr_loss_weights = (self.gamma * snr) / (snr + 1)
        else:
            return loss.mean(), None
            
        final_loss = (loss * snr_loss_weights).mean()
        return final_loss, snr_loss_weights

class FeaturePlugins:
    def __init__(self, config, noise_scheduler, device):
        self.timestep_sampler = TimestepSampler(config, noise_scheduler, device)
        self.input_perturbation = InputPerturbation(config)
        self.conditioning_dropout = ConditioningDropout(config)
        self.base_loss = BaseLoss()
        self.masked_loss = MaskedLoss(config)
        self.min_snr_loss = MinSNRLoss(config, noise_scheduler)
        self.is_v_prediction = noise_scheduler.config.prediction_type == "v_prediction"

class TrainingDiagnostics:
    def __init__(self, accumulation_steps):
        self.accumulation_steps = accumulation_steps
        self.losses = deque(maxlen=accumulation_steps)
        self.grad_norms = deque(maxlen=accumulation_steps)
        self.max_grads = deque(maxlen=accumulation_steps)
        self.timesteps = deque(maxlen=accumulation_steps * 256)

    def step(self, loss, timesteps):
        if loss is not None:
            self.losses.append(loss)
        self.timesteps.extend(timesteps.cpu().numpy())

    def report(self, global_step, lr, trainable_params):
        if not self.losses: return
        
        total_norm, max_grad_val = 0.0, 0.0
        params_with_grad = [p for p in trainable_params if p.grad is not None and torch.isfinite(p.grad).all()]
        
        if params_with_grad:
            for p in params_with_grad:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                current_max = p.grad.data.abs().max().item()
                if current_max > max_grad_val:
                    max_grad_val = current_max
            total_norm = total_norm ** 0.5
        
        avg_loss = sum(self.losses) / len(self.losses)
        vram_gb = torch.cuda.memory_reserved() / 1e9
        
        report_str = (
            f"\n--- Step: {global_step} ---\n"
            f"  Loss:       {avg_loss:<8.5f} | LR: {lr:.2e}\n"
            f"  Grad Norm:  {total_norm:<8.4f} | Max Grad: {max_grad_val:.2e}\n"
            f"  Timesteps:  Min: {min(self.timesteps):<4d} | Mean: {np.mean(self.timesteps):<6.1f} | Max: {max(self.timesteps):<4d}\n"
            f"  VRAM (GB):  {vram_gb:<8.2f}\n"
            f"--------------------"
        )
        tqdm.write(report_str)
        self.reset()

    def reset(self):
        self.losses.clear()
        self.grad_norms.clear()
        self.max_grads.clear()
        self.timesteps.clear()

def filter_scheduler_config(s,c):return{k:v for k,v in s.items() if k in inspect.signature(c.__init__).parameters}
def _generate_hf_to_sd_unet_key_mapping(hf_keys):
    final_map = {}
    for hf_key in hf_keys:
        key = hf_key
        if "resnets" in key:
            new_key = re.sub(r"^down_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"input_blocks.{3*int(m.group(1)) + int(m.group(2)) + 1}.0.", key)
            new_key = re.sub(r"^mid_block\.resnets\.(\d+)\.", lambda m: f"middle_block.{2*int(m.group(1))}.", new_key)
            new_key = re.sub(r"^up_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"output_blocks.{3*int(m.group(1)) + int(m.group(2))}.0.", new_key)
            new_key = new_key.replace("norm1.", "in_layers.0.").replace("conv1.", "in_layers.2.")
            new_key = new_key.replace("norm2.", "out_layers.0.").replace("conv2.", "out_layers.3.")
            new_key = new_key.replace("time_emb_proj.", "emb_layers.1.").replace("conv_shortcut.", "skip_connection.")
            final_map[hf_key] = new_key; continue
        if "attentions" in key:
            new_key = re.sub(r"^down_blocks\.(\d+)\.attentions\.(\d+)\.", lambda m: f"input_blocks.{3*int(m.group(1)) + int(m.group(2)) + 1}.1.", key)
            new_key = re.sub(r"^mid_block\.attentions\.0\.", "middle_block.1.", new_key)
            new_key = re.sub(r"^up_blocks\.(\d+)\.attentions\.(\d+)\.", lambda m: f"output_blocks.{3*int(m.group(1)) + int(m.group(2))}.1.", new_key)
            final_map[hf_key] = new_key; continue
        if "downsamplers" in key:
            new_key = re.sub(r"^down_blocks\.(\d+)\.downsamplers\.0\.conv\.", lambda m: f"input_blocks.{3*(int(m.group(1))+1)}.0.op.", key)
            final_map[hf_key] = new_key; continue
        if "upsamplers" in key:
            new_key = re.sub(r"^up_blocks\.(\d+)\.upsamplers\.0\.", lambda m: f"output_blocks.{3*int(m.group(1)) + 2}.2.", key)
            final_map[hf_key] = new_key; continue
        if key.startswith("conv_in."): final_map[hf_key] = key.replace("conv_in.", "input_blocks.0.0."); continue
        if key.startswith("conv_norm_out."): final_map[hf_key] = key.replace("conv_norm_out.", "out.0."); continue
        if key.startswith("conv_out."): final_map[hf_key] = key.replace("conv_out.", "out.2."); continue
        if key.startswith("time_embedding.linear_1."): final_map[hf_key] = key.replace("time_embedding.linear_1.", "time_embed.0."); continue
        if key.startswith("time_embedding.linear_2."): final_map[hf_key] = key.replace("time_embedding.linear_2.", "time_embed.2."); continue
        if key.startswith("add_embedding.linear_1."): final_map[hf_key] = key.replace("add_embedding.linear_1.", "label_emb.0.0."); continue
        if key.startswith("add_embedding.linear_2."): final_map[hf_key] = key.replace("add_embedding.linear_2.", "label_emb.0.2."); continue
    return final_map

def save_model(base_sd, output_path, unet, trained_unet_param_names, save_dtype):
    """Saves the UNet weights into the base model state_dict and provides a detailed verification report."""
    
    def get_param_category(key_name):
        if 'ff.net' in key_name or 'mlp.fc' in key_name: return "Feed-Forward (ff)"
        if 'attn1' in key_name or 'self_attn' in key_name: return "Self-Attention (attn1)"
        if 'attn2' in key_name or 'cross_attn' in key_name: return "Cross-Attention (attn2)"
        return "Other"

    param_counters = defaultdict(lambda: {'total': 0, 'saved': 0})

    unet_key_map = _generate_hf_to_sd_unet_key_mapping(list(unet.state_dict().keys()))
    print("\nUpdating weights for UNet...")
    
    # Directly get the trained weights from the UNet on the GPU
    model_sd_on_device = unet.state_dict()
   
    # Create a new state dictionary for saving to avoid modifying the original in-memory one.
    # This is still far more memory-efficient than deepcopying the entire base model.
    sd_to_save = copy.deepcopy(base_sd)

    for hf_key in trained_unet_param_names:
        category = get_param_category(hf_key)
        param_counters[category]['total'] += 1

        mapped_part = unet_key_map.get(hf_key)
        if mapped_part:
            sd_key = 'model.diffusion_model.' + mapped_part
            if sd_key in sd_to_save:
                # Move the tensor to CPU and convert dtype before assigning
                sd_to_save[sd_key] = model_sd_on_device[hf_key].to("cpu", dtype=save_dtype)
                param_counters[category]['saved'] += 1
    
    # It's crucial to clean up the GPU-side state dict after use
    del model_sd_on_device
    gc.collect()

    print("\n" + "="*60)
    print(" SAVE MODEL VERIFICATION REPORT")
    print("="*60)
    total_model_params, total_model_saved = 0, 0
    for category, counts in sorted(param_counters.items()):
        saved, total = counts['saved'], counts['total']
        print(f" - {category:<25}: Found {total:>4} -> Saved {saved:>4}")
        if saved < total:
            print(f"   -> WARNING: Skipped {total - saved} params in this category (likely due to key mapping mismatch)!")
        total_model_params += total
        total_model_saved += saved
    print(f" --------------------------------------------------")
    print(f" UNET Summary: {total_model_saved} / {total_model_params} targeted parameters were saved.")
    print("="*60)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Save the newly created state dictionary
    save_file(sd_to_save, output_path)
    print(f"[OK] Save complete: {output_path}")
    
    # Clean up the copied state dict
    del sd_to_save
    gc.collect()


def save_training_checkpoint(config, base_model_state_dict, unet, optimizer, scheduler, step, checkpoint_dir, trainable_param_names, save_dtype):
    print(f"\nSAVING CHECKPOINT AT STEP {step}")

    # --- Step 1: Save optimizer and scheduler state to a file FIRST ---
    state_path = checkpoint_dir / f"training_state_step_{step}.pt"
    torch.save({
        'step': step,
        'optimizer_type': config.OPTIMIZER_TYPE,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, state_path)
    print(f"[OK] Saved optimizer/scheduler state to: {state_path}")

    # --- Step 2: Explicitly delete the optimizer and scheduler to free RAM ---
    # This is the crucial step to prevent OOM.
    del optimizer
    del scheduler
    gc.collect() # Ask the garbage collector to release the memory.
    print("[INFO] Optimizer and scheduler have been released from memory.")

    # --- Step 3: Now, with RAM freed up, save the model weights ---
    checkpoint_model_path = checkpoint_dir / f"checkpoint_step_{step}.safetensors"
    save_model(
        base_sd=base_model_state_dict,
        output_path=checkpoint_model_path,
        unet=unet,
        trained_unet_param_names=trainable_param_names,
        save_dtype=save_dtype
    )

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
    betas = 1 - alphas
    return betas
  
def main():
    config = TrainingConfig()
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    for ds in config.INSTANCE_DATASETS:
        if not Path(ds["path"]).exists(): raise FileNotFoundError(f"Data dir not found: {ds['path']}")
    if not Path(config.SINGLE_FILE_CHECKPOINT_PATH).exists(): raise FileNotFoundError(f"Base model not found: {config.SINGLE_FILE_CHECKPOINT_PATH}")
    
    # --- STEP 1: DEFINE OUR CLOCKS CORRECTLY FROM THE START ---
    # The user-facing "step" is the number of batches processed.
    # We derive the number of optimizer updates from this.
    total_training_steps = config.MAX_TRAIN_STEPS
    total_optimizer_steps = math.ceil(total_training_steps / config.GRADIENT_ACCUMULATION_STEPS)
    print(f"INFO: Training for {total_training_steps} batch steps.")
    print(f"INFO: This will correspond to {total_optimizer_steps} optimizer updates.")

    config.compute_dtype = torch.bfloat16 if config.MIXED_PRECISION == "bfloat16" else torch.float16
    print(f"Using compute dtype: {config.compute_dtype}, Device: {device}")
    use_scaler = config.compute_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print("Loading all components for embedding and latent generation...")
    temp_pipe = StableDiffusionXLPipeline.from_single_file(config.SINGLE_FILE_CHECKPOINT_PATH, torch_dtype=config.compute_dtype, use_safetensors=True)
    precompute_and_cache_latents(config=config, tokenizer1=temp_pipe.tokenizer, tokenizer2=temp_pipe.tokenizer_2, text_encoder1=temp_pipe.text_encoder, text_encoder2=temp_pipe.text_encoder_2, vae=temp_pipe.vae, device_for_encoding=device)
    del temp_pipe; gc.collect(); torch.cuda.empty_cache()

    global_step = 0
    model_to_load = config.SINGLE_FILE_CHECKPOINT_PATH
    latest_state_path = None
    if config.RESUME_TRAINING:
        resume_model_path = Path(config.RESUME_MODEL_PATH) if config.RESUME_MODEL_PATH else None
        resume_state_path = Path(config.RESUME_STATE_PATH) if config.RESUME_STATE_PATH else None
        if resume_model_path and resume_model_path.exists() and resume_state_path and resume_state_path.exists():
            print(f"[OK] Found model: {resume_model_path}\n[OK] Found state: {resume_state_path}")
            model_to_load = str(resume_model_path)
            latest_state_path = resume_state_path
        else:
            print("[WARNING] Resume paths invalid. Reverting to base model.")

    print(f"\nLoading model for training: {model_to_load}")
    pipeline_temp = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=config.compute_dtype, use_safetensors=True)
    unet = pipeline_temp.unet
    scheduler_config = pipeline_temp.scheduler.config
    scheduler_config['prediction_type'] = 'v_prediction'
    print("Loading base model state_dict into memory...")
    base_model_state_dict = load_file(model_to_load)
    del pipeline_temp; gc.collect(); torch.cuda.empty_cache()

    if config.MEMORY_EFFICIENT_ATTENTION == "xformers":
        unet.enable_xformers_memory_efficient_attention()
    else:
        unet.set_attn_processor(AttnProcessor2_0())
    
    unet.to(device).requires_grad_(False)
    unet.enable_gradient_checkpointing()

    print(f"Targeting UNet layers with keywords: {config.UNET_TRAIN_TARGETS}")
    unet_param_names_to_optimize = { name for name, _ in unet.named_parameters() if any(k in name for k in config.UNET_TRAIN_TARGETS) }
    params_to_optimize = [p for n, p in unet.named_parameters() if n in unet_param_names_to_optimize]
    for p in params_to_optimize: p.requires_grad_(True)

    if not params_to_optimize: raise ValueError("No parameters were selected for training.")
    
    if not config.LR_CUSTOM_CURVE:
        raise ValueError("Configuration missing 'LR_CUSTOM_CURVE'.")

    initial_lr = config.LR_CUSTOM_CURVE[0][1]
    optimizer_grouped_parameters = [{"params": params_to_optimize, "lr": initial_lr}]
    
    optimizer_type = config.OPTIMIZER_TYPE.capitalize()
    if optimizer_type == "Raven":
        optimizer = Raven(params=optimizer_grouped_parameters, **config.RAVEN_PARAMS)
    elif optimizer_type == "Adafactor":
        optimizer = Adafactor(params=optimizer_grouped_parameters, **config.ADAFACTOR_PARAMS)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # --- STEP 2: INITIALIZE THE SCHEDULER WITH THE CORRECT TIMELINE ---
    print(f"INFO: LR Scheduler will run for {total_optimizer_steps} optimizer updates.")
    lr_scheduler = CustomCurveScheduler(
        optimizer=optimizer,
        lr_curve_points=config.LR_CUSTOM_CURVE,
        max_steps=total_optimizer_steps # Use the derived optimizer step count
    )

    if latest_state_path:
        print(f"Loading optimizer and scheduler state from {latest_state_path.name}...")
        state = torch.load(latest_state_path, map_location="cpu")
        
        optimizer_step_resumed = state['step']
        global_step = optimizer_step_resumed * config.GRADIENT_ACCUMULATION_STEPS

        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        lr_scheduler.step(current_step=optimizer_step_resumed)

        del state; gc.collect()
        print(f"[OK] Resumed training. Resuming at batch step: {global_step}, Optimizer step: {optimizer_step_resumed}.")

    noise_scheduler = DDPMScheduler(**filter_scheduler_config(scheduler_config, DDPMScheduler))
    if config.USE_ZERO_TERMINAL_SNR:
        noise_scheduler.betas = rescale_zero_terminal_snr(noise_scheduler.betas)
        # Update scheduler internals for consistency
        noise_scheduler.register_to_config(betas=noise_scheduler.betas)
        noise_scheduler._beta_schedule = noise_scheduler.betas  # For lazy updates
        noise_scheduler.alphas = 1.0 - noise_scheduler.betas
        noise_scheduler.alphas_cumprod = torch.cumprod(noise_scheduler.alphas, dim=0)
        noise_scheduler.sqrt_alphas_cumprod = torch.sqrt(noise_scheduler.alphas_cumprod)
        noise_scheduler.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
        noise_scheduler.log_one_minus_alphas_cumprod = torch.log(noise_scheduler.sqrt_one_minus_alphas_cumprod.pow(2))
        noise_scheduler.one_minus_alphas_cumprod = 1 - noise_scheduler.alphas_cumprod  # For velocity, etc.

    plugins = FeaturePlugins(config, noise_scheduler, device)
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)

    train_dataset = ImageTextLatentDataset(config)
    sampler = BucketBatchSampler(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=custom_collate_fn_latent, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    # train_step function remains unchanged
    def train_step(batch):
        latents = batch["latents"]
        noise = torch.randn_like(latents) if config.USE_PER_CHANNEL_NOISE else torch.randn_like(latents[:, :1, :, :]).repeat(1, latents.shape[1], 1, 1)
        timesteps = plugins.timestep_sampler(latents.shape[0])
        perturbed_latents = plugins.input_perturbation(latents)
        noisy_latents = noise_scheduler.add_noise(perturbed_latents, noise, timesteps)
        target = noise_scheduler.get_velocity(perturbed_latents, noise, timesteps) if plugins.is_v_prediction else noise
        prompt_embeds, pooled_embeds = plugins.conditioning_dropout(batch["prompt_embeds"], batch["pooled_prompt_embeds"])
        add_time_ids = torch.cat([torch.tensor(list(s1) + [0,0] + list(s2)).unsqueeze(0) for s1, s2 in zip(batch["original_sizes"], batch["target_sizes"])], dim=0).to(latents.device, dtype=prompt_embeds.dtype)
        with torch.autocast(device_type=latents.device.type, dtype=config.compute_dtype, enabled=use_scaler):
            pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids}).sample
            loss_no_reduction = plugins.base_loss(pred, target, plugins.is_v_prediction)
            loss_per_item = plugins.masked_loss(loss_no_reduction, batch["masks"], batch["mask_focus_factors"], batch["mask_focus_modes"])
            loss, snr_weights = plugins.min_snr_loss(loss_per_item, timesteps)
        if not torch.isfinite(loss):
            tqdm.write(f"\n[WARNING] Detected NaN/Inf loss. Skipping this micro-batch.")
            return None, timesteps
        scaler.scale(loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
        return loss.item(), timesteps

    unet.train()
    
    # --- STEP 3: CONFIGURE THE LOOP AND PROGRESS BAR WITH THE INTUITIVE STEP COUNT ---
    progress_bar = tqdm(range(global_step, total_training_steps), desc="Training Steps", initial=global_step, total=total_training_steps)
    
    done = False
    while not done:
        for batch in train_dataloader:
            if global_step >= total_training_steps:
                done = True; break
            if not batch: continue

            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss_item, timesteps = train_step(batch)
            diagnostics.step(loss_item, timesteps)

            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 10.0)
                
                optimizer_step = (global_step + 1) // config.GRADIENT_ACCUMULATION_STEPS
                
                lr_scheduler.step(current_step=optimizer_step)
                
                current_lr = optimizer.param_groups[0]['lr']
                diagnostics.report(global_step + 1, current_lr, params_to_optimize)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            progress_bar.update(1)
           
            # --- STEP 4: CHECKPOINTING IS NOW ALSO BASED ON THE INTUITIVE BATCH STEP ---
            if global_step > 0 and (global_step % config.SAVE_EVERY_N_STEPS == 0):
                current_optimizer_step = global_step // config.GRADIENT_ACCUMULATION_STEPS
                save_training_checkpoint(
                    config, base_model_state_dict, unet, optimizer, lr_scheduler, 
                    current_optimizer_step, CHECKPOINT_DIR, unet_param_names_to_optimize, config.compute_dtype
                )

                print("\n[INFO] Re-initializing optimizer and scheduler post-checkpointing...")
                
                if optimizer_type == "Raven":
                    optimizer = Raven(params=optimizer_grouped_parameters, **config.RAVEN_PARAMS)
                elif optimizer_type == "Adafactor":
                    optimizer = Adafactor(params=optimizer_grouped_parameters, **config.ADAFACTOR_PARAMS)
                
                state_path = CHECKPOINT_DIR / f"training_state_step_{current_optimizer_step}.pt"
                state = torch.load(state_path, map_location="cpu")
                optimizer.load_state_dict(state['optimizer_state_dict'])

                lr_scheduler = CustomCurveScheduler(
                    optimizer=optimizer, lr_curve_points=config.LR_CUSTOM_CURVE, max_steps=total_optimizer_steps
                )
                lr_scheduler.load_state_dict(state['scheduler_state_dict'])
                lr_scheduler.step(current_step=current_optimizer_step)

                del state; gc.collect()
                print("[OK] Optimizer and scheduler state reloaded.")
                
    progress_bar.close()
    final_optimizer_step = global_step // config.GRADIENT_ACCUMULATION_STEPS
    print(f"--> Training finished at batch step: {global_step}, optimizer step: {final_optimizer_step}")

    # --- NEWLY MODIFIED SECTION WITH DESCRIPTIVE FILENAMES ---
    # Define the base name for the final save, incorporating both step counts.
    file_basename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_final_steps{global_step}_updates{final_optimizer_step}"

    # Save the final optimizer and scheduler state. Its name will now match the model's.
    final_state_path = OUTPUT_DIR / f"{file_basename}_state.pt"
    print(f"\n[INFO] Saving final optimizer and scheduler state to: {final_state_path}")
    torch.save({
        'step': final_optimizer_step,
        'optimizer_type': config.OPTIMIZER_TYPE,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
    }, final_state_path)
    print(f"[OK] Final state saved.")

    # Release memory to prevent OOM on the final model save.
    print("[INFO] Releasing optimizer and scheduler memory before final model save...")
    del optimizer
    del lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()
    print("[OK] Optimizer and scheduler memory released.")

    # Save the final model weights with the new descriptive name.
    final_model_path = OUTPUT_DIR / f"{file_basename}.safetensors"
    
    save_model(base_model_state_dict, final_model_path, unet, unet_param_names_to_optimize, config.compute_dtype)
    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("INFO: Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    os.environ['PYTHONUNBUFFERED'] = '1'
    main()