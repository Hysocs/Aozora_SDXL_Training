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
from torch.optim import AdamW


warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True. Gradients will be None")
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

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.dataset = dataset
        
        self.epoch_indices = []
        for i, (_, _, dataset_config) in enumerate(self.dataset.latent_files):
            self.epoch_indices.append(i)

        tqdm.write(f"INFO: BucketBatchSampler initialized with {len(self.epoch_indices)} total samples for one epoch.")

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.shuffle:
            shuffled_indices = torch.randperm(len(self.epoch_indices), generator=g).tolist()
        else:
            shuffled_indices = list(range(len(self.epoch_indices)))

        buckets = defaultdict(list)
        for i in shuffled_indices:
            bucket_key = self.dataset.bucket_keys[i]
            if bucket_key is not None:
                buckets[bucket_key].append(i)
        
        all_batches = []
        for bucket_key in sorted(buckets.keys()):
            bucket_indices = buckets[bucket_key]
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    all_batches.append(batch)
        
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in perm]
            
        for batch in all_batches:
            yield batch
            
        self.seed += 1

    def __len__(self):
        if self.drop_last:
            return len(self.epoch_indices) // self.batch_size
        else:
            return (len(self.epoch_indices) + self.batch_size - 1) // self.batch_size

class CustomCurveScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_curve_points, max_steps, last_epoch=-1):
        if not lr_curve_points or len(lr_curve_points) < 2:
            raise ValueError("LR_CUSTOM_CURVE must contain at least two points.")
            
        self.absolute_points = sorted([[p[0] * max_steps, p[1]] for p in lr_curve_points])
        
        if self.absolute_points[0][0] > 0:
            self.absolute_points.insert(0, [0, self.absolute_points[0][1]])
        if self.absolute_points[-1][0] < max_steps:
             self.absolute_points.append([max_steps, self.absolute_points[-1][1]])

        super().__init__(optimizer, last_epoch)

    def get_lr(self, current_step=None):
        if current_step is None:
            current_step = self.last_epoch

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
            return [self.absolute_points[-1][1] for _ in self.optimizer.param_groups]

        step1, lr1 = p1
        step2, lr2 = p2

        if step2 == step1:
            return [lr1 for _ in self.optimizer.param_groups]
            
        progress = (current_step - step1) / (step2 - step1)
        current_lr = lr1 + progress * (lr2 - lr1)

        return [current_lr for _ in self.optimizer.param_groups]

    def step(self, current_step=None):
        if current_step is None:
            super().step()
            return

        new_lrs = self.get_lr(current_step=current_step)
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        
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
                        setattr(self, key, value)
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"ERROR: Could not read or parse {path}: {e}. Using default settings.")
            else:
                print(f"WARNING: Specified config {path} does not exist. Using defaults.")
        else:
            print("INFO: No configuration file specified. Using default settings.")
            
        float_keys = ["CLIP_GRAD_NORM", "MIN_SNR_GAMMA", "IP_NOISE_GAMMA", "COND_DROPOUT_PROB","SNR_GAMMA"]
        int_keys = ["MAX_TRAIN_STEPS", "GRADIENT_ACCUMULATION_STEPS", "SEED", "SAVE_EVERY_N_STEPS", "CACHING_BATCH_SIZE", "BATCH_SIZE", "NUM_WORKERS", "TARGET_PIXEL_AREA"]
        str_keys = ["MIN_SNR_VARIANT", "OPTIMIZER_TYPE"]
        bool_keys = [
            "MIRROR_REPEATS", "DARKEN_REPEATS", "RESUME_TRAINING", 
            "USE_SNR_GAMMA", "USE_ZERO_TERMINAL_SNR", 
            "USE_RESIDUAL_SHIFTING", "USE_COND_DROPOUT", "USE_MASKED_TRAINING"
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
            if not isinstance(self.RAVEN_PARAMS, dict):
                print("WARNING: RAVEN_PARAMS from config is not a dictionary. Using defaults.")
                self.RAVEN_PARAMS = default_config.RAVEN_PARAMS

            raven_types = {
                "betas": list, "eps": float, "weight_decay": float,
                "use_grad_centralization": bool, "gc_alpha": float,
            }

            for param, expected_type in raven_types.items():
                val = self.RAVEN_PARAMS.get(param)
                try:
                    if expected_type == bool:
                        self.RAVEN_PARAMS[param] = str(val).lower() in ['true', '1'] if isinstance(val, (str, int)) else bool(val)
                    elif expected_type == float:
                        self.RAVEN_PARAMS[param] = float(val)
                    elif expected_type == list and isinstance(val, list):
                        self.RAVEN_PARAMS[param] = [float(x) for x in val]
                except (ValueError, TypeError):
                    default_val = default_config.RAVEN_PARAMS.get(param)
                    print(f"WARNING: Could not convert RAVEN_PARAMS['{param}']. Using default: {default_val}")
                    self.RAVEN_PARAMS[param] = default_val

        if hasattr(self, 'ADAFACTOR_PARAMS'):
            if not isinstance(self.ADAFACTOR_PARAMS, dict):
                print("WARNING: ADAFACTOR_PARAMS from config is not a dictionary. Using defaults.")
                self.ADAFACTOR_PARAMS = default_config.ADAFACTOR_PARAMS

            ada_types = {
                "eps": list, "clip_threshold": float, "decay_rate": float,
                "beta1": 'optional_float', "weight_decay": float, "scale_parameter": bool,
                "relative_step": bool, "warmup_init": bool
            }

            for param, expected_type in ada_types.items():
                val = self.ADAFACTOR_PARAMS.get(param)
                try:
                    if expected_type == 'optional_float':
                        if val is None or (isinstance(val, str) and val.lower() == 'none'):
                            self.ADAFACTOR_PARAMS[param] = None
                        else:
                            self.ADAFACTOR_PARAMS[param] = float(val)
                    elif expected_type == bool:
                        self.ADAFACTOR_PARAMS[param] = str(val).lower() in ['true', '1'] if isinstance(val, (str, int)) else bool(val)
                    elif expected_type == float:
                        self.ADAFACTOR_PARAMS[param] = float(val)
                    elif expected_type == list and isinstance(val, list):
                        self.ADAFACTOR_PARAMS[param] = [float(x) for x in val]
                except (ValueError, TypeError):
                    default_val = default_config.ADAFACTOR_PARAMS.get(param)
                    print(f"WARNING: Could not convert ADAFACTOR_PARAMS['{param}']. Using default: {default_val}")
                    self.ADAFACTOR_PARAMS[param] = default_val

class ResolutionCalculator:
    def __init__(self, target_area, stride=64):
        self.target_area = target_area
        self.stride = stride
        print(f"INFO: Initialized ResolutionCalculator for a target area of ~{target_area/1e6:.2f}M pixels.")

    def calculate_resolution(self, width, height):
        original_area = width * height
        if original_area > self.target_area:
            aspect_ratio = width / height
            h = math.sqrt(self.target_area / aspect_ratio)
            w = h * aspect_ratio
            w = int(w // self.stride) * self.stride
            h = int(h // self.stride) * self.stride
        else:
            w = int(width // self.stride) * self.stride
            h = int(height // self.stride) * self.stride
        
        if w == 0: w = self.stride
        if h == 0: h = self.stride
        return (w, h)

def resize_to_fit(image, target_w, target_h):
    img_w, img_h = image.size
    
    while img_w > target_w * 2 and img_h > target_h * 2:
        intermediate_w, intermediate_h = img_w // 2, img_h // 2
        if intermediate_w > target_w and intermediate_h > target_h:
            image = image.resize((intermediate_w, intermediate_h), Image.Resampling.BOX)
            img_w, img_h = image.size
        else:
            break

    return image.resize((target_w, target_h), Image.Resampling.BICUBIC)

def apply_sigmoid_contrast(image, gain=10, cutoff=0.5):
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = 1 / (1 + np.exp(gain * (cutoff - arr)))
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def validate_and_assign_resolution(args):
    ip, calculator = args
    if "_truncated" in ip.stem: return None
    try:
        with Image.open(ip) as img:
            img.verify()
        with Image.open(ip) as img:
            img.load()
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                background = Image.new("RGB", img.size, (205, 205, 205))
                background.paste(img, (0, 0), img.split()[-1])
                img = background
            else:
                img = img.convert("RGB")
            w, h = img.size
        cp = ip.with_suffix('.txt')
        caption = ip.stem.replace('_', ' ')
        if cp.exists():
            with open(cp, 'r', encoding='utf-8') as f: caption = f.read().strip()
       
        target_resolution = calculator.calculate_resolution(w, h)
       
        return {
            "ip": ip,
            "caption": caption,
            "target_resolution": target_resolution,
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
    return validate_and_assign_resolution(args)

def precompute_and_cache_latents(config, tokenizer1, tokenizer2, text_encoder1, text_encoder2, vae, device_for_encoding):
    if not hasattr(config, "INSTANCE_DATASETS") or not config.INSTANCE_DATASETS:
        config.INSTANCE_DATASETS = [{"path": config.INSTANCE_DATA_DIR, "repeats": 1}]
   
    calculator = ResolutionCalculator(config.TARGET_PIXEL_AREA)
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
        print(f"Found {len(images_to_process)} images to cache. Now validating and assigning resolutions...")
   
        with Pool(processes=cpu_count()) as pool:
            args = [(ip, calculator) for ip in images_to_process]
            results = list(tqdm(pool.imap(validate_wrapper, args), total=len(args), desc="Validating"))
   
        image_metadata = [res for res in results if res]
        if not image_metadata: 
            print(f"WARNING: No valid images found in {data_root_path}. Skipping.")
            continue
        
        # --- MODIFICATION START: Save metadata to a JSON file ---
        print("INFO: Saving resolution metadata to metadata.json...")
        resolution_metadata = {
            meta['ip'].stem: meta['target_resolution']
            for meta in image_metadata
        }
        metadata_path = data_root_path / "metadata.json"
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                # Convert Path objects to strings for JSON serialization if necessary
                json.dump(resolution_metadata, f, indent=4)
            print(f"[OK] Saved metadata for {len(resolution_metadata)} images to {metadata_path}")
        except Exception as e:
            print(f"[ERROR] Could not save metadata.json: {e}")
        # --- MODIFICATION END ---

        print(f"Validation complete. Found {len(image_metadata)} valid images to cache.")
   
        grouped_metadata = defaultdict(list)
        for meta in image_metadata:
            grouped_metadata[meta["target_resolution"]].append(meta)
   
        all_batches_to_process = []
        for target_res, metadata_list in grouped_metadata.items():
            for i in range(0, len(metadata_list), config.CACHING_BATCH_SIZE):
                all_batches_to_process.append((target_res, metadata_list[i:i + config.CACHING_BATCH_SIZE]))
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
                    with Image.open(meta['ip']) as img:
                        img = img.convert("RGB")

                        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                            background = Image.new("RGB", img.size, (255, 255, 255)) 
                            background.paste(img, (0, 0), img.split()[-1])
                            img = background
                        else:
                            img = img.convert("RGB")

                    contrast_img = apply_sigmoid_contrast(img, gain=10) 
                    
                    processed_img = resize_to_fit(img, target_w, target_h)
                    image_tensors_variants["original"].append(img_transform(processed_img))
                    image_tensors_variants["flipped"].append(img_transform(processed_img.transpose(Image.FLIP_LEFT_RIGHT)))
                    
                    processed_contrast = resize_to_fit(contrast_img, target_w, target_h)
                    image_tensors_variants["contrast"].append(img_transform(processed_contrast))
                    image_tensors_variants["flipped_contrast"].append(img_transform(processed_contrast.transpose(Image.FLIP_LEFT_RIGHT)))

                except Exception as e:
                    tqdm.write(f"\n[ERROR] Skipping bad image during variant creation {meta['ip']}: {e}")

            with torch.no_grad():
                for key, tensors in image_tensors_variants.items():
                    if tensors:
                        tensor_batch = torch.stack(tensors).to(device_for_encoding, dtype=vae.dtype)
                        
                        latents_batch = vae.encode(tensor_batch).latent_dist.mean * vae.config.scaling_factor
                        
                        if torch.isnan(latents_batch).any() or torch.isinf(latents_batch).any():
                            tqdm.write(f"\n[WARNING] NaN/Inf detected post-noising in batch '{key}'. Sanitizing.")
                            latents_batch = torch.nan_to_num(latents_batch, nan=0.0, posinf=0.0, neginf=0.0)

                        variants[key] = latents_batch
                        del tensor_batch
            
            for j, meta in enumerate(batch_metadata):
                image_path = meta['ip']
                cache_dir = image_path.parent / ".precomputed_embeddings_cache"
                cache_dir.mkdir(exist_ok=True)
                save_data = {
                    "original_size_as_tuple": meta["original_size"],
                    "target_size_as_tuple": (target_w, target_h),
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
            
        all_dataset_files = []
        for dataset_config in config.INSTANCE_DATASETS:
            root = Path(dataset_config["path"])
            print(f"Loading dataset '{root.name}' with config: {dataset_config}")
            
            cache_dir = root / ".precomputed_embeddings_cache"
            if not cache_dir.exists() or not any(cache_dir.glob("*.pt")):
                print(f"WARNING: No cached .pt files found in {cache_dir}. Skipping this dataset.")
                continue

            files_in_dataset = sorted(list(cache_dir.glob("*.pt")))
            
            repeats = int(dataset_config.get("repeats", 1))
            should_mirror = dataset_config.get("mirror_repeats", False)
            should_darken = dataset_config.get("darken_repeats", False)
            
            variants_to_use = ["original"]
            if repeats > 1:
                other_variants = []
                if should_mirror: other_variants.append("flipped")
                if should_darken: other_variants.append("contrast")
                if should_mirror and should_darken: other_variants.append("flipped_contrast")
                if other_variants:
                    for i in range(repeats - 1):
                        variants_to_use.append(other_variants[i % len(other_variants)])

            current_dataset_entries = []
            for f in files_in_dataset:
                for i in range(repeats):
                    variant_key = variants_to_use[i % len(variants_to_use)]
                    current_dataset_entries.append((f, variant_key, dataset_config))
            
            all_dataset_files.append(current_dataset_entries)

        self.latent_files = []
        max_len = max(len(d) for d in all_dataset_files)
        for i in range(max_len):
            for dataset_list in all_dataset_files:
                if i < len(dataset_list):
                    self.latent_files.append(dataset_list[i])

        if not self.latent_files:
            raise ValueError("No cached embedding files found across all datasets. Please ensure pre-caching was successful.")
        
        # --- MODIFICATION START: Efficiently load bucket keys from metadata.json ---
        tqdm.write("INFO: Pre-loading bucket keys from metadata.json files...")
        self.bucket_keys = []
        all_metadata = {}

        # Load all metadata from all specified datasets into one dictionary
        for dataset_config in config.INSTANCE_DATASETS:
            root = Path(dataset_config["path"])
            metadata_path = root / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        all_metadata.update(json.load(f))
                except (json.JSONDecodeError, TypeError) as e:
                    tqdm.write(f"WARNING: Could not load or parse {metadata_path}: {e}")
            else:
                tqdm.write(f"WARNING: metadata.json not found for dataset {root}. Bucket info will be missing for its images.")

        # Assign bucket keys by looking up the file stem in the loaded metadata
        for file_path, _, _ in tqdm(self.latent_files, desc="Assigning bucket keys"):
            bucket_key = all_metadata.get(file_path.stem)
            if bucket_key:
                # JSON loads lists, convert back to tuple for dictionary key usage
                self.bucket_keys.append(tuple(bucket_key))
            else:
                tqdm.write(f"WARNING: No metadata found for {file_path.stem}. Marking as invalid for bucketing.")
                self.bucket_keys.append(None)
        # --- MODIFICATION END ---
        
        print(f"Dataset initialized with {len(self.latent_files)} total samples for one scientific epoch after interleaving.")

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
                "mask_focus_mode_cpu": focus_mode,
                "file_path_str": str(file_path.name)
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
        "file_paths": [item["file_path_str"] for item in batch]
    }

class TimestepSampler:
    def __init__(self, config, noise_scheduler, device):
        self.config = config
        self.device = device
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.mode = config.TIMESTEP_CURRICULUM_MODE.lower().replace(" ", "_")
        
        try:
            min_val_str, max_val_str = config.TIMESTEP_CURRICULUM_START_RANGE.split(',')
            self.initial_min = int(min_val_str.strip())
            self.initial_max = int(max_val_str.strip())
        except (ValueError, AttributeError):
            print("WARNING: Could not parse TIMESTEP_CURRICULUM_START_RANGE. Using full range [0, 999].")
            self.initial_min = 0
            self.initial_max = self.num_train_timesteps - 1
        
        self.initial_min = max(0, self.initial_min)
        self.initial_max = min(self.num_train_timesteps - 1, self.initial_max)

        if self.mode == "static_adaptive":
            self._init_static_adaptive_mode()
        elif self.mode == "dynamic_balancing":
            self._init_dynamic_mode()
        else:
            print(f"INFO: Initialized Fixed Timestep Sampling with range: [{self.initial_min}, {self.initial_max}]")

    def _init_static_adaptive_mode(self):
        self.curriculum_duration_steps = int(self.config.MAX_TRAIN_STEPS * (self.config.TIMESTEP_CURRICULUM_END_PERCENT / 100.0))
        print("INFO: Initialized Static Adaptive Timestep Curriculum.")
        print(f"      - Start Range: [{self.initial_min}, {self.initial_max}]")
        print(f"      - Will expand to full range over {self.curriculum_duration_steps} steps.")

    def _init_dynamic_mode(self):
        self.probe_duration_steps = int(self.config.MAX_TRAIN_STEPS * (self.config.DYNAMIC_CHALLENGE_PROBE_PERCENT / 100.0))
        try:
            min_gn_str, max_gn_str = self.config.DYNAMIC_CHALLENGE_TARGET_GRAD_NORM_RANGE.split(',')
            self.target_grad_norm_min = float(min_gn_str.strip())
            self.target_grad_norm_max = float(max_gn_str.strip())
        except (ValueError, AttributeError):
            print("WARNING: Could not parse DYNAMIC_CHALLENGE_TARGET_GRAD_NORM_RANGE. Using defaults [0.25, 0.9].")
            self.target_grad_norm_min = 0.25
            self.target_grad_norm_max = 0.90
        
        self.current_min_ts = self.initial_min
        self.current_max_ts = self.initial_max
        self.adjustment_step_size = max(1, int(self.num_train_timesteps * 0.005)) 

        print("INFO: Initialized Dynamic Challenge Balancing.")
        print(f"      - Probe Phase: {self.probe_duration_steps} steps")
        print(f"      - Target Grad Norm Range: [{self.target_grad_norm_min}, {self.target_grad_norm_max}]")

    def update_range(self, grad_norm, current_step):
        if self.mode != "dynamic_balancing": return

        if current_step <= self.probe_duration_steps:
            if grad_norm < self.target_grad_norm_max:
                self.current_max_ts = min(self.num_train_timesteps - 1, self.current_max_ts + self.adjustment_step_size)
            if grad_norm > self.target_grad_norm_min:
                self.current_min_ts = max(0, self.current_min_ts - self.adjustment_step_size)
        else:
            if grad_norm > self.target_grad_norm_max:
                self.current_max_ts = max(self.current_min_ts, self.current_max_ts - self.adjustment_step_size)
            elif grad_norm < self.target_grad_norm_min:
                self.current_max_ts = min(self.num_train_timesteps - 1, self.current_max_ts + self.adjustment_step_size)
                self.current_min_ts = max(0, self.current_min_ts - self.adjustment_step_size)

    def __call__(self, batch_size, current_step):
        if self.mode == "static_adaptive":
            duration = self.curriculum_duration_steps
            progress = min(1.0, current_step / duration) if duration > 0 else 1.0
            min_ts = int(round(self.initial_min * (1.0 - progress)))
            max_ts = int(round(self.initial_max + (self.num_train_timesteps - 1 - self.initial_max) * progress))
        elif self.mode == "dynamic_balancing":
            min_ts, max_ts = self.current_min_ts, self.current_max_ts
        else:
            min_ts, max_ts = self.initial_min, self.initial_max

        if max_ts < min_ts: max_ts = min_ts
        return torch.randint(min_ts, max_ts + 1, (batch_size,), device=self.device).long()


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
            snr_loss_weights = 1 / (snr + self.gamma)
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
        self.last_grad_norm = 0.75

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
        
        self.last_grad_norm = total_norm
        
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
    def get_param_category(key_name):
        if 'ff.net' in key_name or 'mlp.fc' in key_name: return "Feed-Forward (ff)"
        if 'attn1' in key_name or 'self_attn' in key_name: return "Self-Attention (attn1)"
        if 'attn2' in key_name or 'cross_attn' in key_name: return "Cross-Attention (attn2)"
        return "Other"

    param_counters = defaultdict(lambda: {'total': 0, 'saved': 0})

    unet_key_map = _generate_hf_to_sd_unet_key_mapping(list(unet.state_dict().keys()))
    print("\nUpdating weights for UNet...")
    
    model_sd_on_device = unet.state_dict()
   
    sd_to_save = copy.deepcopy(base_sd)

    for hf_key in trained_unet_param_names:
        category = get_param_category(hf_key)
        param_counters[category]['total'] += 1

        mapped_part = unet_key_map.get(hf_key)
        if mapped_part:
            sd_key = 'model.diffusion_model.' + mapped_part
            if sd_key in sd_to_save:
                sd_to_save[sd_key] = model_sd_on_device[hf_key].to("cpu", dtype=save_dtype)
                param_counters[category]['saved'] += 1
    
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
    save_file(sd_to_save, output_path)
    print(f"[OK] Save complete: {output_path}")
    
    del sd_to_save
    gc.collect()


def save_training_checkpoint(config, base_model_state_dict, unet, optimizer, scheduler, step, checkpoint_dir, trainable_param_names, save_dtype):
    print(f"\nSAVING CHECKPOINT AT STEP {step}")

    state_path = checkpoint_dir / f"training_state_step_{step}.pt"
    torch.save({
        'step': step,
        'optimizer_type': config.OPTIMIZER_TYPE,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, state_path)
    print(f"[OK] Saved optimizer/scheduler state to: {state_path}")

    del optimizer
    del scheduler
    gc.collect()
    print("[INFO] Optimizer and scheduler have been released from memory.")

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
    if config.SEED is not None:
        set_seed(config.SEED)
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    for ds in config.INSTANCE_DATASETS:
        if not Path(ds["path"]).exists(): raise FileNotFoundError(f"Data dir not found: {ds['path']}")
    if not config.SINGLE_FILE_CHECKPOINT_PATH or not Path(config.SINGLE_FILE_CHECKPOINT_PATH).exists():
        if config.RESUME_TRAINING and config.RESUME_MODEL_PATH and Path(config.RESUME_MODEL_PATH).exists():
             pass
        else:
            raise FileNotFoundError(f"Base model not found: {config.SINGLE_FILE_CHECKPOINT_PATH}")

    
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
    if hasattr(temp_pipe, "vae"):
        print("INFO: Enabling VAE slicing and tiling for memory efficiency during caching.")
        temp_pipe.vae.enable_slicing()
        temp_pipe.vae.enable_tiling()
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

    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params_names = {name for name, _ in unet.named_parameters() if any(k in name for k in config.UNET_TRAIN_TARGETS)}
    params_to_optimize = [p for n, p in unet.named_parameters() if n in trainable_params_names]
    trainable_params_count = sum(p.numel() for p in params_to_optimize)
    for p in params_to_optimize: p.requires_grad_(True)
    
    print(f"Targeting UNet layers with keywords: {config.UNET_TRAIN_TARGETS}")
    param_info_str = f"{trainable_params_count/1e6:.2f}M / {total_params/1e6:.2f}M ({trainable_params_count/total_params*100:.2f}%)"
    print(f"GUI_PARAM_INFO::{param_info_str}")

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
    elif optimizer_type == "AdamW":
        optimizer = AdamW(params=optimizer_grouped_parameters)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    print(f"INFO: LR Scheduler will run for {total_optimizer_steps} optimizer updates.")
    lr_scheduler = CustomCurveScheduler(
        optimizer=optimizer,
        lr_curve_points=config.LR_CUSTOM_CURVE,
        max_steps=total_optimizer_steps
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
        noise_scheduler.register_to_config(betas=noise_scheduler.betas)

    class TimestepSampler:
        def __init__(self, config, noise_scheduler, device):
            self.config = config
            self.device = device
            self.num_train_timesteps = noise_scheduler.config.num_train_timesteps
            self.mode = config.TIMESTEP_CURRICULUM_MODE.lower().replace(" ", "_")
            
            try:
                min_val_str, max_val_str = config.TIMESTEP_CURRICULUM_START_RANGE.split(',')
                self.initial_min = int(min_val_str.strip())
                self.initial_max = int(max_val_str.strip())
            except (ValueError, AttributeError):
                print("WARNING: Could not parse TIMESTEP_CURRICULUM_START_RANGE. Using full range [0, 999].")
                self.initial_min = 0
                self.initial_max = self.num_train_timesteps - 1
            
            self.initial_min = max(0, self.initial_min)
            self.initial_max = min(self.num_train_timesteps - 1, self.initial_max)

            if self.mode == "static_adaptive":
                self._init_static_adaptive_mode()
            elif self.mode == "dynamic_balancing":
                self._init_dynamic_mode()
            else:
                print(f"INFO: Initialized Fixed Timestep Sampling with range: [{self.initial_min}, {self.initial_max}]")

        def _init_static_adaptive_mode(self):
            self.curriculum_duration_steps = int(self.config.MAX_TRAIN_STEPS * (self.config.TIMESTEP_CURRICULUM_END_PERCENT / 100.0))
            print("INFO: Initialized Static Adaptive Timestep Curriculum.")
            print(f"      - Start Range: [{self.initial_min}, {self.initial_max}]")
            print(f"      - Will expand to full range over {self.curriculum_duration_steps} steps.")

        def _init_dynamic_mode(self):
            self.probe_duration_steps = int(self.config.MAX_TRAIN_STEPS * (self.config.DYNAMIC_CHALLENGE_PROBE_PERCENT / 100.0))
            try:
                min_gn_str, max_gn_str = self.config.DYNAMIC_CHALLENGE_TARGET_GRAD_NORM_RANGE.split(',')
                self.target_grad_norm_min = float(min_gn_str.strip())
                self.target_grad_norm_max = float(max_gn_str.strip())
            except (ValueError, AttributeError):
                print("WARNING: Could not parse DYNAMIC_CHALLENGE_TARGET_GRAD_NORM_RANGE. Using defaults [0.25, 0.9].")
                self.target_grad_norm_min = 0.25
                self.target_grad_norm_max = 0.90
            
            self.current_min_ts = self.initial_min
            self.current_max_ts = self.initial_max 
            self.adjustment_step_size = max(1, int(self.num_train_timesteps * 0.005)) 

            print("INFO: Initialized Dynamic Challenge Balancing.")
            print(f"      - Probe Phase: {self.probe_duration_steps} steps with oscillating boundary expansion.")
            print(f"      - Target Grad Norm Range: [{self.target_grad_norm_min}, {self.target_grad_norm_max}]")

        def update_range(self, grad_norm, current_step):
            if self.mode != "dynamic_balancing": return

            if current_step <= self.probe_duration_steps:
                if current_step % 2 == 0:
                    if grad_norm < self.target_grad_norm_max:
                        self.current_max_ts = min(self.num_train_timesteps - 1, self.current_max_ts + self.adjustment_step_size)
                else:
                    if grad_norm > self.target_grad_norm_min:
                        self.current_min_ts = max(0, self.current_min_ts - self.adjustment_step_size)
            
            else:
                if grad_norm > self.target_grad_norm_max:
                    self.current_max_ts = max(self.current_min_ts, self.current_max_ts - self.adjustment_step_size)
                elif grad_norm < self.target_grad_norm_min:
                    self.current_max_ts = min(self.num_train_timesteps - 1, self.current_max_ts + self.adjustment_step_size)
                    self.current_min_ts = max(0, self.current_min_ts - self.adjustment_step_size)

        def __call__(self, batch_size, current_step):
            if self.mode == "static_adaptive":
                duration = self.curriculum_duration_steps
                progress = min(1.0, current_step / duration) if duration > 0 else 1.0
                min_ts = int(round(self.initial_min * (1.0 - progress)))
                max_ts = int(round(self.initial_max + (self.num_train_timesteps - 1 - self.initial_max) * progress))
            elif self.mode == "dynamic_balancing":
                min_ts, max_ts = self.current_min_ts, self.current_max_ts
            else:
                min_ts, max_ts = self.initial_min, self.initial_max

            if max_ts < min_ts: max_ts = min_ts
            return torch.randint(min_ts, max_ts + 1, (batch_size,), device=self.device).long()

    class FeaturePlugins:
        def __init__(self, config, noise_scheduler, device):
            self.timestep_sampler = TimestepSampler(config, noise_scheduler, device)
            self.conditioning_dropout = ConditioningDropout(config)
            self.base_loss = BaseLoss()
            self.masked_loss = MaskedLoss(config)
            self.min_snr_loss = MinSNRLoss(config, noise_scheduler)
            self.is_v_prediction = noise_scheduler.config.prediction_type == "v_prediction"

    plugins = FeaturePlugins(config, noise_scheduler, device)
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)

    train_dataset = ImageTextLatentDataset(config)
    sampler = BucketBatchSampler(dataset=train_dataset, batch_size=config.BATCH_SIZE, seed=config.SEED, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=custom_collate_fn_latent, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    def train_step(batch, current_step):
        latents = batch["latents"]

        # --- REPLACEMENT: OFFSET NOISE IMPLEMENTATION ---
        noise = torch.randn_like(latents)
        noise_offset_strength = 0.1
        offset_noise = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
        noise = noise + noise_offset_strength * offset_noise
        # --- END OF REPLACEMENT ---

        timesteps = plugins.timestep_sampler(latents.shape[0], current_step=current_step)
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        target = noise_scheduler.get_velocity(latents, noise, timesteps) if plugins.is_v_prediction else noise
        
        # --- FIX: Changed batch["pooled_embeds"] to the correct key batch["pooled_prompt_embeds"] ---
        prompt_embeds, pooled_embeds = plugins.conditioning_dropout(batch["prompt_embeds"], batch["pooled_prompt_embeds"])
        
        add_time_ids = torch.cat([torch.tensor(list(s1) + [0,0] + list(s2)).unsqueeze(0) for s1, s2 in zip(batch["original_sizes"], batch["target_sizes"])], dim=0).to(latents.device, dtype=prompt_embeds.dtype)
        
        with torch.autocast(device_type=latents.device.type, dtype=config.compute_dtype, enabled=use_scaler):
            pred = unet(
                noisy_latents.to(config.compute_dtype),
                timesteps,
                prompt_embeds.to(config.compute_dtype),
                added_cond_kwargs={
                    "text_embeds": pooled_embeds.to(config.compute_dtype),
                    "time_ids": add_time_ids.to(config.compute_dtype)
                }
            ).sample
            
            loss_no_reduction = plugins.base_loss(pred.float(), target.float(), plugins.is_v_prediction)
            loss_per_item = plugins.masked_loss(loss_no_reduction, batch["masks"], batch["mask_focus_factors"], batch["mask_focus_modes"])
            loss, snr_weights = plugins.min_snr_loss(loss_per_item, timesteps)
        
        if not torch.isfinite(loss):
            tqdm.write(f"\n[WARNING] Detected NaN/Inf loss. Skipping this micro-batch.")
            return None, timesteps
        
        scaler.scale(loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
        return loss.item(), timesteps

    unet.train()
    
    progress_bar = tqdm(range(global_step, total_training_steps), desc="Training Steps", initial=global_step, total=total_training_steps)
    
    done = False
    while not done:
        for batch in train_dataloader:
            if global_step >= total_training_steps:
                done = True; break
            if not batch: continue

            if global_step > 1:
                 tqdm.write(f"Step {global_step}: Processing file: {batch['file_paths'][0]}")

            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss_item, timesteps = train_step(batch, current_step=global_step)
            diagnostics.step(loss_item, timesteps)

            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                if config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, config.CLIP_GRAD_NORM)
                
                optimizer_step = (global_step + 1) // config.GRADIENT_ACCUMULATION_STEPS
                
                lr_scheduler.step(current_step=optimizer_step)
                
                current_lr = optimizer.param_groups[0]['lr']
                diagnostics.report(global_step + 1, current_lr, params_to_optimize)

                plugins.timestep_sampler.update_range(diagnostics.last_grad_norm, global_step)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            progress_bar.update(1)
           
            if global_step > 0 and (global_step % config.SAVE_EVERY_N_STEPS == 0):
                current_optimizer_step = global_step // config.GRADIENT_ACCUMULATION_STEPS
                save_training_checkpoint(
                    config, base_model_state_dict, unet, optimizer, lr_scheduler, 
                    current_optimizer_step, CHECKPOINT_DIR, trainable_params_names, config.compute_dtype
                )

                print("\n[INFO] Re-initializing optimizer and scheduler post-checkpointing...")
                
                if optimizer_type == "Raven":
                    optimizer = Raven(params=optimizer_grouped_parameters, **config.RAVEN_PARAMS)
                elif optimizer_type == "Adafactor":
                    optimizer = Adafactor(params=optimizer_grouped_parameters, **config.ADAFACTOR_PARAMS)
                elif optimizer_type == "AdamW":
                    optimizer = AdamW(params=optimizer_grouped_parameters)

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

    file_basename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_final_steps{global_step}_updates{final_optimizer_step}"

    final_state_path = OUTPUT_DIR / f"{file_basename}_state.pt"
    print(f"\n[INFO] Saving final optimizer and scheduler state to: {final_state_path}")
    torch.save({
        'step': final_optimizer_step,
        'optimizer_type': config.OPTIMIZER_TYPE,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
    }, final_state_path)
    print(f"[OK] Final state saved.")

    print("[INFO] Releasing optimizer and scheduler memory before final model save...")
    del optimizer
    del lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()
    print("[OK] Optimizer and scheduler memory released.")

    final_model_path = OUTPUT_DIR / f"{file_basename}.safetensors"
    
    save_model(base_model_state_dict, final_model_path, unet, trainable_params_names, config.compute_dtype)
    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("INFO: Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    os.environ['PYTHONUNBUFFERED'] = '1'
    main()