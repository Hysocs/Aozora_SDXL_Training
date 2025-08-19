import os
import inspect
import re
from pathlib import Path
import glob
import gc
import json
from collections import defaultdict
import imghdr
import random
import time
import shutil
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from safetensors.torch import save_file, load_file
from PIL import Image, TiffImagePlugin, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    Adafactor,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import logging
import warnings
import config as default_config
import multiprocessing
from multiprocessing import Pool, cpu_count
import argparse
import copy # <--- ADD THIS IMPORT

# --- Global Settings ---
warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True. Gradients will be None")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

class TrainingConfig:
    def __init__(self):
        # First, load all default values
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        
        # --- MODIFIED SECTION START ---
        # Use argparse to accept a configuration file path from the command line
        parser = argparse.ArgumentParser(description="Load a specific training configuration.")
        parser.add_argument("--config", type=str, help="Path to the user configuration JSON file.")
        args, _ = parser.parse_known_args()

        user_config_path = Path(args.config) if args.config else None

        if user_config_path and user_config_path.exists():
            print(f"INFO: Loading user configuration from {user_config_path}")
            try:
                with open(user_config_path, 'r') as f:
                    user_config = json.load(f)
                # Override defaults with values from the specified config file
                for key, value in user_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR: Could not read or parse {user_config_path}: {e}. Using default settings.")
        else:
            # Fallback to look for the default user config if no argument is passed
            default_user_path = Path("user_config.json")
            if default_user_path.exists():
                 print(f"INFO: No --config specified. Found and loaded default {default_user_path}")
                 try:
                    with open(default_user_path, 'r') as f:
                        user_config = json.load(f)
                    for key, value in user_config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                 except (json.JSONDecodeError, TypeError) as e:
                    print(f"ERROR: Could not read or parse {default_user_path}: {e}. Using default settings.")
            else:
                 print("INFO: No configuration file specified or found. Using default settings.")
        # --- MODIFIED SECTION END ---

        # Type conversion for numeric values
        float_keys = ["UNET_LEARNING_RATE", "LR_WARMUP_PERCENT", "CLIP_GRAD_NORM", "MIN_SNR_GAMMA"]
        int_keys = ["MAX_TRAIN_STEPS", "GRADIENT_ACCUMULATION_STEPS", "SEED", "SAVE_EVERY_N_STEPS", "CACHING_BATCH_SIZE", "BATCH_SIZE", "NUM_WORKERS", "TARGET_PIXEL_AREA"]
        
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

class DataPrefetcher:
    """Overlaps host→GPU transfer of the next batch with the GPU work on the current batch."""
    def __init__(self, loader, device, dtype):
        self.loader = iter(loader)
        self.device = device
        self.dtype = dtype
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            if self.next_batch:
                # This dict comprehension will move all tensor values to the GPU asynchronously
                self.next_batch = {
                    k: v.to(self.device, dtype=self.dtype if v.is_floating_point() else None, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in self.next_batch.items()
                }

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            # We record the event to ensure that the tensor is not used before it's ready
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream())
            self.preload()
        return batch

class AspectRatioBucketing:
    """Creates and manages buckets based on aspect ratio and total pixel area."""
    def __init__(self, target_area, aspect_ratios, stride=64):
        self.target_area = target_area
        self.stride = stride
        self.bucket_resolutions = self._generate_bucket_resolutions(aspect_ratios)
        print(f"INFO: Initialized {len(self.bucket_resolutions)} aspect ratio buckets for a target area of ~{target_area/1e6:.2f}M pixels.")
        for i, res in enumerate(self.bucket_resolutions): print(f"  - Bucket {i}: {res} (AR: {res[0]/res[1]:.2f})")

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
    """Resizes an image to fit a target resolution, then center-crops it."""
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

def validate_image_and_assign_bucket(args):
    """Worker to validate a single image, extract metadata, and assign it to a bucket."""
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
        tqdm.write(f"  └─ Reason: {e}")
        try:
            new_name = ip.with_name(f"{ip.stem}_truncated{ip.suffix}")
            ip.rename(new_name)
            if ip.with_suffix('.txt').exists(): ip.with_suffix('.txt').rename(new_name.with_suffix('.txt'))
            tqdm.write(f"  └─ Action: Renamed to quarantine.")
        except Exception as rename_e:
            tqdm.write(f"  └─ [ERROR] Could not rename file: {rename_e}")
        return None

def validate_wrapper(args):
    return validate_image_and_assign_bucket(args)

def precompute_and_cache_latents(config, tokenizer1, tokenizer2, text_encoder1, text_encoder2, vae, device_for_encoding):
    """Caches VAE latents and text embeddings using aspect ratio bucketing."""
    data_root_path = Path(config.INSTANCE_DATA_DIR)
    
    if config.FORCE_RECACHE_LATENTS:
        print("Force recache enabled. Deleting old cache directories...")
        for cache_dir in data_root_path.rglob(".precomputed_embeddings_cache"):
            if cache_dir.is_dir(): shutil.rmtree(cache_dir)

    print("Scanning for images recursively...")
    all_image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in data_root_path.rglob(f"*{ext}")]
    if not all_image_paths: raise ValueError(f"No images found in {config.INSTANCE_DATA_DIR}")

    existing_stems = {p.stem for p in data_root_path.rglob(".precomputed_embeddings_cache/*.pt")}
    images_to_process = [p for p in all_image_paths if p.stem not in existing_stems] if not config.FORCE_RECACHE_LATENTS else all_image_paths
    
    if not images_to_process:
        print("All images are already cached. Nothing to do.")
        vae.cpu(); text_encoder1.cpu(); text_encoder2.cpu(); gc.collect(); torch.cuda.empty_cache()
        return

    print(f"Found {len(images_to_process)} images to cache. Now validating and assigning to buckets...")
    bucketer = AspectRatioBucketing(config.TARGET_PIXEL_AREA, config.BUCKET_ASPECT_RATIOS)
    
    with Pool(processes=cpu_count()) as pool:
        args = [(ip, bucketer) for ip in images_to_process]
        results = list(tqdm(pool.imap(validate_wrapper, args), total=len(args), desc="Validating and bucketing"))
    
    image_metadata = [res for res in results if res]
    if not image_metadata: raise ValueError("No valid images found after scanning.")

    print(f"Validation complete. Found {len(image_metadata)} valid images to cache.")
    
    grouped_metadata = defaultdict(list)
    for meta in image_metadata:
        grouped_metadata[meta["bucket_resolution"]].append(meta)
    
    all_batches_to_process = []
    for bucket_res, metadata_list in grouped_metadata.items():
        for i in range(0, len(metadata_list), config.CACHING_BATCH_SIZE):
            all_batches_to_process.append((bucket_res, metadata_list[i:i + config.CACHING_BATCH_SIZE]))
    random.shuffle(all_batches_to_process)
    
    vae.to(device_for_encoding); text_encoder1.to(device_for_encoding); text_encoder2.to(device_for_encoding)
    vae.enable_tiling()

    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    for (target_w, target_h), batch_metadata in tqdm(all_batches_to_process, desc="Caching embeddings in batches"):
        image_tensors = []
        captions_batch = [meta['caption'] for meta in batch_metadata]

        for meta in batch_metadata:
            try:
                img = Image.open(meta['ip']).convert('RGB')
                processed_img = resize_and_crop(img, target_w, target_h)
                image_tensors.append(img_transform(processed_img))
            except Exception as e:
                tqdm.write(f"\n[ERROR] Skipping bad image {meta['ip']}: {e}")

        if not image_tensors: continue
            
        img_tensor_batch = torch.stack(image_tensors).to(device_for_encoding, dtype=vae.dtype)
        
        with torch.no_grad():
            text_inputs1 = tokenizer1(captions_batch, padding="max_length", max_length=tokenizer1.model_max_length, truncation=True, return_tensors="pt")
            text_inputs2 = tokenizer2(captions_batch, padding="max_length", max_length=tokenizer2.model_max_length, truncation=True, return_tensors="pt")
            
            prompt_embeds_out = text_encoder1(text_inputs1.input_ids.to(device_for_encoding), output_hidden_states=True)
            pooled_prompt_embeds_out = text_encoder2(text_inputs2.input_ids.to(device_for_encoding), output_hidden_states=True)
            
            prompt_embeds_batch = torch.cat([prompt_embeds_out.hidden_states[-2], pooled_prompt_embeds_out.hidden_states[-2]], dim=-1)
            pooled_prompt_embeds_batch = pooled_prompt_embeds_out.text_embeds
            
            latents_batch = vae.encode(img_tensor_batch).latent_dist.mean * vae.config.scaling_factor
            if torch.isnan(latents_batch).any() or torch.isinf(latents_batch).any():
                raise ValueError("NaN or Inf detected in latents—bad encoding!")

        for j, meta in enumerate(batch_metadata):
            image_path = meta['ip']
            cache_dir = image_path.parent / ".precomputed_embeddings_cache"
            cache_dir.mkdir(exist_ok=True)
            
            torch.save({
                "latents_cpu": latents_batch[j].clone().cpu().to(config.compute_dtype),
                "prompt_embeds_cpu": prompt_embeds_batch[j].clone().cpu().to(config.compute_dtype),
                "pooled_prompt_embeds_cpu": pooled_prompt_embeds_batch[j].clone().cpu().to(config.compute_dtype),
                "original_size_as_tuple": meta["original_size"],
                "target_size_as_tuple": (target_h, target_w),
            }, cache_dir / f"{image_path.stem}.pt")
            
        del img_tensor_batch, latents_batch, prompt_embeds_batch, pooled_prompt_embeds_batch
        gc.collect(); torch.cuda.empty_cache()
        
    vae.disable_tiling()
    vae.cpu(); text_encoder1.cpu(); text_encoder2.cpu(); gc.collect(); torch.cuda.empty_cache()

class ImageTextLatentDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.latent_files = sorted(list(self.data_root.rglob(".precomputed_embeddings_cache/*.pt")))
        if not self.latent_files:
            raise ValueError(f"No cached embedding files found in subdirectories of {self.data_root}. Please ensure pre-caching was successful.")
        print(f"Dataset initialized with {len(self.latent_files)} pre-cached samples from all subdirectories.")
    def __len__(self): return len(self.latent_files)
    def __getitem__(self, i):
        try:
            data = torch.load(self.latent_files[i], map_location="cpu")
            if torch.isnan(data["latents_cpu"]).any() or torch.isinf(data["latents_cpu"]).any():
                raise ValueError("NaN/Inf in latents")
            return data
        except Exception as e:
            print(f"WARNING: Skipping bad .pt file {self.latent_files[i]}: {e}")
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
    }

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
    """Saves the model by updating a provided in-memory state dictionary."""
    unet_key_map = _generate_hf_to_sd_unet_key_mapping(list(unet.state_dict().keys()))
    param_counters = defaultdict(lambda: {'total': 0, 'saved': 0})

    def get_param_category(key_name):
        if 'ff.net' in key_name or 'mlp.fc' in key_name: return "Feed-Forward (ff)"
        if 'attn1' in key_name or 'self_attn' in key_name: return "Self-Attention (attn1)"
        if 'attn2' in key_name or 'cross_attn' in key_name: return "Cross-Attention (attn2)"
        return "Other"

    print("\nUpdating weights for UNet...")
    model_sd_on_device = unet.state_dict()
    
    for hf_key in list(trained_unet_param_names):
        category = get_param_category(hf_key)
        param_counters[category]['total'] += 1
        mapped_part = unet_key_map.get(hf_key)
        if mapped_part:
            sd_key = 'model.diffusion_model.' + mapped_part
            if sd_key in base_sd:
                base_sd[sd_key] = model_sd_on_device[hf_key].to("cpu", dtype=save_dtype)
                param_counters[category]['saved'] += 1
    del model_sd_on_device; gc.collect()

    print("\n" + "="*60); print("           SAVE MODEL VERIFICATION REPORT"); print("="*60)
    total_model_params, total_model_saved = 0, 0
    for category, counts in sorted(param_counters.items()):
        saved, total = counts['saved'], counts['total']
        print(f"  - {category:<25}: Found {total:>4} -> Saved {saved:>4}")
        if saved < total: print(f"    -> WARNING: Skipped {total - saved} params!")
        total_model_params += total; total_model_saved += saved
    print(f"  --------------------------------------------------")
    print(f"  UNET Summary: {total_model_saved} / {total_model_params} parameters saved.")
    print("="*60)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(base_sd, output_path)
    print(f"[OK] Save complete: {output_path}")
    gc.collect(); torch.cuda.empty_cache()
    # No return needed as we operated on a copy


def has_nan_in_state(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, dict):
            if has_nan_in_state(value): return True
        elif isinstance(value, torch.Tensor):
            if torch.isnan(value).any() or torch.isinf(value).any(): return True
    return False

def sanitize_state(state_dict):
    for key, value in state_dict.items():
        if isinstance(value, dict):
            sanitize_state(value)
        elif isinstance(value, torch.Tensor):
            state_dict[key] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    return state_dict

def save_training_checkpoint(base_model_state_dict, unet, optimizer, scheduler, step, checkpoint_dir, trainable_param_names, save_dtype):
    checkpoint_model_path = checkpoint_dir / f"checkpoint_step_{step}.safetensors"
    print(f"\nSAVING CHECKPOINT AT STEP {step}")
    
    # --- CHANGE 1: Use a deepcopy of the state dict for saving ---
    # This prevents modifying the original state dict in memory, avoiding potential corruption.
    save_model(
        base_sd=copy.deepcopy(base_model_state_dict), # Create a fresh copy for the save operation
        output_path=checkpoint_model_path,
        unet=unet,
        trained_unet_param_names=trainable_param_names,
        save_dtype=save_dtype
    )
    
    opt_state = optimizer.state_dict()
    sched_state = scheduler.state_dict()
    
    opt_has_nan = has_nan_in_state(opt_state)
    sched_has_nan = has_nan_in_state(sched_state)
    
    if opt_has_nan or sched_has_nan:
        print("WARNING: NaN/Inf detected in optimizer or scheduler state during save! Sanitizing...")
        if opt_has_nan:
            opt_state = sanitize_state(opt_state)
        if sched_has_nan:
            sched_state = sanitize_state(sched_state)
    
    state_path = checkpoint_dir / f"training_state_step_{step}.pt"
    torch.save({
        'step': step,
        'optimizer_state_dict': opt_state,
        'scheduler_state_dict': sched_state,
    }, state_path)
    print(f"[OK] Saved optimizer/scheduler state to: {state_path}")
    
    # --- AGGRESSIVE CLEANUP ---
    del opt_state, sched_state
    gc.collect()
    torch.cuda.empty_cache()
    # --- END AGGRESSIVE CLEANUP ---

    # --- CHANGE 2: No need to return the state dict anymore ---
    # The original state_dict was never modified.

    
def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Rescales betas to have zero terminal SNR from https://arxiv.org/pdf/2305.08891.pdf"""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar / F.pad(alphas_bar[:-1], (1, 0), value=1.0)
    betas = 1 - alphas
    return betas
   
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = TrainingConfig()
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(config.INSTANCE_DATA_DIR).exists(): raise FileNotFoundError(f"Data dir not found: {config.INSTANCE_DATA_DIR}")
    if not Path(config.SINGLE_FILE_CHECKPOINT_PATH).exists(): raise FileNotFoundError(f"Base model not found: {config.SINGLE_FILE_CHECKPOINT_PATH}")
    config.compute_dtype = torch.bfloat16 if config.MIXED_PRECISION == "bfloat16" else torch.float16
    print(f"Using compute dtype: {config.compute_dtype}, Device: {device}")
    
    use_scaler = config.compute_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    
    print("Loading all components for embedding and latent generation...")
    temp_pipe = StableDiffusionXLPipeline.from_single_file(config.SINGLE_FILE_CHECKPOINT_PATH, torch_dtype=config.compute_dtype, use_safetensors=True)
    precompute_and_cache_latents(
        config=config,
        tokenizer1=temp_pipe.tokenizer,
        tokenizer2=temp_pipe.tokenizer_2,
        text_encoder1=temp_pipe.text_encoder,
        text_encoder2=temp_pipe.text_encoder_2,
        vae=temp_pipe.vae,
        device_for_encoding=device,
    )
    del temp_pipe; gc.collect(); torch.cuda.empty_cache()

    global_step = 0
    model_to_load = config.SINGLE_FILE_CHECKPOINT_PATH
    latest_state_path = None
    if config.RESUME_TRAINING:
        print("Resume training enabled. Checking provided paths...")
        resume_model_path = Path(config.RESUME_MODEL_PATH) if config.RESUME_MODEL_PATH else None
        resume_state_path = Path(config.RESUME_STATE_PATH) if config.RESUME_STATE_PATH else None
        if resume_model_path and resume_model_path.exists() and resume_state_path and resume_state_path.exists():
            print(f"[OK] Found model checkpoint: {resume_model_path}\n[OK] Found training state: {resume_state_path}")
            model_to_load = str(resume_model_path); latest_state_path = resume_state_path
        else:
            print("[WARNING] Resume training is ON, but one or more paths are invalid. Reverting to base model.")
    
    print(f"\nLoading model for training: {model_to_load}")
    pipeline_temp = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=config.compute_dtype, use_safetensors=True)
    unet = pipeline_temp.unet
    scheduler_config = pipeline_temp.scheduler.config
    scheduler_config['prediction_type'] = 'v_prediction'
    # Load the base model state dict into CPU memory *once*
    print("Loading base model state_dict into memory...")
    base_model_state_dict = load_file(model_to_load)
    del pipeline_temp.vae, pipeline_temp.tokenizer, pipeline_temp.tokenizer_2, pipeline_temp.text_encoder, pipeline_temp.text_encoder_2, pipeline_temp
    gc.collect(); torch.cuda.empty_cache()
    
    if hasattr(unet, 'enable_xformers_memory_efficient_attention'):
        try:
            # Your proven fastest method
            unet.enable_xformers_memory_efficient_attention()
            print("INFO: xFormers memory-efficient attention is enabled.")
        except Exception as e:
            # Fallback to the PyTorch 2.0 implementation if xFormers is not available
            print(f"WARNING: Could not enable xFormers. Error: {e}. Falling back to PyTorch 2.0's SDPA.")
            try:
                from diffusers.models.attention_processor import AttnProcessor2_0
                unet.set_attn_processor(AttnProcessor2_0())
                print("INFO: PyTorch 2.0's SDPA is enabled.")
            except Exception as e2:
                print(f"WARNING: Could not enable any attention optimization. Training will be slower. Error: {e2}")

    unet.to(device).requires_grad_(False)
    if hasattr(unet, 'enable_gradient_checkpointing'): unet.enable_gradient_checkpointing()
    
    print(f"Targeting UNet layers with keywords: {config.UNET_TRAIN_TARGETS}")
    unet_param_names_to_optimize = { name for name, _ in unet.named_parameters() if any(k in name for k in config.UNET_TRAIN_TARGETS) }
    params_to_optimize = [p for n, p in unet.named_parameters() if n in unet_param_names_to_optimize]
    for p in params_to_optimize: p.requires_grad_(True)
    
    if not params_to_optimize:
        raise ValueError("No parameters were selected for training. Please specify valid UNET_TRAIN_TARGETS.")

    optimizer_grouped_parameters = [{"params": params_to_optimize, "lr": config.UNET_LEARNING_RATE}]
    unet_trainable_params = sum(p.numel() for p in params_to_optimize)
    unet_total_params_count = sum(p.numel() for p in unet.parameters())
    print(f"Trainable UNet params: {unet_trainable_params/1e6:.3f}M / {unet_total_params_count/1e6:.3f}M")
    print(f"GUI_PARAM_INFO::{unet_trainable_params/1e6:.3f}M / {unet_total_params_count/1e6:.3f}M UNet params | LR: {config.UNET_LEARNING_RATE:.2e} | Steps: {config.MAX_TRAIN_STEPS} | Batch: {config.BATCH_SIZE}x{config.GRADIENT_ACCUMULATION_STEPS} (Effective: {config.BATCH_SIZE*config.GRADIENT_ACCUMULATION_STEPS})")
    
    optimizer = Adafactor(optimizer_grouped_parameters, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, weight_decay=0.0, scale_parameter=False, relative_step=False)

    num_update_steps = math.ceil(config.MAX_TRAIN_STEPS / config.GRADIENT_ACCUMULATION_STEPS)
    num_warmup_update_steps = math.ceil(num_update_steps * config.LR_WARMUP_PERCENT)

    lr_scheduler_class = get_linear_schedule_with_warmup if config.LR_SCHEDULER_TYPE == "linear" else get_cosine_schedule_with_warmup
    lr_scheduler = lr_scheduler_class(optimizer=optimizer, num_warmup_steps=num_warmup_update_steps, num_training_steps=num_update_steps)
    
    if latest_state_path:
        print(f"Loading optimizer and scheduler state from {latest_state_path.name}...")
        state = torch.load(latest_state_path, map_location=device)
        
        opt_has_nan = has_nan_in_state(state['optimizer_state_dict'])
        sched_has_nan = has_nan_in_state(state['scheduler_state_dict'])
        
        if opt_has_nan or sched_has_nan:
            print("WARNING: NaN/Inf detected in loaded optimizer or scheduler state! Sanitizing...")
            if opt_has_nan:
                state['optimizer_state_dict'] = sanitize_state(state['optimizer_state_dict'])
            if sched_has_nan:
                state['scheduler_state_dict'] = sanitize_state(state['scheduler_state_dict'])
        
        global_step = state['step']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        del state; gc.collect(); torch.cuda.empty_cache()
        print(f"[OK] Resumed training from step {global_step}.")

    noise_scheduler = DDPMScheduler(**filter_scheduler_config(scheduler_config, DDPMScheduler))
    if config.USE_ZERO_TERMINAL_SNR:
        noise_scheduler.betas = rescale_zero_terminal_snr(noise_scheduler.betas)
    train_dataset = ImageTextLatentDataset(config.INSTANCE_DATA_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_latent, num_workers=config.NUM_WORKERS, persistent_workers=False, pin_memory=True)
    
    is_v_prediction_model = noise_scheduler.config.prediction_type == "v_prediction"
    
    unet.train()
    progress_bar = tqdm(range(global_step, config.MAX_TRAIN_STEPS), desc="Training Steps", initial=global_step, total=config.MAX_TRAIN_STEPS)
    
    prefetcher = DataPrefetcher(train_dataloader, device, config.compute_dtype)
    batch = prefetcher.next()
    ema_loss = None
    ema_decay = 0.99  # Adjust this (e.g., 0.999 for slower smoothing)

    while global_step < config.MAX_TRAIN_STEPS:
        if batch is None:
            prefetcher = DataPrefetcher(train_dataloader, device, config.compute_dtype)
            batch = prefetcher.next()
            if not batch: continue

        data_fetch_time = time.time()
        
        # Tensors are already on the correct device and dtype from the prefetcher
        latents, prompt_embeds, pooled_prompt_embeds = batch["latents"], batch["prompt_embeds"], batch["pooled_prompt_embeds"]
        
        noise = torch.randn_like(latents)
        if config.USE_RESIDUAL_SHIFTING:
            min_timestep = int(0.5 * noise_scheduler.config.num_train_timesteps)
            timesteps = torch.randint(min_timestep, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        else:
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        
        if config.USE_IP_NOISE_GAMMA > 0:
            latents = latents + config.IP_NOISE_GAMMA * torch.randn_like(latents)
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        target = noise_scheduler.get_velocity(latents, noise, timesteps) if is_v_prediction_model else noise
        
        original_sizes, target_sizes = batch["original_sizes"], batch["target_sizes"]
        crops_coords_top_left = [(0, 0) for _ in original_sizes]
        add_time_ids = torch.tensor([list(s1) + list(c) + list(s2) for s1, c, s2 in zip(original_sizes, crops_coords_top_left, target_sizes)], device=device, dtype=prompt_embeds.dtype)
        
        forward_start = time.time()
        with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=use_scaler):
            pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}).sample
            
            if is_v_prediction_model and config.USE_MIN_SNR_GAMMA:
                snr = noise_scheduler.alphas_cumprod.to(device)[timesteps] / (1 - noise_scheduler.alphas_cumprod.to(device)[timesteps])
                snr_loss_weights = torch.min(config.MIN_SNR_GAMMA / snr, torch.ones_like(snr))
                loss = (F.mse_loss(pred.float(), target.float(), reduction="none").mean([1,2,3]) * snr_loss_weights).mean()
            else:
                loss = F.mse_loss(pred.float(), target.float())

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss.item() 

        forward_time = time.time() - forward_start
        
        backward_start = time.time()
        scaler.scale(loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
        backward_time = time.time() - backward_start
        
        optim_start = time.time()
        if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # --- HARDENED OPTIMIZER STEP ---
            scaler.unscale_(optimizer)

            # 1. Check for non-finite gradients before clipping
            all_grads_are_finite = all(torch.isfinite(p.grad).all() for p in params_to_optimize if p.grad is not None)

            if all_grads_are_finite:
                # 2. Clip gradients only if they are valid
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params_to_optimize if p.grad is not None], 
                    config.CLIP_GRAD_NORM
                )
                
                # 3. Step the optimizer
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
            else:
                # 4. Skip the optimizer step and warn the user
                print(f"\n[WARNING] Step {global_step}: Detected NaN/Inf gradients. Skipping optimizer step to prevent corruption. Check learning rate and data.")
                # Optionally, you can clear the corrupted gradients here
                # scaler.update() # Still need to update scaler even if step is skipped

            # Always zero the gradients to start the next accumulation cycle fresh
            optimizer.zero_grad(set_to_none=True)
            # --- END HARDENED OPTIMIZER STEP ---

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ema_loss": f"{ema_loss:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                "timesteps_min_max": f"{timesteps.min().item()}-{timesteps.max().item()}",
                "mem_gb": f"{torch.cuda.memory_reserved() / 1e9:.2f}"
            })
        optim_time = time.time() - optim_start

        # The 'Data' time now represents how long it took to get the pre-fetched batch.
        # The true data loading/transfer is happening in the background.
        print(f"Step {global_step}: Data={data_fetch_time - optim_start:.3f}s, Forward={forward_time:.3f}s, Backward={backward_time:.3f}s, Optim={optim_time:.3f}s")

        if torch.isnan(loss):
            print(f"NaN detected at step {global_step}, timesteps={timesteps}")
            raise ValueError("NaN in loss")

        progress_bar.update(1)
        global_step += 1
        
        batch = prefetcher.next()
        
        if global_step > 0 and global_step % config.SAVE_EVERY_N_STEPS == 0 and global_step < config.MAX_TRAIN_STEPS:
            # --- CHANGE 3: The call no longer re-assigns the state dict ---
            save_training_checkpoint(
                base_model_state_dict=base_model_state_dict, # Pass the original, clean dict
                unet=unet, 
                optimizer=optimizer, 
                scheduler=lr_scheduler, 
                step=global_step, 
                checkpoint_dir=CHECKPOINT_DIR, 
                trainable_param_names=unet_param_names_to_optimize, 
                save_dtype=config.compute_dtype
            )

        # --- PERIODIC GARBAGE COLLECTION ---
        # At the end of each step, clear any python-level objects that might be lingering
        if global_step % 100 == 0: # Adjust frequency as needed
             gc.collect()
        # --- END PERIODIC GARBAGE COLLECTION ---

        if global_step >= config.MAX_TRAIN_STEPS: break

    progress_bar.close()
    print("--> Training finished.")

    print("\nSaving final model...")
    base_fn = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_final_step_{global_step}"
    output_path = OUTPUT_DIR / f"{base_fn}.safetensors"
    save_model(
        base_sd=base_model_state_dict,
        output_path=output_path,
        unet=unet,
        trained_unet_param_names=unet_param_names_to_optimize,
        save_dtype=config.compute_dtype
    )
    print(f"--> Final model saved to {output_path}")
    print("\nTRAINING COMPLETE")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("INFO: Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass # Context already set

    os.environ['PYTHONUNBUFFERED'] = '1'
    main()