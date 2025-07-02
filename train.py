# -*- coding: utf-8 -*-
"""
SDXL Fine-Tuning Training Script

This script provides the core logic for fine-tuning a Stable Diffusion XL (SDXL) model.
It is designed to be executed with a configuration object that specifies all training
parameters, such as model paths, data directories, learning rates, and training steps.

Key features:
- Loads configuration from default settings and a user-provided JSON file.
- Pre-computes and caches VAE latents for efficient training.
- Supports mixed-precision training (bfloat16/float16).
- Implements gradient checkpointing and memory-efficient techniques.
- Optimizes specific parts of the UNet and Text Encoders.
- Supports resuming from saved checkpoints.
- Saves the final trained model as a .safetensors file.
"""

import inspect
import re
from pathlib import Path
import glob
import gc
import os
import json
from collections import defaultdict
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
import bitsandbytes
import logging
import warnings
import config as default_config
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
warnings.filterwarnings("ignore", category=UserWarning, message="None of the inputs have requires_grad=True. Gradients will be None")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

class TrainingConfig:
    def __init__(self):
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        user_config_path = Path("user_config.json")
        if user_config_path.exists():
            print(f"INFO: Loading user configuration from {user_config_path}")
            try:
                with open(user_config_path, 'r') as f:
                    user_config = json.load(f)
                for key, value in user_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR: Could not read or parse user_config.json: {e}. Using default settings.")
        else:
            print("INFO: user_config.json not found. Using default settings.")

# Add this import at the top of your train.py file
from concurrent.futures import ThreadPoolExecutor, as_completed

def precompute_and_cache_latents(data_root, bucket_sizes, tokenizer1, tokenizer2, vae, device_for_encoding, compute_dtype, batch_size, force_recache=False):
    data_root_path = Path(data_root)
    if force_recache:
        print("Force recache enabled. Deleting old cache directories...")
        for cache_dir in data_root_path.rglob(".precomputed_latents_cache"):
            if cache_dir.is_dir():
                print(f"  - Deleting {cache_dir}")
                shutil.rmtree(cache_dir)

    print("Scanning for images recursively...")
    image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in data_root_path.rglob(f"*{ext}")]
    if not image_paths: raise ValueError(f"No images found in {data_root} or its subdirectories.")
    print(f"Found {len(image_paths)} potential image files. Now validating and collecting metadata using multiple threads...")

    image_metadata = []
    
    # --- Start of multithreaded validation ---
    def validate_image(ip):
        """Worker function to validate a single image and extract its metadata."""
        if "_truncated" in ip.stem:
            return None, None
        try:
            with Image.open(ip) as img:
                img.load()
                w, h = img.size
            cp = ip.with_suffix('.txt')
            caption = ip.stem.replace('_', ' ')
            if cp.exists():
                with open(cp, 'r', encoding='utf-8') as f: caption = f.read().strip()
            
            closest_bucket_size = next((b for b in bucket_sizes if max(w, h) <= b), max(bucket_sizes))
            
            meta_info = {
                "ip": ip,
                "caption": caption,
                "bucket_info": (closest_bucket_size, w / h),
            }
            return meta_info, None # Success, no error
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            tqdm.write(f"\n[CORRUPT IMAGE DETECTED] Path: {ip}")
            tqdm.write(f"  └─ Reason: {e}")
            tqdm.write(f"  └─ Action: Renaming file to quarantine it.")
            try:
                new_stem = f"{ip.stem}_truncated"
                new_image_path = ip.with_name(new_stem + ip.suffix)
                ip.rename(new_image_path)
                tqdm.write(f"  └─ Renamed image to: {new_image_path.name}")
                caption_path = ip.with_suffix('.txt')
                if caption_path.exists():
                    new_caption_path = new_image_path.with_suffix('.txt')
                    caption_path.rename(new_caption_path)
                    tqdm.write(f"  └─ Renamed caption to: {new_caption_path.name}")
            except Exception as rename_e:
                tqdm.write(f"  └─ [ERROR] Could not rename file: {rename_e}")
            return None, ip # Failure, return the problematic path
    
    # Use a thread pool to validate images in parallel
    # os.cpu_count() or a fixed number like 16 are good starting points for max_workers
    max_workers = min(32, os.cpu_count() + 4) 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(validate_image, ip): ip for ip in image_paths}
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="Validating and scanning images"):
            metadata, error_path = future.result()
            if metadata:
                image_metadata.append(metadata)
    
    # Sort metadata by image path to ensure deterministic order for tokenization
    image_metadata.sort(key=lambda x: x['ip'])
    
    # Assign the 'original_index' after sorting and filtering
    for i, meta in enumerate(image_metadata):
        meta["original_index"] = i
    # --- End of multithreaded validation ---
    
    if not image_metadata:
        raise ValueError("No valid, non-corrupted images were found after scanning. Please check your dataset.")

    print(f"Validation complete. Found {len(image_metadata)} valid images.")
    all_captions = [m["caption"] for m in image_metadata]
    text_inputs1 = tokenizer1(all_captions, padding="max_length", max_length=tokenizer1.model_max_length, truncation=True, return_tensors="pt")
    text_inputs2 = tokenizer2(all_captions, padding="max_length", max_length=tokenizer2.model_max_length, truncation=True, return_tensors="pt")
    input_ids1, input_ids2 = text_inputs1.input_ids, text_inputs2.input_ids
    
    grouped_metadata = defaultdict(list)
    for meta in image_metadata: grouped_metadata[meta["bucket_info"][0]].append(meta)
    
    all_batches_to_process = []
    for bucket_size, metadata_list in grouped_metadata.items():
        for i in range(0, len(metadata_list), batch_size):
            all_batches_to_process.append((bucket_size, metadata_list[i:i + batch_size]))
    random.shuffle(all_batches_to_process)
    
    vae.to(device_for_encoding)
    for bucket_size, batch_metadata in tqdm(all_batches_to_process, desc="Caching latents in shuffled batches"):
        image_batch_tensors = []
        resized_dimensions_batch = []
        for meta in batch_metadata:
            img_path, aspect_ratio = meta["ip"], meta["bucket_info"][1]
            w, h = (bucket_size, int(bucket_size / aspect_ratio)) if aspect_ratio > 1.0 else (int(bucket_size * aspect_ratio), bucket_size)
            w, h = max(64, (w // 8) * 8), max(64, (h // 8) * 8)
            resized_dimensions_batch.append((h, w))
            img = Image.open(img_path).convert("RGB")
            resized_img = img.resize((w, h), resample=Image.LANCZOS)
            padded_img = Image.new("RGB", (bucket_size, bucket_size), (127, 127, 127))
            paste_coords = ((bucket_size - w) // 2, (bucket_size - h) // 2)
            padded_img.paste(resized_img, paste_coords)
            img_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(padded_img)
            image_batch_tensors.append(img_tensor)
        img_tensor_batch = torch.stack(image_batch_tensors).to(device_for_encoding, dtype=vae.dtype)
        with torch.no_grad():
            latents_batch = vae.encode(img_tensor_batch).latent_dist.mean * vae.config.scaling_factor
        for j, meta in enumerate(batch_metadata):
            original_idx = meta["original_index"]
            image_path = meta['ip']
            cache_dir_for_image = image_path.parent / ".precomputed_latents_cache"
            cache_dir_for_image.mkdir(exist_ok=True)
            save_path = cache_dir_for_image / f"{image_path.stem}.pt"
            resized_h, resized_w = resized_dimensions_batch[j]
            torch.save({
                "latents_cpu": latents_batch[j].clone().cpu().to(compute_dtype),
                "input_ids1_cpu": input_ids1[original_idx].clone(),
                "input_ids2_cpu": input_ids2[original_idx].clone(),
                "original_size_as_tuple": (resized_h, resized_w),
                "target_size_as_tuple": (resized_h, resized_w),
            }, save_path)
    vae.cpu(); gc.collect(); torch.cuda.empty_cache()

class ImageTextLatentDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.latent_files = sorted(list(self.data_root.rglob(".precomputed_latents_cache/*.pt")))
        if not self.latent_files:
            raise ValueError(f"No latent files found in subdirectories of {self.data_root}. Please ensure pre-caching was successful.")
        print(f"Dataset initialized with {len(self.latent_files)} pre-cached samples from all subdirectories.")
    def __len__(self): return len(self.latent_files)
    def __getitem__(self, i): return torch.load(self.latent_files[i], map_location="cpu")

def filter_scheduler_config(s,c):return{k:v for k,v in s.items() if k in inspect.signature(c.__init__).parameters}

def _generate_hf_to_sd_unet_key_mapping(hf_keys):
    m = {hk: hk for hk in hf_keys}
    u_map = [
        ("time_embedding.linear_1.weight", "time_embed.0.weight"),("time_embedding.linear_1.bias", "time_embed.0.bias"),
        ("time_embedding.linear_2.weight", "time_embed.2.weight"),("time_embedding.linear_2.bias", "time_embed.2.bias"),
        ("conv_in.weight", "input_blocks.0.0.weight"),("conv_in.bias", "input_blocks.0.0.bias"),
        ("conv_norm_out.weight", "out.0.weight"),("conv_norm_out.bias", "out.0.bias"),
        ("conv_out.weight", "out.2.weight"),("conv_out.bias", "out.2.bias"),
        ("add_embedding.linear_1.weight", "label_emb.0.0.weight"),("add_embedding.linear_1.bias", "label_emb.0.0.bias"),
        ("add_embedding.linear_2.weight", "label_emb.0.2.weight"),("add_embedding.linear_2.bias", "label_emb.0.2.bias"),
    ]
    r_map = [
        ("norm1", "in_layers.0"),("conv1", "in_layers.2"),("norm2", "out_layers.0"),("conv2", "out_layers.3"),
        ("time_emb_proj", "emb_layers.1"),("conv_shortcut", "skip_connection"),("proj_in", "proj_in"),("proj_out", "proj_out")
    ]
    l_map = []
    for i in range(4):
        for j in range(2):
            l_map.append((f"down_blocks.{i}.resnets.{j}.", f"input_blocks.{3*i+j+1}.0."))
            l_map.append((f"down_blocks.{i}.attentions.{j}.", f"input_blocks.{3*i+j+1}.1."))
        if i < 3:
            l_map.append((f"down_blocks.{i}.downsamplers.0.conv.", f"input_blocks.{3*(i+1)}.0.op."))
    for i in range(3):
        for j in range(3):
            l_map.append((f"up_blocks.{i}.resnets.{j}.", f"output_blocks.{3*i+j}.0."))
            l_map.append((f"up_blocks.{i}.attentions.{j}.", f"output_blocks.{3*i+j}.1."))
        if i < 2:
            l_map.append((f"up_blocks.{i}.upsamplers.0.", f"output_blocks.{3*i+2}.2."))
    l_map.append(("mid_block.attentions.0.", "middle_block.1."))
    for j in range(2):
        l_map.append((f"mid_block.resnets.{j}.", f"middle_block.{2*j}."))
    for h, s in u_map:
        if h in m: m[h] = s
    for ho in list(m.keys()):
        c = m[ho]
        if "resnets" in ho or 'attentions' in ho:
            for hp, sp in r_map: c = c.replace(hp, sp)
        m[ho] = c
    for ho in list(m.keys()):
        c = m[ho]
        for hp, sp in l_map:
            if c.startswith(hp): c = sp + c[len(hp):]
        m[ho] = c
    return m

def save_model(base_model_path, output_path, trained_models, trained_param_names, save_dtype):
    print(f"Loading base model: {base_model_path}")
    try:
        base_sd = load_file(base_model_path, device="cpu")
    except Exception as e:
        print(f"Could not load base model from {base_model_path}. Error: {e}")
        return
    unet_key_map = _generate_hf_to_sd_unet_key_mapping(list(trained_models['unet'].state_dict().keys()))
    param_counters = {
        'unet': defaultdict(lambda: {'total': 0, 'saved': 0, 'skipped_not_in_base': 0, 'skipped_not_in_memory': 0}),
        'text_encoder1': defaultdict(lambda: {'total': 0, 'saved': 0, 'skipped_not_in_base': 0, 'skipped_not_in_memory': 0}),
        'text_encoder2': defaultdict(lambda: {'total': 0, 'saved': 0, 'skipped_not_in_base': 0, 'skipped_not_in_memory': 0}),
    }
    def get_param_category(key_name):
        if 'ff.net' in key_name: return "Feed-Forward (ff)"
        if 'attn1' in key_name: return "Self-Attention (attn1)"
        if 'attn2' in key_name: return "Cross-Attention (attn2)"
        if 'time_emb_proj' in key_name: return "Time Embedding Proj"
        if 'conv_in' in key_name: return "UNet Input Conv"
        if 'conv_out' in key_name: return "UNet Output Conv"
        if 'time_embedding' in key_name: return "Time Embedding Linear"
        if 'token_embedding' in key_name: return "Text Token Embedding"
        return "Other"

    for model_key, model_obj in trained_models.items():
        if model_key not in trained_param_names or not trained_param_names[model_key]:
            continue
        print(f"\nUpdating weights for {model_key}...")
        model_sd_on_device = model_obj.state_dict()
        for hf_key in trained_param_names[model_key]:
            category = get_param_category(hf_key)
            param_counters[model_key][category]['total'] += 1
            if hf_key not in model_sd_on_device:
                param_counters[model_key][category]['skipped_not_in_memory'] += 1
                continue
            param_to_save = model_sd_on_device[hf_key].to("cpu", dtype=save_dtype)
            sd_key = None
            if model_key == 'unet':
                mapped_part = unet_key_map.get(hf_key)
                if mapped_part:
                    sd_key = 'model.diffusion_model.' + mapped_part
            elif model_key == 'text_encoder1' and 'token_embedding' in hf_key:
                sd_key = 'conditioner.embedders.0.transformer.' + hf_key
            elif model_key == 'text_encoder2' and 'token_embedding' in hf_key:
                sd_key = 'conditioner.embedders.1.model.' + hf_key.replace('text_model.embeddings.', '', 1)

            if sd_key and sd_key in base_sd:
                base_sd[sd_key] = param_to_save
                param_counters[model_key][category]['saved'] += 1
            elif sd_key:
                param_counters[model_key][category]['skipped_not_in_base'] += 1
        del model_sd_on_device
        gc.collect()

    print("\n" + "="*60); print("           SAVE MODEL VERIFICATION REPORT"); print("="*60)
    for model_key, categories in param_counters.items():
        if not any(cat['total'] > 0 for cat in categories.values()): continue
        print(f"\n--- Model: {model_key.upper()} ---")
        total_model_params, total_model_saved = 0, 0
        for category, counts in sorted(categories.items()):
            if counts['total'] == 0: continue
            print(f"  - {category:<25}: Found {counts['total']:>4} -> Saved {counts['saved']:>4}")
            if counts['skipped_not_in_base'] > 0: print(f"    -> WARNING: Skipped {counts['skipped_not_in_base']} params (key not in base model). CHECK KEY MAPPING!")
            if counts['skipped_not_in_memory'] > 0: print(f"    -> ERROR: Skipped {counts['skipped_not_in_memory']} params (not in memory). CHECK PARAM NAME GENERATION!")
            total_model_params += counts['total']; total_model_saved += counts['saved']
        print(f"  --------------------------------------------------")
        print(f"  {model_key.upper()} Summary: {total_model_saved} / {total_model_params} parameters saved.")

    print("\n" + "="*60)
    print(f"\nSaving final model to {output_path} (dtype: {save_dtype})")
    save_file(base_sd, output_path)
    print("[OK] Save complete.")
    del base_sd; gc.collect(); torch.cuda.empty_cache()

def custom_collate_fn_latent(batch):
    return {
        "latents": torch.stack([item["latents_cpu"] for item in batch]),
        "input_ids1": torch.stack([item["input_ids1_cpu"] for item in batch]),
        "input_ids2": torch.stack([item["input_ids2_cpu"] for item in batch]),
        "original_sizes": [item["original_size_as_tuple"] for item in batch],
        "target_sizes": [item["target_size_as_tuple"] for item in batch],
    }

def save_training_checkpoint(base_model_path, models, optimizer, scheduler, step, checkpoint_dir, trainable_param_names, save_dtype):
    checkpoint_model_path = checkpoint_dir / f"checkpoint_step_{step}.safetensors"
    print(f"\nSaving full model checkpoint to: {checkpoint_model_path}")
    save_model(
        base_model_path=base_model_path,
        output_path=checkpoint_model_path,
        trained_models=models,
        trained_param_names=trainable_param_names,
        save_dtype=save_dtype
    )
    state = {
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    state_path = checkpoint_dir / f"training_state_step_{step}.pt"
    torch.save(state, state_path)
    print(f"[OK] Saved optimizer/scheduler state to: {state_path}")
    del state; gc.collect()

def main():
    config = TrainingConfig()
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(config.INSTANCE_DATA_DIR).exists(): raise FileNotFoundError(f"Data dir not found: {config.INSTANCE_DATA_DIR}")
    if not Path(config.SINGLE_FILE_CHECKPOINT_PATH).exists(): raise FileNotFoundError(f"Base model not found: {config.SINGLE_FILE_CHECKPOINT_PATH}")
    compute_dtype = torch.bfloat16 if config.MIXED_PRECISION == "bfloat16" else torch.float16
    print(f"Using compute dtype: {compute_dtype}, Device: {device}")
    caches_exist = any(Path(config.INSTANCE_DATA_DIR).rglob(".precomputed_latents_cache/*.pt"))
    if config.FORCE_RECACHE_LATENTS or not caches_exist:
        print("Caching required. Loading minimal components for VAE...")
        temp_pipe = StableDiffusionXLPipeline.from_single_file(config.SINGLE_FILE_CHECKPOINT_PATH, torch_dtype=compute_dtype, use_safetensors=True)
        precompute_and_cache_latents(
            data_root=config.INSTANCE_DATA_DIR,
            bucket_sizes=config.BUCKET_SIZES,
            tokenizer1=temp_pipe.tokenizer,
            tokenizer2=temp_pipe.tokenizer_2,
            vae=temp_pipe.vae,
            device_for_encoding=device,
            compute_dtype=compute_dtype,
            batch_size=config.CACHING_BATCH_SIZE,
            force_recache=config.FORCE_RECACHE_LATENTS
        )
        del temp_pipe; gc.collect(); torch.cuda.empty_cache()
    else:
        print("Latent caches are up-to-date. Skipping caching.")

    # --- MODIFIED RESUME LOGIC ---
    global_step = 0
    model_to_load = config.SINGLE_FILE_CHECKPOINT_PATH
    latest_state_path = None # Will be set if resume is successful

    if config.RESUME_TRAINING:
        print("Resume training enabled. Checking provided paths...")
        resume_model_path = Path(config.RESUME_MODEL_PATH) if config.RESUME_MODEL_PATH else None
        resume_state_path = Path(config.RESUME_STATE_PATH) if config.RESUME_STATE_PATH else None

        if resume_model_path and resume_model_path.exists() and resume_state_path and resume_state_path.exists():
            print(f"[OK] Found model checkpoint: {resume_model_path}")
            print(f"[OK] Found training state: {resume_state_path}")
            model_to_load = str(resume_model_path)
            latest_state_path = resume_state_path # This path will be used to load optimizer state
        else:
            print("[WARNING] Resume training is ON, but one or more paths are invalid or do not exist.")
            print(f"  - Model Path: '{config.RESUME_MODEL_PATH}' (Exists: {resume_model_path.exists() if resume_model_path else 'No'})")
            print(f"  - State Path: '{config.RESUME_STATE_PATH}' (Exists: {resume_state_path.exists() if resume_state_path else 'No'})")
            print("[ACTION] Reverting to base model and starting from step 0. Please check paths in the GUI.")
    # --- END OF MODIFIED RESUME LOGIC ---

    print(f"\nLoading model from: {model_to_load}")
    pipeline_temp = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=compute_dtype, use_safetensors=True)
    unet, text_encoder1, text_encoder2 = pipeline_temp.unet, pipeline_temp.text_encoder, pipeline_temp.text_encoder_2
    scheduler_config = pipeline_temp.scheduler.config
    del pipeline_temp.vae, pipeline_temp.tokenizer, pipeline_temp.tokenizer_2, pipeline_temp
    gc.collect(); torch.cuda.empty_cache()
    print("--> Full training pipeline prepaired.")
    print(f"\nLoading model from: {model_to_load}")
    print("This may take a moment...")
    load_start_time = time.time()
    pipeline_temp = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=compute_dtype, use_safetensors=True)
    load_end_time = time.time()
    print(f"--> Model loaded into memory in {load_end_time - load_start_time:.2f} seconds.")
    print("Extracting and preparing UNet and Text Encoders...")
    unet, text_encoder1, text_encoder2 = pipeline_temp.unet, pipeline_temp.text_encoder, pipeline_temp.text_encoder_2
    scheduler_config = pipeline_temp.scheduler.config
    print("Unloading unnecessary components (VAE, tokenizers)...")
    del pipeline_temp.vae, pipeline_temp.tokenizer, pipeline_temp.tokenizer_2, pipeline_temp
    gc.collect(); torch.cuda.empty_cache()
    print("--> Full training pipeline ready.")
    print("Moving UNet to GPU...")
    all_models = {'unet': unet, 'text_encoder1': text_encoder1, 'text_encoder2': text_encoder2}
    unet.to(device)
    print("--> UNet is on the GPU.")
    unet.requires_grad_(False); text_encoder1.requires_grad_(False); text_encoder2.requires_grad_(False)
    if hasattr(unet, 'enable_gradient_checkpointing'): unet.enable_gradient_checkpointing()
    if hasattr(text_encoder1, 'gradient_checkpointing_enable'): text_encoder1.gradient_checkpointing_enable()
    if hasattr(text_encoder2, 'gradient_checkpointing_enable'): text_encoder2.gradient_checkpointing_enable()
    print("\n" + "="*50); print("       SELECTING TRAINABLE PARAMETERS"); print("="*50)

    # UNet Parameters
    if not config.UNET_TRAIN_TARGETS:
        print("INFO: No UNet training targets specified. Skipping UNet training.")
        unet_param_names_to_optimize = set()
    else:
        print(f"INFO: Targeting UNet layers with keywords: {config.UNET_TRAIN_TARGETS}")
        unet_param_names_to_optimize = {
            name for name, _ in unet.named_parameters() 
            if any(k in name for k in config.UNET_TRAIN_TARGETS)
        }

    # Text Encoder Parameters
    te1_param_names_to_optimize = set()
    te2_param_names_to_optimize = set()
    if config.TEXT_ENCODER_TRAIN_TARGET == "token_embedding_only":
        print("INFO: Targeting Text Encoder token embeddings only.")
        te1_param_names_to_optimize = {name for name, _ in text_encoder1.named_parameters() if 'token_embedding' in name}
        te2_param_names_to_optimize = {name for name, _ in text_encoder2.named_parameters() if 'token_embedding' in name}
    elif config.TEXT_ENCODER_TRAIN_TARGET == "full":
        print("INFO: Targeting FULL Text Encoders.")
        te1_param_names_to_optimize = {name for name, _ in text_encoder1.named_parameters()}
        te2_param_names_to_optimize = {name for name, _ in text_encoder2.named_parameters()}
    else: # "none" or any other value
        print("INFO: Skipping Text Encoder training.")

    all_trainable_param_names = {
        'unet': unet_param_names_to_optimize,
        'text_encoder1': te1_param_names_to_optimize,
        'text_encoder2': te2_param_names_to_optimize
    }

    print("\n" + "="*50); print("        TRAINING TARGET ANALYSIS"); print("="*50)
    unet_params = [p for n, p in unet.named_parameters() if n in unet_param_names_to_optimize]
    te1_params = [p for n, p in text_encoder1.named_parameters() if n in te1_param_names_to_optimize]
    te2_params = [p for n, p in text_encoder2.named_parameters() if n in te2_param_names_to_optimize]
    for p in unet_params + te1_params + te2_params: p.requires_grad_(True)
    optimizer_grouped_parameters = [
        {"params": unet_params, "lr": config.UNET_LEARNING_RATE},
        {"params": te1_params, "lr": config.TEXT_ENCODER_LEARNING_RATE},
        {"params": te2_params, "lr": config.TEXT_ENCODER_LEARNING_RATE},
    ]
    params_to_optimize = unet_params + te1_params + te2_params
    unet_trainable_params = sum(p.numel() for p in unet_params)
    te_trainable_params = sum(p.numel() for p in te1_params) + sum(p.numel() for p in te2_params)
    total_trainable_params = unet_trainable_params + te_trainable_params
    total_params = sum(p.numel() for p in unet.parameters()) + sum(p.numel() for p in text_encoder1.parameters()) + sum(p.numel() for p in text_encoder2.parameters())
    print(f"UNet params to train:      {unet_trainable_params/1e6:.2f}M with LR: {config.UNET_LEARNING_RATE}")
    print(f"Text Encoder params train: {te_trainable_params/1e6:.2f}M with LR: {config.TEXT_ENCODER_LEARNING_RATE}")
    print("-" * 50)
    print(f"OVERALL: Training {total_trainable_params / 1e6:.2f}M parameters out of {total_params / 1e9:.3f}B total.")
    print("="*50 + "\n")
    optimizer = Adafactor(optimizer_grouped_parameters, eps=(1e-30, 1e-3), clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=False, relative_step=False, warmup_init=False)
    num_update_steps = math.ceil(config.MAX_TRAIN_STEPS / config.GRADIENT_ACCUMULATION_STEPS)
    num_warmup_update_steps = math.ceil(int(config.MAX_TRAIN_STEPS * config.LR_WARMUP_PERCENT) / config.GRADIENT_ACCUMULATION_STEPS)

    print(f"INFO: Using LR Scheduler: {config.LR_SCHEDULER_TYPE}")
    print(f"INFO: Scheduler will run for {num_update_steps} optimizer steps.")
    print(f"INFO: Warmup will be for the first {num_warmup_update_steps} optimizer steps.")
    if config.LR_SCHEDULER_TYPE == "linear":
        lr_scheduler_class = get_linear_schedule_with_warmup
    elif config.LR_SCHEDULER_TYPE == "cosine":
        lr_scheduler_class = get_cosine_schedule_with_warmup
    else:
        print(f"WARNING: Scheduler '{config.LR_SCHEDULER_TYPE}' is not available in your 'transformers' version. Defaulting to 'cosine'.")
        lr_scheduler_class = get_cosine_schedule_with_warmup
    lr_scheduler = lr_scheduler_class(optimizer=optimizer, num_warmup_steps=num_warmup_update_steps, num_training_steps=num_update_steps)
    
    # This block now correctly uses the 'latest_state_path' variable set by our new logic
    if latest_state_path:
        print(f"Loading optimizer and scheduler state from {latest_state_path.name}...")
        state = torch.load(latest_state_path, map_location=device)
        global_step = state['step']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state['scheduler_state_dict'])
        del state; gc.collect(); torch.cuda.empty_cache()
        print(f"[OK] Resumed training from step {global_step}.")

    noise_scheduler = DDPMScheduler(**filter_scheduler_config(scheduler_config, DDPMScheduler))
    random.seed(config.SEED); torch.manual_seed(config.SEED); torch.cuda.manual_seed_all(config.SEED)
    train_dataset = ImageTextLatentDataset(config.INSTANCE_DATA_DIR)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_latent, num_workers=config.NUM_WORKERS, persistent_workers=(config.NUM_WORKERS > 0))
    final_loss = 0.0
    unet.train(); text_encoder1.train(); text_encoder2.train()
    training_start_time = time.time()
    progress_bar = tqdm(range(global_step, config.MAX_TRAIN_STEPS), desc="Training Steps", initial=global_step, total=config.MAX_TRAIN_STEPS)
    dataloader_iterator = iter(train_dataloader)
    while global_step < config.MAX_TRAIN_STEPS:
        try: batch = next(dataloader_iterator)
        except StopIteration: dataloader_iterator = iter(train_dataloader); batch = next(dataloader_iterator)
        text_encoder1.to(device); text_encoder2.to(device)
        with torch.no_grad():
            prompt_embeds_out = text_encoder1(batch["input_ids1"].to(device), output_hidden_states=True)
            pooled_prompt_embeds_out = text_encoder2(batch["input_ids2"].to(device), output_hidden_states=True)
            prompt_embeds = torch.cat([prompt_embeds_out.hidden_states[-2], pooled_prompt_embeds_out.hidden_states[-2]], dim=-1).to(dtype=compute_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds_out.text_embeds.to(dtype=compute_dtype)
        text_encoder1.to("cpu"); text_encoder2.to("cpu"); gc.collect(); torch.cuda.empty_cache()
        latents = batch["latents"].to(device, dtype=compute_dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        is_v_prediction = noise_scheduler.config.prediction_type == "v_prediction"
        if is_v_prediction and config.USE_MIN_SNR_GAMMA and global_step == 0:
             print(f"INFO: Detected v-prediction model. Applying Min-SNR gamma weighting with gamma = {config.MIN_SNR_GAMMA}.")
        elif is_v_prediction and not config.USE_MIN_SNR_GAMMA and global_step == 0:
             print(f"INFO: Detected v-prediction model, but Min-SNR gamma is disabled by user.")
        if is_v_prediction:
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        original_sizes, target_sizes, crops_coords_top_left = batch["original_sizes"], batch["target_sizes"], [(0, 0) for _ in batch["original_sizes"]]
        add_time_ids_list = [list(orig_size) + list(crop_coords) + list(targ_size) for orig_size, crop_coords, targ_size in zip(original_sizes, crops_coords_top_left, target_sizes)]
        add_time_ids = torch.tensor(add_time_ids_list, device=device, dtype=prompt_embeds.dtype)
        prompt_embeds.requires_grad_(True)
        with torch.autocast(device_type=device.type, dtype=compute_dtype):
            pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}).sample
            if is_v_prediction and config.USE_MIN_SNR_GAMMA:
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
                snr_gamma = torch.tensor(config.MIN_SNR_GAMMA, device=device)
                snr_loss_weights = (torch.stack([snr, snr_gamma.expand(snr.shape)], dim=1).min(dim=1)[0] / snr.clamp(min=1.0)).detach()
                loss_per_pixel = F.mse_loss(pred.float(), target.float(), reduction="none")
                loss = (loss_per_pixel.mean(dim=list(range(1, len(loss_per_pixel.shape)))) * snr_loss_weights).mean()
            else:
                loss = F.mse_loss(pred.float(), target.float())
        (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()
        if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            params_to_clip = [p for p in params_to_optimize if p.grad is not None]
            if params_to_clip:
                torch.nn.utils.clip_grad_norm_(params_to_clip, config.CLIP_GRAD_NORM)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            lrs = lr_scheduler.get_last_lr()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "unet_lr": f"{lrs[0]:.2e}", "te_lr": f"{lrs[1]:.2e}"})
            final_loss = loss.item()

        progress_bar.update(1); global_step += 1
        if global_step > 0 and global_step % config.SAVE_EVERY_N_STEPS == 0 and global_step < config.MAX_TRAIN_STEPS:
            save_training_checkpoint(
                base_model_path=config.SINGLE_FILE_CHECKPOINT_PATH,
                models=all_models,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                step=global_step,
                checkpoint_dir=CHECKPOINT_DIR,
                trainable_param_names=all_trainable_param_names,
                save_dtype=compute_dtype
            )
        if global_step >= config.MAX_TRAIN_STEPS: break

    progress_bar.close()
    training_end_time = time.time()
    print("--> Training finished.")
    print("\n" + "="*50); print("    POST-TRAINING ANALYSIS & SANITY CHECKS"); print("="*50)
    total_training_time = training_end_time - training_start_time
    avg_step_time = total_training_time / (config.MAX_TRAIN_STEPS - progress_bar.initial) if (config.MAX_TRAIN_STEPS - progress_bar.initial) > 0 else 0
    print("Training Performance:")
    print(f"  - Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"  - Average Step Time:   {avg_step_time:.2f} seconds/step")
    print(f"  - Final Loss:          {final_loss:.5f}")
    final_lrs = lr_scheduler.get_last_lr()
    print(f"  - Final UNet LR:       {final_lrs[0]:.2e}")
    print(f"  - Final TE LR:         {final_lrs[1]:.2e}")
    print("\nSanity Check 1: Verifying weight changes in memory...")
    unet.to(device)
    original_unet_sd = {k: v for k, v in load_file(config.SINGLE_FILE_CHECKPOINT_PATH, device="cpu").items() if k.startswith("model.diffusion_model.")}
    trained_unet_sd_in_mem = unet.state_dict()
    key_map = _generate_hf_to_sd_unet_key_mapping(list(trained_unet_sd_in_mem.keys()))
    total_abs_diff, num_params_checked = 0.0, 0
    for name in all_trainable_param_names['unet']:
        sd_key = "model.diffusion_model." + key_map.get(name, "")
        if sd_key in original_unet_sd and name in trained_unet_sd_in_mem:
            orig_p = original_unet_sd[sd_key].to(device, dtype=torch.float32)
            trained_p = trained_unet_sd_in_mem[name].to(device, dtype=torch.float32)
            total_abs_diff += torch.abs(orig_p - trained_p).sum().item()
            num_params_checked += orig_p.numel()
    mean_abs_diff = total_abs_diff / num_params_checked if num_params_checked > 0 else 0
    print(f"  - Compared {num_params_checked / 1e6:.2f}M UNet parameters.")
    print(f"  - Mean Absolute Difference from Base Model: {mean_abs_diff:.2e}")
    if mean_abs_diff > 1e-7: print("  - [PASS] Model weights have changed significantly from the base model.")
    else: print("  - [FAIL] Model weights have NOT changed. Check learning rate or optimizer.")
    del original_unet_sd; gc.collect(); torch.cuda.empty_cache()
    print("\nSaving final model...")
    base_fn = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_trained_step{global_step}"
    output_path = OUTPUT_DIR / f"{base_fn}.safetensors"
    save_model(config.SINGLE_FILE_CHECKPOINT_PATH, output_path, all_models, all_trainable_param_names, compute_dtype)
    print(f"--> Final model saved to {output_path}")
    print("\nSanity Check 2: Verifying saved file matches in-memory model...")
    if not output_path.exists():
        print("  - [FAIL] Saved file not found at expected path.")
    else:
        saved_model_sd = {k: v for k, v in load_file(output_path, device="cpu").items() if k.startswith("model.diffusion_model.")}
        total_abs_diff_save, num_params_checked_save = 0.0, 0
        for name in all_trainable_param_names['unet']:
            sd_key = "model.diffusion_model." + key_map.get(name, "")
            if sd_key in saved_model_sd and name in trained_unet_sd_in_mem:
                saved_p = saved_model_sd[sd_key].to(device, dtype=torch.float32)
                trained_p = trained_unet_sd_in_mem[name].to(device, dtype=torch.float32)
                total_abs_diff_save += torch.abs(saved_p - trained_p).sum().item()
                num_params_checked_save += saved_p.numel()
        mean_abs_diff_save = total_abs_diff_save / num_params_checked_save if num_params_checked_save > 0 else 0
        print(f"  - Compared {num_params_checked_save / 1e6:.2f}M UNet parameters between memory and file.")
        print(f"  - Mean Absolute Difference from Saved File: {mean_abs_diff_save:.2e}")
        if mean_abs_diff_save < 1e-5: print("  - [PASS] Saved file accurately reflects the trained model.")
        else: print("  - [FAIL] Significant difference between in-memory model and saved file.")
        del saved_model_sd; gc.collect()

    print("\n" + "="*50); print("            TRAINING COMPLETE"); print("="*50)

if __name__ == "__main__":
    os.environ['PYTHONUNBUFFERED'] = '1'
    main()