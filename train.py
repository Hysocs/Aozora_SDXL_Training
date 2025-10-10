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
from diffusers.models.attention_processor import AttnProcessor2_0
from optimizer.raven import RavenAdamW
from transformers.optimization import Adafactor
import multiprocessing
from multiprocessing import Pool, cpu_count
import config as default_config  # Import default configuration


# --- Global Settings ---
warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"INFO: Set random seed to {seed}")


class TrainingConfig:
    """Consolidates all training configurations."""
    def __init__(self):
        # Load all defaults from config.py first
        for key, value in default_config.__dict__.items():
            if not key.startswith('__'):
                setattr(self, key, value)
        
        # Now load from user config file if provided (this will override defaults)
        self._load_from_user_config()
        self._type_check_and_correct()
        
        # Set compute dtype based on mixed precision setting
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
        """Convert string representations of booleans and handle special types."""
        for key, value in list(self.__dict__.items()):
            if isinstance(value, str):
                if value.lower() in ['true', '1', 't', 'y', 'yes']: 
                    setattr(self, key, True)
                elif value.lower() in ['false', '0', 'f', 'n', 'no']: 
                    setattr(self, key, False)
            
            # Handle UNET_EXCLUDE_TARGETS conversion from string to list
            if key == "UNET_EXCLUDE_TARGETS" and isinstance(value, str):
                # Filter out empty strings!
                setattr(self, key, [item.strip() for item in value.split(',') if item.strip()])
            
            # NEW: Also filter if it's already a list with empty strings
            if key == "UNET_EXCLUDE_TARGETS" and isinstance(value, list):
                setattr(self, key, [item for item in value if item])
        
        print("INFO: Configuration types checked and corrected.")


class CustomCurveLRScheduler:
    """Custom LR scheduler that interpolates along a user-defined curve."""
    
    def __init__(self, optimizer, curve_points, max_train_steps):
        """
        Args:
            optimizer: The optimizer to adjust
            curve_points: List of [normalized_x, lr_value] where x is 0.0 to 1.0
            max_train_steps: Total training steps (not optimizer steps)
        """
        self.optimizer = optimizer
        self.curve_points = sorted(curve_points, key=lambda p: p[0])
        self.max_train_steps = max(max_train_steps, 1)
        self.current_training_step = 0
        
        # Validate curve
        if not self.curve_points:
            raise ValueError("LR_CUSTOM_CURVE cannot be empty")
        if self.curve_points[0][0] != 0.0:
            print("WARNING: First LR curve point should start at x=0.0. Prepending [0.0, first_lr].")
            self.curve_points.insert(0, [0.0, self.curve_points[0][1]])
        if self.curve_points[-1][0] != 1.0:
            print("WARNING: Last LR curve point should end at x=1.0. Appending [1.0, last_lr].")
            self.curve_points.append([1.0, self.curve_points[-1][1]])
        
        # Set initial LR
        self._update_lr()
    
    def _interpolate_lr(self, normalized_position):
        """Linear interpolation between curve points."""
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
        """Update LR based on current training step."""
        normalized_position = self.current_training_step / max(self.max_train_steps - 1, 1)
        lr = self._interpolate_lr(normalized_position)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def step(self, training_step):
        """Update LR. Call this ONCE per optimizer step with the current training step."""
        self.current_training_step = training_step
        self._update_lr()
    
    def get_last_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainingDiagnostics:
    """A streamlined class to handle collection and reporting of training metrics."""
    def __init__(self, accumulation_steps, test_param_name):
        self.accumulation_steps = accumulation_steps
        self.test_param_name = test_param_name
        self.losses = deque(maxlen=accumulation_steps)

    def step(self, loss):
        """Record the loss for the current accumulation step."""
        if loss is not None:
            self.losses.append(loss)

    def report(self, global_step, optimizer, raw_grad_norm, clipped_grad_norm, before_val, after_val):
        """Prints a formatted report of the current training state."""
        if not self.losses: return

        avg_loss = sum(self.losses) / len(self.losses)
        current_lr = optimizer.param_groups[0]['lr']
        update_delta = torch.abs(after_val - before_val).max().item()
        update_status = "[OK]" if update_delta > 1e-9 else "[NO UPDATE!]"
        
        # Fixed VRAM reporting - show total training memory used, not just allocated
        vram_reserved_gb = torch.cuda.memory_reserved() / 1e9  # This is total VRAM used by training
        vram_allocated_gb = torch.cuda.memory_allocated() / 1e9  # This is just actively allocated tensors

        report_str = (
            f"\n--- Step: {global_step:<5} | Loss: {avg_loss:<8.5f} | LR: {current_lr:.2e} ---\n"
            f"  Grad Norm (Raw/Clipped): {raw_grad_norm:<8.4f} / {clipped_grad_norm:<8.4f}\n"
            f"  VRAM: Training={vram_reserved_gb:.2f}GB | Model={vram_allocated_gb:.2f}GB\n"
            f"  Test Param: {self.test_param_name}\n"
            f"  |- Update : {update_delta:.4e} {update_status}"
        )
        tqdm.write(report_str)
        self.reset()

    def reset(self):
        self.losses.clear()


class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed, shuffle=True, drop_last=False):
        self.batch_size, self.shuffle, self.drop_last, self.seed, self.dataset = batch_size, shuffle, drop_last, seed, dataset
        self.epoch_indices = list(range(len(self.dataset.latent_files)))

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.seed)
        indices = torch.randperm(len(self.epoch_indices), generator=g).tolist() if self.shuffle else self.epoch_indices
        buckets = defaultdict(list)
        for i in indices:
            if (key := self.dataset.bucket_keys[i]): buckets[key].append(i)
        
        all_batches = []
        for key in sorted(buckets.keys()):
            for i in range(0, len(buckets[key]), self.batch_size):
                batch = buckets[key][i:i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size: all_batches.append(batch)
        
        if self.shuffle: all_batches = [all_batches[i] for i in torch.randperm(len(all_batches), generator=g).tolist()]
        yield from all_batches
        self.seed += 1

    def __len__(self):
        return (len(self.epoch_indices) // self.batch_size if self.drop_last 
                else (len(self.epoch_indices) + self.batch_size - 1) // self.batch_size)


class ResolutionCalculator:
    def __init__(self, target_area, stride=64):
        self.target_area, self.stride = target_area, stride

    def calculate_resolution(self, width, height):
        aspect_ratio = width / height
        h = int(math.sqrt(self.target_area / aspect_ratio) // self.stride) * self.stride
        w = int(h * aspect_ratio // self.stride) * self.stride
        return (w if w > 0 else self.stride, h if h > 0 else self.stride)


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
        # Image validation
        with Image.open(ip) as img: 
            img.verify()
        with Image.open(ip) as img: 
            img.load()
            w, h = img.size
        
        cp = ip.with_suffix('.txt')
        
        # Check for the caption file
        if cp.exists():
            with open(cp, 'r', encoding='utf-8') as f: 
                caption = f.read().strip()
            # If the caption file is empty after stripping whitespace, it's a problem.
            if not caption:
                tqdm.write(f"\n[WARNING] Caption file found for {ip.name}, but it is EMPTY. Image will be skipped.")
                return None
        else:
            # If the caption file does NOT exist, use the filename as a fallback AND WARN THE USER.
            fallback_caption = ip.stem.replace('_', ' ')
            tqdm.write(f"\n[CAPTION WARNING] No .txt file found for {ip.name}. Using fallback caption: '{fallback_caption}'")
            caption = fallback_caption
            
        return {"ip": ip, "caption": caption, "target_resolution": calculator.calculate_resolution(w, h), "original_size": (w, h)}

    except Exception as e:
        tqdm.write(f"\n[CORRUPT IMAGE OR READ ERROR] Skipping {ip}, Reason: {e}")
        return None


def compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device):
    """Process captions ONE AT A TIME to minimize VRAM usage during caching."""
    prompt_embeds_list = []
    pooled_prompt_embeds_list = []
    
    for caption in captions:  # Process individually to save VRAM
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
    """Check if any dataset needs caching before loading heavy models."""
    needs_caching = False
    
    for dataset in config.INSTANCE_DATASETS:
        root = Path(dataset["path"])
        if not root.exists():
            print(f"WARNING: Dataset path {root} does not exist, skipping.")
            continue
            
        # Check for images
        image_paths = [p for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] 
                      for p in root.rglob(f"*{ext}")]
        
        if not image_paths:
            print(f"INFO: No images found in {root}, skipping.")
            continue
        
        # Check for cached files
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
    """Load ONLY the VAE efficiently from a dedicated VAE file."""
    vae_path = config.VAE_PATH
    
    if not vae_path or not Path(vae_path).exists():
        return None  # Signal that we need to use the VAE from the main pipeline
    
    print(f"INFO: Loading dedicated VAE from: {vae_path}")
    # Load VAE directly from the single file
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32)
    
    print(f"INFO: VAE in float32 for high-quality caching")
    print(f"INFO: Moving VAE to {device}...")
    vae = vae.to(device)
    
    # Report actual VRAM usage after loading
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"INFO: VAE loaded. Current VRAM usage: {vram_gb:.2f} GB")
    
    return vae


def precompute_and_cache_latents(config, t1, t2, te1, te2, vae, device):
    """Streamlined caching with minimal memory footprint."""
    
    # Check if caching is even needed
    if not check_if_caching_needed(config):
        print("\n" + "="*60)
        print("INFO: All datasets are already cached. Skipping caching step.")
        print("="*60 + "\n")
        return
    
    print("\n" + "="*60)
    print("STARTING CACHING PROCESS")
    print("="*60 + "\n")
    
    calc = ResolutionCalculator(config.TARGET_PIXEL_AREA)
    
    # VAE is already loaded and passed in
    vae.enable_tiling()
    
    # Load text encoders
    print("INFO: Loading text encoders to device...")
    te1.to(device)
    te2.to(device)
    
    # Report memory usage
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
        
        # Check what's already cached
        cache_dir = root / ".precomputed_embeddings_cache"
        cache_dir.mkdir(exist_ok=True)
        
        stems = {p.stem for p in cache_dir.rglob("*.pt")}
        to_process = [p for p in paths if p.stem not in stems]
        
        if not to_process: 
            print(f"INFO: All {len(paths)} images already cached for this dataset.")
            continue
        
        print(f"INFO: Found {len(to_process)} images to cache (out of {len(paths)} total)")
        
        # Validate and assign resolutions using multiprocessing
        print("INFO: Validating images and calculating resolutions...")
        with Pool(processes=min(cpu_count(), 8)) as pool:  # Limit to 8 processes max
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
        
        # Save metadata for bucketing
        res_meta = {m['ip'].stem: m['target_resolution'] for m in metadata}
        metadata_path = root / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f: 
            json.dump(res_meta, f, indent=4)
        print(f"INFO: Saved metadata to {metadata_path}")

        # Group images by resolution for efficient batch processing
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
            # Encode text
            captions = [m['caption'] for m in batch_meta]
            embeds, pooled = compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device)
            
            # Load and process images
            images, valid_meta = [], []
            for m in batch_meta:
                try:
                    with Image.open(m['ip']) as img: 
                        img = img.convert("RGB")
                    images.append(transform(resize_to_fit(img, w, h)))
                    valid_meta.append(m)
                except Exception as e: 
                    tqdm.write(f"\n[ERROR] Skipping {m['ip']}: {e}")
            
            if not images: 
                continue

            # Encode latents
            image_batch_tensor = torch.stack(images).to(device, dtype=torch.float32)
            
            with torch.no_grad():
                latents = vae.encode(image_batch_tensor).latent_dist.mean * vae.config.scaling_factor
            
            # Save each item
            for j, m in enumerate(valid_meta):
                cache_path = cache_dir / f"{m['ip'].stem}.pt"
                torch.save({
                    "original_size": m["original_size"], 
                    "target_size": (w, h),
                    "embeds": embeds[j].cpu().to(torch.float32), 
                    "pooled": pooled[j].cpu().to(torch.float32),
                    "latents": latents[j].cpu().to(torch.float32)
                }, cache_path)
            
            # Clear memory periodically
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        print(f"INFO: Completed caching for {root}")
    
    # Cleanup - move text encoders back to CPU (VAE cleanup handled by main())
    print("\nINFO: Moving text encoders back to CPU...")
    te1.cpu()
    te2.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("CACHING COMPLETE")
    print("="*60 + "\n")


class ImageTextLatentDataset(Dataset):
    def __init__(self, config):
        self.latent_files = []
        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / ".precomputed_embeddings_cache"
            if not cache_dir.exists(): continue
            
            # Load files and shuffle
            files = list(cache_dir.glob("*.pt")) 
            self.latent_files.extend(files * int(ds.get("repeats", 1)))
            
        if not self.latent_files: raise ValueError("No cached files found.")
        
        # Shuffle the entire collection of files ONCE at the start.
        print("INFO: Shuffling the entire dataset order...")
        random.shuffle(self.latent_files)
        
        self.bucket_keys = []
        all_meta = {}
        for ds in config.INSTANCE_DATASETS:
            meta_path = Path(ds["path"]) / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f: all_meta.update(json.load(f))
        
        # Build bucket keys based on the now-shuffled file list
        for f in self.latent_files:
            key = all_meta.get(f.stem)
            self.bucket_keys.append(tuple(key) if key else None)
        print(f"Dataset initialized with {len(self.latent_files)} samples.")

    def __len__(self): return len(self.latent_files)
    
    def __getitem__(self, i):
        try:
            data = torch.load(self.latent_files[i], map_location="cpu")
            return {
                "latents": data["latents"], "embeds": data["embeds"], "pooled": data["pooled"],
                "original_sizes": data["original_size"], "target_sizes": data["target_size"],
                "latent_path": str(self.latent_files[i])
            }
        except Exception as e:
            print(f"WARNING: Skipping {self.latent_files[i]}: {e}")
            return None

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
    alphas = 1.0 - betas; alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone(); alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar / F.pad(alphas_bar[:-1], (1, 0), value=1.0)
    return 1 - alphas


def filter_scheduler_config(config, scheduler_class):
    return {k: v for k, v in config.items() if k in inspect.signature(scheduler_class.__init__).parameters}


def create_optimizer(config, params_to_optimize):
    """Create optimizer based on config settings."""
    optimizer_type = config.OPTIMIZER_TYPE.lower()
    
    if optimizer_type == "raven":
        print("\n--- Initializing Raven Optimizer ---")
        raven_params = getattr(config, 'RAVEN_PARAMS', default_config.RAVEN_PARAMS)
        print(f"Raven Parameters: {raven_params}")
        
        optimizer = RavenAdamW(
            params_to_optimize,
            lr=config.LEARNING_RATE,
            betas=tuple(raven_params.get('betas', [0.9, 0.999])),
            eps=raven_params.get('eps', 1e-8),
            weight_decay=raven_params.get('weight_decay', 0.01)
        )
    
    elif optimizer_type == "adafactor":
        print("\n--- Initializing Adafactor Optimizer ---")
        adafactor_params = getattr(config, 'ADAFACTOR_PARAMS', default_config.ADAFACTOR_PARAMS)
        print(f"Adafactor Parameters: {adafactor_params}")
        
        optimizer = Adafactor(
            params_to_optimize,
            lr=config.LEARNING_RATE,
            eps=tuple(adafactor_params.get('eps', [1e-30, 1e-3])),
            clip_threshold=adafactor_params.get('clip_threshold', 1.0),
            decay_rate=adafactor_params.get('decay_rate', -0.8),
            beta1=adafactor_params.get('beta1', None),
            weight_decay=adafactor_params.get('weight_decay', 0.01),
            scale_parameter=adafactor_params.get('scale_parameter', True),
            relative_step=adafactor_params.get('relative_step', False),
            warmup_init=adafactor_params.get('warmup_init', False)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'raven' or 'adafactor'.")
    
    return optimizer


def main():
    config = TrainingConfig()
    if config.SEED: set_seed(config.SEED)
        
    OUTPUT_DIR = Path(config.OUTPUT_DIR)
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Only load models for caching if caching is needed
    if check_if_caching_needed(config):
        print("="*60)
        print("LOADING MODELS FOR CACHING")
        print("="*60)
        
        # Check if we have a dedicated VAE file
        vae = load_vae_only(config, device)
        
        if vae is None:
            # No dedicated VAE, need to load from main checkpoint
            print("WARNING: No dedicated VAE file. Loading pipeline once to extract text encoders AND VAE...")
            print(f"Loading pipeline to CPU from: {config.SINGLE_FILE_CHECKPOINT_PATH}")
            
            pipe = StableDiffusionXLPipeline.from_single_file(
                config.SINGLE_FILE_CHECKPOINT_PATH, 
                torch_dtype=torch.float32,
                device_map=None  # Keep on CPU
            )
            
            # Extract everything we need
            tokenizer = pipe.tokenizer
            tokenizer_2 = pipe.tokenizer_2
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            vae = pipe.vae
            
            # Move VAE to device
            print(f"INFO: Moving VAE to {device}...")
            vae = vae.to(device)
            
            # Delete pipeline
            del pipe
            gc.collect()
            
            print("INFO: Extracted all components from pipeline")
        else:
            # We have a dedicated VAE, just load text encoders
            print("Loading text encoders from main checkpoint...")
            pipe = StableDiffusionXLPipeline.from_single_file(
                config.SINGLE_FILE_CHECKPOINT_PATH, 
                torch_dtype=config.compute_dtype
            )
            tokenizer = pipe.tokenizer
            tokenizer_2 = pipe.tokenizer_2
            text_encoder = pipe.text_encoder
            text_encoder_2 = pipe.text_encoder_2
            
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        
        # Now cache with all the components
        precompute_and_cache_latents(
            config, 
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            vae,
            device
        )
        
        # Clean up after caching
        del tokenizer, tokenizer_2, text_encoder, text_encoder_2, vae
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("\n" + "="*60)
        print("SKIPPING CACHING - All datasets already cached")
        print("="*60 + "\n")

    model_to_load = Path(config.RESUME_MODEL_PATH) if config.RESUME_TRAINING and Path(config.RESUME_MODEL_PATH).exists() else Path(config.SINGLE_FILE_CHECKPOINT_PATH)
    print(f"\nLoading UNet for training from: {model_to_load}")

    pipe = StableDiffusionXLPipeline.from_single_file(model_to_load, torch_dtype=config.compute_dtype)
    original_scheduler_config = pipe.scheduler.config
    unet = pipe.unet
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    base_model_state_dict = load_file(model_to_load)

    print("\n--- Using Original Scheduler Config as Base ---")
    print(f"Original model's scheduler config: {original_scheduler_config}")

    # --- Dynamic Noise Scheduler Selection ---
    SCHEDULER_MAP = {
        "DDPMScheduler": DDPMScheduler,
        "DDIMScheduler": DDIMScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
    }

    scheduler_name_from_config = getattr(config, 'NOISE_SCHEDULER', 'DDPMScheduler')
    scheduler_name = scheduler_name_from_config.replace(" (Experimental)", "")
    scheduler_class = SCHEDULER_MAP.get(scheduler_name)

    if not scheduler_class:
        raise ValueError(f"Unknown noise scheduler: '{scheduler_name_from_config}'. Available options are: {list(SCHEDULER_MAP.keys())}")
        
    training_scheduler_config = original_scheduler_config.copy()
    training_scheduler_config['prediction_type'] = config.PREDICTION_TYPE

    valid_scheduler_config = filter_scheduler_config(training_scheduler_config, scheduler_class)
    noise_scheduler = scheduler_class.from_config(valid_scheduler_config)
    
    print(f"INFO: Training with {type(noise_scheduler).__name__} and prediction type '{noise_scheduler.config.prediction_type}'")
    if "Euler" in scheduler_name:
        print("WARNING: EulerDiscreteScheduler is experimental for training and may produce unstable results.")

    # Sanity checks for scheduler compatibility
    if config.PREDICTION_TYPE == 'v_prediction' and not hasattr(noise_scheduler, 'get_velocity'):
        raise ValueError(f"Scheduler {scheduler_name} does not support 'v_prediction'. Please use 'epsilon' or select a different scheduler (e.g., DDPMScheduler).")

    if config.USE_ZERO_TERMINAL_SNR:
        print("INFO: Rescaling betas for Zero Terminal SNR.")
        if hasattr(noise_scheduler, 'betas'):
            noise_scheduler.betas = rescale_zero_terminal_snr(noise_scheduler.betas)
            noise_scheduler.alphas = 1.0 - noise_scheduler.betas
            noise_scheduler.alphas_cumprod = torch.cumprod(noise_scheduler.alphas, dim=0)
        else:
            print(f"WARNING: Scheduler {scheduler_name} does not have a 'betas' attribute. Cannot apply Zero Terminal SNR.")
    
    unet.enable_xformers_memory_efficient_attention() if config.MEMORY_EFFICIENT_ATTENTION == "xformers" else unet.set_attn_processor(AttnProcessor2_0())
    unet.to(device)
    unet.enable_gradient_checkpointing()
    
    print("\n--- UNet Layer Selection Report ---")
    print(f"Exclusion Keywords: {config.UNET_EXCLUDE_TARGETS}")
    
    params_to_optimize = []
    trainable_layer_names = []
    frozen_layer_names = []
    
    for name, param in unet.named_parameters():
        param.requires_grad = True
        if any(k in name for k in config.UNET_EXCLUDE_TARGETS):
            param.requires_grad = False
            frozen_layer_names.append(name)
        else:
            trainable_layer_names.append(name)

    params_to_optimize = [p for p in unet.parameters() if p.requires_grad]
    
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params_count = sum(p.numel() for p in params_to_optimize)
    
    print(f"Total UNet Parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params_count / 1e6:.2f}M ({trainable_params_count/total_params*100:.2f}%)")
    
    print("\nExample Trainable Layers:")
    for name in trainable_layer_names[:5]: print(f"  + {name}")
    print("\nExample Frozen Layers:")
    for name in frozen_layer_names[:5]: print(f"  - {name}")
    print("---------------------------------")
    
    if not params_to_optimize: raise ValueError("No parameters were selected for training. Check exclusion list.")

    optimizer = create_optimizer(config, params_to_optimize)

    lr_custom_curve = getattr(config, 'LR_CUSTOM_CURVE', [[0.0, 0.0], [0.1, config.LEARNING_RATE], [1.0, 0.0]])

    print(f"\n--- Custom LR Scheduler ---")
    print(f"Curve Points: {lr_custom_curve}")
    print(f"Max Training Steps: {config.MAX_TRAIN_STEPS}")

    lr_scheduler = CustomCurveLRScheduler(
        optimizer=optimizer,
        curve_points=lr_custom_curve,
        max_train_steps=config.MAX_TRAIN_STEPS
    )
    
    dataset = ImageTextLatentDataset(config)
    sampler = BucketBatchSampler(dataset, config.BATCH_SIZE, config.SEED, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=custom_collate_fn, num_workers=config.NUM_WORKERS)
    
    global_step = 0
    state_path = Path(config.RESUME_STATE_PATH)
    if config.RESUME_TRAINING and state_path.exists():
        print(f"Loading optimizer state from {state_path.name}...")
        state = torch.load(state_path, map_location="cpu")
        global_step = state['step'] * config.GRADIENT_ACCUMULATION_STEPS
        optimizer.load_state_dict(state['optimizer_state_dict'])
        del state
        gc.collect()
        print(f"Resumed at step {global_step}")

    is_fp32_model = next(unet.parameters()).dtype == torch.float32
    use_grad_scaler = is_fp32_model and config.MIXED_PRECISION in ["float16", "fp16"]
    scaler = torch.amp.GradScaler('cuda', enabled=use_grad_scaler)

    if use_grad_scaler:
        print("INFO: Base model is FP32 and MIXED_PRECISION is 'fp16'. GradScaler is ENABLED.")
    else:
        print(f"INFO: GradScaler is DISABLED. (Reason: Is base model FP32? {is_fp32_model}. Is precision 'fp16'? {config.MIXED_PRECISION in ['float16', 'fp16']})")

    is_v_pred = config.PREDICTION_TYPE == "v_prediction"
    
    test_param_name = trainable_layer_names[0] if trainable_layer_names else None
    if not test_param_name: raise ValueError("No trainable layers found to monitor.")
    test_param = dict(unet.named_parameters())[test_param_name]
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS, test_param_name)
    
    unet.train()
    progress_bar = tqdm(range(global_step, config.MAX_TRAIN_STEPS), desc="Training Steps", initial=global_step, total=config.MAX_TRAIN_STEPS)
    
    accumulated_latent_paths = []
    done = False
    while not done:
        for batch in dataloader:
            if global_step >= config.MAX_TRAIN_STEPS: done = True; break
            if not batch: continue
            
            if "latent_path" in batch:
                accumulated_latent_paths.extend(batch["latent_path"])

            latents = batch["latents"].to(device, non_blocking=True)
            embeds = batch["embeds"].to(device, non_blocking=True)
            pooled = batch["pooled"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):
                time_ids = torch.cat([torch.tensor(list(s1) + [0,0] + list(s2)).unsqueeze(0) for s1, s2 in zip(batch["original_sizes"], batch["target_sizes"])], dim=0).to(device, dtype=embeds.dtype)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                target = noise_scheduler.get_velocity(latents, noise, timesteps) if is_v_pred else noise

                pred = unet(noisy_latents, timesteps, embeds, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
            
            diagnostics.step(loss.item())
            scaler.scale(loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                before_val = test_param.data.clone()
                scaler.unscale_(optimizer)

                raw_grad_norm = 0.0
                for p in params_to_optimize:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2).item()
                        raw_grad_norm += param_norm ** 2
                raw_grad_norm = raw_grad_norm ** 0.5
                
                if config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, config.CLIP_GRAD_NORM)
                    clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM)
                else:
                    clipped_grad_norm = raw_grad_norm

                if raw_grad_norm > config.GRAD_SPIKE_THRESHOLD_HIGH or raw_grad_norm < config.GRAD_SPIKE_THRESHOLD_LOW:
                    tqdm.write("\n" + "="*20 + " GRADIENT ANOMALY DETECTED " + "="*20)
                    tqdm.write(f"  Step: {global_step + 1}")
                    tqdm.write(f"  Raw Gradient Norm: {raw_grad_norm:.4f}")
                    tqdm.write(f"  Clipped Gradient Norm: {clipped_grad_norm:.4f}")
                    tqdm.write(f"  Thresholds (Low/High): {config.GRAD_SPIKE_THRESHOLD_LOW} / {config.GRAD_SPIKE_THRESHOLD_HIGH}")
                    tqdm.write("  Images in Accumulated Batch:")
                    for path in accumulated_latent_paths:
                        tqdm.write(f"    - {Path(path).stem}")
                    tqdm.write("="*65 + "\n")
                
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step(global_step + 1)
                optimizer.zero_grad(set_to_none=True)
                after_val = test_param.data.clone()
                
                avg_loss = sum(diagnostics.losses) / len(diagnostics.losses) if diagnostics.losses else 0
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
                
                diagnostics.report(global_step + 1, optimizer, raw_grad_norm, clipped_grad_norm, before_val, after_val)
                accumulated_latent_paths.clear()

                current_optim_step = (global_step + 1) // config.GRADIENT_ACCUMULATION_STEPS
                if current_optim_step > 0 and (current_optim_step % config.SAVE_EVERY_N_STEPS == 0):
                    print(f"\nSaving checkpoint at optimizer step {current_optim_step}...")
                    state_path = CHECKPOINT_DIR / f"state_step_{current_optim_step}.pt"
                    torch.save({'step': current_optim_step, 'optimizer_state_dict': optimizer.state_dict()}, state_path)
                    
                    model_path = CHECKPOINT_DIR / f"model_step_{current_optim_step}.safetensors"
                    sd_to_save = base_model_state_dict.copy()
                    unet_sd = unet.state_dict()
                    key_map = _generate_hf_to_sd_unet_key_mapping(list(unet_sd.keys()))
                    for hf_key in trainable_layer_names:
                        if (mapped := key_map.get(hf_key)) and (sd_key := 'model.diffusion_model.' + mapped) in sd_to_save:
                            sd_to_save[sd_key] = unet_sd[hf_key].cpu().to(config.compute_dtype)
                    save_file(sd_to_save, model_path)
                    del sd_to_save, unet_sd
                    print(f"Checkpoint saved.")
            
            global_step += 1
            progress_bar.update(1)

    progress_bar.close()
    print("\nTraining complete.")

    final_optim_step = config.MAX_TRAIN_STEPS // config.GRADIENT_ACCUMULATION_STEPS
    basename = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_final"
    
    print("Saving final model and state...")
    torch.save({'step': final_optim_step, 'optimizer_state_dict': optimizer.state_dict()}, OUTPUT_DIR / f"{basename}_state.pt")
    
    final_model_path = OUTPUT_DIR / f"{basename}.safetensors"
    final_sd = base_model_state_dict.copy()
    unet_sd = unet.state_dict()
    key_map = _generate_hf_to_sd_unet_key_mapping(list(unet_sd.keys()))
    for hf_key in trainable_layer_names:
        if (mapped := key_map.get(hf_key)) and (sd_key := 'model.diffusion_model.' + mapped) in final_sd:
            final_sd[sd_key] = unet_sd[hf_key].cpu().to(config.compute_dtype)
    save_file(final_sd, final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()