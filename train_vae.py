import inspect
import re
from pathlib import Path
import glob
import gc
import os
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
from diffusers import AutoencoderKL
from transformers import Adafactor, get_cosine_schedule_with_warmup
import cv2
import numpy as np
from PIL import Image, TiffImagePlugin, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Pool
import warnings
from safetensors.torch import load_file
try:
    import lpips
except ImportError:
    lpips = None
try:
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
except ImportError:
    MemoryEfficientAttentionFlashAttentionOp = None
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
warnings.filterwarnings("ignore", category=UserWarning, module=TiffImagePlugin.__name__, message="Corrupt EXIF data")
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

class VaeTrainingConfig:
    def __init__(self):
        # Default configuration values
        self.VAE_MODEL_PATH = r"C:\Users\Administrator\Documents\VAE\sdxl_vae-fp16fix-blessed.safetensors"  # Path to pre-trained VAE
        self.INSTANCE_DATA_DIR = r"C:\Users\Administrator\Pictures\GameCg"  # Folder with training images
        self.OUTPUT_DIR = "./vae_output"  # Where to save fine-tuned VAE
        self.RESOLUTION = 512  # Target image resolution (SDXL VAE native)
        self.BATCH_SIZE = 1  # Minimum batch size to reduce VRAM
        self.NUM_WORKERS = 2  # Reduced to minimize CPU load
        self.MAX_TRAIN_STEPS = 5000  # Total training steps
        self.LR = 1e-5  # Learning rate
        self.LR_WARMUP_PERCENT = 0.1  # Warmup steps as percentage of max steps
        self.KL_WEIGHT = 1e-6  # KL divergence loss weight
        self.USE_PERCEPTUAL_LOSS = False  # Disabled to save VRAM
        self.PERCEPTUAL_LOSS_WEIGHT = 0.5  # Weight for perceptual loss (if enabled)
        self.MIXED_PRECISION = "float16"  # fp16 for lower VRAM
        self.SEED = 42  # Random seed
        self.SAVE_EVERY_N_STEPS = 5000  # Save checkpoint frequency
        self.CHECKPOINT_DIR = Path(self.OUTPUT_DIR) / "checkpoints"  # Checkpoint directory
        self.GRADIENT_ACCUMULATION_STEPS = 16  # Increased for effective larger batch
        self.TILE_SIZE = 512  # Tile size for VAE processing
        self.TEST_IMAGE_PATH = None  # Optional: Set to specific image path for testing, e.g., "C:\path\to\test.jpg"
        
        # Load user config if exists
        user_config_path = Path("vae_config.json")
        if user_config_path.exists():
            print(f"INFO: Loading user configuration from {user_config_path}")
            try:
                with open(user_config_path, 'r') as f:
                    user_config = json.load(f)
                for key, value in user_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR: Could not read or parse vae_config.json: {e}. Using default settings.")
        
        # Convert types for safety
        float_keys = ["LR", "LR_WARMUP_PERCENT", "KL_WEIGHT", "PERCEPTUAL_LOSS_WEIGHT"]
        int_keys = ["MAX_TRAIN_STEPS", "BATCH_SIZE", "NUM_WORKERS", "SEED", "SAVE_EVERY_N_STEPS", "RESOLUTION", "GRADIENT_ACCUMULATION_STEPS", "TILE_SIZE"]
        for key in float_keys:
            if hasattr(self, key):
                try:
                    setattr(self, key, float(getattr(self, key)))
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Could not convert {key}: {e}. Using default.")
        for key in int_keys:
            if hasattr(self, key):
                try:
                    setattr(self, key, int(getattr(self, key)))
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Could not convert {key}: {e}. Using default.")

def validate_image(ip, resolution):
    """Validate a single image and extract metadata."""
    if "_truncated" in ip.stem:
        return None, None
    try:
        file_type = imghdr.what(ip)
        if file_type not in ['jpeg', 'png', 'webp', 'bmp']:
            raise ValueError(f"Invalid image format: {file_type}")
        img_array = np.fromfile(str(ip), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load image with cv2")
        meta_info = {"ip": ip}
        return meta_info, None
    except Exception as e:
        tqdm.write(f"\n[CORRUPT IMAGE DETECTED] Path: {ip}")
        tqdm.write(f"  └─ Reason: {e}")
        tqdm.write(f"  └─ Action: Renaming file to quarantine it.")
        try:
            new_stem = f"{ip.stem}_truncated"
            new_image_path = ip.with_name(new_stem + ip.suffix)
            ip.rename(new_image_path)
            tqdm.write(f"  └─ Renamed image to: {new_image_path.name}")
        except Exception as rename_e:
            tqdm.write(f"  └─ [ERROR] Could not rename file: {rename_e}")
        return None, ip

def validate_wrapper(args):
    ip, resolution = args
    return validate_image(ip, resolution)

class VaeDataset(Dataset):
    def __init__(self, data_root, resolution=512):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.image_paths = [Path(p) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] for p in self.data_root.rglob(f"*{ext}")]
        if not self.image_paths:
            raise ValueError(f"No images found in {data_root} or its subdirectories.")
        
        print(f"Found {len(self.image_paths)} potential image files. Validating...")
        image_args = [(ip, resolution) for ip in self.image_paths]
        num_processes = min(4, multiprocessing.cpu_count())  # Limited to 4 to reduce CPU load
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(validate_wrapper, image_args), total=len(self.image_paths), desc="Validating images"))
        self.image_paths = [meta["ip"] for meta, _ in results if meta]
        
        if not self.image_paths:
            raise ValueError("No valid, non-corrupted images found after scanning.")
        print(f"Validation complete. Found {len(self.image_paths)} valid images.")
        
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # SDXL VAE normalization
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, i):
        try:
            img_array = np.fromfile(str(self.image_paths[i]), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to load image with cv2")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            return self.transform(img)
        except Exception as e:
            print(f"WARNING: Skipping bad image {self.image_paths[i]}: {e}")
            return None

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.stack(batch)

def generate_reconstruction(vae, test_image_path, output_path, compute_dtype, device):
    """Generate and save a reconstruction of a test image, return MSE."""
    if not Path(test_image_path).exists():
        print(f"Test image not found: {test_image_path}")
        return None
    try:
        img_array = np.fromfile(str(test_image_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load test image with cv2")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
    except Exception as e:
        print(f"ERROR: Failed to load test image {test_image_path}: {e}")
        return None
    
    # Save original image for comparison
    original_output_path = output_path.parent / f"{output_path.stem}_original.png"
    img.save(original_output_path)
    print(f"Saved original test image to: {original_output_path}")
    
    transform = transforms.Compose([
        transforms.Resize((vae.config.sample_size, vae.config.sample_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device, dtype=compute_dtype)
    
    vae.eval()  # Set to evaluation mode
    with torch.no_grad():
        posterior = vae.encode(img_tensor).latent_dist
        latents = posterior.sample()
        reconstruction = vae.decode(latents / vae.config.scaling_factor).sample
    reconstruction = (reconstruction.squeeze(0) + 1) / 2
    reconstruction = reconstruction.clamp(0, 1)
    
    # Compute MSE
    mse = F.mse_loss(reconstruction, transform(img).to(device, dtype=compute_dtype)).item()
    print(f"MSE for {output_path.name}: {mse:.6f}")
    
    reconstruction_img = transforms.ToPILImage()(reconstruction.cpu())
    reconstruction_img.save(output_path)
    print(f"Saved reconstruction to: {output_path}")
    
    # Cleanup
    del img_tensor, posterior, latents, reconstruction
    gc.collect()
    torch.cuda.empty_cache()
    return mse

def save_vae_checkpoint(vae, step, checkpoint_dir, config, compute_dtype):
    checkpoint_path = checkpoint_dir / f"vae_checkpoint_step_{step}"
    vae.save_pretrained(checkpoint_path, torch_dtype=compute_dtype)
    print(f"[OK] Saved VAE checkpoint to: {checkpoint_path}")
    gc.collect()
    torch.cuda.empty_cache()

def fine_tune_vae(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float16  # Hardcode fp16 for consistency
    print(f"Using compute dtype: {compute_dtype}, Device: {device}")
    
    # Optimize CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # Create output directories
    output_dir = Path(config.OUTPUT_DIR)
    checkpoint_dir = config.CHECKPOINT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Hardcoded SDXL VAE parameters
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        sample_size=config.RESOLUTION,
        scaling_factor=0.18215,  # Corrected for SDXL VAE
        force_upcast=False,
    )
    
    # Load weights
    vae_path = Path(config.VAE_MODEL_PATH)
    if vae_path.is_file():
        weight_file = vae_path
    else:
        weight_file = next(vae_path.glob("*.safetensors"), None) or next(vae_path.glob("*.bin"), None) or next(vae_path.glob("*.pt"), None)
        if weight_file is None:
            raise FileNotFoundError("No weights file found in VAE directory (.safetensors, .bin, or .pt)")
    
    print(f"Loading VAE weights from: {weight_file}")
    with torch.no_grad():
        if weight_file.suffix == ".safetensors":
            state_dict = load_file(weight_file)
        else:
            state_dict = torch.load(weight_file, map_location="cpu")
        # Convert all weights to float16 and check for corruption
        state_dict = {k: v.to(dtype=compute_dtype) for k, v in state_dict.items()}
        sample_tensor = next(iter(state_dict.values()))
        print(f"Sample weight tensor stats: min={sample_tensor.min().item():.4f}, max={sample_tensor.max().item():.4f}")
        if torch.isnan(sample_tensor).any() or torch.isinf(sample_tensor).any():
            raise ValueError("Loaded weights contain NaN or Inf values, likely corrupted.")
    
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    vae.to(device, dtype=compute_dtype)
    vae.train()
    vae.requires_grad_(True)
    
    # Enhanced tiling with specified tile size
    def set_vae_tiling(vae, tile_size):
        vae.enable_tiling()
        if hasattr(vae, 'set_tiling_parameters'):
            vae.set_tiling_parameters(tile_size=tile_size, overlap=16)
        print(f"Enabled VAE tiling with tile size {tile_size}x{tile_size}")
    
    set_vae_tiling(vae, config.TILE_SIZE)
    vae.enable_slicing()
    
    # Enable gradient checkpointing
    if hasattr(vae, 'enable_gradient_checkpointing'):
        vae.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing for VAE.")
    
    # Enable xFormers if available
    if MemoryEfficientAttentionFlashAttentionOp is not None:
        vae.enable_xformers_memory_efficient_attention()
        print("Enabled xFormers memory-efficient attention.")
    
    # Optionally compile the model
    try:
        vae = torch.compile(vae, mode="reduce-overhead")
        print("Compiled VAE model for performance.")
    except Exception as e:
        print(f"WARNING: Could not compile VAE model: {e}")
    
    # Optimizer: Prefer 8-bit Adam if available
    if bnb is not None:
        optimizer = bnb.optim.Adam8bit(vae.parameters(), lr=config.LR)
        print("Using 8-bit Adam optimizer for lower VRAM.")
    else:
        optimizer = Adafactor(vae.parameters(), lr=config.LR, scale_parameter=False, relative_step=False)
        print("Falling back to Adafactor optimizer.")
    
    num_update_steps = config.MAX_TRAIN_STEPS
    num_warmup_steps = int(config.MAX_TRAIN_STEPS * config.LR_WARMUP_PERCENT)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_update_steps
    )
    
    # Losses
    mse_loss = torch.nn.MSELoss()
    perceptual_loss = lpips.LPIPS(net='vgg').to(device, dtype=compute_dtype) if config.USE_PERCEPTUAL_LOSS and lpips else None
    
    # Dataset and Dataloader with custom collate
    dataset = VaeDataset(config.INSTANCE_DATA_DIR, config.RESOLUTION)
    
    # Select test images (try up to 3 to avoid bad ones)
    test_image_path = None
    if config.TEST_IMAGE_PATH and Path(config.TEST_IMAGE_PATH).exists():
        test_image_path = Path(config.TEST_IMAGE_PATH)
        print(f"Using user-specified test image: {test_image_path}")
    else:
        valid_images = dataset.image_paths
        if not valid_images:
            raise ValueError("No valid images found for reconstruction test.")
        for _ in range(3):  # Try up to 3 images
            candidate = random.choice(valid_images)
            if validate_image(candidate, config.RESOLUTION)[0] is not None:
                test_image_path = candidate
                break
        if test_image_path is None:
            raise ValueError("Could not find a valid test image after 3 attempts.")
        print(f"Selected test image: {test_image_path}")
    
    # Pre-training reconstruction
    print("Generating pre-training reconstruction...")
    pre_recon_path = output_dir / "base_reconstruction.png"
    pre_mse = generate_reconstruction(vae, test_image_path, pre_recon_path, compute_dtype, device)
    
    # Dataset and Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.NUM_WORKERS > 0),
        drop_last=True,
        collate_fn=custom_collate
    )
    
    # Training Loop with gradient accumulation
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    global_step = 0
    accum_step = 0
    progress_bar = tqdm(range(config.MAX_TRAIN_STEPS), desc="VAE Fine-Tuning Steps")
    
    for epoch in range(int(math.ceil(config.MAX_TRAIN_STEPS / len(dataloader)))):
        for batch in dataloader:
            if batch is None:
                continue  # Skip empty batches from collate
            if global_step >= config.MAX_TRAIN_STEPS:
                break
                
            images = batch.to(device, dtype=compute_dtype)
            
            # Encode and decode with CPU offloading
            with torch.autocast(device_type=device.type, dtype=compute_dtype):
                posterior = vae.encode(images).latent_dist
                latents = posterior.sample().cpu()  # Offload to CPU
                reconstructions = vae.decode(latents.to(device, dtype=compute_dtype) / vae.config.scaling_factor).sample
                
                # Losses
                recon_loss = mse_loss(reconstructions, images)
                kl_loss = posterior.kl().mean()
                loss = recon_loss + config.KL_WEIGHT * kl_loss
                
                if perceptual_loss:
                    perc_loss = perceptual_loss(reconstructions, images).mean()
                    loss += config.PERCEPTUAL_LOSS_WEIGHT * perc_loss
            
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            accum_step += 1
            if accum_step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                accum_step = 0
            
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}", "vram": f"{torch.cuda.memory_allocated()/1e9:.2f}GB"})
            
            if global_step > 0 and global_step % config.SAVE_EVERY_N_STEPS == 0:
                save_vae_checkpoint(vae, global_step, checkpoint_dir, config, compute_dtype)
                
            gc.collect()
            torch.cuda.empty_cache()
    
    progress_bar.close()
    
    # Post-training reconstruction
    print("Generating post-training reconstruction...")
    post_recon_path = output_dir / "fine_tuned_reconstruction.png"
    post_mse = generate_reconstruction(vae, test_image_path, post_recon_path, compute_dtype, device)
    
    # Compare MSE
    if pre_mse is not None and post_mse is not None:
        print(f"Reconstruction MSE improvement: Pre={pre_mse:.6f}, Post={post_mse:.6f}, Reduction={(pre_mse - post_mse)/pre_mse*100:.2f}%")
    
    # Save final VAE
    final_vae_path = output_dir / "fine_tuned_vae"
    vae.save_pretrained(final_vae_path, torch_dtype=compute_dtype)
    print(f"Final VAE saved to: {final_vae_path}")
    
    # Cleanup
    vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return final_vae_path

def main():
    config = VaeTrainingConfig()
    if not Path(config.INSTANCE_DATA_DIR).exists():
        raise FileNotFoundError(f"Image directory not found: {config.INSTANCE_DATA_DIR}")
    if not Path(config.VAE_MODEL_PATH).exists():
        raise FileNotFoundError(f"VAE model path not found: {config.VAE_MODEL_PATH}")
    
    fine_tune_vae(config)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("INFO: Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"INFO: Multiprocessing context already set or an error occurred: {e}")
    
    os.environ['PYTHONUNBUFFERED'] = '1'
    main()