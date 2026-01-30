import torch
from safetensors.torch import load_file, save_file
import os

# ================= CONFIGURATION =================
# 1. The "Broken" Model (We keep the UNet and CLIP from here)
#    This model MUST be the one that "loads with no crashes" currently.
BROKEN_MODEL_PATH = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\StableDiffusion\noobaiFlux2vae_03Aesthetic.safetensors"

# 2. The "Good" Flux VAE (ae.safetensors from Black Forest Labs)
FLUX_VAE_PATH = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\VAE\ae.safetensors"

# 3. Output Filename
OUTPUT_PATH = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\StableDiffusion\Trained\fixed_sdxl_flux_hybrid_SWAP_ONLY.safetensors"
# =================================================

def swap_vae():
    if not os.path.exists(BROKEN_MODEL_PATH):
        print(f"Error: Base model not found at {BROKEN_MODEL_PATH}")
        return
    if not os.path.exists(FLUX_VAE_PATH):
        print(f"Error: Flux VAE not found at {FLUX_VAE_PATH}")
        return

    print(f"Loading Base Model: {BROKEN_MODEL_PATH}...")
    base_state = load_file(BROKEN_MODEL_PATH)
    
    print(f"Loading Flux VAE: {FLUX_VAE_PATH}...")
    vae_state = load_file(FLUX_VAE_PATH)
    
    new_state_dict = {}

    print("Processing weights...")

    # --- STEP 1: Copy UNet and CLIP from Base ---
    # We copy everything EXCEPT the VAE keys.
    # VAE keys in SDXL usually start with "first_stage_model." or "vae."
    
    kept_count = 0
    for key, value in base_state.items():
        if key.startswith("first_stage_model.") or key.startswith("vae."):
            # Skip the old "Bad" VAE
            continue
        else:
            # Keep UNet (model.diffusion_model) and CLIP (conditioner)
            new_state_dict[key] = value
            kept_count += 1
            
    print(f"   -> Kept {kept_count} keys from Base Model (UNet + CLIP).")

    # --- STEP 2: Inject New Flux VAE ---
    # Flux VAE keys usually start with "encoder." or "decoder."
    # We map them to "first_stage_model." so SDXL recognizes them.
    
    vae_count = 0
    for key, value in vae_state.items():
        new_key = None
        
        # Standard Flux VAE mapping
        if key.startswith("encoder.") or key.startswith("decoder."):
            new_key = f"first_stage_model.{key}"
        elif key.startswith("quant_conv") or key.startswith("post_quant_conv"):
             new_key = f"first_stage_model.{key}"
        
        # Some VAEs might already have the prefix, handle gracefully
        elif key.startswith("first_stage_model."):
            new_key = key

        if new_key:
            new_state_dict[new_key] = value
            vae_count += 1

    print(f"   -> Injected {vae_count} keys from Flux VAE.")

    # --- STEP 3: Save ---
    print(f"Saving to {OUTPUT_PATH}...")
    save_file(new_state_dict, OUTPUT_PATH)
    print("Done! This model has the Base UNet + Official Flux VAE.")

if __name__ == "__main__":
    swap_vae()