#!/usr/bin/env python3
"""
SDXL Model Inspector
Extracts training scheduler info and settings from checkpoint files
"""

import json
import argparse
from pathlib import Path
from safetensors.torch import safe_open, load_file
import torch

def inspect_model(model_path):
    """Extract all training-relevant information from a model checkpoint."""
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    print("="*70)
    print(f"INSPECTING: {model_path.name}")
    print("="*70)
    
    # Load metadata
    with safe_open(model_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    
    if not metadata:
        print("\n⚠️  WARNING: No metadata found in this checkpoint!")
        print("This is an older format. Falling back to weight analysis...\n")
        analyze_weights_only(model_path)
        return
    
    print("\n" + "="*70)
    print("📋 RAW METADATA")
    print("="*70)
    for key, value in metadata.items():
        print(f"{key}: {value[:200]}..." if len(str(value)) > 200 else f"{key}: {value}")
    
    # Parse scheduler config if present
    scheduler_config = None
    for key in ['scheduler_config', 'noise_scheduler_config', 'scheduler']:
        if key in metadata:
            try:
                scheduler_config = json.loads(metadata[key])
                print(f"\n✅ Found scheduler config in metadata key: '{key}'")
                break
            except:
                pass
    
    # Check for diffusers format config
    if not scheduler_config:
        for key, value in metadata.items():
            if 'scheduler' in key.lower() and isinstance(value, str):
                try:
                    scheduler_config = json.loads(value)
                    print(f"\n✅ Found scheduler config in: '{key}'")
                    break
                except:
                    pass
    
    print("\n" + "="*70)
    print("🔧 TRAINING SCHEDULER SETTINGS")
    print("="*70)
    
    if scheduler_config:
        print("\n📊 Scheduler Configuration:")
        print(json.dumps(scheduler_config, indent=2))
        
        # Extract key training parameters
        print("\n" + "-"*70)
        print("KEY TRAINING PARAMETERS:")
        print("-"*70)
        
        prediction_type = scheduler_config.get('prediction_type', 'UNKNOWN')
        class_name = scheduler_config.get('_class_name', 'UNKNOWN')
        num_train_timesteps = scheduler_config.get('num_train_timesteps', 'UNKNOWN')
        beta_schedule = scheduler_config.get('beta_schedule', 'UNKNOWN')
        beta_start = scheduler_config.get('beta_start', 'UNKNOWN')
        beta_end = scheduler_config.get('beta_end', 'UNKNOWN')
        
        print(f"  Prediction Type:     {prediction_type}")
        print(f"  Recommended Sampler: {class_name}")
        print(f"  Training Timesteps:  {num_train_timesteps}")
        print(f"  Beta Schedule:       {beta_schedule}")
        print(f"  Beta Range:          {beta_start} → {beta_end}")
        
        # Check for zero terminal SNR
        zero_snr = scheduler_config.get('rescale_betas_zero_snr', False)
        print(f"  Zero Terminal SNR:   {zero_snr}")
        
        print("\n" + "="*70)
        print("💡 TRAINING RECOMMENDATIONS")
        print("="*70)
        
        print(f"\n🎯 Use this in your config.json:")
        print("-"*70)
        
        # Determine actual training scheduler
        training_scheduler = "DDIMScheduler"
        if "ddpm" in class_name.lower():
            training_scheduler = "DDPMScheduler"
        
        print(f"""
{{
    "PREDICTION_TYPE": "{prediction_type}",
    "USE_ZERO_TERMINAL_SNR": {str(zero_snr).lower()},
    
    // Training Scheduler Settings (use {training_scheduler})
    "NUM_TRAIN_TIMESTEPS": {num_train_timesteps},
    "BETA_SCHEDULE": "{beta_schedule}",
    "BETA_START": {beta_start},
    "BETA_END": {beta_end}
}}
""")
        
        # V-prediction specific warnings
        if prediction_type == "v_prediction":
            print("\n⚠️  V-PREDICTION MODEL DETECTED")
            print("-"*70)
            print("V-prediction models have higher gradient magnitudes!")
            print("Recommended additional settings:")
            print("""
{{
    "USE_MIN_SNR_GAMMA": true,
    "MIN_SNR_GAMMA": 5.0,
    "LEARNING_RATE": 1e-06,  // Lower LR for v-pred
    "CLIP_GRAD_NORM": 0.5     // More aggressive clipping
}}
""")
        
        if prediction_type == "epsilon":
            print("\n✅ EPSILON PREDICTION MODEL")
            print("-"*70)
            print("Standard stable diffusion training.")
            print("Recommended settings:")
            print("""
{{
    "LEARNING_RATE": 2e-06,
    "CLIP_GRAD_NORM": 1.0
}}
""")
            
    else:
        print("\n❌ No scheduler configuration found in metadata.")
        print("This checkpoint may be using default SDXL settings.")
        print("\nTry these default SDXL training settings:")
        print("""
{
    "PREDICTION_TYPE": "epsilon",
    "USE_ZERO_TERMINAL_SNR": false,
    "NUM_TRAIN_TIMESTEPS": 1000,
    "BETA_SCHEDULE": "scaled_linear",
    "BETA_START": 0.00085,
    "BETA_END": 0.012
}
""")
    
    # Analyze model architecture
    print("\n" + "="*70)
    print("🏗️  MODEL ARCHITECTURE ANALYSIS")
    print("="*70)
    
    analyze_architecture(model_path)
    
    print("\n" + "="*70)
    print("✅ INSPECTION COMPLETE")
    print("="*70)


def analyze_weights_only(model_path):
    """Analyze model when no metadata is available."""
    print("Analyzing model weights for clues...")
    
    try:
        state_dict = load_file(model_path)
        
        # Check for v-prediction indicators in key names
        vpred_indicators = [
            'v_pred', 'velocity', 'v_prediction'
        ]
        
        has_vpred = any(
            any(indicator in key.lower() for indicator in vpred_indicators)
            for key in state_dict.keys()
        )
        
        if has_vpred:
            print("✅ Model appears to use V-PREDICTION (based on weight names)")
        else:
            print("✅ Model likely uses EPSILON prediction (standard)")
        
        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"\nTotal parameters: {total_params / 1e9:.2f}B")
        
        # Check if it's SDXL (should be ~2.5B for UNet)
        if 2.0e9 < total_params < 3.5e9:
            print("Model type: SDXL (2.5B parameters detected)")
            print("\nRecommended default SDXL settings:")
            print("""
{
    "PREDICTION_TYPE": "epsilon",
    "NUM_TRAIN_TIMESTEPS": 1000,
    "BETA_SCHEDULE": "scaled_linear",
    "BETA_START": 0.00085,
    "BETA_END": 0.012
}
""")
        
    except Exception as e:
        print(f"Error analyzing weights: {e}")


def analyze_architecture(model_path):
    """Analyze model architecture for additional insights."""
    
    try:
        state_dict = load_file(model_path)
        
        # Look for specific architecture markers
        unet_keys = [k for k in state_dict.keys() if 'diffusion_model' in k or 'unet' in k]
        
        if not unet_keys:
            print("⚠️  Warning: No UNet keys found. This may not be a standard checkpoint.")
            return
        
        # Check for SDXL-specific features
        has_add_embedding = any('add_embedding' in k for k in unet_keys)
        has_label_emb = any('label_emb' in k for k in unet_keys)
        
        if has_add_embedding or has_label_emb:
            print("✅ SDXL architecture detected (has additional embeddings)")
        else:
            print("⚠️  May be SD 1.5/2.1 architecture (no SDXL embeddings found)")
        
        # Count attention layers (helps understand model complexity)
        attention_keys = [k for k in unet_keys if 'attn' in k.lower()]
        resnet_keys = [k for k in unet_keys if 'resnet' in k.lower() or 'conv' in k.lower()]
        
        print(f"\nArchitecture components:")
        print(f"  Attention layers: ~{len(attention_keys)} parameters")
        print(f"  ResNet/Conv layers: ~{len(resnet_keys)} parameters")
        
        # Check for potential training modifications
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        if lora_keys:
            print(f"\n⚠️  WARNING: LoRA weights detected ({len(lora_keys)} keys)")
            print("   This appears to be a LoRA checkpoint, not a full model.")
        
    except Exception as e:
        print(f"Error during architecture analysis: {e}")


def main():
    # HARDCODED MODEL PATH
    model_path = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\StableDiffusion\rouwei_v080Vpred.safetensors"
    
    print(f"Inspecting hardcoded model: {model_path}\n")
    inspect_model(model_path)


if __name__ == "__main__":
    main()