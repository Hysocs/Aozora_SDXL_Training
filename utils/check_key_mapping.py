# check_key_mapping.py (v4 - Definitive Fix)
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline
import sys
import os
from collections import defaultdict
import re

def convert_hf_key_to_sd(hf_key):
    """
    Converts a single HuggingFace diffusers UNet key to its original
    .safetensors format. This is the final, corrected version.
    """
    key = hf_key

    # Rule 1: Handle ResNet blocks
    if "resnets" in key:
        new_key = re.sub(r"^down_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"input_blocks.{3*int(m.group(1)) + int(m.group(2)) + 1}.0.", key)
        new_key = re.sub(r"^mid_block\.resnets\.(\d+)\.", lambda m: f"middle_block.{2*int(m.group(1))}.", new_key)
        new_key = re.sub(r"^up_blocks\.(\d+)\.resnets\.(\d+)\.", lambda m: f"output_blocks.{3*int(m.group(1)) + int(m.group(2))}.0.", new_key)
        
        new_key = new_key.replace("norm1.", "in_layers.0.")
        new_key = new_key.replace("conv1.", "in_layers.2.")
        new_key = new_key.replace("norm2.", "out_layers.0.")
        new_key = new_key.replace("conv2.", "out_layers.3.")
        new_key = new_key.replace("time_emb_proj.", "emb_layers.1.")
        new_key = new_key.replace("conv_shortcut.", "skip_connection.")
        return new_key

    # Rule 2: Handle Attention blocks (The corrected part)
    if "attentions" in key:
        new_key = re.sub(r"^down_blocks\.(\d+)\.attentions\.(\d+)\.", lambda m: f"input_blocks.{3*int(m.group(1)) + int(m.group(2)) + 1}.1.", key)
        new_key = re.sub(r"^mid_block\.attentions\.0\.", "middle_block.1.", new_key)
        new_key = re.sub(r"^up_blocks\.(\d+)\.attentions\.(\d+)\.", lambda m: f"output_blocks.{3*int(m.group(1)) + int(m.group(2))}.1.", new_key)
        # The ".replace('transformer_blocks.0.', '')" line has been REMOVED. This was the bug.
        return new_key
            
    # Rule 3: Handle Downsamplers/Upsamplers
    if "downsamplers" in key:
        return re.sub(r"^down_blocks\.(\d+)\.downsamplers\.0\.conv\.", lambda m: f"input_blocks.{3*(int(m.group(1))+1)}.0.op.", key)
    if "upsamplers" in key:
        return re.sub(r"^up_blocks\.(\d+)\.upsamplers\.0\.", lambda m: f"output_blocks.{3*int(m.group(1)) + 2}.2.", key)

    # Rule 4: Handle top-level layers
    if key.startswith("conv_in."): return key.replace("conv_in.", "input_blocks.0.0.")
    if key.startswith("conv_norm_out."): return key.replace("conv_norm_out.", "out.0.")
    if key.startswith("conv_out."): return key.replace("conv_out.", "out.2.")
    if key.startswith("time_embedding.linear_1."): return key.replace("time_embedding.linear_1.", "time_embed.0.")
    if key.startswith("time_embedding.linear_2."): return key.replace("time_embedding.linear_2.", "time_embed.2.")
    if key.startswith("add_embedding.linear_1."): return key.replace("add_embedding.linear_1.", "label_emb.0.0.")
    if key.startswith("add_embedding.linear_2."): return key.replace("add_embedding.linear_2.", "label_emb.0.2.")
    
    return None

def verify_keys():
    base_model_path = "C:/Users/Administrator/Documents/Models/aozoraXLVpred_v01AlphaVpred.safetensors"
    if not os.path.exists(base_model_path):
        print("="*80, "\n!!! ERROR: FILE NOT FOUND !!!\n", f"Path '{base_model_path}' is incorrect.", "\n" + "="*80)
        return

    print(f"Loading base model: {os.path.basename(base_model_path)}")
    try:
        base_sd = load_file(base_model_path, device="cpu")
        base_unet_keys = {k for k in base_sd.keys() if k.startswith("model.diffusion_model.")}
        del base_sd
        print(f" -> Found {len(base_unet_keys)} UNet keys in the base safetensors file.")
    except Exception as e: print(f"ERROR: {e}"); return

    print("\nLoading diffusers pipeline...")
    try:
        pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_path, torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True)
        unet_hf_keys = list(pipe.unet.state_dict().keys())
        del pipe
        print(f" -> Found {len(unet_hf_keys)} keys in the diffusers UNet representation.")
    except Exception as e: print(f"ERROR: {e}"); return

    print("\nGenerating key map and checking for mismatches...")
    unmatched_keys, error_summary = [], defaultdict(int)
    for hf_key in unet_hf_keys:
        sd_key = convert_hf_key_to_sd(hf_key)
        if sd_key:
            sd_key_to_check = "model.diffusion_model." + sd_key
            if sd_key_to_check not in base_unet_keys:
                unmatched_keys.append((hf_key, sd_key_to_check))
                error_summary['Mismatched Mapping'] += 1
        else:
            unmatched_keys.append((hf_key, "!!! NO MAPPING RULE FOUND !!!"))
            error_summary['No Mapping Rule'] += 1
    
    print("\n" + "="*80, "\n              KEY MAPPING VERIFICATION REPORT\n" + "="*80)
    total_keys, total_unmatched = len(unet_hf_keys), len(unmatched_keys)
    total_matched = total_keys - total_unmatched
    success_rate = (total_matched / total_keys) * 100 if total_keys > 0 else 0

    print(f"\n[+] SUMMARY:")
    print(f"    - Total Diffusers UNet Keys:  {total_keys}")
    print(f"    - Successfully Matched Keys:  {total_matched}")
    print(f"    - Unmatched Keys:             {total_unmatched}")
    print(f"    - Success Rate:               {success_rate:.2f}%")

    if total_unmatched > 0:
        print("\n[+] UNMATCHED KEY CATEGORIES:")
        for category, count in sorted(error_summary.items()): print(f"    - {category:<30}: {count} keys")
        print("\n[+] EXAMPLE OF FAILED MAPPINGS (first 5):")
        for i, (hf_key, sd_key) in enumerate(unmatched_keys[:5]):
            print(f"  {i+1}. Diffusers Key: {hf_key}\n     Mapped To:     {sd_key} (NOT FOUND)\n")
    else:
        print("\n\n--- ✅✅✅ SUCCESS! All {total_keys} keys were matched successfully! ✅✅✅ ---")

if __name__ == "__main__":
    verify_keys()