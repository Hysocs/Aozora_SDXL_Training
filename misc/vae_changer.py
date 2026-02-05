import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm # Used for a progress bar

def swap_vae_and_save_single_file(model_path, vae_path, output_path):
    """
    Loads an SDXL model from a single .safetensors file, swaps its VAE from
    another .safetensors file, and saves the result as a new single
    .safetensors file.

    Args:
        model_path (str): Path to the source SDXL .safetensors model.
        vae_path (str): Path to the standalone .safetensors VAE.
        output_path (str): Path to save the new .safetensors model file.
    """
    try:
        # Load the state dictionaries from the model and VAE files
        print(f"Loading SDXL model from: {model_path}")
        model_state_dict = load_file(model_path)
        
        print(f"Loading new VAE from: {vae_path}")
        vae_state_dict = load_file(vae_path)

        # Create a new state dictionary for the output model, starting with the original
        output_state_dict = model_state_dict.copy()

        # Isolate the VAE keys from the new VAE state dict
        # Typically, VAE keys in standalone files don't have a prefix.
        # Inside the main model, they are prefixed with "first_stage_model."
        vae_keys = {k for k in vae_state_dict.keys()}
        
        print(f"Found {len(vae_keys)} tensors in the new VAE file.")
        
        # Replace the corresponding VAE tensors in the main model's state dict
        print("Swapping VAE tensors...")
        for key in tqdm(vae_keys, desc="Replacing VAE keys"):
            # The key in the main model is usually prefixed
            prefixed_key = f"first_stage_model.{key}"
            
            # Check if this key exists in the main model before replacing
            if prefixed_key in output_state_dict:
                output_state_dict[prefixed_key] = vae_state_dict[key]
            else:
                # Fallback for models that might not use the prefix (less common)
                if key in output_state_dict:
                    output_state_dict[key] = vae_state_dict[key]

        # Save the modified state dictionary to a new single .safetensors file
        print(f"Saving new model to: {output_path}")
        # The `metadata` parameter ensures any original model metadata is preserved.
        save_file(output_state_dict, output_path, metadata=model_state_dict.get('__metadata__'))
        
        print("\nModel with swapped VAE saved successfully as a single file!")
        print(f"New model location: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Make sure to use the raw string prefix (r"...") or double backslashes (\\) for Windows paths.
    
    # Path to your original SDXL model file
    original_model_path = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\StableDiffusion\hikarimagineXL_hikarimagineExp.safetensors"
    
    # Path to your new VAE file
    new_vae_path = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\VAE\Goodv1.safetensors"
    
    # Path where the new SINGLE .safetensors file will be saved
    new_model_output_path = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\StableDiffusion\hikarimagineXL_hikarimagineExp_Baked_VAE.safetensors"

    # --- Run the script ---
    swap_vae_and_save_single_file(original_model_path, new_vae_path, new_model_output_path)