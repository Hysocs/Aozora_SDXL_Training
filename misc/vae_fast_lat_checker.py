from safetensors.torch import load_file

# Replace with the path to your VAE or Checkpoint file
model_path = r"C:\Users\Administrator\Desktop\StabilityMatrix-win-x64\Data\Models\StableDiffusion\chenkinnoobXLV02_v02.safetensors"

# Load the file
tensors = load_file(model_path)

# Key layers to check for channel size:
# Encoder output (Standard SDXL VAE)
if "first_stage_model.encoder.conv_out.weight" in tensors:
    shape = tensors["first_stage_model.encoder.conv_out.weight"].shape
    print(f"Encoder Output Channels: {shape[0]}") # Should be 8 (mean + logvar, so 4*2)
    
# Decoder input (Standard SDXL VAE)
if "first_stage_model.decoder.conv_in.weight" in tensors:
    shape = tensors["first_stage_model.decoder.conv_in.weight"].shape
    print(f"Decoder Input Channels: {shape[1]}") # Should be 4

# Check for separate VAE file naming conventions
if "encoder.conv_out.weight" in tensors:
    shape = tensors["encoder.conv_out.weight"].shape
    print(f"VAE Encoder Output Channels: {shape[0]}") # Should be 8 (4 latent channels * 2)