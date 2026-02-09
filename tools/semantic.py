import cv2
import numpy as np
import torch
from PIL import Image

def generate_character_map(pil_image):
    """
    Generates a map focusing on broad color regions and subjects.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    np_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Bilateral filter smooths colors while preserving edges
    bilateral = cv2.bilateralFilter(np_image_bgr, d=5, sigmaColor=50, sigmaSpace=50)
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Use color deviation to find salient regions
    mean_a, mean_b = np.mean(a_channel), np.mean(b_channel)
    saliency_a = np.abs(a_channel.astype(np.float32) - mean_a)
    saliency_b = np.abs(b_channel.astype(np.float32) - mean_b)
    color_saliency = saliency_a + saliency_b
   
    if color_saliency.max() > 1e-6:
        color_saliency_norm = color_saliency / color_saliency.max()
    else:
        color_saliency_norm = np.zeros_like(color_saliency, dtype=np.float32)
   
    # **FIX:** Reduced blur kernel to (9, 9) to match visualizer.
    # This creates a tighter, more accurate mask around the subject.
    final_map_np = cv2.GaussianBlur(color_saliency_norm, (9, 9), 0)
   
    return final_map_np # Return numpy array directly

def generate_detail_map(pil_image):
    """
    Generates a map focusing on lines, edges, and high-frequency details.
    """
    np_image_gray = np.array(pil_image.convert("L"))
    
    # Sobel filters detect edges
    sobelx = cv2.Sobel(np_image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(np_image_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
   
    if magnitude.max() > 1e-6:
        magnitude_norm = magnitude / magnitude.max()
    else:
        magnitude_norm = np.zeros_like(magnitude, dtype=np.float32)
   
    # **FIX:** Reduced blur kernel to (3, 3) to match visualizer.
    # This provides just enough anti-aliasing without "boxing" empty space.
    final_map_np = cv2.GaussianBlur(magnitude_norm.astype(np.float32), (3, 3), 0)
   
    return final_map_np # Return numpy array directly


def generate_semantic_map_batch_softmax(images, char_weight, detail_weight, 
                                        target_size, device, dtype, num_channels):
    """
    The final function called by the trainer.
    Now perfectly aligned with the visualizer's logic.
    """
    batch_maps = []
   
    for img in images:
        if img is None:
            # Use target_size (W, H) to create a tensor of (H, W)
            batch_maps.append(torch.zeros((target_size[1], target_size[0]), dtype=dtype))
            continue
        
        # 1. Generate Raw Maps (now returns numpy)
        char_np = generate_character_map(img)
        detail_np = generate_detail_map(img)
        
        # 2. Linear Combination (Stable and Aligned)
        combined = (char_np * char_weight) + (detail_np * detail_weight)
        
        # Clip max value to prevent extreme spikes, but allow high values
        combined = np.clip(combined, 0.0, 10.0) 
        
        # 3. Downsample Safely to Target Latent Size
        # `target_size` from the trainer is the latent resolution (e.g., 128x128)
        # We must use BILINEAR to correctly average the values, matching the visualizer.
        # This is the step that makes the map look "weaker", but is mathematically correct.
        
        # Convert to PIL for reliable resizing
        weight_map_pil = Image.fromarray(combined, mode='F')
        
        if weight_map_pil.size != target_size:
            weight_map_pil = weight_map_pil.resize(
                target_size, Image.Resampling.BILINEAR
            )
        
        # Convert final resized map to a tensor
        weight_map_tensor = torch.from_numpy(np.array(weight_map_pil)).to(dtype)
        batch_maps.append(weight_map_tensor)
    
    # Stack batch and expand to match latent channels
    final_map_batch = torch.stack(batch_maps).to(device)
    final_map_batch = final_map_batch.unsqueeze(1).expand(-1, num_channels, -1, -1)
    
    return final_map_batch