import cv2
import numpy as np
import torch
from PIL import Image

def generate_character_map(pil_image):
    """Generates a map focusing on the overall character/object region using color and structure."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    np_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    bilateral = cv2.bilateralFilter(np_image_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    mean_a, mean_b = np.mean(a_channel), np.mean(b_channel)
    saliency_a = np.abs(a_channel.astype(np.float32) - mean_a)
    saliency_b = np.abs(b_channel.astype(np.float32) - mean_b)
    color_saliency = saliency_a + saliency_b
   
    if color_saliency.max() > 0:
        color_saliency_norm = color_saliency / color_saliency.max()
    else:
        color_saliency_norm = np.zeros_like(color_saliency, dtype=np.float32)
   
    color_saliency_uint8 = (color_saliency_norm * 255).astype(np.uint8)
   
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(color_saliency_uint8, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)
   
    eroded_float = eroded.astype(np.float32) / 255.0
    final_map = cv2.GaussianBlur(eroded_float, (11, 11), 0)
   
    return Image.fromarray(final_map)

def generate_detail_map(pil_image):
    """Generates a map focusing on edges and lineart."""
    np_image_gray = np.array(pil_image.convert("L"))
    
    sobelx = cv2.Sobel(np_image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(np_image_gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
   
    if magnitude.max() > 0:
        magnitude_norm = magnitude / magnitude.max()
    else:
        magnitude_norm = np.zeros_like(magnitude, dtype=np.float32)
   
    final_map = cv2.GaussianBlur(magnitude_norm.astype(np.float32), (1, 1), 0)
   
    return Image.fromarray(final_map)


# Training generate_semantic_map_batch_softmax
def generate_semantic_map_batch_softmax(images, char_weight, detail_weight, 
                                        target_size, device, dtype, num_channels):
    batch_maps = []
   
    for img in images:
        if img is None:
            batch_maps.append(torch.zeros(target_size, dtype=dtype))
            continue
           
        char_map_pil = generate_character_map(img)
        detail_map_pil = generate_detail_map(img)
        
        char_np = np.array(char_map_pil).astype(np.float32)
        detail_np = np.array(detail_map_pil).astype(np.float32)
        
        # --- ASYMMETRIC SOFT MAX ---
        # Character: gamma 0.7 lifts mid-tones (prevents washout)
        # Detail: linear preserves sharp peaks
        char_boosted = np.power(char_np, 0.7) * char_weight
        detail_scaled = detail_np * detail_weight
        
        # Soft max - detail sharpness dominates denominator
        combined = np.logaddexp(char_boosted * 4, detail_scaled * 4) / 4
        combined = np.clip(combined, 0, 1)
        
        weight_map = np.power(combined, 0.8)
       
        weight_map_pil = Image.fromarray(weight_map, mode='F').resize(
            target_size, Image.Resampling.LANCZOS
        )
        weight_map_tensor = torch.from_numpy(np.array(weight_map_pil)).float()
        batch_maps.append(weight_map_tensor)

    final_map_batch = torch.stack(batch_maps).unsqueeze(1).to(device, dtype=dtype)
    return final_map_batch.expand(-1, num_channels, -1, -1)