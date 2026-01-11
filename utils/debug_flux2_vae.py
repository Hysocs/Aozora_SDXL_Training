import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk, ImageDraw, ImageOps
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import numpy as np
import threading
from pathlib import Path
import math
import time
import os
import requests
import cv2
import sys

# ==============================================================================
#  AI ENGINE (RealESRGAN) - Optional
# ==============================================================================
class AIEngine:
    def __init__(self, device):
        self.device = device
        self.has_basicsr = False
        try:
            # Compatibility Patch for newer Torchvision
            import torchvision.transforms.functional as F
            sys.modules["torchvision.transforms.functional_tensor"] = F

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            self.has_basicsr = True
            
            model_path = 'weights/RealESRGAN_x4plus.pth'
            if not os.path.exists(model_path):
                os.makedirs('weights', exist_ok=True)
                url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                r = requests.get(url, allow_redirects=True)
                with open(model_path, 'wb') as f: f.write(r.content)
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True if "cuda" in str(device) else False, device=device)
        except: pass

    def upscale(self, pil_image):
        # Input is guaranteed to be RGB (No Alpha) by the time it gets here
        if not self.has_basicsr: return pil_image
        img_np = np.array(pil_image)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        try:
            output, _ = self.upsampler.enhance(img_cv2, outscale=4)
            return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        except: return pil_image

# ==============================================================================
#  CORE LOGIC (Grey Fill + Smart Resize)
# ==============================================================================

def fill_alpha_grey(img, color=(124, 124, 124)):
    """
    Detects transparency and fills it with a solid grey color
    BEFORE any resizing occurs to prevent black edge artifacts.
    """
    # Handle Palette images with transparency info
    if img.mode == 'P':
        img = img.convert('RGBA')
    
    # Handle Grayscale + Alpha
    if img.mode == 'LA':
        img = img.convert('RGBA')

    # If image has an Alpha channel
    if img.mode == 'RGBA':
        # Create a solid grey background
        bg = Image.new('RGB', img.size, color)
        # Paste the image on top using the alpha channel as a mask
        bg.paste(img, mask=img.split()[3])
        return bg
    
    # If it's just RGB/L, verify it is RGB
    return img.convert("RGB")

def smart_resize(image, target_w, target_h, use_lanczos=True):
    # Image is already flattened to RGB Grey here.
    src_w, src_h = image.size
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h
    
    # Calculate Float Crop Box to preserve aspect ratio
    if src_aspect > target_aspect:
        crop_w = src_h * target_aspect
        crop_h = src_h
        x_offset = (src_w - crop_w) / 2.0
        y_offset = 0.0
    else:
        crop_w = src_w
        crop_h = src_w / target_aspect
        x_offset = 0.0
        y_offset = (src_h - crop_h) / 2.0
        
    box = (x_offset, y_offset, x_offset + crop_w, y_offset + crop_h)
    
    # Single Pass Resampling
    filter_type = Image.Resampling.LANCZOS if use_lanczos else Image.Resampling.BICUBIC
    return image.resize((target_w, target_h), resample=filter_type, box=box)

class ResolutionCalculator:
    def __init__(self, target_area, stride=64):
        self.target_area = target_area
        self.stride = stride
        
    def calculate_resolution(self, width, height):
        # 1. Scale based on pixel AREA (Dynamic aspect ratio)
        current_area = width * height
        scale = math.sqrt(self.target_area / current_area)
        
        # 2. Calculate ideal dimensions
        scaled_w = width * scale
        scaled_h = height * scale
        
        # 3. Snap to stride
        target_w = int(round(scaled_w / self.stride) * self.stride)
        target_h = int(round(scaled_h / self.stride) * self.stride)
        return max(target_w, self.stride), max(target_h, self.stride)

# ==============================================================================
#  FULL RESOLUTION INSPECTOR WINDOW
# ==============================================================================
class FullResWindow(Toplevel):
    def __init__(self, master, pil_a, pil_b, title="Inspector"):
        super().__init__(master)
        self.title(f"{title} (1:1 Pixel Perfect)")
        self.geometry("1200x800")
        self.configure(bg="#050505")
        
        self.img_b = pil_b
        w, h = self.img_b.size
        
        # Match A to B size for comparison slider
        if pil_a.size != (w, h):
            self.img_a = smart_resize(pil_a, w, h, use_lanczos=True)
        else:
            self.img_a = pil_a

        self.tk_img = None
        self.slider_pos = 0.5
        
        self.frame = tk.Frame(self, bg="#050505")
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.frame, bg="#050505", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Motion>", self.on_move)
        self.bind("<Escape>", lambda e: self.destroy())
        self.render()

    def on_move(self, event):
        cw = self.canvas.winfo_width()
        self.slider_pos = max(0, min(1, event.x / cw)) if cw > 0 else 0.5
        self.render()

    def render(self):
        w, h = self.img_b.size
        split = int(w * self.slider_pos)
        comp = self.img_b.copy()
        if split > 0:
            comp.paste(self.img_a.crop((0,0,split,h)), (0,0))
            
        draw = ImageDraw.Draw(comp)
        draw.line([(split,0), (split,h)], fill="#00ff00", width=1)
        
        self.tk_img = ImageTk.PhotoImage(comp)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.config(scrollregion=(0,0,w,h))

# ==============================================================================
#  MAIN GUI
# ==============================================================================
class ZoomableSlider(tk.Canvas):
    def __init__(self, parent, label_text):
        super().__init__(parent, bg="#151515", highlightthickness=1, highlightbackground="#333")
        self.img_a = None
        self.img_b = None
        self.tk_img = None
        self.slider_pos = 0.5
        self.bind("<Motion>", self.on_move)
        self.bind("<Button-1>", self.on_click) 
        self.bind("<Configure>", self.render)
        self.label_text = label_text

    def load_images(self, pil_a, pil_b):
        self.img_a = pil_a
        self.img_b = pil_b 
        self.render()

    def on_click(self, event):
        if self.img_a and self.img_b:
            FullResWindow(self.winfo_toplevel(), self.img_a, self.img_b, self.label_text)

    def on_move(self, event):
        self.slider_pos = event.x / self.winfo_width()
        self.render()

    def render(self, event=None):
        if not self.img_b: return
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10: return
        
        ratio = min(w/self.img_b.width, h/self.img_b.height)
        dw, dh = int(self.img_b.width * ratio), int(self.img_b.height * ratio)
        
        display_b = self.img_b.resize((dw, dh), Image.Resampling.BILINEAR)
        display_a = smart_resize(self.img_a, dw, dh, use_lanczos=False)
        
        split = int(dw * self.slider_pos)
        comp = display_b.copy()
        if split > 0:
            comp.paste(display_a.crop((0, 0, split, dh)), (0, 0))
            
        draw = ImageDraw.Draw(comp)
        draw.line([(split, 0), (split, dh)], fill="cyan", width=2)
        
        self.tk_img = ImageTk.PhotoImage(comp)
        self.delete("all")
        ox, oy = (w - dw) // 2, (h - dh) // 2
        self.create_image(ox, oy, anchor=tk.NW, image=self.tk_img)
        self.create_text(10, 10, anchor=tk.NW, text=f"{self.label_text} (Click for 1:1)", fill="white", font=("Arial", 9, "bold"))

class VAEDebugger(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Pipeline Inspector (Grey Fill Fix)")
        self.geometry("1400x900")
        self.configure(bg="#101010")
        
        self.vae = None
        self.ai_engine = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_files = []
        
        threading.Thread(target=self.init_ai, daemon=True).start()
        
        bar = tk.Frame(self, bg="#202020", height=50)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="1. Dataset", command=self.sel_in, bg="#444", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="2. Load VAE", command=self.sel_vae, bg="#444", fg="white").pack(side=tk.LEFT, padx=5)
        self.res_var = tk.IntVar(value=1024*1024)
        tk.Entry(bar, textvariable=self.res_var, width=10).pack(side=tk.LEFT, padx=5)
        self.use_ai = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="AI Upscale", variable=self.use_ai, bg="#202020", fg="#00ff00", selectcolor="#333").pack(side=tk.LEFT, padx=15)
        self.use_packing = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Flux Mode", variable=self.use_packing, bg="#202020", fg="cyan", selectcolor="#333").pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="▶ PROCESS", command=self.run_process, bg="#007acc", fg="white", font=("Arial", 10, "bold")).pack(side=tk.RIGHT, padx=20)
        self.status = tk.Label(bar, text="Init AI...", bg="#202020", fg="yellow")
        self.status.pack(side=tk.RIGHT, padx=10)

        container = tk.Frame(self, bg="#101010")
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)
        
        self.view_bucket = ZoomableSlider(container, "Resize Check")
        self.view_bucket.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.view_vae = ZoomableSlider(container, "VAE Check")
        self.view_vae.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        self.lbl_info = tk.Label(self, text="...", bg="#000", fg="#ccc", font=("Consolas", 10))
        self.lbl_info.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    def init_ai(self):
        self.ai_engine = AIEngine(self.device)
        self.status.config(text="AI Ready" if self.ai_engine.has_basicsr else "AI Missing", fg="#00ff00" if self.ai_engine.has_basicsr else "red")

    def sel_in(self):
        d = filedialog.askdirectory()
        if d: 
            self.input_files = [p for p in Path(d).glob("*") if p.suffix.lower() in {'.jpg','.png','.webp','.bmp'}]
            self.status.config(text=f"Images: {len(self.input_files)}")

    def sel_vae(self):
        f = filedialog.askopenfilename(filetypes=[("Safetensors", "*.safetensors")])
        if f: threading.Thread(target=self.load_vae, args=(f,), daemon=True).start()

    def load_vae(self, path):
        try:
            try: self.vae = AutoencoderKL.from_single_file(path).to(self.device)
            except: self.vae = AutoencoderKL.from_single_file(path, latent_channels=32, ignore_mismatched_sizes=True).to(self.device)
            self.vae.enable_tiling()
            ch = self.vae.config.latent_channels
            self.status.config(text=f"VAE: {ch}ch", fg="#00ff00")
            if ch == 16: self.use_packing.set(True)
        except Exception as e: self.status.config(text=f"VAE Err: {e}", fg="red")

    def run_process(self):
        if not self.input_files: return
        import random
        path = random.choice(self.input_files)
        threading.Thread(target=self.process_image, args=(path,), daemon=True).start()

    def process_image(self, path):
        self.status.config(text=f"Processing {path.name}...")
        calc = ResolutionCalculator(self.res_var.get())
        
        try:
            with Image.open(path) as img:
                w_orig, h_orig = img.size
                
                # 1. Fill Alpha with Grey IMMEDIATELY (Prevents Black Edges)
                img_clean = fill_alpha_grey(img)
                
                target_w, target_h = calc.calculate_resolution(w_orig, h_orig)
                
                # 2. Resize / Upscale (Using the already flattened RGB image)
                if self.use_ai.get() and self.ai_engine.has_basicsr:
                    high_res = self.ai_engine.upscale(img_clean)
                    bucket_img = smart_resize(high_res, target_w, target_h)
                else:
                    bucket_img = smart_resize(img_clean, target_w, target_h)

                # 3. VAE
                t_val = transforms.ToTensor()(bucket_img).unsqueeze(0).to(self.device)
                t_val = (t_val - 0.5) * 2.0 
                
                with torch.no_grad():
                    dist = self.vae.encode(t_val.to(dtype=torch.float32)).latent_dist
                    latents = dist.sample()
                    decoded = self.vae.decode(latents).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    decoded_np = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
                    vae_rec = Image.fromarray((decoded_np * 255).astype(np.uint8))
            
            # Update GUI: Show the flattened Original vs Flattened Bucket
            self.view_bucket.load_images(img_clean, bucket_img)
            self.view_vae.load_images(bucket_img, vae_rec)
            self.lbl_info.config(text=f"Orig: {w_orig}x{h_orig} | Bucket: {target_w}x{target_h}")
            self.status.config(text="Done.", fg="#00ff00")
        except Exception as e:
            self.status.config(text=f"Err: {e}", fg="red")
            print(e)

if __name__ == "__main__":
    app = VAEDebugger()
    app.mainloop()