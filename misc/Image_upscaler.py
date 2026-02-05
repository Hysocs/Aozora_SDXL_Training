# Save this file as gui_upscaler.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import threading
import os
import sys
import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageTk
from pathlib import Path
import random
import math

# --- MODEL CONFIGURATION ---
MODELS = {
    "Anime (Fast, Lightweight)": {
        "name": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "params": {"num_block": 6, "num_feat": 64}
    },
    "General (Slow, Powerful)": {
        "name": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "params": {"num_block": 23, "num_feat": 64}
    }
}

# --- CORE LOGIC ---
class AIEngine:
    def __init__(self, device, model_key, tile_size=0):
        self.device = device
        self.upsampler = None
        self.model_key = model_key # MODIFICATION: Store the key to know which model is loaded
        if model_key not in MODELS: raise ValueError(f"Model key '{model_key}' not found.")
        model_config = MODELS[model_key]
        try:
            import torchvision.transforms.functional as F
            sys.modules["torchvision.transforms.functional_tensor"] = F
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            model_name = model_config["name"]
            model_path = os.path.join('weights', model_name)
            if not os.path.exists(model_path):
                print(f"INFO: Downloading AI model ({model_name})...")
                os.makedirs('weights', exist_ok=True)
                r = requests.get(model_config["url"], allow_redirects=True)
                with open(model_path, 'wb') as f: f.write(r.content)
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=model_config["params"]["num_feat"],
                            num_block=model_config["params"]["num_block"], num_grow_ch=32, scale=4)
            self.upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=tile_size,
                                          tile_pad=10, pre_pad=0, half=True if "cuda" in str(device) else False,
                                          device=device)
        except Exception as e:
            raise RuntimeError(f"Could not initialize AI Engine: {e}\nPlease run: pip install basicsr realesrgan opencv-python")

    def upscale(self, pil_image):
        # The image passed here will already be a flattened RGB image
        if not self.upsampler: return None
        img_np = np.array(pil_image)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        try:
            output, _ = self.upsampler.enhance(img_cv2, outscale=4)
            return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"WARNING: AI upscale failed: {e}")
            return None

def fill_alpha_grey(img, color=(124, 124, 124)):
    """
    Detects transparency and fills it with a solid grey color
    BEFORE any resizing occurs to prevent black edge artifacts.
    """
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, color)
        bg.paste(img, mask=img.split()[3]) # Use alpha channel as mask
        return bg
    return img.convert("RGB")

def smart_resize(image, target_w, target_h):
    src_w, src_h = image.size
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h
    
    if src_aspect > target_aspect:
        crop_w, crop_h = src_h * target_aspect, src_h
        x_offset, y_offset = (src_w - crop_w) / 2.0, 0.0
    else:
        crop_w, crop_h = src_w, src_w / target_aspect
        x_offset, y_offset = 0.0, (src_h - crop_h) / 2.0
        
    box = (x_offset, y_offset, x_offset + crop_w, y_offset + crop_h)
    return image.resize((target_w, target_h), resample=Image.Resampling.LANCZOS, box=box)

def calculate_resolution(width, height, target_area, stride=64):
    scale = math.sqrt(target_area / (width * height))
    target_w = int(round((width * scale) / stride) * stride)
    target_h = int(round((height * scale) / stride) * stride)
    return max(target_w, stride), max(target_h, stride)

# --- PREVIEW WINDOW ---
class PreviewWindow(Toplevel):
    def __init__(self, master, pil_a, pil_b):
        super().__init__(master)
        self.title("Preview (1:1) - Original vs. Processed - Click or ESC to Close")
        self.attributes('-fullscreen', True)
        self.configure(bg="#050505")
        
        self.img_b = pil_b # Final processed image
        w, h = self.img_b.size
        # Resize original to match final for slider comparison
        self.img_a = smart_resize(pil_a, w, h)
        
        self.tk_img = None
        self.slider_pos = 0.5
        
        self.canvas = tk.Canvas(self, bg="#050505", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<Motion>", self.on_move)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Button-1>", lambda e: self.destroy())
        self.render()

    def on_move(self, event):
        self.slider_pos = max(0, min(1, event.x / self.canvas.winfo_width()))
        self.render()

    def render(self):
        w, h = self.img_b.size
        split = int(w * self.slider_pos)
        comp = self.img_b.copy()
        if split > 0: comp.paste(self.img_a.crop((0,0,split,h)), (0,0))
        draw = ImageDraw.Draw(comp)
        draw.line([(split,0), (split,h)], fill="#00ff00", width=2)
        
        self.tk_img = ImageTk.PhotoImage(comp)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

# --- MAIN GUI ---
class UpscalerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dataset AI Upscaler & Resizer")
        self.geometry("650x450")
        self.configure(bg="#2E2E2E")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_selection = tk.StringVar(value=list(MODELS.keys())[0])
        self.pixel_area = tk.StringVar(value="1048576") # 1024*1024
        self.tile_size = tk.StringVar(value="0")
        self.status_text = tk.StringVar(value="Ready. Select a folder to begin.")
        self.progress_var = tk.DoubleVar()
        
        # MODIFICATION: Add an attribute to hold the persistent AI engine instance
        self.ai_engine = None

        main_frame = tk.Frame(self, padx=15, pady=15, bg="#2E2E2E")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(1, weight=1)

        tk.Label(main_frame, text="Input Folder:", bg="#2E2E2E", fg="white").grid(row=0, column=0, sticky="w", pady=2)
        tk.Entry(main_frame, textvariable=self.input_path, state="readonly").grid(row=0, column=1, sticky="ew")
        tk.Button(main_frame, text="Browse...", command=self.select_input).grid(row=0, column=2, padx=5)

        tk.Label(main_frame, text="Output Folder:", bg="#2E2E2E", fg="white").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(main_frame, textvariable=self.output_path, state="readonly").grid(row=1, column=1, sticky="ew")
        tk.Button(main_frame, text="Browse...", command=self.select_output).grid(row=1, column=2, padx=5)

        options_frame = tk.LabelFrame(main_frame, text="Settings", padx=10, pady=10, bg="#2E2E2E", fg="white")
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=15)
        options_frame.grid_columnconfigure(1, weight=1)

        tk.Label(options_frame, text="AI Model:", bg="#2E2E2E", fg="white").grid(row=0, column=0, sticky="w")
        ttk.Combobox(options_frame, textvariable=self.model_selection, values=list(MODELS.keys()), state="readonly").grid(row=0, column=1, columnspan=2, sticky="ew", padx=5)
        
        tk.Label(options_frame, text="Target Pixel Area:", bg="#2E2E2E", fg="white").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(options_frame, textvariable=self.pixel_area, width=15).grid(row=1, column=1, sticky="w", padx=5)

        tk.Label(options_frame, text="Tile Size (VRAM):", bg="#2E2E2E", fg="white").grid(row=2, column=0, sticky="w")
        tk.Entry(options_frame, textvariable=self.tile_size, width=15).grid(row=2, column=1, sticky="w", padx=5)

        action_frame = tk.Frame(main_frame, bg="#2E2E2E")
        action_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
        action_frame.columnconfigure(0, weight=1); action_frame.columnconfigure(1, weight=1)

        self.preview_button = tk.Button(action_frame, text="Test Random Image", command=self.start_preview_thread, bg="#555555", fg="white", relief=tk.FLAT, padx=10, pady=5)
        self.preview_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        
        self.start_button = tk.Button(action_frame, text="START BATCH PROCESS", command=self.start_processing_thread, bg="#007ACC", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT, padx=10, pady=5)
        self.start_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        tk.Label(main_frame, textvariable=self.status_text, bg="#2E2E2E", fg="#CCCCCC").grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Progressbar(main_frame, variable=self.progress_var, mode='determinate').grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)

    def select_input(self):
        path = filedialog.askdirectory()
        if path:
            self.input_path.set(path)
            if not self.output_path.get():
                self.output_path.set(str(Path(path).parent / f"{Path(path).name}_upscaled"))

    def select_output(self):
        path = filedialog.askdirectory()
        if path: self.output_path.set(path)

    # MODIFICATION: New method to manage the AI engine instance
    def get_ai_engine(self):
        """
        Creates or retrieves the AI engine instance.
        Reloads the engine if the model selection has changed.
        """
        model_key = self.model_selection.get()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if the engine needs to be created or reloaded
        if self.ai_engine is None or self.ai_engine.model_key != model_key:
            try:
                self.status_text.set(f"Loading AI model: {model_key}...")
                self.update_idletasks()  # Force GUI update
                
                # Explicitly free memory if an old engine exists
                if self.ai_engine:
                    del self.ai_engine.upsampler
                    del self.ai_engine
                    if "cuda" in device:
                        torch.cuda.empty_cache()

                tile = int(self.tile_size.get())
                self.ai_engine = AIEngine(device, model_key, tile_size=tile)
                self.status_text.set("AI Model loaded successfully.")
            except Exception as e:
                self.ai_engine = None  # Reset on failure
                messagebox.showerror("Model Load Error", str(e))
                self.status_text.set(f"Error loading model: {e}")
                return None
        return self.ai_engine
        
    def _get_image_files(self):
        input_p = Path(self.input_path.get())
        if not input_p.is_dir(): return []
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        return [p for ext in valid_exts for p in input_p.rglob(f"*{ext}")]

    def _process_single_image(self, img_path, ai_engine):
        target_area = int(self.pixel_area.get())
        with Image.open(img_path) as img:
            # STEP 1: Fill alpha with grey IMMEDIATELY after loading.
            img_clean = fill_alpha_grey(img)
            w_orig, h_orig = img_clean.size
            
            # STEP 2: Decide if upscaling is needed.
            source_for_resize = img_clean
            if (w_orig * h_orig) < target_area:
                upscaled = ai_engine.upscale(img_clean)
                if upscaled: source_for_resize = upscaled

            # STEP 3: Resize to final target dimensions.
            target_w, target_h = calculate_resolution(w_orig, h_orig, target_area)
            final_img = smart_resize(source_for_resize, target_w, target_h)
            return img_clean, final_img # Return the CLEANED original for preview

    def start_preview_thread(self):
        image_files = self._get_image_files()
        if not image_files:
            messagebox.showerror("Error", "No images found in input folder.")
            return
        
        self.preview_button.config(state="disabled", text="Testing...")
        random_image_path = random.choice(image_files)
        thread = threading.Thread(target=self.run_preview, args=(random_image_path,), daemon=True)
        thread.start()
        
    def run_preview(self, img_path):
        try:
            # MODIFICATION: Use the get_ai_engine method instead of creating a new instance
            ai_engine = self.get_ai_engine()
            if not ai_engine:
                self.status_text.set("Preview failed: AI model not loaded.")
                return

            self.status_text.set(f"Previewing: {img_path.name}")
            original_img_clean, final_img = self._process_single_image(img_path, ai_engine)
            
            self.after(0, lambda: PreviewWindow(self, original_img_clean, final_img))
            self.status_text.set("Preview complete. Press ESC or click to close.")
        except Exception as e:
            messagebox.showerror("Preview Error", str(e))
            self.status_text.set(f"Preview error: {e}")
        finally:
            self.preview_button.config(state="normal", text="Test Random Image")

    def start_processing_thread(self):
        if not self.input_path.get() or not self.output_path.get():
            messagebox.showerror("Error", "Please select input and output folders.")
            return
        self.start_button.config(state="disabled", text="Processing...")
        self.preview_button.config(state="disabled")
        self.progress_var.set(0)
        thread = threading.Thread(target=self.processing_loop, daemon=True)
        thread.start()

    def processing_loop(self):
        try:
            # MODIFICATION: Use the get_ai_engine method instead of creating a new instance
            ai_engine = self.get_ai_engine()
            if not ai_engine:
                self.status_text.set("Processing failed: AI model not loaded.")
                return
            
            input_p, output_p = Path(self.input_path.get()), Path(self.output_path.get())
            image_files = self._get_image_files()
            total_files = len(image_files)
            if total_files == 0:
                self.status_text.set("No images found."); return

            for i, img_path in enumerate(image_files):
                self.status_text.set(f"Processing [{i+1}/{total_files}]: {img_path.name}")
                try:
                    _, final_img = self._process_single_image(img_path, ai_engine)
                    if final_img:
                        relative_path = img_path.relative_to(input_p)
                        output_file = output_p / relative_path
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        final_img.save(output_file.with_suffix('.png'), "PNG")
                except Exception as e:
                    print(f"Skipping {img_path.name} due to error: {e}")
                
                self.progress_var.set((i + 1) / total_files * 100)

            self.status_text.set(f"Success! {total_files} images processed.")
            messagebox.showinfo("Success", f"All images processed and saved to:\n{output_p}")
        except Exception as e:
            messagebox.showerror("Fatal Error", str(e))
            self.status_text.set(f"Fatal error during processing: {e}")
        finally:
            self.start_button.config(state="normal", text="START BATCH PROCESS")
            self.preview_button.config(state="normal")

if __name__ == "__main__":
    app = UpscalerApp()
    app.mainloop()