import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import numpy as np
import threading
from pathlib import Path
import math

# --- LOGIC: Resolution & Aspect Ratio ---
class ResolutionCalculator:
    def __init__(self, target_area, stride=64, should_upscale=False):
        self.target_area = target_area
        self.stride = stride
        self.should_upscale = should_upscale

    def calculate_resolution(self, width, height):
        aspect_ratio = width / height
        if not self.should_upscale:
            h = int(math.sqrt(self.target_area / aspect_ratio) // self.stride) * self.stride
            w = int(h * aspect_ratio // self.stride) * self.stride
            return max(w, self.stride), max(h, self.stride)
        scale = math.sqrt(self.target_area / (width * height))
        new_w = int((width * scale) // self.stride) * self.stride
        new_h = int((height * scale) // self.stride) * self.stride
        return max(new_w, self.stride), max(new_h, self.stride)

def fix_alpha(img):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P': img = img.convert('RGBA')
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")

# --- CUSTOM WIDGET: Zoomable Wipe Slider ---
class ZoomableSlider(tk.Canvas):
    def __init__(self, parent, label_text, bg_color="#151515"):
        super().__init__(parent, bg=bg_color, highlightthickness=1, highlightbackground="#333")
        
        # Data
        self.img_a_full = None  # Full resolution PIL
        self.img_b_full = None
        self.tk_img = None
        
        # View State
        self.zoom = 1.0
        self.view_x = 0  # Top-left X of the visible crop in image coords
        self.view_y = 0  # Top-left Y of the visible crop
        self.slider_pos_screen = 0 # Slider position in screen pixels
        
        # Bindings
        self.bind("<Motion>", self.on_mouse_move)
        self.bind("<MouseWheel>", self.on_zoom)     # Windows
        self.bind("<Button-4>", self.on_zoom)       # Linux Scroll Up
        self.bind("<Button-5>", self.on_zoom)       # Linux Scroll Down
        self.bind("<ButtonPress-3>", self.start_pan) # Right Click Pan
        self.bind("<B3-Motion>", self.do_pan)
        self.bind("<Configure>", self.render)
        
        # Overlay Text
        self.label_text = label_text
        self.lbl_id = None

    def load_images(self, pil_a, pil_b):
        self.img_a_full = pil_a
        self.img_b_full = pil_b
        self.reset_view()

    def reset_view(self):
        if not self.img_a_full: return
        self.zoom = 1.0
        self.view_x = 0
        self.view_y = 0
        self.render()

    def start_pan(self, event):
        self.last_pan_x = event.x
        self.last_pan_y = event.y

    def do_pan(self, event):
        if not self.img_a_full: return
        dx = (event.x - self.last_pan_x) / (self.zoom * self.get_fit_ratio())
        dy = (event.y - self.last_pan_y) / (self.zoom * self.get_fit_ratio())
        
        self.view_x -= dx
        self.view_y -= dy
        self.validate_bounds()
        
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.render()

    def on_zoom(self, event):
        if not self.img_a_full: return
        
        # Determine scroll direction
        if event.num == 5 or event.delta < 0: factor = 0.9
        else: factor = 1.1

        # Mouse position relative to canvas (0.0 to 1.0)
        mx_ratio = event.x / self.winfo_width()
        my_ratio = event.y / self.winfo_height()

        # Current width/height of the view in image pixels
        view_w, view_h = self.get_view_size()
        
        # Mouse position in IMAGE coordinates
        mouse_img_x = self.view_x + view_w * mx_ratio
        mouse_img_y = self.view_y + view_h * my_ratio
        
        # Apply Zoom
        new_zoom = max(1.0, min(self.zoom * factor, 50.0))
        if new_zoom == self.zoom: return
        self.zoom = new_zoom
        
        # Calculate new view size
        new_view_w, new_view_h = self.get_view_size()
        
        # Adjust view_x/y so mouse stays under cursor
        self.view_x = mouse_img_x - new_view_w * mx_ratio
        self.view_y = mouse_img_y - new_view_h * my_ratio
        
        self.validate_bounds()
        self.render()

    def get_fit_ratio(self):
        # Ratio to fit the FULL image into current canvas
        cw, ch = self.winfo_width(), self.winfo_height()
        iw, ih = self.img_a_full.width, self.img_a_full.height
        return min(cw/iw, ch/ih)

    def get_view_size(self):
        # Size of the visible crop in image pixels
        iw, ih = self.img_a_full.width, self.img_a_full.height
        # At zoom 1.0, we see the whole image
        return iw / self.zoom, ih / self.zoom

    def validate_bounds(self):
        vw, vh = self.get_view_size()
        iw, ih = self.img_a_full.width, self.img_a_full.height
        
        # Clamp bounds
        self.view_x = max(0, min(self.view_x, iw - vw))
        self.view_y = max(0, min(self.view_y, ih - vh))

    def on_mouse_move(self, event):
        self.slider_pos_screen = event.x
        self.render(lazy=True) # Lazy render just updates the slider line if possible

    def render(self, event=None, lazy=False):
        if not self.img_a_full: return
        
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 5 or ch < 5: return

        # 1. Determine Crop
        vw, vh = self.get_view_size()
        crop_box = (int(self.view_x), int(self.view_y), int(self.view_x + vw), int(self.view_y + vh))
        
        # Crop Original
        crop_a = self.img_a_full.crop(crop_box)
        crop_b = self.img_b_full.crop(crop_box)
        
        # Resize to Canvas (Fit aspect ratio)
        # We want to fill the canvas as much as possible while keeping aspect
        ratio = min(cw / crop_a.width, ch / crop_a.height)
        disp_w = int(crop_a.width * ratio)
        disp_h = int(crop_a.height * ratio)
        
        # Bilinear for smoothness, Nearest for pixel peeping? Bilinear is safer for artifacts check.
        # Use Nearest if zoom is very high > 10
        resample = Image.Resampling.NEAREST if self.zoom > 8 else Image.Resampling.BILINEAR
        
        disp_a = crop_a.resize((disp_w, disp_h), resample)
        disp_b = crop_b.resize((disp_w, disp_h), resample)
        
        # Center in canvas
        off_x = (cw - disp_w) // 2
        off_y = (ch - disp_h) // 2
        
        # 2. Composite based on Slider
        comp = disp_b.copy()
        
        # Calculate slider position relative to the displayed image
        rel_slider_x = self.slider_pos_screen - off_x
        
        if rel_slider_x > 0:
            crop_w = min(rel_slider_x, disp_w)
            if crop_w > 0:
                left_crop = disp_a.crop((0, 0, crop_w, disp_h))
                comp.paste(left_crop, (0, 0))
                
        # Draw Divider
        draw = ImageDraw.Draw(comp)
        if 0 <= rel_slider_x <= disp_w:
            draw.line([(rel_slider_x, 0), (rel_slider_x, disp_h)], fill="cyan", width=2)
            
        # 3. Output to Canvas
        self.tk_img = ImageTk.PhotoImage(comp)
        self.delete("all")
        self.create_image(off_x, off_y, anchor=tk.NW, image=self.tk_img)
        
        # HUD Text
        info = f"{self.label_text} | Zoom: {self.zoom:.1f}x"
        self.create_text(10, 10, anchor=tk.NW, text=info, fill="white", font=("Arial", 10, "bold"), tags="hud")
        self.create_text(10, ch-20, anchor=tk.NW, text="Scroll to Zoom | Right-Click to Pan", fill="#888", font=("Arial", 8), tags="hud")

# --- MAIN APP ---
class UltimateDebugger(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Training Inspector Pro (Zoom + Latent X-Ray)")
        self.geometry("1600x900")
        self.configure(bg="#101010")
        try: self.state('zoomed')
        except: pass

        self.vae = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_files = []
        
        # --- CONTROL BAR ---
        bar = tk.Frame(self, bg="#202020", height=60)
        bar.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_in = tk.Button(bar, text="1. Dataset", command=self.sel_in, bg="#333", fg="white", font=("Arial",9,"bold"))
        self.btn_in.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.btn_vae = tk.Button(bar, text="2. Model/VAE", command=self.sel_vae, bg="#333", fg="white", font=("Arial",9,"bold"))
        self.btn_vae.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(bar, text="Res:", bg="#202020", fg="white").pack(side=tk.LEFT, padx=(15,2))
        self.res_var = tk.IntVar(value=1024)
        tk.Entry(bar, textvariable=self.res_var, width=5).pack(side=tk.LEFT)
        
        tk.Label(bar, text="Filter:", bg="#202020", fg="white").pack(side=tk.LEFT, padx=(15,2))
        self.filter_var = tk.StringVar(value="LANCZOS")
        ttk.Combobox(bar, textvariable=self.filter_var, values=('LANCZOS', 'BICUBIC', 'BOX'), width=8).pack(side=tk.LEFT)
        
        self.btn_go = tk.Button(bar, text="🎲 PROCESS IMAGE", command=self.process, bg="#007acc", fg="white", font=("Arial",10,"bold"))
        self.btn_go.pack(side=tk.RIGHT, padx=20)
        
        self.status = tk.Label(bar, text="Ready.", bg="#202020", fg="#00ff00")
        self.status.pack(side=tk.RIGHT, padx=10)

        # --- 3 PANES ---
        grid = tk.Frame(self, bg="#101010")
        grid.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        grid.rowconfigure(0, weight=1)
        
        self.p1 = ZoomableSlider(grid, "1. BUCKETING\n< Original | Bucket >")
        self.p1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        self.p2 = ZoomableSlider(grid, "2. COMPRESSION\n< Bucket | VAE Rec >")
        self.p2.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        self.p3 = ZoomableSlider(grid, "3. LATENT X-RAY (MACHINE VIEW)\n< VAE Rec | Raw Latents >")
        self.p3.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

    def log(self, msg, col="#00ff00"):
        self.status.config(text=msg, fg=col)
        print(msg)

    def sel_in(self):
        path = filedialog.askdirectory()
        if path:
            self.input_files = [p for p in Path(path).glob("*") if p.suffix.lower() in ['.jpg','.png','.webp']]
            self.log(f"Files: {len(self.input_files)}")

    def sel_vae(self):
        path = filedialog.askopenfilename(filetypes=[("Model", "*.safetensors")])
        if path: threading.Thread(target=self.load_vae, args=(path,), daemon=True).start()

    def load_vae(self, path):
        self.log("Loading VAE...", "orange")
        try:
            self.vae = AutoencoderKL.from_single_file(path).to(self.device, dtype=torch.float32)
            self.vae.enable_tiling()
            self.vae.eval()
            self.log("VAE Loaded (FP32 Tiling)", "#00ff00")
        except Exception as e:
            self.log(f"Error: {e}", "red")

    def visualize_latents(self, latents, target_size):
        # Latents shape: (1, 4, H, W)
        # We want to map channels 0,1,2 to R,G,B to see the "structure"
        # SDXL Latents range approx -3 to +3
        
        l_vis = latents[0, :3, :, :].clone().cpu().float()
        
        # Normalize roughly [-3, 3] -> [0, 1]
        l_vis = (l_vis + 2.0) / 4.0
        l_vis = torch.clamp(l_vis, 0, 1)
        
        # To Image
        l_vis = l_vis.permute(1, 2, 0).numpy() # H, W, C
        l_vis = (l_vis * 255).astype(np.uint8)
        img = Image.fromarray(l_vis)
        
        # Resize nearest neighbor to match target (so we see the raw latent pixels clearly)
        return img.resize(target_size, Image.Resampling.NEAREST)

    def process_thread(self, path):
        calc = ResolutionCalculator(self.res_var.get()**2)
        flt = Image.Resampling.LANCZOS if self.filter_var.get() == "LANCZOS" else Image.Resampling.BICUBIC
        
        try:
            with Image.open(path) as raw:
                img = fix_alpha(raw)
                tw, th = calc.calculate_resolution(*img.size)
                
                # Bucketing Logic
                w, h = img.size
                if w/tw < h/th: new_w, new_h = tw, int(h*tw/w)
                else: new_w, new_h = int(w*th/h), th
                
                # Bucket Image
                bucket = img.resize((new_w, new_h), flt)
                l, t = (new_w-tw)//2, (new_h-th)//2
                bucket = bucket.crop((l, t, l+tw, t+th))
                
                # Original resized to match bucket (for comparison 1)
                orig_view = img.resize(bucket.size, Image.Resampling.BILINEAR)

                vae_rec = Image.new("RGB", bucket.size, (50,0,0))
                latent_view = Image.new("RGB", bucket.size, (0,0,50))

                if self.vae:
                    t_in = transforms.ToTensor()(bucket).unsqueeze(0).to(self.device, dtype=torch.float32)
                    t_in = t_in * 2.0 - 1.0
                    
                    with torch.no_grad():
                        latents = self.vae.encode(t_in).latent_dist.sample()
                        decoded = self.vae.decode(latents).sample
                    
                    # 1. Decoded View
                    decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
                    vae_rec = Image.fromarray((decoded[0] * 255).astype(np.uint8))
                    
                    # 2. Latent X-Ray View
                    # Shows the actual 4-channel tensor mapped to colors
                    latent_view = self.visualize_latents(latents, bucket.size)

            # Update GUI
            self.p1.load_images(orig_view, bucket)
            self.p2.load_images(bucket, vae_rec)
            self.p3.load_images(vae_rec, latent_view)
            self.log(f"Done: {path.name}", "#00ff00")
            
        except Exception as e:
            self.log(f"Error: {e}", "red")
            import traceback
            traceback.print_exc()

    def process(self):
        if not self.input_files: return
        import random
        target = random.choice(self.input_files)
        self.log(f"Processing {target.name}...", "yellow")
        threading.Thread(target=self.process_thread, args=(target,), daemon=True).start()

if __name__ == "__main__":
    app = UltimateDebugger()
    app.mainloop()