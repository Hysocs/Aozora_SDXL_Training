import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageChops
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

def generate_difference_map(img_a, img_b, boost=10.0):
    arr_a = np.array(img_a, dtype=np.float32)
    arr_b = np.array(img_b, dtype=np.float32)
    diff = np.abs(arr_a - arr_b)
    diff_boost = np.clip(diff * boost, 0, 255).astype(np.uint8)
    return Image.fromarray(diff_boost)

# --- FLUX/PACKED VAE LOGIC (Ported from Comfy Node) ---
class FluxPackedLogic:
    """
    Handles the Pixel Shuffle/Unshuffle logic required for the Flux2/Packed VAE.
    """
    @staticmethod
    def pack(latent, packed_channels=32, sf=2):
        """
        Simulates the 'Encode' post-processing.
        Converts High-Channel (128) Low-Res -> Low-Channel (32) High-Res (Pixel Shuffle).
        Corresponds to _from_vae_latent in Comfy node.
        """
        # Input: (B, 128, H, W)
        b, c, h, w = latent.shape
        target_channels = c
        
        # Verify dimensions align with logic
        if packed_channels * (sf ** 2) != target_channels:
            print(f"Warning: Channel mismatch. Input: {c}, Expected Inner: {packed_channels * sf**2}")
            return latent # Return as-is if mismatch

        # Reshape to split channels: (B, 32, 2, 2, H, W)
        latent = latent.reshape(b, packed_channels, sf, sf, h, w)
        # Permute to move spatial factor to spatial dims: (B, 32, H, 2, W, 2) -> (B, 32, H*2, W*2)
        latent = latent.permute(0, 1, 4, 2, 5, 3).reshape(b, packed_channels, h * sf, w * sf)
        return latent

    @staticmethod
    def unpack(latent, packed_channels=32, sf=2):
        """
        Simulates the 'Decode' pre-processing.
        Converts Low-Channel (32) High-Res -> High-Channel (128) Low-Res (Pixel Unshuffle).
        Corresponds to _to_vae_latent in Comfy node.
        """
        # Input: (B, 32, H_big, W_big)
        b, c, h, w = latent.shape
        
        # Check padding requirements (Comfy logic)
        if h % sf != 0 or w % sf != 0:
             pad_h = (sf - (h % sf)) % sf
             pad_w = (sf - (w % sf)) % sf
             latent = torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h))
             h, w = latent.shape[-2], latent.shape[-1]

        # Reshape to extract spatial blocks: (B, 32, H//2, 2, W//2, 2)
        latent = latent.reshape(b, packed_channels, h // sf, sf, w // sf, sf)
        # Permute to stack into channels: (B, 32, 2, 2, H//2, W//2) -> (B, 128, H//2, W//2)
        latent = latent.permute(0, 1, 3, 5, 2, 4).reshape(b, packed_channels * (sf**2), h // sf, w // sf)
        return latent

# --- CUSTOM WIDGET: Zoomable Wipe Slider ---
class ZoomableSlider(tk.Canvas):
    def __init__(self, parent, label_text, bg_color="#151515"):
        super().__init__(parent, bg=bg_color, highlightthickness=1, highlightbackground="#333")
        self.img_a_full = None
        self.img_b_full = None
        self.tk_img = None
        self.zoom = 1.0
        self.view_x = 0 
        self.view_y = 0 
        self.slider_pos_screen = 0
        self.bind("<Motion>", self.on_mouse_move)
        self.bind("<MouseWheel>", self.on_zoom)
        self.bind("<Button-4>", self.on_zoom)
        self.bind("<Button-5>", self.on_zoom)
        self.bind("<ButtonPress-3>", self.start_pan)
        self.bind("<B3-Motion>", self.do_pan)
        self.bind("<Configure>", self.render)
        self.label_text = label_text

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
        if event.num == 5 or event.delta < 0: factor = 0.9
        else: factor = 1.1
        mx_ratio = event.x / self.winfo_width()
        my_ratio = event.y / self.winfo_height()
        view_w, view_h = self.get_view_size()
        mouse_img_x = self.view_x + view_w * mx_ratio
        mouse_img_y = self.view_y + view_h * my_ratio
        new_zoom = max(1.0, min(self.zoom * factor, 50.0))
        if new_zoom == self.zoom: return
        self.zoom = new_zoom
        new_view_w, new_view_h = self.get_view_size()
        self.view_x = mouse_img_x - new_view_w * mx_ratio
        self.view_y = mouse_img_y - new_view_h * my_ratio
        self.validate_bounds()
        self.render()

    def get_fit_ratio(self):
        cw, ch = self.winfo_width(), self.winfo_height()
        iw, ih = self.img_a_full.width, self.img_a_full.height
        return min(cw/iw, ch/ih)

    def get_view_size(self):
        iw, ih = self.img_a_full.width, self.img_a_full.height
        return iw / self.zoom, ih / self.zoom

    def validate_bounds(self):
        vw, vh = self.get_view_size()
        iw, ih = self.img_a_full.width, self.img_a_full.height
        self.view_x = max(0, min(self.view_x, iw - vw))
        self.view_y = max(0, min(self.view_y, ih - vh))

    def on_mouse_move(self, event):
        self.slider_pos_screen = event.x
        self.render(lazy=True)

    def render(self, event=None, lazy=False):
        if not self.img_a_full: return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 5 or ch < 5: return
        vw, vh = self.get_view_size()
        crop_box = (int(self.view_x), int(self.view_y), int(self.view_x + vw), int(self.view_y + vh))
        crop_a = self.img_a_full.crop(crop_box)
        crop_b = self.img_b_full.crop(crop_box)
        ratio = min(cw / crop_a.width, ch / crop_a.height)
        disp_w = int(crop_a.width * ratio)
        disp_h = int(crop_a.height * ratio)
        resample = Image.Resampling.NEAREST if self.zoom > 8 else Image.Resampling.BILINEAR
        disp_a = crop_a.resize((disp_w, disp_h), resample)
        disp_b = crop_b.resize((disp_w, disp_h), resample)
        off_x = (cw - disp_w) // 2
        off_y = (ch - disp_h) // 2
        comp = disp_b.copy()
        rel_slider_x = self.slider_pos_screen - off_x
        if rel_slider_x > 0:
            crop_w = min(rel_slider_x, disp_w)
            if crop_w > 0:
                left_crop = disp_a.crop((0, 0, crop_w, disp_h))
                comp.paste(left_crop, (0, 0))
        draw = ImageDraw.Draw(comp)
        if 0 <= rel_slider_x <= disp_w:
            draw.line([(rel_slider_x, 0), (rel_slider_x, disp_h)], fill="cyan", width=2)
        self.tk_img = ImageTk.PhotoImage(comp)
        self.delete("all")
        self.create_image(off_x, off_y, anchor=tk.NW, image=self.tk_img)
        info = f"{self.label_text} | Zoom: {self.zoom:.1f}x"
        self.create_text(10, 10, anchor=tk.NW, text=info, fill="white", font=("Arial", 10, "bold"), tags="hud")

# --- MAIN APP ---
class UltimateDebugger(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Training Inspector (Flux/Packed VAE Supported)")
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
        
        self.btn_vae = tk.Button(bar, text="2. Load VAE", command=self.sel_vae, bg="#333", fg="white", font=("Arial",9,"bold"))
        self.btn_vae.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Flux Toggle
        self.is_flux_packed = tk.BooleanVar(value=False)
        self.chk_flux = tk.Checkbutton(bar, text="Enable Flux2/Packed Mode (32ch)", variable=self.is_flux_packed, bg="#202020", fg="#00ff00", selectcolor="#444", activebackground="#202020")
        self.chk_flux.pack(side=tk.LEFT, padx=20)
        
        tk.Label(bar, text="Res:", bg="#202020", fg="white").pack(side=tk.LEFT, padx=(15,2))
        self.res_var = tk.IntVar(value=1024)
        tk.Entry(bar, textvariable=self.res_var, width=5).pack(side=tk.LEFT)
        
        self.btn_go = tk.Button(bar, text="🎲 PROCESS IMAGE", command=self.process, bg="#007acc", fg="white", font=("Arial",10,"bold"))
        self.btn_go.pack(side=tk.RIGHT, padx=20)
        
        self.status = tk.Label(bar, text="Ready.", bg="#202020", fg="#00ff00")
        self.status.pack(side=tk.RIGHT, padx=10)

        # --- TABS ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --- TAB 1: PIPELINE VIEW ---
        self.tab_pipeline = tk.Frame(self.notebook, bg="#101010")
        self.notebook.add(self.tab_pipeline, text="Pipeline View")
        
        grid = tk.Frame(self.tab_pipeline, bg="#101010")
        grid.pack(fill=tk.BOTH, expand=True)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        grid.rowconfigure(0, weight=1)
        
        self.p1 = ZoomableSlider(grid, "1. BUCKETING\n< Original | Bucket >")
        self.p1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        self.p2 = ZoomableSlider(grid, "2. COMPRESSION\n< Bucket | VAE Rec >")
        self.p2.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        self.p3 = ZoomableSlider(grid, "3. LATENT CACHE VIEW\n< VAE Rec | Cached Latents >")
        self.p3.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

        # --- TAB 2: VAE FORENSICS ---
        self.tab_vae = tk.Frame(self.notebook, bg="#101010")
        self.notebook.add(self.tab_vae, text="VAE Forensics")
        
        vae_grid = tk.Frame(self.tab_vae, bg="#101010")
        vae_grid.pack(fill=tk.BOTH, expand=True)
        vae_grid.columnconfigure(0, weight=1)
        vae_grid.columnconfigure(1, weight=1)
        vae_grid.rowconfigure(0, weight=1)

        self.p_vae_pixel = ZoomableSlider(vae_grid, "PIXEL PEEPING\n< Bucket (Input) | VAE (Output) >")
        self.p_vae_pixel.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.p_vae_diff = ZoomableSlider(vae_grid, "ARTIFACT DETECTOR\n< VAE (Output) | Diff x10 (Black = Perfect) >")
        self.p_vae_diff.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

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
            # Attempt 1: Try loading with standard config
            # We pass ignore_mismatched_sizes=False first to catch the error and diagnose
            self.vae = AutoencoderKL.from_single_file(path).to(self.device, dtype=torch.float32)
            
        except Exception as e:
            # Check for the specific shape mismatch error regarding conv_out
            err_msg = str(e)
            if "conv_out.weight" in err_msg and "64" in err_msg:
                self.log("Detected 32-channel VAE (Flux/Custom). Retrying...", "yellow")
                try:
                    # Attempt 2: Force latent_channels=32
                    # We usually need to override the config. 
                    # Note: Passing kwargs to from_single_file is the cleanest way in modern diffusers
                    self.vae = AutoencoderKL.from_single_file(
                        path, 
                        latent_channels=32,
                        ignore_mismatched_sizes=True 
                    ).to(self.device, dtype=torch.float32)
                except Exception as e2:
                    self.log(f"Retry Failed: {e2}", "red")
                    return
            else:
                self.log(f"Error: {e}", "red")
                return

        # Setup VAE
        try:
            self.vae.enable_tiling()
            self.vae.eval()
            
            # Auto-configure the GUI based on what we loaded
            ch = self.vae.config.latent_channels
            self.log(f"VAE Loaded! (Latent Channels: {ch})", "#00ff00")
            
            if ch == 32:
                # If the VAE natively outputs 32 channels, 
                # we usually DON'T need the packing logic (Pixel Shuffle), 
                # OR the packing logic is internal to the model. 
                # The Comfy node implies the packing is for 128ch -> 32ch.
                # If your VAE is already 32ch, we disable the manual packing toggle 
                # to avoid messing up the dimensions, unless you know for sure otherwise.
                self.is_flux_packed.set(False) 
                self.log("Note: VAE is native 32ch. Packing disabled by default.", "yellow")
                
            elif ch == 128:
                self.is_flux_packed.set(True)
                self.log("VAE is 128ch. Auto-enabling Pack/Unpack mode.", "cyan")
                
        except Exception as e:
             self.log(f"Post-Load Error: {e}", "red")

    def visualize_latents(self, latents, target_size):
        # latents: (B, C, H, W)
        # We visualize the first 3 channels normalized
        l_vis = latents[0, :3, :, :].clone().cpu().float()
        
        # Normalize roughly for visualization
        l_vis = (l_vis - l_vis.min()) / (l_vis.max() - l_vis.min() + 1e-5)
        
        if l_vis.shape[0] < 3:
            # Handle 1 or 2 channel latents if they exist
            zeros = torch.zeros_like(l_vis[0:1])
            l_vis = torch.cat([l_vis, zeros, zeros], dim=0)[:3]
            
        l_vis = l_vis.permute(1, 2, 0).numpy() # H, W, C
        l_vis = (l_vis * 255).astype(np.uint8)
        img = Image.fromarray(l_vis)
        return img.resize(target_size, Image.Resampling.NEAREST)

    def process_thread(self, path):
        calc = ResolutionCalculator(self.res_var.get()**2)
        flt = Image.Resampling.LANCZOS
        
        try:
            with Image.open(path) as raw:
                img = fix_alpha(raw)
                tw, th = calc.calculate_resolution(*img.size)
                
                # Bucketing Logic
                w, h = img.size
                if w/tw < h/th: new_w, new_h = tw, int(h*tw/w)
                else: new_w, new_h = int(w*th/h), th
                
                bucket = img.resize((new_w, new_h), flt)
                l, t = (new_w-tw)//2, (new_h-th)//2
                bucket = bucket.crop((l, t, l+tw, t+th))
                
                orig_view = img.resize(bucket.size, Image.Resampling.BILINEAR)

                vae_rec = Image.new("RGB", bucket.size, (50,0,0))
                latent_view = Image.new("RGB", bucket.size, (0,0,50))
                diff_map = Image.new("RGB", bucket.size, (0,0,0))

                if self.vae:
                    t_in = transforms.ToTensor()(bucket).unsqueeze(0).to(self.device, dtype=torch.float32)
                    t_in = t_in * 2.0 - 1.0
                    
                    with torch.no_grad():
                        # 1. ENCODE
                        dist = self.vae.encode(t_in).latent_dist
                        latents_raw = dist.sample()
                        
                        # --- CACHE SIMULATION ---
                        if self.is_flux_packed.get():
                            # This is what gets saved to .pt file
                            latents_cached = FluxPackedLogic.pack(latents_raw)
                            
                            cache_shape = latents_cached.shape
                            self.log(f"Packed (Cached) Shape: {list(cache_shape)}", "cyan")
                            
                            # Visualize the PACKED state (32ch)
                            latent_view = self.visualize_latents(latents_cached, bucket.size)
                            
                            # --- LOAD SIMULATION ---
                            # This happens during training
                            latents_for_decode = FluxPackedLogic.unpack(latents_cached)
                        else:
                            # Standard SDXL
                            latents_cached = latents_raw
                            latents_for_decode = latents_raw
                            self.log(f"Latent Shape: {list(latents_cached.shape)}", "cyan")
                            latent_view = self.visualize_latents(latents_cached, bucket.size)

                        # 2. DECODE
                        decoded = self.vae.decode(latents_for_decode).sample
                    
                    decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
                    vae_rec = Image.fromarray((decoded[0] * 255).astype(np.uint8))
                    
                    diff_map = generate_difference_map(bucket, vae_rec, boost=10.0)

            # Update GUI
            self.p1.load_images(orig_view, bucket)
            self.p2.load_images(bucket, vae_rec)
            self.p3.load_images(vae_rec, latent_view)
            self.p_vae_pixel.load_images(bucket, vae_rec)
            self.p_vae_diff.load_images(vae_rec, diff_map)
            
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