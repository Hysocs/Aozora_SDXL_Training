import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
import numpy as np
import threading
import time
import random
from pathlib import Path

# ==============================================================================
#  HELPER: Debouncer
# ==============================================================================
class DebouncedWorker:
    def __init__(self, process_func, callback_func, delay=0.1):
        self.process_func = process_func
        self.callback_func = callback_func
        self.delay = delay
        self.last_time = 0
        self.pending_args = None
        self.is_running = False
        self.daemon = threading.Thread(target=self._monitor, daemon=True)
        self.daemon.start()

    def request(self, args):
        self.pending_args = args
        self.last_time = time.time()

    def _monitor(self):
        while True:
            time.sleep(0.05)
            if self.pending_args is not None and not self.is_running:
                if time.time() - self.last_time > self.delay:
                    self.is_running = True
                    args = self.pending_args
                    self.pending_args = None
                    try:
                        res = self.process_func(args)
                        self.callback_func(res)
                    except Exception as e:
                        print(f"Worker Error: {e}")
                    finally:
                        self.is_running = False

# ==============================================================================
#  MAIN APPLICATION
# ==============================================================================
class VAEStatisticsTool:
    def __init__(self, root):
        self.root = root
        self.root.title("VAE Latent Statistics Calculator")
        self.root.geometry("1400x900")
        self.root.configure(bg="#101010")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vae = None
        self.image_paths = []
        self.current_idx = 0
        
        # Scan Settings
        self.var_pixel_limit = tk.IntVar(value=1024*1024) 
        self.var_scan_count = tk.IntVar(value=200)
        
        # Calculated Results
        self.calc_shift = tk.StringVar(value="0.0000")
        self.calc_scale = tk.StringVar(value="1.0000")
        
        # Manual Sliders (Linked to calc results if applied)
        self.var_manual_shift = tk.DoubleVar(value=0.0)
        self.var_manual_scale = tk.DoubleVar(value=0.13025)
        
        self.worker = DebouncedWorker(self.process_preview, self.update_preview_ui)
        self.setup_ui()

    def setup_ui(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg="#1e1e1e", pady=10)
        header.pack(side=tk.TOP, fill=tk.X)
        
        btn_opts = {"bg": "#333", "fg": "white", "relief": "flat", "padx": 15, "pady": 5}
        
        tk.Button(header, text="1. Load VAE", command=self.load_vae_thread, **btn_opts).pack(side=tk.LEFT, padx=10)
        tk.Button(header, text="2. Select Image Folder", command=self.select_folder, **btn_opts).pack(side=tk.LEFT, padx=10)
        
        self.lbl_status = tk.Label(header, text="Status: Waiting for input...", bg="#1e1e1e", fg="cyan", font=("Consolas", 11))
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        # --- CONTENT SPLIT ---
        content = tk.Frame(self.root, bg="#121212")
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === LEFT: STATISTICS & CALCULATOR ===
        stats_frame = tk.Frame(content, bg="#181818", width=500)
        stats_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        stats_frame.pack_propagate(False)
        
        tk.Label(stats_frame, text="DATASET SCANNER", bg="#181818", fg="white", font=("Arial", 14, "bold")).pack(pady=15)
        
        # Controls
        ctrl = tk.Frame(stats_frame, bg="#181818")
        ctrl.pack(fill=tk.X, padx=20)
        tk.Label(ctrl, text="Scan Count:", bg="#181818", fg="#aaa").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(ctrl, textvariable=self.var_scan_count, bg="#222", fg="white", width=10).grid(row=0, column=1, padx=10)
        tk.Label(ctrl, text="Resize Area:", bg="#181818", fg="#aaa").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(ctrl, textvariable=self.var_pixel_limit, bg="#222", fg="white", width=10).grid(row=1, column=1, padx=10)

        tk.Button(stats_frame, text="▶ RUN STATISTICAL ANALYSIS", command=self.start_scan, bg="#007acc", fg="white", font=("Arial", 11, "bold"), pady=10).pack(fill=tk.X, padx=20, pady=20)
        
        self.progress = ttk.Progressbar(stats_frame, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, padx=20)

        # DATA OUTPUT BOX
        self.txt_output = tk.Text(stats_frame, height=12, bg="#000", fg="#0f0", font=("Consolas", 10))
        self.txt_output.pack(fill=tk.X, padx=20, pady=15)
        self.txt_output.insert(tk.END, "Ready to scan.\n")

        # MAGIC NUMBERS
        res_box = tk.Frame(stats_frame, bg="#223322", highlightthickness=2, highlightbackground="#00ff00")
        res_box.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(res_box, text="OPTIMAL CONFIG VALUES", bg="#223322", fg="#fff", font=("Arial", 10, "bold")).pack(pady=5)
        
        r1 = tk.Frame(res_box, bg="#223322")
        r1.pack(fill=tk.X, pady=2)
        tk.Label(r1, text="SHIFT:", bg="#223322", fg="#aaa", width=8).pack(side=tk.LEFT)
        tk.Entry(r1, textvariable=self.calc_shift, bg="#223322", fg="yellow", font=("Consolas", 14, "bold"), width=12, bd=0).pack(side=tk.LEFT)
        
        r2 = tk.Frame(res_box, bg="#223322")
        r2.pack(fill=tk.X, pady=2)
        tk.Label(r2, text="SCALE:", bg="#223322", fg="#aaa", width=8).pack(side=tk.LEFT)
        tk.Entry(r2, textvariable=self.calc_scale, bg="#223322", fg="yellow", font=("Consolas", 14, "bold"), width=12, bd=0).pack(side=tk.LEFT)
        
        tk.Button(res_box, text="Test on Sliders ->", command=self.apply_to_sliders, bg="#446644", fg="white", relief="flat").pack(fill=tk.X, padx=5, pady=5)

        # === RIGHT: VISUAL VERIFICATION ===
        viz_frame = tk.Frame(content, bg="black")
        viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Sliders
        sliders_f = tk.Frame(viz_frame, bg="#222")
        sliders_f.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(sliders_f, text="MANUAL PREVIEW (Shift/Scale)", bg="#222", fg="white").pack(pady=5)
        
        s1 = tk.Frame(sliders_f, bg="#222")
        s1.pack(fill=tk.X, padx=10)
        tk.Label(s1, text="Shift:", bg="#222", fg="#aaa", width=6).pack(side=tk.LEFT)
        tk.Scale(s1, variable=self.var_manual_shift, from_=-2.0, to=2.0, resolution=0.0001, orient=tk.HORIZONTAL, bg="#222", fg="white", command=self.trigger_preview).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        s2 = tk.Frame(sliders_f, bg="#222")
        s2.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(s2, text="Scale:", bg="#222", fg="#aaa", width=6).pack(side=tk.LEFT)
        tk.Scale(s2, variable=self.var_manual_scale, from_=0.01, to=2.0, resolution=0.0001, orient=tk.HORIZONTAL, bg="#222", fg="white", command=self.trigger_preview).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Image Area
        self.lbl_preview = tk.Label(viz_frame, bg="black", text="Visual Check Area")
        self.lbl_preview.pack(fill=tk.BOTH, expand=True)
        
        # Nav
        nav = tk.Frame(viz_frame, bg="#000", height=30)
        nav.pack(side=tk.TOP, fill=tk.X)
        tk.Button(nav, text="< Prev", command=self.prev_image, bg="#333", fg="white").pack(side=tk.LEFT)
        self.lbl_fname = tk.Label(nav, text="--", bg="#000", fg="#666")
        self.lbl_fname.pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="Next >", command=self.next_image, bg="#333", fg="white").pack(side=tk.RIGHT)

    # ================= LOGIC =================

    def load_vae_thread(self):
        threading.Thread(target=self.load_vae, daemon=True).start()

    def load_vae(self):
        path = filedialog.askopenfilename(filetypes=[("Safetensors", "*.safetensors")])
        if not path: return
        self.lbl_status.config(text="Loading VAE...", fg="yellow")
        try:
            try:
                self.vae = AutoencoderKL.from_single_file(path).to(self.device, dtype=torch.float32)
            except:
                self.vae = AutoencoderKL.from_single_file(path, latent_channels=32, ignore_mismatched_sizes=True).to(self.device, dtype=torch.float32)
            self.lbl_status.config(text=f"VAE Loaded", fg="#00ff00")
            self.trigger_preview()
        except Exception as e:
            self.lbl_status.config(text=f"VAE Error: {e}", fg="red")

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            exts = {'.jpg', '.png', '.jpeg', '.webp', '.bmp'}
            self.image_paths = sorted([p for p in Path(path).rglob("*") if p.suffix.lower() in exts])
            self.lbl_status.config(text=f"Images: {len(self.image_paths)}")
            if self.image_paths:
                self.trigger_preview()

    # --- SCANNING LOGIC ---

    def start_scan(self):
        if not self.image_paths or not self.vae:
            messagebox.showerror("Error", "Load VAE and Images first.")
            return
        threading.Thread(target=self.run_scan, daemon=True).start()

    def run_scan(self):
        self.lbl_status.config(text="Scanning...", fg="yellow")
        self.txt_output.delete(1.0, tk.END)
        self.txt_output.insert(tk.END, "Initializing Scan...\n")
        
        # Sample
        total = len(self.image_paths)
        count = self.var_scan_count.get()
        samples = random.sample(self.image_paths, min(count, total))
        limit = self.var_pixel_limit.get()
        
        # We will collect per-image means/stds to calculate global
        # Doing it this way avoids holding huge tensors in VRAM
        all_means = []
        all_stds = []
        global_min = float('inf')
        global_max = float('-inf')

        self.progress['maximum'] = len(samples)
        self.progress['value'] = 0
        
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
        
        with torch.no_grad():
            for i, p in enumerate(samples):
                try:
                    img = Image.open(p).convert("RGB")
                    w, h = img.size
                    # Resize
                    if w*h > limit:
                        s = (limit / (w*h)) ** 0.5
                        w, h = int(w*s), int(h*s)
                    w, h = (w//64)*64, (h//64)*64
                    if w<64 or h<64: continue
                    
                    img = img.resize((w,h), Image.Resampling.LANCZOS)
                    t = tf(img).unsqueeze(0).to(self.device)
                    
                    # Encode -> Latent Dist -> Mean
                    latents = self.vae.encode(t).latent_dist.mean
                    
                    # Accumulate Stats
                    all_means.append(latents.mean().item())
                    all_stds.append(latents.std().item())
                    
                    curr_min = latents.min().item()
                    curr_max = latents.max().item()
                    if curr_min < global_min: global_min = curr_min
                    if curr_max > global_max: global_max = curr_max
                    
                    if i % 5 == 0: self.root.after(0, lambda v=i: self.progress.configure(value=v))
                except: pass

        if not all_means:
            self.lbl_status.config(text="Scan Failed", fg="red")
            return
            
        # --- CALCULATE STATISTICS ---
        # Note: Calculating mean of means is roughly accurate for uniform sized batches
        raw_mean = np.mean(all_means)
        raw_std = np.mean(all_stds)
        
        self.log(f"--- RAW STATISTICS ---")
        self.log(f"Images Scanned: {len(samples)}")
        self.log(f"Global Min: {global_min:.4f}")
        self.log(f"Global Max: {global_max:.4f}")
        self.log(f"Raw Mean  : {raw_mean:.4f}")
        self.log(f"Raw Std   : {raw_std:.4f}")
        self.log(f"----------------------")

        # --- CALCULATE NORMALIZATION ---
        # Goal: (Raw - Shift) * Scale -> Mean=0, Std=1
        
        calc_shift = raw_mean
        calc_scale = 1.0 / raw_std
        
        self.log(f"--- CALCULATED CONFIG ---")
        self.log(f"Target Mean: 0.0")
        self.log(f"Target Std : 1.0")
        self.log(f"Rec. Shift : {calc_shift:.4f}")
        self.log(f"Rec. Scale : {calc_scale:.4f}")
        self.log(f"----------------------")
        
        # --- VERIFICATION STEP ---
        # Let's apply this math to our gathered stats to prove it works
        # Norm = (Raw - Shift) * Scale
        # NormMean = (RawMean - Shift) * Scale
        # NormStd  = RawStd * Scale
        
        verify_mean = (raw_mean - calc_shift) * calc_scale
        verify_std = raw_std * calc_scale
        
        self.log(f"--- VERIFICATION ---")
        self.log(f"New Mean   : {verify_mean:.4f} (Ideal: 0)")
        self.log(f"New Std    : {verify_std:.4f}  (Ideal: 1)")
        
        self.root.after(0, lambda: self.update_results(calc_shift, calc_scale))

    def log(self, text):
        self.root.after(0, lambda: self.txt_output.insert(tk.END, text + "\n"))
        self.root.after(0, lambda: self.txt_output.see(tk.END))

    def update_results(self, shift, scale):
        self.progress['value'] = self.progress['maximum']
        self.lbl_status.config(text="Scan Complete", fg="#00ff00")
        self.calc_shift.set(f"{shift:.4f}")
        self.calc_scale.set(f"{scale:.4f}")

    def apply_to_sliders(self):
        try:
            s = float(self.calc_shift.get())
            sc = float(self.calc_scale.get())
            self.var_manual_shift.set(s)
            self.var_manual_scale.set(sc)
            self.trigger_preview()
        except: pass

    # --- VISUAL PREVIEW ---

    def next_image(self):
        if self.image_paths:
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            self.trigger_preview()

    def prev_image(self):
        if self.image_paths:
            self.current_idx = (self.current_idx - 1) % len(self.image_paths)
            self.trigger_preview()

    def trigger_preview(self, _=None):
        if not self.image_paths or not self.vae: return
        args = {
            "path": self.image_paths[self.current_idx],
            "shift": self.var_manual_shift.get(),
            "scale": self.var_manual_scale.get()
        }
        self.lbl_fname.config(text=f"{args['path'].name}")
        self.worker.request(args)

    def process_preview(self, args):
        img = Image.open(args["path"]).convert("RGB")
        w, h = img.size
        # Resize for preview speed
        if w*h > 512*512:
            s = (512*512 / (w*h)) ** 0.5
            w, h = int(w*s), int(h*s)
        w, h = (w//64)*64, (h//64)*64
        if w<64: w=64
        if h<64: h=64
        
        img = img.resize((w, h), Image.Resampling.LANCZOS)
        t_pix = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        t_pix = (t_pix - 0.5) * 2.0
        
        with torch.no_grad():
            raw = self.vae.encode(t_pix).latent_dist.mean
            
            # Apply Math (Simulate Normalization)
            shift = args["shift"]
            scale = args["scale"]
            if scale == 0: scale = 0.0001
            
            # This is what gets saved to cache
            stored = (raw - shift) * scale
            
            # CLAMPING CHECK (Training Integrity)
            clamped = torch.clamp(stored, -4.0, 4.0)
            
            # Reverse
            rec_input = (clamped / scale) + shift
            decoded = self.vae.decode(rec_input).sample
            
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            dec_np = decoded.cpu().permute(0, 2, 3, 1).numpy()[0]
            
        rec_img = Image.fromarray((dec_np * 255).astype(np.uint8))
        
        # Side By Side
        comp = Image.new("RGB", (w*2, h))
        comp.paste(img, (0,0))
        comp.paste(rec_img, (w,0))
        return comp

    def update_preview_ui(self, pil_img):
        dw = self.lbl_preview.winfo_width()
        dh = self.lbl_preview.winfo_height()
        if dw < 10: dw=500
        if dh < 10: dh=500
        
        r = min(dw/pil_img.width, dh/pil_img.height)
        ns = (int(pil_img.width*r), int(pil_img.height*r))
        self.tk_prev = ImageTk.PhotoImage(pil_img.resize(ns, Image.Resampling.LANCZOS))
        self.lbl_preview.config(image=self.tk_prev)

if __name__ == "__main__":
    root = tk.Tk()
    app = VAEStatisticsTool(root)
    root.mainloop()