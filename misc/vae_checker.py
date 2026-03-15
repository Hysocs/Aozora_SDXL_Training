"""
VAE Encode/Decode Diagnostic Tool
Visually test how your VAE encodes and reconstructs images,
replicating exactly how training caches and uses latents.

Requirements:
    pip install torch torchvision diffusers safetensors pillow
"""

import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
import math

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    import numpy as np
except ImportError as e:
    import tkinter.messagebox as mb
    root = tk.Tk(); root.withdraw()
    mb.showerror("Missing Dependency", f"{e}\n\nRun: pip install torch torchvision diffusers safetensors pillow")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# VAE Logic (mirrors training code exactly)
# ─────────────────────────────────────────────────────────────────────────────

def detect_vae_info(path: str):
    """
    Returns (channels, dtype_string) by inspecting the safetensors file directly.
    """
    try:
        tensors = load_file(path, device="cpu")
        
        # 1. Detect Channels
        channels = 4
        for k in ["first_stage_model.quant_conv.weight", "quant_conv.weight"]:
            if k in tensors:
                channels = tensors[k].shape[0] // 2
                break
        
        # 2. Detect Precision (fp16 vs fp32)
        # We look for any weight tensor inside the VAE (first_stage_model)
        dtype_str = "unknown"
        for k, tensor in tensors.items():
            # Check for standard VAE prefixes in checkpoints
            if k.startswith("first_stage_model.") or k.startswith("vae."):
                dtype_str = str(tensor.dtype).replace("torch.", "")
                # Just need to check one tensor, break loop
                break
                
        return channels, dtype_str

    except Exception as e:
        print(f"Error detecting VAE info: {e}")
        return 4, "unknown"


def load_vae(path: str, latent_channels: int) -> AutoencoderKL:
    kwargs = dict(torch_dtype=torch.float32, low_cpu_mem_usage=False)
    if latent_channels != 4:
        kwargs["latent_channels"] = latent_channels
        kwargs["ignore_mismatched_sizes"] = True
    vae = AutoencoderKL.from_single_file(path, **kwargs)
    vae.eval()
    return vae


def set_vae_factors(vae, shift: float, scale: float):
    vae.config.shift_factor = shift
    vae.config.scaling_factor = scale


def image_to_tensor(img: Image.Image, w: int, h: int) -> torch.Tensor:
    img = img.convert("RGB").resize((w, h), Image.LANCZOS)
    t = transforms.ToTensor()(img)           # [3, H, W]  range [0,1]
    t = transforms.Normalize([0.5], [0.5])(t)  # range [-1,1]
    return t.unsqueeze(0)                    # [1, 3, H, W]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).float().clamp(-1, 1)   # [3, H, W]
    t = (t + 1) / 2                          # [0, 1]
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def encode_decode(vae, pixel_tensor, shift, scale, device):
    """Mirrors training cache logic exactly."""
    pixel_tensor = pixel_tensor.to(device)
    vae = vae.to(device)

    with torch.no_grad():
        # Encode
        dist = vae.encode(pixel_tensor).latent_dist
        latents_raw = dist.mean                          # no sampling for clean test

        # Normalize (same as training cache)
        if shift != 0.0:
            latents_norm = (latents_raw - shift) * scale
        else:
            latents_norm = latents_raw * scale

        # Denormalize for decode (inverse of above)
        if shift != 0.0:
            latents_decode = latents_norm / scale + shift
        else:
            latents_decode = latents_norm / scale

        # Decode
        decoded = vae.decode(latents_decode).sample

    return latents_raw, latents_norm, decoded


def latent_to_preview(latents: torch.Tensor) -> Image.Image:
    """Visualize all latent channels as a grid of grayscale maps."""
    l = latents.squeeze(0).float().cpu()   # [C, H, W]
    c, h, w = l.shape
    cols = 8
    rows = math.ceil(c / cols)
    cell = max(w, 32)
    pad = 2
    canvas_w = cols * (cell + pad) + pad
    canvas_h = rows * (cell + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 24))
    draw = ImageDraw.Draw(canvas)

    for i in range(c):
        ch = l[i]
        mn, mx = ch.min().item(), ch.max().item()
        rng = mx - mn if mx != mn else 1.0
        norm = ((ch - mn) / rng * 255).byte().numpy()
        ch_img = Image.fromarray(norm, mode="L").resize((cell, cell), Image.NEAREST)
        ch_rgb = Image.merge("RGB", [ch_img, ch_img, ch_img])
        col = i % cols
        row = i // cols
        x = pad + col * (cell + pad)
        y = pad + row * (cell + pad)
        canvas.paste(ch_rgb, (x, y))
        draw.text((x + 2, y + 2), str(i), fill=(255, 220, 80))

    return canvas


def compute_stats(latents_raw: torch.Tensor, latents_norm: torch.Tensor) -> dict:
    r = latents_raw.float().cpu()
    n = latents_norm.float().cpu()
    stats = {
        "channels":    r.shape[1],
        "spatial":     f"{r.shape[-2]} × {r.shape[-1]}",
        "raw_mean":    r.mean().item(),
        "raw_std":     r.std().item(),
        "raw_min":     r.min().item(),
        "raw_max":     r.max().item(),
        "norm_mean":   n.mean().item(),
        "norm_std":    n.std().item(),
        "norm_min":    n.min().item(),
        "norm_max":    n.max().item(),
    }
    # Per-channel stats
    ch_means = [r[0, i].mean().item() for i in range(r.shape[1])]
    ch_stds  = [r[0, i].std().item()  for i in range(r.shape[1])]
    stats["ch_means"] = ch_means
    stats["ch_stds"]  = ch_stds
    return stats


def psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = F.mse_loss(original.float(), reconstructed.float()).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

BG       = "#0d0f14"
BG2      = "#13161e"
BG3      = "#1a1e2a"
ACCENT   = "#7eb8f7"
ACCENT2  = "#f7c97e"
TEXT     = "#e8eaf0"
MUTED    = "#5a6070"
SUCCESS  = "#7ef7a8"
WARN     = "#f7e07e"
DANGER   = "#f77e7e"
FONT_UI  = ("Consolas", 10)
FONT_HDR = ("Consolas", 11, "bold")
FONT_SM  = ("Consolas", 9)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VAE Diagnostic Tool")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1100, 700)

        self.vae = None
        self.vae_path = tk.StringVar()
        self.img_path = tk.StringVar()
        self.shift_var = tk.DoubleVar(value=0.0760)
        self.scale_var = tk.DoubleVar(value=0.6043)
        self.width_var  = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.detected_ch = tk.IntVar(value=0)

        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left panel
        left = tk.Frame(self, bg=BG2, width=320)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        self._build_controls(left)

        # Right panel
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)
        self._build_display(right)

    def _build_controls(self, parent):
        def section(text):
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=10, pady=(12, 2))
            tk.Label(f, text=text, font=FONT_HDR, bg=BG2, fg=ACCENT).pack(anchor="w")
            tk.Frame(parent, bg=ACCENT, height=1).pack(fill="x", padx=10)

        def row(parent, label, widget_fn):
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=10, pady=3)
            tk.Label(f, text=label, font=FONT_SM, bg=BG2, fg=MUTED, width=14, anchor="w").pack(side="left")
            widget_fn(f)

        # ── Model ──
        section("MODEL")
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill="x", padx=10, pady=4)
        tk.Entry(f, textvariable=self.vae_path, font=FONT_SM, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat", bd=4).pack(side="left", fill="x", expand=True)
        self._btn(f, "…", self._browse_model).pack(side="right", padx=(4,0))

        row(parent, "Detected ch:", lambda p: tk.Label(p, textvariable=self.detected_ch,
            font=FONT_SM, bg=BG2, fg=ACCENT2).pack(side="left"))

        self._load_btn = self._big_btn(parent, "LOAD VAE", self._load_vae_thread, ACCENT)

        # ── Image ──
        section("IMAGE")
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill="x", padx=10, pady=4)
        tk.Entry(f, textvariable=self.img_path, font=FONT_SM, bg=BG3, fg=TEXT,
                 insertbackground=TEXT, relief="flat", bd=4).pack(side="left", fill="x", expand=True)
        self._btn(f, "…", self._browse_image).pack(side="right", padx=(4,0))

        row(parent, "Width:",  lambda p: tk.Spinbox(p, textvariable=self.width_var,
            from_=64, to=4096, increment=64, width=7, font=FONT_SM,
            bg=BG3, fg=TEXT, buttonbackground=BG3, relief="flat").pack(side="left"))
        row(parent, "Height:", lambda p: tk.Spinbox(p, textvariable=self.height_var,
            from_=64, to=4096, increment=64, width=7, font=FONT_SM,
            bg=BG3, fg=TEXT, buttonbackground=BG3, relief="flat").pack(side="left"))

        # ── VAE Settings ──
        section("VAE FACTORS")

        def scale_row(p, label, var, lo, hi, digits):
            tk.Label(p, text=label, font=FONT_SM, bg=BG2, fg=MUTED, width=14, anchor="w").pack(side="left")
            s = tk.Scale(p, variable=var, from_=lo, to=hi, resolution=10**-digits,
                         orient="horizontal", length=120, bg=BG2, fg=TEXT,
                         troughcolor=BG3, highlightthickness=0, bd=0, showvalue=True,
                         digits=digits+1)
            s.pack(side="left", fill="x", expand=True)

        f = tk.Frame(parent, bg=BG2); f.pack(fill="x", padx=10, pady=3)
        scale_row(f, "Shift factor:", self.shift_var, -1.0, 1.0, 4)
        f = tk.Frame(parent, bg=BG2); f.pack(fill="x", padx=10, pady=3)
        scale_row(f, "Scale factor:", self.scale_var, 0.01, 2.0, 4)

        # Presets
        f = tk.Frame(parent, bg=BG2)
        f.pack(fill="x", padx=10, pady=4)
        tk.Label(f, text="Presets:", font=FONT_SM, bg=BG2, fg=MUTED).pack(side="left")
        self._btn(f, "NoobAI Flux2", lambda: self._preset(0.0760, 0.6043)).pack(side="left", padx=2)
        self._btn(f, "SDXL std",     lambda: self._preset(0.0,    0.13025)).pack(side="left", padx=2)
        self._btn(f, "Flux1",        lambda: self._preset(0.0609, 0.3611)).pack(side="left", padx=2)

        # ── Device ──
        section("DEVICE")
        f = tk.Frame(parent, bg=BG2); f.pack(fill="x", padx=10, pady=4)
        for d in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
            tk.Radiobutton(f, text=d.upper(), variable=self.device_var, value=d,
                           font=FONT_SM, bg=BG2, fg=TEXT, selectcolor=BG3,
                           activebackground=BG2, activeforeground=TEXT).pack(side="left", padx=4)

        # ── Run ──
        section("RUN")
        self._run_btn = self._big_btn(parent, "ENCODE → DECODE", self._run_thread, ACCENT2)

        # ── Log ──
        section("LOG")
        self.log = tk.Text(parent, height=10, font=FONT_SM, bg=BG3, fg=TEXT,
                           insertbackground=TEXT, relief="flat", bd=4, wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.log.config(state="disabled")

    def _build_display(self, parent):
        # Tab strip
        tabs = tk.Frame(parent, bg=BG)
        tabs.pack(fill="x", pady=(0, 6))

        self.panels = {}
        self.tab_btns = {}
        self.active_tab = tk.StringVar(value="comparison")
        tab_defs = [
            ("comparison", "Comparison"),
            ("latents",    "Latent Channels"),
            ("stats",      "Statistics"),
        ]
        for key, label in tab_defs:
            b = tk.Button(tabs, text=label, font=FONT_HDR, bg=BG3, fg=MUTED,
                          relief="flat", bd=0, padx=16, pady=6,
                          command=lambda k=key: self._switch_tab(k))
            b.pack(side="left", padx=2)
            self.tab_btns[key] = b

        self.display_frame = tk.Frame(parent, bg=BG)
        self.display_frame.pack(fill="both", expand=True)

        # ---------------------------------------------------------
        # Comparison panel (FIXED)
        # ---------------------------------------------------------
        p = tk.Frame(self.display_frame, bg=BG)
        self.panels["comparison"] = p

        # Create image panels (Frames) and store them
        self.orig_panel  = self._image_panel(p, "ORIGINAL")
        self.recon_panel = self._image_panel(p, "RECONSTRUCTED")
        self.diff_panel  = self._image_panel(p, "DIFFERENCE  ×5")

        # Pack the returned frames
        for panel in [self.orig_panel, self.recon_panel, self.diff_panel]:
            panel.pack(side="left", fill="both", expand=True, padx=4)

        # ---------------------------------------------------------
        # Latent panel
        # ---------------------------------------------------------
        p2 = tk.Frame(self.display_frame, bg=BG)
        self.panels["latents"] = p2
        self.lat_label = tk.Label(p2, bg=BG, fg=MUTED, font=FONT_SM,
                                  text="Run encode to see latent channels")
        self.lat_label.pack(fill="both", expand=True)

        # ---------------------------------------------------------
        # Stats panel
        # ---------------------------------------------------------
        p3 = tk.Frame(self.display_frame, bg=BG)
        self.panels["stats"] = p3
        self.stats_text = tk.Text(p3, font=("Consolas", 11), bg=BG3, fg=TEXT,
                                  relief="flat", bd=8, wrap="word", state="disabled")
        self.stats_text.pack(fill="both", expand=True, padx=8, pady=8)

        self._switch_tab("comparison")

    def _image_panel(self, parent, title):
        """
        Creates a Frame containing a title and an image label.
        Returns the Frame (which contains .label for the image).
        """
        f = tk.Frame(parent, bg=BG3, bd=0)
        
        # Title Label
        tk.Label(f, text=title, font=FONT_HDR, bg=BG3, fg=MUTED).pack(pady=(6, 2))
        
        # Image Label (packed inside 'f')
        lbl = tk.Label(f, bg=BG3, text="—", fg=MUTED, font=FONT_SM)
        lbl.pack(fill="both", expand=True, padx=4, pady=(0, 6))
        
        # Attach label to frame so we can access it later
        f.label = lbl
        return f

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _switch_tab(self, key):
        for k, p in self.panels.items():
            p.pack_forget()
        self.panels[key].pack(fill="both", expand=True)
        for k, b in self.tab_btns.items():
            b.config(bg=BG3 if k != key else ACCENT, fg=MUTED if k != key else BG)
        self.active_tab.set(key)

    # ── Widgets ───────────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, font=FONT_SM,
                         bg=BG3, fg=ACCENT, relief="flat", bd=0, padx=8, pady=4,
                         cursor="hand2", activebackground=ACCENT, activeforeground=BG)

    def _big_btn(self, parent, text, cmd, color):
        b = tk.Button(parent, text=text, command=cmd, font=FONT_HDR,
                      bg=color, fg=BG, relief="flat", bd=0, padx=16, pady=10,
                      cursor="hand2", activebackground=TEXT, activeforeground=BG)
        b.pack(fill="x", padx=10, pady=6)
        return b

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _preset(self, shift, scale):
        self.shift_var.set(shift)
        self.scale_var.set(scale)

    def _browse_model(self):
        p = filedialog.askopenfilename(
            title="Select model / VAE checkpoint",
            filetypes=[("Safetensors / CKPT", "*.safetensors *.ckpt *.pt"), ("All", "*.*")]
        )
        if p:
            self.vae_path.set(p)
            
            # Use the new detection function
            ch, dtype = detect_vae_info(p)
            
            self.detected_ch.set(ch)
            self._log(f"Detected: {ch} channels, Precision: {dtype.upper()}")
            
            # Auto-presets based on channels
            if ch == 32:
                self._preset(0.0760, 0.6043)
                self._log("Auto-set NoobAI Flux2 shift/scale presets.")
            elif ch == 4:
                self._preset(0.0, 0.13025)
                self._log("Auto-set standard SDXL shift/scale presets.")

    def _browse_image(self):
        p = filedialog.askopenfilename(
            title="Select test image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")]
        )
        if p:
            self.img_path.set(p)
            with Image.open(p) as img:
                self._log(f"Image: {img.size[0]}×{img.size[1]} {img.mode}")

    def _log(self, msg: str):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self._run_btn.config(state=state)
        self._load_btn.config(state=state)

    # ── VAE Loading ───────────────────────────────────────────────────────────

    def _load_vae_thread(self):
        threading.Thread(target=self._load_vae, daemon=True).start()

    def _load_vae(self):
        path = self.vae_path.get()
        if not path or not Path(path).exists():
            self._log("ERROR: Select a valid model path first.")
            return
        self._set_busy(True)
        self._log(f"Loading VAE from {Path(path).name} ...")
        try:
            ch = self.detected_ch.get() or detect_vae_info(path)
            self.detected_ch.set(ch)
            self.vae = load_vae(path, ch)
            set_vae_factors(self.vae, self.shift_var.get(), self.scale_var.get())
            self._log(f"VAE loaded. Channels={ch}, "
                      f"shift={self.shift_var.get():.4f}, scale={self.scale_var.get():.4f}")
        except Exception as e:
            self._log(f"ERROR loading VAE: {e}")
        finally:
            self._set_busy(False)

    # ── Encode/Decode Run ─────────────────────────────────────────────────────

    def _run_thread(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        if self.vae is None:
            self._log("ERROR: Load a VAE first.")
            return
        if not self.img_path.get() or not Path(self.img_path.get()).exists():
            self._log("ERROR: Select an image first.")
            return

        self._set_busy(True)
        try:
            w = self.width_var.get()
            h = self.height_var.get()
            shift = self.shift_var.get()
            scale = self.scale_var.get()
            device = self.device_var.get()

            # Update factors before run
            set_vae_factors(self.vae, shift, scale)

            self._log(f"Running encode→decode | {w}×{h} | shift={shift:.4f} scale={scale:.4f} | {device}")

            img = Image.open(self.img_path.get())
            pixel_tensor = image_to_tensor(img, w, h)

            latents_raw, latents_norm, decoded = encode_decode(
                self.vae, pixel_tensor, shift, scale, device
            )

            # Move back to CPU for display
            pixel_tensor = pixel_tensor.cpu()
            decoded      = decoded.cpu()
            latents_raw  = latents_raw.cpu()
            latents_norm = latents_norm.cpu()

            # Images
            orig_pil  = tensor_to_pil(pixel_tensor)
            recon_pil = tensor_to_pil(decoded)

            # Difference ×5
            diff = ((decoded - pixel_tensor).float().clamp(-1, 1) * 5).clamp(-1, 1)
            diff_pil = tensor_to_pil(diff)

            # PSNR
            # Normalize both to [0,1] for PSNR
            o = (pixel_tensor.float() + 1) / 2
            r = (decoded.float() + 1) / 2
            psnr_val = psnr(o, r)

            stats = compute_stats(latents_raw, latents_norm)
            norm_ok = abs(stats["norm_mean"]) < 0.3 and 0.6 < stats["norm_std"] < 1.6

            self._log(f"Latent shape: {list(latents_raw.shape)}")
            self._log(f"Raw   mean={stats['raw_mean']:.4f}  std={stats['raw_std']:.4f}")
            self._log(f"Norm  mean={stats['norm_mean']:.4f}  std={stats['norm_std']:.4f}  "
                      f"{'✓ GOOD' if norm_ok else '✗ CHECK shift/scale'}")
            self._log(f"PSNR: {psnr_val:.2f} dB  {'✓ good' if psnr_val > 28 else '⚠ low — check decode path'}")

            # Update UI on main thread
            self.after(0, lambda: self._update_display(
                orig_pil, recon_pil, diff_pil, latents_raw, latents_norm, stats, psnr_val, norm_ok
            ))

        except Exception as e:
            import traceback
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
        finally:
            self._set_busy(False)

    # ── Display Update ────────────────────────────────────────────────────────

    def _update_display(self, orig, recon, diff, latents_raw, latents_norm, stats, psnr_val, norm_ok):
        # Pass the Frame (panel) to set canvas
        self._set_canvas(self.orig_panel,  orig)
        self._set_canvas(self.recon_panel, recon)
        self._set_canvas(self.diff_panel,  diff)

        # Latent channel preview
        lat_img = latent_to_preview(latents_raw)
        self._switch_to_latent_img(lat_img)

        # Stats
        self._update_stats(stats, psnr_val, norm_ok)

    def _set_canvas(self, frame: tk.Frame, img: Image.Image):
        """
        Updates the image label inside the given frame.
        """
        label = frame.label
        self.update_idletasks()
        
        # Calculate available size inside the frame
        W = max(frame.winfo_width() - 16, 100)
        H = max(self.display_frame.winfo_height() - 60, 100)
        
        img_r = img.copy()
        img_r.thumbnail((W, H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img_r)
        
        label.config(image=photo, text="")
        label._photo = photo # Keep reference

    def _switch_to_latent_img(self, lat_img: Image.Image):
        p = self.panels["latents"]
        for child in p.winfo_children():
            child.destroy()
        self.update_idletasks()
        W = max(p.winfo_width(),  900)
        H = max(p.winfo_height(), 500)
        lat_img.thumbnail((W, H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(lat_img)
        self.lat_label = tk.Label(p, image=photo, bg=BG)
        self.lat_label._photo = photo
        self.lat_label.pack(fill="both", expand=True)

    def _update_stats(self, stats, psnr_val, norm_ok):
        lines = []
        lines.append("━" * 52)
        lines.append("  LATENT STATISTICS")
        lines.append("━" * 52)
        lines.append(f"  Channels   : {stats['channels']}")
        lines.append(f"  Spatial    : {stats['spatial']}")
        lines.append("")
        lines.append("  ── Raw Latent (post-encode, pre-norm) ──")
        lines.append(f"  Mean  : {stats['raw_mean']:+.6f}")
        lines.append(f"  Std   : {stats['raw_std']:.6f}")
        lines.append(f"  Min   : {stats['raw_min']:+.6f}")
        lines.append(f"  Max   : {stats['raw_max']:+.6f}")
        lines.append("")
        lines.append("  ── Normalized Latent (what training stores) ──")
        lines.append(f"  Mean  : {stats['norm_mean']:+.6f}  (target ≈ 0.0)")
        lines.append(f"  Std   : {stats['norm_std']:.6f}  (target ≈ 1.0)")
        lines.append(f"  Min   : {stats['norm_min']:+.6f}")
        lines.append(f"  Max   : {stats['norm_max']:+.6f}")
        lines.append(f"  Status: {'✓ Normalization looks correct' if norm_ok else '✗ Check shift/scale factors'}")
        lines.append("")
        lines.append("  ── Reconstruction Quality ──")
        lines.append(f"  PSNR  : {psnr_val:.2f} dB")
        if psnr_val > 35:
            lines.append("  Grade : ✓ Excellent — VAE round-trip is clean")
        elif psnr_val > 28:
            lines.append("  Grade : ✓ Good — acceptable for training")
        elif psnr_val > 20:
            lines.append("  Grade : ⚠ Mediocre — check decode path / factors")
        else:
            lines.append("  Grade : ✗ Poor — something is wrong with decode")
        lines.append("")
        lines.append("  ── Per-Channel Means (first 16) ──")
        for i, (m, s) in enumerate(zip(stats["ch_means"][:16], stats["ch_stds"][:16])):
            bar = "█" * int(min(abs(m) * 10, 20))
            lines.append(f"  ch{i:02d}  mean={m:+.4f}  std={s:.4f}  {bar}")
        if stats["channels"] > 16:
            lines.append(f"  ... and {stats['channels'] - 16} more channels")
        lines.append("━" * 52)

        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("end", "\n".join(lines))
        self.stats_text.config(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()