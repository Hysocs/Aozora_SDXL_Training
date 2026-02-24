"""
SDXL UNet 32-Channel Surgery & VAE Merge Tool
Converts a standard 4-channel SDXL UNet to 32-channels AND merges a 32ch VAE (like FLUX 2).
"""

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import torch
from safetensors.torch import load_file, save_file
import struct
import json
from pathlib import Path
import os

# ─────────────────────────────────────────────────────────────────────
#  VAE SNIFFER (To Auto-Detect Channels)
# ─────────────────────────────────────────────────────────────────────

def sniff_vae_channels(path: str) -> int:
    """Read the safetensors header to detect latent channels without loading weights."""
    try:
        with open(path, 'rb') as f:
            n_bytes = f.read(8)
            if len(n_bytes) < 8: return 4
            n = struct.unpack('<Q', n_bytes)[0]
            hdr = json.loads(f.read(n).decode('utf-8'))
            
        keys = set(k for k in hdr if k != '__metadata__')
        prefix = 'first_stage_model.' if any(k.startswith('first_stage_model.') for k in keys) else ''
        
        probes = {
            f'{prefix}quant_conv.weight':       lambda s: s[0] // 2,
            f'{prefix}encoder.conv_out.weight': lambda s: s[0] // 2,
            f'{prefix}post_quant_conv.weight':  lambda s: s[1],
            f'{prefix}decoder.conv_in.weight':  lambda s: s[1],
        }
        
        for k, fn in probes.items():
            if k in keys:
                shape = hdr[k].get('shape', [])
                if shape: return fn(shape)
                
        if 'encoder.conv_out.weight' in keys:
            shape = hdr['encoder.conv_out.weight'].get('shape', [])
            if shape: return shape[0] // 2
            
    except Exception as e:
        print(f"Failed to sniff VAE: {e}")
    return 32 # Default to 32 for the new Flux2 VAE target

# ─────────────────────────────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────────────────────────────

class UNetSurgeryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDXL UNet Surgery & VAE Merger Tool (32-Channel Edition)")
        self.geometry("900x680")
        self.configure(bg="#080808")
        
        self.input_ckpt = tk.StringVar()
        self.target_vae = tk.StringVar()
        self.output_ckpt = tk.StringVar()
        self.channels_var = tk.IntVar(value=32)  # Defaulted to 32
        
        self.init_modes = [
            "Zero-Pad New Channels (Keep Orig 4ch)",
            "Random Normal New Channels (Keep Orig 4ch)",
            "Full Re-Init (Kaiming In / Zero Out - Best for FLUX2 32ch VAE)"
        ]
        self.init_mode_var = tk.StringVar(value=self.init_modes[2])
        self.merge_vae_var = tk.BooleanVar(value=True)
        
        self._build_ui()

    def _build_ui(self):
        btn_style = {"bg": "#1a3a5c", "fg": "white", "relief": tk.FLAT, "padx": 10, "pady": 6, "font": ("Consolas", 10, "bold")}
        lbl_style = {"bg": "#080808", "fg": "#00ffcc", "font": ("Consolas", 10, "bold"), "anchor": "w"}
        ent_style = {"bg": "#1e1e1e", "fg": "white", "insertbackground": "white", "relief": tk.FLAT, "font": ("Consolas", 10)}
        
        main_frame = tk.Frame(self, bg="#080808", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="SDXL UNET SURGERY + 32-CH VAE MERGER", 
                 bg="#080808", fg="#ffaa00", font=("Consolas", 14, "bold")).pack(anchor="w", pady=(0, 20))
                 
        # ── Input Checkpoint ──
        tk.Label(main_frame, text="1. Select Base SDXL Checkpoint (4ch)", **lbl_style).pack(fill=tk.X)
        row1 = tk.Frame(main_frame, bg="#080808")
        row1.pack(fill=tk.X, pady=(4, 15))
        tk.Entry(row1, textvariable=self.input_ckpt, state='readonly', **ent_style).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        tk.Button(row1, text="BROWSE", command=self._sel_input, **btn_style).pack(side=tk.LEFT, padx=(10, 0))

        # ── Target VAE ──
        tk.Label(main_frame, text="2. Select Target 32ch VAE (Will be merged into the new checkpoint)", **lbl_style).pack(fill=tk.X)
        row2 = tk.Frame(main_frame, bg="#080808")
        row2.pack(fill=tk.X, pady=(4, 5))
        tk.Entry(row2, textvariable=self.target_vae, state='readonly', **ent_style).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        tk.Button(row2, text="BROWSE", command=self._sel_vae, **btn_style).pack(side=tk.LEFT, padx=(10, 0))
        
        # Checkbox for VAE merge
        tk.Checkbutton(main_frame, text="Delete old SDXL 4ch VAE and inject this new 32ch VAE", 
                       variable=self.merge_vae_var, bg="#080808", fg="#aaa", selectcolor="#222", 
                       font=("Consolas", 9)).pack(anchor="w", pady=(0, 15))

        # ── Settings ──
        row_cfg = tk.Frame(main_frame, bg="#080808")
        row_cfg.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(row_cfg, text="Target Channels:", bg="#080808", fg="#aaa", font=("Consolas", 10)).pack(side=tk.LEFT)
        tk.Entry(row_cfg, textvariable=self.channels_var, width=5, **ent_style).pack(side=tk.LEFT, padx=(5, 20), ipady=5)
        
        tk.Label(row_cfg, text="UNet Init Strategy:", bg="#080808", fg="#aaa", font=("Consolas", 10)).pack(side=tk.LEFT)
        cb = ttk.Combobox(row_cfg, textvariable=self.init_mode_var, values=self.init_modes, state="readonly", width=60)
        cb.pack(side=tk.LEFT, padx=(5, 0))
        
        # ── Output Checkpoint ──
        tk.Label(main_frame, text="3. Save Output Checkpoint As", **lbl_style).pack(fill=tk.X)
        row3 = tk.Frame(main_frame, bg="#080808")
        row3.pack(fill=tk.X, pady=(4, 20))
        tk.Entry(row3, textvariable=self.output_ckpt, state='readonly', **ent_style).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        tk.Button(row3, text="BROWSE", command=self._sel_output, **btn_style).pack(side=tk.LEFT, padx=(10, 0))

        # ── Run Button ──
        run_btn = tk.Button(main_frame, text="⚡ EXECUTE 32-CH SURGERY ⚡", command=self._run_surgery, 
                            bg="#006644", fg="white", relief=tk.FLAT, font=("Consolas", 14, "bold"), pady=10)
        run_btn.pack(fill=tk.X, pady=(10, 20))

        # ── Console Log ──
        self.log_box = tk.Text(main_frame, bg="#0d0d0d", fg="#ccc", font=("Consolas", 9), relief=tk.FLAT, height=12)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log("Ready. Awaiting inputs...")

    def log(self, msg: str):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.update_idletasks()

    def _sel_input(self):
        f = filedialog.askopenfilename(title="Select Base SDXL Safetensors", filetypes=[("Safetensors", "*.safetensors")])
        if f: self.input_ckpt.set(f)

    def _sel_vae(self):
        f = filedialog.askopenfilename(title="Select Target VAE", filetypes=[("Safetensors", "*.safetensors")])
        if f:
            self.target_vae.set(f)
            ch = sniff_vae_channels(f)
            self.channels_var.set(ch)
            self.log(f"Detected target VAE channels: {ch}")

    def _sel_output(self):
        f = filedialog.asksaveasfilename(title="Save Modified Model", defaultextension=".safetensors", 
                                         filetypes=[("Safetensors", "*.safetensors")])
        if f: self.output_ckpt.set(f)

    def _run_surgery(self):
        if not self.input_ckpt.get() or not self.output_ckpt.get():
            self.log("ERROR: Please select an input and output checkpoint.")
            return
            
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        in_path = self.input_ckpt.get()
        out_path = self.output_ckpt.get()
        target_ch = self.channels_var.get()
        init_mode = self.init_mode_var.get()
        vae_path = self.target_vae.get()
        merge_vae = self.merge_vae_var.get()
        
        self.log(f"\n[{'-'*50}]")
        self.log(f"Starting surgery: 4ch -> {target_ch}ch")
        self.log(f"Loading {os.path.basename(in_path)} into memory...")
        
        try:
            # 1. LOAD BASE CHECKPOINT
            tensors = load_file(in_path, device="cpu")
            is_ldm = any(k.startswith("model.diffusion_model.") for k in tensors.keys())
            
            if is_ldm:
                in_w_key = "model.diffusion_model.input_blocks.0.0.weight"
                out_w_key = "model.diffusion_model.out.2.weight"
                out_b_key = "model.diffusion_model.out.2.bias"
                self.log("Detected Format: Single File Checkpoint (LDM)")
            else:
                in_w_key = "conv_in.weight"
                out_w_key = "conv_out.weight"
                out_b_key = "conv_out.bias"
                self.log("Detected Format: Diffusers Checkpoint")

            # 2. UNET SURGERY
            old_in_w = tensors[in_w_key]
            old_out_w = tensors[out_w_key]
            old_out_b = tensors.get(out_b_key, None)

            c_out_in, c_in_in, k1, k2 = old_in_w.shape
            c_out_out, c_in_out, k3, k4 = old_out_w.shape

            new_in_w = torch.zeros((c_out_in, target_ch, k1, k2), dtype=old_in_w.dtype)
            new_out_w = torch.zeros((target_ch, c_in_out, k3, k4), dtype=old_out_w.dtype)
            new_out_b = torch.zeros((target_ch,), dtype=old_out_b.dtype) if old_out_b is not None else None

            if "Zero-Pad" in init_mode:
                new_in_w[:, :c_in_in, :, :] = old_in_w
                new_out_w[:c_out_out, :, :, :] = old_out_w
                if new_out_b is not None: new_out_b[:c_out_out] = old_out_b
                
            elif "Random Normal" in init_mode:
                new_in_w[:, :c_in_in, :, :] = old_in_w
                torch.nn.init.normal_(new_in_w[:, c_in_in:, :, :], mean=0.0, std=old_in_w.std().item())
                new_out_w[:c_out_out, :, :, :] = old_out_w
                torch.nn.init.normal_(new_out_w[c_out_out:, :, :, :], mean=0.0, std=old_out_w.std().item())
                if new_out_b is not None: new_out_b[:c_out_out] = old_out_b
                
            elif "Full Re-Init" in init_mode:
                torch.nn.init.kaiming_normal_(new_in_w, mode='fan_out', nonlinearity='relu')
                torch.nn.init.constant_(new_out_w, 0.0)
                if new_out_b is not None: torch.nn.init.constant_(new_out_b, 0.0)

            tensors[in_w_key] = new_in_w
            tensors[out_w_key] = new_out_w
            if new_out_b is not None: tensors[out_b_key] = new_out_b
            self.log(f"✓ UNet expanded to {target_ch} channels.")

            # 3. VAE MERGE
            if vae_path and merge_vae:
                self.log(f"\nMerging {target_ch}ch VAE from {os.path.basename(vae_path)}...")
                vae_tensors = load_file(vae_path, device="cpu")
                
                # Strip old VAE weights from base checkpoint
                old_vae_keys = [k for k in tensors.keys() if k.startswith("first_stage_model.") or k.startswith("vae.")]
                for k in old_vae_keys:
                    del tensors[k]
                self.log(f"Deleted {len(old_vae_keys)} old VAE tensors.")

                # Inject new VAE weights (handling prefixing)
                injected_count = 0
                for k, v in vae_tensors.items():
                    if is_ldm:
                        new_k = f"first_stage_model.{k}" if not k.startswith("first_stage_model.") else k
                    else:
                        new_k = k 
                        
                    tensors[new_k] = v.to(old_in_w.dtype)
                    injected_count += 1
                    
                self.log(f"✓ Injected {injected_count} new VAE tensors.")

            # 4. SAVE TO DISK
            self.log("\nSaving modified weights to disk (This may take a moment)...")
            save_file(tensors, out_path)
            self.log(f"SUCCESS! 32-Channel model saved to:")
            self.log(out_path)
            self.log(f"[{'-'*50}]\n")

        except Exception as e:
            self.log(f"\nCRITICAL ERROR during surgery: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = UNetSurgeryApp()
    app.mainloop()