"""
SDXL UNet Channel Surgery GUI — 4ch → 12ch
==========================================
Works with single .safetensors files (full models or standalone unets).
Run with: python expand_unet_gui.py
"""

import tkinter as tk
from tkinter import filedialog
import threading
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from pathlib import Path


# ── Core surgery logic ────────────────────────────────────────────────────────

def run_surgery(model_path, output_path, log_fn):
    path = Path(model_path)

    log_fn(f"Loading: {path.name}\n")
    state_dict = load_file(str(path))

    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    log_fn(f"Keys loaded: {len(state_dict)}\n")

    # Detect format
    has_diffusion = any("model.diffusion_model" in k for k in state_dict)
    log_fn(f"Format: {'full model (ldm/ComfyUI)' if has_diffusion else 'standalone unet'}\n")

    # Find conv_in key
    conv_key = None
    candidates =[
        "model.diffusion_model.input_blocks.0.0.weight",
        "conv_in.weight",
    ]
    for c in candidates:
        if c in state_dict:
            conv_key = c
            break
    if conv_key is None:
        for k in state_dict:
            if "conv_in" in k and k.endswith(".weight"):
                conv_key = k
                break
    if conv_key is None:
        raise ValueError(
            "Could not find conv_in.weight.\nSample keys:\n" +
            "\n".join(list(state_dict.keys())[:8])
        )

    old_weight = state_dict[conv_key]
    log_fn(f"Key:       {conv_key}\n")
    log_fn(f"Old shape: {tuple(old_weight.shape)}\n")

    if old_weight.shape[1] == 12:
        raise ValueError("Model already has 12 input channels — nothing to do.")
    if old_weight.shape[1] != 4:
        raise ValueError(f"Expected 4 input channels, got {old_weight.shape[1]}.")

    target_dtype = old_weight.dtype
    log_fn(f"Precision: {target_dtype}\n")

    # Expand conv_in — zero init new channels
    out_ch, _, kH, kW = old_weight.shape
    new_weight = torch.zeros(out_ch, 12, kH, kW, dtype=target_dtype)
    new_weight[:, :4, :, :] = old_weight

    state_dict[conv_key] = new_weight
    log_fn(f"New shape: {tuple(new_weight.shape)}\n")
    log_fn("Channels 0-3: original weights preserved\n")
    log_fn("Channels 4-11: zero initialized\n")

    log_fn(f"\nSaving: {Path(output_path).name}\n")
    save_file(state_dict, str(output_path), metadata=metadata)
    log_fn("Saved successfully.\n")


# ── GUI ───────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UNet Channel Surgery")
        self.resizable(False, False)
        self.configure(bg="#0e0e10")
        self.BG     = "#0e0e10"
        self.PANEL  = "#16161a"
        self.BORDER = "#2a2a30"
        self.ACCENT = "#00e5ff"
        self.RED    = "#ff4d6d"
        self.FG     = "#e8e8f0"
        self.DIM    = "#6b6b80"
        self.MONO   = ("Courier New", 9)
        self._build()
        self._center()

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build(self):
        # Header
        tk.Frame(self, bg=self.BG, height=24).pack()
        row = tk.Frame(self, bg=self.BG)
        row.pack(fill="x", padx=28)
        tk.Label(row, text="◈", font=("Courier New", 18, "bold"),
                 fg=self.ACCENT, bg=self.BG).pack(side="left", padx=(0, 10))
        tk.Label(row, text="UNet Channel Surgery",
                 font=("Georgia", 16, "bold"), fg=self.FG, bg=self.BG).pack(side="left")
        tk.Label(row, text="4ch → 12ch", font=self.MONO,
                 fg=self.ACCENT, bg=self.BG).pack(side="left", padx=(10, 0))

        tk.Label(self, text="Expand input channels for reference latent conditioning.",
                 font=("Georgia", 9, "italic"), fg=self.DIM, bg=self.BG
                 ).pack(anchor="w", padx=42, pady=(4, 14))
        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=self.BG, padx=28, pady=24)
        body.pack(fill="both", expand=True)

        self._field(body, "Model (.safetensors)", "Browse", self._browse_model, "model_var")
        tk.Frame(body, bg=self.BG, height=12).pack()
        self._field(body, "Output (.safetensors)", "Save As", self._browse_output, "output_var")
        tk.Frame(body, bg=self.BG, height=20).pack()

        self.run_btn = tk.Button(
            body, text="RUN SURGERY",
            font=("Courier New", 11, "bold"),
            fg="#0e0e10", bg=self.ACCENT,
            activebackground="#00b8d9", activeforeground="#0e0e10",
            relief="flat", bd=0, cursor="hand2",
            padx=32, pady=12, command=self._run
        )
        self.run_btn.pack(fill="x")
        tk.Frame(body, bg=self.BG, height=16).pack()

        # Log header
        lh = tk.Frame(body, bg=self.BG)
        lh.pack(fill="x", pady=(0, 6))
        tk.Label(lh, text="LOG", font=("Courier New", 8, "bold"),
                 fg=self.DIM, bg=self.BG).pack(side="left")
        self.dot = tk.Label(lh, text="●", font=("Courier New", 8),
                            fg=self.BORDER, bg=self.BG)
        self.dot.pack(side="left", padx=6)

        # Log box
        lw = tk.Frame(body, bg=self.PANEL, highlightthickness=1,
                      highlightbackground=self.BORDER)
        lw.pack(fill="both", expand=True)
        self.log_box = tk.Text(
            lw, height=14, bg=self.PANEL, fg="#a0a0b8",
            font=self.MONO, relief="flat", bd=0,
            insertbackground=self.ACCENT, selectbackground=self.BORDER,
            wrap="word", padx=12, pady=10, state="disabled"
        )
        sb = tk.Scrollbar(lw, command=self.log_box.yview, bg=self.PANEL,
                          troughcolor=self.PANEL, activebackground=self.BORDER)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True)

        self._log("Ready. Select a .safetensors model file to begin.\n")
        self._log("Supports full models (ldm/ComfyUI) and standalone unet files.\n")

    def _field(self, parent, label, btn_text, cmd, attr):
        tk.Label(parent, text=label, font=("Courier New", 8, "bold"),
                 fg=self.DIM, bg=self.BG).pack(anchor="w", pady=(0, 4))
        row = tk.Frame(parent, bg=self.PANEL, highlightthickness=1,
                       highlightbackground=self.BORDER)
        row.pack(fill="x")
        var = tk.StringVar()
        setattr(self, attr, var)
        tk.Entry(row, textvariable=var, bg=self.PANEL, fg=self.FG,
                 font=self.MONO, relief="flat", bd=0,
                 insertbackground=self.ACCENT,
                 selectbackground=self.BORDER).pack(
                     side="left", fill="x", expand=True, padx=(12, 0), pady=10)
        tk.Button(row, text=btn_text,
                  font=("Courier New", 8, "bold"),
                  fg="#0e0e10", bg=self.ACCENT,
                  activebackground="#00b8d9", activeforeground="#0e0e10",
                  relief="flat", bd=0, cursor="hand2",
                  padx=12, pady=6, command=cmd).pack(side="right", padx=8, pady=6)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")]
        )
        if path:
            self.model_var.set(path)
            self._log(f"Model: {Path(path).name}\n")
            if not self.output_var.get():
                p = Path(path)
                self.output_var.set(str(p.parent / (p.stem + "_12ch.safetensors")))
                self._log(f"Output auto-set: {p.stem}_12ch.safetensors\n")

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=".safetensors",
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")]
        )
        if path:
            self.output_var.set(path)
            self._log(f"Output: {Path(path).name}\n")

    def _log(self, msg, color=None):
        self.log_box.configure(state="normal")
        if color:
            tag = f"c{color.replace('#','')}"
            self.log_box.tag_configure(tag, foreground=color)
            self.log_box.insert("end", msg, tag)
        else:
            self.log_box.insert("end", msg)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _dot(self, state):
        self.dot.configure(fg={"idle": self.BORDER, "running": "#ffcc00",
                                "done": self.ACCENT, "error": self.RED}.get(state, self.BORDER))

    def _run(self):
        model  = self.model_var.get().strip()
        output = self.output_var.get().strip()
        if not model:
            self._log("✗ Please select a model file.\n", self.RED); return
        if not output:
            self._log("✗ Please set an output path.\n", self.RED); return
        if not output.endswith(".safetensors"):
            output += ".safetensors"
            self.output_var.set(output)
        self.run_btn.configure(state="disabled", text="RUNNING...")
        self._dot("running")
        threading.Thread(target=self._worker, args=(model, output), daemon=True).start()

    def _worker(self, model, output):
        def log(msg, color=None):
            self.after(0, lambda m=msg, c=color: self._log(m, c))
        try:
            log(f"\n── Surgery starting ──────────────────────────\n", self.ACCENT)
            run_surgery(model, output, log)
            log(f"\n── Complete ───────────────────────────────────\n", self.ACCENT)
            log("Surgery successful.\n", self.ACCENT)
            log("\nTrainer usage:\n")
            log("  ref = torch.zeros_like(latents)\n")
            log("  inp = torch.cat([noisy, ref], dim=1)\n")
            log("  pred = unet(inp, t, ...)\n")
            log("  ref = (noisy - t_exp * pred).detach()\n")
            self.after(0, lambda: self._dot("done"))
        except Exception as e:
            log(f"\n✗ Error: {e}\n", self.RED)
            self.after(0, lambda: self._dot("error"))
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal", text="RUN SURGERY"))


if __name__ == "__main__":
    app = App()
    app.mainloop()