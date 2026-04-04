import tkinter as tk
from tkinter import filedialog
import threading
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  Core surgery
#
#  What we actually do:
#    time_embed.0 (or time_embedding.linear_1) is a Linear layer:
#      weight shape: (time_embed_dim, model_channels) = (1280, 320)
#
#    We expand it to (1280, 640) — init both halves at 0.5 × original.
#
#    Why 0.5 × ?
#      When t_high ≈ t_low (small dt), emb_high ≈ emb_low, so:
#        0.5*W @ emb_high + 0.5*W @ emb_low ≈ W @ emb_high
#      Output is identical to the pretrained model — zero disruption.
#      When t_high and t_low differ, the two halves produce different
#      contributions and the gradient teaches them to specialise.
#
#    Why this works where cond_proj failed:
#      cond_proj added a side branch the model could zero out.
#      This expansion IS the only path from sinusoidal → time MLP.
#      There is no bypass. The model cannot ignore t_low.
#
#  Training / inference:
#    Feed: cat([sinusoidal(t_high * 1000, 320), sinusoidal(t_low * 1000, 320)])
#    Shape: (B, 640) — straight into the expanded linear.
#    Use a forward_pre_hook on time_embed to inject this vector.
#    No new parameters, no architecture changes beyond the weight shape.
# ─────────────────────────────────────────────────────────────

# Keys to try in order — handles ComfyUI/ldm and diffusers checkpoints
_LINEAR1_CANDIDATES = [
    "model.diffusion_model.time_embed.0.weight",   # ComfyUI / ldm format
    "time_embedding.linear_1.weight",               # diffusers HF format
]

_BIAS1_CANDIDATES = [
    "model.diffusion_model.time_embed.0.bias",
    "time_embedding.linear_1.bias",
]

_STALE_PREFIXES = ["target_t_proj", "cond_proj"]


def run_embed_surgery(model_path, output_path, log_fn):
    path = Path(model_path)
    log_fn(f"Loading: {path.name}\n")

    state_dict = load_file(str(path))
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    log_fn(f"Keys loaded: {len(state_dict)}\n\n")

    # -- Remove stale surgery keys from old attempts
    stale = [k for k in list(state_dict.keys())
             if any(k.startswith(p) or p in k for p in _STALE_PREFIXES)
             and k != "__aozora_delta__"]
    if stale:
        for k in stale:
            del state_dict[k]
        log_fn(f"Removed {len(stale)} stale keys: {stale}\n\n")

    # -- Find the first linear weight
    linear1_key = next((k for k in _LINEAR1_CANDIDATES if k in state_dict), None)
    if linear1_key is None:
        raise ValueError(
            f"Could not find time_embed linear_1.\n"
            f"Tried: {_LINEAR1_CANDIDATES}\n"
            f"Available keys (first 20): {list(state_dict.keys())[:20]}"
        )

    w = state_dict[linear1_key]
    out_features, in_features = w.shape
    dtype = w.dtype

    log_fn(f"Found : {linear1_key}\n")
    log_fn(f"Shape : ({out_features}, {in_features})   dtype={dtype}\n")

    # -- Guard: only run on original unmodified checkpoint
    if in_features != 320:
        raise ValueError(
            f"Expected in_features=320 (model_channels for SDXL).\n"
            f"Got {in_features} — this model may already be surgically modified.\n"
            f"Run surgery only once on the original base checkpoint."
        )

    model_channels = in_features     # 320
    time_embed_dim = out_features     # 1280

    log_fn(f"\nmodel_channels = {model_channels}\n")
    log_fn(f"time_embed_dim = {time_embed_dim}\n\n")

    # -- Build expanded weight (1280, 640)
    # Both halves = 0.5 × original weight.
    # See module docstring above for why 0.5.
    new_w = torch.zeros(time_embed_dim, model_channels * 2, dtype=dtype)
    new_w[:, :model_channels] = w * 0.5   # first half  → processes t_high embedding
    new_w[:, model_channels:] = w * 0.5   # second half → processes t_low  embedding

    state_dict[linear1_key] = new_w

    log_fn(f"Expanded weight:\n")
    log_fn(f"  ({time_embed_dim}, {model_channels})  →  ({time_embed_dim}, {model_channels * 2})\n")
    log_fn(f"  t_high path : cols [0    : {model_channels}] = 0.5 × W_orig\n")
    log_fn(f"  t_low  path : cols [{model_channels} : {model_channels * 2}] = 0.5 × W_orig\n\n")

    # -- Bias stays the same shape — no touch needed
    bias1_key = next((k for k in _BIAS1_CANDIDATES if k in state_dict), None)
    if bias1_key:
        log_fn(f"Bias {bias1_key}: shape {tuple(state_dict[bias1_key].shape)} — unchanged\n\n")

    # -- Sentinel so inference auto-detects dual-endpoint mode
    state_dict["__aozora_delta__"] = torch.tensor([1.0], dtype=torch.float32)
    log_fn(f"Injected __aozora_delta__ sentinel\n\n")

    # -- Save
    log_fn(f"Saving → {Path(output_path).name}\n")
    save_file(state_dict, str(output_path), metadata=metadata)

    log_fn(
        f"\n{'─'*54}\n"
        f"Done.\n\n"
        f"What changed:\n"
        f"  {linear1_key}\n"
        f"    ({time_embed_dim}, {model_channels}) → ({time_embed_dim}, {model_channels * 2})\n\n"
        f"How to use:\n"
        f"  Training  — hook on time_embedding, inject:\n"
        f"    cat([sinusoidal(t_high*1000, {model_channels}),\n"
        f"         sinusoidal(t_low *1000, {model_channels})]) → shape (B, {model_channels*2})\n\n"
        f"  Inference — same hook in ComfyUI wrapper,\n"
        f"    auto-detected via __aozora_delta__ sentinel.\n\n"
        f"  No save-code changes needed. The expanded weight\n"
        f"  saves and loads like any other tensor.\n"
        f"{'─'*54}\n"
    )


# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    BG     = "#0e0e10"
    PANEL  = "#16161a"
    BORDER = "#2a2a30"
    ACCENT = "#00e5ff"
    RED    = "#ff4d6d"
    FG     = "#e8e8f0"
    DIM    = "#6b6b80"
    MONO   = ("Courier New", 9)

    def __init__(self):
        super().__init__()
        self.title("Aozora — Time Embed Surgery")
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self._build()

    def _build(self):
        tk.Frame(self, bg=self.BG, height=20).pack()

        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=28)
        tk.Label(header, text="o", font=("Courier New", 18, "bold"),
                 fg=self.ACCENT, bg=self.BG).pack(side="left", padx=(0, 10))
        tk.Label(header, text="Time Embed Surgery",
                 font=("Georgia", 15, "bold"), fg=self.FG, bg=self.BG).pack(side="left")

        tk.Label(
            self,
            text=(
                "Expands time_embed.linear_1: (1280, 320) → (1280, 640).\n"
                "First 320 cols = t_high signal.  Last 320 cols = t_low signal.\n"
                "Both halves init at 0.5× original — model is stable from step 1.\n"
                "No bypass path. The model physically must use both endpoints."
            ),
            font=("Georgia", 9, "italic"), fg=self.DIM, bg=self.BG, justify="left"
        ).pack(anchor="w", padx=42, pady=(4, 14))

        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=self.BG, padx=28, pady=20)
        body.pack(fill="both", expand=True)

        self._file_row(body, "Model (.safetensors)", "Browse",  self._browse_model,  "model_var")
        tk.Frame(body, bg=self.BG, height=10).pack()
        self._file_row(body, "Output (.safetensors)", "Save As", self._browse_output, "output_var")
        tk.Frame(body, bg=self.BG, height=16).pack()

        self.run_btn = tk.Button(
            body, text="EXPAND TIME EMBED",
            font=("Courier New", 11, "bold"),
            fg="#0e0e10", bg=self.ACCENT,
            activebackground="#00b8d9", relief="flat", bd=0,
            cursor="hand2", padx=32, pady=12,
            command=self._run
        )
        self.run_btn.pack(fill="x")
        tk.Frame(body, bg=self.BG, height=14).pack()

        lw = tk.Frame(body, bg=self.PANEL, highlightthickness=1,
                      highlightbackground=self.BORDER)
        lw.pack(fill="both", expand=True)

        self.log_box = tk.Text(
            lw, height=16, bg=self.PANEL, fg="#a0a0b8",
            font=self.MONO, relief="flat", bd=0,
            wrap="word", padx=12, pady=10, state="disabled"
        )
        sb = tk.Scrollbar(lw, command=self.log_box.yview,
                          bg=self.PANEL, troughcolor=self.PANEL)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True)

        self._log("Ready.\n")
        self._log("Run surgery only once on the original base checkpoint.\n")

    def _file_row(self, parent, label, btn_text, cmd, attr):
        tk.Label(parent, text=label,
                 font=("Courier New", 8, "bold"),
                 fg=self.DIM, bg=self.BG).pack(anchor="w", pady=(0, 4))
        row = tk.Frame(parent, bg=self.PANEL, highlightthickness=1,
                       highlightbackground=self.BORDER)
        row.pack(fill="x")
        var = tk.StringVar()
        setattr(self, attr, var)
        tk.Entry(row, textvariable=var, bg=self.PANEL, fg=self.FG,
                 font=self.MONO, relief="flat", bd=0,
                 insertbackground=self.ACCENT
                 ).pack(side="left", fill="x", expand=True, padx=(12, 0), pady=10)
        tk.Button(row, text=btn_text,
                  font=("Courier New", 8, "bold"),
                  fg="#0e0e10", bg=self.ACCENT, relief="flat", bd=0,
                  cursor="hand2", padx=12, pady=6,
                  command=cmd).pack(side="right", padx=8, pady=6)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            filetypes=[("Safetensors", "*.safetensors")]
        )
        if path:
            self.model_var.set(path)
            p = Path(path)
            if not self.output_var.get():
                self.output_var.set(str(p.parent / (p.stem + "_expanded.safetensors")))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".safetensors",
            filetypes=[("Safetensors", "*.safetensors")]
        )
        if path:
            self.output_var.set(path)

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

    def _run(self):
        model  = self.model_var.get().strip()
        output = self.output_var.get().strip()
        if not model:
            self._log("Select a model.\n", self.RED); return
        if not output:
            self._log("Set output path.\n", self.RED); return
        self.run_btn.configure(state="disabled", text="RUNNING...")
        threading.Thread(target=self._worker, args=(model, output), daemon=True).start()

    def _worker(self, model, output):
        def log(msg, color=None):
            self.after(0, lambda m=msg, c=color: self._log(m, c))
        try:
            log(f"\n{'─'*54}\n", self.ACCENT)
            run_embed_surgery(model, output, log)
        except Exception as e:
            log(f"\nError: {e}\n", self.RED)
            import traceback
            log(traceback.format_exc(), self.RED)
        finally:
            self.after(0, lambda: self.run_btn.configure(
                state="normal", text="EXPAND TIME EMBED"
            ))


if __name__ == "__main__":
    App().mainloop()