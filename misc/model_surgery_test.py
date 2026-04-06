import tkinter as tk
from tkinter import filedialog
import threading
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  Core Surgery: NATIVE 400-dim Expansion
#
#  MODIFIES the model architecture permanently (no monkey-patch):
#
#  Before Surgery:
#    time_embed.0: (1280, 320)  ← standard SDXL
#    time_embed.2: (1280, 1280)
#
#  After Surgery:
#    time_embed.0: (1280, 400)  ← EXPANDED to accept dual input
#      └─ cols [0:320]   = original delta weights (unchanged)
#      └─ cols [320:400] = timestep weights (0.25x scaled slice, zero-init region)
#
#    time_embed.2: (1280, 1280) ← unchanged, now processes mixed 1280-dim
#
#    timestep_embed.0: (320, 80)   ← stored for reference / diagnostics
#    timestep_embed.2: (1280, 320) ← stored, initialized to ZERO
#
#  Why this works:
#    At Step 0, cols [320:400] are near-zero, so the layer behaves
#    identically to the original (320-dim) version. As training
#    progresses, gradients flow through the new columns, allowing
#    the timestep signal to be learned without breaking the model.
#
#  Signal Path (Native, No Monkey-Patch):
#    time_proj outputs 400-dim: [sinusoidal(delta_cond, 320) | sinusoidal(t_high, 80)]
#    → time_embed.0 (1280, 400) → 1280-dim
#    → time_embed.2 (1280, 1280) → 1280-dim
#    → downstream UNet
# ─────────────────────────────────────────────────────────────

_STALE_PREFIXES = ["target_t_proj", "cond_proj", "aux_proj"]

def run_dual_chain_surgery(model_path, output_path, log_fn):
    path = Path(model_path)
    log_fn(f"Loading: {path.name}\n")

    state_dict = load_file(str(path))
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}

    log_fn(f"Keys loaded: {len(state_dict)}\n\n")

    # ── Guard: refuse if already processed ──────────────────────
    already = next((k for k in state_dict.keys() if "timestep_embed" in k), None)
    if already is not None:
        raise ValueError(
            f"Dual chain already exists ({already}).\n"
            f"Run surgery only once on the original delta checkpoint."
        )

    # ── Remove stale keys ──────────────────────────────────────
    stale = [
        k for k in list(state_dict.keys())
        if any(k.startswith(p) or p in k for p in _STALE_PREFIXES)
        and k not in ("__aozora_delta__", "__aozora_delta_plus__", "__aozora_dual_chain__")
    ]
    if stale:
        for k in stale:
            del state_dict[k]
        log_fn(f"Removed {len(stale)} stale keys: {stale}\n\n")

    # ── Identify Format (SD vs HF) ──────────────────────────────
    prefix = ""
    is_hf = False
    
    if "model.diffusion_model.time_embed.0.weight" in state_dict:
        prefix = "model.diffusion_model."
        l1_name = "time_embed.0"
        l2_name = "time_embed.2"
        new_chain = "timestep_embed"
    elif "time_embedding.linear_1.weight" in state_dict:
        is_hf = True
        l1_name = "time_embedding.linear_1"
        l2_name = "time_embedding.linear_2"
        new_chain = "timestep_embedding"
    else:
        raise ValueError("Could not find standard time_embed linear_1/linear_2 weights.")

    # ── Read Original Weights ───────────────────────────────────
    w0 = state_dict[f"{prefix}{l1_name}.weight"]  # (1280, 320)
    b0 = state_dict[f"{prefix}{l1_name}.bias"]     # (1280,)
    w2 = state_dict[f"{prefix}{l2_name}.weight"]  # (1280, 1280)
    b2 = state_dict[f"{prefix}{l2_name}.bias"]     # (1280,)
    
    orig_in     = w0.shape[1]  # 320
    orig_hidden = w0.shape[0]  # 1280
    orig_out    = w2.shape[0]  # 1280

    dtype = w0.dtype

    log_fn(f"Original time_embed found:\n")
    log_fn(f"  Linear 1: ({orig_hidden}, {orig_in})  dtype={dtype}\n")
    log_fn(f"  Linear 2: ({orig_out}, {orig_hidden})  dtype={dtype}\n\n")

    if orig_in != 320:
        raise ValueError(f"Expected in_features=320, got {orig_in}.")

    # ── Calculate Dimensions ────────────────────────────────────
    delta_dim   = orig_in           # 320
    ts_dim      = orig_in // 4      # 80   (0.25x capacity)
    total_dim   = delta_dim + ts_dim # 400

    log_fn(f"Expanding to NATIVE dual-chain architecture:\n")
    log_fn(f"  delta_dim   = {delta_dim}\n")
    log_fn(f"  ts_dim      = {ts_dim}  (0.25x)\n")
    log_fn(f"  total_dim   = {total_dim}  (expanded input)\n\n")

    # ══════════════════════════════════════════════════════════
    #  STEP 1: Expand time_embed.0 from (1280, 320) → (1280, 400)
    # ══════════════════════════════════════════════════════════
    
    w0_fp32 = w0.float()
    b0_fp32 = b0.float()
    
    # Create expanded weight matrix
    new_w0 = torch.zeros((orig_hidden, total_dim), dtype=torch.float32)  # (1280, 400)
    new_b0 = b0_fp32.clone()  # Bias unchanged (1280,)
    
    # --- Region A: Delta columns [0:320] = EXACT COPY of original ---
    # This preserves 100% of original behavior for the delta signal
    new_w0[:, :delta_dim] = w0_fp32  # (1280, 320) copied exactly
    


    # Apply to the new columns
    new_w0[:, delta_dim:] = 0.0
    
    # Convert back to original dtype
    new_w0 = new_w0.to(dtype)
    new_b0 = new_b0.to(dtype)
    
    # Write EXPANDED weights back to time_embed.0
    state_dict[f"{prefix}{l1_name}.weight"] = new_w0  # Now (1280, 400)!
    state_dict[f"{prefix}{l1_name}.bias"]   = new_b0
    
    log_fn(f"EXPANDED {prefix}{l1_name}:\n")
    log_fn(f"  Weight: ({orig_hidden}, {total_dim})  ← was ({orig_hidden}, {orig_in})\n")
    log_fn(f"    cols [0:{delta_dim}]       = original (preserved)\n")
    log_fn(f"    cols [{delta_dim}:{total_dim}] = zero init (silent until trained)\n")
    log_fn(f"  Bias:   ({orig_hidden},)  (unchanged)\n\n")
    
    # ══════════════════════════════════════════════════════════
    #  STEP 2: Store reference timestep_embed weights
    #  (Not used by standard forward, but available for inspection
    #   and potential future use in custom inference code)
    # ══════════════════════════════════════════════════════════
    
    # Layer 1: (320, 80) - matches what the old surgery created
    # AFTER
    ref_ts_w0 = torch.zeros((orig_hidden // 4, ts_dim), dtype=dtype)
    ref_ts_b0 = torch.zeros((orig_hidden // 4,), dtype=dtype)
    
    # Layer 2: (1280, 320) - ZERO INIT (the "volume knob")
    ref_ts_w2 = torch.zeros((orig_out, orig_hidden // 4), dtype=dtype)  # (1280, 320)
    ref_ts_b2 = torch.zeros((orig_out,), dtype=dtype)                   # (1280,)
    
    if not is_hf:
        state_dict[f"{prefix}{new_chain}.0.weight"] = ref_ts_w0
        state_dict[f"{prefix}{new_chain}.0.bias"]   = ref_ts_b0
        state_dict[f"{prefix}{new_chain}.2.weight"] = ref_ts_w2
        state_dict[f"{prefix}{new_chain}.2.bias"]   = ref_ts_b2
    else:
        state_dict[f"{new_chain}.linear_1.weight"] = ref_ts_w0
        state_dict[f"{new_chain}.linear_1.bias"]   = ref_ts_b0
        state_dict[f"{new_chain}.linear_2.weight"] = ref_ts_w2
        state_dict[f"{new_chain}.linear_2.bias"]   = ref_ts_b2
    
    log_fn(f"Stored reference {new_chain} (for diagnostics):\n")
    log_fn(f"  Linear 1: ({orig_hidden // 4}, {ts_dim})\n")
    log_fn(f"  Linear 2: ({orig_out}, {orig_hidden // 4})  <- ZERO INIT\n\n")
    
    # ══════════════════════════════════════════════════════════
    #  STEP 3: Sentinel & Metadata
    # ══════════════════════════════════════════════════════════
    
    # Clean old sentinels
    for old in ("__aozora_delta__", "__aozora_delta_plus__"):
        if old in state_dict:
            del state_dict[old]

    # New sentinel indicating native dual-chain architecture
    state_dict["__aozora_dual_chain__"] = torch.tensor([1.0], dtype=torch.float32)
    
    # Store dimension info for easy reading
    state_dict["__aozora_dc_dims__"] = torch.tensor([delta_dim, ts_dim, total_dim], dtype=torch.float32)
    
    log_fn(f"Signal path (NATIVE ARCHITECTURE):\n")
    log_fn(f"  time_proj → sinusoidal({total_dim}) → time_embed.0({orig_hidden},{total_dim})\n")
    log_fn(f"                                         → time_embed.2({orig_out},{orig_hidden})\n")
    log_fn(f"                                         → UNet Downstream\n\n")
    
    log_fn(f"Input layout to time_embed.0:\n")
    log_fn(f"  [{0:>3}:{delta_dim:>3}] = delta signal (e.g., dt or blended cond)\n")
    log_fn(f"  [{delta_dim:>3}:{total_dim:>3}] = timestep signal (e.g., t_high)\n\n")
    
    log_fn(f"Saving → {Path(output_path).name}\n")
    save_file(state_dict, str(output_path), metadata=metadata)
    log_fn(f"Done.\n")


# ─────────────────────────────────────────────────────────────
#  UI (unchanged)
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
        self.title("Aozora — Dual Chain Surgery (NATIVE 400-dim)")
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self._build()

    def _build(self):
        tk.Frame(self, bg=self.BG, height=20).pack()

        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=28)
        tk.Label(header, text="◈", font=("Courier New", 18, "bold"),
                 fg=self.ACCENT, bg=self.BG).pack(side="left", padx=(0, 10))
        tk.Label(header, text="Dual Chain Surgery (NATIVE)",
                 font=("Georgia", 15, "bold"), fg=self.FG, bg=self.BG).pack(side="left")

        tk.Label(
            self,
            text=(
                "Permanently expands time_embed.0 to 400-dim input.\n"
                "No monkey-patching needed — works natively in ComfyUI.\n"
                "Columns [0:320] = delta (preserved) | [320:400] = timestep (learned).\n"
                "Run only once on the original delta checkpoint."
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
            body, text="EXPAND TO 400-DIM",
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
            lw, height=18, bg=self.PANEL, fg="#a0a0b8",
            font=self.MONO, relief="flat", bd=0,
            wrap="word", padx=12, pady=10, state="disabled"
        )
        sb = tk.Scrollbar(lw, command=self.log_box.yview,
                          bg=self.PANEL, troughcolor=self.PANEL)
        self.log_box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_box.pack(fill="both", expand=True)

        self._log("Ready.\n")
        self._log("This permanently modifies the model architecture.\n")
        self._log("time_embed.0: (1280, 320) → (1280, 400)\n\n")

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
        path = filedialog.askopenfilename(filetypes=[("Safetensors", "*.safetensors")])
        if path:
            self.model_var.set(path)
            p = Path(path)
            if not self.output_var.get():
                self.output_var.set(str(p.parent / (p.stem + "_dualchain_native.safetensors")))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".safetensors", filetypes=[("Safetensors", "*.safetensors")])
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
            run_dual_chain_surgery(model, output, log)
        except Exception as e:
            log(f"\nError: {e}\n", self.RED)
            import traceback
            log(traceback.format_exc(), self.RED)
        finally:
            self.after(0, lambda: self.run_btn.configure(state="normal", text="EXPAND TO 400-DIM"))

if __name__ == "__main__":
    App().mainloop()