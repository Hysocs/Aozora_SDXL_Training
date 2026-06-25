
"""
Model Layer Compare — v3
Supports: safetensors, pt, pth, bin, ckpt

Detection improvements over v2:
  - Random projection SVD for large matrices (was silently skipped at >4096)
  - avg_delta_per_element: baked LoRAs have 3-4x higher per-element delta than real trains
  - merge artifact detector: singular value gap clustering (B@A structure leaves a hard cliff)
  - untouched norm/embed flag: consistent with adapter merge or frozen-module training
  - same_tensor_ratio signal: >55% untouched tensors = narrow update scope / merge-like pattern
  - calibrated hard caps on full_tune_score when rank_ratio + low_rank_ratio both betray it
  - all new signals shown in the log and saved to lora_detection_summary.csv
"""

import csv
import json
import math
import queue
import threading
import traceback
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkfont

import torch

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except Exception:
    safe_open = None
    HAS_SAFETENSORS = False

# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_EXTS = (".safetensors", ".safetensor", ".pt", ".pth", ".bin", ".ckpt")
MODEL_FILE_PATTERN = "*.safetensors *.safetensor *.pt *.pth *.bin *.ckpt"

ANIMA_MERGE_TARGET_SUFFIXES = (
    ".adaln_modulation_cross_attn.1.weight",
    ".adaln_modulation_cross_attn.2.weight",
    ".adaln_modulation_mlp.1.weight",
    ".adaln_modulation_mlp.2.weight",
    ".adaln_modulation_self_attn.1.weight",
    ".adaln_modulation_self_attn.2.weight",
    ".cross_attn.k_proj.weight",
    ".cross_attn.o_proj.weight",
    ".cross_attn.output_proj.weight",
    ".cross_attn.q_proj.weight",
    ".cross_attn.v_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.o_proj.weight",
    ".self_attn.output_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.v_proj.weight",
    ".mlp.0.weight",
    ".mlp.layer1.weight",
    ".mlp.layer2.weight",
    ".mlp.2.weight",
)

ANIMA_MERGE_TARGET_PREFIXES = (
    "net.blocks.",
    "blocks.",
    "llm_adapter.blocks.",
)

ANIMA_PROTECTED_HINTS = (
    ".norm.",
    "_norm.",
    ".adaln_modulation_",
    ".final_layer.",
    ".llm_adapter.",
    ".t_embedder.",
    ".t_embedding_norm.",
    ".x_embedder.",
    ".patch_embed.",
    ".embed.",
    ".embedding.",
    ".bias",
)

# DiT-specific LoRA targets (FLUX, SD3, Wan, HunYuan, CogVideoX, SDXL UNet)
LORA_TARGET_HINTS = (
    "to_q", "to_k", "to_v", "to_out",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query", "key", "value",
    "attn.qkv", "attn.proj", "attn.to_q", "attn.to_k", "attn.to_v",
    "attn1.to_q", "attn1.to_k", "attn1.to_v",
    "attn2.to_q", "attn2.to_k", "attn2.to_v",
    "context_attn", "cross_attn",
    "adaLN_modulation", "mod_fc", "modulation.lin",
    "ffn", "mlp", "gate", "up", "down",
    "fc1", "fc2", "ff.net",
    "linear", "proj_in", "proj_out",
    "ip_adapter", "image_proj",
)

NON_LORA_HINTS = (
    "norm", "ln", "layernorm", "rms_norm",
    "embed", "embedding", "text_embed",
    "pos_embed", "time_embed", "t_embedder",
    "final_layer", "patch_embed",
    "out_proj.bias",
)

BLOCK_PATTERNS = [
    ("double_blocks",      "Double blocks",    "#c084fc"),
    ("single_blocks",      "Single blocks",    "#67e8f9"),
    ("joint_blocks",       "Joint blocks",     "#a78bfa"),
    ("transformer_blocks", "Transformer",      "#818cf8"),
    ("input_blocks",       "Input blocks",     "#f472b6"),
    ("middle_block",       "Middle block",     "#fb923c"),
    ("output_blocks",      "Output blocks",    "#34d399"),
    ("time_embed",         "Time embed",       "#facc15"),
    ("label_emb",          "Label embed",      "#fbbf24"),
    ("final_layer",        "Final layer",      "#f87171"),
    ("patch_embed",        "Patch embed",      "#60a5fa"),
    ("pos_embed",          "Pos embed",        "#94a3b8"),
    ("norm",               "Norms",            "#6b7280"),
    ("embed",              "Embeddings",       "#a3e635"),
]

# ─── Theme ────────────────────────────────────────────────────────────────────

DARK_BG      = "#0d0d0f"
DARK_BG2     = "#141418"
DARK_BG3     = "#1c1c22"
DARK_BORDER  = "#2a2a35"
DARK_ACCENT  = "#7c3aed"
DARK_ACCENT2 = "#a78bfa"
DARK_FG      = "#e2e2ee"
DARK_FG_MUT  = "#8888aa"
DARK_GREEN   = "#34d399"
DARK_RED     = "#f87171"
DARK_YELLOW  = "#facc15"
DARK_CYAN    = "#67e8f9"
DARK_ORANGE  = "#fb923c"

# ─── IO helpers ───────────────────────────────────────────────────────────────

def is_safetensors(path: str) -> bool:
    return str(path).lower().endswith((".safetensors", ".safetensor"))

def clean_path(value):
    return str(value or "").strip().strip("\"'")

def parse_prefixes(text: str):
    return [p.strip() for p in text.split(",") if p.strip()]

def strip_prefix(key: str, prefixes):
    for prefix in prefixes:
        if prefix and key.startswith(prefix):
            return key[len(prefix):]
    return key

def layer_name(key: str, depth: int):
    parts = key.split(".")
    return ".".join(parts[:depth]) if len(parts) > depth else key

def looks_like_lora_target(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in LORA_TARGET_HINTS)

def looks_like_anima_merge_target(key: str) -> bool:
    k = key.lower()
    return any(k.startswith(prefix) for prefix in ANIMA_MERGE_TARGET_PREFIXES) and any(k.endswith(suffix) for suffix in ANIMA_MERGE_TARGET_SUFFIXES)

def looks_like_anima_protected_area(key: str) -> bool:
    if looks_like_anima_merge_target(key):
        return False
    k = f".{key.lower()}"
    return any(h in k for h in ANIMA_PROTECTED_HINTS)

def looks_like_llm_adapter_area(key: str) -> bool:
    return key.lower().startswith("llm_adapter.")

def looks_like_anchor_area(key: str) -> bool:
    k = f".{key.lower()}"
    return any(h in k for h in (
        ".final_layer.",
        ".t_embedder.",
        ".t_embedding_norm.",
        ".x_embedder.",
        ".patch_embed.",
        ".embed.",
        ".embedding.",
        ".bias",
    ))

def looks_like_non_lora_area(key: str) -> bool:
    k = key.lower()
    return any(h in k for h in NON_LORA_HINTS)

def get_block_group(key: str):
    k = key.lower()
    for pattern, label, color in BLOCK_PATTERNS:
        if pattern in k:
            return label, color
    return "Other", "#4b5563"

def load_torch_state_dict(path: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for k in ("state_dict", "model", "module", "ema", "params"):
            if k in data and isinstance(data[k], dict):
                data = data[k]
                break
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported torch checkpoint format: {path}")
    return {str(k): v.detach().cpu() for k, v in data.items() if torch.is_tensor(v)}


# ─── Checkpoint reader ────────────────────────────────────────────────────────

class CheckpointReader:
    def __init__(self, path: str, prefixes):
        self.path = str(path)
        self.prefixes = prefixes
        self.state = None
        self.simple_to_raw = {}

        if is_safetensors(self.path):
            if safe_open is None:
                raise RuntimeError("safetensors not installed. Run: pip install safetensors")
            with safe_open(self.path, framework="pt", device="cpu") as f:
                raw_keys = list(f.keys())
        else:
            self.state = load_torch_state_dict(self.path)
            raw_keys = list(self.state.keys())

        for raw_key in raw_keys:
            simple_key = strip_prefix(raw_key, self.prefixes)
            if simple_key not in self.simple_to_raw:
                self.simple_to_raw[simple_key] = raw_key

    @property
    def keys(self):
        return set(self.simple_to_raw.keys())

    def get_tensor(self, simple_key):
        raw_key = self.simple_to_raw[simple_key]
        if is_safetensors(self.path):
            with safe_open(self.path, framework="pt", device="cpu") as f:
                return f.get_tensor(raw_key)
        return self.state[raw_key]


# ─── Tensor comparison ────────────────────────────────────────────────────────

def compare_tensors(base, other, atol, rtol, compute_diff):
    if tuple(base.shape) != tuple(other.shape):
        return {
            "status": "shape_mismatch", "same": False,
            "shape_base": list(base.shape), "shape_other": list(other.shape),
            "dtype_base": str(base.dtype).replace("torch.", ""),
            "dtype_other": str(other.dtype).replace("torch.", ""),
            "numel": int(base.numel()),
            "different_elements": None, "different_percent": None,
            "max_abs_diff": None, "mean_abs_diff": None,
            "total_abs_diff": None, "total_abs_diff_scientific": None,
        }

    numel = int(base.numel())
    if torch.is_floating_point(base) or torch.is_floating_point(other):
        b, o = base.float(), other.float()
        close = torch.isclose(b, o, atol=atol, rtol=rtol, equal_nan=True)
        different_elements = int((~close).sum().item())
        same = different_elements == 0
        max_abs_diff = mean_abs_diff = total_abs_diff = None
        if compute_diff:
            diff = (b - o).abs()
            max_abs_diff   = float(diff.max().item())  if diff.numel() else 0.0
            mean_abs_diff  = float(diff.mean().item()) if diff.numel() else 0.0
            total_abs_diff = float(diff.sum().item())  if diff.numel() else 0.0
    else:
        eq = base.eq(other)
        different_elements = int((~eq).sum().item())
        same = different_elements == 0
        max_abs_diff = mean_abs_diff = total_abs_diff = None

    return {
        "status": "same" if same else "different", "same": same,
        "shape_base": list(base.shape), "shape_other": list(other.shape),
        "dtype_base": str(base.dtype).replace("torch.", ""),
        "dtype_other": str(other.dtype).replace("torch.", ""),
        "numel": numel,
        "different_elements": different_elements,
        "different_percent": (different_elements / max(numel, 1)) * 100.0,
        "max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff,
        "total_abs_diff": total_abs_diff,
        "total_abs_diff_scientific": f"{total_abs_diff:.8e}" if total_abs_diff is not None else None,
    }


# ─── LoRA detection ───────────────────────────────────────────────────────────

def _svd_of_delta(delta: torch.Tensor, max_dim: int = 4096) -> torch.Tensor | None:
    """
    Return singular values of delta, using random projection for matrices
    larger than max_dim on either axis.  Returns None on failure.

    Random projection preserves the rank order: if delta is low-rank,
    the projected version is also low-rank.  We use k=512 which is large
    enough to detect ranks up to ~200 reliably.
    """
    rows, cols = delta.shape
    d = delta.float()

    if max(rows, cols) > max_dim:
        k = 512
        # Project rows down
        if rows > k:
            pr = torch.randn(k, rows) / math.sqrt(k)
            d = pr @ d          # k × cols
        # Project cols down
        if cols > k:
            pc = torch.randn(cols, k) / math.sqrt(k)
            d = d @ pc          # (rows or k) × k

    if not torch.isfinite(d).all():
        return None
    try:
        return torch.linalg.svdvals(d)
    except Exception:
        return None


def detect_merge_artifact(s: torch.Tensor) -> float:
    """
    A baked LoRA (weight += alpha * B @ A) has singular values with one
    hard cliff: the first r values are large, everything after is near-zero.
    A gradient-trained weight has a smooth exponential decay.

    Returns a clustering score 0-100.  High = likely merge artifact.
    The key signal is max_consecutive_ratio / mean_consecutive_ratio.
    """
    if s.numel() < 4:
        return 0.0
    s_cpu = s.float()
    # Consecutive ratios (larger / smaller)
    ratios = s_cpu[:-1] / (s_cpu[1:] + 1e-9)
    max_ratio  = float(ratios.max().item())
    mean_ratio = float(ratios.mean().item())
    # A cliff gives max_ratio >> mean_ratio
    cliff_score = (max_ratio / max(mean_ratio, 1.0)) * 10.0
    return min(100.0, cliff_score)


def lora_delta_rank_stats(base, other, energy_threshold=0.95, max_dim=4096):
    """Full rank statistics for a changed weight matrix."""
    if not (torch.is_floating_point(base) or torch.is_floating_point(other)):
        return None
    if base.ndim != 2 or other.ndim != 2:
        return None
    if tuple(base.shape) != tuple(other.shape):
        return None

    rows, cols = base.shape
    was_projected = max(rows, cols) > max_dim

    delta = (other.float() - base.float()).cpu()
    total_energy = float((delta * delta).sum().item())
    if total_energy <= 0:
        return None

    s = _svd_of_delta(delta, max_dim)
    if s is None or s.numel() == 0:
        return None

    energy     = s * s
    energy_sum = float(energy.sum().item())
    if energy_sum <= 0:
        return None

    cumulative    = torch.cumsum(energy, dim=0) / energy_sum
    # Effective rank: how many singular values to explain 95% of energy
    nonzero_idx   = (cumulative >= energy_threshold).nonzero(as_tuple=False)
    if nonzero_idx.numel() == 0:
        return None
    effective_rank_95 = int(nonzero_idx[0].item() + 1)

    # When projected, max_possible_rank is capped at projection dim
    proj_dim          = min(512, rows, cols) if was_projected else int(min(rows, cols))
    max_possible_rank = proj_dim
    rank_ratio_95     = effective_rank_95 / max(max_possible_rank, 1)

    top1  = float(cumulative[0].item())
    top4  = float(cumulative[min(3,  cumulative.numel()-1)].item())
    top8  = float(cumulative[min(7,  cumulative.numel()-1)].item())
    top16 = float(cumulative[min(15, cumulative.numel()-1)].item())

    # Spectral decay: ratio of s[15] to s[0] — closer to 0 = steeper cliff
    top_s = s[:min(16, s.numel())].tolist()
    decay_ratio = top_s[-1] / max(top_s[0], 1e-8) if len(top_s) >= 2 else 1.0

    # Merge artifact score
    merge_artifact_score = detect_merge_artifact(s)

    return {
        "effective_rank_95":     effective_rank_95,
        "max_possible_rank":     max_possible_rank,
        "rank_ratio_95":         rank_ratio_95,
        "top1_energy":           top1,
        "top4_energy":           top4,
        "top8_energy":           top8,
        "top16_energy":          top16,
        "spectral_decay_ratio":  decay_ratio,
        "merge_artifact_score":  merge_artifact_score,
        "was_projected":         was_projected,
    }


def lora_likelihood_label(score):
    if score >= 80: return "High"
    if score >= 60: return "Medium"
    if score >= 35: return "Low"
    return "Very low"


def diffcheck_comment_block(result):
    raw_result_type = result["likely_update_type"]
    if raw_result_type == "Classic LoRA merge":
        result_type = "LoRA merge"
    elif raw_result_type == "LoKR/LyCORIS merge":
        result_type = "LoKR merge"
    else:
        result_type = raw_result_type
    merge_chance = float(result.get("lora_likelihood_score", 0.0))
    changed = int(result.get("changed_tensors", 0))
    total = int(result.get("base_tensors", 0))
    changed_target = int(result.get("changed_anima_merge_target_tensors", 0))
    total_target = int(result.get("anima_merge_target_tensors", 0))
    changed_protected = int(result.get("changed_anima_protected_tensors", 0))
    total_protected = int(result.get("anima_protected_tensors", 0))
    stats = [
        f"- Changed tensors: {changed:,} / {total:,}",
        f"- Adapter-target changes: {changed_target:,} / {total_target:,}",
        f"- Protected-layer changes: {changed_protected:,} / {total_protected:,}",
        f"- LLM adapter drift: {result.get('llm_adapter_changed_ratio', 0.0):.3f}",
        f"- Final/embed/bias drift: {result.get('anchor_changed_ratio', 0.0):.3f}",
        f"- Base mismatch: {result.get('base_mismatch_score', 0.0):.1f}/100",
        f"- Model merge score: {result.get('model_merge_score', 0.0):.1f}/100",
        f"- Embedded adapter signal: {result.get('embedded_adapter_score', 0.0):.1f}/100",
    ]
    return "\n".join([
        "[DiffCheck]",
        "",
        f"Result: {result_type}",
        f"Merge likelihood: {merge_chance:.1f}%",
        "",
        "Stats:",
        "",
        *stats,
        "",
        "Note: using this as a Training base or stacking other LoRAs on top may cause LoRAs to behave differently, require different strengths, or not work as intended.",
        "",
        "Suggestion: this should be labeled as a merge/merged model rather than trained/fine-tuned to avoid confusion.",
        "",
        "- (If you disagree with this please let me know)",
    ])


# ─── Scoring ──────────────────────────────────────────────────────────────────

def build_lora_score(lora_stats, totals):
    changed = lora_stats["changed_tensors"]
    base_total = lora_stats["base_tensors"]
    changed_percent = (changed / max(base_total, 1)) * 100.0
    generic_target_ratio = lora_stats["changed_lora_target_tensors"] / max(changed, 1)
    anima_target_ratio = lora_stats["changed_anima_merge_target_tensors"] / max(changed, 1)
    protected_ratio = lora_stats["changed_anima_protected_tensors"] / max(changed, 1)
    non_lora_ratio = max(lora_stats["changed_non_lora_area_tensors"] / max(changed, 1), protected_ratio)
    target_ratio = anima_target_ratio if lora_stats.get("anima_merge_target_tensors", 0) else generic_target_ratio
    rank_ratios = lora_stats["rank_ratio_95_values"]
    top8_values = lora_stats["top8_energy_values"]
    decay_values = lora_stats.get("spectral_decay_values", [])
    artifact_values = lora_stats.get("merge_artifact_scores", [])
    avg_rank_ratio = sum(rank_ratios) / max(len(rank_ratios), 1) if rank_ratios else 1.0
    avg_top8 = sum(top8_values) / max(len(top8_values), 1) if top8_values else 0.0
    avg_decay = sum(decay_values) / max(len(decay_values), 1) if decay_values else 1.0
    avg_artifact = sum(artifact_values) / max(len(artifact_values), 1) if artifact_values else 0.0
    low_rank_ratio = lora_stats["low_rank_like_tensors"] / max(lora_stats["svd_tested_tensors"], 1)
    total_abs_diff = totals.get("total_abs_diff", 0.0)
    changed_elements = totals.get("changed_elements", 1)
    base_elements = totals.get("base_elements", 1)
    avg_delta_per_el = total_abs_diff / max(changed_elements, 1)
    same_tensor_ratio = totals.get("same_tensors", 0) / max(totals.get("base_tensors", 1), 1)
    changed_element_percent = (changed_elements / max(base_elements, 1)) * 100.0
    zero_norm_changes = lora_stats["changed_anima_protected_tensors"] == 0 and changed > 0
    llm_adapter_changed_ratio = lora_stats.get("changed_llm_adapter_tensors", 0) / max(lora_stats.get("llm_adapter_tensors", 0), 1)
    llm_adapter_changed_element_ratio = lora_stats.get("changed_llm_adapter_elements", 0) / max(lora_stats.get("llm_adapter_elements", 0), 1)
    anchor_changed_ratio = lora_stats.get("changed_anchor_tensors", 0) / max(lora_stats.get("anchor_tensors", 0), 1)
    anchor_changed_element_ratio = lora_stats.get("changed_anchor_elements", 0) / max(lora_stats.get("anchor_elements", 0), 1)

    if changed == 0 or changed_elements == 0:
        return {
            "lora_likelihood_score": 0.0,
            "lora_likelihood": lora_likelihood_label(0.0),
            "likely_update_type": "No tensor changes detected",
            "classic_lora_score": 0.0,
            "lokr_lycoris_score": 0.0,
            "merge_scope_score": 0.0,
            "full_tune_score": 0.0,
            "model_merge_score": 0.0,
            "base_mismatch_score": 0.0,
            "changed_percent": changed_percent,
            "changed_element_percent": changed_element_percent,
            "changed_lora_target_ratio": target_ratio,
            "changed_non_lora_area_ratio": non_lora_ratio,
            "changed_anima_merge_target_ratio": anima_target_ratio,
            "changed_anima_protected_ratio": protected_ratio,
            "low_rank_like_ratio": low_rank_ratio,
            "avg_rank_ratio_95": avg_rank_ratio,
            "avg_top8_energy": avg_top8,
            "avg_spectral_decay": avg_decay,
            "avg_merge_artifact_score": avg_artifact,
            "embedded_adapter_score": 0.0,
            "avg_delta_per_element": avg_delta_per_el,
            "same_tensor_ratio": same_tensor_ratio,
            "zero_norm_embed_changes": False,
            "llm_adapter_changed_ratio": llm_adapter_changed_ratio,
            "llm_adapter_changed_element_ratio": llm_adapter_changed_element_ratio,
            "anchor_changed_ratio": anchor_changed_ratio,
            "anchor_changed_element_ratio": anchor_changed_element_ratio,
            "flags": ["No tensors changed; merge likely failed or files are identical"],
        }

    base_mismatch = 0.0
    if changed_percent >= 85:
        base_mismatch += 30
    elif changed_percent >= 78:
        base_mismatch += 15
    if changed_element_percent >= 82:
        base_mismatch += 25
    elif changed_element_percent >= 78:
        base_mismatch += 10
    if same_tensor_ratio <= 0.15:
        base_mismatch += 25
    elif same_tensor_ratio <= 0.25:
        base_mismatch += 10
    if protected_ratio >= 0.20:
        base_mismatch += 20
    elif protected_ratio >= 0.10:
        base_mismatch += 10
    if llm_adapter_changed_ratio >= 0.60:
        base_mismatch += 25
    elif llm_adapter_changed_ratio >= 0.35:
        base_mismatch += 12
    if same_tensor_ratio <= 0.15 and anchor_changed_ratio >= 0.70:
        base_mismatch += 15
    elif same_tensor_ratio <= 0.20 and anchor_changed_ratio >= 0.40:
        base_mismatch += 8
    if avg_delta_per_el >= 4.0e-4:
        base_mismatch += 10
    base_mismatch = max(0.0, min(100.0, base_mismatch))

    embedded_adapter = 0.0
    if target_ratio >= 0.85:
        embedded_adapter += 20
    elif target_ratio >= 0.55:
        embedded_adapter += 12
    elif target_ratio >= 0.35:
        embedded_adapter += 6
    if low_rank_ratio >= 0.75:
        embedded_adapter += 25
    elif low_rank_ratio >= 0.45:
        embedded_adapter += 15
    elif low_rank_ratio >= 0.30:
        embedded_adapter += 8
    if avg_rank_ratio <= 0.08:
        embedded_adapter += 20
    elif avg_rank_ratio <= 0.18:
        embedded_adapter += 12
    elif avg_rank_ratio <= 0.35:
        embedded_adapter += 6
    if avg_top8 >= 0.78:
        embedded_adapter += 15
    elif avg_top8 >= 0.60:
        embedded_adapter += 8
    elif avg_top8 >= 0.42:
        embedded_adapter += 4
    if avg_artifact >= 70:
        embedded_adapter += 10
    elif avg_artifact >= 45:
        embedded_adapter += 6
    if protected_ratio <= 0.10:
        embedded_adapter += 10
    elif protected_ratio >= 0.45 and base_mismatch >= 75:
        embedded_adapter = min(embedded_adapter, 55.0)
    embedded_adapter = max(0.0, min(100.0, embedded_adapter))

    merge_scope = 0.0
    if anima_target_ratio >= 0.98:
        merge_scope += 35
    elif anima_target_ratio >= 0.90:
        merge_scope += 25
    if protected_ratio <= 0.01:
        merge_scope += 25
    elif protected_ratio <= 0.05:
        merge_scope += 10
    if same_tensor_ratio >= 0.55:
        merge_scope += 20
    elif same_tensor_ratio >= 0.45:
        merge_scope += 10
    if 30 <= changed_percent <= 55:
        merge_scope += 15
    elif 20 <= changed_percent <= 70:
        merge_scope += 8
    if zero_norm_changes:
        merge_scope += 5
    if base_mismatch >= 75:
        merge_scope = min(merge_scope, 20.0)
    merge_scope = max(0.0, min(100.0, merge_scope))

    model_merge = 0.0
    if 55 <= changed_percent <= 70:
        model_merge += 25
    elif 45 <= changed_percent < 55 or 70 < changed_percent <= 82:
        model_merge += 15
    if target_ratio >= 0.90:
        model_merge += 25
    elif target_ratio >= 0.75:
        model_merge += 15
    if 0.02 <= protected_ratio <= 0.12:
        model_merge += 20
    elif protected_ratio <= 0.20:
        model_merge += 8
    if 0.32 <= same_tensor_ratio <= 0.48:
        model_merge += 15
    elif 0.25 <= same_tensor_ratio < 0.32 or 0.48 < same_tensor_ratio <= 0.55:
        model_merge += 8
    if 0.08 <= avg_rank_ratio <= 0.24:
        model_merge += 10
    elif avg_rank_ratio <= 0.35:
        model_merge += 5
    if low_rank_ratio >= 0.45:
        model_merge += 8
    if avg_delta_per_el >= 2.5e-4:
        model_merge += 7
    if changed_element_percent >= 65 and avg_decay <= 0.35:
        model_merge += 20
    elif changed_element_percent >= 55 and avg_decay <= 0.42:
        model_merge += 10
    if low_rank_ratio >= 0.30 and avg_top8 >= 0.40:
        model_merge += 10
    if llm_adapter_changed_ratio == 0.0 and anchor_changed_ratio <= 0.05 and changed_element_percent >= 55:
        model_merge += 10
    if changed_element_percent >= 75 and low_rank_ratio >= 0.30 and avg_rank_ratio <= 0.32:
        model_merge += 10
    if changed_element_percent <= 35 and low_rank_ratio <= 0.15 and avg_top8 <= 0.35:
        model_merge = min(model_merge, 30.0)
    if base_mismatch >= 75:
        model_merge = min(model_merge, 35.0)
    if merge_scope >= 90 and protected_ratio <= 0.01:
        model_merge = min(model_merge, 25.0)
    model_merge = max(0.0, min(100.0, model_merge))

    classic = 0.0
    if merge_scope >= 80:
        classic += 35
    elif merge_scope >= 60:
        classic += 20
    if low_rank_ratio >= 0.80:
        classic += 30
    elif low_rank_ratio >= 0.50:
        classic += 15
    if avg_rank_ratio <= 0.05:
        classic += 20
    elif avg_rank_ratio <= 0.12:
        classic += 10
    if avg_top8 >= 0.80:
        classic += 15
    elif avg_top8 >= 0.60:
        classic += 8
    if base_mismatch >= 75:
        classic = min(classic, 20.0)
    classic = max(0.0, min(100.0, classic))

    lokr = 0.0
    if merge_scope >= 80:
        lokr += 45
    elif merge_scope >= 60:
        lokr += 25
    if low_rank_ratio <= 0.20:
        lokr += 20
    elif low_rank_ratio <= 0.45:
        lokr += 8
    if avg_rank_ratio >= 0.25:
        lokr += 20
    elif avg_rank_ratio >= 0.16:
        lokr += 8
    if avg_top8 <= 0.40:
        lokr += 15
    elif avg_top8 <= 0.60:
        lokr += 6
    if base_mismatch >= 75:
        lokr = min(lokr, 20.0)
    if merge_scope < 60:
        classic = min(classic, 40.0)
        lokr = min(lokr, 40.0)
    lokr = max(0.0, min(100.0, lokr))

    full = 0.0
    if protected_ratio >= 0.30:
        full += 40
    elif protected_ratio >= 0.12:
        full += 25
    if same_tensor_ratio <= 0.35:
        full += 25
    elif same_tensor_ratio <= 0.45:
        full += 10
    if changed_percent >= 65:
        full += 20
    elif changed_percent >= 50:
        full += 10
    if not zero_norm_changes:
        full += 15
    if base_mismatch >= 75 or merge_scope >= 80:
        full = min(full, 30.0)
    full = max(0.0, min(100.0, full))

    adapter_score = max(classic, lokr)
    detection_score = max(adapter_score, model_merge)
    low_change_finetune = (
        base_mismatch < 50
        and changed_element_percent <= 35
        and low_rank_ratio <= 0.15
        and avg_top8 <= 0.35
        and avg_rank_ratio >= 0.40
    )
    if base_mismatch >= 75:
        if embedded_adapter >= 30:
            adapter_score = max(adapter_score, min(max(embedded_adapter, 35.0), 55.0))
            detection_score = max(detection_score, adapter_score)
        if embedded_adapter >= 30:
            likely_type = "Wrong base/version, possible adapter merge"
        else:
            likely_type = "Likely different Anima base/version mismatch"
    elif classic >= 80 and classic >= lokr:
        likely_type = "Classic LoRA merge"
    elif lokr >= 80 and lokr > classic:
        likely_type = "LoKR/LyCORIS merge"
    elif model_merge >= 70 and model_merge >= adapter_score:
        likely_type = "Structured model merge / hybrid adapter merge"
    elif model_merge >= 60 and changed_element_percent >= 55 and (low_rank_ratio >= 0.30 or avg_rank_ratio <= 0.30 or avg_top8 >= 0.40):
        likely_type = "Structured model merge / hybrid adapter merge"
    elif merge_scope >= 80 and adapter_score >= 60:
        likely_type = "Adapter merge, rank type ambiguous"
    elif low_change_finetune and full >= 55:
        likely_type = "Fine-tune or non-adapter edit"
    elif full >= 65 and full >= adapter_score:
        likely_type = "Fine-tune or non-adapter edit"
    elif adapter_score >= 45:
        likely_type = "Weak adapter-merge signal"
    else:
        likely_type = "Inconclusive"

    flags = []
    if base_mismatch >= 75:
        flags.append(f"Likely wrong base/version: {changed_percent:.1f}% tensors changed, {changed_element_percent:.1f}% elements changed, {same_tensor_ratio*100:.1f}% tensors identical")
    if base_mismatch >= 75 and embedded_adapter >= 30:
        flags.append(f"Adapter-like rank signal remains under base mismatch: embedded score {embedded_adapter:.1f}/100")
    if merge_scope >= 80:
        flags.append("Only Anima mergeable block weights changed")
    if model_merge >= 70:
        flags.append(f"Structured model-merge pattern: target-heavy changes with protected drift {protected_ratio:.3f} and same tensor ratio {same_tensor_ratio:.3f}")
    if llm_adapter_changed_ratio >= 0.35:
        flags.append(f"LLM adapter drift: {llm_adapter_changed_ratio:.3f} of tensors changed")
    if anchor_changed_ratio >= 0.40:
        flags.append(f"Final/embed/bias drift: {anchor_changed_ratio:.3f} of tensors changed")
    if protected_ratio >= 0.10:
        flags.append("Protected/non-mergeable tensors changed")
    if classic >= 80:
        flags.append(f"Classic LoRA rank pattern: low-rank tensors {low_rank_ratio:.3f}, avg rank ratio {avg_rank_ratio:.4f}, top-8 energy {avg_top8:.3f}")
    if lokr >= 80 and lokr > classic:
        flags.append(f"LoKR/LyCORIS rank pattern: low-rank tensors {low_rank_ratio:.3f}, avg rank ratio {avg_rank_ratio:.4f}, top-8 energy {avg_top8:.3f}")

    return {
        "lora_likelihood_score": detection_score,
        "lora_likelihood": lora_likelihood_label(detection_score),
        "likely_update_type": likely_type,
        "classic_lora_score": classic,
        "lokr_lycoris_score": lokr,
        "merge_scope_score": merge_scope,
        "full_tune_score": full,
        "model_merge_score": model_merge,
        "base_mismatch_score": base_mismatch,
        "changed_percent": changed_percent,
        "changed_element_percent": changed_element_percent,
        "changed_lora_target_ratio": target_ratio,
        "changed_non_lora_area_ratio": non_lora_ratio,
        "changed_anima_merge_target_ratio": anima_target_ratio,
        "changed_anima_protected_ratio": protected_ratio,
        "low_rank_like_ratio": low_rank_ratio,
        "avg_rank_ratio_95": avg_rank_ratio,
        "avg_top8_energy": avg_top8,
        "avg_spectral_decay": avg_decay,
        "avg_merge_artifact_score": avg_artifact,
        "embedded_adapter_score": embedded_adapter,
        "avg_delta_per_element": avg_delta_per_el,
        "same_tensor_ratio": same_tensor_ratio,
        "zero_norm_embed_changes": zero_norm_changes,
        "llm_adapter_changed_ratio": llm_adapter_changed_ratio,
        "llm_adapter_changed_element_ratio": llm_adapter_changed_element_ratio,
        "anchor_changed_ratio": anchor_changed_ratio,
        "anchor_changed_element_ratio": anchor_changed_element_ratio,
        "flags": flags,
    }


# ─── Main comparison logic ────────────────────────────────────────────────────

def compare_one_model(base_reader, tune_reader, label, atol, rtol, depth,
                      compute_diff, detect_lora, progress_cb):
    base_keys = base_reader.keys
    tune_keys  = tune_reader.keys

    shared  = sorted(base_keys & tune_keys)
    missing = sorted(base_keys - tune_keys)
    extra   = sorted(tune_keys - base_keys)

    rows, lora_rows = [], []

    layers = defaultdict(lambda: {
        "tensors": 0, "same_tensors": 0, "different_tensors": 0,
        "shape_mismatch_tensors": 0, "missing_tensors": 0, "extra_tensors": 0,
        "elements": 0, "changed_elements": 0,
        "total_abs_diff": 0.0, "total_abs_diff_scientific": "0.00000000e+00",
    })

    blocks = defaultdict(lambda: {
        "label": "", "color": "#4b5563",
        "total": 0, "changed": 0, "same": 0, "missing": 0,
        "elements": 0, "changed_elements": 0, "total_abs_diff": 0.0,
    })

    totals = {
        "label": label,
        "base_tensors": len(base_keys), "tune_tensors": len(tune_keys),
        "shared_tensors": len(shared), "same_tensors": 0, "different_tensors": 0,
        "shape_mismatch_tensors": 0, "missing_tensors": len(missing),
        "extra_tensors": len(extra), "base_elements": 0, "changed_elements": 0,
        "changed_percent_of_base_elements": 0.0,
        "total_abs_diff": 0.0, "total_abs_diff_scientific": "0.00000000e+00",
    }

    lora_stats = {
        "model": label, "base_tensors": len(base_keys),
        "changed_tensors": 0,
        "changed_lora_target_tensors": 0, "changed_non_lora_area_tensors": 0,
        "anima_merge_target_tensors": 0, "changed_anima_merge_target_tensors": 0,
        "anima_protected_tensors": 0, "changed_anima_protected_tensors": 0,
        "llm_adapter_tensors": 0, "changed_llm_adapter_tensors": 0,
        "llm_adapter_elements": 0, "changed_llm_adapter_elements": 0,
        "anchor_tensors": 0, "changed_anchor_tensors": 0,
        "anchor_elements": 0, "changed_anchor_elements": 0,
        "svd_tested_tensors": 0, "low_rank_like_tensors": 0,
        "rank_ratio_95_values": [], "top8_energy_values": [],
        "spectral_decay_values": [], "merge_artifact_scores": [],
        "projected_tensor_count": 0,
    }

    total_work = len(shared) + len(missing) + len(extra)
    done = 0

    for key in shared:
        base_tensor = base_reader.get_tensor(key)
        tune_tensor = tune_reader.get_tensor(key)
        stats       = compare_tensors(base_tensor, tune_tensor, atol, rtol, compute_diff)
        lyr         = layer_name(key, depth)
        blk_label, blk_color = get_block_group(key)
        is_anima_target = looks_like_anima_merge_target(key)
        is_anima_protected = looks_like_anima_protected_area(key)
        is_llm_adapter = looks_like_llm_adapter_area(key)
        is_anchor = looks_like_anchor_area(key)
        if detect_lora:
            if is_anima_target:
                lora_stats["anima_merge_target_tensors"] += 1
            if is_anima_protected:
                lora_stats["anima_protected_tensors"] += 1
            if is_llm_adapter:
                lora_stats["llm_adapter_tensors"] += 1
            if is_anchor:
                lora_stats["anchor_tensors"] += 1

        row = {"model": label, "key": key, "layer": lyr, "block_group": blk_label, **stats}

        blocks[blk_label]["label"]    = blk_label
        blocks[blk_label]["color"]    = blk_color
        blocks[blk_label]["total"]   += 1
        blocks[blk_label]["elements"] += int(stats["numel"] or 0)
        totals["base_elements"]       += int(stats["numel"] or 0)
        if detect_lora:
            numel = int(stats["numel"] or 0)
            if is_llm_adapter:
                lora_stats["llm_adapter_elements"] += numel
            if is_anchor:
                lora_stats["anchor_elements"] += numel

        if stats["status"] == "same":
            totals["same_tensors"]         += 1
            layers[lyr]["same_tensors"]    += 1
            blocks[blk_label]["same"]      += 1

        elif stats["status"] == "shape_mismatch":
            totals["shape_mismatch_tensors"]      += 1
            layers[lyr]["shape_mismatch_tensors"] += 1
            blocks[blk_label]["changed"]          += 1

        else:
            totals["different_tensors"]       += 1
            layers[lyr]["different_tensors"]  += 1
            blocks[blk_label]["changed"]      += 1

            changed  = int(stats["different_elements"] or 0)
            abs_diff = float(stats["total_abs_diff"] or 0.0)

            totals["changed_elements"]             += changed
            layers[lyr]["changed_elements"]        += changed
            blocks[blk_label]["changed_elements"]  += changed
            blocks[blk_label]["total_abs_diff"]    += abs_diff
            totals["total_abs_diff"]               += abs_diff
            layers[lyr]["total_abs_diff"]          += abs_diff

            if detect_lora:
                lora_stats["changed_tensors"] += 1
                is_lora_target   = looks_like_lora_target(key)
                is_non_lora_area = looks_like_non_lora_area(key)
                if is_lora_target:   lora_stats["changed_lora_target_tensors"]  += 1
                if is_non_lora_area: lora_stats["changed_non_lora_area_tensors"] += 1
                if is_anima_target: lora_stats["changed_anima_merge_target_tensors"] += 1
                if is_anima_protected: lora_stats["changed_anima_protected_tensors"] += 1
                if is_llm_adapter:
                    lora_stats["changed_llm_adapter_tensors"] += 1
                    lora_stats["changed_llm_adapter_elements"] += changed
                if is_anchor:
                    lora_stats["changed_anchor_tensors"] += 1
                    lora_stats["changed_anchor_elements"] += changed

                rank_stats = lora_delta_rank_stats(base_tensor, tune_tensor)
                if rank_stats is not None:
                    lora_stats["svd_tested_tensors"] += 1
                    lora_stats["rank_ratio_95_values"].append(rank_stats["rank_ratio_95"])
                    lora_stats["top8_energy_values"].append(rank_stats["top8_energy"])
                    lora_stats["spectral_decay_values"].append(rank_stats["spectral_decay_ratio"])
                    lora_stats["merge_artifact_scores"].append(rank_stats["merge_artifact_score"])
                    if rank_stats["was_projected"]:
                        lora_stats["projected_tensor_count"] += 1

                    if rank_stats["rank_ratio_95"] <= 0.20 or rank_stats["top8_energy"] >= 0.90:
                        lora_stats["low_rank_like_tensors"] += 1

                    lora_rows.append({
                        "model": label, "key": key, "layer": lyr,
                        "is_lora_target_name": is_lora_target,
                        "is_non_lora_area_name": is_non_lora_area,
                        "is_anima_merge_target": is_anima_target,
                        "is_anima_protected_area": is_anima_protected,
                        **rank_stats,
                    })

        layers[lyr]["tensors"]  += 1
        layers[lyr]["elements"] += int(stats["numel"] or 0)
        rows.append(row)

        done += 1
        if done % 10 == 0 or done == total_work:
            progress_cb(label, done, total_work, key)
        del base_tensor, tune_tensor

    for key in missing:
        lyr = layer_name(key, depth)
        layers[lyr]["missing_tensors"] += 1
        blk_label, _ = get_block_group(key)
        blocks[blk_label]["missing"] += 1
        rows.append({
            "model": label, "key": key, "layer": lyr,
            "status": "missing_in_tune", "same": False,
            "shape_base": None, "shape_other": None,
            "dtype_base": None, "dtype_other": None, "numel": None,
            "different_elements": None, "different_percent": None,
            "max_abs_diff": None, "mean_abs_diff": None,
            "total_abs_diff": None, "total_abs_diff_scientific": None,
        })
        done += 1

    for key in extra:
        lyr = layer_name(key, depth)
        layers[lyr]["extra_tensors"] += 1
        rows.append({
            "model": label, "key": key, "layer": lyr,
            "status": "extra_in_tune", "same": False,
            "shape_base": None, "shape_other": None,
            "dtype_base": None, "dtype_other": None, "numel": None,
            "different_elements": None, "different_percent": None,
            "max_abs_diff": None, "mean_abs_diff": None,
            "total_abs_diff": None, "total_abs_diff_scientific": None,
        })
        done += 1

    totals["changed_percent_of_base_elements"] = (
        totals["changed_elements"] / max(totals["base_elements"], 1)
    ) * 100.0
    totals["total_abs_diff_scientific"] = f"{totals['total_abs_diff']:.8e}"

    layer_rows = []
    for lyr, data in sorted(layers.items()):
        data["changed_percent"]           = (data["changed_elements"] / max(data["elements"], 1)) * 100.0
        data["total_abs_diff_scientific"] = f"{data['total_abs_diff']:.8e}"
        layer_rows.append({"model": label, "layer": lyr, **data})

    lora_summary = None
    if detect_lora:
        score_data = build_lora_score(lora_stats, totals)
        lora_summary = {
            "model":                       label,
            "lora_likelihood":             score_data["lora_likelihood"],
            "lora_likelihood_score":       score_data["lora_likelihood_score"],
            "likely_update_type":          score_data["likely_update_type"],
            "classic_lora_score":          score_data["classic_lora_score"],
            "lokr_lycoris_score":          score_data["lokr_lycoris_score"],
            "merge_scope_score":           score_data["merge_scope_score"],
            "full_tune_score":             score_data["full_tune_score"],
            "model_merge_score":           score_data["model_merge_score"],
            "base_mismatch_score":         score_data["base_mismatch_score"],
            "base_tensors":                lora_stats["base_tensors"],
            "changed_tensors":             lora_stats["changed_tensors"],
            "changed_tensor_percent":      score_data["changed_percent"],
            "changed_element_percent":     score_data["changed_element_percent"],
            "changed_lora_target_ratio":   score_data["changed_lora_target_ratio"],
            "changed_non_lora_area_ratio": score_data["changed_non_lora_area_ratio"],
            "changed_anima_merge_target_ratio": score_data["changed_anima_merge_target_ratio"],
            "changed_anima_protected_ratio": score_data["changed_anima_protected_ratio"],
            "anima_merge_target_tensors": lora_stats["anima_merge_target_tensors"],
            "changed_anima_merge_target_tensors": lora_stats["changed_anima_merge_target_tensors"],
            "anima_protected_tensors": lora_stats["anima_protected_tensors"],
            "changed_anima_protected_tensors": lora_stats["changed_anima_protected_tensors"],
            "llm_adapter_tensors": lora_stats["llm_adapter_tensors"],
            "changed_llm_adapter_tensors": lora_stats["changed_llm_adapter_tensors"],
            "llm_adapter_changed_ratio": score_data["llm_adapter_changed_ratio"],
            "llm_adapter_changed_element_ratio": score_data["llm_adapter_changed_element_ratio"],
            "anchor_tensors": lora_stats["anchor_tensors"],
            "changed_anchor_tensors": lora_stats["changed_anchor_tensors"],
            "anchor_changed_ratio": score_data["anchor_changed_ratio"],
            "anchor_changed_element_ratio": score_data["anchor_changed_element_ratio"],
            "svd_tested_tensors":          lora_stats["svd_tested_tensors"],
            "projected_tensor_count":      lora_stats["projected_tensor_count"],
            "low_rank_like_tensors":       lora_stats["low_rank_like_tensors"],
            "low_rank_like_ratio":         lora_stats["low_rank_like_tensors"] / max(lora_stats["svd_tested_tensors"], 1),
            "avg_rank_ratio_95":           score_data["avg_rank_ratio_95"],
            "avg_top8_energy":             score_data["avg_top8_energy"],
            "avg_spectral_decay":          score_data["avg_spectral_decay"],
            "avg_merge_artifact_score":    score_data["avg_merge_artifact_score"],
            "embedded_adapter_score":      score_data["embedded_adapter_score"],
            "avg_delta_per_element":       score_data["avg_delta_per_element"],
            "same_tensor_ratio":           score_data["same_tensor_ratio"],
            "zero_norm_embed_changes":     score_data["zero_norm_embed_changes"],
            "flags":                       " | ".join(score_data["flags"]),
            "note": "Scope-calibrated from known Anima base, fine-tune, and LoKR samples. Base mismatch warnings mean retry against the other known base before judging adapter merges.",
        }

    block_data = [dict(v, label=k) for k, v in sorted(blocks.items())]
    return rows, layer_rows, totals, lora_rows, lora_summary, block_data


# ─── CSV / JSON output ────────────────────────────────────────────────────────

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_compare(base_path, tune1_path, tune2_path, output_dir, prefixes,
                atol, rtol, depth, compute_diff, detect_lora, progress_cb):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_reader = CheckpointReader(base_path, prefixes)
    tunes = []
    if tune1_path: tunes.append(("fine_tune_1", tune1_path))
    if tune2_path: tunes.append(("fine_tune_2", tune2_path))
    if not tunes:
        raise ValueError("Select at least one fine-tune model.")

    all_tensor_rows, all_layer_rows, all_lora_rows = [], [], []
    summaries, lora_summaries, all_block_data = [], [], []

    for label, path in tunes:
        tune_reader = CheckpointReader(path, prefixes)
        tensor_rows, layer_rows, summary, lora_rows, lora_summary, block_data = compare_one_model(
            base_reader, tune_reader, label, atol, rtol, depth,
            compute_diff, detect_lora, progress_cb,
        )
        all_tensor_rows.extend(tensor_rows)
        all_layer_rows.extend(layer_rows)
        all_lora_rows.extend(lora_rows)
        summaries.append(summary)
        all_block_data.append({"model": label, "blocks": block_data})
        if lora_summary is not None:
            lora_summaries.append(lora_summary)

    write_csv(output_dir / "tensor_comparison.csv",  all_tensor_rows)
    write_csv(output_dir / "layer_summary.csv",       all_layer_rows)
    if detect_lora:
        write_csv(output_dir / "lora_detection_rank_details.csv", all_lora_rows)
        write_csv(output_dir / "lora_detection_summary.csv",      lora_summaries)

    report = {
        "base_model":  str(base_path),
        "fine_tune_1": str(tune1_path) if tune1_path else None,
        "fine_tune_2": str(tune2_path) if tune2_path else None,
        "settings": {
            "prefixes_stripped": prefixes,
            "atol": atol, "rtol": rtol, "layer_depth": depth,
            "compute_diff": compute_diff, "detect_lora": detect_lora,
        },
        "summary":             summaries,
        "lora_detection":      lora_summaries if detect_lora else None,
        "block_visualization": all_block_data,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# ─── Block Diagram Canvas ─────────────────────────────────────────────────────

class BlockDiagram(tk.Canvas):
    PAD_X   = 20
    PAD_Y   = 20
    BAR_H   = 34
    BAR_GAP = 8
    LABEL_W = 170
    METER_W = 380

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=DARK_BG2, highlightthickness=0, **kw)
        self._models_data = []

    def update_data(self, models_data):
        self._models_data = models_data
        self._draw()

    def _draw(self):
        self.delete("all")
        if not self._models_data:
            self.create_text(300, 120,
                text="Run a comparison to see the block diagram",
                fill=DARK_FG_MUT, font=("Courier New", 11))
            return

        y = self.PAD_Y
        for model_label, block_data in self._models_data:
            sorted_blocks = sorted(
                block_data,
                key=lambda b: (b["changed"] / max(b["total"], 1)),
                reverse=True,
            )

            self.create_text(self.PAD_X, y + 10,
                text=f"▸ {model_label}",
                fill=DARK_ACCENT2, font=("Courier New", 11, "bold"), anchor="w")
            y += 26

            self.create_text(self.PAD_X,                          y, text="Block",          fill=DARK_FG_MUT, font=("Courier New", 9), anchor="w")
            self.create_text(self.PAD_X + self.LABEL_W,           y, text="Change %",       fill=DARK_FG_MUT, font=("Courier New", 9), anchor="w")
            self.create_text(self.PAD_X + self.LABEL_W + self.METER_W + 8, y, text="Changed / Total", fill=DARK_FG_MUT, font=("Courier New", 9), anchor="w")
            y += 16

            for blk in sorted_blocks:
                total   = blk["total"]
                changed = blk["changed"]
                label   = blk.get("label", "?")
                color   = blk.get("color", "#4b5563")
                pct     = (changed / max(total, 1)) * 100.0

                bx1 = self.PAD_X + self.LABEL_W
                bx2 = bx1 + self.METER_W
                by1, by2 = y, y + self.BAR_H
                self.create_rectangle(bx1, by1, bx2, by2, fill=DARK_BG3, outline=DARK_BORDER, width=1)

                fill_w = int(self.METER_W * (pct / 100.0))
                if fill_w > 0:
                    self.create_rectangle(bx1, by1, bx1 + fill_w, by2, fill=color, outline="")

                self.create_text(self.PAD_X, y + self.BAR_H // 2,
                    text=label, fill=DARK_FG, font=("Courier New", 10), anchor="w")

                pct_txt = f"{pct:.1f}%"
                tx = bx1 + fill_w + 6 if fill_w <= self.METER_W - 50 else bx1 + 6
                self.create_text(tx, y + self.BAR_H // 2,
                    text=pct_txt, fill=DARK_FG, font=("Courier New", 10, "bold"), anchor="w")

                self.create_text(bx2 + 10, y + self.BAR_H // 2,
                    text=f"{changed:,} / {total:,}",
                    fill=DARK_FG_MUT, font=("Courier New", 9), anchor="w")

                y += self.BAR_H + self.BAR_GAP

            y += 20

        self.configure(scrollregion=(0, 0, 680, y + self.PAD_Y))


# ─── Tkinter GUI ──────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Layer Compare - DiT LoRA Detector v3")
        self.geometry("1100x800")
        self.minsize(920, 660)
        self.configure(bg=DARK_BG)

        self.msg_queue = queue.Queue()
        self.worker    = None

        self.base_var   = tk.StringVar()
        self.tune1_var  = tk.StringVar()
        self.tune2_var  = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "model_compare_report"))

        self.prefix_var       = tk.StringVar(
            value="pipe.dit.,dit.,model.diffusion_model.,diffusion_model.,model.,net."
        )
        self.atol_var         = tk.StringVar(value="0")
        self.rtol_var         = tk.StringVar(value="0")
        self.depth_var        = tk.StringVar(value="3")
        self.compute_diff_var = tk.BooleanVar(value=True)
        self.detect_lora_var  = tk.BooleanVar(value=True)

        self._apply_dark_style()
        self._build_ui()
        self.after(100, self._poll_queue)

    # ── Style ──────────────────────────────────────────────────────────────────

    def _apply_dark_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".",
            background=DARK_BG, foreground=DARK_FG,
            troughcolor=DARK_BG2, selectbackground=DARK_ACCENT,
            selectforeground=DARK_FG, fieldbackground=DARK_BG3,
            insertcolor=DARK_FG, borderwidth=1, relief="flat",
        )
        style.configure("TFrame",     background=DARK_BG)
        style.configure("TLabel",     background=DARK_BG,  foreground=DARK_FG, font=("Courier New", 10))
        style.configure("TEntry",     fieldbackground=DARK_BG3, foreground=DARK_FG, insertcolor=DARK_FG, borderwidth=1, relief="flat")
        style.configure("TButton",    background=DARK_BG3, foreground=DARK_FG, borderwidth=1, relief="flat", font=("Courier New", 10))
        style.map("TButton",
            background=[("active", DARK_ACCENT), ("pressed", "#5b21b6")],
            foreground=[("active", "#fff")])
        style.configure("TCheckbutton", background=DARK_BG, foreground=DARK_FG, font=("Courier New", 10))
        style.map("TCheckbutton",
            background=[("active", DARK_BG)],
            indicatorcolor=[("selected", DARK_ACCENT), ("!selected", DARK_BG3)])
        style.configure("TLabelframe",       background=DARK_BG, foreground=DARK_ACCENT2, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_ACCENT2, font=("Courier New", 10, "bold"))
        style.configure("Horizontal.TProgressbar", troughcolor=DARK_BG3, background=DARK_ACCENT, thickness=6)
        style.configure("TNotebook",     background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=DARK_BG3, foreground=DARK_FG_MUT, padding=[14, 6], font=("Courier New", 10))
        style.map("TNotebook.Tab",
            background=[("selected", DARK_BG), ("active", DARK_BG2)],
            foreground=[("selected", DARK_ACCENT2), ("active", DARK_FG)])

    # ── Layout ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        header = tk.Frame(self, bg=DARK_BG2, height=42)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)
        tk.Label(header, text="MODEL LAYER COMPARE  v3", bg=DARK_BG2, fg=DARK_ACCENT2,
                 font=("Courier New", 13, "bold")).pack(side="left", padx=16, pady=8)
        if not HAS_SAFETENSORS:
            tk.Label(header, text="⚠ safetensors not installed", bg=DARK_BG2, fg=DARK_YELLOW,
                     font=("Courier New", 9)).pack(side="right", padx=12)

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content, width=430)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._build_config(left)

        tk.Frame(content, bg=DARK_BORDER, width=1).pack(side="left", fill="y")

        right = ttk.Frame(content)
        right.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        log_frame  = ttk.Frame(self.notebook)
        diag_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame,  text="  Log / Summary  ")
        self.notebook.add(diag_frame, text="  Block Diagram  ")

        self._build_log(log_frame)
        self._build_diagram(diag_frame)
        self._build_statusbar()

    def _build_config(self, parent):
        inner = ttk.Frame(parent)
        inner.pack(fill="both", expand=True, padx=14, pady=10)

        files_lf = ttk.LabelFrame(inner, text=" Model Files ")
        files_lf.pack(fill="x", pady=(0, 10))
        self._file_row(files_lf, "Base model *",  self.base_var,   0, True)
        self._file_row(files_lf, "Fine-tune 1",   self.tune1_var,  1, True)
        self._file_row(files_lf, "Fine-tune 2",   self.tune2_var,  2, True)
        self._file_row(files_lf, "Output folder", self.output_var, 3, False)

        sett_lf = ttk.LabelFrame(inner, text=" Comparison Settings ")
        sett_lf.pack(fill="x", pady=(0, 10))
        ttk.Label(sett_lf, text="Prefixes to strip (comma-sep)").grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2))
        ttk.Entry(sett_lf, textvariable=self.prefix_var).grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        ttk.Label(sett_lf, text="atol").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(sett_lf, textvariable=self.atol_var, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(sett_lf, text="rtol").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(sett_lf, textvariable=self.rtol_var, width=10).grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(sett_lf, text="Layer depth").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(sett_lf, textvariable=self.depth_var, width=10).grid(row=4, column=1, sticky="w", padx=8, pady=4)
        sett_lf.columnconfigure(1, weight=1)

        opt_lf = ttk.LabelFrame(inner, text=" Detection Options ")
        opt_lf.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(opt_lf, text="Compute max/mean/total abs diff", variable=self.compute_diff_var).pack(anchor="w", padx=8, pady=4)
        ttk.Checkbutton(opt_lf, text="Detect LoRA / LyCORIS / low-rank delta (SVD)", variable=self.detect_lora_var).pack(anchor="w", padx=8, pady=(0, 8))

        run_frame = ttk.Frame(inner)
        run_frame.pack(fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(run_frame, text="▶  Compare Models", command=self._start_compare)
        self.run_btn.pack(fill="x", pady=(0, 8))
        self.progress = ttk.Progressbar(run_frame, mode="determinate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x")

    def _build_log(self, parent):
        self.log = tk.Text(parent, wrap="word", bg=DARK_BG, fg=DARK_FG,
            insertbackground=DARK_FG, selectbackground=DARK_ACCENT,
            font=("Courier New", 10), relief="flat", borderwidth=0, padx=12, pady=10)
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("header", foreground=DARK_ACCENT2, font=("Courier New", 10, "bold"))
        self.log.tag_configure("good",   foreground=DARK_GREEN)
        self.log.tag_configure("warn",   foreground=DARK_YELLOW)
        self.log.tag_configure("bad",    foreground=DARK_RED)
        self.log.tag_configure("info",   foreground=DARK_CYAN)
        self.log.tag_configure("flag",   foreground=DARK_ORANGE)
        self.log.tag_configure("muted",  foreground=DARK_FG_MUT)

    def _build_diagram(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        vsb = tk.Scrollbar(frame, orient="vertical", bg=DARK_BG2, troughcolor=DARK_BG3)
        vsb.pack(side="right", fill="y")
        self.diagram = BlockDiagram(frame, yscrollcommand=vsb.set)
        self.diagram.pack(fill="both", expand=True)
        vsb.configure(command=self.diagram.yview)
        self.diagram.bind("<MouseWheel>", lambda e: self.diagram.yview_scroll(int(-1*(e.delta/120)), "units"))
        self.diagram.bind("<Button-4>",   lambda e: self.diagram.yview_scroll(-1, "units"))
        self.diagram.bind("<Button-5>",   lambda e: self.diagram.yview_scroll(1,  "units"))
        self.diagram._draw()

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=DARK_BG2, height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready - load models and press Compare")
        tk.Label(bar, textvariable=self.status_var, bg=DARK_BG2, fg=DARK_FG_MUT,
                 font=("Courier New", 9)).pack(side="left", padx=10)
        self.tensor_count_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.tensor_count_var, bg=DARK_BG2, fg=DARK_FG_MUT,
                 font=("Courier New", 9)).pack(side="right", padx=10)

    # ── Helper widgets ─────────────────────────────────────────────────────────

    def _file_row(self, parent, label, var, row, file_mode):
        ttk.Label(parent, text=label).grid(row=row*2,   column=0, sticky="w", padx=8, pady=(8, 0))
        ttk.Entry(parent, textvariable=var).grid(row=row*2+1, column=0, sticky="ew", padx=8, pady=(2, 6))
        cmd = (lambda v=var: self._pick_file(v)) if file_mode else (lambda v=var: self._pick_folder(v))
        ttk.Button(parent, text="…", command=cmd, width=3).grid(row=row*2+1, column=1, padx=(0, 8), pady=(2, 6))
        parent.columnconfigure(0, weight=1)

    def _pick_file(self, var):
        path = filedialog.askopenfilename(title="Select model checkpoint",
            filetypes=[("Model checkpoints", MODEL_FILE_PATTERN), ("All files", "*.*")])
        if path: var.set(path)

    def _pick_folder(self, var):
        path = filedialog.askdirectory(title="Select output folder")
        if path: var.set(path)

    # ── Log helpers ────────────────────────────────────────────────────────────

    def _log(self, text, tag=None):
        self.log.configure(state="normal")
        if tag:
            self.log.insert("end", text + "\n", tag)
        else:
            self.log.insert("end", text + "\n")
        self.log.see("end")

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self):
        base_path  = clean_path(self.base_var.get())
        tune1_path = clean_path(self.tune1_var.get())
        tune2_path = clean_path(self.tune2_var.get())
        output_dir = clean_path(self.output_var.get())

        if not base_path:                                    raise ValueError("Select a base model file.")
        if not Path(base_path).exists():                     raise ValueError(f"Base model not found: {base_path}")
        if Path(base_path).suffix.lower() not in MODEL_EXTS: raise ValueError(f"Unsupported extension: {base_path}")

        selected = []
        for name, path in (("Fine-tune 1", tune1_path), ("Fine-tune 2", tune2_path)):
            if not path: continue
            if not Path(path).exists():                     raise ValueError(f"{name} not found: {path}")
            if Path(path).suffix.lower() not in MODEL_EXTS: raise ValueError(f"Unsupported extension for {name}: {path}")
            selected.append(path)

        if not selected:   raise ValueError("Select at least one fine-tune model.")
        if not output_dir: raise ValueError("Select an output folder.")

        return {
            "base_path":    base_path,
            "tune1_path":   tune1_path or None,
            "tune2_path":   tune2_path or None,
            "output_dir":   output_dir,
            "prefixes":     parse_prefixes(self.prefix_var.get()),
            "atol":         float(self.atol_var.get()),
            "rtol":         float(self.rtol_var.get()),
            "depth":        max(1, int(self.depth_var.get())),
            "compute_diff": bool(self.compute_diff_var.get()),
            "detect_lora":  bool(self.detect_lora_var.get()),
        }

    # ── Worker ────────────────────────────────────────────────────────────────

    def _start_compare(self):
        if self.worker and self.worker.is_alive():
            return

        try:
            args = self._validate()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.run_btn.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)
        self.log.delete("1.0", "end")
        self._log("MODEL LAYER COMPARE  v3", "header")
        self._log("─" * 60, "muted")
        self._log(f"Base:   {args['base_path']}", "muted")
        if args['tune1_path']: self._log(f"Tune 1: {args['tune1_path']}", "muted")
        if args['tune2_path']: self._log(f"Tune 2: {args['tune2_path']}", "muted")
        self._log("─" * 60, "muted")
        if args["detect_lora"]:
            self._log("LoRA detection: scope + base mismatch guard + optional SVD signals", "info")
        self.status_var.set("Comparing…")

        def progress_cb(label, done, total, key):
            self.msg_queue.put(("progress", label, done, total, key))

        def worker():
            try:
                report = run_compare(progress_cb=progress_cb, **args)
                self.msg_queue.put(("done", report, args["output_dir"]))
            except Exception:
                self.msg_queue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _poll_queue(self):
        try:
            while True:
                item = self.msg_queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, label, done, total, key = item
                    pct = (done / max(total, 1)) * 100.0
                    self.progress.configure(value=pct)
                    self.status_var.set(f"{label}  {done:,}/{total:,} tensors")
                    self.tensor_count_var.set(key[-60:] if len(key) > 60 else key)

                elif kind == "done":
                    _, report, out_dir = item
                    self.run_btn.configure(state="normal")
                    self.progress.configure(value=100)
                    self.status_var.set("Done")
                    self.tensor_count_var.set("")

                    self._log("")
                    self._log("✓ Comparison complete", "good")
                    self._log(f"Output: {out_dir}", "muted")
                    self._log("")

                    for s in report["summary"]:
                        self._log(f"┌─ {s['label']}", "header")
                        self._log(f"│  same tensors    : {s['same_tensors']:,}")
                        self._log(f"│  different       : {s['different_tensors']:,}",
                                  "bad" if s["different_tensors"] > 0 else "good")
                        self._log(f"│  shape mismatch  : {s['shape_mismatch_tensors']:,}",
                                  "warn" if s["shape_mismatch_tensors"] else None)
                        self._log(f"│  missing/extra   : {s['missing_tensors']:,} / {s['extra_tensors']:,}")
                        self._log(f"│  changed elements: {s['changed_elements']:,}  ({s['changed_percent_of_base_elements']:.6f}%)")
                        self._log(f"└  total |Δ|       : {s['total_abs_diff_scientific']}")
                        self._log("")

                    lora = report.get("lora_detection") or []
                    if lora:
                        self._log("─── LoRA / LyCORIS Detection ─────────────────────", "header")
                        for r in lora:
                            score = r["lora_likelihood_score"]
                            score_tag = "bad" if score >= 75 else ("warn" if score >= 50 else "muted")

                            self._log(f"┌─ {r['model']}  →  {r['likely_update_type']}", "header")
                            self._log(f"│  likelihood          : {r['lora_likelihood']}  ({score:.1f}/100)", score_tag)
                            self._log(f"│  classic LoRA        : {r['classic_lora_score']:.1f}")
                            self._log(f"│  LoKR/LyCORIS        : {r['lokr_lycoris_score']:.1f}")
                            self._log(f"│  merge scope         : {r.get('merge_scope_score', 0.0):.1f}")
                            self._log(f"│  full fine-tune      : {r['full_tune_score']:.1f}")
                            self._log(f"│  model merge         : {r.get('model_merge_score', 0.0):.1f}")
                            self._log(f"│  base mismatch       : {r.get('base_mismatch_score', 0.0):.1f}",
                                      "warn" if r.get("base_mismatch_score", 0.0) >= 75 else None)
                            self._log(f"│  embedded adapter    : {r.get('embedded_adapter_score', 0.0):.1f}")
                            self._log(f"│  ── tensor stats ──────────────────────────")
                            self._log(f"│  changed tensors     : {r['changed_tensor_percent']:.1f}%")
                            self._log(f"│  changed elements    : {r.get('changed_element_percent', 0.0):.1f}%")
                            self._log(f"│  same tensor ratio   : {r['same_tensor_ratio']:.3f}",
                                      "warn" if r["same_tensor_ratio"] > 0.55 else None)
                            self._log(f"│  target ratio        : {r['changed_lora_target_ratio']:.3f}")
                            self._log(f"│  Anima merge targets : {r.get('changed_anima_merge_target_ratio', 0.0):.3f}")
                            self._log(f"│  protected drift     : {r.get('changed_anima_protected_ratio', 0.0):.3f}",
                                      "warn" if r.get("changed_anima_protected_ratio", 0.0) >= 0.10 else None)
                            self._log(f"│  LLM adapter drift   : {r.get('llm_adapter_changed_ratio', 0.0):.3f}",
                                      "warn" if r.get("llm_adapter_changed_ratio", 0.0) >= 0.35 else None)
                            self._log(f"│  final/embed/bias    : {r.get('anchor_changed_ratio', 0.0):.3f}",
                                      "warn" if r.get("anchor_changed_ratio", 0.0) >= 0.40 else None)
                            self._log(f"│  zero norm changes   : {r['zero_norm_embed_changes']}",
                                      "warn" if r["zero_norm_embed_changes"] else None)
                            self._log(f"│  ── SVD signals ───────────────────────────")
                            self._log(f"│  low-rank ratio      : {r['low_rank_like_ratio']:.3f}",
                                      "warn" if r["low_rank_like_ratio"] > 0.4 else None)
                            self._log(f"│  avg rank ratio 95   : {r['avg_rank_ratio_95']:.4f}",
                                      "warn" if r["avg_rank_ratio_95"] < 0.25 else None)
                            self._log(f"│  avg top-8 energy    : {r['avg_top8_energy']:.4f}")
                            self._log(f"│  avg spectral decay  : {r['avg_spectral_decay']:.4f}")
                            self._log(f"│  merge artifact score: {r['avg_merge_artifact_score']:.1f}/100",
                                      "warn" if r["avg_merge_artifact_score"] > 60 else None)
                            self._log(f"│  ── delta magnitude ───────────────────────")
                            adpe = r["avg_delta_per_element"]
                            adpe_tag = "warn" if adpe >= 1.8e-4 else None
                            self._log(f"│  avg Δ per element   : {adpe:.4e}", adpe_tag)
                            svd_proj = r.get("projected_tensor_count", 0)
                            if svd_proj:
                                self._log(f"│  (SVD via projection : {svd_proj} large tensors)", "muted")

                            # Flags — print each on its own line in orange
                            flags_str = r.get("flags", "")
                            if flags_str:
                                self._log(f"│  ── flags ─────────────────────────────────")
                                for flag in flags_str.split(" | "):
                                    if flag.strip():
                                        self._log(f"│  {flag}", "flag")

                            self._log("└")
                            self._log("")
                            self._log("─── DiffCheck Copy Block ─────────────────────", "header")
                            for line in diffcheck_comment_block(r).splitlines():
                                self._log(line)
                            self._log("")

                    blk_viz = report.get("block_visualization") or []
                    if blk_viz:
                        self.diagram.update_data([(m["model"], m["blocks"]) for m in blk_viz])
                        self.notebook.select(1)

                    messagebox.showinfo("Done", f"Comparison complete.\n\nReports saved to:\n{out_dir}")

                elif kind == "error":
                    _, err = item
                    self.run_btn.configure(state="normal")
                    self.status_var.set("Error")
                    self._log("")
                    self._log("✗ Error", "bad")
                    self._log(err, "bad")
                    messagebox.showerror("Error", err[-3000:])

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    App().mainloop()
