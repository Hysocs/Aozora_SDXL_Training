import gc
import fnmatch
import json
import math
import os
import random
import re
import time
import traceback
import uuid
from collections import defaultdict, deque
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from train import (
    AsyncReporter,
    BucketBatchSampler,
    CustomCurveLRScheduler,
    TitanAdamW,
    TrainingDiagnostics,
    cache_float_dtype,
    cache_float_dtype_name,
    cache_options_match_for_image_cache,
    caption_signature_for_image,
    CAPTION_JSON_PRIMARY_TYPE,
    caption_types_for_cache,
    caption_source_type,
    consume_force_save_flag,
    create_optimizer,
    fix_alpha_channel,
    get_json_caption_weights,
    get_max_bucket_resolution_for_config,
    get_multi_bucket_resolutions,
    get_text_conditioning_scale_range,
    json_caption_mode_enabled,
    make_bucket_variant_metadata,
    normalize_cache_options_for_image_cache,
    null_conditioning_cache_needed,
    json_caption_cache_suffix,
    remove_cache_files_for_stem,
    selected_caption_variant_path,
    set_seed,
    smart_resize,
    strip_json_caption_suffix,
    text_cache_float_dtype,
    text_cache_float_dtype_name,
    text_cache_paths_for_index_item,
    validate_and_assign_resolution,
    vae_cache_float_dtype,
    vae_cache_float_dtype_name,
    BUCKET_LAYOUT_VERSION,
)

AnimaImagePipeline = None
ModelConfig = None


def normalize_anima_config(config):
    if getattr(config, "DIT_VAE_PATH", ""):
        config.VAE_PATH = config.DIT_VAE_PATH
    if getattr(config, "RESUME_TRAINING", False):
        if not getattr(config, "ANIMA_RESUME_MODEL_PATH", "") and getattr(config, "RESUME_MODEL_PATH", ""):
            config.ANIMA_RESUME_MODEL_PATH = config.RESUME_MODEL_PATH
        if not getattr(config, "ANIMA_RESUME_STATE_PATH", "") and getattr(config, "RESUME_STATE_PATH", ""):
            config.ANIMA_RESUME_STATE_PATH = config.RESUME_STATE_PATH
    exclude_targets = getattr(config, "DIT_EXCLUDE_TARGETS", [])
    if isinstance(exclude_targets, str):
        config.DIT_EXCLUDE_TARGETS = [item.strip() for item in exclude_targets.split(",") if item.strip()]
    elif isinstance(exclude_targets, list):
        config.DIT_EXCLUDE_TARGETS = [item for item in exclude_targets if item]
    else:
        config.DIT_EXCLUDE_TARGETS = []


def anima_cache_folder_name(config):
    return getattr(config, "ANIMA_CACHE_FOLDER_NAME", ".precomputed_anima_dit_cache")


def anima_cache_float_dtype(config):
    return cache_float_dtype(config)


def anima_cache_float_dtype_name(config):
    return cache_float_dtype_name(config)


def anima_text_cache_float_dtype(config):
    return text_cache_float_dtype(config)


def anima_text_cache_float_dtype_name(config):
    return text_cache_float_dtype_name(config)


def anima_vae_cache_float_dtype(config):
    return vae_cache_float_dtype(config)


def anima_vae_cache_float_dtype_name(config):
    return vae_cache_float_dtype_name(config)


def anima_caption_types_for_cache(config, json_caption_mode=None):
    if json_caption_mode is None:
        json_caption_mode = json_caption_mode_enabled(config)
    if not json_caption_mode:
        return ("txt",)

    weights = get_json_caption_weights(config)
    active_types = tuple(
        caption_type
        for caption_type in caption_types_for_cache(True)
        if int(weights.get(caption_type, 0) or 0) > 0
    )
    return active_types or (CAPTION_JSON_PRIMARY_TYPE,)


def anima_caption_variant_index(variant_paths):
    return {
        caption_type: {"te_path": str(path)}
        for caption_type, path in variant_paths.items()
    }


def anima_dataset_roots(config):
    return [
        Path(ds["path"])
        for ds in getattr(config, "INSTANCE_DATASETS", [])
        if ds.get("path")
    ]


ANIMA_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_anima_image_paths(root):
    return [
        p
        for ext in ANIMA_IMAGE_EXTENSIONS
        for p in root.rglob(f"*{ext}")
    ]


def anima_cache_stem_for_image(root, image_path, cache_suffix=""):
    return str(image_path.relative_to(root).with_suffix("")).replace(os.sep, "_") + cache_suffix


def anima_cache_base_stem(cache_path):
    stem = Path(cache_path).stem
    stem = re.sub(r"_(te|lat)$", "", stem)
    stem = strip_json_caption_suffix(stem)
    return re.sub(r"_mb\d+$", "", stem)


def anima_text_cache_paths_for_index_item(item):
    return text_cache_paths_for_index_item(
        item,
        path_key="te_path",
        primary_key="te_path",
        fallback_key="cache_path",
    )


def anima_cache_paths_for_index_item(item):
    paths = list(anima_text_cache_paths_for_index_item(item))
    if item.get("lat_path"):
        paths.append(item["lat_path"])
    return paths


def anima_lat_path_for_item(item):
    if item.get("lat_path"):
        return item["lat_path"]
    return item.get("cache_path")


def remove_anima_cache_file(cache_path):
    try:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cache_path.unlink()
    except OSError as e:
        print(f"WARNING: Could not remove stale Anima cache file {cache_path}: {e}")


def anima_expected_cache_paths(root, cache_dir, meta, active_caption_types, json_caption_mode):
    safe_filename = anima_cache_stem_for_image(root, meta["ip"], meta.get("cache_suffix", ""))
    text_paths = {
        caption_type: cache_dir / f"{safe_filename}{json_caption_cache_suffix(caption_type, json_caption_mode)}_te.pt"
        for caption_type in active_caption_types
    }
    return text_paths, cache_dir / f"{safe_filename}_lat.pt"


def anima_metadata_matches(payload, meta, check_caption=True):
    if not isinstance(payload, dict):
        return False
    if check_caption and payload.get("caption_signature") != meta.get("caption_signature"):
        return False
    relative_path = str(meta["ip"].relative_to(meta["root"]))
    return (
        payload.get("relative_path") == relative_path
        and tuple(payload.get("original_size", ())) == tuple(meta["original_size"])
        and tuple(payload.get("scaled_size", ())) == tuple(meta["scaled_size"])
        and tuple(payload.get("target_size", ())) == tuple(meta["target_resolution"])
        and tuple(payload.get("crop_coords", (0, 0))) == tuple(meta.get("crop_coords", (0, 0)))
        and int(payload.get("bucket_variant_index", 0) or 0) == int(meta.get("bucket_variant_index", 0) or 0)
    )


def anima_text_cache_valid(path, meta, caption_type, caption, text_cache_dtype):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        prompt_emb = payload.get("prompt_emb")
        t5xxl_ids = payload.get("t5xxl_ids")
        return (
            prompt_emb is not None
            and t5xxl_ids is not None
            and prompt_emb.dtype == text_cache_dtype
            and payload.get("caption_type") == caption_type
            and payload.get("caption") == caption
            and anima_metadata_matches(payload, meta, check_caption=True)
        )
    except Exception:
        return False


def anima_lat_cache_valid(path, meta, vae_cache_dtype):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latents = payload.get("latents") if isinstance(payload, dict) else None
        return (
            latents is not None
            and latents.dtype == vae_cache_dtype
            and not torch.isnan(latents).any()
            and not torch.isinf(latents).any()
            and anima_metadata_matches(payload, meta, check_caption=False)
        )
    except Exception:
        return False


def get_anima_cache_options(config):
    return {
        "version": 5,
        "bucket_layout": BUCKET_LAYOUT_VERSION,
        "text_cache_float_dtype": anima_text_cache_float_dtype_name(config),
        "vae_cache_float_dtype": anima_vae_cache_float_dtype_name(config),
        "caption_source_type": caption_source_type(config),
        "caption_json_types": list(anima_caption_types_for_cache(config)),
        "max_bucket_resolution": get_max_bucket_resolution_for_config(config),
        "should_upscale": bool(getattr(config, "SHOULD_UPSCALE", False)),
        "multi_bucket_enabled": bool(getattr(config, "MULTI_BUCKET_ENABLED", False)),
        "multi_bucket_extra_buckets": int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0),
        "vae_caching_tiled": bool(getattr(config, "VAE_CACHING_TILED", True)),
        "vae_caching_tile_size": list(getattr(config, "VAE_CACHING_TILE_SIZE", [96, 96])),
        "vae_caching_tile_stride": list(getattr(config, "VAE_CACHING_TILE_STRIDE", [72, 72])),
    }


def normalize_anima_cache_options(cache_options, expected_options=None):
    if not isinstance(cache_options, dict):
        return cache_options

    normalized = normalize_cache_options_for_image_cache(cache_options)
    for key in (
        "dit_path",
        "anima_base_dit_path",
        "vae_path",
        "text_encoder_path",
        "tokenizer_path",
        "tokenizer_t5xxl_path",
    ):
        normalized.pop(key, None)
    return normalized


def anima_cache_options_match(cached_options, expected_options):
    return (
        normalize_anima_cache_options(cached_options, expected_options)
        == normalize_anima_cache_options(expected_options)
    )


def anima_cache_rebuild_needed_for_root(config, root, expected_options=None, cache_name=None):
    expected_options = expected_options or get_anima_cache_options(config)
    cache_name = cache_name or anima_cache_folder_name(config)
    cache_dir = root / cache_name
    index_path = cache_dir / "dataset_index.json"
    if not cache_dir.exists() or not index_path.exists():
        print(
            "INFO: Anima cache rebuild needed for "
            f"{root}: cache_dir_exists={cache_dir.exists()}, "
            f"index_exists={index_path.exists()}."
        )
        return True
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        image_paths = collect_anima_image_paths(root)
        current_cache_stems = {anima_cache_stem_for_image(root, p) for p in image_paths}
        cached_options = index_data.get("cache_options")
        if not anima_cache_options_match(cached_options, expected_options):
            print(f"INFO: Anima cache rebuild needed for {root}: cache options changed.")
            cached_norm = normalize_anima_cache_options(cached_options, expected_options)
            expected_norm = normalize_anima_cache_options(expected_options)
            if isinstance(cached_norm, dict) and isinstance(expected_norm, dict):
                all_keys = sorted(set(cached_norm) | set(expected_norm))
                for key in all_keys:
                    cached_value = cached_norm.get(key, "<missing>")
                    expected_value = expected_norm.get(key, "<missing>")
                    if cached_value != expected_value:
                        print(f"  - {key}: cached={cached_value!r}, expected={expected_value!r}")
            else:
                print(f"  cached_options={cached_options!r}")
                print(f"  expected_options={expected_options!r}")
            return True
        files = index_data.get("files", [])
        if not files:
            print(f"INFO: Anima cache rebuild needed for {root}: dataset_index has no files.")
            return True
        indexed_base_stems = {
            anima_cache_base_stem(cache_path)
            for item in files
            for cache_path in anima_cache_paths_for_index_item(item)
        }
        if not current_cache_stems.issubset(indexed_base_stems):
            print(f"INFO: Anima cache rebuild needed for {root}: new image(s) are not cached.")
            return True
        if any(stem not in current_cache_stems for stem in indexed_base_stems):
            print(f"INFO: Anima cache rebuild needed for {root}: cached image(s) were removed from the dataset.")
            return True
        for item in files:
            cache_paths = anima_cache_paths_for_index_item(item)
            if not cache_paths:
                print(f"INFO: Anima cache rebuild needed for {root}: missing cached path in dataset index.")
                return True
            for cache_path in cache_paths:
                cache_path = Path(cache_path)
                if not cache_path.exists():
                    print(f"INFO: Anima cache rebuild needed for {root}: missing cached item {cache_path}.")
                    return True
            relative_path = item.get("relative_path")
            if relative_path:
                try:
                    if caption_signature_for_image(root / relative_path, caption_source_type(config)) != item.get("caption_signature"):
                        print(f"INFO: Anima cache rebuild needed for {root}: caption changed for {relative_path}.")
                        return True
                except Exception as e:
                    print(f"INFO: Anima cache rebuild needed for {root}: caption invalid for {relative_path}: {e}")
                    return True
    except Exception:
        print(f"INFO: Anima cache rebuild needed for {root}: failed to read/validate {index_path}.")
        traceback.print_exc()
        return True
    return False


def anima_roots_needing_cache_rebuild(config):
    roots = anima_dataset_roots(config)
    if bool(getattr(config, "REBUILD_CACHE", False)):
        print("INFO: Rebuilding Anima DiT cache because REBUILD_CACHE=True.")
        return roots

    expected_options = get_anima_cache_options(config)
    cache_name = anima_cache_folder_name(config)
    return [
        root
        for root in roots
        if anima_cache_rebuild_needed_for_root(config, root, expected_options, cache_name)
    ]


def check_if_anima_caching_needed(config):
    return bool(anima_roots_needing_cache_rebuild(config))


def find_anima_base_dit_path(config):
    candidates = []
    configured = getattr(config, "ANIMA_BASE_DIT_PATH", "") or getattr(config, "DIT_BASE_PATH", "")
    if configured:
        candidates.append(Path(configured))

    dit_path = Path(getattr(config, "DIT_PATH", "") or "")
    if dit_path:
        candidates.extend([
            dit_path.with_name("anima-preview.safetensors"),
            dit_path.with_name("anima_preview.safetensors"),
        ])

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def default_huggingface_cache_root():
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return str(Path(hf_home).expanduser())

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return str(Path(xdg_cache_home).expanduser() / "huggingface")

    return str(Path.home() / ".cache" / "huggingface")


def local_tokenizer_path(model_id, origin_file_pattern):
    origin = (origin_file_pattern or "./").rstrip("/")
    relative_origin = "" if origin in ("", ".") else origin
    candidates = []

    model_path = Path(model_id).expanduser()
    if model_path.exists():
        candidates.append(model_path / relative_origin if relative_origin else model_path)

    for base in (Path("models"), Path(default_huggingface_cache_root())):
        candidates.append(base / model_id / relative_origin if relative_origin else base / model_id)

    required_files = ("tokenizer.json", "spiece.model", "tokenizer.model", "vocab.json")
    for candidate in candidates:
        if candidate.exists() and any((candidate / name).exists() for name in required_files):
            return str(candidate)
    return None


def make_anima_tokenizer_config(model_id, origin_file_pattern):
    local_path = local_tokenizer_path(model_id, origin_file_pattern)
    if local_path:
        return ModelConfig(path=local_path, skip_download=True)

    return ModelConfig(
        model_id=model_id,
        origin_file_pattern=origin_file_pattern,
        download_source="huggingface",
        local_model_path=default_huggingface_cache_root(),
    )


def make_anima_t5xxl_tokenizer_config(model_id, origin_file_pattern):
    local_path = local_tokenizer_path(model_id, origin_file_pattern)
    if local_path:
        return ModelConfig(path=local_path, skip_download=True)

    if model_id == "stabilityai/stable-diffusion-3.5-large":
        print(
            "INFO: T5XXL tokenizer from stabilityai/stable-diffusion-3.5-large is gated on Hugging Face. "
            "Using google/t5-v1_1-xxl tokenizer instead."
        )
        return make_anima_tokenizer_config("google/t5-v1_1-xxl", "./")

    return make_anima_tokenizer_config(model_id, origin_file_pattern)


def ensure_anima_diffsynth_available():
    global AnimaImagePipeline, ModelConfig
    if AnimaImagePipeline is not None and ModelConfig is not None:
        return
    try:
        from diffsynth.pipelines.anima_image import AnimaImagePipeline as _AnimaImagePipeline
        from diffsynth.pipelines.anima_image import ModelConfig as _ModelConfig
    except Exception as e:
        raise RuntimeError(
            "Anima DiT training requires diffsynth. Install/enable diffsynth, or use the SDXL architecture mode."
        ) from e
    AnimaImagePipeline = _AnimaImagePipeline
    ModelConfig = _ModelConfig


def normalize_anima_dit_state_key(key):
    for prefix in ("pipe.dit.", "dit.", "model.diffusion_model.", "diffusion_model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return key.replace("net.", "")


def detect_anima_dit_key_prefix(path):
    prefixes = ("pipe.dit.", "dit.", "model.diffusion_model.", "diffusion_model.", "model.", "net.")
    counts = {prefix: 0 for prefix in prefixes}
    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            total += 1
            for prefix in prefixes:
                if key.startswith(prefix):
                    counts[prefix] += 1
                    break
    if total <= 0:
        return ""
    best_prefix, best_count = max(counts.items(), key=lambda item: item[1])
    return best_prefix if best_count / total >= 0.8 else ""


def load_anima_dit_weights(dit, path):
    path = str(path)
    target_shapes = {name: tuple(param.shape) for name, param in dit.state_dict().items()}
    remapped = {}
    skipped_shape = []
    skipped_unknown = 0

    with safe_open(path, framework="pt", device="cpu") as f:
        for source_key in f.keys():
            target_key = normalize_anima_dit_state_key(source_key)
            if target_key not in target_shapes:
                skipped_unknown += 1
                continue

            tensor = f.get_tensor(source_key)
            if tuple(tensor.shape) != target_shapes[target_key]:
                skipped_shape.append((source_key, tuple(tensor.shape), target_key, target_shapes[target_key]))
                continue
            remapped[target_key] = tensor

    if not remapped:
        raise RuntimeError(
            f"No matching DiT tensors were found in {path}. "
            "Expected raw Anima DiT keys or keys prefixed with pipe.dit./dit./model."
        )

    incompatible = dit.load_state_dict(remapped, strict=False, assign=True)
    print(
        f"INFO: Loaded fine-tuned Anima DiT weights from {Path(path).name}: "
        f"{len(remapped):,}/{len(target_shapes):,} tensors matched."
    )
    if skipped_unknown:
        print(f"INFO: Ignored {skipped_unknown:,} non-DiT or unknown tensor(s).")
    if skipped_shape:
        print(f"WARNING: Ignored {len(skipped_shape):,} tensor(s) with shape mismatches.")
        for source_key, source_shape, target_key, target_shape in skipped_shape[:5]:
            print(f"  {source_key} {source_shape} -> {target_key} expected {target_shape}")
        if len(skipped_shape) > 5:
            print(f"  ... and {len(skipped_shape) - 5} more")
    if incompatible.missing_keys:
        print(f"WARNING: {len(incompatible.missing_keys):,} DiT key(s) were not supplied by the fine-tune.")
    detected_prefix = detect_anima_dit_key_prefix(path)
    del remapped
    gc.collect()
    return detected_prefix


def load_anima_pipe(config, device):
    ensure_anima_diffsynth_available()

    original_dit_path = config.DIT_PATH
    model_configs = [
        ModelConfig(path=config.DIT_PATH),
        ModelConfig(path=config.TEXT_ENCODER_PATH),
        ModelConfig(path=config.VAE_PATH),
    ]

    tokenizer_model_id = "Qwen/Qwen3-0.6B"
    tokenizer_origin = "./"
    if isinstance(config.TOKENIZER_PATH, str) and ":" in config.TOKENIZER_PATH:
        tokenizer_model_id, tokenizer_origin = config.TOKENIZER_PATH.split(":", 1)
    tokenizer_config = make_anima_tokenizer_config(tokenizer_model_id, tokenizer_origin)

    t5xxl_model_id = "stabilityai/stable-diffusion-3.5-large"
    t5xxl_origin = "tokenizer_3/"
    if isinstance(config.TOKENIZER_T5XXL_PATH, str) and ":" in config.TOKENIZER_T5XXL_PATH:
        t5xxl_model_id, t5xxl_origin = config.TOKENIZER_T5XXL_PATH.split(":", 1)
    tokenizer_t5xxl_config = make_anima_t5xxl_tokenizer_config(t5xxl_model_id, t5xxl_origin)

    loaded_template_path = config.DIT_PATH
    try:
        pipe = AnimaImagePipeline.from_pretrained(
            torch_dtype=config.compute_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            tokenizer_t5xxl_config=tokenizer_t5xxl_config,
        )
    except ValueError as e:
        message = str(e)
        if "Cannot detect the model type" not in message:
            raise
        base_dit_path = find_anima_base_dit_path(config)
        if not base_dit_path:
            raise RuntimeError(
                "DiffSynth could not auto-detect the selected DiT file, and no base template was found. "
                "Set 'Base DiT Template' in the GUI to the original anima-preview.safetensors, then keep "
                "your fine-tune in 'DiT Model'."
            ) from e

        print(
            "WARNING: DiffSynth could not auto-detect the selected DiT file. "
            f"Using {Path(base_dit_path).name} as the architecture template, then overlaying fine-tune weights."
        )
        model_configs[0] = ModelConfig(path=base_dit_path)
        loaded_template_path = base_dit_path
        pipe = AnimaImagePipeline.from_pretrained(
            torch_dtype=config.compute_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            tokenizer_t5xxl_config=tokenizer_t5xxl_config,
        )
        loaded_prefix = load_anima_dit_weights(pipe.dit, original_dit_path)
    else:
        loaded_prefix = detect_anima_dit_key_prefix(original_dit_path)

    for attr in ("dit", "vae", "text_encoder"):
        if hasattr(pipe, attr):
            getattr(pipe, attr).cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    config.DIT_PATH = original_dit_path
    config.ANIMA_LOADED_DIT_TEMPLATE_PATH = loaded_template_path
    if str(getattr(config, "ANIMA_DIT_SAVE_PREFIX", "auto")).lower() == "auto":
        config.ANIMA_DIT_SAVE_PREFIX = loaded_prefix
    return pipe


@torch.no_grad()
def encode_prompt_anima(pipe, caption, device):
    if isinstance(caption, str):
        caption = [caption]

    text_inputs = pipe.tokenizer(
        caption,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()
    prompt_embeds = pipe.text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-1]

    t5xxl_text_inputs = pipe.tokenizer_t5xxl(
        caption,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    return prompt_embeds.to(pipe.torch_dtype), t5xxl_text_inputs.input_ids.to(device)


@torch.no_grad()
def encode_image_anima(pipe, image, device, tiled=True, tile_size=(96, 96), tile_stride=(72, 72), dtype=None):
    dtype = dtype or pipe.torch_dtype
    image_tensor = pipe.preprocess_image(image).to(device="cpu", dtype=dtype)
    latents = pipe.vae.encode(
        image_tensor.unsqueeze(2),
        device=device,
        tiled=tiled,
        tile_size=tuple(tile_size),
        tile_stride=tuple(tile_stride),
    ).squeeze(2)
    return latents


def save_anima_null_conditioning_cache(cache_dir, null_prompt_emb, null_t5xxl_ids, text_cache_dtype):
    torch.save(
        {
            "prompt_emb": null_prompt_emb[0].to(dtype=text_cache_dtype).cpu(),
            "t5xxl_ids": null_t5xxl_ids[0].to(dtype=torch.long).cpu(),
        },
        cache_dir / "null_embeds.pt",
    )


def ensure_anima_null_conditioning_cache(config, pipe, device):
    if not null_conditioning_cache_needed(config):
        return

    cache_name = anima_cache_folder_name(config)
    missing_cache_dirs = []
    for root in anima_dataset_roots(config):
        cache_dir = root / cache_name
        if (cache_dir / "dataset_index.json").exists() and not (cache_dir / "null_embeds.pt").exists():
            missing_cache_dirs.append(cache_dir)

    if not missing_cache_dirs:
        return

    print(f"INFO: Creating Anima null conditioning cache for {len(missing_cache_dirs)} dataset(s).")
    if hasattr(pipe, "dit"):
        pipe.dit.cpu()
    pipe.vae.cpu()
    pipe.text_encoder.to(device=device, dtype=config.compute_dtype)
    pipe.text_encoder.eval()

    with torch.no_grad():
        null_prompt_emb, null_t5xxl_ids = encode_prompt_anima(pipe, "", device)
    text_cache_dtype = anima_text_cache_float_dtype(config)
    for cache_dir in missing_cache_dirs:
        save_anima_null_conditioning_cache(cache_dir, null_prompt_emb, null_t5xxl_ids, text_cache_dtype)

    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def precompute_and_cache_anima(config, pipe, device):
    roots_to_rebuild = anima_roots_needing_cache_rebuild(config)
    if not roots_to_rebuild:
        ensure_anima_null_conditioning_cache(config, pipe, device)
        print("\n" + "=" * 60 + "\nINFO: Anima DiT datasets already cached.\n" + "=" * 60 + "\n")
        return

    cache_name = anima_cache_folder_name(config)
    text_cache_dtype = anima_text_cache_float_dtype(config)
    vae_cache_dtype = anima_vae_cache_float_dtype(config)
    expected_options = get_anima_cache_options(config)
    caption_mode = caption_source_type(config)
    json_caption_mode = json_caption_mode_enabled(config)
    active_caption_types = anima_caption_types_for_cache(config, json_caption_mode)
    print(
        "INFO: Anima cache precision: "
        f"text={anima_text_cache_float_dtype_name(config)}, "
        f"vae={anima_vae_cache_float_dtype_name(config)}."
    )
    multi_bucket_extra = (
        max(0, int(getattr(config, "MULTI_BUCKET_EXTRA_BUCKETS", 0) or 0))
        if getattr(config, "MULTI_BUCKET_ENABLED", False)
        else 0
    )

    for root in roots_to_rebuild:
        if not root.exists():
            print(f"WARNING: Dataset folder does not exist: {root}")
            continue

        cache_dir = root / cache_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / "dataset_index.json"
        force_recaching = bool(getattr(config, "REBUILD_CACHE", False))
        if index_path.exists() and not force_recaching:
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    existing_index = json.load(f)
                if not anima_cache_options_match(existing_index.get("cache_options"), expected_options):
                    print(
                        f"INFO: Anima cache options changed for {root.name}; "
                        "reusing compatible cached files and filling only missing variants."
                    )
            except Exception:
                print(f"INFO: Anima cache index for {root.name} is unreadable; rebuilding index from cache files.")

        image_paths = collect_anima_image_paths(root)
        current_cache_stems = {anima_cache_stem_for_image(root, p) for p in image_paths}
        if not image_paths:
            stale_files = list(cache_dir.glob("*.pt"))
            if stale_files:
                print(f"INFO: Removing {len(stale_files)} stale Anima cache item(s) because {root.name} has no images.")
            for f in stale_files:
                remove_anima_cache_file(f)
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({"version": 5, "cache_options": expected_options, "files": []}, f)
            print(f"WARNING: No images found in {root}")
            continue

        if force_recaching:
            for f in cache_dir.glob("*.pt"):
                remove_anima_cache_file(f)
        else:
            stale_files = [
                f for f in cache_dir.glob("*.pt")
                if f.name != "null_embeds.pt" and anima_cache_base_stem(f) not in current_cache_stems
            ]
            if stale_files:
                print(f"INFO: Removing {len(stale_files)} stale Anima cache item(s) for deleted images in {root.name}.")
            for f in stale_files:
                remove_anima_cache_file(f)

        max_bucket_resolution = get_max_bucket_resolution_for_config(config)
        max_bucket_area = max_bucket_resolution * max_bucket_resolution
        print(f"INFO: Validating Anima DiT images and assigning buckets for {root.name} ({len(image_paths)} image(s))...")
        with Pool(processes=min(cpu_count(), 8)) as pool:
            results = list(tqdm(
                pool.imap(
                    validate_and_assign_resolution,
                    [(p, max_bucket_area, 64, config.SHOULD_UPSCALE, caption_mode) for p in image_paths],
                ),
                total=len(image_paths),
            ))

        expanded_results = []
        for m in (r for r in results if r):
            buckets = get_multi_bucket_resolutions(
                m["original_size"][0],
                m["original_size"][1],
                max_bucket_area,
                config.SHOULD_UPSCALE,
                multi_bucket_extra,
            )
            for variant_index, (target_w, target_h) in enumerate(buckets):
                variant_meta = make_bucket_variant_metadata(m, target_w, target_h, variant_index)
                variant_meta["root"] = root
                expanded_results.append(variant_meta)

        index_data = []
        text_jobs = []
        lat_jobs = []
        reused_text_count = 0
        reused_lat_count = 0
        expected_cache_files = set()
        for m in expanded_results:
            caption_variants = m.get("caption_variants") or {"txt": m["caption"]}
            text_paths, lat_path = anima_expected_cache_paths(root, cache_dir, m, active_caption_types, json_caption_mode)
            expected_cache_files.add(lat_path.resolve())
            variant_cache_paths = {}
            for caption_type in active_caption_types:
                te_path = text_paths[caption_type]
                expected_cache_files.add(te_path.resolve())
                variant_cache_paths[caption_type] = str(te_path)
                caption = caption_variants[caption_type]
                if te_path.exists() and anima_text_cache_valid(te_path, m, caption_type, caption, text_cache_dtype):
                    reused_text_count += 1
                else:
                    text_jobs.append((m, caption_type, caption, te_path))

            if lat_path.exists() and anima_lat_cache_valid(lat_path, m, vae_cache_dtype):
                reused_lat_count += 1
            else:
                lat_jobs.append((m, lat_path))

            primary_te_path = variant_cache_paths.get(CAPTION_JSON_PRIMARY_TYPE) or next(iter(variant_cache_paths.values()))
            item = {
                "te_path": primary_te_path,
                "lat_path": str(lat_path),
                "relative_path": str(m["ip"].relative_to(root)),
                "target_size": tuple(m["target_resolution"]),
                "original_size": m["original_size"],
                "scaled_size": m["scaled_size"],
                "crop_coords": m.get("crop_coords", (0, 0)),
                "bucket_variant_index": m.get("bucket_variant_index", 0),
                "caption_signature": m.get("caption_signature"),
            }
            if json_caption_mode:
                item["caption_variants"] = anima_caption_variant_index(variant_cache_paths)
            index_data.append(item)

        obsolete_files = [
            f for f in cache_dir.glob("*.pt")
            if f.name != "null_embeds.pt"
            and anima_cache_base_stem(f) in current_cache_stems
            and f.resolve() not in expected_cache_files
        ]
        if obsolete_files:
            print(f"INFO: Removing {len(obsolete_files)} obsolete Anima cache variant(s) for {root.name}.")
        for f in obsolete_files:
            remove_anima_cache_file(f)

        if not text_jobs and not lat_jobs:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump({"version": 5, "cache_options": expected_options, "files": index_data}, f)
            print(f"INFO: Anima DiT cache current for {root.name}: {len(index_data)} item(s).")
            continue

        print(
            f"INFO: Anima cache reuse for {root.name}: "
            f"{reused_text_count}/{len(expanded_results) * len(active_caption_types)} text item(s), "
            f"{reused_lat_count}/{len(expanded_results)} latent item(s)."
        )

        if hasattr(pipe, "dit"):
            pipe.dit.cpu()
        pipe.vae.cpu()
        pipe.text_encoder.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"INFO: Anima cache phase 1/2: text encoder for {root.name}")
        if text_jobs or null_conditioning_cache_needed(config):
            pipe.text_encoder.to(device=device, dtype=config.compute_dtype)
            pipe.text_encoder.eval()
            pipe.vae.cpu()

        if null_conditioning_cache_needed(config) and (text_jobs or not (cache_dir / "null_embeds.pt").exists()):
            with torch.no_grad():
                null_prompt_emb, null_t5xxl_ids = encode_prompt_anima(pipe, "", device)
        else:
            null_prompt_emb, null_t5xxl_ids = None, None

        if text_jobs:
            with torch.no_grad():
                with tqdm(total=len(text_jobs), desc=f"Caching Anima text {root.name}", unit="item") as pbar:
                    for m, caption_type, caption, te_path in text_jobs:
                        prompt_emb, t5xxl_ids = encode_prompt_anima(pipe, caption, device)
                        torch.save({
                            "prompt_emb": prompt_emb[0].to(dtype=text_cache_dtype).cpu(),
                            "t5xxl_ids": t5xxl_ids[0].to(dtype=torch.long).cpu(),
                            "caption_type": caption_type,
                            "caption": caption,
                            "caption_signature": m.get("caption_signature"),
                            "image_path": str(m["ip"]),
                            "relative_path": str(m["ip"].relative_to(root)),
                            "original_size": m["original_size"],
                            "scaled_size": m["scaled_size"],
                            "target_size": tuple(m["target_resolution"]),
                            "crop_coords": m.get("crop_coords", (0, 0)),
                            "bucket_variant_index": m.get("bucket_variant_index", 0),
                            "cache_options": expected_options,
                        }, te_path)
                        pbar.update(1)

        if null_prompt_emb is not None and null_t5xxl_ids is not None:
            save_anima_null_conditioning_cache(cache_dir, null_prompt_emb, null_t5xxl_ids, text_cache_dtype)

        pipe.text_encoder.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tile_size = tuple(getattr(config, "VAE_CACHING_TILE_SIZE", [96, 96]))
        tile_stride = tuple(getattr(config, "VAE_CACHING_TILE_STRIDE", [72, 72]))
        if lat_jobs:
            print(
                "INFO: Anima cache phase 2/2: VAE "
                f"(tiled={getattr(config, 'VAE_CACHING_TILED', True)}, "
                f"tile_size={tile_size}, tile_stride={tile_stride})"
            )
            pipe.vae.to(device=device, dtype=vae_cache_dtype)
            pipe.vae.eval()

            with torch.inference_mode():
                with tqdm(total=len(lat_jobs), desc=f"Caching Anima VAE {root.name}", unit="item") as pbar:
                    for m, lat_path in lat_jobs:
                        image_path = Path(m["ip"])
                        w, h = m["target_resolution"]
                        try:
                            with Image.open(image_path) as img:
                                image = smart_resize(fix_alpha_channel(img), w, h)
                            latents = encode_image_anima(
                                pipe,
                                image,
                                device,
                                tiled=bool(getattr(config, "VAE_CACHING_TILED", True)),
                                tile_size=tile_size,
                                tile_stride=tile_stride,
                                dtype=vae_cache_dtype,
                            )
                            cached_latents = latents[0].to(dtype=vae_cache_dtype).cpu()
                            torch.save({
                                "latents": cached_latents,
                                "image_path": str(image_path),
                                "relative_path": str(m["ip"].relative_to(root)),
                                "original_size": m["original_size"],
                                "scaled_size": m["scaled_size"],
                                "target_size": tuple(m["target_resolution"]),
                                "crop_coords": m.get("crop_coords", (0, 0)),
                                "bucket_variant_index": m.get("bucket_variant_index", 0),
                                "caption_signature": m.get("caption_signature"),
                                "cache_options": expected_options,
                            }, lat_path)
                        except Exception as e:
                            print(f"[SKIP ANIMA VAE] {image_path.name}: {e}")
                            remove_anima_cache_file(lat_path)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        pbar.update(1)
        else:
            print(f"INFO: Anima cache phase 2/2: VAE cache already current for {root.name}.")

        pipe.vae.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        valid_index_data = [
            item for item in index_data
            if all(Path(path).exists() for path in anima_cache_paths_for_index_item(item))
        ]
        with open(cache_dir / "dataset_index.json", "w", encoding="utf-8") as f:
            json.dump({"version": 5, "cache_options": expected_options, "files": valid_index_data}, f)
        print(f"INFO: Cached {len(valid_index_data)} Anima DiT items to {cache_dir}")

    ensure_anima_null_conditioning_cache(config, pipe, device)


class AnimaCachedDataset(Dataset):
    def __init__(self, config):
        self.items = []
        self.bucket_keys = []
        self.seed = config.SEED if config.SEED else 42
        self.worker_rng = None
        self.json_caption_mode = json_caption_mode_enabled(config)
        self.caption_weights = get_json_caption_weights(config)
        cache_name = anima_cache_folder_name(config)
        self.null_prompt_emb = None
        self.null_t5xxl_ids = None
        self.cond_scale_min, self.cond_scale_max = get_text_conditioning_scale_range(config)
        self.cond_scale_enabled = self.cond_scale_min < 1.0 or self.cond_scale_max > 1.0
        self.dropout_prob = (
            min(max(float(getattr(config, "UNCONDITIONAL_DROPOUT_CHANCE", 0.0)), 0.0), 1.0)
            if getattr(config, "UNCONDITIONAL_DROPOUT", False)
            else 0.0
        )

        for ds in getattr(config, "INSTANCE_DATASETS", []):
            root = Path(ds["path"])
            index_path = root / cache_name / "dataset_index.json"
            if not index_path.exists():
                print(f"WARNING: Anima DiT index missing at {index_path}.")
                continue
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            repeats = int(ds.get("repeats", 1))
            for _ in range(repeats):
                for item in index_data["files"]:
                    self.items.append(item)
                    self.bucket_keys.append(tuple(item["target_size"]))

        if not self.items:
            raise ValueError("No cached Anima DiT files found.")

        combined = list(zip(self.items, self.bucket_keys))
        random.shuffle(combined)
        self.items, self.bucket_keys = zip(*combined)
        self.items = list(self.items)
        self.bucket_keys = list(self.bucket_keys)
        if self.dropout_prob > 0 or self.cond_scale_enabled:
            try:
                null_data = torch.load(
                    Path(config.INSTANCE_DATASETS[0]["path"]) / cache_name / "null_embeds.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.null_prompt_emb = null_data["prompt_emb"].squeeze(0) if null_data["prompt_emb"].dim() == 3 else null_data["prompt_emb"]
                self.null_t5xxl_ids = null_data["t5xxl_ids"].squeeze(0)
            except Exception:
                self.dropout_prob = 0.0
                self.cond_scale_enabled = False

    def __len__(self):
        return len(self.items)

    def _init_worker_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_rng = random.Random(self.seed + (worker_info.id if worker_info else 0))

    def _align_null_prompt_emb(self, prompt_emb):
        null_prompt_emb = self.null_prompt_emb
        if null_prompt_emb is None or prompt_emb.shape == null_prompt_emb.shape:
            return prompt_emb, null_prompt_emb
        if prompt_emb.dim() != 2 or null_prompt_emb.dim() != 2 or prompt_emb.shape[1] != null_prompt_emb.shape[1]:
            return prompt_emb, null_prompt_emb

        prompt_len = prompt_emb.shape[0]
        null_len = null_prompt_emb.shape[0]
        if prompt_len < null_len:
            prompt_pad = null_prompt_emb[prompt_len:null_len].to(dtype=prompt_emb.dtype)
            prompt_emb = torch.cat([prompt_emb, prompt_pad], dim=0)
        elif prompt_len > null_len:
            pad = null_prompt_emb[-1:].expand(prompt_len - null_len, -1)
            null_prompt_emb = torch.cat([null_prompt_emb, pad], dim=0)
        return prompt_emb, null_prompt_emb.to(dtype=prompt_emb.dtype)

    def __getitem__(self, i):
        try:
            if self.worker_rng is None:
                self._init_worker_rng()
            item = self.items[i]
            te_path = selected_caption_variant_path(
                item,
                self.worker_rng,
                self.caption_weights,
                fallback_key="cache_path",
                enabled=self.json_caption_mode,
            )
            lat_path = item.get("lat_path", item.get("cache_path"))
            data_te = torch.load(te_path, map_location="cpu", weights_only=True)
            data_lat = torch.load(lat_path, map_location="cpu", weights_only=True)
            latents = data_lat["latents"]
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                return None
            item_data = {
                "latents": latents,
                "prompt_emb": data_te["prompt_emb"],
                "t5xxl_ids": data_te["t5xxl_ids"],
                "target_size": item["target_size"],
                "latent_path": lat_path,
            }

            if self.dropout_prob > 0 and self.worker_rng.random() < self.dropout_prob:
                _, null_prompt_emb = self._align_null_prompt_emb(item_data["prompt_emb"])
                if null_prompt_emb is not None:
                    item_data["prompt_emb"] = null_prompt_emb
                if null_prompt_emb is not None and self.null_t5xxl_ids is not None:
                    item_data["t5xxl_ids"] = self.null_t5xxl_ids
            elif self.cond_scale_enabled:
                scale = self.worker_rng.uniform(self.cond_scale_min, self.cond_scale_max)
                prompt_emb, null_prompt_emb = self._align_null_prompt_emb(item_data["prompt_emb"])
                if null_prompt_emb is not None:
                    item_data["prompt_emb"] = null_prompt_emb + (prompt_emb - null_prompt_emb) * scale

            return item_data
        except Exception as e:
            print(f"[ANIMA DATASET] Failed to load item {i}: {e}")
            return None


def anima_collate_fn(batch):
    batch = list(filter(None, batch))
    if not batch:
        return {}

    max_t5_len = max(item["t5xxl_ids"].shape[0] for item in batch)
    t5_ids = []
    for item in batch:
        ids = item["t5xxl_ids"]
        if ids.shape[0] < max_t5_len:
            ids = torch.cat([ids, torch.zeros(max_t5_len - ids.shape[0], dtype=ids.dtype)], dim=0)
        t5_ids.append(ids)

    return {
        "latents": torch.stack([item["latents"] for item in batch]),
        "prompt_emb": torch.stack([item["prompt_emb"] for item in batch]),
        "t5xxl_ids": torch.stack(t5_ids),
        "target_size": [item["target_size"] for item in batch],
        "latent_path": [item["latent_path"] for item in batch],
    }


def print_anima_dataset_resolution_sample(dataset, sample_count=5):
    sample_count = min(sample_count, len(dataset.items))
    if sample_count <= 0:
        return

    print(f"INFO: Anima dataset resolution sample ({sample_count} cached item{'s' if sample_count != 1 else ''}):")
    for item in dataset.items[:sample_count]:
        orig_w, orig_h = item["original_size"]
        targ_w, targ_h = item["target_size"]
        orig_ar = orig_w / orig_h if orig_h else 1.0
        targ_ar = targ_w / targ_h if targ_h else 1.0
        ar_error_pct = (abs(orig_ar - targ_ar) / orig_ar * 100) if orig_ar else 0.0
        target_area = targ_w * targ_h
        stem = Path(item.get("lat_path", item.get("cache_path", ""))).stem.replace("_lat", "")
        variant_label = f", variant {item.get('bucket_variant_index', 0)}" if item.get("bucket_variant_index", 0) else ""
        print(
            f"INFO:   {stem}: original {orig_w}x{orig_h} (AR {orig_ar:.4f}) -> "
            f"target {targ_w}x{targ_h} ({target_area:,} px, AR {targ_ar:.4f})"
            f"{variant_label}, AR diff {ar_error_pct:.2f}%, cropped not stretched"
        )


class AnimaTimestepSampler:
    def __init__(self, config, total_timestep_count):
        self.config = config
        self.total_timestep_count = total_timestep_count
        self.total_tickets_needed = config.MAX_TRAIN_STEPS * config.BATCH_SIZE
        self.seed = config.SEED if config.SEED else 42
        self.ticket_pool = self._build_ticket_pool(getattr(config, "TIMESTEP_ALLOCATION", None))
        self.pool_index = 0

    def _build_ticket_pool(self, allocation):
        if not allocation or "counts" not in allocation or "bin_size" not in allocation or sum(allocation["counts"]) == 0:
            bin_size = max(1, self.total_timestep_count // 10)
            bins = max(1, math.ceil(self.total_timestep_count / bin_size))
            counts = [self.total_tickets_needed // bins] * bins
            for i in range(self.total_tickets_needed % bins):
                counts[i] += 1
        else:
            bin_size = max(1, int(allocation["bin_size"]))
            counts = allocation["counts"]

        pool = []
        rng = np.random.Generator(np.random.PCG64(self.seed))
        scale_factor = self.total_timestep_count / 1000.0
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            start_t = int(i * bin_size * scale_factor)
            end_t = min(self.total_timestep_count, max(start_t + 1, int((i + 1) * bin_size * scale_factor)))
            if start_t >= self.total_timestep_count:
                break
            pool.extend(rng.integers(start_t, end_t, size=max(1, int(count))).tolist())

        random.seed(self.seed)
        random.shuffle(pool)
        if not pool:
            pool = [random.randint(0, self.total_timestep_count - 1) for _ in range(self.total_tickets_needed)]
        while len(pool) < self.total_tickets_needed:
            pool.extend(pool[: self.total_tickets_needed - len(pool)])
        return pool[:self.total_tickets_needed]

    def set_current_step(self, micro_step):
        self.pool_index = (micro_step * self.config.BATCH_SIZE) % len(self.ticket_pool)

    def sample(self, batch_size):
        indices = []
        for _ in range(batch_size):
            if self.pool_index >= len(self.ticket_pool):
                self.pool_index = 0
            indices.append(self.ticket_pool[self.pool_index])
            self.pool_index += 1
        return torch.tensor(indices, dtype=torch.long), indices[0]


def freeze_dit_layers(dit, exclude_patterns):
    total_params = frozen_params = trainable_params = 0
    for name, param in dit.named_parameters():
        total_params += param.numel()
        should_freeze = any(fnmatch.fnmatch(name, pattern if "*" in pattern else f"*{pattern}*") for pattern in exclude_patterns)
        param.requires_grad_(not should_freeze)
        if should_freeze:
            frozen_params += param.numel()
        else:
            trainable_params += param.numel()

    print("=" * 56)
    print("INFO: DiT Parameter Statistics")
    print(f"- Total Parameters:     {total_params:,}")
    print(f"- Frozen Parameters:    {frozen_params:,}")
    print(f"- Trainable Parameters: {trainable_params:,}")
    print(f"- Percentage Frozen:    {(frozen_params / max(total_params, 1)) * 100:.2f}%")
    print("=" * 56)


def save_dit_model(output_path, dit, compute_dtype, key_prefix=""):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nINFO: Saving DiT to: {output_path.name}")

    named_params = dict(dit.named_parameters())
    named_buffers = dict(dit.named_buffers())
    param_keys = set(named_params.keys())
    buffer_keys = set(named_buffers.keys())
    total_param_count = sum(p.numel() for p in named_params.values())
    trainable_param_count = sum(p.numel() for p in named_params.values() if p.requires_grad)
    frozen_param_count = total_param_count - trainable_param_count

    state = {}
    key_prefix = "" if str(key_prefix).lower() == "auto" else str(key_prefix or "")
    raw_state = dit.state_dict()
    saved_param_count = 0
    saved_trainable_param_count = 0
    saved_frozen_param_count = 0
    saved_buffer_count = 0
    saved_bytes = 0
    dtype_counts = defaultdict(lambda: {"tensors": 0, "elements": 0})
    for k, v in raw_state.items():
        out_key = f"{key_prefix}{k}"
        tensor_cpu = v.detach().cpu().to(dtype=compute_dtype) if torch.is_floating_point(v) else v.detach().cpu()
        state[out_key] = tensor_cpu
        saved_bytes += tensor_cpu.numel() * tensor_cpu.element_size()
        dtype_key = str(tensor_cpu.dtype).replace("torch.", "")
        dtype_counts[dtype_key]["tensors"] += 1
        dtype_counts[dtype_key]["elements"] += tensor_cpu.numel()
        if k in param_keys:
            saved_param_count += tensor_cpu.numel()
            if named_params[k].requires_grad:
                saved_trainable_param_count += tensor_cpu.numel()
            else:
                saved_frozen_param_count += tensor_cpu.numel()
        elif k in buffer_keys:
            saved_buffer_count += tensor_cpu.numel()

    missing_params = sorted(param_keys - set(raw_state.keys()))
    missing_trainable_params = [k for k in missing_params if named_params[k].requires_grad]
    extra_state_keys = sorted(set(raw_state.keys()) - param_keys - buffer_keys)

    print("INFO: DiT save accounting:")
    print(f"- Model parameters:       {total_param_count:,}")
    print(f"- Parameters saved:       {saved_param_count:,} / {total_param_count:,} ({(saved_param_count / max(total_param_count, 1)) * 100:.2f}%)")
    print(f"- Trainable saved:        {saved_trainable_param_count:,} / {trainable_param_count:,} ({(saved_trainable_param_count / max(trainable_param_count, 1)) * 100:.2f}%)")
    print(f"- Frozen saved:           {saved_frozen_param_count:,} / {frozen_param_count:,} ({(saved_frozen_param_count / max(frozen_param_count, 1)) * 100:.2f}%)")
    print(f"- Buffer elements saved:  {saved_buffer_count:,}")
    print(f"- State tensors saved:    {len(state):,}")
    print(f"- Estimated tensor bytes: {saved_bytes / (1024 ** 2):.2f} MiB")
    if dtype_counts:
        dtype_summary = ", ".join(
            f"{dtype}: {info['tensors']:,} tensors / {info['elements']:,} elems"
            for dtype, info in sorted(dtype_counts.items())
        )
        print(f"- Saved dtypes:           {dtype_summary}")
    if missing_params:
        print(f"WARNING: {len(missing_params):,} named parameters were not present in state_dict().")
        if missing_trainable_params:
            print(f"WARNING: {len(missing_trainable_params):,} missing parameters were trainable.")
        for name in missing_params[:10]:
            print(f"  -> missing param: {name}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10:,} more")
    else:
        print("INFO: All named DiT parameters are present in the saved state.")
    if extra_state_keys:
        print(f"INFO: State dict has {len(extra_state_keys):,} non-parameter/non-buffer tensor keys.")
        for name in extra_state_keys[:5]:
            print(f"  -> extra state: {name}")
        if len(extra_state_keys) > 5:
            print(f"  ... and {len(extra_state_keys) - 5:,} more")
    if key_prefix:
        print(f"INFO: Saved DiT keys with prefix: {key_prefix}")
    save_file(state, str(output_path))
    file_size = output_path.stat().st_size if output_path.exists() else 0
    try:
        with safe_open(str(output_path), framework="pt", device="cpu") as f:
            disk_keys = set(f.keys())
        expected_keys = set(state.keys())
        missing_disk_keys = sorted(expected_keys - disk_keys)
        unexpected_disk_keys = sorted(disk_keys - expected_keys)
        print(f"INFO: Safetensors file keys: {len(disk_keys):,} / expected {len(expected_keys):,}")
        print(f"INFO: Safetensors file size: {file_size / (1024 ** 2):.2f} MiB")
        if missing_disk_keys or unexpected_disk_keys:
            print(
                "WARNING: Safetensors key verification mismatch "
                f"(missing={len(missing_disk_keys):,}, unexpected={len(unexpected_disk_keys):,})."
            )
            for name in missing_disk_keys[:10]:
                print(f"  -> missing on disk: {name}")
            for name in unexpected_disk_keys[:10]:
                print(f"  -> unexpected on disk: {name}")
        else:
            print("INFO: Safetensors key verification passed.")
    except Exception as e:
        print(f"WARNING: Could not verify saved safetensors keys: {e}")
    print("INFO: DiT save complete.")


def save_anima_checkpoint_pt(global_step, micro_step, dit, optimizer, lr_scheduler, sampler, config, noise_generator=None, batch_sampler=None):
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"{Path(config.DIT_PATH).stem}_step_{global_step}.safetensors"
    state_filename = f"training_state_step_{global_step}.pt"
    save_dit_model(
        output_dir / model_filename,
        dit,
        config.compute_dtype,
        getattr(config, "ANIMA_DIT_SAVE_PREFIX", ""),
    )
    optim_state = optimizer.save_cpu_state() if hasattr(optimizer, "save_cpu_state") else optimizer.state_dict()
    torch.save({
        "global_step": global_step,
        "micro_step": micro_step,
        "optimizer_state": optim_state,
        "sampler_seed": sampler.seed,
        "sampler_pool_index": sampler.pool_index,
        "batch_sampler_epoch": max(batch_sampler.epoch - 1, 0) if batch_sampler is not None else 0,
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_cpu_state": torch.get_rng_state(),
        "torch_cuda_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "noise_generator_state": noise_generator.get_state() if noise_generator is not None else None,
    }, output_dir / state_filename)


def safe_set_anima_scheduler_training(pipe):
    try:
        pipe.scheduler.set_timesteps(1000, training=True)
    except TypeError:
        pipe.scheduler.set_timesteps(1000)
        try:
            pipe.scheduler.training = True
        except Exception:
            pass


def run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config):
    model_output = dit(
        x=noisy_latents.unsqueeze(2),
        timesteps=timesteps / 1000,
        context=prompt_emb,
        t5xxl_ids=t5xxl_ids,
        use_gradient_checkpointing=bool(getattr(config, "USE_GRADIENT_CHECKPOINTING", True)),
        use_gradient_checkpointing_offload=bool(getattr(config, "USE_GRADIENT_CHECKPOINTING_OFFLOAD", True)),
    )
    return model_output.squeeze(2)


def flowmatch_noise_and_target(input_latents, noise, sigmas):
    sigmas = sigmas.view((sigmas.shape[0],) + (1,) * (input_latents.ndim - 1))
    return (1 - sigmas) * input_latents + sigmas * noise, noise - input_latents


def weighted_flowmatch_mse(model_pred, training_target, weights):
    per_sample_loss = (model_pred.float() - training_target.float()).pow(2).flatten(1).mean(dim=1)
    return (per_sample_loss * weights.float()).mean()



def run_anima_dit_training(config):
    normalize_anima_config(config)

    is_resuming = getattr(config, "RESUME_TRAINING", False)
    required_paths = ["VAE_PATH", "TEXT_ENCODER_PATH"]
    required_paths.extend(["ANIMA_RESUME_MODEL_PATH", "ANIMA_RESUME_STATE_PATH"] if is_resuming else ["DIT_PATH"])
    for attr in required_paths:
        value = getattr(config, attr, "")
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{attr} is required for Anima DiT training: {value}")

    if config.SEED:
        set_seed(config.SEED)

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    force_save_flag = Path(__file__).resolve().with_name("force_save.flag")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    micro_step = optimizer_step = 0
    optimizer_state = None
    saved_sampler_pool_index = None
    saved_noise_generator_state = None
    saved_batch_sampler_epoch = 0
    if is_resuming:
        print("\n" + "=" * 50 + "\n--- RESUMING ANIMA DIT TRAINING SESSION ---\n")
        training_state = torch.load(Path(config.ANIMA_RESUME_STATE_PATH), map_location="cpu", weights_only=False)
        micro_step = training_state.get("micro_step", training_state.get("global_step", 0) * config.GRADIENT_ACCUMULATION_STEPS)
        optimizer_step = micro_step // config.GRADIENT_ACCUMULATION_STEPS
        optimizer_state = training_state.get("optimizer_state")
        saved_sampler_pool_index = training_state.get("sampler_pool_index")
        saved_noise_generator_state = training_state.get("noise_generator_state")
        saved_batch_sampler_epoch = training_state.get("batch_sampler_epoch", 0)
        config.DIT_PATH = config.ANIMA_RESUME_MODEL_PATH
        if "random_state" in training_state:
            random.setstate(training_state["random_state"])
        if "numpy_state" in training_state:
            np.random.set_state(training_state["numpy_state"])
        if "torch_cpu_state" in training_state:
            torch.set_rng_state(training_state["torch_cpu_state"])
        if "torch_cuda_state" in training_state and training_state["torch_cuda_state"] is not None:
            torch.cuda.set_rng_state(training_state["torch_cuda_state"])
    else:
        print("\n" + "=" * 50 + "\n--- STARTING ANIMA DIT TRAINING ---\n" + "=" * 50 + "\n")

    print("INFO: Loading Anima pipeline components on CPU...")
    pipe = load_anima_pipe(config, torch.device("cpu"))
    safe_set_anima_scheduler_training(pipe)

    print("INFO: Precomputing Anima DiT cache if needed...")
    precompute_and_cache_anima(config, pipe, device)

    print("INFO: Moving VAE/Text Encoder off GPU for DiT training...")
    pipe.vae.cpu()
    pipe.text_encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dit = pipe.dit
    dit.to(device=device, dtype=config.compute_dtype)
    dit.train()
    freeze_dit_layers(dit, getattr(config, "DIT_EXCLUDE_TARGETS", []))

    params_to_optimize = [{"params": [p for p in dit.parameters() if p.requires_grad], "lr_scale": 1.0}]
    all_trainable_params = [p for group in params_to_optimize for p in group["params"]]
    optimizer = create_optimizer(config, params_to_optimize)
    lr_scheduler = CustomCurveLRScheduler(optimizer=optimizer, curve_points=config.LR_CUSTOM_CURVE, total_micro_steps=config.MAX_TRAIN_STEPS)
    if optimizer_state:
        try:
            optimizer.load_cpu_state(optimizer_state) if hasattr(optimizer, "load_cpu_state") else optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            print(f"WARNING: Could not load optimizer state: {e}")
        lr_scheduler.step(micro_step)

    dataset = AnimaCachedDataset(config)
    print(f"INFO: Cached Anima DiT dataset items: {len(dataset)}")
    print_anima_dataset_resolution_sample(dataset, sample_count=5)
    batch_sampler = BucketBatchSampler(dataset, config.BATCH_SIZE, config.SEED, shuffle=True)
    if is_resuming:
        batch_sampler.set_epoch(saved_batch_sampler_epoch)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=anima_collate_fn,
    )

    total_scheduler_timesteps = len(pipe.scheduler.timesteps)
    timestep_sampler = AnimaTimestepSampler(config, total_scheduler_timesteps)
    if saved_sampler_pool_index is not None:
        timestep_sampler.pool_index = int(saved_sampler_pool_index) % len(timestep_sampler.ticket_pool)
    elif micro_step > 0:
        timestep_sampler.set_current_step(micro_step)
    scheduler_timesteps = pipe.scheduler.timesteps.to(device=device)
    scheduler_sigmas = pipe.scheduler.sigmas.to(device=device, dtype=config.compute_dtype)
    scheduler_weights = getattr(pipe.scheduler, "linear_timesteps_weights", torch.ones_like(pipe.scheduler.timesteps))
    scheduler_weights = scheduler_weights.to(device=device, dtype=torch.float32)

    reporter = AsyncReporter(total_steps=config.MAX_TRAIN_STEPS, test_param_name="dit")
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.SEED if config.SEED else 42)
    if saved_noise_generator_state is not None:
        generator.set_state(saved_noise_generator_state)
    accumulated_latent_paths = set()
    optim_step_times = deque(maxlen=20)
    global_step_times = deque(maxlen=50)
    training_start_time = time.time()
    last_step_time = time.time()
    last_optim_step_log_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    dataloader_iter = iter(dataloader)
    batches_to_skip = micro_step % len(dataloader) if is_resuming and len(dataloader) > 0 else 0

    while micro_step < config.MAX_TRAIN_STEPS:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        if batches_to_skip > 0:
            batches_to_skip -= 1
            continue
        if not batch:
            continue

        step_start = time.time()
        lr_scheduler.step(micro_step)
        input_latents = batch["latents"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        prompt_emb = batch["prompt_emb"].to(device=device, dtype=config.compute_dtype, non_blocking=True)
        t5xxl_ids = batch["t5xxl_ids"].to(device=device, non_blocking=True)

        batch_size = input_latents.shape[0]
        timestep_indices, timestep_str = timestep_sampler.sample(batch_size)
        timestep_indices = timestep_indices.to(device=device)
        timesteps = scheduler_timesteps[timestep_indices].to(device=device, dtype=config.compute_dtype)
        sigmas = scheduler_sigmas[timestep_indices]
        loss_weights = scheduler_weights[timestep_indices]
        noise = torch.randn(input_latents.shape, device=device, dtype=config.compute_dtype, generator=generator)

        noisy_latents, training_target = flowmatch_noise_and_target(input_latents, noise, sigmas)
        model_pred = run_dit_forward(dit, noisy_latents, timesteps, prompt_emb, t5xxl_ids, config)
        loss = weighted_flowmatch_mse(model_pred, training_target, loss_weights)
        raw_loss_value = loss.detach().float().item()
        (loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

        diagnostics.step(raw_loss_value)
        accumulated_latent_paths.update(batch["latent_path"])
        diag_data_to_log = None

        if (micro_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            raw_grad_norm = (
                optimizer.clip_grad_norm(config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float("inf"))
                if isinstance(optimizer, TitanAdamW)
                else torch.nn.utils.clip_grad_norm_(
                    all_trainable_params,
                    config.CLIP_GRAD_NORM if config.CLIP_GRAD_NORM > 0 else float("inf"),
                )
            )
            if isinstance(raw_grad_norm, torch.Tensor):
                raw_grad_norm = raw_grad_norm.item()
            clipped_grad_norm = min(raw_grad_norm, config.CLIP_GRAD_NORM) if config.CLIP_GRAD_NORM > 0 else raw_grad_norm

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            optim_step_time = time.time() - last_optim_step_log_time
            optim_step_times.append(optim_step_time)
            last_optim_step_log_time = time.time()

            diag_data_to_log = {
                "optim_step": optimizer_step,
                "avg_loss": diagnostics.get_average_loss(),
                "current_lr": optimizer.param_groups[-1]["lr"],
                "raw_grad_norm": raw_grad_norm,
                "clipped_grad_norm": clipped_grad_norm,
                "update_delta": 1.0 if raw_grad_norm > 0 else 0.0,
                "optim_step_time": optim_step_time,
                "avg_optim_step_time": sum(optim_step_times) / len(optim_step_times),
            }
            diagnostics.reset()
            accumulated_latent_paths.clear()

            scheduled_save = (
                config.SAVE_EVERY_N_STEPS > 0
                and optimizer_step > 0
                and optimizer_step % config.SAVE_EVERY_N_STEPS == 0
            )
            force_save = consume_force_save_flag(force_save_flag)
            if scheduled_save or force_save:
                reason = "Emergency checkpoint requested" if force_save and not scheduled_save else "Saving checkpoint"
                reporter.log_message(f"\n--- {reason} at optimizer step {optimizer_step} ---")
                save_anima_checkpoint_pt(
                    optimizer_step,
                    micro_step + 1,
                    dit,
                    optimizer,
                    lr_scheduler,
                    timestep_sampler,
                    config,
                    generator,
                    batch_sampler,
                )

        step_duration = time.time() - last_step_time
        global_step_times.append(step_duration)
        last_step_time = time.time()
        avg_step_time = sum(global_step_times) / len(global_step_times) if global_step_times else 0.0
        reporter.log_step(
            micro_step,
            timing_data={
                "raw_step_time": time.time() - step_start,
                "elapsed_time": time.time() - training_start_time,
                "eta": (config.MAX_TRAIN_STEPS - micro_step) * avg_step_time,
                "loss": raw_loss_value,
                "timestep": int(timestep_str),
            },
            diag_data=diag_data_to_log,
        )
        micro_step += 1

    reporter.log_message("\nTraining complete.")
    reporter.shutdown()
    final_path = output_dir / f"{Path(config.DIT_PATH).stem}_trained_full_{str(uuid.uuid4())[:4]}.safetensors"
    save_dit_model(final_path, dit, config.compute_dtype, getattr(config, "ANIMA_DIT_SAVE_PREFIX", ""))
    print("All tasks complete. Final DiT saved.")


def main():
    from train import TrainingConfig

    config = TrainingConfig()
    run_anima_dit_training(config)


if __name__ == "__main__":
    main()
