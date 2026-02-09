# Aozora SDXL Trainer

> [!WARNING]
> **Beta Software:** This is still under active development and in beta. It is fully functional, successfully training 80–90% of the full UNet using only 11.80 GB VRAM, with ~1.55 seconds per iteration and ~15 seconds per optimizer step.
>
> Some rough edges and minor issues are to be expected. The primary goal is to make SDXL training practical and efficient on 12 GB VRAM GPUs without the usual compromises or complexity.
>
> **Note:** Sorry for any previously failed attempts at training. Getting this stable on consumer hardware is tricky.

A GUI trainer for SDXL fine-tuning that fits on single consumer GPUs. Built because existing tools either need 24GB+ or require you to train at lower resolutions.

Works with v-prediction and Flow Matching models (NoobAI, Illustrious, etc.) out of the box.

<img src="https://images2.imgbox.com/9c/f2/iQZQ9LHl_o.png" alt="Gui" width="850"/>
<img src="https://images2.imgbox.com/81/c8/yB61nX4p_o.png" alt="Gui" width="850"/>

## Who This Is For

- You have **12GB-16GB VRAM** (RTX 3060/4060/4070 etc.)
- You want to train on **one GPU** (no multi-GPU setup needed)
- You're tired of editing JSON configs and want buttons that work

## What It Actually Does

**Custom Optimizers**
Two built-in optimizers designed specifically for 12GB training:

- **RavenAdamW**: Pre-allocates GPU buffers so you don't get OOM errors mid-training. Keeps momentum/variance states on CPU, computes updates on GPU using shared buffers. Has proper weight decay ordering (decay happens before the update, not after) and optional gradient centralization. About 20% faster than standard optimizers for SDXL.

- **VeloRMS**: Even more memory efficient. Uses velocity + RMS normalization with a small "leak" of gradient info to keep things stable on sparse updates (like when training specific layers). Also CPU-offloads states. Includes verbose logging mode that prints diagnostics every N steps so you can see if your model is about to explode.

**Ticket Pool Timesteps**
Instead of random timestep sampling, you get a visual "ticket pool" system. You can allocate how many training steps go to early vs middle vs late timesteps using a bar chart. Want to train mostly on the middle 200-800 range for Flow models? Drag the bars. The system handles the distribution math for you.

**Graph Learning Rate Scheduler**
Draw your LR curve visually in the GUI. Want a warmup that spikes then cosine decays? Click the points, drag them around. The graph interpolates between points so you get exact control over the schedule without editing config files.

**Layer Targeting**
Pick exactly which UNet blocks train. Want only attention layers? Just check those boxes. Saves VRAM and prevents overfitting compared to training everything.

**v-pred & Flow Support**
Handles both standard epsilon-prediction and modern flow-matching models. Includes the shift scheduling and timestep distributions that actually work for SDXL (not just SD 1.5 ported over).

**Frozen Text Encoder Training**
Caches text embeddings so you can train with text encoder updates off by default (saves ~4GB VRAM). Optional token-embedding-only mode for teaching new concepts without destroying the base model's understanding.

**Smart Latent Caching**
Pre-processes your images through the VAE once, stores them compressed. Training runs faster because it's not encoding images every step.

**Dynamic Resolution Shifting**
Automatically adjusts flow-shift strength based on your image resolution. Training 1536px images uses different noise scheduling than 1024px - it's handled automatically.

**Memory Stack**
- Flash Attention 2 support
- Gradient checkpointing 
- BF16/FP16 mixed precision
- The optimizers handle their own CPU offloading so you don't run out of VRAM keeping optimizer states

## Quick Start

1. Put your images in a folder with matching .txt captions
2. Run `setup.bat` (pick Flash Attention if you have RTX 20-series or newer)
3. Run `start_gui.bat`
4. Select your base model, set your folder, adjust features to your liking or use a preset, hit Train

Checkpoints save to `output/checkpoints/` every N steps. If it crashes, resume from the latest `.pt` state file.

## Requirements

- Windows 10/11
- Python 3.10
- NVIDIA GPU with 12GB+ VRAM
- About 20GB free disk space for the environment
- 32GB of ram for offloading

## Notes

- Only works with standard SDXL format models (fp16/bf16 safetensors)
- Layer targeting is hand-tuned for specific architectures - using merged models might behave weird
- Flow models need the shift factor set right (GUI has presets)
- Raven and Velo optimizers are designed for single-GPU training; they offload to CPU to save VRAM which will be slower on multi-GPU setups
