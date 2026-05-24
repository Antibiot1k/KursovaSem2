# ============================================================
# STABLE DIFFUSION RESEARCH ENVIRONMENT
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import json
import gc
import warnings

warnings.filterwarnings('ignore')

from diffusers import StableDiffusionPipeline, DDIMScheduler
from torchvision import transforms

# ── CONFIG ──────────────────────────────────────────────────

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'runwayml/stable-diffusion-v1-5'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

OUT_DIR = Path('/content/results')
OUT_DIR.mkdir(exist_ok=True)

# ── MATPLOTLIB STYLE ────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'figure.dpi': 110,
})

# ── DEVICE INFO ─────────────────────────────────────────────

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Device: {DEVICE}')

if DEVICE == 'cuda':
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {total:.1f} GB')

# ── LOAD PIPELINE ───────────────────────────────────────────

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe = pipe.to(DEVICE)

if DEVICE == 'cuda':
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

print('\n✅ Stable Diffusion loaded successfully')

# ── TEST GENERATION ─────────────────────────────────────────

prompt = (
    "simple pixel art necromancer, black robe, "
    "green glowing eyes, indie game sprite, "
    "transparent background"
)

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    width=512,
    height=512
).images[0]

# ── SAVE RESULT ─────────────────────────────────────────────

save_path = OUT_DIR / 'necromancer.png'
image.save(save_path)

print(f'\n✅ Image saved: {save_path}')

# ── DISPLAY ─────────────────────────────────────────────────

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.show()
