# ============================================================
# FULL CLEAN INSTALL FOR STABLE DIFFUSION COLAB (T4 READY)
# ============================================================

import subprocess

# ── GPU CHECK ───────────────────────────────────────────────

result = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True,
    text=True
)

print('🖥️ GPU:')
print(result.stdout.strip() or 'GPU not found')

# ── CLEAN OLD PACKAGES ──────────────────────────────────────

!pip uninstall -y \
diffusers \
transformers \
accelerate \
peft \
safetensors

# ── INSTALL COMPATIBLE STACK ────────────────────────────────

!pip install -q \
diffusers==0.30.3 \
transformers==4.44.2 \
accelerate==0.34.2 \
peft==0.12.0 \
safetensors==0.4.5 \
torchmetrics[image] \
lpips \
open-clip-torch

print("\n✅ Dependencies installed successfully")

# ── RESTART RUNTIME (IMPORTANT) ─────────────────────────────

import os
os.kill(os.getpid(), 9)
