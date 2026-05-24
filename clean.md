# ============================================================
# 1. CLEAN & INSTALL CORRECT STACK (ALL IN ONE)
# ============================================================

# Видаляємо старі пакети, що викликали конфлікт
!pip uninstall -y torch torchvision torchaudio diffusers transformers accelerate xformers huggingface_hub peft sentence-transformers timm

# Ставимо фіксований робочий стек однією командою
!pip install -q \
torch==2.5.1 \
torchvision==0.20.1 \
torchaudio==2.5.1 \
diffusers==0.35.1 \
transformers==4.45.0 \
accelerate==1.0.0 \
huggingface_hub==0.26.0 \
safetensors \
ftfy \
xformers==0.0.28.post1 \
--index-url https://download.pytorch.org/whl/cu121
