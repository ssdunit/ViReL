pip install --upgrade pip && pip install uv

uv pip install --no-deps transformers>=4.45.0
uv pip install --no-deps datasets>=3.0.0
uv pip install --no-deps trl>=0.12.0
uv pip install --no-deps accelerate>=0.30.0
uv pip install --no-deps peft>=0.13.0
uv pip install --no-deps bitsandbytes>=0.43.0
uv pip install --no-deps wandb>=0.18.0
uv pip install --no-deps tqdm numpy ollama
uv pip install platformdirs

pip cache purge && uv cache clean