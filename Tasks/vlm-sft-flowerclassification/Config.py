"""
Structural / architectural config only.

Anything that affects what a run *produces* (learning rate, batch size, LoRA
rank, epochs, seed, dataset, prompt, quantization type, etc.) has been moved
to CLI arguments in train.py / eval.py so every run is fully specified by its
command line and can be reproduced exactly. What's left here is fixed
plumbing that isn't meant to vary between experiments.
"""

from dataclasses import dataclass
import torch

cuda_available = torch.cuda.is_available()


@dataclass
class ModelConfig:
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    attn_implementation: str = "sdpa"


model_config = ModelConfig()


def build_bnb_config(load_in_4bit: bool, bnb_quant_type: str, bnb_double_quant: bool):
    """Build a BitsAndBytesConfig from CLI-provided quantization flags."""
    from transformers import BitsAndBytesConfig

    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bnb_double_quant,
        bnb_4bit_quant_type=bnb_quant_type,
        bnb_4bit_compute_dtype=torch.float16 if cuda_available else torch.float32,
    )