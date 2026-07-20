from dataclasses import dataclass,field
from typing import List,Optional
import torch
from transformers import BitsAndBytesConfig
cuda_available = torch.cuda.is_available()

@dataclass
class DataConfig():
    dataset:str = "dpdl-benchmark/oxford_flowers102"
    prompt:str = "Identify the flower species in this image and output structured JSON."
    seed:int = 42

@dataclass
class hyperparameters():
    device:str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MODEL_ID:str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    epochs:int = 2
    learning_rate:float = 3e-5
    train_batch_size:int = 2
    gradient_accumulation_steps:int = 4

@dataclass
class TrainingConfig():
    logging_steps:int = 1
    hf_save:bool = False
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str="cosine"
    bf16:bool=True
    max_length:int = 2048
    warmup_ratio:float=0.05
    weight_decay:float=0.01
    save_steps:int=200
    eval_steps:int=200

@dataclass
class LoRaConfig():
    lora_r:int = 16
    lora_alpha:int = 32 
    lora_dropout:float = 0.1 
    lora_target_modules: tuple = ("q_proj","v_proj","k_proj","o_proj")
    bias:str = "None"
    task_type:str="CAUSAL_LM"

@dataclass
class ModelConfig():
    bnb_config: BitsAndBytesConfig= field(
        default_factory=lambda:BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if cuda_available else torch.float32
        )
    )
    device_map:str="auto"
    torch_dtype:torch=torch.float16
    attn_implementation:str="sdpa"

@dataclass
class Config():
    data: DataConfig = field(default_factory=DataConfig)
    hyp: hyperparameters = field(default_factory=hyperparameters)
    tc : TrainingConfig = field(default_factory=TrainingConfig)
    LR: LoRaConfig = field(default_factory=LoRaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: str = "wandb"
    wandb_run: Optional[str] = "res768-smolvlm2-fc-v2"
    model_output_dir : str = "finalmodel/"