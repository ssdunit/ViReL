import torch
from dataclasses import dataclass

@dataclass
class hyperparameters():
    device:str = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    MODEL_ID:str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    epochs:int = 2
    learning_rate:float = 3e-5
    train_batch_size:int = 2
    gradient_accumulation_steps:int = 4
    lora_r:int = 16
    lora_alpha:int = 32 
    lora_dropout:float = 0.1 