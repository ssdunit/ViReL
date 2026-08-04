#configuration file
"""
config.py

Configuration file for Robo2VLM GRPO training.
"""

from dataclasses import dataclass
import torch 


# ==========================================================
# Model Configuration
# ==========================================================

@dataclass
class ModelConfig:
    MODEL_NAME: str = "unsloth/Qwen3-VL-32B-Thinking"

    MAX_SEQ_LENGTH:int=4096

    LOAD_IN_4BIT: bool=False 

    TORCH_DTYPE: torch.dtype =torch.bfloat16

    USE_GRADIENT_CHECKPOINTING: str="unsloth"

    DEVICE_MAP: str="auto"




# ==========================================================
# Dataset Configuration
# ==========================================================

@dataclass
class DataConfig:
    DATASET_NAME: str = "keplerccc/Robo2VLM-1"

    TRAIN_SPLIT: str = "train"

    TEST_SPLIT: str = "test"

    MAX_TRAIN_SAMPLES: int | None = None

    MAX_TEST_SAMPLES: int | None = None

    SEED: int = 42


# ==========================================================
# Prompt Configuration
# ==========================================================

SYSTEM_PROMPT = """
You are an expert robotics vision-language assistant.

You are given an image from a robot manipulation task together
with a multiple-choice question.

Reason carefully about the image before selecting the answer.

Rules:

1. Think inside <think></think> tags.
2. Output ONLY one capital letter inside <answer></answer>.
3. The answer must be one of the provided options.
4. Do not explain anything outside the tags.

Example:

<think>
The red cube is closest to the gripper.
</think>

<answer>
B
</answer>
"""

@dataclass
class TrainConfig:

    OUTPUT_DIR: str = "./outputs"

    NUM_EPOCHS: int = 2

    BATCH_SIZE: int =4

    LEARNING_RATE: float = 2e-5

    WEIGHT_DECAY: float =0.01

    OPTIM: str="adamw_8bit"

    ADAM_BETA1:float= 0.9

    ADAM_BETA2:float=0.999

    ADAM_EPSILON: float =1e-8

    GRADIENT_ACCUMULATION_STEPS: int = 4

    LOGGING_STEPS: int = 10

    SAVE_STEPS: int = 500

    EVAL_STEPS: int = 500

    EVAL_STRATEGY:str  = "steps"

    EVAL_BATCH_SIZE: int = 4
    
    RESUME_FROM_CHECKPOINT: str | None = None

    SAVE_TOTAL_LIMIT:int  =2

    SAVE_STRATEGY: str= "steps"

    LR_SCHEDULER_TYPE: str ="cosine"

    WARMUP_RATIO: float=0.03

    MAX_GRAD_NORM: float=1.0

    RANK: int = 128

    ALPHA: int = 256

    DROPOUT: float = 0.0

    BIAS: str="none"

    BF16: bool=True

    FP16: bool=False

    REMOVE_UNUSED_COLUMNS: bool = False

    LOGGING_FIRST_STEP: bool = True
    
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


@dataclass
class GRPOConfig:

    NUM_GENERATIONS: int = 8

    TEMPERATURE: float = 1.0

    TOP_P: float = 0.95

    MAX_PROMPT_LENGTH: int=2048

    MAX_COMPLETION_LENGTH: int = 512

    BETA:float  =0.04

@dataclass
class LoggingConfig:
    REPORT_TO: str="wandb"
    RUN_NAME: str ="robo2vlm-grpo"
    PROJECT: str ="Robo2VLM"
    LOG_LEVEL: str ="info"
