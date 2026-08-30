"""
config.py

Configuration file for Robo2VLM GRPO training.
"""

from dataclasses import dataclass
import torch 

SYSTEM_PROMPT = """
You are an expert robotics vision-language assistant.

You are given an image from a robot manipulation task together with a multiple-choice question.

Your task is to reason carefully about the image and select the correct answer.

Rules:
1. You MUST start your response with <think>.
2. Do all of your reasoning inside <think> and </think> tags.
3. After thinking, output your final choice inside <answer> and </answer> tags.
4. The content of the <answer> tag MUST be exactly ONE capital letter (e.g., A, B, C, D).
5. Do not output any other text before <reason> or after </answer>.
6. Keep your thinking length less than 400 words.

response:
<think>
Within this should be the visual reasoning
</think>
<answer>
[correct option letter ]
</answer>
"""

@dataclass
class ModelConfig:
    MODEL_NAME: str = "unsloth/Qwen3-VL-8B-Thinking"

    MAX_SEQ_LENGTH:int=2048

    TORCH_DTYPE: torch.dtype =torch.bfloat16

    USE_GRADIENT_CHECKPOINTING: str="unsloth"

    DEVICE_MAP: str="auto"

    Load_in_4bit: bool = True


@dataclass
class DataConfig:
    DATASET_NAME: str = "Rauneet/robo2vlm-erqa-plus-combined"

    TRAIN_SPLIT: str = "train"

    TEST_SPLIT: str = "test"

    MAX_TRAIN_SAMPLES: int | None = None

    MAX_TEST_SAMPLES: int | None = None

    SEED: int = 42


@dataclass
class TrainConfig:

    OUTPUT_DIR: str = "./outputs_grpo"

    NUM_EPOCHS: int = 2

    BATCH_SIZE: int =2

    LEARNING_RATE: float = 1e-6

    WEIGHT_DECAY: float =0.01

    OPTIM: str="paged_adamw_8bit"

    ADAM_BETA1:float= 0.9

    ADAM_BETA2:float=0.999

    ADAM_EPSILON: float =1e-8

    GRADIENT_ACCUMULATION_STEPS: int = 4

    LOGGING_STEPS: int = 1

    SAVE_STEPS: int = 50

    EVAL_STEPS: int = 500

    EVAL_STRATEGY:str  = "steps"

    EVAL_BATCH_SIZE: int = 8
    
    RESUME_FROM_CHECKPOINT: str | None = None

    SAVE_TOTAL_LIMIT:int  = 3

    SAVE_STRATEGY: str= "steps"

    LR_SCHEDULER_TYPE: str ="cosine"

    WARMUP_RATIO: float=0.03

    WARMUP_STEPS: int=20

    MAX_GRAD_NORM: float=1.0

    RANK: int = 16

    ALPHA: int = 32

    DROPOUT: float = 0.0

    BIAS: str="none"

    BF16: bool=True

    FP16: bool=False

    REMOVE_UNUSED_COLUMNS: bool = True

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
    
    RANDOM_STATE = 3407

@dataclass
class GRPOConfig:

    NUM_GENERATIONS: int = 8

    TEMPERATURE: float = 0.9

    TOP_P: float = 0.9

    MAX_PROMPT_LENGTH: int=1536

    MAX_COMPLETION_LENGTH: int =256

    BETA:float  = 0.15

@dataclass
class LoggingConfig:
    REPORT_TO: str="wandb"
    RUN_NAME: str ="robo2vlm-grpo"
    PROJECT: str ="Robo2VLM"
    LOG_LEVEL: str ="info"