from dataclasses import dataclass, field
from typing import Optional,List

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    max_seq_length: int = 2048


@dataclass
class GRPOConfig:
    num_generations: int = 4
    max_completion_length: int = 1024
    temperature: float = 0.9
    top_p: float = 1.0
    learning_rate: float = 5e-6
    beta: float = 0.005
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    log_level: str = "info"
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    eval_strategy: str = "no"
    bf16: bool = False
    output_dir: str = "grpo-gsm8k-output"
    report_to: str = "wandb"
    seed: int = 42
    
@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    wandb_project: str = "grpo-gsm8k"
    wandb_run_name: Optional[str] = "slm-grpo-gsm8k-v2"

@dataclass
class DynamicSLMThinking:
    model_name: str = "qwen2.5:1.5b"
    host_url: str = "https://localhost:11434"
    timeout: int = 60
    temperature: float = 0.7 
    max_tokens: int = 100
    stop_sequences: List[str] = field(default_factory=lambda: ["\n\n", "Math Problem:"])
    system_prompt: str = (
        "You are a data labeling assistant. Read the math problem and solution. "
        "Write a concise 1-2 sentence plan stating the required mathematical operations "
        "and formulas. Do NOT solve the problem or include numbers."
    )

@dataclass
class FormattingTags:
    #Formatting tags
    reasoning_start:str = "<REASONING>"
    reasoning_end:str = "</REASONING>"
    thinking_start:str = "<THINKING>"
    thinking_end:str = "</THINKING>"
    solution_start:str = "<SOLUTION>"
    solution_end:str = "</SOLUTION>"

@dataclass
class DataConfig:
    DST:DynamicSLMThinking= field(default_factory=DynamicSLMThinking)
    F_tags: FormattingTags = field(default_factory=FormattingTags)
    dataset: str = "openai/gsm8k"
