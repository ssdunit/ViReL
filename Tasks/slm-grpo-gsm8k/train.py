#Unsloth
from unsloth import FastLanguageModel,PatchFastRL
#PatchFastRL("GRPO",FastLanguageModel)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer

from config import ExperimentConfig
from data_utils import load_gsm8k_dataset
from rewards import (
    correctness_reward,
    strict_format_reward,
    xml_formatting_reward,
    cosine_scaled_reward,
    repetition_penalty_reward,
    reasoning_length_reward
)

from utils import print_trainable_parameters, set_seed, setup_wandb

def main() -> None:
    config = ExperimentConfig()

    set_seed(config.grpo.seed)
    setup_wandb(config.wandb_project, config.wandb_run_name)

    model,tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=config.model.lora_r,
        gpu_memory_utilization=0.6,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = None
    if config.model.use_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(config.model.lora_target_modules),
            use_gradient_checkpointing="unsloth", 
            random_state=config.grpo.seed,
        )

    print_trainable_parameters(model)

    train_dataset, test_dataset = load_gsm8k_dataset(config=DataConfig())

    training_args = TRLGRPOConfig(
        output_dir=config.grpo.output_dir,
        num_generations=config.grpo.num_generations,
        max_completion_length=config.grpo.max_completion_length,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        learning_rate=config.grpo.learning_rate,
        beta=config.grpo.beta,
        num_train_epochs=config.grpo.num_train_epochs,
        per_device_train_batch_size=config.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        warmup_ratio=config.grpo.warmup_ratio,
        weight_decay=config.grpo.weight_decay,
        max_grad_norm=config.grpo.max_grad_norm,
        log_level=config.grpo.log_level,
        logging_steps=config.grpo.logging_steps,
        save_steps=config.grpo.save_steps,
        save_total_limit=config.grpo.save_total_limit,
        eval_strategy=config.grpo.eval_strategy,
        bf16=config.grpo.bf16,
        use_cpu=False,
        report_to=config.grpo.report_to,
        seed=config.grpo.seed,
        remove_unused_columns=False,
        use_vllm=False,
        max_prompt_length=config.model.max_seq_length - config.grpo.max_completion_length,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward,
            strict_format_reward,
            xml_formatting_reward,
            cosine_scaled_reward,
            repetition_penalty_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.save_model(config.grpo.output_dir)
    tokenizer.save_pretrained(config.grpo.output_dir)
    print(f"Model saved to {config.grpo.output_dir}")


if __name__ == "__main__":
    main()