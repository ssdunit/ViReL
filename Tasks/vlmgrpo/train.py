#training file 
import os 
import torch 
from unsloth import FastVisionModel
from trl import GRPOTrainer,GRPOConfig
from data_util import load_from_disk
from data_util import preprocess_example,load_robo2vlm
from config import(
    ModelConfig,
    DataConfig,
    TrainConfig,
    GRPOConfig as GRPOTrainConfig,
    LoggingConfig,
    SYSTEM_PROMPT,
)
from reward import(
    answer_correctness_reward,
    format_reward,
)

def setup_logging():
    os.environ["WANDB_PROJECT"]=LoggingConfig.PROJECT
    os.environ["WANDB_RUN_NAME"]=LoggingConfig.RUN_NAME

def load_model():
    model,tokenizer = FastVisionModel.from_pretrained(
        model_name=ModelConfig.MODEL_NAME,
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dtype=ModelConfig.TORCH_DTYPE,
        load_in_4bit=TrainConfig.Load_in_4bit,
        device_map=ModelConfig.DEVICE_MAP,
        #max_pixels=262144, 
        
        # 2. Enable Unsloth's vLLM backend for ultra-fast, memory-efficient GRPO generation
        fast_inference=False,
        
        # 3. Restrict vLLM VRAM allocation so PyTorch has room to train
        gpu_memory_utilization=0.2,
    )

    return model, tokenizer

def apply_lora(model):
    model = FastVisionModel.get_peft_model(
        model,
        r=TrainConfig.RANK,
        lora_alpha=TrainConfig.ALPHA,
        lora_dropout=TrainConfig.DROPOUT,
        target_modules=TrainConfig.TARGET_MODULES,
        bias=TrainConfig.BIAS,
        use_gradient_checkpointing=ModelConfig.USE_GRADIENT_CHECKPOINTING,
        random_state=TrainConfig.RANDOM_STATE,
    )
    return model
"""
def prepare_dataset():
    train_dataset= load_from_disk(
        "./datasets/ROBO2VLM_train_10k"

    )
    train_dataset=train_dataset.map(
        lambda x:preprocess_example(
            x,
            SYSTEM_PROMPT,
        )
    )
    return train_dataset
"""

def build_grpo_config():
    training_args = GRPOConfig(
        # Training
        output_dir=TrainConfig.OUTPUT_DIR,
        num_train_epochs=TrainConfig.NUM_EPOCHS,
        per_device_train_batch_size=TrainConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainConfig.GRADIENT_ACCUMULATION_STEPS,
        
        # Optimizer
        learning_rate=TrainConfig.LEARNING_RATE,
        weight_decay=TrainConfig.WEIGHT_DECAY,
        optim=TrainConfig.OPTIM,
        adam_beta1=TrainConfig.ADAM_BETA1,
        adam_beta2=TrainConfig.ADAM_BETA2,
        adam_epsilon=TrainConfig.ADAM_EPSILON,

        # Scheduler
        lr_scheduler_type=TrainConfig.LR_SCHEDULER_TYPE,
        #warmup_ratio=TrainConfig.WARMUP_RATIO,
        max_grad_norm=TrainConfig.MAX_GRAD_NORM,

        # Precision
        bf16=TrainConfig.BF16,
        fp16=TrainConfig.FP16,

        # Logging
        logging_strategy="steps",
        logging_steps=TrainConfig.LOGGING_STEPS,
        report_to=LoggingConfig.REPORT_TO,
        run_name=LoggingConfig.RUN_NAME,

        # Evaluation
        eval_strategy=TrainConfig.EVAL_STRATEGY,
        eval_steps=TrainConfig.EVAL_STEPS,

        # Checkpointing
        save_strategy=TrainConfig.SAVE_STRATEGY,
        save_steps=TrainConfig.SAVE_STEPS,
        save_total_limit=TrainConfig.SAVE_TOTAL_LIMIT,

        # GRPO
        num_generations=GRPOTrainConfig.NUM_GENERATIONS,
        temperature=GRPOTrainConfig.TEMPERATURE,
        top_p=GRPOTrainConfig.TOP_P,
        max_prompt_length=GRPOTrainConfig.MAX_PROMPT_LENGTH,
        max_completion_length=GRPOTrainConfig.MAX_COMPLETION_LENGTH,
        beta=GRPOTrainConfig.BETA,

        # Reproducibility
        seed=TrainConfig.RANDOM_STATE,
        warmup_steps=TrainConfig.WARMUP_STEPS
    )

    return training_args

def create_trainer(model, tokenizer, train_dataset,test_dataset, training_args):
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        reward_funcs=[
            answer_correctness_reward,
            format_reward,
            #reasoning_length_reward,
        ],
        processing_class=tokenizer,
    )

    return trainer
def train_model(trainer):
    trainer.train()


def merge_and_save_model(model,tokenizer,):
    merge_model=model.merge_and_unload()

    merge_model.save_pretrained(
        TrainConfig.OUTPUT_DIR,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(
        TrainConfig.OUTPUT_DIR
    )
def main():
    setup_logging()

    model, tokenizer = load_model()

    model = apply_lora(model)

    train_dataset = load_robo2vlm(dataset_name="robo2vlm",system_prompt = SYSTEM_PROMPT,tokenizer=tokenizer,split="train")
    test_dataset = load_robo2vlm(dataset_name="robo2vlm",system_prompt=SYSTEM_PROMPT,tokenizer=tokenizer, split="test")

    training_args = build_grpo_config()

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        training_args=training_args,
    )

    train_model(trainer)
    merge_and_save_model(
        model=model,
        tokenizer=tokenizer,
    )

if __name__=="__main__":
    main()



