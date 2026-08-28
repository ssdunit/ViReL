#training file 
import os 
import torch 
from unsloth import FastVisionModel
from trl import GRPOTrainer,GRPOConfig
from data_util import load_robo2vlm
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
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator 

'''def build_sft_config():
    return SFTConfig(
        output_dir=TrainConfig.OUTPUT_DIR + "_sft_coldstart",
        num_train_epochs=1,
        per_device_train_batch_size=TrainConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainConfig.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=TrainConfig.BF16,
        fp16=TrainConfig.FP16,
        logging_steps=TrainConfig.LOGGING_STEPS,
        eval_steps = 100,
        eval_strategy = "steps",
        per_device_eval_batch_size = 4,
        save_strategy="steps",
        save_steps = 100,
        save_total_limit = 3,
        report_to=LoggingConfig.REPORT_TO,
        run_name=LoggingConfig.RUN_NAME + "_sft_coldstart",
        remove_unused_columns=False,
        dataset_text_field="",       # required by SFTConfig for vision collators
        max_seq_length=8192,
    )

def run_sft_cold_start(model, tokenizer):
    FastVisionModel.for_training(model)

    sft_train = load_spatialladder_sft(SYSTEM_PROMPT, tokenizer, split="train")
    sft_eval = load_spatialladder_sft(SYSTEM_PROMPT,tokenizer,split="test")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=sft_train,
        eval_dataset = sft_eval,
        args=build_sft_config(),
    )
    trainer.train()
    return model'''

def setup_logging():
    os.environ["WANDB_PROJECT"]=LoggingConfig.PROJECT
    os.environ["WANDB_RUN_NAME"]=LoggingConfig.RUN_NAME

def load_model(model_path=None):
    model,tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path or ModelConfig.MODEL_NAME,
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dtype=ModelConfig.TORCH_DTYPE,
        #load_in_8bit=TrainConfig.Load_in_4bit,
        load_in_4bit=True,
        device_map=ModelConfig.DEVICE_MAP,
        #max_pixels=262144,    
    )
    #tokenizer.image_processor.min_pixels=256*28*28
    #tokenizer.image_processor.max_pixels=512*28*28

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
        log_completions=True,
        num_completions_to_print=5,  
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
        #max_prompt_length=GRPOTrainConfig.MAX_PROMPT_LENGTH,
        max_prompt_length = None,
        max_completion_length=GRPOTrainConfig.MAX_COMPLETION_LENGTH,
        beta=GRPOTrainConfig.BETA,

        # Reproducibility
        seed=TrainConfig.RANDOM_STATE,
        warmup_steps=TrainConfig.WARMUP_STEPS,
        repetition_penalty=1.15,
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
    '''SFT_CHECKPOINT = "./outputs_sft_coldstart/checkpoint-500"'''
    model, tokenizer = load_model(model_path=None)
    model = apply_lora(model)

    #model = run_sft_cold_start(model, tokenizer)
    
    train_dataset = load_robo2vlm(dataset_name="Rauneet/robo2vlm-erqa-plus-combined",system_prompt=SYSTEM_PROMPT,tokenizer=tokenizer,split="train")
    sample = train_dataset[0]
    print("\n========== IMAGE TEST ==========")

    print("Keys:", sample.keys())

    print("Image object:", sample["image"])
    print("Image type:", type(sample["image"]))
    print("Image size:", sample["image"].size)
    print("Image mode:", sample["image"].mode)

    print("\nPrompt:")
    print(sample["prompt"])

    print("================================")
    test_dataset = load_robo2vlm(dataset_name="Rauneet/robo2vlm-erqa-plus-combined",system_prompt=SYSTEM_PROMPT,tokenizer=tokenizer,split="test")
    sample = train_dataset[0]

    

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



