import warnings
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import EarlyStoppingCallback,BitsAndBytesConfig, AutoProcessor, AutoModelForImageTextToText
from peft import get_peft_model,LoraConfig
from trl import SFTConfig, SFTTrainer
import torch

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"]="smolvlm2-flowerclassification-v1"
os.environ["WANDB_LOG_MODEL"]="checkpoint"

import wandb
wandb.finish()

from data_format import dataset_format
from Config import Config

train_formatted_datasets,eval_formatted_datasets,_ = dataset_format()
config = Config()


processor = AutoProcessor.from_pretrained(
    config.MODEL_ID,
    size={"longest_edge": 384},
)
model = AutoModelForImageTextToText.from_pretrained(
    config.MODEL_ID,
    quantization_config=config.model.bnb_config,
    device_map=config.model.device_map,
    torch_dtype=config.model.torch_dtype,
    attn_implementation=config.model.attn_implementation,
)

peft_config = LoraConfig(
    lora_alpha=config.LR.lora_alpha,
    lora_dropout=config.LR.lora_dropout,
    r=config.LR.lora_r,
    bias=config.LR.bias,
    task_type=config.LR.task_type,
    target_modules=list(config.LR.lora_target_modules)
)

target_model = model.model if hasattr(model, "model") and hasattr(model.model, "inputs_merger") else model

### Use only if need to explicitly cast float instead of bfloat else not needed
if hasattr(target_model, "inputs_merger"):
    old_merger = target_model.inputs_merger
    def patched_merger(input_ids, inputs_embeds, image_hidden_states):
        # Explicitly cast image_hidden_states to match inputs_embeds (float16)
        return old_merger(input_ids, inputs_embeds, image_hidden_states.to(inputs_embeds.dtype))
    target_model.inputs_merger = patched_merger

model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
model = model.to(torch.bfloat16)
model.print_trainable_parameters()

for name, param in model.named_parameters():
    if param.requires_grad and param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)

training_args = SFTConfig(
    output_dir="results",
    num_train_epochs=config.hyp.epochs,
    per_device_train_batch_size=config.hyp.train_batch_size,
    gradient_accumulation_steps=config.hyp.gradient_accumulation_steps,
    learning_rate=config.hyp.learning_rate,
    logging_steps=config.tc.logging_steps,
    #save_strategy="epoch",
    #eval_strategy="epoch",
    remove_unused_columns=False,
    push_to_hub=False,
    report_to=config.wandb,
    run_name =config.wandb_run,
    optim=config.tc.optimizer,
    lr_scheduler_type=config.tc.lr_scheduler_type,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    #fp16=False,
    bf16=config.tc.bf16,
    max_length=config.tc.max_length,
    use_liger_kernel=False,
    disable_tqdm = False,
    packing=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    loss_type="nll",
    warmup_ratio=config.tc.warmup_ratio,
    weight_decay=config.tc.weight_decay,
    save_strategy="steps",
    save_steps=config.tc.save_steps,
    eval_strategy="steps",
    eval_steps=config.tc.eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    assistant_only_loss = False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_formatted_datasets,
    eval_dataset=eval_formatted_datasets,
    args=training_args,
    peft_config=None,
    data_collator=None,
    processing_class=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

wandb.finish()
output_dir = config.model_output_dir
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Model and processor saved to {output_dir}")