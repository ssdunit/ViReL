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
from hyperparameters import hyperparameters

train_formatted_datasets,eval_formatted_datasets,_ = dataset_format()
config = hyperparameters()

#BitsandBytes Config
cuda_available = torch.cuda.is_available()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 if cuda_available else torch.float32
)

processor = AutoProcessor.from_pretrained(
    config.MODEL_ID,
    size={"longest_edge": 384},
)
model = AutoModelForImageTextToText.from_pretrained(
    config.MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
)

peft_config = LoraConfig(
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    r=config.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","v_proj","k_proj","o_proj"]
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
    num_train_epochs=config.epochs,
    per_device_train_batch_size=config.train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    logging_steps=1,
    #save_strategy="epoch",
    #eval_strategy="epoch",
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="wandb",
    run_name ="res768-smolvlm2-fc-v2",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    fp16=False,
    bf16=True,
    max_length=2048,
    use_liger_kernel=False,
    disable_tqdm = False,
    packing=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    loss_type="nll",
    warmup_ratio=0.05,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
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
output_dir = "finalmodel/"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Model and processor saved to {output_dir}")