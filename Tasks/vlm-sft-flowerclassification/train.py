"""
train.py

Trains a LoRA adapter on top of a quantized vision-language model
(default: SmolVLM2-2.2B-Instruct) to classify flower species from images,
using the Oxford Flowers 102 dataset and TRL's SFTTrainer.

The pipeline:
    1. Loads and formats the dataset into chat-style SFT examples.
    2. Loads the base VLM in 4-bit and wraps it with a LoRA adapter.
    3. Trains with early stopping on eval loss.
    4. Saves the resulting adapter, processor, and a label map for
       downstream evaluation.

Every run also writes a run_config.json capturing exactly how it was
launched, so a saved checkpoint can always be traced back to the settings
that produced it.
"""

import warnings
warnings.filterwarnings("ignore", message=".*processor_kwargs.*")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import argparse
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import EarlyStoppingCallback, AutoProcessor, AutoModelForImageTextToText, set_seed
from peft import get_peft_model, LoraConfig
from trl import SFTConfig, SFTTrainer

from data_format import dataset_format, DEFAULT_DATASET, DEFAULT_PROMPT
from Config import model_config, build_bnb_config


def parse_args():
    p = argparse.ArgumentParser(
        description="SFT training for VLM flower classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                    help="HuggingFace dataset id to train/eval on.")
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                    help="Instruction prompt paired with each training image.")
    p.add_argument("--seed", type=int, default=42, required=False,
                    help="Random seed for dataset shuffling, weight init, and training.")

    # Model
    p.add_argument("--model_id", type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                    help="Base vision-language model to fine-tune.")
    p.add_argument("--image_longest_edge", type=int, default=384,
                    help="Resize images so their longest edge is this many pixels.")

    # Quantization
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True,
                    help="Load the base model in 4-bit precision via bitsandbytes.")
    p.add_argument("--bnb_quant_type", type=str, default="nf4",
                    help="bitsandbytes 4-bit quantization type.")
    p.add_argument("--bnb_double_quant", action=argparse.BooleanOptionalAction, default=True,
                    help="Use nested (double) quantization for the 4-bit quant constants.")

    # Core training hyperparameters
    p.add_argument("--epochs", type=float, default=2,
                    help="Number of training epochs.")
    p.add_argument("--learning_rate", type=float, default=3e-5,
                    help="Peak learning rate for the optimizer.")
    p.add_argument("--train_batch_size", type=int, default=2,
                    help="Per-device training batch size.")
    p.add_argument("--eval_batch_size", type=int, default=2,
                    help="Per-device evaluation batch size.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Number of steps to accumulate gradients before an optimizer step.")
    p.add_argument("--max_length", type=int, default=2048,
                    help="Maximum sequence length for training examples.")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                    help="Fraction of total steps used for learning-rate warmup.")
    p.add_argument("--weight_decay", type=float, default=0.01,
                    help="Weight decay applied by the optimizer.")
    p.add_argument("--optimizer", type=str, default="paged_adamw_8bit",
                    help="Optimizer to use for training.")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                    help="Learning rate schedule type.")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                    help="Train in bfloat16 mixed precision.")

    # Logging / checkpointing
    p.add_argument("--logging_steps", type=int, default=1,
                    help="Log training metrics every N steps.")
    p.add_argument("--save_steps", type=int, default=200,
                    help="Save a checkpoint every N steps.")
    p.add_argument("--eval_steps", type=int, default=200,
                    help="Run evaluation every N steps.")
    p.add_argument("--save_total_limit", type=int, default=2,
                    help="Maximum number of checkpoints to keep on disk.")
    p.add_argument("--early_stopping_patience", type=int, default=3,
                    help="Stop training after this many evals without eval-loss improvement.")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16,
                    help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha (scaling factor).")
    p.add_argument("--lora_dropout", type=float, default=0.1,
                    help="Dropout applied within the LoRA adapter layers.")
    p.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
                    help="Which bias terms to train alongside the LoRA weights.")
    p.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        help="Model submodule names to attach LoRA adapters to.",
    )

    # Output / tracking
    p.add_argument("--output_dir", type=str, default="results",
                    help="Directory for Trainer checkpoints and logs.")
    p.add_argument("--model_output_dir", type=str, default="finalmodel/",
                    help="Directory to save the final adapter, processor, label map, and run config.")
    p.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"],
                    help="Experiment tracking backend to report metrics to.")
    p.add_argument("--wandb_project", type=str, default="smolvlm2-flowerclassification-v1",
                    help="W&B project name (used when --report_to wandb).")
    p.add_argument("--wandb_run_name", type=str, default="res768-smolvlm2-fc-v2",
                    help="W&B run name (used when --report_to wandb).")

    return p.parse_args()


def main():
    args = parse_args()

    # Seed everything (python/random, numpy, torch, cuda) up front so data
    # shuffling, weight init, and sampling are all reproducible from this
    # single flag.
    set_seed(args.seed)

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        import wandb
        wandb.finish()

    train_formatted_datasets, eval_formatted_datasets, _, label_map = dataset_format(
        dataset=args.dataset, prompt=args.prompt, seed=args.seed
    )

    bnb_config = build_bnb_config(args.load_in_4bit, args.bnb_quant_type, args.bnb_double_quant)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        size={"longest_edge": args.image_longest_edge},
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map=model_config.device_map,
        torch_dtype=model_config.torch_dtype,
        attn_implementation=model_config.attn_implementation,
    )

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=list(args.lora_target_modules),
    )

    target_model = model.model if hasattr(model, "model") and hasattr(model.model, "inputs_merger") else model

    # Use only if need to explicitly cast float instead of bfloat, else not needed
    if hasattr(target_model, "inputs_merger"):
        old_merger = target_model.inputs_merger

        def patched_merger(input_ids, inputs_embeds, image_hidden_states):
            return old_merger(input_ids, inputs_embeds, image_hidden_states.to(inputs_embeds.dtype))

        target_model.inputs_merger = patched_merger

    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        bf16=args.bf16,
        max_length=args.max_length,
        use_liger_kernel=False,
        disable_tqdm=False,
        packing=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        loss_type="nll",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        assistant_only_loss=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_formatted_datasets,
        eval_dataset=eval_formatted_datasets,
        args=training_args,
        peft_config=None,
        data_collator=None,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()

    if args.report_to == "wandb":
        import wandb
        wandb.finish()

    output_dir = args.model_output_dir
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save the label map alongside the model (handy for eval.py / downstream
    # use), and the exact resolved CLI args this run was launched with, so
    # the run is fully reproducible from this one file.
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Model and processor saved to {output_dir}")
    print(f"Run config saved to {os.path.join(output_dir, 'run_config.json')}")


if __name__ == "__main__":
    main()