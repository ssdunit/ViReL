"""
Inference script for the LoRA/GRPO-tuned GSM8K model.

Design notes (why things are wired the way they are):
- Every "training truth" value (model name, torch dtype, attn impl, LoRA rank,
  reasoning/solution tags, system prompt shape, temperature, top_p,
  max_completion_length, seed) is pulled from config.py's ExperimentConfig /
  DataConfig. Nothing is hardcoded or re-tuned here, so generation stays
  faithful to what the model was actually trained on.
- The prompt format mirrors data_utils.py's format_custom_gsm8k exactly
  (same system prompt string, same "start your response immediately with
  <REASONING>" nudge in the user turn) so the model sees the same
  distribution at inference time that it saw during RL training.
- Answer extraction reuses the same regex logic as rewards.correctness_reward
  so that eval-mode accuracy numbers are computed the same way the reward
  model scored them during training.
- Loading assumes trainer.save_model() saved a PEFT adapter (the standard
  behavior when config.model.use_lora=True), so we load the base model with
  transformers and attach the adapter with peft. If no adapter is found at
  the given path, the script falls back to treating the path as a full
  (merged) model directory.
"""

import argparse
import json
import re
import sys
from pathlib import Path
import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DataConfig, ExperimentConfig
from utils import set_seed

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


#Config
exp_config = ExperimentConfig()
data_config = DataConfig()

MODEL_NAME = exp_config.model.model_name
TORCH_DTYPE = exp_config.model.torch_dtype
ATTN_IMPL = exp_config.model.attn_implementation
MAX_SEQ_LENGTH = exp_config.model.max_seq_length

TEMPERATURE = exp_config.grpo.temperature
TOP_P = exp_config.grpo.top_p
MAX_COMPLETION_LENGTH = exp_config.grpo.max_completion_length
SEED = exp_config.grpo.seed
DEFAULT_ADAPTER_DIR = exp_config.grpo.output_dir

REASONING_START = data_config.F_tags.reasoning_start
REASONING_END = data_config.F_tags.reasoning_end
SOLUTION_START = data_config.F_tags.solution_start
SOLUTION_END = data_config.F_tags.solution_end

# Identical to the system prompt built inside data_utils.load_gsm8k_dataset
SYSTEM_PROMPT = f"""You are a strict mathematics reasoning assistant. 
You must solve the math problem step-by-step. Your entire response MUST be formatted exactly like this template:

{REASONING_START}
Write your step-by-step mathematical logic and calculations here.
{REASONING_END}
{SOLUTION_START}
Write ONLY the final numerical answer here.
{SOLUTION_END}"""

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}



def load_model(adapter_path: str, load_in_4bit: bool = False, device_map: str = "auto"):
    """Load the base model (per config.py) and attach the trained adapter if present."""
    torch_dtype = _DTYPE_MAP.get(TORCH_DTYPE, torch.bfloat16)

    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {MODEL_NAME} (dtype={TORCH_DTYPE}, attn={ATTN_IMPL})")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        attn_implementation=ATTN_IMPL,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_dir = Path(adapter_path)
    is_adapter = (adapter_dir / "adapter_config.json").exists()

    if is_adapter:
        if not _HAS_PEFT:
            raise RuntimeError(
                f"Found adapter_config.json at {adapter_path} but 'peft' is not installed. "
                f"Run: pip install peft"
            )
        print(f"Attaching LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        # Try to load the fine-tuned tokenizer too (saved alongside the adapter),
        # falls back to the base tokenizer if not present.
        if (adapter_dir / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
    elif adapter_dir.exists() and (adapter_dir / "config.json").exists():
        print(f"No adapter_config.json found — loading {adapter_path} as a full/merged model.")
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_dir),
            torch_dtype=torch_dtype,
            attn_implementation=ATTN_IMPL,
            device_map=device_map,
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"No checkpoint found at {adapter_path} — using base model with no fine-tuning applied.")
        model = base_model

    model.eval()
    return model, tokenizer

def build_prompt(question: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\nStart your response immediately with {REASONING_START}\n"},
    ]


@torch.no_grad()
def generate(model, tokenizer, question: str, do_sample: bool = True) -> str:
    messages = build_prompt(question)
    
    # Store as 'inputs' since it returns a dict with 'input_ids' and 'attention_mask'
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=MAX_COMPLETION_LENGTH,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    # Unpack both input_ids and attention_mask into generate()
    output_ids = model.generate(**inputs, **gen_kwargs)
    
    # Index into the dict to get the actual tensor shape
    prompt_length = inputs["input_ids"].shape[-1]
    completion_ids = output_ids[0][prompt_length:]
    
    return tokenizer.decode(completion_ids, skip_special_tokens=True)

def extract_final_answer(text: str):
    match = re.search(rf"{SOLUTION_START}\s*(.*?)\s*{SOLUTION_END}", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    model_answer = match.group(1).strip()
    numbers = re.findall(r"\d+\.?\d*", model_answer)
    return numbers[-1] if numbers else None


def run_interactive(model, tokenizer, do_sample: bool):
    print("\nInteractive mode. Type a GSM8K-style question ('quit' to exit).\n")
    while True:
        try:
            question = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        completion = generate(model, tokenizer, question, do_sample=do_sample)
        print(completion)
        answer = extract_final_answer(completion)
        print(f"\n[extracted answer]: {answer}\n")


def run_single(model, tokenizer, question: str, do_sample: bool):
    completion = generate(model, tokenizer, question, do_sample=do_sample)
    answer = extract_final_answer(completion)
    print(completion)
    print(f"\n[extracted answer]: {answer}")


def run_eval(model, tokenizer, num_samples: int, do_sample: bool):
    from data_utils import load_gsm8k_dataset

    _, test_dataset = load_gsm8k_dataset()
    if num_samples > 0:
        test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

    correct = 0
    total = 0
    results = []

    progress_bar = tqdm(test_dataset, total=len(test_dataset), desc="Evaluating", unit="q")

    for example in test_dataset:
        question = example["prompt"][1]["content"]
        gt_answer = example["answer"]

        completion = generate(model, tokenizer, question, do_sample=do_sample)
        pred_answer = extract_final_answer(completion)

        gt_numeric = "".join(c for c in str(gt_answer) if c.isdigit() or c == ".")
        is_correct = pred_answer is not None and pred_answer == gt_numeric

        correct += int(is_correct)
        total += 1
        results.append(
            {
                "question": question,
                "ground_truth": gt_answer,
                "prediction": pred_answer,
                "correct": is_correct,
                "completion": completion,
            }
        )
        progress_bar.set_postfix(
            acc=f"{correct}/{total} ({correct/total:.2%})",
            gt = gt_numeric,
            pred = pred_answer,
            correct="True" if is_correct else correct="False",)
        #print(f"[{total}/{len(test_dataset)}] correct={is_correct}  pred={pred_answer}  gt={gt_answer}")


    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}")

    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps({"accuracy": accuracy, "results": results}, indent=2))
    print(f"Full results written to {out_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference for the GRPO-tuned GSM8K model.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=DEFAULT_ADAPTER_DIR,
        help=f"Path to the trained LoRA adapter or full model dir (default: config.grpo.output_dir = '{DEFAULT_ADAPTER_DIR}').",
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the base model in 4-bit (bitsandbytes).")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of the training sampling params.")
    parser.add_argument("--question", type=str, default=None, help="Run a single question and exit.")
    parser.add_argument("--eval", action="store_true", help="Evaluate on the GSM8K test split.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test examples for --eval (0 = full test set).")
    args = parser.parse_args()

    set_seed(SEED)

    model, tokenizer = load_model(args.adapter_path, load_in_4bit=args.load_in_4bit)
    do_sample = not args.greedy

    if args.eval:
        run_eval(model, tokenizer, num_samples=args.num_samples, do_sample=do_sample)
    elif args.question:
        run_single(model, tokenizer, args.question, do_sample=do_sample)
    else:
        run_interactive(model, tokenizer, do_sample=do_sample)


if __name__ == "__main__":
    sys.exit(main())
