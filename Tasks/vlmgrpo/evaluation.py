import argparse
import io
import re
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Default model — unsloth/Qwen3-VL-8B-Thinking is the full BF16 weight repo
# (not FP8/GGUF), so it loads fine through the standard transformers
# Auto* classes below. Requires a recent `transformers` build:
#   pip install git+https://github.com/huggingface/transformers
# Older releases raise "model type not recognized" for Qwen3-VL.
DEFAULT_MODEL_PATH = "unsloth/Qwen3-VL-8B-Thinking"


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def strip_thinking(text):
    """
    Remove the <think>...</think> block emitted by Qwen3-VL-Thinking
    before the final response, so it can never contaminate answer
    parsing. Returns the trace and the "final" text separately.
    """
    if not text:
        return "", ""

    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)

    if match:
        thinking = match.group(1).strip()
        final_text = text[match.end():].strip()
        return thinking, final_text

    # No closing tag found (e.g. generation was truncated mid-think).
    # Treat everything as thinking and leave nothing to parse an
    # answer from — this makes truncation visible instead of guessing.
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), ""

    # Model didn't use think tags at all — nothing to strip.
    return "", text.strip()


def extract_answer(text):
    """
    Extract A/B/C/D/E from the model's FINAL (post-thinking) text only.

    Handles outputs such as:
        A
        The answer is B.
        <answer>C</answer>
        Therefore, the correct answer is D.
    """

    if not text:
        return None

    # Prefer explicit answer tag
    tag_match = re.search(
        r"<answer>\s*([A-E])\s*</answer>",
        text,
        re.IGNORECASE
    )

    if tag_match:
        return tag_match.group(1).upper()

    # Look for phrases indicating final answer
    final_patterns = [
        r"(?:final answer|final|answer)\s*(?:is|:)?\s*\(?([A-E])\)?",
        r"(?:correct answer)\s*(?:is|:)?\s*\(?([A-E])\)?",
    ]

    for pattern in final_patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).upper()

    # Fallback: last standalone A-E letter.
    # IMPORTANT: case-sensitive on purpose. With IGNORECASE this would
    # also match the lowercase word "a" (the English article), which
    # shows up constantly in ordinary prose and silently corrupts
    # results. Use lookaround instead of \b so we don't match letters
    # that are part of a bigger word/token.
    matches = re.findall(r"(?<![A-Za-z])[A-E](?![A-Za-z])", text)

    if matches:
        return matches[-1].upper()

    return None


# ============================================================
# MODEL
# ============================================================

def load_model(model_path):

    print(f"\nLoading model from: {model_path}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Qwen3-VL (both Instruct and Thinking variants) is not Qwen2.5-VL —
    # use the auto class so it resolves to the correct architecture
    # (Qwen3VLForConditionalGeneration under the hood). Requires a recent
    # `transformers` — Qwen's own model card recommends building from
    # source (`pip install git+https://github.com/huggingface/transformers`)
    # since Qwen3-VL support landed after the last release at time of
    # writing. If this raises a "model type not recognized" error, that's
    # almost certainly why — upgrade transformers first.
    attn_impl = "sdpa"
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        pass

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
    except (ValueError, ImportError):
        # Some transformers/model combos don't accept attn_implementation
        # for this architecture yet — retry without forcing it.
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model.eval()

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"GPU memory allocated after load: {allocated_gb:.2f} GB")

    print("Model loaded successfully.\n")

    return model, processor


# ============================================================
# PROMPT
# ============================================================

def build_prompt(question, choices):
    """
    Build RoboVista multiple-choice prompt.
    """

    choice_text = ""

    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        choice_text += f"{letter}. {choice}\n"

    prompt = f"""
You are a visual reasoning assistant.

Analyze the image carefully and answer the question.

Question:
{question}

Choices:
{choice_text}

Think through the visual evidence and reasoning internally.

At the end, provide your final answer in this exact format:
<think>
provide reasoning for  your approach to that answer 
</think>
<answer>X</answer>

where X is one of A, B, C, D, or E.
"""

    return prompt.strip()


# ============================================================
# IMAGE HANDLING
# ============================================================

def get_image(example):
    """
    Handles common RoboVista image representations.
    """

    image = example.get("image", None)

    if image is None:
        image = example.get("images", None)

    if image is None:
        raise ValueError(
            "Could not find image/images field in dataset example."
        )

    # Some datasets return a list of images
    if isinstance(image, list):
        image = image[0]

    # PIL image
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # HF image object / dictionary
    if isinstance(image, dict):

        if "path" in image and image["path"]:
            return Image.open(image["path"]).convert("RGB")

        if "bytes" in image and image["bytes"]:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")

    # Local path
    if isinstance(image, str):
        return Image.open(image).convert("RGB")

    raise ValueError(
        f"Unsupported image type: {type(image)}"
    )


# ============================================================
# SINGLE INFERENCE
# ============================================================

@torch.inference_mode()
def run_inference(
    model,
    processor,
    image,
    question,
    choices,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    min_p=0.0,
):

    prompt = build_prompt(
        question,
        choices
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # Single-step chat template application, per the official Qwen3-VL
    # model card. This lets the processor place image tokens itself,
    # which is more robust than tokenizing text and calling processor()
    # separately (the two-step version can occasionally mismatch image
    # placeholder counts if the template changes between releases).
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device)
        if torch.is_tensor(v)
        else v
        for k, v in inputs.items()
    }

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )

    if do_sample:
        gen_kwargs.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        if min_p is not None:
            gen_kwargs.update(min_p=min_p)

    start_time = time.time()

    generated_ids = model.generate(**inputs, **gen_kwargs)

    inference_time = time.time() - start_time

    # Remove prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs["input_ids"],
            generated_ids
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    thinking_text, final_text = strip_thinking(output_text)

    # If the model never closed </think> (truncated by max_new_tokens),
    # final_text will be empty and predicted_answer will correctly
    # come back None instead of a guess.
    predicted_answer = extract_answer(final_text)

    truncated = "<think>" in output_text and "</think>" not in output_text

    return {
        "full_output": output_text,
        "thinking": thinking_text,
        "final_answer_text": final_text,
        "predicted_answer": predicted_answer,
        "inference_time": inference_time,
        "truncated": truncated,
    }


# ============================================================
# DATASET
# ============================================================

def load_robovista(domain=None):

    print("Loading RoboVista dataset...")

    dataset = load_dataset(
        "sy-xie/robovista",
        split="train"
    )

    print(f"Total RoboVista samples: {len(dataset)}")

    if len(dataset) > 0:
        print(f"Example fields: {list(dataset[0].keys())}")

    if domain is not None:

        dataset = dataset.filter(
            lambda x: x.get("domain") == domain
        )

        print(f"Samples in domain '{domain}': {len(dataset)}")

        if len(dataset) == 0:
            print(
                "WARNING: 0 samples matched this domain string. "
                "Check the dataset's actual domain values (see "
                "'Example fields' above, or inspect dataset[0]['domain'])."
            )

    return dataset


# ============================================================
# CHOICE PARSING
# ============================================================

def get_choices(example):

    choices = example.get("choices")

    if choices is None:
        raise ValueError(
            "Dataset example does not contain 'choices'."
        )

    # Already a list
    if isinstance(choices, list):
        return choices

    # Sometimes choices may be a dictionary
    if isinstance(choices, dict):

        ordered = []

        for letter in ["A", "B", "C", "D", "E"]:
            if letter in choices:
                ordered.append(choices[letter])

        return ordered

    raise ValueError(
        f"Unsupported choices type: {type(choices)}"
    )


# ============================================================
# BENCHMARK EVALUATION
# ============================================================

def evaluate_robovista(
    model,
    processor,
    dataset,
    output_file,
    max_samples=None,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    min_p=0.0,
):

    results = []

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total = len(dataset)

    print(f"\nStarting evaluation on {total} samples...\n")

    correct = 0
    invalid = 0
    errored = 0
    truncated_count = 0

    for idx, example in enumerate(dataset):

        print(f"[{idx + 1}/{total}]", end=" ", flush=True)

        try:

            image = get_image(example)
            question = example["question"]
            choices = get_choices(example)
            ground_truth = str(example["correct_answer"]).strip().upper()

            result = run_inference(
                model=model,
                processor=processor,
                image=image,
                question=question,
                choices=choices,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )

            predicted_answer = result["predicted_answer"]

            is_correct = (
                predicted_answer is not None
                and predicted_answer == ground_truth
            )

            if is_correct:
                correct += 1

            if predicted_answer is None:
                invalid += 1

            if result["truncated"]:
                truncated_count += 1

            print(
                f"GT={ground_truth} PRED={predicted_answer} "
                f"{'✓' if is_correct else '✗'}"
                f"{' [TRUNCATED]' if result['truncated'] else ''}"
            )

            if is_correct:
                error_type = "correct"
            elif predicted_answer is None:
                error_type = (
                    "truncated_before_answer"
                    if result["truncated"]
                    else "invalid_answer"
                )
            else:
                error_type = "wrong_answer"

            results.append({
                "sample_id": example.get("id", idx),
                "domain": example.get("domain", ""),
                "task": example.get("task", ""),
                "question": question,
                "choices": " | ".join(
                    f"{chr(ord('A') + i)}. {c}"
                    for i, c in enumerate(choices)
                ),
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "correct": is_correct,
                "model_thinking": result["thinking"],
                "model_final_answer_text": result["final_answer_text"],
                "model_full_output": result["full_output"],
                "truncated": result["truncated"],
                "inference_time_sec": round(result["inference_time"], 3),
                "dataset_reference_reasoning": example.get("reasoning", ""),
                "ability": example.get("ability", ""),
                "error_type": error_type,
            })

        except Exception as e:

            errored += 1

            print(f"ERROR: {e}")

            results.append({
                "sample_id": example.get("id", idx),
                "domain": example.get("domain", ""),
                "task": example.get("task", ""),
                "question": example.get("question", ""),
                "choices": str(example.get("choices", "")),
                "ground_truth": str(
                    example.get("correct_answer", "")
                ).strip().upper(),
                "predicted_answer": None,
                "correct": False,
                "model_thinking": "",
                "model_final_answer_text": "",
                "model_full_output": "",
                "truncated": False,
                "inference_time_sec": None,
                "dataset_reference_reasoning": example.get("reasoning", ""),
                "ability": example.get("ability", ""),
                "error_type": f"inference_error: {str(e)}",
            })

        # Long thinking traces generate/free a lot of KV-cache memory per
        # sample. Periodically clearing the cache prevents fragmentation
        # from creeping up over a long RoboVista run. Doesn't reduce peak
        # per-sample memory, just keeps overall usage flat across samples.
        if torch.cuda.is_available() and (idx + 1) % 25 == 0:
            torch.cuda.empty_cache()

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    df = pd.DataFrame(results)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_path, index=False)

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = (correct / total * 100) if total > 0 else 0

    print("\n")
    print("=" * 50)
    print("RoboVista Evaluation Complete")
    print("=" * 50)
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Incorrect     : {total - correct - errored}")
    print(f"Invalid       : {invalid}")
    print(f"Errored       : {errored}")
    print(f"Truncated     : {truncated_count}  (raise --max_new_tokens if > 0)")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"Results saved : {output_path}")

    return df


# ============================================================
# SINGLE QUESTION MODE
# ============================================================

def single_question(
    model,
    processor,
    image_path,
    question,
    choices,
    max_new_tokens,
    do_sample,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    min_p=0.0,
):

    image = Image.open(image_path).convert("RGB")

    result = run_inference(
        model=model,
        processor=processor,
        image=image,
        question=question,
        choices=choices,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
    )

    print("\n")
    print("=" * 60)
    print("MODEL THINKING")
    print("=" * 60)
    print(result["thinking"] or "(none captured — no <think> block found)")

    print("\n")
    print("=" * 60)
    print("MODEL FINAL ANSWER TEXT")
    print("=" * 60)
    print(result["final_answer_text"])

    print("\nPredicted answer:", result["predicted_answer"])
    print(f"Inference time: {result['inference_time']:.3f}s")

    if result["truncated"]:
        print(
            "\nWARNING: generation hit max_new_tokens before the model "
            "finished thinking. Increase --max_new_tokens."
        )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-Thinking on RoboVista"
    )

    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path or HuggingFace ID of trained model "
             f"(default: {DEFAULT_MODEL_PATH})"
    )

    parser.add_argument(
        "--mode", type=str, choices=["single", "robovista"], required=True
    )

    # RoboVista arguments
    parser.add_argument(
        "--domain", type=str, default="open datasets",
        help="RoboVista domain to evaluate"
    )
    parser.add_argument(
        "--output", type=str,
        default="evaluation_results/robovista_results.xlsx"
    )
    parser.add_argument("--max_samples", type=int, default=None)

    # Single question arguments
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument(
        "--choices", nargs="+", default=None,
        help="Example: --choices 'cup' 'box' 'tool' 'plate'"
    )

    # Generation settings — defaults follow the Qwen3-VL-Thinking card's
    # recommended sampling recipe (temperature=0.6, top_p=0.95, top_k=20,
    # min_p=0). These differ from the Instruct variant's recipe
    # (temp=0.7, top_p=0.8), so don't reuse Instruct settings here.
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Thinking traces are long — keep this generous."
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding instead of sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Qwen3-VL-Thinking card recommended default: 0.6."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Thinking variant recommended default: 0.95."
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--min_p", type=float, default=0.0,
        help="Thinking variant recommended default: 0."
    )

    args = parser.parse_args()

    model, processor = load_model(args.model_path)

    do_sample = not args.greedy

    if args.mode == "single":

        if args.image is None:
            parser.error("--image is required for single mode")
        if args.question is None:
            parser.error("--question is required for single mode")
        if args.choices is None:
            parser.error("--choices is required for single mode")

        single_question(
            model=model,
            processor=processor,
            image_path=args.image,
            question=args.question,
            choices=args.choices,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
        )
        return

    elif args.mode == "robovista":

        dataset = load_robovista(domain=args.domain)

        evaluate_robovista(
            model=model,
            processor=processor,
            dataset=dataset,
            output_file=args.output,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
        )


if __name__ == "__main__":
    main()
