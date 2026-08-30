"""
Run this ONCE before training. It answers three questions:

  1. Does every row actually produce image tokens?
  2. At your current max_prompt_length, how many rows lose them to truncation?
  3. Does the generation config stop on the token your chat template emits?

Usage:
    python preflight.py --model Qwen/Qwen2-VL-7B-Instruct \
                        --dataset your/dataset --split train \
                        --max-prompt-length 512
"""

import argparse
from collections import Counter

# Fallbacks for models whose config does not expose image_token_id.
IMAGE_TOKEN_CANDIDATES = [
    "<|image_pad|>",   # Qwen2-VL / Qwen2.5-VL
    "<image>",         # LLaVA, Idefics
    "<IMG_CONTEXT>",   # InternVL
    "[IMG]",           # Pixtral
]


def resolve_image_token_id(processor, model_config=None):
    """Return (token_id, token_string). Prefers the config, falls back to
    probing the tokenizer vocabulary."""
    tok = getattr(processor, "tokenizer", processor)

    for attr in ("image_token_id", "image_token_index"):
        tid = getattr(model_config, attr, None)
        if isinstance(tid, int) and tid >= 0:
            return tid, tok.convert_ids_to_tokens(tid)

    for cand in IMAGE_TOKEN_CANDIDATES:
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid != tok.unk_token_id and tid >= 0:
            return tid, cand

    raise RuntimeError(
        "Could not resolve the image token. Print processor.tokenizer.additional_special_tokens "
        "and add the right one to IMAGE_TOKEN_CANDIDATES."
    )


def build_inputs(processor, example, image_key="image", question_key="question"):
    """Adapt this to match how your training script builds prompts.
    It MUST mirror it exactly, or the diagnostic measures the wrong thing."""
    image = example[image_key]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": example[question_key]},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt")


def scan(processor, dataset, image_token_id, max_prompt_length, n=None,
         truncation_side="left", **kw):
    rows = dataset if n is None else dataset.select(range(min(n, len(dataset))))

    stats = Counter()
    blind = []          # zero image tokens even before truncation
    truncated_blind = []  # image tokens exist but truncation removes them all
    lengths = []

    for i, ex in enumerate(rows):
        try:
            batch = build_inputs(processor, ex, **kw)
        except Exception as e:
            stats["build_failed"] += 1
            blind.append((i, f"build error: {type(e).__name__}: {e}"))
            continue

        ids = batch["input_ids"][0]
        total = ids.numel()
        n_img = int((ids == image_token_id).sum())
        lengths.append(total)

        if n_img == 0:
            stats["blind_before_truncation"] += 1
            blind.append((i, "no image tokens produced"))
            continue

        if max_prompt_length is not None and total > max_prompt_length:
            kept = ids[-max_prompt_length:] if truncation_side == "left" else ids[:max_prompt_length]
            n_img_kept = int((kept == image_token_id).sum())
            if n_img_kept == 0:
                stats["blinded_by_truncation"] += 1
                truncated_blind.append((i, total, n_img))
            elif n_img_kept < n_img:
                stats["partially_blinded"] += 1
            else:
                stats["ok_but_truncated"] += 1
        else:
            stats["ok"] += 1

    lengths.sort()
    def pct(p):
        return lengths[min(int(len(lengths) * p), len(lengths) - 1)] if lengths else 0

    print(f"\nscanned {len(rows)} rows")
    print(f"prompt length  min={lengths[0] if lengths else 0}  "
          f"p50={pct(0.5)}  p95={pct(0.95)}  max={lengths[-1] if lengths else 0}")
    print(f"max_prompt_length={max_prompt_length}  truncation_side={truncation_side}\n")

    for k, v in stats.most_common():
        print(f"  {k:26s} {v:6d}  ({100*v/max(len(rows),1):.1f}%)")

    if truncated_blind:
        print(f"\nrows blinded purely by truncation (first 20):")
        for i, total, n_img in truncated_blind[:20]:
            print(f"    row {i}: {total} tokens, {n_img} image tokens, all cut")
        print("  -> set max_prompt_length=None")

    if blind:
        print(f"\nrows with no image tokens at all (first 20):")
        for i, why in blind[:20]:
            print(f"    row {i}: {why}")
        print("  -> dataset or prompt-construction bug, not truncation")

    return stats


def check_eos(processor, model=None):
    tok = getattr(processor, "tokenizer", processor)
    print(f"\neos_token = {tok.eos_token!r}  id = {tok.eos_token_id}")

    template_enders = ["<|im_end|>", "<|eot_id|>", "<end_of_turn>", "</s>"]
    found = []
    for t in template_enders:
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid >= 0 and tid != tok.unk_token_id:
            found.append((t, tid))
            print(f"  template ender present: {t!r} -> {tid}")

    ids = {tid for _, tid in found}
    if ids and tok.eos_token_id not in ids:
        print("\n  MISMATCH: the template ends turns with a token that is not eos_token_id.")
        print("  Generation will not stop and will run to max_completion_length.")
        print("  Fix:")
        print(f"    gen_cfg.eos_token_id = {sorted(ids | {tok.eos_token_id})}")
    else:
        print("  eos looks consistent with the chat template.")


if __name__ == "__main__":
    from transformers import AutoConfig, AutoProcessor
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name or path, e.g. unsloth/Qwen3-VL-8B-Thinking")
    ap.add_argument("--dataset", required=True, help="Dataset id, e.g. keplerccc/Robo2VLM-1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--n", type=int, default=500, help="rows to scan; 0 = all")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    ds = load_dataset(args.dataset, split=args.split)

    tid, tstr = resolve_image_token_id(processor, config)
    print(f"image token: {tstr!r} -> id {tid}")

    scan(processor, ds, tid,
         max_prompt_length=args.max_prompt_length,
         n=(None if args.n == 0 else args.n))
    check_eos(processor)